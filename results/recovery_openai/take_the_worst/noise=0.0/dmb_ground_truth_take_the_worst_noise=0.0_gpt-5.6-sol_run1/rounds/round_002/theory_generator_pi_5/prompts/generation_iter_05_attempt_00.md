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
- THEORY 1 = `pi_3`
- THEORY 2 = `pi_4`
- The recommendation below acts on THEORY 2 (= `pi_4`).

Replace pi_4 with a task-invariant multi-format comparison theory rather than modifying its parameter ranges. The new theory should propose three load-dependent representational regimes driven by the observable number of discriminating cues: diffuse equal-weight comparison at low load; an intermediate-load sequential-retrieval regime in which display position or recency can compete with communicated validity; and validity-led compression at high load. Recruitment should be gradual and subject-heterogeneous, but the intermediate positional component must permit the negative Experiment 2 gradient rather than enforcing monotonic preference for higher-validity ranks. Equal communicated validities should trigger a distinct tie-resolution rule: stable individuals may either select the first displayed member of a tied tier or pool evidence across the entire tied tier. This predicts the approximately half-and-half classification and high between-subject variance in Experiment 1 without treating choices as homogeneous noise. At loads 6–8, the validity-led regime should become dominant enough to preserve the strong leading-cue adherence in Experiment 6 and the near-zero matched contrast in Experiment 3. At low load it should retain majority behavior matching Experiment 4. Broad transition widths, persistent intermediate-regime mass, and stable subject-level differences should reduce the exaggerated crossover predicted in Experiment 5. The theory must make display position, validity-tier structure, and discriminating-cue load explicit observable determinants, with ordinary response noise applied only after those representations generate evidence.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_3` (overall score: 0.980)

**Description**
Prospective Capacity-Gated Comparison theory: before choosing, people perceptually register how many cues discriminate between the options and compare that comparison load with a stable cognitive capacity. If the discriminating set is manageable, they integrate all discriminating cues with an equal-weight majority rule. If it exceeds capacity, they compress the comparison into a validity-ranked lexicographic search and rely on the first discriminating cue. Strategy recruitment changes smoothly near the capacity boundary, reflecting variability in momentary load, but is determined by an observable property of the option pair rather than by experiment identity. Ordinary response noise is applied only after a strategy has produced its intended choice.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Capacity-Gated Comparison expects shape (2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.shape[0] != n_features:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {n_features}."
        )

    differences = a - b
    discriminating = np.flatnonzero(differences != 0)
    load = float(discriminating.size)
    if load == 0:
        return np.array([0.5, 0.5], dtype=float)

    capacity = float(parameters["capacity"])
    gate_slope = float(parameters["gate_slope"])

    # Probability that exhaustive equal-weight integration is recruited.
    # This stable logistic gate approaches one below capacity and zero above it.
    gate_logit = gate_slope * (capacity - load)
    if gate_logit >= 0.0:
        exp_neg = np.exp(-gate_logit)
        p_integrate = 1.0 / (1.0 + exp_neg)
    else:
        exp_pos = np.exp(gate_logit)
        p_integrate = exp_pos / (1.0 + exp_pos)

    # Manageable-load strategy: all discriminating cues receive equal weight.
    tally_margin = float(np.sum(np.sign(differences)))
    if tally_margin > 0.0:
        tally_scores = np.array([1.0, 0.0])
    elif tally_margin < 0.0:
        tally_scores = np.array([0.0, 1.0])
    else:
        tally_scores = np.array([0.0, 0.0])

    # Overload strategy: validity-ranked compression to the first diagnostic cue.
    cue_order = np.argsort(-validities, kind="stable")
    lex_scores = np.array([0.0, 0.0])
    for cue in cue_order:
        if differences[cue] > 0.0:
            lex_scores = np.array([1.0, 0.0])
            break
        if differences[cue] < 0.0:
            lex_scores = np.array([0.0, 1.0])
            break

    beta = float(parameters["beta"])

    def noisy_strategy_choice(scores):
        logits = beta * np.asarray(scores, dtype=float)
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs /= probs.sum()
        return probs

    # Response noise follows, rather than determines, strategy recruitment.
    p_tally = noisy_strategy_choice(tally_scores)
    p_lex = noisy_strategy_choice(lex_scores)
    p_core = p_integrate * p_tally + (1.0 - p_integrate) * p_lex

    epsilon = float(parameters["epsilon"])
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
- validities: validities
- capacity: [4.5, 5.5]
- gate_slope: [2.0, 5.0]
- beta: [1.55, 2.05]
- epsilon: [0.0, 0.08]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.0300 (var=0.0056) vs this=-0.0028 (var=0.0068)
- Experiment 2: real=0.6683 (var=0.0367) vs this=0.6600 (var=0.0188)
- Experiment 3: real=0.3612 (var=0.0106) vs this=0.6512 (var=0.0114)
- Experiment 4: real=0.8756 (var=0.0078) vs this=0.8291 (var=0.0019)
- Experiment 5: real=0.5000 (var=0.2000) vs this=1.0000 (var=0.0000)
- Experiment 6: real=-0.1132 (var=0.0031) vs this=0.0009 (var=0.0007)


---

### `pi_4` (overall score: 0.413)

**Description**
Precision-Annealed Rank Integration theory: people always combine the signed evidence supplied by every discriminating cue. Comparison load does not recruit a different heuristic; instead, it continuously changes the precision with which cue priorities can be represented. At low load, the attention distribution is diffuse and discriminating cues receive nearly equal weight, approximating tallying. As the number of discriminating cues exceeds a stable representational capacity, attention is progressively concentrated toward cues higher in the communicated-validity ordering. Attention is implemented as a Gibbs distribution over validity ranks, with its concentration increasing smoothly with load. The load curve is mildly powered while anchored at high load, reducing premature concentration without weakening overload attention. The highest-validity cue retains its original Gibbs attention mass, while the conditional distribution over remaining cues is only mildly flattened, preserving collective tail evidence without excessively reallocating attention toward the lowest-validity cues. The normalized weighted evidence is then translated into a choice through ordinary logistic response noise. Stable individual differences generate heterogeneous but gradual crossovers without categorical switching between tallying and Take The Best.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Precision-Annealed Rank Integration expects shape (2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.shape[0] != n_features:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {n_features}."
        )

    differences = a - b
    signs = np.sign(differences)
    discriminating = np.flatnonzero(signs != 0.0)
    load = float(discriminating.size)

    if discriminating.size == 0:
        return np.array([0.5, 0.5], dtype=float)

    # A cue's rank is the number of cues having strictly greater communicated
    # validity. Thus equally valid cues receive exactly equal attention.
    ranks = np.asarray(
        [np.sum(validities > validities[j]) for j in range(n_features)],
        dtype=float,
    )

    capacity = float(parameters["capacity"])
    transition_width = float(parameters["transition_width"])
    concentration_gain = float(parameters["concentration_gain"])
    schedule_power = float(parameters["schedule_power"])

    # Smoothly increasing attention concentration. Powering the curve relative
    # to a load-7 anchor reduces premature low-load concentration while leaving
    # the overload concentration of the accepted model approximately intact.
    def stable_sigmoid(value):
        if value >= 0.0:
            return 1.0 / (1.0 + np.exp(-value))
        ex = np.exp(value)
        return ex / (1.0 + ex)

    x = (load - capacity) / transition_width
    concentration_fraction = stable_sigmoid(x)
    anchor_fraction = stable_sigmoid((7.0 - capacity) / transition_width)
    concentration = concentration_gain * anchor_fraction * (
        concentration_fraction / anchor_fraction
    ) ** schedule_power

    active_ranks = ranks[discriminating]
    log_attention = -concentration * active_ranks
    log_attention -= np.max(log_attention)
    attention = np.exp(log_attention)
    attention /= attention.sum()

    # Preserve the Gibbs mass assigned to the highest-validity active cue(s),
    # but flatten only the conditional distribution over the remaining cues.
    top_mask = active_ranks == np.min(active_ranks)
    tail_mask = ~top_mask
    if np.any(tail_mask):
        top_mass = float(np.sum(attention[top_mask]))
        tail_flatten = float(parameters["tail_flatten"])
        tail_logits = tail_flatten * log_attention[tail_mask]
        tail_logits -= np.max(tail_logits)
        tail_distribution = np.exp(tail_logits)
        tail_distribution /= tail_distribution.sum()

        adjusted_attention = np.zeros_like(attention)
        adjusted_attention[top_mask] = attention[top_mask]
        adjusted_attention[tail_mask] = (1.0 - top_mass) * tail_distribution
        attention = adjusted_attention

    # Positive evidence favors A and negative evidence favors B. Normalizing
    # attention keeps response precision comparable across different loads.
    evidence = float(np.dot(attention, signs[discriminating]))

    beta = float(parameters["beta"])
    logits = np.array([0.5 * beta * evidence, -0.5 * beta * evidence], dtype=float)
    logits -= np.max(logits)
    probs = np.exp(logits)
    probs /= probs.sum()

    epsilon = float(parameters["epsilon"])
    probs = (1.0 - epsilon) * probs + epsilon * np.array([0.5, 0.5], dtype=float)
    probs = np.clip(probs, 0.0, 1.0)
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
- validities: validities
- capacity: [5.3, 5.8]
- transition_width: [0.6, 0.95]
- concentration_gain: [1.15, 1.55]
- schedule_power: [1.15, 1.35]
- tail_flatten: [0.75, 0.95]
- beta: [4.2, 5.4]
- epsilon: [0.0, 0.06]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.0300 (var=0.0056) vs this=0.0328 (var=0.0015)
- Experiment 2: real=0.6683 (var=0.0367) vs this=0.5833 (var=0.0114)
- Experiment 3: real=0.3612 (var=0.0106) vs this=0.6541 (var=0.0103)
- Experiment 4: real=0.8756 (var=0.0078) vs this=0.8331 (var=0.0025)
- Experiment 5: real=0.5000 (var=0.2000) vs this=0.0000 (var=0.0000)
- Experiment 6: real=-0.1132 (var=0.0031) vs this=0.0970 (var=0.0005)


---

### `pi_1` (overall score: 0.173)

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
- Experiment 1: real=-0.0300 (var=0.0056) vs this=-0.0167 (var=0.0052)
- Experiment 2: real=0.6683 (var=0.0367) vs this=-0.6542 (var=0.0564)
- Experiment 3: real=0.3612 (var=0.0106) vs this=0.0041 (var=0.0094)
- Experiment 4: real=0.8756 (var=0.0078) vs this=0.8391 (var=0.0151)
- Experiment 5: real=0.5000 (var=0.2000) vs this=1.0000 (var=0.0000)
- Experiment 6: real=-0.1132 (var=0.0031) vs this=0.0003 (var=0.0005)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.1660 -> ACCEPTED
- iter 2: loss=0.1194 -> ACCEPTED
- iter 3: loss=0.1122 -> ACCEPTED
- iter 4: loss=0.0795 -> ACCEPTED
- iter 5: loss=0.0843 -> REJECTED
Running-best (last ACCEPTED) base: iter 4 at loss=0.0795 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Contrast top-cue adherence between tally-congruent and
    # tally-conflicting trials, matching on absolute tally margin.
    cells = {(d, c): [] for d in (2, 4, 6) for c in (-1, 1)}

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        response = int(row["response"])

        # Feature 0 is the highest-validity cue in this experiment.
        if a[0] > b[0]:
            top_choice = 0
            top_relative_margin = int(np.sum(a > b) - np.sum(b > a))
        elif b[0] > a[0]:
            top_choice = 1
            top_relative_margin = int(np.sum(b > a) - np.sum(a > b))
        else:
            continue

        d = abs(top_relative_margin)
        if d not in (2, 4, 6) or top_relative_margin == 0:
            continue

        congruence = 1 if top_relative_margin > 0 else -1
        top_adherence = 1.0 if response == top_choice else -1.0
        cells[(d, congruence)].append(top_adherence)

    matched_contrasts = []
    for d in (2, 4, 6):
        congruent = cells[(d, 1)]
        conflicting = cells[(d, -1)]
        if len(congruent) == 0 or len(conflicting) == 0:
            continue
        # Division by two keeps the metric in [-1, 1].
        matched_contrasts.append(
            0.5 * (float(np.mean(congruent)) - float(np.mean(conflicting)))
        )

    if len(matched_contrasts) == 0:
        return float("nan")
    return float(np.mean(matched_contrasts))
```

**Observed (real) value:** -0.0300 (var=0.0056)
**Previous candidate values (this loop):**
  - iter 1: 0.1011 (var=0.0061) (Δ vs real +0.1311)
  - iter 2: 0.0889 (var=0.0089) (Δ vs real +0.1189)
  - iter 3: 0.0572 (var=0.0037) (Δ vs real +0.0872)
  - iter 4: 0.0472 (var=0.0073) (Δ vs real +0.0772)
  - iter 5 (most recent): 0.0433 (var=0.0095) (Δ vs real +0.0733)
**Other theories' values on this metric (for reference):**
- pi_1: -0.0167 (var=0.0052)
- pi_2: 0.8033 (var=0.0243)
- pi_3: -0.0028 (var=0.0068)
- pi_4: 0.0328 (var=0.0015)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    scores = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        d = a - b

        # +1 denotes an A prediction and -1 a B prediction.
        tally_margin = np.sum(np.sign(d))
        if tally_margin == 0:
            continue
        tally_direction = 1.0 if tally_margin > 0 else -1.0

        # Validities decrease with feature index in this experiment.
        discriminating = np.flatnonzero(d != 0)
        if discriminating.size == 0:
            continue
        first = int(discriminating[0])
        ttb_direction = 1.0 if d[first] > 0 else -1.0

        # Agreement trials cannot distinguish the theories.
        if tally_direction == ttb_direction:
            continue

        observed_direction = 1.0 if int(row["response"]) == 0 else -1.0
        scores.append(observed_direction * tally_direction)

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** 0.6683 (var=0.0367)
**Previous candidate values (this loop):**
  - iter 1: 0.6983 (var=0.0137) (Δ vs real +0.0300)
  - iter 2: 0.7142 (var=0.0083) (Δ vs real +0.0458)
  - iter 3: 0.7608 (var=0.0133) (Δ vs real +0.0925)
  - iter 4: 0.6308 (var=0.0148) (Δ vs real -0.0375)
  - iter 5 (most recent): 0.6433 (var=0.0124) (Δ vs real -0.0250)
**Other theories' values on this metric (for reference):**
- pi_2: 0.7008 (var=0.0514)
- pi_1: -0.6542 (var=0.0564)
- pi_3: 0.6600 (var=0.0188)
- pi_4: 0.5833 (var=0.0114)

### Experiment 3
**Design**
  A=[0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 0, 1, 0, 1]  B=[1, 0, 0, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Return a parity-matched low-to-high load crossover index."""
    if data is None or len(data) == 0:
        return float("nan")

    loads = []
    majority_consistent = []

    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"],
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        diff = a - b
        loads.append(int(np.count_nonzero(diff)))

        tally_margin = float(np.sum(np.sign(diff)))
        if tally_margin > 0:
            majority_response = 0
        elif tally_margin < 0:
            majority_response = 1
        else:
            majority_response = -1

        majority_consistent.append(
            float(int(response) == majority_response)
            if majority_response >= 0 else np.nan
        )

    work = pd.DataFrame({
        "load": np.asarray(loads, dtype=int),
        "majority_consistent": np.asarray(majority_consistent, dtype=float),
    })

    p = {}
    for load in (3, 4, 6, 7, 8):
        values = work.loc[
            work["load"] == load, "majority_consistent"
        ].dropna()
        if len(values) == 0:
            return float("nan")
        p[load] = float(values.mean())

    odd_crossover = p[3] - p[7]
    even_crossover = p[4] - 0.5 * (p[6] + p[8])
    return float(0.5 * (odd_crossover + even_crossover))

```

**Observed (real) value:** 0.3612 (var=0.0106)
**Previous candidate values (this loop):**
  - iter 1: 0.6303 (var=0.0073) (Δ vs real +0.2691)
  - iter 2: 0.5947 (var=0.0051) (Δ vs real +0.2334)
  - iter 3: 0.5866 (var=0.0052) (Δ vs real +0.2253)
  - iter 4: 0.5237 (var=0.0059) (Δ vs real +0.1625)
  - iter 5 (most recent): 0.5141 (var=0.0072) (Δ vs real +0.1528)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6512 (var=0.0114)
- pi_2: 0.0066 (var=0.0037)
- pi_1: 0.0041 (var=0.0094)
- pi_4: 0.6541 (var=0.0103)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0, 0, 1, 0]  B=[1, 1, 0, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Rate of validity-leading-cue-consistent choices on overload trials."""
    scores = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        differences = a - b
        discriminating = np.flatnonzero(differences != 0)

        # Focus on the 6- and 8-cue overload conditions.
        if discriminating.size < 6:
            continue

        # Validities are strictly descending by feature index, so the first
        # discriminating index is the highest-validity diagnostic cue.
        leading_cue = int(discriminating[0])
        lex_response = 0 if differences[leading_cue] > 0 else 1
        scores.append(float(int(row["response"]) == lex_response))

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** 0.8756 (var=0.0078)
**Previous candidate values (this loop):**
  - iter 1: 0.8759 (var=0.0023) (Δ vs real +0.0003)
  - iter 2: 0.8781 (var=0.0018) (Δ vs real +0.0025)
  - iter 3: 0.9113 (var=0.0012) (Δ vs real +0.0356)
  - iter 4: 0.8859 (var=0.0016) (Δ vs real +0.0103)
  - iter 5 (most recent): 0.8753 (var=0.0025) (Δ vs real -0.0003)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5031 (var=0.0037)
- pi_3: 0.8291 (var=0.0019)
- pi_1: 0.8391 (var=0.0151)
- pi_4: 0.8331 (var=0.0025)

### Experiment 5
**Design**
  A=[1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 1]  B=[0, 0, 0, 1, 1, 1, 0]
  A=[1, 1, 0, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1, 1]  B=[1, 1, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0, 1]  B=[1, 0, 0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def subject_classification(df):
        aligned = []
        for _, row in df.iterrows():
            a = np.asarray(row['option_a_ratings'], dtype=float)
            b = np.asarray(row['option_b_ratings'], dtype=float)
            d = np.sign(a - b)

            # Select trials where Experts 2 and 3 agree with each other but
            # oppose Expert 1: a 2-to-1 conflict within the top-validity tier.
            if d.size >= 3 and d[0] != 0 and d[1] == d[2] and d[1] == -d[0]:
                response = int(row['response'])
                chose_a = 1.0 if response == 0 else -1.0
                aligned.append(float(chose_a * d[0] > 0))

        if len(aligned) == 0:
            return np.nan
        rate = float(np.mean(aligned))
        if rate > 0.5:
            return 1.0
        if rate < 0.5:
            return 0.0
        return 0.5

    values = []
    for _, subj_df in data.groupby('subject_id', sort=False):
        value = subject_classification(subj_df)
        if np.isfinite(value):
            values.append(value)

    if len(values) == 0:
        return float('nan')
    return float(np.mean(values))
```

**Observed (real) value:** 0.5000 (var=0.2000)
**Previous candidate values (this loop):**
  - iter 1: 0.4800 (var=0.2496) (Δ vs real -0.0200)
  - iter 2: 0.5000 (var=0.2500) (Δ vs real +0.0000)
  - iter 3: 0.5200 (var=0.2496) (Δ vs real +0.0200)
  - iter 4: 0.5200 (var=0.2496) (Δ vs real +0.0200)
  - iter 5 (most recent): 0.6000 (var=0.2400) (Δ vs real +0.1000)
**Other theories' values on this metric (for reference):**
- pi_3: 1.0000 (var=0.0000)
- pi_4: 0.0000 (var=0.0000)
- pi_1: 1.0000 (var=0.0000)
- pi_2: 1.0000 (var=0.0000)

### Experiment 6
**Design**
  A=[0, 0, 1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 0, 0, 0]  B=[0, 1, 0, 0, 1, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0, 1, 1]  B=[0, 1, 0, 1, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Contrast majority-choice rates across the four positions of the
    # second cue opposing the majority. Coefficients sum to zero, so any
    # profile that is flat across positions has expected metric zero.
    weights = {1: -0.24, 3: -0.05, 5: 0.10, 7: 0.19}
    outcomes = {1: [], 3: [], 5: [], 7: []}

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        signs = np.sign(a - b)
        active = np.flatnonzero(signs != 0)
        if active.size == 0:
            continue

        majority_sign = np.sign(np.sum(signs[active]))
        if majority_sign == 0:
            continue

        # Expert 1 (array index 0) is the highest-validity cue. The target
        # position is the other active cue pointing in the same direction.
        top_sign = signs[0]
        opposing = [int(j) for j in active if j != 0 and signs[j] == top_sign]
        if len(opposing) != 1 or opposing[0] not in outcomes:
            continue

        response = int(row["response"])
        chose_majority = ((response == 0 and majority_sign > 0) or
                          (response == 1 and majority_sign < 0))
        outcomes[opposing[0]].append(float(chose_majority))

    if any(len(outcomes[j]) == 0 for j in outcomes):
        return float("nan")

    rates = {j: float(np.mean(outcomes[j])) for j in outcomes}
    return float(sum(weights[j] * rates[j] for j in weights))

```

**Observed (real) value:** -0.1132 (var=0.0031)
**Previous candidate values (this loop):**
  - iter 1: -0.0330 (var=0.0013) (Δ vs real +0.0802)
  - iter 2: -0.0765 (var=0.0009) (Δ vs real +0.0367)
  - iter 3: -0.1028 (var=0.0005) (Δ vs real +0.0103)
  - iter 4: -0.1006 (var=0.0012) (Δ vs real +0.0126)
  - iter 5 (most recent): -0.0999 (var=0.0008) (Δ vs real +0.0133)
**Other theories' values on this metric (for reference):**
- pi_4: 0.0970 (var=0.0005)
- pi_3: 0.0009 (var=0.0007)
- pi_1: 0.0003 (var=0.0005)
- pi_2: 0.0004 (var=0.0004)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Load-Indexed Format Arbitration theory: people first register the observable number of cues that discriminate between the options and use it to recruit among three comparison formats. At low discriminating-cue load, they diffusely pool all diagnostic cues with equal weight, but the confidence attached to this normalized majority evidence varies stably across individuals. At intermediate load, they retrieve diagnostic cues sequentially from the display; retrieval accessibility combines communicated-validity tier with a recency gradient that is attenuated at very low loads and reaches full strength by overload loads. At high load, they compress the display to the highest-validity active tier, with a format-specific and person-specific evidence gain. The high-load transition distinguishes loads 6–7 from load 8, and its location varies coherently with recency strength. Observable validity-tier structure changes compressibility: a distinctive partially tied leading tier facilitates compression, whereas an entirely undifferentiated validity display delays it. Equal-validity tiers invoke a stable person-specific convention: some individuals use the first displayed diagnostic member of the tier, whereas others pool all diagnostic members of that tier. These representations and their format-specific confidence generate signed evidence before ordinary response noise is applied.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Load-Indexed Format Arbitration expects shape (2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.shape[0] != n_features:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {n_features}."
        )

    signs = np.sign(a - b)
    active = np.flatnonzero(signs != 0.0)
    load = float(active.size)
    if active.size == 0:
        return np.array([0.5, 0.5], dtype=float)

    def stable_sigmoid(x):
        x = float(x)
        if x >= 0.0:
            z = np.exp(-x)
            return 1.0 / (1.0 + z)
        z = np.exp(x)
        return z / (1.0 + z)

    # Observable load gradually opens sequential retrieval and, later,
    # validity-tier compression. The low transition begins slightly earlier
    # so retrieval can contribute on three-cue conflicts.
    low_width = float(parameters["transition_width"])
    entered_retrieval = stable_sigmoid(
        (load - float(parameters["low_center"])) / low_width
    )

    global_best = float(np.max(validities))
    global_top_tier = np.flatnonzero(np.isclose(
        validities, global_best, rtol=1e-9, atol=1e-12
    ))
    partial_top_tie = 1.0 if 1 < global_top_tier.size < n_features else 0.0
    all_equal = 1.0 if global_top_tier.size == n_features else 0.0

    # Recency strength and compression timing are stably correlated within a
    # person. Strong-recency subjects retain retrieval longer; validity-led
    # subjects compress earlier. Distinct partial tiers facilitate chunking,
    # whereas an all-equal display supplies no validity boundary to compress.
    recency_strength = float(parameters["recency_strength"])
    recency_profile = (recency_strength - 5.8) / 1.0
    effective_high_center = (
        float(parameters["high_center"])
        + float(parameters["recency_compression_coupling"]) * recency_profile
        - partial_top_tie * float(parameters["partial_tie_acceleration"])
        + all_equal * float(parameters["all_equal_delay"])
    )
    compressed = stable_sigmoid(
        (load - effective_high_center)
        / float(parameters["high_transition_width"])
    )

    w_low = 1.0 - entered_retrieval
    w_high = entered_retrieval * compressed
    w_mid = entered_retrieval * (1.0 - compressed)

    # Low-load format: diffuse equal-weight comparison. Normalization makes
    # the evidence the diagnostic-cue majority margin rather than display size.
    low_evidence = float(np.mean(signs[active]))

    # Construct communicated-validity tiers. Cues tied in communicated
    # validity have the same rank; display position remains separately visible.
    tier_rank = np.asarray(
        [np.sum(validities > validities[j] + 1e-12) for j in range(n_features)],
        dtype=float,
    )
    rank_scale = max(1.0, float(n_features - 1))
    normalized_rank = tier_rank / rank_scale
    if n_features > 1:
        display_position = np.arange(n_features, dtype=float) / float(n_features - 1)
    else:
        display_position = np.zeros(1, dtype=float)

    # Intermediate format: expected evidence from sequential retrieval.
    # Positional accessibility is mildly attenuated at three-cue load and
    # reaches its original strength by load six, preserving overload gradients.
    recency_load_scale = 0.85 + 0.15 * np.clip((load - 3.0) / 3.0, 0.0, 1.0)
    retrieval_log_access = (
        recency_load_scale * recency_strength * display_position[active]
        - float(parameters["validity_strength"]) * normalized_rank[active]
    )
    retrieval_log_access -= np.max(retrieval_log_access)
    retrieval_weights = np.exp(retrieval_log_access)
    retrieval_weights /= retrieval_weights.sum()
    mid_evidence = float(np.dot(retrieval_weights, signs[active]))

    # High-load format: compress to the best communicated-validity tier that
    # is diagnostic on this trial. Stable tie_rule=0 uses its first displayed
    # member; tie_rule=1 pools the complete active tier before taking direction.
    best_validity = float(np.max(validities[active]))
    top_tier = active[np.isclose(
        validities[active], best_validity, rtol=1e-9, atol=1e-12
    )]

    tie_rule = int(parameters["tie_rule"])
    if tie_rule == 0:
        first_member = int(np.min(top_tier))
        high_evidence = float(signs[first_member])
    else:
        tier_sum = float(np.sum(signs[top_tier]))
        if tier_sum > 0.0:
            high_evidence = 1.0
        elif tier_sum < 0.0:
            high_evidence = -1.0
        else:
            high_evidence = 0.0

    beta = float(parameters["beta"])

    def evidence_probabilities(evidence, gain=1.0):
        logits = np.array([
            0.5 * beta * float(gain) * float(evidence),
            -0.5 * beta * float(gain) * float(evidence),
        ], dtype=float)
        logits -= np.max(logits)
        p = np.exp(logits)
        p /= p.sum()
        return p

    # Format recruitment is part of representation, not response noise.
    # Stable format-specific gains calibrate confidence without weakening the
    # intermediate positional profile that is needed at overload loads.
    p_core = (
        w_low * evidence_probabilities(
            low_evidence, float(parameters["low_evidence_gain"])
        )
        + w_mid * evidence_probabilities(mid_evidence)
        + w_high * evidence_probabilities(
            high_evidence, float(parameters["high_evidence_gain"])
        )
    )

    # Ordinary lapses occur only after each format has generated evidence.
    epsilon = float(parameters["epsilon"])
    probs = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=float)
    return probs / total


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
- validities: validities
- low_center: [4.3, 4.8]
- high_center: [7.0, 7.35]
- transition_width: [0.9, 1.3]
- high_transition_width: [0.3, 0.5]
- partial_tie_acceleration: [0.9, 1.4]
- all_equal_delay: [0.9, 1.3]
- recency_compression_coupling: [0.25, 0.45]
- recency_strength: [4.8, 6.8]
- validity_strength: [0.2, 0.65]
- tie_rule: {0, 1}
- low_evidence_gain: [0.6, 0.85]
- high_evidence_gain: [0.4, 0.65]
- beta: [4.8, 6.2]
- epsilon: [0.0, 0.06]

`rationale`: This is a minimal confidence-calibration edit to the accepted three-format model. The recruitment gates, recency mechanism, validity tiers, and tied-tier convention are retained. Two stable subject-level gains are added only at the response mapping of the low and high representations. The weaker low-format gain reduces overconfident three-cue majority choices, targeting Experiment 2 and the low-load side of Experiment 3. The weaker high-format gain softens excessive leading-cue certainty, targeting Experiment 4 and raising high-load majority consistency enough to reduce Experiment 3's exaggerated crossover. Their broad ranges introduce stable heterogeneity rather than homogeneous trial noise. Intermediate evidence remains at its original gain so the successful Experiment 6 positional contrast is protected. Recency is only mildly attenuated at load 3 and is unchanged from load 6 upward, preventing low-load retrieval from becoming too position-dominated without disturbing the overload gradient. Finally, the all-equal-tier delay is increased modestly and in isolation, preserving more intermediate retrieval when no validity boundary exists and moving Experiment 1 toward its near-zero contrast.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed three-regime mechanism and captures several important invariants. Experiment 2 is close to the observed majority preference (0.698 versus 0.668), Experiment 4 reproduces overload leading-cue adherence almost exactly (0.876 versus 0.876), and the stable tied-tier rule successfully produces the heterogeneous half-and-half classification in Experiment 5 (0.480 versus 0.500, with appropriately high between-subject variance). However, three systematic discrepancies remain. First, Experiment 3’s load crossover is much too large (0.630 versus 0.361), indicating that recruitment changes too decisively from low-load pooling to high-load top-tier compression. Second, Experiment 6 has the correct negative sign but only about one third of the required positional gradient (-0.033 versus -0.113), so sequential retrieval/recency has too little effective influence at overload loads. Third, Experiment 1 produces a positive congruence effect (0.101) where the observed matched contrast is approximately zero or slightly negative (-0.030), suggesting excessive residual sensitivity to the tally configuration in those high-load trials. The model also understates subject variance in Experiments 2, 4, and 6, consistent with overly narrow heterogeneity in regime recruitment and the recency-to-validity balance.
Rationale: Keep the current mechanism family and the successful tied-tier rule, but recalibrate recruitment and heterogeneity. Do not globally lower beta or raise lapse noise, because that would sacrifice the excellent Experiment 4 fit and likely weaken Experiment 2. Instead, allow a larger and more heterogeneous intermediate-regime contribution specifically around loads 6–7, with a wider subject-level distribution of high-transition centers and recency-to-validity ratios. Strengthen recency among the persistent-intermediate subjects so Experiment 6’s negative gradient can approach -0.113, while retaining a substantial validity-compression subgroup to preserve Experiment 4’s leading-cue adherence. Consider separate low- and high-transition widths or a tier-structure-dependent high-transition center: this can delay compression at loads 6–7 without eliminating strong compression at load 8, thereby reducing Experiment 3’s exaggerated crossover while avoiding additional tally-congruence sensitivity in Experiment 1. The next fit should target the joint constraint—smaller Experiment 3 crossover, more negative Experiment 6 gradient, and near-zero Experiment 1 contrast—rather than changing response noise or abandoning the three-format arbitration account.

**Outcome of this advice:** iter 1 candidate loss=0.1660 -> iter 2 candidate loss=0.1194 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The candidate is faithful to the arbiter’s three-format mechanism and the accepted revision clearly improved the joint fit: Experiment 3’s crossover fell from 0.630 to 0.595, Experiment 6’s positional gradient became substantially more negative (-0.033 to -0.077), and Experiment 1 moved toward zero. It also retains the excellent leading-cue adherence in Experiment 4 (0.878 versus 0.876) and exactly captures Experiment 5’s half-and-half classification, including strong subject heterogeneity. Nevertheless, the main cross-experiment discrepancy remains substantial. Experiment 3 still predicts much too sharp a low-to-high-load crossover (0.595 versus 0.361), while Experiment 1 still predicts an appreciable positive tally-congruence effect where humans are near zero or slightly negative (0.089 versus -0.030). Experiment 6 now has the correct sign but still only about two thirds of the observed positional gradient (-0.077 versus -0.113). Experiment 2 has also moved slightly away from the target (0.714 versus 0.668), and its simulated between-subject variance remains too low. Thus, the representational regimes are qualitatively correct, but their recruitment remains too homogeneous and insufficiently differentiated between loads 6–7 and load 8.
Rationale: The previous advice was accepted and improved all three targeted metrics, so retain the three regimes, the tied-tier convention, and the strategy of preserving sequential retrieval at overload. Extend it selectively rather than globally increasing recency, which would endanger Experiment 4. First, further separate loads 6–7 from load 8: retain more intermediate-format mass at 6–7 to reduce Experiment 3’s crossover and strengthen Experiment 6’s negative gradient, while making validity compression approach saturation at load 8 so Experiment 1’s residual positive congruence contrast moves toward zero. This can be achieved with a modestly later high-transition center plus a transition normalization or shape that still saturates by load 8, rather than simply widening the gate everywhere. Second, introduce stronger stable, correlated subject heterogeneity within the prescribed family: subjects with stronger recency should also have later compression, whereas validity-led subjects should have earlier compression and weaker recency. That mixture can increase the currently understated variances and improve Experiments 3 and 6 while preserving the pooled leading-cue rate in Experiment 4. Validity-tier distinctiveness may modulate this gate, but it must be defined from observable tier structure rather than experiment identity. Finally, slightly advance—not delay—the low-to-intermediate transition: the last shift toward later low-load recruitment worsened Experiment 2 from 0.698 to 0.714, whereas a small amount of additional retrieval at its low-load conflicts should reduce excessive majority choice toward 0.668. Keep beta, lapse noise, and the successful tie-rule mixture essentially unchanged.

**Outcome of this advice:** iter 2 candidate loss=0.1194 -> iter 3 candidate loss=0.1122 -> the gate marked it ACCEPTED.

### Iteration 3
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed three-regime, load-indexed mechanism and the latest accepted revision improves aggregate fit. In particular, Experiment 6 is now very close to the observed negative positional gradient (-0.103 versus -0.113), Experiment 1 continues moving toward its near-zero contrast (0.057 versus -0.030), and Experiment 5 retains the crucial approximately half-and-half subject classification (0.520 versus 0.500) with strong heterogeneity. However, the fit is not yet experiment-invariant. Experiment 3 remains the principal failure: the predicted crossover is 0.587 versus 0.361, indicating substantially too much separation between confident low-load majority responding and high-load validity-led responding. Moreover, the latest gating/recency changes produced collateral deterioration: Experiment 2 now overpredicts tally choice (0.761 versus 0.668), and Experiment 4 overpredicts leading-cue adherence (0.911 versus 0.876). Between-subject variance also remains much too low in Experiments 2, 4, and 6. Thus the qualitative formats are appropriate, but evidence strength within the formats is too extreme and homogeneous.
Rationale: The previous recommendation was accepted and successfully repaired Experiment 6, so retain the current three formats, observable load and tier gates, recency mechanism, and tied-tier convention. But do not again globally strengthen recency or further sharpen/delay the high-load transition: Experiment 6 is already nearly correct, while those changes have pushed Experiments 2 and 4 away from their targets. Instead, add or recalibrate format-specific evidence gains/temperatures. Use a weaker, heterogeneous low-load gain for diffuse equal-weight evidence so three-cue majority choices remain above chance but fall toward Experiment 2's 0.668; this should also reduce the low-load side of Experiment 3's excessive crossover without altering the successful overload positional gradient. Because intermediate retrieval at load 3 currently emphasizes the late majority cues, its recency contribution should ramp with discriminating-cue load—weak at load 3 but unchanged around loads 6–8—rather than changing recency strength globally. A modestly weaker high-format gain can reduce Experiment 4's excessive 0.911 leading-cue adherence toward 0.876 and soften Experiment 3's high-load extreme while leaving the position-dependent mid-format contrast largely intact. Permit stable subject heterogeneity in these format-specific gains to address the understated variances. Experiment 1 still needs a small all-equal-tier-specific adjustment toward greater intermediate-format influence, but this should be modest and isolated from strictly ordered-validity displays. The next revision should therefore preserve the now-successful Experiment 6 gate and target confidence calibration within regimes, especially the joint Experiment 2/3 error.

**Outcome of this advice:** iter 3 candidate loss=0.1122 -> iter 4 candidate loss=0.0795 -> the gate marked it ACCEPTED.

### Iteration 4
Verdict: regenerate
Interpretation: The current candidate is faithful to the prescribed multi-format mechanism and is the strongest accepted iteration so far. The prior confidence-calibration advice was productive: aggregate loss fell from 0.1122 to 0.0795; Experiment 2 moved from excessive tally choice to a small underprediction (0.631 versus 0.668); Experiment 4 returned close to observed leading-cue adherence (0.886 versus 0.876); and Experiment 6 retains the required negative positional gradient (-0.101 versus -0.113). Experiment 5 also preserves the crucial heterogeneous half-and-half classification. Nevertheless, Experiment 3 remains a clear systematic failure: its crossover is 0.524 versus 0.361, meaning the model still separates low-load majority behavior from high-load behavior too strongly. Experiment 1 also retains a positive matched congruence contrast (0.047) where humans are slightly negative (-0.030). Finally, subject variance remains markedly too low in Experiments 2, 4, and 6, while Experiment 5 is somewhat too polarized. Thus the model captures the qualitative cross-experiment pattern, but its transition among formats remains too sharp and homogeneous for experiment-invariant adequacy.
Rationale: Retain the three regimes, load indexing, recency mechanism, validity-tier representation, and stable tied-tier conventions. The accepted low/high confidence edit should not simply be repeated more strongly: weakening low-format evidence further would worsen Experiment 2, which has now crossed below its target. Keep the low gain near its current range rather than lowering it again. Instead, target Experiment 3 from the high-load side using an observable tier-structure adjustment. Modestly reduce the partial-top-tier acceleration or otherwise preserve more intermediate-regime mass at loads 6–7 specifically for partially tied leading tiers, while leaving compression for strictly ordered displays and at load 8 largely intact. This should reduce the Experiment 3 crossover, can move Experiment 6 the remaining small distance toward its negative target, and should preserve Experiment 4's already accurate strictly ordered leading-cue adherence. It may also make Experiment 5 subjects less perfectly polarized, bringing its variance down from 0.250 toward 0.200 without destroying the half-and-half mean. In addition, widen stable subject heterogeneity in high-format gain and compression timing around the current means; this can raise the understated variances and slightly soften average leading-cue certainty without using homogeneous lapse noise. For Experiment 1, make only a small further all-equal-tier-specific increase in persistent retrieval or broaden that transition, since the existing isolated adjustment has moved the contrast in the correct direction. Avoid global increases in recency, global delays in compression, or another broad reduction in low evidence strength, because Experiments 2, 4, and 6 are already close.

**Outcome of this advice:** iter 4 candidate loss=0.0795 -> iter 5 candidate loss=0.0843 -> the gate marked it REJECTED.

### Iteration 5 (most recent — address this)
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed three-regime family and still captures several important patterns: Experiment 4 is essentially exact (0.8753 versus 0.8756), Experiment 2 is close (0.6433 versus 0.6683), Experiment 6 retains the correct and nearly sufficient negative positional gradient (-0.0999 versus -0.1132), and Experiment 1 moved slightly toward zero. However, this iteration was rejected because aggregate loss increased from 0.0795 to 0.0843. The attempted reduction in partial-tier acceleration produced only a small improvement in Experiment 3 (0.5237 to 0.5141, still far above 0.3612) while substantially damaging Experiment 5: its classification mean shifted from 0.520 to 0.600 instead of the observed 0.500. Widening compression timing and high-format gain also failed to improve the persistently understated variances in Experiments 2, 4, and 6. Thus, the dominant remaining error is still Experiment 3's exaggerated crossover, but the latest attempt to correct it from the high-load recruitment side had an unfavorable tradeoff.
Rationale: Because the current candidate was rejected, the next attempt should use the unchanged iteration-4 accepted base and pursue a different in-family adjustment rather than again reducing partial-tier acceleration, increasing the all-equal delay, or broadly widening high-format parameters. Target Experiment 3 from its low-load side without changing strictly ordered low-load behavior: introduce a modest observable tier-structure-dependent attenuation of diffuse low-format confidence specifically for partially tied validity structures at loads 3–4. This should lower the low-load majority-consistency terms driving Experiment 3's crossover while leaving Experiment 2's already slightly low majority preference, Experiment 4's leading-cue adherence, and Experiment 6's recency gradient largely intact. A second useful normalization fix is to make the pooling tie rule genuinely pool graded evidence: use the mean signed evidence across the active tied tier rather than reducing its sum immediately to ±1. The current sign collapse makes a weak 2-to-1 tied-tier majority as confident as unanimity, encouraging overly polarized classifications. Graded tied-tier evidence could reduce Experiment 5's excessive polarization and soften Experiment 1's residual positive congruence contrast without altering the first-member subgroup or adding homogeneous noise. Keep the accepted recency profile, high-load gate, and global low/high gains otherwise fixed; the rejected broad transition and heterogeneity push should not be repeated.

**Outcome of this advice:** iter 5 candidate loss=0.0843 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Load-Indexed Format Arbitration theory: people first register the observable number of cues that discriminate between the options and use it to recruit among three comparison formats. At low discriminating-cue load, they diffusely pool all diagnostic cues with equal weight, but the confidence attached to this normalized majority evidence varies stably across individuals. When the display contains a partially tied leading-validity tier, confidence in diffuse pooling is selectively attenuated at loads 3–4 because the tied priority structure interferes with forming a unitary low-load comparison. At intermediate load, people retrieve diagnostic cues sequentially from the display; retrieval accessibility combines communicated-validity tier with a recency gradient that is attenuated at very low loads and reaches full strength by overload loads. At high load, they compress the display to the highest-validity active tier, with a format-specific and person-specific evidence gain. The high-load transition distinguishes loads 6–7 from load 8, and its location varies coherently with recency strength. Observable validity-tier structure changes compressibility: a distinctive partially tied leading tier facilitates compression, whereas an entirely undifferentiated validity display delays it. Equal-validity tiers invoke a stable person-specific convention: some individuals use the first displayed diagnostic member of the tier, whereas others pool graded signed evidence across all diagnostic members of that tier. These representations and their format-specific confidence generate signed evidence before ordinary response noise is applied.",
  "predict": "def predict(parameters, state, history) -> np.ndarray:\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"Load-Indexed Format Arbitration expects shape (2, n_features); got {stim.shape}.\"\n        )\n\n    a, b = stim[0], stim[1]\n    n_features = stim.shape[1]\n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    if validities.ndim != 1 or validities.shape[0] != n_features:\n        raise ValueError(\n            f\"validities length {validities.shape[0]} != n_features {n_features}.\"\n        )\n\n    signs = np.sign(a - b)\n    active = np.flatnonzero(signs != 0.0)\n    load = float(active.size)\n    if active.size == 0:\n        return np.array([0.5, 0.5], dtype=float)\n\n    def stable_sigmoid(x):\n        x = float(x)\n        if x >= 0.0:\n            z = np.exp(-x)\n            return 1.0 / (1.0 + z)\n        z = np.exp(x)\n        return z / (1.0 + z)\n\n    # Observable load gradually opens sequential retrieval and, later,\n    # validity-tier compression. The low transition begins slightly earlier\n    # so retrieval can contribute on three-cue conflicts.\n    low_width = float(parameters[\"transition_width\"])\n    entered_retrieval = stable_sigmoid(\n        (load - float(parameters[\"low_center\"])) / low_width\n    )\n\n    global_best = float(np.max(validities))\n    global_top_tier = np.flatnonzero(np.isclose(\n        validities, global_best, rtol=1e-9, atol=1e-12\n    ))\n    partial_top_tie = 1.0 if 1 < global_top_tier.size < n_features else 0.0\n    all_equal = 1.0 if global_top_tier.size == n_features else 0.0\n\n    # Recency strength and compression timing are stably correlated within a\n    # person. Strong-recency subjects retain retrieval longer; validity-led\n    # subjects compress earlier. Distinct partial tiers facilitate chunking,\n    # whereas an all-equal display supplies no validity boundary to compress.\n    recency_strength = float(parameters[\"recency_strength\"])\n    recency_profile = (recency_strength - 5.8) / 1.0\n    effective_high_center = (\n        float(parameters[\"high_center\"])\n        + float(parameters[\"recency_compression_coupling\"]) * recency_profile\n        - partial_top_tie * float(parameters[\"partial_tie_acceleration\"])\n        + all_equal * float(parameters[\"all_equal_delay\"])\n    )\n    compressed = stable_sigmoid(\n        (load - effective_high_center)\n        / float(parameters[\"high_transition_width\"])\n    )\n\n    w_low = 1.0 - entered_retrieval\n    w_high = entered_retrieval * compressed\n    w_mid = entered_retrieval * (1.0 - compressed)\n\n    # Low-load format: diffuse equal-weight comparison. Normalization makes\n    # the evidence the diagnostic-cue majority margin rather than display size.\n    low_evidence = float(np.mean(signs[active]))\n\n    # A partially tied leading tier selectively lowers confidence in diffuse\n    # pooling at loads 3-4, without changing strictly ordered low-load displays.\n    low_gain_multiplier = 1.0\n    if partial_top_tie > 0.0 and 3.0 <= load <= 4.0:\n        low_gain_multiplier = float(parameters[\"partial_tie_low_attenuation\"])\n\n    # Construct communicated-validity tiers. Cues tied in communicated\n    # validity have the same rank; display position remains separately visible.\n    tier_rank = np.asarray(\n        [np.sum(validities > validities[j] + 1e-12) for j in range(n_features)],\n        dtype=float,\n    )\n    rank_scale = max(1.0, float(n_features - 1))\n    normalized_rank = tier_rank / rank_scale\n    if n_features > 1:\n        display_position = np.arange(n_features, dtype=float) / float(n_features - 1)\n    else:\n        display_position = np.zeros(1, dtype=float)\n\n    # Intermediate format: expected evidence from sequential retrieval.\n    # Positional accessibility is mildly attenuated at three-cue load and\n    # reaches its original strength by load six, preserving overload gradients.\n    recency_load_scale = 0.85 + 0.15 * np.clip((load - 3.0) / 3.0, 0.0, 1.0)\n    retrieval_log_access = (\n        recency_load_scale * recency_strength * display_position[active]\n        - float(parameters[\"validity_strength\"]) * normalized_rank[active]\n    )\n    retrieval_log_access -= np.max(retrieval_log_access)\n    retrieval_weights = np.exp(retrieval_log_access)\n    retrieval_weights /= retrieval_weights.sum()\n    mid_evidence = float(np.dot(retrieval_weights, signs[active]))\n\n    # High-load format: compress to the best communicated-validity tier that\n    # is diagnostic on this trial. Stable tie_rule=0 uses its first displayed\n    # member; tie_rule=1 pools graded evidence over the complete active tier.\n    best_validity = float(np.max(validities[active]))\n    top_tier = active[np.isclose(\n        validities[active], best_validity, rtol=1e-9, atol=1e-12\n    )]\n\n    tie_rule = int(parameters[\"tie_rule\"])\n    if tie_rule == 0:\n        first_member = int(np.min(top_tier))\n        high_evidence = float(signs[first_member])\n    else:\n        high_evidence = float(np.mean(signs[top_tier]))\n\n    beta = float(parameters[\"beta\"])\n\n    def evidence_probabilities(evidence, gain=1.0):\n        logits = np.array([\n            0.5 * beta * float(gain) * float(evidence),\n            -0.5 * beta * float(gain) * float(evidence),\n        ], dtype=float)\n        logits -= np.max(logits)\n        p = np.exp(logits)\n        p /= p.sum()\n        return p\n\n    # Format recruitment is part of representation, not response noise.\n    # Stable format-specific gains calibrate confidence without weakening the\n    # intermediate positional profile that is needed at overload loads.\n    p_core = (\n        w_low * evidence_probabilities(\n            low_evidence,\n            low_gain_multiplier * float(parameters[\"low_evidence_gain\"])\n        )\n        + w_mid * evidence_probabilities(mid_evidence)\n        + w_high * evidence_probabilities(\n            high_evidence, float(parameters[\"high_evidence_gain\"])\n        )\n    )\n\n    # Ordinary lapses occur only after each format has generated evidence.\n    epsilon = float(parameters[\"epsilon\"])\n    probs = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])\n    probs = np.clip(probs, 0.0, None)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        return np.array([0.5, 0.5], dtype=float)\n    return probs / total\n",
  "policy": "def policy(probs) -> int:\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.clip(probs, 0.0, None)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        probs = np.ones(len(probs), dtype=np.float64) / len(probs)\n    else:\n        probs /= total\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "validities": "validities",
    "low_center": "[4.3, 4.8]",
    "high_center": "[7.0, 7.35]",
    "transition_width": "[0.9, 1.3]",
    "high_transition_width": "[0.3, 0.5]",
    "partial_tie_acceleration": "[0.9, 1.4]",
    "all_equal_delay": "[0.9, 1.3]",
    "recency_compression_coupling": "[0.25, 0.45]",
    "recency_strength": "[4.8, 6.8]",
    "validity_strength": "[0.2, 0.65]",
    "tie_rule": "{0, 1}",
    "partial_tie_low_attenuation": "[0.35, 0.6]",
    "low_evidence_gain": "[0.6, 0.85]",
    "high_evidence_gain": "[0.4, 0.65]",
    "beta": "[4.8, 6.2]",
    "epsilon": "[0.0, 0.06]"
  },
  "rationale": "This applies two localized edits to the accepted iteration-4 model while retaining its recruitment gates, recency profile, compression timing, and global format gains. First, diffuse low-format confidence is attenuated only when an observable partially tied leading tier occurs at discriminating loads 3–4. This targets the low-load terms responsible for Experiment 3's exaggerated crossover without weakening the strictly ordered low-load trials in Experiment 2 or changing the overload mechanisms supporting Experiments 4 and 6. The attenuation is sampled per subject, preserving stable heterogeneity rather than introducing trial noise. Second, the pooling tied-tier convention now uses the mean signed evidence instead of collapsing every nonzero tier sum to full-strength evidence. Consequently, a 2-to-1 tier majority is weaker than unanimity while retaining its direction. This should reduce excessive subject polarization in Experiment 5 and soften Experiment 1's residual congruence effect, while leaving the first-member subgroup unchanged. No rejected broad change to partial-tier acceleration, all-equal delay, high-format heterogeneity, or compression timing is repeated."
}
```

## Usage

```json
{
  "input_tokens": 19805,
  "output_tokens": 3323
}
```
