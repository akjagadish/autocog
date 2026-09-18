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
- THEORY 1 = `pi_11`
- THEORY 2 = `pi_13`
- The recommendation below acts on THEORY 2 (= `pi_13`).

Replace pi_13 with a new theory of Coherence-Constrained Source Clustering. Observers should infer a small set of latent reliability strata for each agreeing coalition rather than classify every positive gap as individuation evidence. Coalition value should combine: (1) weak contextual reliability weighting; (2) a redundancy cost when multiple experts are assigned to one exact or near-exact source; (3) an independence benefit for distinguishable sources; (4) a consensus-coherence term that declines when individuated sources are too dispersed; and (5) a structural-compressibility bonus when several observations support a small, balanced set of reliability clusters. This creates a Goldilocks relation between reliability gap and corroboration rather than either theory's monotonic rule. An exact dyad can be treated as one redundant source, a slightly nonidentical dyad as coherent independent corroboration, and a broader dyad as less consensual, addressing Experiments 2 and 24. A tetrad with two replicated endpoints can be represented as two well-supported source strata and beat four weakly supported singleton levels, addressing Experiment 1, while a second duplicate group can still incur an accumulating redundancy cost as in Experiment 18. The theory should retain exchangeable cue salience, smooth cardinality saturation, and no intrinsic position, onset, closure, or recency gradient, preserving the many null positional results. Its model should use continuous Bayesian or soft-clustering assignments, a nonmonotonic gap-to-corroboration function, and separate parameters for within-cluster redundancy, between-cluster coherence, cluster-support confidence, and capacity saturation. Parameter ranges should permit strong narrow-over-broad and patterned-over-unstructured preferences without forcing broad reliability-gap effects in Experiments 13 and 20–22.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_7` (overall score: 0.677)

**Description**
Reliability-Schema Source Inference proposes that decision makers treat expert agreement as evidence generated by latent informational sources rather than as conditionally independent votes. Advertised validities identify experts' reliability schemas. Dependence inference operates at two scales: an extremely narrow component assigns nearly identical, jointly extreme experts a high probability of sharing a source, while a narrower but stronger shoulder captures partial dependence among merely similar reliability schemas. For two-member coalitions, an additional equality-specific jointly extreme source prior produces nonlinear skepticism toward exactly matched reliable experts without increasing dependence for larger coalitions. Thus equally extreme coalitions can be discounted sharply without treating moderately heterogeneous coalitions as fully independent. Every discriminating cue contributes, but its contribution is divided by inferred latent-source overlap. Coalition evidence has diminishing returns, and nominal count corroboration is capped after three members. The transition from two to three members receives an additional corroboration gain only to the extent that the coalition represents informationally diverse sources. Stable subject-level priors over source dependence and count-versus-independence tradeoffs produce heterogeneous three-versus-two judgments without rewarding arbitrarily large coalitions. Communicated reliability differences are strongly compressed, preventing a single advertised-best expert from dominating a diverse majority. The computation is exchangeable over expert identities and contains no onset, closure, recency, adjacency, entry, exit, or other spatial parsing mechanism.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Reliability-Schema Source Inference expects shape "
            f"(2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = int(stim.shape[1])

    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    validity_compression = float(parameters["validity_compression"])
    validity_reliance = float(parameters["validity_reliance"])
    coalition_saturation = float(parameters["coalition_saturation"])
    dependence_prior = float(parameters["dependence_prior"])
    equality_pair_strength = float(parameters["equality_pair_strength"])
    similarity_bandwidth = float(parameters["similarity_bandwidth"])
    shoulder_strength = float(parameters["shoulder_strength"])
    shoulder_bandwidth = float(parameters["shoulder_bandwidth"])
    extremity_threshold = float(parameters["extremity_threshold"])
    extremity_slope = float(parameters["extremity_slope"])
    overlap_curvature = float(parameters["overlap_curvature"])
    count_independence_tradeoff = float(
        parameters["count_independence_tradeoff"]
    )
    diverse_third_gain = float(parameters["diverse_third_gain"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Advertised reliability is represented, but ordinal differences are
    # strongly compressed before use. The mean normalization keeps overall
    # evidence scale comparable across experiments with different validity
    # schedules.
    v = np.clip(validities, 0.500001, 0.999999)
    diagnosticity = np.log(v / (1.0 - v))
    compressed = np.power(np.maximum(diagnosticity, 1e-12), validity_compression)
    compressed /= max(float(np.mean(compressed)), 1e-12)
    cue_weights = (
        (1.0 - validity_reliance) * np.ones(n_features, dtype=np.float64)
        + validity_reliance * compressed
    )
    cue_weights /= max(float(np.mean(cue_weights)), 1e-12)

    differences = a - b
    a_support = np.flatnonzero(differences > 0.0)
    b_support = np.flatnonzero(differences < 0.0)

    if a_support.size == 0 and b_support.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    def coalition_signal(indices):
        idx = np.asarray(indices, dtype=int)
        n = int(idx.size)
        if n == 0:
            return 0.0
        if n == 1:
            return float(cue_weights[idx[0]])

        coalition_v = v[idx]
        overlap = np.zeros(n, dtype=np.float64)
        pair_posteriors = []

        # The narrow component detects almost identical reliability schemas.
        # A moderate-width shoulder represents partial common-source
        # inference for nonidentical but similar experts. An extra narrow
        # prior applies only to jointly extreme two-member coalitions, keeping
        # exact-pair skepticism separate from dependence among larger groups.
        for i in range(n):
            for j in range(i + 1, n):
                reliability_gap = float(coalition_v[i] - coalition_v[j])
                narrow_similarity = np.exp(
                    -0.5 * (reliability_gap / similarity_bandwidth) ** 2
                )
                shoulder_similarity = np.exp(
                    -0.5 * (reliability_gap / shoulder_bandwidth) ** 2
                )
                pair_extremity = 0.5 * float(coalition_v[i] + coalition_v[j])
                extreme_gate = 1.0 / (
                    1.0
                    + np.exp(
                        -np.clip(
                            extremity_slope
                            * (pair_extremity - extremity_threshold),
                            -50.0,
                            50.0,
                        )
                    )
                )

                narrow_affinity = (
                    dependence_prior
                    * narrow_similarity
                    * (0.01 + 0.99 * extreme_gate)
                )
                if n == 2:
                    narrow_affinity += (
                        equality_pair_strength
                        * narrow_similarity
                        * extreme_gate
                    )
                shoulder_affinity = (
                    shoulder_strength
                    * shoulder_similarity
                    * (0.10 + 0.90 * extreme_gate)
                )
                source_affinity = narrow_affinity + shoulder_affinity
                overlap[i] += source_affinity
                overlap[j] += source_affinity
                pair_posteriors.append(
                    source_affinity / (1.0 + source_affinity)
                )

        # Shared-source overlap is converted superlinearly into redundancy.
        # This permits sharp skepticism toward exactly equal extreme pairs
        # while preserving useful evidence from heterogeneous additions.
        independent_contributions = cue_weights[idx] / np.power(
            1.0 + overlap, overlap_curvature
        )
        independent_total = float(np.sum(independent_contributions))

        signal = independent_total / (float(n) ** coalition_saturation)

        # Nominal count corroboration is capped after the third member rather
        # than growing exponentially throughout a large coalition.
        capped_count_steps = float(min(n - 1, 2))
        signal *= np.exp(
            np.clip(
                count_independence_tradeoff * capped_count_steps,
                -10.0,
                10.0,
            )
        )

        # Crossing from two to at least three cues provides corroboration only
        # when those cues are inferred to represent informationally diverse
        # sources. The bonus is then capped, so fourth and later cues do not
        # receive repeated generic count rewards.
        if n >= 3:
            mean_shared_source = float(np.mean(pair_posteriors))
            effective_diversity = np.clip(
                1.0 - mean_shared_source, 0.0, 1.0
            )
            signal *= np.exp(
                np.clip(diverse_third_gain * effective_diversity, -10.0, 10.0)
            )

        return float(signal)

    evidence_a = coalition_signal(a_support)
    evidence_b = coalition_signal(b_support)
    net_a = evidence_a - evidence_b

    logits = np.array(
        [0.5 * beta * net_a, -0.5 * beta * net_a], dtype=np.float64
    )
    logits -= np.max(logits)
    probs = np.exp(logits)
    probs /= probs.sum()

    probs = (1.0 - epsilon) * probs + epsilon * 0.5
    probs = np.maximum(probs, 0.0)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
    return probs / total


`policy(probs) -> int`:
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.maximum(probs, 0.0)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- validities: validities
- validity_compression: [0.14, 0.30]
- validity_reliance: [0.01, 0.06]
- coalition_saturation: [0.62, 0.72]
- dependence_prior: [2.10, 3.90]
- equality_pair_strength: [0.60, 1.80]
- similarity_bandwidth: [0.010, 0.022]
- shoulder_strength: [0.62, 0.92]
- shoulder_bandwidth: [0.032, 0.065]
- extremity_threshold: [0.72, 0.78]
- extremity_slope: [20.0, 30.0]
- overlap_curvature: [2.10, 3.10]
- count_independence_tradeoff: [-0.02, 0.16]
- diverse_third_gain: [0.20, 0.40]
- beta: [1.20, 1.48]
- epsilon: [0.02, 0.06]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.1217 (var=0.0117) vs this=-0.3083 (var=0.0280)
- Experiment 2: real=-0.2642 (var=0.0087) vs this=-0.0892 (var=0.0075)
- Experiment 3: real=-0.0410 (var=0.0003) vs this=-0.0224 (var=0.0002)
- Experiment 4: real=0.5087 (var=0.0038) vs this=0.4932 (var=0.0039)
- Experiment 5: real=0.5021 (var=0.0024) vs this=0.5052 (var=0.0023)
- Experiment 6: real=0.4400 (var=0.2464) vs this=0.3200 (var=0.2176)
- Experiment 7: real=0.0219 (var=0.0094) vs this=-0.0108 (var=0.0126)
- Experiment 8: real=0.0117 (var=0.0101) vs this=-0.0021 (var=0.0101)
- Experiment 9: real=0.4958 (var=0.0041) vs this=0.4946 (var=0.0027)
- Experiment 10: real=0.1723 (var=0.0081) vs this=0.1814 (var=0.0023)
- Experiment 11: real=0.0317 (var=0.0030) vs this=0.2729 (var=0.0090)
- Experiment 12: real=0.6692 (var=0.0068) vs this=0.5575 (var=0.0028)
- Experiment 13: real=0.3242 (var=0.0194) vs this=0.3412 (var=0.0162)
- Experiment 14: real=0.5869 (var=0.0047) vs this=0.5816 (var=0.0036)
- Experiment 15: real=0.0142 (var=0.0045) vs this=0.0317 (var=0.0068)
- Experiment 16: real=0.6779 (var=0.0032) vs this=0.6562 (var=0.0033)
- Experiment 17: real=-0.3486 (var=0.0131) vs this=-0.3008 (var=0.0082)
- Experiment 18: real=0.4983 (var=0.0018) vs this=0.4525 (var=0.0026)
- Experiment 19: real=-0.0283 (var=0.0129) vs this=0.0192 (var=0.0098)
- Experiment 20: real=0.5042 (var=0.0027) vs this=0.4792 (var=0.0023)
- Experiment 21: real=0.6400 (var=0.2304) vs this=1.0000 (var=0.0000)
- Experiment 22: real=0.6817 (var=0.0026) vs this=0.5088 (var=0.0022)
- Experiment 23: real=-0.3542 (var=0.0111) vs this=-0.0146 (var=0.0112)
- Experiment 24: real=0.3208 (var=0.0038) vs this=0.5267 (var=0.0024)


---

### `pi_13` (overall score: 0.672)

**Description**
Adaptive Source-Resolution Compression with Allocated Source Quality proposes that observers represent an agreeing coalition as a probabilistic collection of latent information sources rather than as a tally of experts. Exact equality in communicated reliability is a discrete but non-deterministic common-source cue, whereas any nonzero reliability difference supports source individuation. Pairwise dependence posteriors determine both a continuous effective-source count and cue-specific uniqueness credits. These credits allocate the inferred source count across coalition members: a cue strongly dependent on its partners receives less ownership of source evidence, while an individuated cue retains more ownership. Coalition quality is therefore computed from the reliability and salience of the cues that own independent-source credit, rather than from the unadjusted arithmetic mean of all agreeing cues. The first inferred source is represented fully, while additional sources undergo subject-specific, smoothly saturating compression. Reliability-range and ordinal-stratum coverage provide a diversity benefit that decays with coalition load. The same quality-adjusted compressed source representation enters opponent-relative conflict, allowing several independent moderate cues to outweigh a highly reliable but redundant cue without introducing cardinality-specific rules. Stable heterogeneity in equality inference, compression capacity, diversity sensitivity, source-credit allocation, conflict weighting, and exchangeable cue salience produces graded population prevalence and preserves positional exchangeability.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Adaptive Source-Resolution Compression expects shape "
            f"(2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = int(stim.shape[1])

    v = np.asarray(parameters["validities"], dtype=np.float64)
    if v.ndim != 1 or v.size != n_features:
        raise ValueError(
            f"validities length {v.size} != n_features {n_features}."
        )
    v = np.clip(v, 0.500001, 0.999999)

    raw_salience = np.asarray(
        parameters["cue_salience_profile"], dtype=np.float64
    )
    if raw_salience.ndim != 1 or raw_salience.size != n_features:
        raise ValueError(
            f"cue_salience_profile length {raw_salience.size} != "
            f"n_features {n_features}."
        )

    source_dependence_prior = float(parameters["source_dependence_prior"])
    equality_evidence = float(parameters["equality_evidence"])
    nonzero_separation = float(parameters["nonzero_separation"])
    separation_floor = float(parameters["separation_floor"])
    positive_gap_rank_gain = float(parameters["positive_gap_rank_gain"])
    compression_capacity = float(parameters["compression_capacity"])
    load_individuation = float(parameters["load_individuation"])
    source_signal_curvature = float(parameters["source_signal_curvature"])
    source_allocation_curvature = float(
        parameters["source_allocation_curvature"]
    )
    rank_reliance = float(parameters["rank_reliance"])
    diversity_sensitivity = float(parameters["diversity_sensitivity"])
    coverage_mix = float(parameters["coverage_mix"])
    salience_strength = float(parameters["salience_strength"])
    conflict_gain = float(parameters["conflict_gain"])
    conflict_scale = float(parameters["conflict_scale"])
    best_cue_tradeoff = float(parameters["best_cue_tradeoff"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Contextual ordinal reliability representation. Ties share a rank, and
    # cue identity does not enter the population-level weighting rule.
    unique_v, inverse, counts = np.unique(
        v, return_inverse=True, return_counts=True
    )
    n_levels = int(unique_v.size)
    if n_levels == 1:
        level_rank = np.full(1, 0.5, dtype=np.float64)
    else:
        cumulative = np.cumsum(counts).astype(np.float64)
        midranks = cumulative - 0.5 * counts
        level_rank = midranks / float(n_features)

    cue_rank = level_rank[inverse]
    rank_contrast = 2.0 * (cue_rank - float(np.mean(cue_rank)))
    reliability_quality = np.exp(
        np.clip(rank_reliance * rank_contrast, -10.0, 10.0)
    )

    # Stable salience is centered and exchangeable across identities. It can
    # create individual configuration preferences without a common onset,
    # spacing, closure, or recency gradient.
    centered_salience = raw_salience - float(np.mean(raw_salience))
    attention = np.exp(
        np.clip(salience_strength * centered_salience, -10.0, 10.0)
    )
    attention /= max(float(np.mean(attention)), 1e-12)

    cue_quality = reliability_quality * attention
    cue_quality /= max(float(np.mean(cue_quality)), 1e-12)

    # Build the task-relative distribution of strictly positive reliability
    # differences. Equality remains a separate probabilistic cue.
    positive_gaps = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            gap = abs(float(v[i] - v[j]))
            if gap > 0.0:
                positive_gaps.append(gap)
    positive_gaps = np.sort(np.asarray(positive_gaps, dtype=np.float64))

    task_range = float(np.max(v) - np.min(v))
    task_rank_sd = float(np.std(cue_rank))

    difference = a - b
    a_support = np.flatnonzero(difference > 0.0)
    b_support = np.flatnonzero(difference < 0.0)

    if a_support.size == 0 and b_support.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    def coalition_representation(indices):
        idx = np.asarray(indices, dtype=np.int64)
        n = int(idx.size)
        if n == 0:
            return 0.0, 0.0, 0.0

        dependence_load = np.zeros(n, dtype=np.float64)

        if n == 1:
            effective_sources = 1.0
        else:
            dependence_sum = 0.0

            for ii in range(n):
                for jj in range(ii + 1, n):
                    gap = abs(float(v[idx[ii]] - v[idx[jj]]))

                    if gap == 0.0:
                        # Equality is strong but nonabsolute, with broad
                        # subject-level variation in its interpretation.
                        logit_dependence = (
                            source_dependence_prior + equality_evidence
                        )
                    else:
                        # Positive gaps primarily share one separate-source
                        # category. Gap rank supplies only a small contextual
                        # modulation, avoiding a broad metric gradient.
                        if positive_gaps.size == 0:
                            gap_percentile = 1.0
                        else:
                            gap_percentile = float(
                                np.searchsorted(
                                    positive_gaps, gap, side="right"
                                )
                            ) / float(positive_gaps.size)
                        separation = np.clip(
                            separation_floor
                            + positive_gap_rank_gain * gap_percentile,
                            0.0,
                            1.0,
                        )
                        logit_dependence = (
                            source_dependence_prior
                            - nonzero_separation * separation
                        )

                    dependence = 1.0 / (
                        1.0 + np.exp(-np.clip(logit_dependence, -50.0, 50.0))
                    )

                    # Dependence information declines gradually with load.
                    # The reduced decline preserves equality effects when a
                    # dyad becomes part of a triad without introducing a gate.
                    load = float(max(n - 2, 0))
                    retained_dependence = dependence * np.exp(
                        -load_individuation
                        * load
                        / max(compression_capacity, 1e-12)
                    )
                    dependence_sum += retained_dependence
                    dependence_load[ii] += retained_dependence
                    dependence_load[jj] += retained_dependence

            mean_dependence_load = 2.0 * dependence_sum / float(n)
            effective_sources = float(n) / (1.0 + mean_dependence_load)
            effective_sources = float(
                np.clip(effective_sources, 1.0, float(n))
            )

        # Allocate the accepted scalar source count across individual cues.
        # Dependence lowers a cue's ownership of independent-source evidence.
        # Normalization preserves the original effective-source magnitude, so
        # allocation changes source quality without adding a second count term.
        uniqueness = np.power(
            1.0 + dependence_load,
            -source_allocation_curvature,
        )
        uniqueness_total = max(float(np.sum(uniqueness)), 1e-12)
        source_credits = uniqueness * effective_sources / uniqueness_total
        allocated_quality = float(
            np.dot(source_credits, cue_quality[idx])
            / max(effective_sources, 1e-12)
        )

        # The first inferred source is fully represented. Additional sources
        # are capacity-limited and then compressed by a common smooth
        # curvature. This sharply reduces raw-count leverage without a
        # cardinality-specific rule.
        additional_sources = compression_capacity * (
            1.0
            - np.exp(
                -(effective_sources - 1.0)
                / max(compression_capacity, 1e-12)
            )
        )
        compressed_sources = 1.0 + additional_sources
        source_code = compressed_sources ** source_signal_curvature
        quality_adjusted_source_code = source_code * allocated_quality

        coalition_v = v[idx]
        coalition_ranks = cue_rank[idx]

        if n <= 1 or task_range <= 1e-12:
            range_coverage = 0.0
        else:
            range_coverage = float(
                (np.max(coalition_v) - np.min(coalition_v)) / task_range
            )

        if n <= 1 or task_rank_sd <= 1e-12:
            stratum_coverage = 0.0
        else:
            stratum_coverage = float(
                np.clip(
                    np.std(coalition_ranks) / task_rank_sd,
                    0.0,
                    1.0,
                )
            )

        coverage = (
            coverage_mix * range_coverage
            + (1.0 - coverage_mix) * stratum_coverage
        )
        coverage = float(np.clip(coverage, 0.0, 1.0))

        # Diversity is available at low load but decays increasingly strongly
        # for tetrads and larger coalitions through one continuous function.
        load_retention = np.exp(
            -float(max(n - 1, 0))
            / max(compression_capacity, 1e-12)
        )
        diversity_factor = 1.0 + (
            diversity_sensitivity * coverage * load_retention
        )

        best_quality = float(np.max(cue_quality[idx]))
        signal = quality_adjusted_source_code * diversity_factor
        return (
            float(signal),
            float(quality_adjusted_source_code),
            best_quality,
        )

    signal_a, sources_a, best_a = coalition_representation(a_support)
    signal_b, sources_b, best_b = coalition_representation(b_support)

    source_difference = sources_a - sources_b
    best_difference = best_a - best_b

    # Conflict arbitration uses the same quality-adjusted, capacity-limited
    # source code as coalition evidence. Thus neither raw cardinality nor an
    # unallocated arithmetic mean can bypass probabilistic source inference.
    conflict_input = (
        source_difference + best_cue_tradeoff * best_difference
    ) / max(conflict_scale, 1e-12)
    relational_conflict = conflict_gain * np.tanh(conflict_input)

    net_a = (signal_a - signal_b) + relational_conflict

    logits = np.array(
        [0.5 * beta * net_a, -0.5 * beta * net_a], dtype=np.float64
    )
    logits -= np.max(logits)
    probs = np.exp(logits)
    probs /= probs.sum()

    probs = (1.0 - epsilon) * probs + epsilon * 0.5
    probs = np.maximum(probs, 0.0)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
    return probs / total


`policy(probs) -> int`:
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.maximum(probs, 0.0)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- validities: validities
- cue_salience_profile: [(-1.0, 1.0)] * n_features
- source_dependence_prior: [-2.20, 0.10]
- equality_evidence: [1.80, 5.20]
- nonzero_separation: [3.20, 5.80]
- separation_floor: [0.58, 0.78]
- positive_gap_rank_gain: [0.03, 0.14]
- compression_capacity: [1.00, 2.80]
- load_individuation: [0.18, 0.55]
- source_signal_curvature: [0.40, 0.66]
- source_allocation_curvature: [0.70, 1.60]
- rank_reliance: [0.008, 0.035]
- diversity_sensitivity: [0.12, 0.85]
- coverage_mix: [0.45, 0.75]
- salience_strength: [0.12, 0.34]
- conflict_gain: [0.04, 0.16]
- conflict_scale: [0.55, 1.10]
- best_cue_tradeoff: [0.05, 0.20]
- beta: [1.35, 1.90]
- epsilon: [0.025, 0.070]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.1217 (var=0.0117) vs this=-0.4079 (var=0.0298)
- Experiment 2: real=-0.2642 (var=0.0087) vs this=0.0375 (var=0.0119)
- Experiment 3: real=-0.0410 (var=0.0003) vs this=0.0037 (var=0.0003)
- Experiment 4: real=0.5087 (var=0.0038) vs this=0.5116 (var=0.0099)
- Experiment 5: real=0.5021 (var=0.0024) vs this=0.5056 (var=0.0030)
- Experiment 6: real=0.4400 (var=0.2464) vs this=0.7400 (var=0.1924)
- Experiment 7: real=0.0219 (var=0.0094) vs this=0.0113 (var=0.0089)
- Experiment 8: real=0.0117 (var=0.0101) vs this=0.0233 (var=0.0214)
- Experiment 9: real=0.4958 (var=0.0041) vs this=0.4948 (var=0.0023)
- Experiment 10: real=0.1723 (var=0.0081) vs this=0.3822 (var=0.0076)
- Experiment 11: real=0.0317 (var=0.0030) vs this=0.1450 (var=0.0154)
- Experiment 12: real=0.6692 (var=0.0068) vs this=0.5746 (var=0.0084)
- Experiment 13: real=0.3242 (var=0.0194) vs this=0.1496 (var=0.0263)
- Experiment 14: real=0.5869 (var=0.0047) vs this=0.5269 (var=0.0058)
- Experiment 15: real=0.0142 (var=0.0045) vs this=-0.1233 (var=0.0085)
- Experiment 16: real=0.6779 (var=0.0032) vs this=0.5477 (var=0.0053)
- Experiment 17: real=-0.3486 (var=0.0131) vs this=-0.1662 (var=0.0141)
- Experiment 18: real=0.4983 (var=0.0018) vs this=0.5371 (var=0.0037)
- Experiment 19: real=-0.0283 (var=0.0129) vs this=0.0083 (var=0.0145)
- Experiment 20: real=0.5042 (var=0.0027) vs this=0.5681 (var=0.0054)
- Experiment 21: real=0.6400 (var=0.2304) vs this=0.8200 (var=0.1476)
- Experiment 22: real=0.6817 (var=0.0026) vs this=0.6694 (var=0.0048)
- Experiment 23: real=-0.3542 (var=0.0111) vs this=0.1104 (var=0.0254)
- Experiment 24: real=0.3208 (var=0.0038) vs this=0.4952 (var=0.0056)


---

### `pi_10` (overall score: 0.645)

**Description**
Categorical Source-Budget Inference proposes that people recode instructed expert reliabilities into a small number of latent information-source categories. Category membership is inferred locally: experts with exactly or nearly matching represented reliabilities are likely to be treated as exchangeable observations from one source, whereas this coassignment declines sharply outside a narrow subject-specific neighborhood. Within each option-supporting coalition, members assigned to one source draw from a shared evidence budget whose value grows as a strongly concave power of cluster membership. Exact reliability matches additionally invoke a categorical equality atom, producing an almost flat shared budget without changing how nonzero reliability gaps are partitioned. Budgets belonging to distinct inferred sources are added. A duplicate pair is therefore redundant in a dyad, an unrelated outsider adds a separate budget without changing that duplicate cost, and two duplicate clusters consume two separate budgets. The total is then subjected to generic nominal-coalition normalization and a separately calibrated saturating release that is concentrated between two and three effective sources. Instructed validity differences have little direct influence after strong compression. Stable, exchangeable, zero-centered cue salience permits individual configuration preferences but creates no population-level serial-position mechanism. With no correctness feedback, source assignments and cue salience remain fixed across trials.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Categorical Source-Budget Inference expects shape "
            f"(2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = int(stim.shape[1])

    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    raw_salience = np.asarray(
        parameters["cue_salience_profile"], dtype=np.float64
    )
    if raw_salience.ndim != 1 or raw_salience.size != n_features:
        raise ValueError(
            f"cue_salience_profile length {raw_salience.size} != "
            f"n_features {n_features}."
        )

    direct_validity_compression = float(
        parameters["direct_validity_compression"]
    )
    validity_reliance = float(parameters["validity_reliance"])
    salience_strength = float(parameters["salience_strength"])
    category_width = float(parameters["category_width"])
    category_boundary_shift = float(parameters["category_boundary_shift"])
    within_source_curvature = float(parameters["within_source_curvature"])
    equality_source_curvature = float(
        parameters["equality_source_curvature"]
    )
    coalition_normalization = float(parameters["coalition_normalization"])
    source_count_curvature = float(parameters["source_count_curvature"])
    source_count_midpoint = float(parameters["source_count_midpoint"])
    source_count_slope = float(parameters["source_count_slope"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    v = np.clip(validities, 0.500001, 0.999999)

    # Reliability has only a weak direct effect. Diagnosticity differences
    # are strongly compressed around a fixed .75 reference so the scale is
    # comparable across experiments.
    diagnosticity = np.log(v / (1.0 - v))
    reference = np.log(0.75 / 0.25)
    represented_direct = np.power(
        np.maximum(diagnosticity, 1e-12), direct_validity_compression
    )
    represented_direct /= max(
        float(reference ** direct_validity_compression), 1e-12
    )
    reliability_weight = (
        (1.0 - validity_reliance) * np.ones(n_features, dtype=np.float64)
        + validity_reliance * represented_direct
    )

    # Idiosyncratic salience is exchangeable across identities and explicitly
    # centered, so it cannot generate a common onset, recency, or closure rule.
    centered_salience = raw_salience - float(np.mean(raw_salience))
    attention = np.exp(
        np.clip(salience_strength * centered_salience, -10.0, 10.0)
    )
    attention /= max(float(np.mean(attention)), 1e-12)

    cue_weights = reliability_weight * attention
    cue_weights /= max(float(np.mean(cue_weights)), 1e-12)

    # Category uncertainty controls a narrow local coassignment radius.
    # Exact matches are always coassigned; nonidentical reliabilities are
    # coassigned only inside a sharply localized subject-specific region.
    coassignment_radius = category_width * np.exp(category_boundary_shift)

    difference = a - b
    a_support = np.flatnonzero(difference > 0.0)
    b_support = np.flatnonzero(difference < 0.0)

    if a_support.size == 0 and b_support.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    def coalition_signal(indices):
        idx = np.asarray(indices, dtype=int)
        n = int(idx.size)
        if n == 0:
            return 0.0

        # Infer a local source partition by joining only reliability-near
        # coalition members. Connected components are exchangeable in expert
        # identity and produce no repeated global boundary artifacts.
        parent = np.arange(n, dtype=np.int64)

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = int(parent[i])
            return i

        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_j] = root_i

        coalition_v = v[idx]
        for i in range(n):
            for j in range(i + 1, n):
                if abs(float(coalition_v[i] - coalition_v[j])) <= coassignment_radius:
                    union(i, j)

        labels = np.asarray([find(i) for i in range(n)], dtype=np.int64)
        unique_labels = np.unique(labels)
        source_budget_total = 0.0

        # Experts assigned to one latent source share a cluster-level budget.
        # Exact reliability matches invoke a categorical equality atom with
        # near-flat membership growth. Nonidentical local clusters retain the
        # accepted model's original pooling curvature.
        for label in unique_labels:
            local_members = np.flatnonzero(labels == label)
            members = idx[local_members]
            membership = float(members.size)
            source_quality = float(np.mean(cue_weights[members]))
            local_validities = coalition_v[local_members]
            exact_match = (
                members.size > 1
                and float(np.max(local_validities) - np.min(local_validities))
                <= 1e-12
            )
            cluster_curvature = (
                equality_source_curvature
                if exact_match
                else within_source_curvature
            )
            source_budget_total += source_quality * (
                membership ** cluster_curvature
            )

        effective_source_count = float(unique_labels.size)

        # Source budgets already add independent-source evidence. The extra
        # count calibration is therefore bounded and releases primarily when
        # effective source count crosses from two to three, rather than
        # multiplying evidence by an unbounded second power of source count.
        release = 1.0 / (
            1.0
            + np.exp(
                -np.clip(
                    source_count_slope
                    * (effective_source_count - source_count_midpoint),
                    -50.0,
                    50.0,
                )
            )
        )
        count_factor = np.exp(source_count_curvature * release)
        nominal_normalizer = float(n) ** coalition_normalization
        signal = source_budget_total * count_factor / max(
            nominal_normalizer, 1e-12
        )
        return float(max(signal, 0.0))

    evidence_a = coalition_signal(a_support)
    evidence_b = coalition_signal(b_support)
    net_a = evidence_a - evidence_b

    logits = np.array(
        [0.5 * beta * net_a, -0.5 * beta * net_a], dtype=np.float64
    )
    logits -= np.max(logits)
    probs = np.exp(logits)
    probs /= probs.sum()

    probs = (1.0 - epsilon) * probs + epsilon * 0.5
    probs = np.maximum(probs, 0.0)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
    return probs / total


`policy(probs) -> int`:
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.maximum(probs, 0.0)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- validities: validities
- cue_salience_profile: [(-1.0, 1.0)] * n_features
- direct_validity_compression: [0.08, 0.14]
- validity_reliance: [0.006, 0.024]
- salience_strength: [0.18, 0.40]
- category_width: [0.008, 0.018]
- category_boundary_shift: [-0.30, 0.30]
- within_source_curvature: [0.08, 0.22]
- equality_source_curvature: [0.00, 0.05]
- coalition_normalization: [0.88, 0.98]
- source_count_curvature: [0.18, 0.26]
- source_count_midpoint: [2.45, 2.65]
- source_count_slope: [7.0, 11.0]
- beta: [1.45, 1.85]
- epsilon: [0.03, 0.08]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.1217 (var=0.0117) vs this=-0.1767 (var=0.0260)
- Experiment 2: real=-0.2642 (var=0.0087) vs this=0.0396 (var=0.0103)
- Experiment 3: real=-0.0410 (var=0.0003) vs this=0.0007 (var=0.0004)
- Experiment 4: real=0.5087 (var=0.0038) vs this=0.4788 (var=0.0086)
- Experiment 5: real=0.5021 (var=0.0024) vs this=0.4923 (var=0.0026)
- Experiment 6: real=0.4400 (var=0.2464) vs this=0.0600 (var=0.0564)
- Experiment 7: real=0.0219 (var=0.0094) vs this=-0.0065 (var=0.0117)
- Experiment 8: real=0.0117 (var=0.0101) vs this=-0.0167 (var=0.0081)
- Experiment 9: real=0.4958 (var=0.0041) vs this=0.4925 (var=0.0029)
- Experiment 10: real=0.1723 (var=0.0081) vs this=0.3205 (var=0.0042)
- Experiment 11: real=0.0317 (var=0.0030) vs this=0.0933 (var=0.0083)
- Experiment 12: real=0.6692 (var=0.0068) vs this=0.7271 (var=0.0030)
- Experiment 13: real=0.3242 (var=0.0194) vs this=0.4242 (var=0.0096)
- Experiment 14: real=0.5869 (var=0.0047) vs this=0.6191 (var=0.0035)
- Experiment 15: real=0.0142 (var=0.0045) vs this=0.0337 (var=0.0098)
- Experiment 16: real=0.6779 (var=0.0032) vs this=0.6777 (var=0.0028)
- Experiment 17: real=-0.3486 (var=0.0131) vs this=-0.1210 (var=0.0132)
- Experiment 18: real=0.4983 (var=0.0018) vs this=0.2913 (var=0.0022)
- Experiment 19: real=-0.0283 (var=0.0129) vs this=-0.0008 (var=0.0157)
- Experiment 20: real=0.5042 (var=0.0027) vs this=0.4062 (var=0.0025)
- Experiment 21: real=0.6400 (var=0.2304) vs this=1.0000 (var=0.0000)
- Experiment 22: real=0.6817 (var=0.0026) vs this=0.5156 (var=0.0026)
- Experiment 23: real=-0.3542 (var=0.0111) vs this=0.0008 (var=0.0095)
- Experiment 24: real=0.3208 (var=0.0038) vs this=0.5956 (var=0.0110)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.2497 -> ACCEPTED
- iter 2: loss=0.2525 -> REJECTED
- iter 3: loss=0.2471 -> ACCEPTED
- iter 4: loss=0.2329 -> ACCEPTED
- iter 5: loss=0.2705 -> REJECTED
- iter 6: loss=0.2269 -> ACCEPTED
- iter 7: loss=0.2338 -> REJECTED
Running-best (last ACCEPTED) base: iter 6 at loss=0.2269 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Mean signed endorsement of the highest-validity cue's winner."""
    scores = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        response = int(row["response"])

        # Feature 0 is the uniquely highest-validity cue in this design.
        if a[0] > b[0]:
            top_cue_winner = 0
        elif b[0] > a[0]:
            top_cue_winner = 1
        else:
            continue

        scores.append(1.0 if response == top_cue_winner else -1.0)

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** -0.1217 (var=0.0117)
**Previous candidate values (this loop):**
  - iter 1: -0.0912 (var=0.0217) (Δ vs real +0.0304)
  - iter 2: 0.1046 (var=0.0238) (Δ vs real +0.2263)
  - iter 3: -0.2017 (var=0.0225) (Δ vs real -0.0800)
  - iter 4: -0.1046 (var=0.0175) (Δ vs real +0.0171)
  - iter 5: 0.1379 (var=0.0285) (Δ vs real +0.2596)
  - iter 6: -0.0833 (var=0.0330) (Δ vs real +0.0383)
  - iter 7 (most recent): -0.0746 (var=0.0251) (Δ vs real +0.0471)
**Other theories' values on this metric (for reference):**
- pi_1: 0.7129 (var=0.0389)
- pi_2: -0.6354 (var=0.0266)
- pi_3: -0.1033 (var=0.0092)
- pi_4: -0.1246 (var=0.0114)
- pi_5: -0.1179 (var=0.0105)
- pi_6: -0.0779 (var=0.0176)
- pi_7: -0.3083 (var=0.0280)
- pi_8: -0.1358 (var=0.0111)
- pi_9: -0.2600 (var=0.0153)
- pi_10: -0.1767 (var=0.0260)
- pi_11: -0.2383 (var=0.0360)
- pi_12: -0.3588 (var=0.0139)
- pi_13: -0.4079 (var=0.0298)

### Experiment 2
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 1, 1, 0, 1]
  A=[0, 0, 0, 1, 1, 1, 0, 1]  B=[1, 1, 1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0, 1, 0, 1]
  A=[0, 1, 1, 0, 0, 1, 0, 1]  B=[1, 0, 0, 1, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 1, 1, 0, 0]  B=[1, 0, 1, 1, 0, 0, 1, 0]
  A=[1, 0, 1, 1, 0, 0, 1, 0]  B=[1, 1, 0, 0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 1, 1, 0]  B=[1, 0, 0, 1, 1, 0, 0, 1]
  A=[1, 0, 0, 1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Mean signed agreement with the highest-validity discriminating cue."""
    if data is None or len(data) == 0:
        return 0.0

    signed_agreements = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)

        # Features are already ordered from highest to lowest validity.
        differing = np.flatnonzero(a != b)
        if differing.size == 0:
            continue

        j = int(differing[0])
        ttb_response = 0 if a[j] > b[j] else 1
        observed_response = int(row["response"])
        signed_agreements.append(1.0 if observed_response == ttb_response else -1.0)

    if not signed_agreements:
        return 0.0
    return float(np.mean(signed_agreements))
```

**Observed (real) value:** -0.2642 (var=0.0087)
**Previous candidate values (this loop):**
  - iter 1: -0.0046 (var=0.0103) (Δ vs real +0.2596)
  - iter 2: 0.0025 (var=0.0091) (Δ vs real +0.2667)
  - iter 3: 0.0250 (var=0.0107) (Δ vs real +0.2892)
  - iter 4: 0.0358 (var=0.0086) (Δ vs real +0.3000)
  - iter 5: 0.0654 (var=0.0062) (Δ vs real +0.3296)
  - iter 6: 0.0417 (var=0.0122) (Δ vs real +0.3058)
  - iter 7 (most recent): 0.0479 (var=0.0117) (Δ vs real +0.3121)
**Other theories' values on this metric (for reference):**
- pi_2: -0.0008 (var=0.0102)
- pi_1: 0.7221 (var=0.0226)
- pi_3: -0.1892 (var=0.0123)
- pi_4: -0.2667 (var=0.0120)
- pi_5: -0.2217 (var=0.0089)
- pi_6: -0.2321 (var=0.0159)
- pi_7: -0.0892 (var=0.0075)
- pi_8: -0.2221 (var=0.0114)
- pi_9: -0.0008 (var=0.0090)
- pi_10: 0.0396 (var=0.0103)
- pi_11: 0.0096 (var=0.0132)
- pi_12: 0.0117 (var=0.0092)
- pi_13: 0.0375 (var=0.0119)

### Experiment 3
**Design**
  A=[1, 1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 0, 0, 1]
  A=[1, 0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0, 1]  B=[1, 0, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if len(data) == 0:
        return 0.0

    # Fixed, preregistered midpoint instantiation of the advocated model.
    validities = np.array([0.52, 0.58, 0.65, 0.72, 0.80, 0.88, 0.95], dtype=float)
    balance = 0.30
    recency = 2.10
    curvature = 0.825
    saturation = 0.725
    beta = 0.90
    epsilon = 0.05

    v = np.clip(validities, 0.500001, 0.999999)
    diagnosticity = np.log(v / (1.0 - v))
    diagnosticity /= np.max(diagnosticity)
    diagnosticity = np.power(np.maximum(diagnosticity, 0.0), curvature)
    diagnosticity /= np.mean(diagnosticity)

    position = np.arange(len(validities), dtype=float) / float(len(validities) - 1)
    accessibility = np.exp(recency * position)
    accessibility /= np.mean(accessibility)

    cue_weights = balance * diagnosticity + (1.0 - balance) * accessibility
    cue_weights /= np.mean(cue_weights)

    predicted_majority_probs = []
    chose_majority = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        d = a - b

        # Orient every pair toward the option winning the 3-versus-2 tally.
        tally_margin = float(np.sum(d > 0) - np.sum(d < 0))
        if tally_margin == 0.0 or len(d) != len(cue_weights):
            continue
        majority_is_a = tally_margin > 0.0
        oriented = d if majority_is_a else -d

        positive = oriented > 0.0
        negative = oriented < 0.0
        n_pos = max(int(np.sum(positive)), 1)
        n_neg = max(int(np.sum(negative)), 1)
        min_n = min(n_pos, n_neg)

        pos_evidence = float(np.dot(cue_weights, np.maximum(oriented, 0.0)))
        neg_evidence = float(np.dot(cue_weights, np.maximum(-oriented, 0.0)))
        evidence = (
            pos_evidence / (max(1.0, float(n_pos) / float(min_n)) ** saturation)
            - neg_evidence / (max(1.0, float(n_neg) / float(min_n)) ** saturation)
        )

        core_p = 1.0 / (1.0 + np.exp(-np.clip(beta * evidence, -50.0, 50.0)))
        predicted_p = (1.0 - epsilon) * core_p + epsilon * 0.5
        predicted_majority_probs.append(predicted_p)

        response = int(row["response"])
        chose_majority.append(float((response == 0) if majority_is_a else (response == 1)))

    if len(predicted_majority_probs) == 0:
        return 0.0

    p = np.asarray(predicted_majority_probs, dtype=float)
    y = np.asarray(chose_majority, dtype=float)
    contrast = p - np.mean(p)
    return float(np.mean(contrast * (y - 0.5)))

```

**Observed (real) value:** -0.0410 (var=0.0003)
**Previous candidate values (this loop):**
  - iter 1: 0.0004 (var=0.0003) (Δ vs real +0.0414)
  - iter 2: 0.0057 (var=0.0003) (Δ vs real +0.0467)
  - iter 3: 0.0103 (var=0.0003) (Δ vs real +0.0513)
  - iter 4: 0.0066 (var=0.0003) (Δ vs real +0.0476)
  - iter 5: 0.0168 (var=0.0003) (Δ vs real +0.0578)
  - iter 6: 0.0073 (var=0.0004) (Δ vs real +0.0483)
  - iter 7 (most recent): 0.0092 (var=0.0003) (Δ vs real +0.0502)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0860 (var=0.0002)
- pi_2: -0.0017 (var=0.0001)
- pi_1: 0.0899 (var=0.0009)
- pi_4: -0.0260 (var=0.0002)
- pi_5: -0.0237 (var=0.0003)
- pi_6: -0.0366 (var=0.0003)
- pi_7: -0.0224 (var=0.0002)
- pi_8: -0.0376 (var=0.0003)
- pi_9: 0.0012 (var=0.0003)
- pi_10: 0.0007 (var=0.0004)
- pi_11: 0.0039 (var=0.0005)
- pi_12: 0.0144 (var=0.0003)
- pi_13: 0.0037 (var=0.0003)

### Experiment 4
**Design**
  A=[1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[1, 1, 1, 0, 0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 1, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 1, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 1, 1, 0, 1]
  A=[1, 1, 1, 0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    weighted_choices = []
    weights = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.shape != b.shape or a.ndim != 1 or a.size < 2:
            continue

        differing = np.flatnonzero(a != b)
        if differing.size != 2:
            continue

        early = int(np.min(differing))
        late = int(np.max(differing))
        late_supports_a = bool(a[late] > b[late])
        later_option_response = 0 if late_supports_a else 1
        chose_later_supported_option = float(int(row["response"]) == later_option_response)

        # Approximate matched-filter weight for the competing theory's
        # exponential positional-accessibility contrast.
        scale = float(a.size - 1)
        weight = np.exp(2.1 * late / scale) - np.exp(2.1 * early / scale)
        if np.isfinite(weight) and weight > 0.0:
            weighted_choices.append(chose_later_supported_option)
            weights.append(weight)

    if len(weights) == 0:
        return float("nan")

    return float(np.average(np.asarray(weighted_choices), weights=np.asarray(weights)))
```

**Observed (real) value:** 0.5087 (var=0.0038)
**Previous candidate values (this loop):**
  - iter 1: 0.4763 (var=0.0052) (Δ vs real -0.0324)
  - iter 2: 0.5055 (var=0.0065) (Δ vs real -0.0032)
  - iter 3: 0.4978 (var=0.0052) (Δ vs real -0.0109)
  - iter 4: 0.4922 (var=0.0070) (Δ vs real -0.0164)
  - iter 5: 0.5146 (var=0.0058) (Δ vs real +0.0060)
  - iter 6: 0.4816 (var=0.0055) (Δ vs real -0.0270)
  - iter 7 (most recent): 0.4992 (var=0.0030) (Δ vs real -0.0095)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5017 (var=0.0041)
- pi_3: 0.6281 (var=0.0049)
- pi_1: 0.1476 (var=0.0116)
- pi_4: 0.5061 (var=0.0029)
- pi_5: 0.4957 (var=0.0036)
- pi_6: 0.5046 (var=0.0065)
- pi_7: 0.4932 (var=0.0039)
- pi_8: 0.4759 (var=0.0052)
- pi_9: 0.5033 (var=0.0044)
- pi_10: 0.4788 (var=0.0086)
- pi_11: 0.4771 (var=0.0066)
- pi_12: 0.5132 (var=0.0055)
- pi_13: 0.5116 (var=0.0099)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if len(data) == 0:
        return 0.5

    a = np.vstack(data["option_a_ratings"].apply(lambda x: np.asarray(x, dtype=float)).to_numpy())
    b = np.vstack(data["option_b_ratings"].apply(lambda x: np.asarray(x, dtype=float)).to_numpy())
    responses = pd.to_numeric(data["response"], errors="coerce").to_numpy(dtype=float)

    n_features = a.shape[1]
    if n_features == 1:
        positional_weights = np.ones(1, dtype=float)
    else:
        positions = np.arange(n_features, dtype=float) / float(n_features - 1)
        positional_weights = np.exp(2.1 * positions)

    # Positive evidence means positional accessibility favors A; negative
    # evidence means it favors B.
    accessibility_difference = np.dot(a - b, positional_weights)
    valid = np.isfinite(responses) & (accessibility_difference != 0.0)
    if not np.any(valid):
        return 0.5

    chose_a = responses[valid] == 0.0
    accessibility_favors_a = accessibility_difference[valid] > 0.0
    accessibility_aligned = chose_a == accessibility_favors_a
    return float(np.mean(accessibility_aligned))
```

**Observed (real) value:** 0.5021 (var=0.0024)
**Previous candidate values (this loop):**
  - iter 1: 0.4975 (var=0.0020) (Δ vs real -0.0046)
  - iter 2: 0.5062 (var=0.0038) (Δ vs real +0.0042)
  - iter 3: 0.5071 (var=0.0024) (Δ vs real +0.0050)
  - iter 4: 0.5081 (var=0.0033) (Δ vs real +0.0060)
  - iter 5: 0.5002 (var=0.0035) (Δ vs real -0.0019)
  - iter 6: 0.4871 (var=0.0026) (Δ vs real -0.0150)
  - iter 7 (most recent): 0.4994 (var=0.0034) (Δ vs real -0.0027)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5692 (var=0.0021)
- pi_4: 0.3798 (var=0.0029)
- pi_1: 0.1575 (var=0.0119)
- pi_2: 0.4992 (var=0.0023)
- pi_5: 0.5098 (var=0.0033)
- pi_6: 0.5123 (var=0.0025)
- pi_7: 0.5052 (var=0.0023)
- pi_8: 0.4940 (var=0.0029)
- pi_9: 0.4973 (var=0.0028)
- pi_10: 0.4923 (var=0.0026)
- pi_11: 0.5060 (var=0.0029)
- pi_12: 0.4983 (var=0.0020)
- pi_13: 0.5056 (var=0.0030)

### Experiment 6
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if data is None or len(data) == 0:
        return float("nan")

    df = data[["subject_id", "option_a_ratings", "option_b_ratings", "response"]].copy()

    def early_coalition_choice(row):
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.size < 3 or b.size < 3:
            return np.nan
        score_a = float(np.sum(a[:3]))
        score_b = float(np.sum(b[:3]))
        if score_a == score_b:
            return np.nan
        early_side = 0 if score_a > score_b else 1
        return float(int(row["response"]) == early_side)

    df["chose_early_coalition"] = df.apply(early_coalition_choice, axis=1)
    rates = df.groupby("subject_id", sort=False)["chose_early_coalition"].mean().dropna()
    if len(rates) == 0:
        return float("nan")

    # Fraction of subjects whose majority choice favors the fixed early
    # three-cue coalition rather than the later two-cue coalition.
    return float(np.mean(rates.to_numpy(dtype=float) > 0.5))
```

**Observed (real) value:** 0.4400 (var=0.2464)
**Previous candidate values (this loop):**
  - iter 1: 0.0200 (var=0.0196) (Δ vs real -0.4200)
  - iter 2: 0.0600 (var=0.0564) (Δ vs real -0.3800)
  - iter 3: 0.0800 (var=0.0736) (Δ vs real -0.3600)
  - iter 4: 0.1200 (var=0.1056) (Δ vs real -0.3200)
  - iter 5: 0.1600 (var=0.1344) (Δ vs real -0.2800)
  - iter 6: 0.1600 (var=0.1344) (Δ vs real -0.2800)
  - iter 7 (most recent): 0.1800 (var=0.1476) (Δ vs real -0.2600)
**Other theories' values on this metric (for reference):**
- pi_4: 0.9800 (var=0.0196)
- pi_3: 0.0000 (var=0.0000)
- pi_1: 1.0000 (var=0.0000)
- pi_2: 1.0000 (var=0.0000)
- pi_5: 0.3200 (var=0.2176)
- pi_6: 0.3600 (var=0.2304)
- pi_7: 0.3200 (var=0.2176)
- pi_8: 0.5000 (var=0.2500)
- pi_9: 0.1600 (var=0.1344)
- pi_10: 0.0600 (var=0.0564)
- pi_11: 0.1600 (var=0.1344)
- pi_12: 0.8400 (var=0.1344)
- pi_13: 0.7400 (var=0.1924)

### Experiment 7
**Design**
  A=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    entry_sum = 0.0
    entry_weight = 0.0
    exit_sum = 0.0
    exit_weight = 0.0

    for row in data.itertuples(index=False):
        a = np.asarray(row.option_a_ratings, dtype=float)
        b = np.asarray(row.option_b_ratings, dtype=float)
        diff = a - b
        pos = np.flatnonzero(diff > 0)
        neg = np.flatnonzero(diff < 0)

        # Identify the three-cue coalition and whether it supports A or B.
        if pos.size == 3 and neg.size == 2:
            coalition = pos
            endorsed_three = 1.0 if int(row.response) == 0 else 0.0
        elif neg.size == 3 and pos.size == 2:
            coalition = neg
            endorsed_three = 1.0 if int(row.response) == 1 else 0.0
        else:
            continue

        gaps = np.diff(coalition).astype(float)
        if gaps.size != 2:
            continue

        first_compact = bool(gaps[0] == 1 and gaps[1] > 1)
        last_compact = bool(gaps[1] == 1 and gaps[0] > 1)
        if not (first_compact or last_compact):
            continue

        # Larger separation makes entry- versus exit-compact organization
        # more distinct. This bounded weight is the difference between the
        # two adjacent-link binding strengths.
        weight = abs(np.exp(-(gaps[0] - 1.0)) - np.exp(-(gaps[1] - 1.0)))

        if first_compact:
            entry_sum += weight * endorsed_three
            entry_weight += weight
        else:
            exit_sum += weight * endorsed_three
            exit_weight += weight

    if entry_weight <= 0.0 or exit_weight <= 0.0:
        return 0.0

    return float(entry_sum / entry_weight - exit_sum / exit_weight)

```

**Observed (real) value:** 0.0219 (var=0.0094)
**Previous candidate values (this loop):**
  - iter 1: -0.0069 (var=0.0101) (Δ vs real -0.0288)
  - iter 2: -0.0086 (var=0.0111) (Δ vs real -0.0305)
  - iter 3: -0.0129 (var=0.0099) (Δ vs real -0.0348)
  - iter 4: -0.0103 (var=0.0111) (Δ vs real -0.0322)
  - iter 5: 0.0050 (var=0.0093) (Δ vs real -0.0168)
  - iter 6: -0.0145 (var=0.0076) (Δ vs real -0.0364)
  - iter 7 (most recent): -0.0148 (var=0.0091) (Δ vs real -0.0366)
**Other theories' values on this metric (for reference):**
- pi_5: 0.2360 (var=0.0123)
- pi_4: 0.0009 (var=0.0095)
- pi_1: 0.0069 (var=0.0048)
- pi_2: -0.0035 (var=0.0050)
- pi_3: -0.0376 (var=0.0081)
- pi_6: 0.0257 (var=0.0065)
- pi_7: -0.0108 (var=0.0126)
- pi_8: 0.0088 (var=0.0097)
- pi_9: -0.0034 (var=0.0107)
- pi_10: -0.0065 (var=0.0117)
- pi_11: 0.0143 (var=0.0064)
- pi_12: -0.0353 (var=0.0117)
- pi_13: 0.0113 (var=0.0089)

### Experiment 8
**Design**
  A=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    early_choices = []
    late_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        differences = a - b
        a_support = np.flatnonzero(differences > 0)
        b_support = np.flatnonzero(differences < 0)
        response = int(row["response"])

        if a_support.size == 3 and b_support.size == 2:
            onset_difference = int(a_support[0]) - int(b_support[0])
            chose_three = float(response == 0)
        elif a_support.size == 2 and b_support.size == 3:
            onset_difference = int(b_support[0]) - int(a_support[0])
            chose_three = float(response == 1)
        else:
            continue

        if onset_difference < 0:
            early_choices.append(chose_three)
        elif onset_difference > 0:
            late_choices.append(chose_three)

    if len(early_choices) == 0 or len(late_choices) == 0:
        return 0.0

    return float(np.mean(early_choices) - np.mean(late_choices))
```

**Observed (real) value:** 0.0117 (var=0.0101)
**Previous candidate values (this loop):**
  - iter 1: -0.0179 (var=0.0134) (Δ vs real -0.0296)
  - iter 2: -0.0042 (var=0.0166) (Δ vs real -0.0158)
  - iter 3: 0.0304 (var=0.0136) (Δ vs real +0.0188)
  - iter 4: -0.0033 (var=0.0134) (Δ vs real -0.0150)
  - iter 5: -0.0129 (var=0.0099) (Δ vs real -0.0246)
  - iter 6: -0.0025 (var=0.0126) (Δ vs real -0.0142)
  - iter 7 (most recent): -0.0038 (var=0.0093) (Δ vs real -0.0154)
**Other theories' values on this metric (for reference):**
- pi_4: 0.2729 (var=0.0146)
- pi_5: -0.0100 (var=0.0101)
- pi_1: 0.7325 (var=0.0409)
- pi_2: -0.0042 (var=0.0038)
- pi_3: -0.4775 (var=0.0097)
- pi_6: -0.0154 (var=0.0165)
- pi_7: -0.0021 (var=0.0101)
- pi_8: -0.0042 (var=0.0116)
- pi_9: 0.0033 (var=0.0081)
- pi_10: -0.0167 (var=0.0081)
- pi_11: -0.0054 (var=0.0118)
- pi_12: 0.0358 (var=0.0069)
- pi_13: 0.0233 (var=0.0214)

### Experiment 9
**Design**
  A=[1, 0, 1, 0, 0, 1, 1, 0]  B=[0, 1, 0, 1, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 0, 1, 1, 0]  B=[0, 1, 1, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 1, 1, 1, 0]  B=[0, 1, 1, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 0, 1, 1, 0]  B=[1, 0, 0, 1, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1, 1, 0]  B=[1, 0, 1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 1, 1, 0]  B=[1, 0, 1, 1, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 1, 1, 0]  B=[1, 1, 0, 1, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 1, 0]
  A=[0, 1, 1, 1, 0, 0, 1, 0]  B=[1, 0, 0, 0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 1, 0, 1, 0]  B=[0, 1, 1, 0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 1, 0]
  A=[1, 0, 1, 1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1, 1, 0]
  A=[1, 1, 0, 1, 0, 0, 1, 0]  B=[0, 0, 1, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    closure_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=np.float64)
        b = np.asarray(row["option_b_ratings"], dtype=np.float64)
        difference = a - b
        discriminating = np.flatnonzero(difference != 0.0)

        if discriminating.size == 0:
            continue

        # The closure coalition is the option supported by the final
        # discriminating cue, irrespective of whether it is displayed as A or B.
        last_index = int(discriminating[-1])
        closure_response = 0 if difference[last_index] > 0.0 else 1
        closure_choices.append(float(int(row["response"]) == closure_response))

    if len(closure_choices) == 0:
        return float("nan")

    return float(np.mean(np.asarray(closure_choices, dtype=np.float64)))
```

**Observed (real) value:** 0.4958 (var=0.0041)
**Previous candidate values (this loop):**
  - iter 1: 0.5038 (var=0.0029) (Δ vs real +0.0079)
  - iter 2: 0.5040 (var=0.0031) (Δ vs real +0.0081)
  - iter 3: 0.5006 (var=0.0025) (Δ vs real +0.0048)
  - iter 4: 0.5077 (var=0.0031) (Δ vs real +0.0119)
  - iter 5: 0.4977 (var=0.0027) (Δ vs real +0.0019)
  - iter 6: 0.5040 (var=0.0024) (Δ vs real +0.0081)
  - iter 7 (most recent): 0.5242 (var=0.0034) (Δ vs real +0.0283)
**Other theories' values on this metric (for reference):**
- pi_5: 0.6108 (var=0.0033)
- pi_6: 0.4996 (var=0.0028)
- pi_1: 0.4104 (var=0.0015)
- pi_2: 0.4985 (var=0.0025)
- pi_3: 0.6115 (var=0.0020)
- pi_4: 0.6271 (var=0.0029)
- pi_7: 0.4946 (var=0.0027)
- pi_8: 0.5056 (var=0.0026)
- pi_9: 0.4960 (var=0.0023)
- pi_10: 0.4925 (var=0.0029)
- pi_11: 0.5090 (var=0.0024)
- pi_12: 0.4890 (var=0.0018)
- pi_13: 0.4948 (var=0.0023)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    validities = np.asarray([
        0.68, 0.74, 0.80, 0.86, 0.92, 0.97, 0.55, 0.70,
        0.68, 0.74, 0.80, 0.86, 0.92, 0.97, 0.60, 0.70
    ], dtype=np.float64)
    bounded = np.clip(2.0 * validities - 1.0, 0.0, 1.0)

    weighted_choices = []
    weights = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=np.float64)
        b = np.asarray(row["option_b_ratings"], dtype=np.float64)
        if a.size != validities.size or b.size != validities.size:
            continue

        differences = a - b
        a_support = np.flatnonzero(differences > 0.0)
        b_support = np.flatnonzero(differences < 0.0)
        if a_support.size != 2 or b_support.size != 2:
            continue

        a_equal = bool(np.isclose(validities[a_support[0]], validities[a_support[1]]))
        b_equal = bool(np.isclose(validities[b_support[0]], validities[b_support[1]]))
        if a_equal == b_equal:
            continue

        load_a = float(bounded[a_support[0]] * bounded[a_support[1]])
        load_b = float(bounded[b_support[0]] * bounded[b_support[1]])
        weight = abs(load_a - load_b)
        if not np.isfinite(weight) or weight <= 0.0:
            continue

        response = int(row["response"])
        chose_equal = float((response == 0) if a_equal else (response == 1))
        weighted_choices.append(weight * chose_equal)
        weights.append(weight)

    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        return 0.5
    return float(np.sum(weighted_choices) / total_weight)

```

**Observed (real) value:** 0.1723 (var=0.0081)
**Previous candidate values (this loop):**
  - iter 1: 0.4301 (var=0.0047) (Δ vs real +0.2578)
  - iter 2: 0.5977 (var=0.0065) (Δ vs real +0.4253)
  - iter 3: 0.4743 (var=0.0080) (Δ vs real +0.3020)
  - iter 4: 0.4808 (var=0.0056) (Δ vs real +0.3085)
  - iter 5: 0.5806 (var=0.0058) (Δ vs real +0.4083)
  - iter 6: 0.4696 (var=0.0055) (Δ vs real +0.2973)
  - iter 7 (most recent): 0.4930 (var=0.0046) (Δ vs real +0.3207)
**Other theories' values on this metric (for reference):**
- pi_6: 0.2057 (var=0.0035)
- pi_5: 0.5023 (var=0.0031)
- pi_1: 0.8796 (var=0.0067)
- pi_2: 0.4978 (var=0.0039)
- pi_3: 0.5414 (var=0.0029)
- pi_4: 0.3753 (var=0.0039)
- pi_7: 0.1814 (var=0.0023)
- pi_8: 0.1646 (var=0.0019)
- pi_9: 0.2645 (var=0.0031)
- pi_10: 0.3205 (var=0.0042)
- pi_11: 0.2817 (var=0.0060)
- pi_12: 0.3006 (var=0.0042)
- pi_13: 0.3822 (var=0.0076)

### Experiment 11
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def subject_contrast(df):
        high_gap = []
        low_gap = []

        for _, row in df.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float)
            b = np.asarray(row["option_b_ratings"], dtype=float)
            d = a - b

            # Target coalitions occupy cues 0--15; cues 16--17 are the
            # fixed reference coalition. Each target coalition is one of
            # the adjacent cue pairs (0,1), (2,3), ..., (14,15).
            target_idx = np.flatnonzero(d[:16] != 0.0)
            if target_idx.size != 2:
                continue

            pair_id = int(np.min(target_idx) // 2)
            direction = float(np.mean(d[target_idx]))
            if direction == 0.0:
                continue

            target_response = 0 if direction > 0.0 else 1
            chose_target = float(int(row["response"]) == target_response)

            # Low gaps comprise 0.00 and 0.04; high gaps comprise 0.10
            # and 0.18. This uses all 96 observations per subject.
            if pair_id <= 3:
                low_gap.append(chose_target)
            else:
                high_gap.append(chose_target)

        if len(high_gap) == 0 or len(low_gap) == 0:
            return np.nan
        return float(np.mean(high_gap) - np.mean(low_gap))

    if "subject_id" in data.columns:
        values = [subject_contrast(df) for _, df in data.groupby("subject_id", sort=False)]
    else:
        values = [subject_contrast(data)]

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.mean(values))
```

**Observed (real) value:** 0.0317 (var=0.0030)
**Previous candidate values (this loop):**
  - iter 1: -0.0258 (var=0.0150) (Δ vs real -0.0575)
  - iter 2: -0.0329 (var=0.0095) (Δ vs real -0.0646)
  - iter 3: 0.0187 (var=0.0097) (Δ vs real -0.0129)
  - iter 4: 0.0046 (var=0.0139) (Δ vs real -0.0271)
  - iter 5: 0.0237 (var=0.0083) (Δ vs real -0.0079)
  - iter 6: 0.0333 (var=0.0106) (Δ vs real +0.0017)
  - iter 7 (most recent): 0.0250 (var=0.0119) (Δ vs real -0.0067)
**Other theories' values on this metric (for reference):**
- pi_7: 0.2729 (var=0.0090)
- pi_6: 0.0217 (var=0.0086)
- pi_1: 0.0167 (var=0.0064)
- pi_2: -0.0292 (var=0.0133)
- pi_3: 0.1604 (var=0.0061)
- pi_4: 0.0054 (var=0.0079)
- pi_5: 0.0146 (var=0.0117)
- pi_8: 0.0125 (var=0.0063)
- pi_9: 0.1046 (var=0.0086)
- pi_10: 0.0933 (var=0.0083)
- pi_11: 0.1204 (var=0.0114)
- pi_12: 0.1071 (var=0.0100)
- pi_13: 0.1450 (var=0.0154)

### Experiment 12
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    distributed = {(3, 4, 5), (9, 10, 11)}

    def subject_score(df):
        records = []
        for _, row in df.iterrows():
            a = np.asarray(row['option_a_ratings'], dtype=float)
            b = np.asarray(row['option_b_ratings'], dtype=float)
            diff = a - b
            a_support = tuple(np.flatnonzero(diff > 0).tolist())
            b_support = tuple(np.flatnonzero(diff < 0).tolist())
            a_is_distributed = a_support in distributed
            b_is_distributed = b_support in distributed
            if a_is_distributed == b_is_distributed:
                continue
            chose_a = int(row['response']) == 0
            chose_distributed = (
                (chose_a and a_is_distributed) or
                ((not chose_a) and b_is_distributed)
            )
            records.append((a_is_distributed, float(chose_distributed)))

        if not records:
            return np.nan

        records = pd.DataFrame(records, columns=['distributed_on_a', 'choice'])
        role_means = records.groupby('distributed_on_a')['choice'].mean()
        if len(role_means) == 2:
            return float(0.5 * (role_means.loc[False] + role_means.loc[True]))
        return float(records['choice'].mean())

    if len(data) == 0:
        return float('nan')

    if 'subject_id' in data.columns:
        scores = [subject_score(df) for _, df in data.groupby('subject_id', sort=False)]
    else:
        scores = [subject_score(data)]
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    return float(np.mean(scores)) if scores.size else float('nan')
```

**Observed (real) value:** 0.6692 (var=0.0068)
**Previous candidate values (this loop):**
  - iter 1: 0.5450 (var=0.0045) (Δ vs real -0.1242)
  - iter 2: 0.5052 (var=0.0036) (Δ vs real -0.1640)
  - iter 3: 0.5023 (var=0.0038) (Δ vs real -0.1669)
  - iter 4: 0.4948 (var=0.0041) (Δ vs real -0.1744)
  - iter 5: 0.4227 (var=0.0043) (Δ vs real -0.2465)
  - iter 6: 0.5006 (var=0.0039) (Δ vs real -0.1685)
  - iter 7 (most recent): 0.5125 (var=0.0047) (Δ vs real -0.1567)
**Other theories' values on this metric (for reference):**
- pi_6: 0.4883 (var=0.0028)
- pi_7: 0.5575 (var=0.0028)
- pi_1: 0.1550 (var=0.0096)
- pi_2: 0.5073 (var=0.0027)
- pi_3: 0.6663 (var=0.0014)
- pi_4: 0.5715 (var=0.0020)
- pi_5: 0.5602 (var=0.0027)
- pi_8: 0.7031 (var=0.0082)
- pi_9: 0.6831 (var=0.0036)
- pi_10: 0.7271 (var=0.0030)
- pi_11: 0.6735 (var=0.0042)
- pi_12: 0.6571 (var=0.0061)
- pi_13: 0.5746 (var=0.0084)

### Experiment 13
**Design**
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if data is None or len(data) == 0:
        return float('nan')

    cell_values = {}
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=int)
        b = np.asarray(row['option_b_ratings'], dtype=int)
        response = int(row['response'])

        # In this design, distributed coalitions occupy cues 6--11 and
        # clustered coalitions occupy cues 0--5. Identify which display side
        # contains the distributed coalition from discriminating support.
        a_support = np.flatnonzero(a > b)
        b_support = np.flatnonzero(b > a)
        a_is_distributed = (
            a_support.size > 0 and np.all(a_support >= 6)
        )
        b_is_distributed = (
            b_support.size > 0 and np.all(b_support >= 6)
        )

        if a_is_distributed == b_is_distributed:
            continue

        chose_distributed = (
            (response == 0 and a_is_distributed) or
            (response == 1 and b_is_distributed)
        )
        signed_choice = 1.0 if chose_distributed else -1.0

        # Equal-weight the eight unique configurations so that the statistic
        # remains mirror-corrected even if a few trials are missing.
        key = (tuple(a.tolist()), tuple(b.tolist()))
        if key not in cell_values:
            cell_values[key] = []
        cell_values[key].append(signed_choice)

    if not cell_values:
        return float('nan')

    cell_means = [float(np.mean(values)) for values in cell_values.values()]
    return float(np.mean(cell_means))

```

**Observed (real) value:** 0.3242 (var=0.0194)
**Previous candidate values (this loop):**
  - iter 1: 0.0429 (var=0.0112) (Δ vs real -0.2813)
  - iter 2: -0.0204 (var=0.0121) (Δ vs real -0.3446)
  - iter 3: 0.0042 (var=0.0151) (Δ vs real -0.3200)
  - iter 4: -0.0196 (var=0.0208) (Δ vs real -0.3438)
  - iter 5: -0.1812 (var=0.0119) (Δ vs real -0.5054)
  - iter 6: -0.0108 (var=0.0226) (Δ vs real -0.3350)
  - iter 7 (most recent): -0.0112 (var=0.0149) (Δ vs real -0.3354)
**Other theories' values on this metric (for reference):**
- pi_8: 0.5346 (var=0.0171)
- pi_6: 0.0158 (var=0.0163)
- pi_1: -0.6938 (var=0.0472)
- pi_2: 0.0129 (var=0.0129)
- pi_3: 0.6592 (var=0.0081)
- pi_4: 0.2250 (var=0.0097)
- pi_5: 0.2188 (var=0.0122)
- pi_7: 0.3412 (var=0.0162)
- pi_9: 0.3688 (var=0.0124)
- pi_10: 0.4242 (var=0.0096)
- pi_11: 0.3646 (var=0.0242)
- pi_12: 0.3475 (var=0.0228)
- pi_13: 0.1496 (var=0.0263)

### Experiment 14
**Design**
  A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Concentrate on the direct endpoint comparison and the three early
    # reliability-gap transitions where region release should be largest.
    diagnostic_pairs = {(0, 1), (1, 2), (2, 3), (0, 5)}
    distributed_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=np.int8)
        b = np.asarray(row["option_b_ratings"], dtype=np.int8)
        diff = a - b
        a_support = np.flatnonzero(diff > 0)
        b_support = np.flatnonzero(diff < 0)

        if a_support.size == 0 or b_support.size == 0:
            continue

        # Each coalition occupies one of six consecutive three-cue blocks.
        a_block = int(a_support[0] // 3)
        b_block = int(b_support[0] // 3)
        pair = tuple(sorted((a_block, b_block)))
        if pair not in diagnostic_pairs:
            continue

        response = int(row["response"])
        chosen_block = a_block if response == 0 else b_block
        distributed_block = max(a_block, b_block)
        distributed_choices.append(float(chosen_block == distributed_block))

    if not distributed_choices:
        return float("nan")
    return float(np.mean(distributed_choices))
```

**Observed (real) value:** 0.5869 (var=0.0047)
**Previous candidate values (this loop):**
  - iter 1: 0.5184 (var=0.0025) (Δ vs real -0.0684)
  - iter 2: 0.4944 (var=0.0036) (Δ vs real -0.0925)
  - iter 3: 0.5019 (var=0.0044) (Δ vs real -0.0850)
  - iter 4: 0.4953 (var=0.0049) (Δ vs real -0.0916)
  - iter 5: 0.4675 (var=0.0053) (Δ vs real -0.1194)
  - iter 6: 0.5072 (var=0.0043) (Δ vs real -0.0797)
  - iter 7 (most recent): 0.5016 (var=0.0041) (Δ vs real -0.0853)
**Other theories' values on this metric (for reference):**
- pi_6: 0.5012 (var=0.0028)
- pi_8: 0.6616 (var=0.0045)
- pi_1: 0.1222 (var=0.0090)
- pi_2: 0.5122 (var=0.0043)
- pi_3: 0.6784 (var=0.0035)
- pi_4: 0.6269 (var=0.0040)
- pi_5: 0.6112 (var=0.0040)
- pi_7: 0.5816 (var=0.0036)
- pi_9: 0.5988 (var=0.0038)
- pi_10: 0.6191 (var=0.0035)
- pi_11: 0.5844 (var=0.0036)
- pi_12: 0.5878 (var=0.0035)
- pi_13: 0.5269 (var=0.0058)

### Experiment 15
**Design**
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Triadic onset in preference for reliability-separated coalitions."""
    dyad_choices = []
    triad_choices = []

    for row in data.itertuples(index=False):
        a = np.asarray(row.option_a_ratings, dtype=np.int8)
        b = np.asarray(row.option_b_ratings, dtype=np.int8)
        d = a - b

        a_support = np.flatnonzero(d > 0)
        b_support = np.flatnonzero(d < 0)
        if a_support.size != b_support.size or a_support.size not in (2, 3):
            continue

        # The reliability-separated pair uses cues 4:8 (validities .65 and
        # .7666667); the matched pair uses two cues 0:4 (validity .70).
        a_mid = int(np.sum((a_support >= 4) & (a_support < 8)))
        b_mid = int(np.sum((b_support >= 4) & (b_support < 8)))
        a_equal = int(np.sum(a_support < 4))
        b_equal = int(np.sum(b_support < 4))

        if a_mid == 2 and b_equal == 2:
            separated_is_a = True
        elif b_mid == 2 and a_equal == 2:
            separated_is_a = False
        else:
            continue

        response = int(row.response)
        chose_separated = float(
            (separated_is_a and response == 0)
            or ((not separated_is_a) and response == 1)
        )

        if a_support.size == 2:
            dyad_choices.append(chose_separated)
        else:
            triad_choices.append(chose_separated)

    if len(dyad_choices) == 0 or len(triad_choices) == 0:
        return float("nan")

    return float(np.mean(triad_choices) - np.mean(dyad_choices))
```

**Observed (real) value:** 0.0142 (var=0.0045)
**Previous candidate values (this loop):**
  - iter 1: 0.0254 (var=0.0097) (Δ vs real +0.0112)
  - iter 2: 0.0904 (var=0.0101) (Δ vs real +0.0762)
  - iter 3: -0.0292 (var=0.0092) (Δ vs real -0.0433)
  - iter 4: -0.0067 (var=0.0094) (Δ vs real -0.0208)
  - iter 5: -0.0858 (var=0.0082) (Δ vs real -0.1000)
  - iter 6: -0.0104 (var=0.0075) (Δ vs real -0.0246)
  - iter 7 (most recent): -0.0108 (var=0.0061) (Δ vs real -0.0250)
**Other theories' values on this metric (for reference):**
- pi_8: 0.2675 (var=0.0085)
- pi_9: -0.0350 (var=0.0068)
- pi_1: -0.3600 (var=0.0100)
- pi_2: -0.0133 (var=0.0099)
- pi_3: -0.0054 (var=0.0075)
- pi_4: -0.1288 (var=0.0113)
- pi_5: 0.0125 (var=0.0091)
- pi_6: -0.0029 (var=0.0108)
- pi_7: 0.0317 (var=0.0068)
- pi_10: 0.0337 (var=0.0098)
- pi_11: -0.0717 (var=0.0091)
- pi_12: -0.1379 (var=0.0071)
- pi_13: -0.1233 (var=0.0085)

### Experiment 16
**Design**
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Mirror-corrected probability of choosing a single-duplicate coalition
    # (features 8:16) over a double-duplicate coalition (features 0:8).
    chosen_single = []
    single_on_a = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=np.float64)
        b = np.asarray(row["option_b_ratings"], dtype=np.float64)
        if a.size < 16 or b.size < 16:
            continue

        a_is_single = float(np.sum(a[8:16])) > float(np.sum(a[0:8]))
        b_is_single = float(np.sum(b[8:16])) > float(np.sum(b[0:8]))
        if a_is_single == b_is_single:
            continue

        response = int(row["response"])
        chose_single = ((response == 0 and a_is_single) or
                         (response == 1 and b_is_single))
        chosen_single.append(float(chose_single))
        single_on_a.append(bool(a_is_single))

    if not chosen_single:
        return float("nan")

    y = np.asarray(chosen_single, dtype=np.float64)
    side = np.asarray(single_on_a, dtype=bool)

    # Give the two mirror orientations equal weight, canceling stable A/B bias.
    if np.any(side) and np.any(~side):
        return float(0.5 * (np.mean(y[side]) + np.mean(y[~side])))
    return float(np.mean(y))

```

**Observed (real) value:** 0.6779 (var=0.0032)
**Previous candidate values (this loop):**
  - iter 1: 0.5454 (var=0.0027) (Δ vs real -0.1325)
  - iter 2: 0.5885 (var=0.0036) (Δ vs real -0.0894)
  - iter 3: 0.5040 (var=0.0027) (Δ vs real -0.1740)
  - iter 4: 0.5275 (var=0.0040) (Δ vs real -0.1504)
  - iter 5: 0.4708 (var=0.0035) (Δ vs real -0.2071)
  - iter 6: 0.5123 (var=0.0027) (Δ vs real -0.1656)
  - iter 7 (most recent): 0.5483 (var=0.0034) (Δ vs real -0.1296)
**Other theories' values on this metric (for reference):**
- pi_9: 0.6575 (var=0.0035)
- pi_8: 0.5031 (var=0.0031)
- pi_1: 0.8390 (var=0.0107)
- pi_2: 0.5010 (var=0.0023)
- pi_3: 0.8723 (var=0.0016)
- pi_4: 0.6344 (var=0.0024)
- pi_5: 0.6090 (var=0.0024)
- pi_6: 0.4815 (var=0.0033)
- pi_7: 0.6562 (var=0.0033)
- pi_10: 0.6777 (var=0.0028)
- pi_11: 0.6365 (var=0.0053)
- pi_12: 0.7210 (var=0.0079)
- pi_13: 0.5477 (var=0.0053)

### Experiment 17
**Design**
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Signed, inverse-variance-weighted cardinality profile of choices for the
    # reliability-diverse coalition. The duplicate pair identifying each
    # cardinality is fixed by this experimental design.
    duplicate_pair = {2: (0, 1), 3: (4, 5), 4: (10, 11)}
    weights = {2: -1.00, 3: 0.53, 4: -0.51}

    def subject_score(df):
        chosen_diverse = {2: [], 3: [], 4: []}

        for _, row in df.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float)
            b = np.asarray(row["option_b_ratings"], dtype=float)
            d = a - b
            cardinality = int(np.sum(d > 0))
            if cardinality not in duplicate_pair:
                continue

            i, j = duplicate_pair[cardinality]
            if d[i] > 0 and d[j] > 0:
                # The duplicate coalition supports A, so B is diverse.
                y = float(int(row["response"]) == 1)
            elif d[i] < 0 and d[j] < 0:
                # The duplicate coalition supports B, so A is diverse.
                y = float(int(row["response"]) == 0)
            else:
                continue
            chosen_diverse[cardinality].append(y)

        if any(len(chosen_diverse[k]) == 0 for k in (2, 3, 4)):
            return np.nan

        rates = {k: float(np.mean(chosen_diverse[k])) for k in (2, 3, 4)}
        return float(sum(weights[k] * (rates[k] - 0.5) for k in (2, 3, 4)))

    scores = []
    for _, subj_df in data.groupby("subject_id", sort=False):
        value = subject_score(subj_df)
        if np.isfinite(value):
            scores.append(value)

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** -0.3486 (var=0.0131)
**Previous candidate values (this loop):**
  - iter 1: -0.0726 (var=0.0196) (Δ vs real +0.2760)
  - iter 2: 0.0404 (var=0.0233) (Δ vs real +0.3890)
  - iter 3: -0.0328 (var=0.0148) (Δ vs real +0.3158)
  - iter 4: -0.0253 (var=0.0132) (Δ vs real +0.3232)
  - iter 5: -0.0005 (var=0.0112) (Δ vs real +0.3481)
  - iter 6: -0.0283 (var=0.0152) (Δ vs real +0.3203)
  - iter 7 (most recent): -0.0308 (var=0.0141) (Δ vs real +0.3178)
**Other theories' values on this metric (for reference):**
- pi_10: -0.1210 (var=0.0132)
- pi_9: -0.2371 (var=0.0104)
- pi_1: 0.0290 (var=0.0066)
- pi_2: 0.0005 (var=0.0172)
- pi_3: -0.1762 (var=0.0147)
- pi_4: -0.1228 (var=0.0094)
- pi_5: 0.0260 (var=0.0189)
- pi_6: -0.0595 (var=0.0208)
- pi_7: -0.3008 (var=0.0082)
- pi_8: -0.0731 (var=0.0163)
- pi_11: -0.1886 (var=0.0173)
- pi_12: -0.2260 (var=0.0128)
- pi_13: -0.1662 (var=0.0141)

### Experiment 18
**Design**
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Equal-weighted, mirror-corrected probability of choosing the chain coalition."""
    if data is None or len(data) == 0:
        return float("nan")

    records = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.size < 20 or b.size != a.size:
            continue

        d = a - b
        chain_direction = float(np.sum(d[:10]))
        if abs(chain_direction) < 1e-12:
            continue

        # Positive means that A is supported by one of the two chain blocks;
        # negative means that B is supported by a chain block.
        chain_is_a = chain_direction > 0.0
        response = int(row["response"])
        chain_chosen = float((response == 0 and chain_is_a) or
                             (response == 1 and not chain_is_a))

        # Identify the crossed chain/comparison configuration independently
        # of whether it was displayed in its original or mirrored orientation.
        chain_block = int(np.sum(np.abs(d[5:10])) > np.sum(np.abs(d[0:5])))
        comparison_block = int(
            np.sum(np.abs(d[15:20])) > np.sum(np.abs(d[10:15]))
        )
        family = 2 * chain_block + comparison_block
        records.append((row["subject_id"], family, chain_chosen))

    if not records:
        return float("nan")

    scored = pd.DataFrame(records, columns=["subject_id", "family", "score"])
    # Equal weighting of the four crossed configurations prevents stable cue
    # salience or an uneven number of usable repetitions from dominating.
    family_rates = scored.groupby(["subject_id", "family"])["score"].mean()
    subject_rates = family_rates.groupby(level=0).mean()
    return float(subject_rates.mean())
```

**Observed (real) value:** 0.4983 (var=0.0018)
**Previous candidate values (this loop):**
  - iter 1: 0.7340 (var=0.0062) (Δ vs real +0.2356)
  - iter 2: 0.6108 (var=0.0091) (Δ vs real +0.1125)
  - iter 3: 0.7365 (var=0.0050) (Δ vs real +0.2381)
  - iter 4: 0.6781 (var=0.0077) (Δ vs real +0.1798)
  - iter 5: 0.6025 (var=0.0057) (Δ vs real +0.1042)
  - iter 6: 0.6227 (var=0.0046) (Δ vs real +0.1244)
  - iter 7 (most recent): 0.6508 (var=0.0041) (Δ vs real +0.1525)
**Other theories' values on this metric (for reference):**
- pi_9: 0.7094 (var=0.0054)
- pi_10: 0.2913 (var=0.0022)
- pi_1: 0.1706 (var=0.0080)
- pi_2: 0.4950 (var=0.0026)
- pi_3: 0.1296 (var=0.0016)
- pi_4: 0.3533 (var=0.0025)
- pi_5: 0.3854 (var=0.0022)
- pi_6: 0.5079 (var=0.0037)
- pi_7: 0.4525 (var=0.0026)
- pi_8: 0.5229 (var=0.0036)
- pi_11: 0.4998 (var=0.0039)
- pi_12: 0.4948 (var=0.0035)
- pi_13: 0.5371 (var=0.0037)

### Experiment 19
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    narrow_choices = []
    wide_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        response = int(row["response"])

        a_support = np.flatnonzero(a > b)
        b_support = np.flatnonzero(b > a)

        # Features 0--3 form the two narrow-gap target dyads;
        # features 4--7 form the two wide-gap target dyads.
        a_narrow = np.any(a_support < 4)
        b_narrow = np.any(b_support < 4)
        a_wide = np.any((a_support >= 4) & (a_support < 8))
        b_wide = np.any((b_support >= 4) & (b_support < 8))

        if a_narrow and not b_narrow:
            narrow_choices.append(float(response == 0))
        elif b_narrow and not a_narrow:
            narrow_choices.append(float(response == 1))
        elif a_wide and not b_wide:
            wide_choices.append(float(response == 0))
        elif b_wide and not a_wide:
            wide_choices.append(float(response == 1))

    if len(narrow_choices) == 0 or len(wide_choices) == 0:
        return float("nan")

    return float(np.mean(wide_choices) - np.mean(narrow_choices))
```

**Observed (real) value:** -0.0283 (var=0.0129)
**Previous candidate values (this loop):**
  - iter 1: 0.1008 (var=0.0094) (Δ vs real +0.1292)
  - iter 2: 0.0596 (var=0.0191) (Δ vs real +0.0879)
  - iter 3: 0.1246 (var=0.0093) (Δ vs real +0.1529)
  - iter 4: 0.0971 (var=0.0109) (Δ vs real +0.1254)
  - iter 5: 0.0933 (var=0.0115) (Δ vs real +0.1217)
  - iter 6: 0.1113 (var=0.0125) (Δ vs real +0.1396)
  - iter 7 (most recent): 0.1362 (var=0.0070) (Δ vs real +0.1646)
**Other theories' values on this metric (for reference):**
- pi_11: -0.0029 (var=0.0090)
- pi_9: 0.1900 (var=0.0096)
- pi_1: -0.0088 (var=0.0046)
- pi_2: 0.0033 (var=0.0099)
- pi_3: 0.0775 (var=0.0094)
- pi_4: 0.0067 (var=0.0102)
- pi_5: 0.0087 (var=0.0124)
- pi_6: 0.0029 (var=0.0113)
- pi_7: 0.0192 (var=0.0098)
- pi_8: -0.0146 (var=0.0165)
- pi_10: -0.0008 (var=0.0157)
- pi_12: -0.0354 (var=0.0045)
- pi_13: 0.0083 (var=0.0145)

### Experiment 20
**Design**
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Score each response as 1 when the chain-validity coalition was chosen
    # and 0 when the exact-duplicate comparison coalition was chosen.
    chain_when_a = []
    chain_when_b = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=np.float64)
        b = np.asarray(row["option_b_ratings"], dtype=np.float64)
        d = a - b

        # In this design, chain experts occupy indices 0:8. Exactly one side
        # is supported by four of these experts on every experimental trial.
        a_chain_support = int(np.sum(d[:8] > 0.0))
        b_chain_support = int(np.sum(d[:8] < 0.0))
        response = int(row["response"])

        if a_chain_support > b_chain_support:
            chain_when_a.append(float(response == 0))
        elif b_chain_support > a_chain_support:
            chain_when_b.append(float(response == 1))

    # Average the two response-side orientations separately so the statistic
    # is explicitly mirror-corrected rather than sensitive to side imbalance.
    orientation_means = []
    if len(chain_when_a) > 0:
        orientation_means.append(float(np.mean(chain_when_a)))
    if len(chain_when_b) > 0:
        orientation_means.append(float(np.mean(chain_when_b)))

    if len(orientation_means) == 0:
        return float("nan")
    return float(np.mean(orientation_means))
```

**Observed (real) value:** 0.5042 (var=0.0027)
**Previous candidate values (this loop):**
  - iter 1: 0.7460 (var=0.0056) (Δ vs real +0.2419)
  - iter 2: 0.6219 (var=0.0104) (Δ vs real +0.1177)
  - iter 3: 0.7515 (var=0.0065) (Δ vs real +0.2473)
  - iter 4: 0.7110 (var=0.0063) (Δ vs real +0.2069)
  - iter 5: 0.6365 (var=0.0097) (Δ vs real +0.1323)
  - iter 6: 0.6625 (var=0.0042) (Δ vs real +0.1583)
  - iter 7 (most recent): 0.6540 (var=0.0054) (Δ vs real +0.1498)
**Other theories' values on this metric (for reference):**
- pi_9: 0.7292 (var=0.0039)
- pi_11: 0.5040 (var=0.0037)
- pi_1: 0.1427 (var=0.0067)
- pi_2: 0.5029 (var=0.0031)
- pi_3: 0.1819 (var=0.0027)
- pi_4: 0.3612 (var=0.0019)
- pi_5: 0.3902 (var=0.0029)
- pi_6: 0.5054 (var=0.0028)
- pi_7: 0.4792 (var=0.0023)
- pi_8: 0.6019 (var=0.0034)
- pi_10: 0.4062 (var=0.0025)
- pi_12: 0.5085 (var=0.0024)
- pi_13: 0.5681 (var=0.0054)

### Experiment 21
**Design**
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Fraction of subjects showing a supermajority preference for diverse coalitions."""
    if data is None or len(data) == 0:
        return 0.0

    records = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.size != 16 or b.size != 16:
            continue

        diff = a - b
        a_support = np.flatnonzero(diff > 0)
        b_support = np.flatnonzero(diff < 0)

        # In this design, indices 0--7 form the two reliability-diverse
        # coalitions, whereas indices 8--15 form the exact-duplicate blocks.
        a_diverse = (a_support.size == 4 and np.all(a_support < 8))
        b_diverse = (b_support.size == 4 and np.all(b_support < 8))
        a_duplicate = (a_support.size == 4 and np.all(a_support >= 8))
        b_duplicate = (b_support.size == 4 and np.all(b_support >= 8))

        if a_diverse and b_duplicate:
            chose_diverse = int(row["response"] == 0)
        elif b_diverse and a_duplicate:
            chose_diverse = int(row["response"] == 1)
        else:
            continue

        records.append((row["subject_id"], chose_diverse))

    if len(records) == 0:
        return 0.0

    scored = pd.DataFrame(records, columns=["subject_id", "chose_diverse"])
    subject_rates = scored.groupby("subject_id")["chose_diverse"].mean()

    # A 2/3 threshold lies well above the competing theory's near-chance
    # prediction but below the advocated theory's predicted range.
    return float(np.mean(subject_rates >= (2.0 / 3.0)))
```

**Observed (real) value:** 0.6400 (var=0.2304)
**Previous candidate values (this loop):**
  - iter 1: 0.8600 (var=0.1204) (Δ vs real +0.2200)
  - iter 2: 0.5000 (var=0.2500) (Δ vs real -0.1400)
  - iter 3: 0.8400 (var=0.1344) (Δ vs real +0.2000)
  - iter 4: 0.8600 (var=0.1204) (Δ vs real +0.2200)
  - iter 5: 0.1000 (var=0.0900) (Δ vs real -0.5400)
  - iter 6: 0.9000 (var=0.0900) (Δ vs real +0.2600)
  - iter 7 (most recent): 0.9000 (var=0.0900) (Δ vs real +0.2600)
**Other theories' values on this metric (for reference):**
- pi_11: 1.0000 (var=0.0000)
- pi_12: 0.0000 (var=0.0000)
- pi_1: 0.9600 (var=0.0384)
- pi_2: 0.0000 (var=0.0000)
- pi_3: 0.0000 (var=0.0000)
- pi_4: 0.0000 (var=0.0000)
- pi_5: 0.0000 (var=0.0000)
- pi_6: 0.0000 (var=0.0000)
- pi_7: 1.0000 (var=0.0000)
- pi_8: 0.0000 (var=0.0000)
- pi_9: 1.0000 (var=0.0000)
- pi_10: 1.0000 (var=0.0000)
- pi_13: 0.8200 (var=0.1476)

### Experiment 22
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # In this fixed design, features 0--3 form the two near-duplicate
    # coalitions, whereas features 4--7 form the two exact-duplicate
    # coalitions. Every trial pits one type against the other.
    near_choices = []

    for row in data.itertuples(index=False):
        a = np.asarray(row.option_a_ratings, dtype=np.float64)
        b = np.asarray(row.option_b_ratings, dtype=np.float64)

        a_near_mass = float(np.sum(a[:4]))
        a_exact_mass = float(np.sum(a[4:8]))
        b_near_mass = float(np.sum(b[:4]))
        b_exact_mass = float(np.sum(b[4:8]))

        a_is_near = a_near_mass > a_exact_mass
        b_is_near = b_near_mass > b_exact_mass

        # Retain only the intended near-versus-exact diagnostic trials.
        if a_is_near == b_is_near:
            continue

        response = int(row.response)
        chose_near = a_is_near if response == 0 else b_is_near
        near_choices.append(float(chose_near))

    if len(near_choices) == 0:
        return 0.5
    return float(np.mean(near_choices))
```

**Observed (real) value:** 0.6817 (var=0.0026)
**Previous candidate values (this loop):**
  - iter 1: 0.6342 (var=0.0043) (Δ vs real -0.0475)
  - iter 2: 0.5079 (var=0.0048) (Δ vs real -0.1737)
  - iter 3: 0.6813 (var=0.0070) (Δ vs real -0.0004)
  - iter 4: 0.6856 (var=0.0084) (Δ vs real +0.0040)
  - iter 5: 0.6610 (var=0.0074) (Δ vs real -0.0206)
  - iter 6: 0.6696 (var=0.0109) (Δ vs real -0.0121)
  - iter 7 (most recent): 0.6967 (var=0.0084) (Δ vs real +0.0150)
**Other theories' values on this metric (for reference):**
- pi_12: 0.7017 (var=0.0030)
- pi_11: 0.4998 (var=0.0029)
- pi_1: 0.8473 (var=0.0103)
- pi_2: 0.5046 (var=0.0017)
- pi_3: 0.4006 (var=0.0025)
- pi_4: 0.3654 (var=0.0023)
- pi_5: 0.4827 (var=0.0022)
- pi_6: 0.4890 (var=0.0027)
- pi_7: 0.5088 (var=0.0022)
- pi_8: 0.4983 (var=0.0026)
- pi_9: 0.7244 (var=0.0023)
- pi_10: 0.5156 (var=0.0026)
- pi_13: 0.6694 (var=0.0048)

### Experiment 23
**Design**
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if len(data) == 0:
        return float('nan')

    a = np.vstack([np.asarray(x, dtype=np.float64) for x in data['option_a_ratings']])
    b = np.vstack([np.asarray(x, dtype=np.float64) for x in data['option_b_ratings']])
    responses = data['response'].to_numpy(dtype=np.int64)

    # Features 8:16 comprise the two fully separated coalitions. Their signed
    # rating difference identifies whether the separated coalition is on A
    # (+1) or B (-1), independently of the particular coalition identities.
    separated_side = np.sign(np.sum(a[:, 8:16] - b[:, 8:16], axis=1))
    valid = (separated_side != 0) & ((responses == 0) | (responses == 1))
    if not np.any(valid):
        return float('nan')

    # +1 denotes choosing A and -1 choosing B. Multiplication mirror-corrects
    # every response so +1 always means choosing the separated coalition.
    chosen_side = 1.0 - 2.0 * responses[valid].astype(np.float64)
    return float(np.mean(chosen_side * separated_side[valid]))

```

**Observed (real) value:** -0.3542 (var=0.0111)
**Previous candidate values (this loop):**
  - iter 1: -0.0550 (var=0.0265) (Δ vs real +0.2992)
  - iter 2: -0.0508 (var=0.0203) (Δ vs real +0.3033)
  - iter 3: -0.0963 (var=0.0228) (Δ vs real +0.2579)
  - iter 4: -0.0887 (var=0.0238) (Δ vs real +0.2654)
  - iter 5: -0.0746 (var=0.0209) (Δ vs real +0.2796)
  - iter 6: -0.0971 (var=0.0084) (Δ vs real +0.2571)
  - iter 7 (most recent): -0.0408 (var=0.0204) (Δ vs real +0.3133)
**Other theories' values on this metric (for reference):**
- pi_11: 0.0071 (var=0.0085)
- pi_13: 0.1104 (var=0.0254)
- pi_1: -0.7033 (var=0.0507)
- pi_2: -0.0025 (var=0.0119)
- pi_3: 0.5962 (var=0.0107)
- pi_4: 0.2871 (var=0.0068)
- pi_5: 0.2362 (var=0.0116)
- pi_6: -0.0187 (var=0.0122)
- pi_7: -0.0146 (var=0.0112)
- pi_8: 0.1246 (var=0.0138)
- pi_9: -0.0104 (var=0.0124)
- pi_10: 0.0008 (var=0.0095)
- pi_12: 0.0108 (var=0.0081)

### Experiment 24
**Design**
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if data is None or len(data) == 0:
        return 0.0

    validities = np.asarray([
        0.55, 0.554, 0.56, 0.574, 0.58, 0.594, 0.6, 0.604,
        0.65, 0.67, 0.69, 0.71, 0.73, 0.75, 0.77, 0.79,
        0.81, 0.83, 0.85, 0.87, 0.89, 0.91, 0.93, 0.95
    ], dtype=float)

    broad_when_a = []
    broad_when_b = []

    def coalition_gap(ratings):
        x = np.asarray(ratings, dtype=float)
        idx = np.flatnonzero(x > 0.5)
        if idx.size != 2:
            return np.nan
        return float(abs(validities[idx[1]] - validities[idx[0]]))

    for _, row in data.iterrows():
        gap_a = coalition_gap(row["option_a_ratings"])
        gap_b = coalition_gap(row["option_b_ratings"])
        if not np.isfinite(gap_a) or not np.isfinite(gap_b):
            continue

        # The design contains narrow gaps of .004 and broad gaps of .014;
        # .009 is their fixed midpoint and does not depend on responses.
        a_is_broad = gap_a > 0.009
        b_is_broad = gap_b > 0.009
        if a_is_broad == b_is_broad:
            continue

        response = int(row["response"])
        chose_broad = float(
            (a_is_broad and response == 0) or
            (b_is_broad and response == 1)
        )

        if a_is_broad:
            broad_when_a.append(chose_broad)
        else:
            broad_when_b.append(chose_broad)

    # Average the two mirror orientations separately before combining them.
    # This exactly removes a stable A/B response tendency in the balanced
    # schedule and remains well-defined for a single subject.
    orientation_means = []
    if len(broad_when_a) > 0:
        orientation_means.append(float(np.mean(broad_when_a)))
    if len(broad_when_b) > 0:
        orientation_means.append(float(np.mean(broad_when_b)))

    if len(orientation_means) == 0:
        return 0.0
    return float(np.mean(orientation_means))

```

**Observed (real) value:** 0.3208 (var=0.0038)
**Previous candidate values (this loop):**
  - iter 1: 0.5692 (var=0.0064) (Δ vs real +0.2483)
  - iter 2: 0.4492 (var=0.0057) (Δ vs real +0.1283)
  - iter 3: 0.4217 (var=0.0122) (Δ vs real +0.1008)
  - iter 4: 0.3940 (var=0.0135) (Δ vs real +0.0731)
  - iter 5: 0.3925 (var=0.0092) (Δ vs real +0.0717)
  - iter 6: 0.3917 (var=0.0096) (Δ vs real +0.0708)
  - iter 7 (most recent): 0.3898 (var=0.0136) (Δ vs real +0.0690)
**Other theories' values on this metric (for reference):**
- pi_13: 0.4952 (var=0.0056)
- pi_11: 0.7165 (var=0.0043)
- pi_1: 0.5023 (var=0.0008)
- pi_2: 0.5117 (var=0.0028)
- pi_3: 0.5012 (var=0.0019)
- pi_4: 0.4933 (var=0.0018)
- pi_5: 0.5035 (var=0.0025)
- pi_6: 0.4967 (var=0.0029)
- pi_7: 0.5267 (var=0.0024)
- pi_8: 0.4975 (var=0.0031)
- pi_9: 0.5210 (var=0.0033)
- pi_10: 0.5956 (var=0.0110)
- pi_12: 0.4994 (var=0.0025)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Coherence-Constrained Reliability-Strata Inference proposes that observers interpret an agreeing coalition by inferring a compact posterior distribution over latent reliability strata. Experts are assigned continuously rather than categorically to one through four possible source strata. Exact or virtually exact experts have a high probability of sharing a source and therefore incur accumulating redundancy. A small but nonzero reliability difference instead provides maximal corroboration because it supports distinguishable sources that remain mutually coherent. Larger separations progressively lose consensus coherence, producing a Goldilocks-shaped relation between reliability gap and coalition value. Coalition structure also matters: a small, balanced collection of compact strata supported by multiple experts is easier to compress and receives greater confidence than either one undifferentiated duplicate mass or many weakly supported singleton levels. Replication protection is granted only when posterior cluster mass is concentrated tightly around a stratum center; diffuse chains cannot obtain this protection merely through broad coassignment. For coalitions beyond triadic capacity, independence credit is pooled at the latent-stratum level and smoothly limited by the strongest compact, balanced, supported relation between distinguishable strata. Thus a chain cannot accumulate corroboration from every locally narrow expert pair. A dyad can progress from redundant at equality, to coherently corroborative at a narrow gap, and back toward weak consensus at a broader gap. A replicated two-stratum tetrad can outperform four unsupported reliability levels, although each compact stratum's excess membership adds a separate redundancy cost. Instructed reliability has a weak but consequential bounded influence after compression. Coalition magnitude saturates smoothly under finite capacity, with a modest corroborative release at the two-to-three transition that decays beyond three. Exchangeable idiosyncratic cue salience introduces individual variation without population-level position, onset, closure, adjacency, or recency effects.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Coherence-Constrained Reliability-Strata Inference expects "
            f"shape (2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = int(stim.shape[1])

    v = np.asarray(parameters["validities"], dtype=np.float64)
    if v.ndim != 1 or v.size != n_features:
        raise ValueError(
            f"validities length {v.size} != n_features {n_features}."
        )
    v = np.clip(v, 0.500001, 0.999999)

    raw_salience = np.asarray(
        parameters["cue_salience_profile"], dtype=np.float64
    )
    if raw_salience.ndim != 1 or raw_salience.size != n_features:
        raise ValueError(
            f"cue_salience_profile length {raw_salience.size} != "
            f"n_features {n_features}."
        )

    validity_compression = float(parameters["validity_compression"])
    validity_reliance = float(parameters["validity_reliance"])
    salience_strength = float(parameters["salience_strength"])
    redundancy_width = float(parameters["redundancy_width"])
    redundancy_cost = float(parameters["redundancy_cost"])
    compact_excess_cost = float(parameters["compact_excess_cost"])
    corroboration_peak = float(parameters["corroboration_peak"])
    corroboration_width = float(parameters["corroboration_width"])
    independence_benefit = float(parameters["independence_benefit"])
    coherence_width = float(parameters["coherence_width"])
    coherence_cost = float(parameters["coherence_cost"])
    cluster_width = float(parameters["cluster_width"])
    cluster_complexity = float(parameters["cluster_complexity"])
    cluster_temperature = float(parameters["cluster_temperature"])
    support_threshold = float(parameters["support_threshold"])
    support_slope = float(parameters["support_slope"])
    cluster_support_bonus = float(parameters["cluster_support_bonus"])
    capacity_saturation = float(parameters["capacity_saturation"])
    count_curvature = float(parameters["count_curvature"])
    triad_transition_gain = float(parameters["triad_transition_gain"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Communicated reliability is used, but diagnosticity differences are
    # strongly compressed so that source organization can dominate isolated
    # numerical reliability advantages.
    diagnosticity = np.log(v / (1.0 - v))
    represented = np.power(
        np.maximum(diagnosticity, 1e-12), validity_compression
    )
    represented /= max(float(np.mean(represented)), 1e-12)
    reliability_weight = (
        (1.0 - validity_reliance) * np.ones(n_features, dtype=np.float64)
        + validity_reliance * represented
    )

    # The salience profile is exchangeable across expert identities and is
    # centered within subject. It therefore adds stable heterogeneity without
    # imposing a common serial-position or spatial-organization gradient.
    centered_salience = raw_salience - float(np.mean(raw_salience))
    attention = np.exp(
        np.clip(salience_strength * centered_salience, -10.0, 10.0)
    )
    attention /= max(float(np.mean(attention)), 1e-12)
    cue_quality = reliability_weight * attention
    cue_quality /= max(float(np.mean(cue_quality)), 1e-12)

    difference = a - b
    a_support = np.flatnonzero(difference > 0.0)
    b_support = np.flatnonzero(difference < 0.0)

    if a_support.size == 0 and b_support.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    def soft_partition(x, k):
        """Deterministic soft clustering for one candidate stratum count."""
        n = int(x.size)
        if k == 1:
            assignments = np.ones((n, 1), dtype=np.float64)
            centers = np.array([float(np.mean(x))], dtype=np.float64)
            return assignments, centers

        xmin = float(np.min(x))
        xmax = float(np.max(x))
        if xmax - xmin <= 1e-15:
            centers = np.full(k, xmin, dtype=np.float64)
        else:
            quantiles = np.linspace(0.0, 1.0, k)
            centers = np.quantile(x, quantiles).astype(np.float64)

        assignments = np.full((n, k), 1.0 / float(k), dtype=np.float64)
        for _ in range(16):
            squared = (x[:, None] - centers[None, :]) ** 2
            logits = -0.5 * squared / max(cluster_width ** 2, 1e-12)
            logits -= np.max(logits, axis=1, keepdims=True)
            assignments = np.exp(logits)
            assignments /= np.maximum(
                np.sum(assignments, axis=1, keepdims=True), 1e-12
            )
            masses = np.sum(assignments, axis=0)
            updated = np.sum(assignments * x[:, None], axis=0) / np.maximum(
                masses, 1e-12
            )
            centers = 0.65 * centers + 0.35 * updated

        order = np.argsort(centers)
        return assignments[:, order], centers[order]

    def coalition_signal(indices):
        idx = np.asarray(indices, dtype=np.int64)
        n = int(idx.size)
        if n == 0:
            return 0.0
        if n == 1:
            return float(cue_quality[idx[0]])

        x = v[idx]
        qualities = cue_quality[idx]

        # Smooth Bayesian model comparison over a small number of reliability
        # strata. The likelihood rewards fit, while the complexity term makes
        # unsupported singleton strata expensive. When all values coincide,
        # the resolution gate strongly favors a single latent source.
        max_k = int(min(4, n))
        models = []
        model_scores = []
        observed_range = float(np.max(x) - np.min(x))
        resolution_gate = 1.0 - np.exp(
            -0.5 * (observed_range / max(cluster_width, 1e-12)) ** 2
        )

        for k in range(1, max_k + 1):
            assignments, centers = soft_partition(x, k)
            masses = np.sum(assignments, axis=0)
            residual = np.sum(
                assignments * (x[:, None] - centers[None, :]) ** 2
            )
            fit_cost = residual / max(
                float(n) * cluster_width ** 2, 1e-12
            )

            support = 1.0 / (
                1.0
                + np.exp(
                    -np.clip(
                        support_slope * (masses - support_threshold),
                        -50.0,
                        50.0,
                    )
                )
            )
            unsupported_cost = float(np.mean(1.0 - support))
            extra_strata = float(k - 1)
            resolution_penalty = (
                cluster_complexity
                * extra_strata
                / max(0.12 + resolution_gate, 1e-12)
            )
            score = -fit_cost - resolution_penalty - 0.35 * unsupported_cost
            models.append((assignments, centers, masses, support))
            model_scores.append(score)

        model_scores = np.asarray(model_scores, dtype=np.float64)
        posterior_logits = model_scores / max(cluster_temperature, 1e-12)
        posterior_logits -= np.max(posterior_logits)
        posterior = np.exp(posterior_logits)
        posterior /= max(float(np.sum(posterior)), 1e-12)

        # Posterior coassignment is a continuous estimate that two experts
        # derive from the same latent source.
        coassignment = np.zeros((n, n), dtype=np.float64)
        compressibility = 0.0
        expected_strata = 0.0
        compact_excess = 0.0
        stratum_link_credit = 0.0
        for probability, model in zip(posterior, models):
            assignments, centers, masses, support = model
            k = int(assignments.shape[1])
            coassignment += probability * (assignments @ assignments.T)
            expected_strata += probability * float(k)

            # Replication requires both excess posterior mass and low
            # within-stratum variance. Broad assignments over a smooth chain
            # therefore do not count as replicated support.
            within_variance = np.sum(
                assignments * (x[:, None] - centers[None, :]) ** 2,
                axis=0,
            ) / np.maximum(masses, 1e-12)
            concentration = np.exp(
                -0.5
                * within_variance
                / max(redundancy_width ** 2, 1e-12)
            )
            model_compact_excess = float(
                np.sum(
                    np.maximum(masses - 1.0, 0.0)
                    * support
                    * concentration
                )
                / float(n)
            )
            compact_excess += probability * model_compact_excess

            if k > 1:
                proportions = masses / max(float(np.sum(masses)), 1e-12)
                entropy = -float(
                    np.sum(proportions * np.log(np.maximum(proportions, 1e-12)))
                ) / max(np.log(float(k)), 1e-12)
                supported_mass = float(
                    np.sum(proportions * support * concentration)
                )
                compactness = 1.0 - float(k - 1) / float(max(n, 2))
                model_compressibility = (
                    entropy
                    * supported_mass
                    * max(compactness, 0.0)
                    * (float(k - 1) / float(k))
                )
                compressibility += probability * model_compressibility

                # Independence is represented once between inferred strata,
                # rather than once for every expert pair. A link receives
                # credit only when both strata are compact and supported, and
                # when their posterior masses are reasonably balanced. Taking
                # the strongest link is a smooth capacity cap: a reliability
                # chain cannot collect several rewards from adjacent gaps.
                model_links = []
                for ci in range(k):
                    for cj in range(ci + 1, k):
                        center_gap = abs(float(centers[ci] - centers[cj]))
                        center_resolution = 1.0 - np.exp(
                            -0.5
                            * (
                                center_gap
                                / max(redundancy_width, 1e-12)
                            ) ** 2
                        )
                        center_goldilocks = np.exp(
                            -0.5
                            * (
                                (center_gap - corroboration_peak)
                                / max(corroboration_width, 1e-12)
                            ) ** 2
                        )
                        balanced_mass = (
                            2.0 * min(float(masses[ci]), float(masses[cj]))
                            / max(float(n), 1e-12)
                        )
                        compact_support = np.sqrt(
                            max(float(support[ci] * support[cj]), 0.0)
                            * max(
                                float(concentration[ci] * concentration[cj]),
                                0.0,
                            )
                        )
                        model_links.append(
                            center_resolution
                            * center_goldilocks
                            * np.clip(balanced_mass, 0.0, 1.0)
                            * compact_support
                        )
                if model_links:
                    stratum_link_credit += probability * float(
                        np.max(model_links)
                    )

        # Pairwise source evidence has a Goldilocks shape. Equality implies
        # common-source redundancy; a resolvable narrow gap is corroborative;
        # and only the tail beyond that narrow region loses coherence.
        redundant_load = 0.0
        corroborative_load = 0.0
        broad_dispersion = 0.0
        pair_count = 0.0

        tail_scale = max(0.5 * corroboration_width, 1e-12)
        tail_onset = corroboration_peak + corroboration_width
        tail_floor = 1.0 / (
            1.0 + np.exp(np.clip(tail_onset / tail_scale, -50.0, 50.0))
        )

        for i in range(n):
            for j in range(i + 1, n):
                gap = abs(float(x[i] - x[j]))
                exact_similarity = np.exp(
                    -0.5 * (gap / max(redundancy_width, 1e-12)) ** 2
                )
                redundant_load += coassignment[i, j] * exact_similarity

                goldilocks = np.exp(
                    -0.5
                    * (
                        (gap - corroboration_peak)
                        / max(corroboration_width, 1e-12)
                    ) ** 2
                )
                direct_resolution = 1.0 - np.exp(
                    -0.5 * (gap / max(redundancy_width, 1e-12)) ** 2
                )
                corroborative_load += direct_resolution * goldilocks

                tail = 1.0 / (
                    1.0
                    + np.exp(
                        -np.clip(
                            (gap - tail_onset) / tail_scale,
                            -50.0,
                            50.0,
                        )
                    )
                )
                tail = np.clip(
                    (tail - tail_floor) / max(1.0 - tail_floor, 1e-12),
                    0.0,
                    1.0,
                )
                broad_dispersion += tail
                pair_count += 1.0

        pair_count = max(pair_count, 1.0)
        mean_corroboration = corroborative_load / pair_count
        mean_dispersion = broad_dispersion / pair_count

        # Dyads and triads retain the accepted pairwise Goldilocks relation.
        # Beyond triadic capacity, corroboration is increasingly constrained
        # by the single best compact and balanced link between latent strata.
        # This local aggregation correction leaves all other coalition terms
        # unchanged and avoids a global span penalty.
        higher_order_load = float(max(n - 3, 0))
        stratum_pooling = 1.0 - np.exp(-higher_order_load)
        link_limited_corroboration = min(
            mean_corroboration, stratum_link_credit
        )
        mean_corroboration = (
            (1.0 - stratum_pooling) * mean_corroboration
            + stratum_pooling * link_limited_corroboration
        )

        # Only compact, genuinely replicated strata protect a coalition from
        # dispersion. Posterior overlap by itself cannot shield a smooth chain.
        replication_guard = 1.0 / (1.0 + 1.5 * compact_excess)
        contextual_advantage = max(
            float(np.mean(represented[idx])) - 1.0, 0.0
        )
        quality_guard = np.exp(-2.0 * contextual_advantage)
        mean_dispersion *= replication_guard * quality_guard

        # Redundancy accumulates with the number of close within-source pairs.
        # Compact excess membership adds a smaller incremental cost, allowing
        # a second supported duplicate group to remain consequential.
        redundancy_per_member = redundant_load / float(n)

        additional = capacity_saturation * (
            1.0
            - np.exp(
                -float(n - 1) / max(capacity_saturation, 1e-12)
            )
        )
        represented_count = (1.0 + additional) ** count_curvature

        # A smooth two-to-three corroborative release supports heterogeneous
        # three-versus-two judgments but decays beyond three under capacity.
        triad_transition = 1.0 / (
            1.0 + np.exp(-np.clip((float(n) - 2.5) / 0.18, -50.0, 50.0))
        )
        beyond_three_decay = np.exp(
            -float(max(n - 3, 0)) / max(capacity_saturation, 1e-12)
        )
        represented_count *= np.exp(
            triad_transition_gain * triad_transition * beyond_three_decay
        )

        # Source-credit weighting gives less influence to cues heavily
        # coassigned with coalition partners without introducing cue position.
        dependence_load = np.sum(coassignment, axis=1) - 1.0
        uniqueness = 1.0 / np.maximum(1.0 + dependence_load, 1e-12)
        quality_weights = 0.55 + 0.45 * uniqueness / max(
            float(np.mean(uniqueness)), 1e-12
        )
        coalition_quality = float(
            np.sum(quality_weights * qualities)
            / max(float(np.sum(quality_weights)), 1e-12)
        )

        structural_log_value = (
            -redundancy_cost * redundancy_per_member
            -compact_excess_cost * compact_excess
            + independence_benefit * mean_corroboration
            - coherence_cost * mean_dispersion
            + cluster_support_bonus * compressibility
        )
        structural_factor = np.exp(
            np.clip(structural_log_value, -8.0, 8.0)
        )

        # Expected strata mildly calibrate confidence but are capacity-limited;
        # they do not create an unbounded second count term.
        source_confidence = 1.0 + 0.06 * (
            1.0
            - np.exp(
                -(expected_strata - 1.0)
                / max(capacity_saturation, 1e-12)
            )
        )

        signal = (
            represented_count
            * coalition_quality
            * structural_factor
            * source_confidence
        )
        return float(max(signal, 0.0))

    evidence_a = coalition_signal(a_support)
    evidence_b = coalition_signal(b_support)
    net_a = evidence_a - evidence_b

    logits = np.array(
        [0.5 * beta * net_a, -0.5 * beta * net_a], dtype=np.float64
    )
    logits -= np.max(logits)
    probs = np.exp(logits)
    probs /= max(float(np.sum(probs)), 1e-12)

    probs = (1.0 - epsilon) * probs + epsilon * 0.5
    probs = np.maximum(probs, 0.0)
    total = float(np.sum(probs))
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
    return probs / total


`policy(probs) -> int`:
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.maximum(probs, 0.0)
    total = float(np.sum(probs))
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- validities: validities
- cue_salience_profile: [(-1.0, 1.0)] * n_features
- validity_compression: [0.08, 0.18]
- validity_reliance: [0.020, 0.065]
- salience_strength: [0.10, 0.28]
- redundancy_width: [0.0015, 0.0045]
- redundancy_cost: [0.70, 1.35]
- compact_excess_cost: [0.06, 0.20]
- corroboration_peak: [0.004, 0.007]
- corroboration_width: [0.003, 0.007]
- independence_benefit: [0.18, 0.55]
- coherence_width: [0.012, 0.040]
- coherence_cost: [0.28, 0.72]
- cluster_width: [0.004, 0.014]
- cluster_complexity: [0.10, 0.28]
- cluster_temperature: [0.45, 0.90]
- support_threshold: [1.35, 1.75]
- support_slope: [4.0, 8.0]
- cluster_support_bonus: [0.30, 0.82]
- capacity_saturation: [0.90, 3.10]
- count_curvature: [0.34, 0.74]
- triad_transition_gain: [0.04, 0.24]
- beta: [1.25, 1.75]
- epsilon: [0.025, 0.070]

`rationale`: This is a minimal edit to the accepted iteration-4 candidate. All validated parameter ranges, the dyadic Goldilocks function, compact-replication criterion, contextual reliability representation, redundancy terms, count saturation, and two-to-three transition are unchanged. The only substantive change is the aggregation of higher-order independence credit. For each posterior clustering model, the code now computes a single capped stratum-level corroboration value from the strongest link between compact, supported, and balanced latent strata. Starting only beyond triadic load, the original expert-pair corroboration is smoothly constrained by this value. Consequently, four- and five-member chains cannot collect structural value merely because several adjacent expert pairs fall near the narrow corroboration peak. Dyads are completely unchanged, protecting the Experiment 22 fit, and triads retain the accepted cardinality mechanism needed for Experiment 6 and the distributed-coalition tests. Unlike the rejected iteration-5 package, this edit introduces no global-span multiplier, no post-clustering mean-validity bonus, no duplicate-stratum surcharge, and no expanded triad-gain range. It therefore targets the remaining chain excesses in Experiments 18 and 20, and excessive diverse-tetrad confidence in Experiments 21 and 23, without broadly suppressing the useful configurations in Experiments 1 and 12–16.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed coherence-constrained clustering family and improves substantially on pi_13 in several critical tests, including Experiments 1, 15, 22, and 23. It also preserves the intended positional nulls in Experiments 4, 5, and 7–9. However, the central Goldilocks relation is not calibrated correctly. Broad or chain-like reliability structures remain too valuable: broad-choice rates are 0.569 versus 0.321 in Experiment 24, the wide-minus-narrow contrast is +0.101 versus -0.028 in Experiment 19, and chain choices are 0.734 versus 0.498 and 0.746 versus 0.504 in Experiments 18 and 20. The corresponding aversion to fully separated tetrads is also much too weak in Experiment 23 (-0.055 versus -0.354). At the same time, the desired narrow-gap advantage is slightly too weak in Experiment 22 (0.634 versus 0.682). Thus the implemented relation is too broad, not merely too weak. Important structural and cardinality patterns also remain underfit: Experiments 2, 6, 10, 12–14, 16, and 17 all show large discrepancies. In particular, the model is nearly indifferent in Experiment 2, almost categorically rejects the three-cue coalition in Experiment 6, underproduces distributed-coalition preferences in Experiments 12–14, and does not generate enough accumulating duplicate cost in Experiment 16. Between-subject behavior is also too categorical in Experiment 6 and too homogeneous in Experiment 21.
Rationale: Continue within the current mechanism family, but sharpen and renormalize its components. First, move the corroboration peak firmly toward the narrow-gap regime and reduce its width; the present peak range up to .011 and coherence widths up to .040 allow the .014 broad dyads in Experiment 24 to remain corroborative with little dispersion penalty. A narrower peak near .004–.006, narrower corroboration width, and an earlier/stronger coherence decline should simultaneously lower Experiments 19, 20, and 24, make Experiment 23 more negative, and preserve or strengthen Experiment 22. Second, fix the soft-clustering/compressibility calculation so diffuse chains cannot be represented as a few artificially well-supported strata merely because assignments are broad. Use effective occupied, distinguishable clusters rather than nominal k, tighten cluster assignments, and penalize overlapping or weakly separated centers. Reserve the compressibility bonus for balanced clusters with genuine replicated support. Third, replace the current pair-count-divided-by-n redundancy normalization with posterior expected excess membership within sources, such as a smooth sum over cluster masses above one. This lets a second duplicate group add cost as required by Experiments 16 and 18 while keeping that cost distinct from broad-gap incoherence. Counterbalance this with a somewhat stronger but smoothly saturating count term so exact triads are not almost universally beaten by dyads in Experiment 6. Independence benefits should saturate more strongly with coalition cardinality, reducing the excessive tetrad-diversity rate in Experiment 21 without sacrificing the dyadic near-over-exact effect in Experiment 22. These are parameterization and normalization corrections to the prescribed coherence-constrained source-clustering model, not a change of mechanism family.

**Outcome of this advice:** iter 1 candidate loss=0.2497 -> iter 2 candidate loss=0.2525 -> the gate marked it REJECTED.

### Iteration 2
Verdict: regenerate
Interpretation: The current candidate remains faithful to Coherence-Constrained Source Clustering, but it was rejected by the accept gate (loss 0.2525 versus the accepted base’s 0.2497). The previous broad intervention improved the intended broad/chain diagnostics—Experiments 18, 20, and 24 moved substantially toward the data, and Experiment 16 also improved—but these gains were outweighed by severe regressions elsewhere. Experiment 1 reversed from a near-correct majority preference (-0.091 versus -0.122) to an incorrect top-cue preference (+0.105). Experiment 10’s already excessive preference for exact dyads rose from 0.430 to 0.598 against 0.172. The narrow-over-exact effect in Experiment 22 collapsed from 0.634 to 0.508 against 0.682. Experiments 12–14 moved farther from their distributed-coalition effects, and Experiment 17 changed sign and became markedly worse. Experiment 15 also developed an excessive cardinality interaction. Thus the revised clustering penalties suppress not only broad incoherent coalitions but also the narrow, distributed, and majority coalitions that should retain value. Major failures remain in Experiments 2, 6, 10, 13, 17, and 23, with especially categorical under-selection of the three-cue coalition in Experiment 6 and inadequate aversion to fully separated tetrads in Experiment 23. Positional nulls in Experiments 4, 5, and 7–9 remain appropriately preserved.
Rationale: Because the last package of stronger coherence, accumulated redundancy, tighter clustering, and altered cardinality saturation was rejected, do not repeat that package or push those parameters farther. Build from the unchanged accepted iteration-1 base and make a smaller, more selective correction. The key implementation problem is that narrow-gap independence is multiplied by posterior distinctness: with cluster widths overlapping the .004–.006 regime, near-duplicate experts can remain highly coassigned, eliminating the very corroboration peak needed in Experiment 22. Meanwhile, a generic dispersion penalty can overpower reliability quality in Experiment 10. Use the currently loaded but effectively unused `redundancy_width` to define a direct smooth resolution gate, such as `1-exp(-gap^2/(2*redundancy_width^2))`, and apply it to the Goldilocks benefit independently of posterior coassignment. Keep exact gaps redundant, make .004–.006 gaps reliably distinguishable, and add only a bounded tail penalty beyond the narrow peak. Do not globally replace iteration 1’s redundancy normalization or count term again. To distinguish Experiment 10 from low-reliability broad-gap tests such as Experiment 24, let the prescribed weak contextual reliability component moderate the tail penalty: a coalition with a substantial mean-quality advantage should not be rejected solely for internal spread, whereas equally strong coalitions should be decided by coherence. This contextual moderation should be local to structural valuation rather than a global increase in cue validity reliance, which could worsen Experiments 1 and 2. For larger coalitions, reserve strong incoherence penalties for genuinely unsupported singleton-like partitions using effective occupied clusters; retain the accepted base’s treatment of replicated or majority-supported structure so Experiments 1, 15, 17, and 22 are not sacrificed again. Finally, Experiment 6 requires heterogeneity around the three-versus-two boundary rather than a uniform shift: modestly widen subject-level capacity or count-curvature ranges around the accepted settings, without increasing exchangeable salience or introducing position. This is a targeted recalibration within the prescribed clustering family.

**Outcome of this advice:** iter 2 candidate loss=0.2525 -> iter 3 candidate loss=0.2471 -> the gate marked it ACCEPTED.

### Iteration 3
Verdict: regenerate
Interpretation: The accepted candidate is a genuine improvement and remains faithful to Coherence-Constrained Source Clustering. The direct resolution gate successfully captures the central near-over-exact result in Experiment 22 almost perfectly (0.681 versus 0.682), improves the broad-gap result in Experiment 24 relative to iteration 1, and moves Experiment 23 in the correct negative direction. It also preserves the required positional nulls in Experiments 4, 5, and 7–9. Nevertheless, the fit is not yet experiment-invariant. Broad or chain-like coalitions remain substantially overvalued in Experiments 18 and 20 (0.737 and 0.752 versus approximately 0.50), and wide dyads are preferred rather than weakly disfavored in Experiment 19 (+0.125 versus -0.028). The model also still strongly underestimates broad-over-narrow rejection in Experiment 24 (0.422 versus 0.321) and separated-tetrad rejection in Experiment 23 (-0.096 versus -0.354). Separately, its weak reliability representation leaves it near chance where humans systematically favor particular reliability configurations in Experiments 12–14, and it chooses exact dyads far too often in Experiment 10 (0.474 versus 0.172). Cardinality and redundancy remain miscalibrated: only 8% rather than 44% of subjects favor the three-cue coalition in Experiment 6; the single-duplicate advantage in Experiment 16 is absent (0.504 versus 0.678); and the cardinality profile in Experiment 17 is much too shallow. Experiment 2 also remains essentially at chance rather than showing the observed opposition to the highest-validity discriminating cue. Thus the accepted gain is real but concentrated around Experiment 22 rather than constituting a satisfactory cross-experiment account.
Rationale: Build on the accepted iteration-3 model and preserve its direct narrow-gap resolution gate; that targeted change was validated by the gate and should not be undone. The next revision should use different, selective in-family adjustments rather than repeating the rejected iteration-1 package of globally tighter clustering and stronger coherence penalties. First, the current `replication_guard` is too permissive: posterior coassignment alone lets smooth chains masquerade as replicated strata, shielding them from dispersion in Experiments 18 and 20. Define replicated support using actual concentrated mass around a center—combining posterior mass with low within-cluster variance or close-pair similarity—and grant protection only to compact, well-supported strata. This should retain the replicated-endpoint benefit required by Experiment 1 while removing protection from chains and unsupported dispersed tetrads. Apply only a modest additional tail penalty around gaps of roughly .014 and above to continue the accepted improvement in Experiments 19, 23, and 24 without narrowing the .004–.006 peak that now fits Experiment 22.

Second, the local quality guard has not supplied enough reliability sensitivity. Modestly raise the lower and upper bounds of `validity_reliance` or reduce validity compression slightly, rather than making a large change. Greater contextual quality influence should move Experiment 1 back toward -0.122, help unequal high-quality dyads defeat exact dyads in Experiment 10, and recover the reliability-configuration effects in Experiments 12–14. Keep this contribution bounded at the coalition level so it does not generate the broad monotonic gap effects the arbiter warned against.

Third, adjust smooth cardinality saturation specifically at the two-to-three transition: slightly increase count credit there while retaining strong saturation beyond three. The present range moved Experiment 6 only from 0.02 to 0.08, so its population is still far from the required decision boundary. Pair this with broader subject-level variation in count curvature or capacity so Experiment 6 produces a mixture near 0.44 rather than another categorical result. Finally, add an incremental within-stratum excess-membership term alongside—rather than replacing—the accepted redundancy normalization. A small cost for each additional supported duplicate group is needed for Experiment 16 and the cardinality profile in Experiment 17. These changes preserve the prescribed clustering mechanism and the successful narrow-gap result while targeting the remaining false replication, weak reliability sensitivity, and cardinality-specific failures.

**Outcome of this advice:** iter 3 candidate loss=0.2471 -> iter 4 candidate loss=0.2329 -> the gate marked it ACCEPTED.

### Iteration 4
Verdict: regenerate
Interpretation: Iteration 4 is the best candidate so far and remains faithful to the prescribed Coherence-Constrained Source Clustering family. The accepted changes produced meaningful gains: Experiment 1 is now close to the human value (-0.105 versus -0.122), Experiment 22 retains an excellent narrow-over-exact fit (0.686 versus 0.682), Experiment 24 continues toward the required narrow-over-broad preference, Experiment 6 moves from 0.08 to 0.12, and the chain excesses in Experiments 18 and 20 decline somewhat. The positional nulls in Experiments 4, 5, and 7–9 also remain intact. However, the theory is not yet experiment-invariant. Its largest remaining structural error is that multi-member chains still collect too much value from locally narrow pairwise gaps: chain choice remains 0.678 versus 0.498 in Experiment 18 and 0.711 versus 0.504 in Experiment 20; wide dyads are still favored in Experiment 19 (+0.097 versus -0.028); and rejection of fully separated tetrads remains much too weak in Experiment 23 (-0.089 versus -0.354). Experiment 21 likewise overproduces strong diverse-tetrad preferences (0.86 versus 0.64). Thus compact-replication filtering helped, but local pairwise corroboration and compressibility can still overwhelm coalition-level incoherence. Reliability-configuration effects are also largely absent: Experiments 12–14 remain near chance rather than showing the observed distributed-coalition preferences, Experiment 10 predicts 0.481 rather than 0.172, and Experiment 2 predicts slight agreement rather than strong opposition to the highest-validity discriminating cue. Finally, redundancy and cardinality effects remain too shallow in Experiments 6, 16, and 17. The single-duplicate preference is only 0.528 versus 0.678, and the Experiment 17 profile is -0.025 versus -0.349. These discrepancies are too large and systematic to ship despite the improved aggregate loss.
Rationale: Build on iteration 4: the most recent advice was accepted, so preserve the direct dyadic resolution gate, compact replication criterion, modest validity increase, and two-to-three transition mechanism. The next adjustment should target a different in-family failure rather than repeat the previously rejected package of globally tighter clustering and stronger penalties. First, add a coalition-level coherence gate for coalitions of three or more experts. At present, a broad chain can earn corroboration from several locally narrow pairs and a compressibility bonus even when its global reliability span implies weak consensus. Multiply independence and compressibility credit by a smooth function of posterior global spread, unsupported occupied strata, or maximum/robust pairwise separation. Keep this gate weak or absent for dyads so the excellent Experiment 22 fit is protected. This should selectively lower Experiments 18, 20, 21, and 23 without indiscriminately suppressing compact replicated structures such as Experiment 1. For Experiment 24, only a small dyadic tail adjustment is warranted because the current 0.394 is already moving toward 0.321; do not narrow the successful corroboration peak itself.

Second, revise how contextual reliability enters coalition value rather than merely increasing the same tiny `validity_reliance` range again. The current within-experiment normalization and coalition averaging reduce communicated validity to a nearly negligible perturbation. Use a bounded coalition-level mean-diagnosticity or quality-contrast term after clustering, separate from the dispersion guard, so reliability configurations can influence Experiments 10 and 12–14 without turning every gap into monotonic individuation evidence. In particular, high mean quality should offset some dispersion cost, but it should not erase the global coherence gate for chains. This remains the recommendation's weak contextual reliability component, not a switch of mechanism family.

Third, extend the accepted compact-excess idea selectively: charge a small incremental cost for each additional well-supported redundant stratum, rather than mainly for total excess mass. The human contrast in Experiment 16 concerns accumulation across duplicate groups, while Experiment 1 requires two compact replicated endpoints still to beat unsupported singleton levels. A bounded second-stratum surcharge can express both. Finally, increase the upper range of `triad_transition_gain` modestly and widen subject heterogeneity specifically in that gain or count curvature, because Experiment 6 is improving in the correct direction but remains far below the observed 0.44 and has too little between-subject variance. Avoid increasing general salience heterogeneity, which would unnecessarily inflate variance in equal-cardinality diagnostics such as Experiment 24. These are focused normalization and gating changes within the prescribed clustering model and retain the components that the accept gate has validated.

**Outcome of this advice:** iter 4 candidate loss=0.2329 -> iter 5 candidate loss=0.2705 -> the gate marked it REJECTED.

### Iteration 5
Verdict: regenerate
Interpretation: The current candidate remains nominally faithful to Coherence-Constrained Source Clustering, but it was decisively rejected by the accept gate: loss rose from the accepted iteration-4 value of 0.2329 to 0.2705. The new package achieved its narrow intended benefit for chains—Experiments 18 and 20 moved from 0.678/0.711 to 0.603/0.637, and Experiment 6 improved from 0.12 to 0.16—but caused much larger reversals across the core evidence. Experiment 1 changed from an excellent -0.105 to the wrong-sign +0.138. Distributed-coalition effects collapsed in Experiments 12–14: 0.423 versus 0.669, -0.181 versus +0.324, and 0.468 versus 0.587. Experiment 21 fell catastrophically from 0.86 to 0.10 against 0.64. Experiment 10’s excessive exact-coalition choice rose from 0.481 to 0.581, Experiment 15 worsened from -0.007 to -0.086, and Experiment 16 fell from 0.528 to 0.471 despite the added redundant-stratum surcharge. The near-over-exact result in Experiment 22 remains reasonably close, and positional nulls remain intact, but those successes were already present in iteration 4. The pattern indicates that the new global coherence gate and coalition-level mean-diagnosticity term are not selective: they suppress many useful multi-expert configurations along with chains, while the redundant-stratum statistic either fails to distinguish the intended duplicate structures or is overwhelmed by the other additions.
Rationale: Because the latest advice produced a rejected candidate, do not repeat or intensify its package of global-span gating, post-clustering mean-quality bonuses, second-stratum surcharge, and expanded triad gain. Build again from the unchanged accepted iteration-4 base. Remove the new coalition-level coherence multiplier and bounded mean-diagnosticity term rather than trying to retune their current ranges; their joint introduction is associated with the wrong-sign reversals in Experiments 1, 12–14, and 21. Also restore the iteration-4 triad-gain range: its small improvement in Experiment 6 was outweighed by the large Experiment 15 regression.

For the next attempt, make one more local in-family correction to chain overvaluation: change how corroboration is aggregated, not how the entire coalition is penalized. Compute independence credit once between posterior reliability strata, with a smooth cap or soft maximum-matching over distinguishable stratum pairs, rather than allowing every locally narrow expert pair to contribute. Weight each stratum-level link by compact support and balanced cluster mass. This should prevent chains in Experiments 18 and 20 from accumulating several narrow-link benefits while preserving compact replicated endpoints and other coherent multi-member coalitions that the rejected global-span gate destroyed. Keep the successful dyadic Goldilocks function unchanged to protect Experiment 22.

Before reintroducing any duplicate-stratum cost, unit-test its sufficient statistic on the exact configurations from Experiments 1 and 16. The current surcharge moved Experiment 16 in the opposite of its intended direction, showing that expected `replicated_strata` does not reliably order single- versus double-duplicate structures under the soft posterior. Any later correction should operate on posterior-supported excess mass per occupied stratum and demonstrably assign greater redundancy load to the double-duplicate coalition while retaining the compressibility advantage of two balanced replicated endpoints over unsupported singleton levels. Make this separable from corroboration and quality terms. Finally, do not pursue Experiments 10 and 12–14 with another global mean-validity term; the rejected result shows that this normalization has the wrong cross-experiment consequences. First test whether cluster-support/compressibility normalization itself correctly ranks their configurations, then consider only a very small signed quality contrast at the final comparison stage if necessary.

**Outcome of this advice:** iter 5 candidate loss=0.2705 -> iter 6 candidate loss=0.2269 -> the gate marked it ACCEPTED.

### Iteration 6
Verdict: regenerate
Interpretation: Iteration 6 is accepted and improves the running-best loss from 0.2329 to 0.2269, so the prior recommendation to replace global coalition penalties with local stratum-level corroboration pooling was productive. It reduces the chain excesses in Experiments 18 and 20 from 0.678/0.711 to 0.623/0.663 while preserving the excellent dyadic Goldilocks result in Experiment 22 (0.670 versus 0.682). It also retains the intended positional nulls in Experiments 4, 5, and 7–9 and remains faithful to Coherence-Constrained Source Clustering. Nevertheless, the account is not experiment-invariant. Chains remain substantially overvalued in Experiments 18 and 20, wide dyads have the wrong contrast in Experiment 19 (+0.111 versus -0.028), and rejection of fully separated tetrads remains much too weak in Experiment 23 (-0.097 versus -0.354). Experiment 21 also remains overly categorical toward diverse tetrads (0.90 versus 0.64). The strongest-link cap therefore still allows an isolated favorable link within an otherwise unsupported coalition to confer too much corroboration. Major independent failures also persist: Experiment 2 has the wrong sign (+0.042 versus -0.264), Experiment 10 greatly overselects exact dyads (0.470 versus 0.172), Experiments 12–14 miss distributed-coalition preferences, and Experiments 16–17 show far too little accumulating redundancy/cardinality structure. Experiment 6 remains well below the human three-versus-two rate (0.16 versus 0.44). Several successful point estimates are also too heterogeneous, especially Experiments 1 and 22, whereas Experiment 6 lacks the heterogeneity implied by its target mean.
Rationale: Build on iteration 6 and preserve the accepted local stratum-pooling idea, the direct dyadic resolution gate, and the compact-replication criterion. Do not return to the rejected iteration-5 package of global-span gating, post-clustering mean-quality bonuses, or a separate duplicate-stratum surcharge. Instead, refine the accepted aggregation locally: the current maximum stratum link only asks whether one good compact pair exists. Weight that selected link by how much of the coalition is covered by its two supported strata, and discount posterior mass left in unsupported or weakly occupied residual strata. A genuine balanced 2+2 structure should have nearly complete coverage, whereas a chain or four singleton-like levels should not receive full value from one locally narrow link. Strengthen the transition to this coverage-limited credit modestly for n>=4, rather than imposing a global spread penalty. This should continue the accepted improvements in Experiments 18, 20, 21, and 23 while leaving dyads and triads—and therefore Experiment 22—unchanged.

Next, unit-test and recalibrate the existing `compact_excess` statistic rather than adding the previously rejected second-stratum surcharge. It should demonstrably assign more redundancy to the double-duplicate coalition in Experiment 16 than to the single-duplicate coalition, while preserving the compressibility advantage of a balanced replicated 2+2 tetrad over four unsupported levels in Experiment 1. If the posterior currently collapses both patterns into similar low-k solutions, adjust cluster support or complexity normalization so supported mass-2 strata are resolved consistently; then modestly increase the existing compact-excess cost only if that ordering is correct. This same correction should deepen the shallow Experiment 17 profile.

Finally, tune the two-to-three cardinality release separately from general structure. A modest isolated increase or widening of `triad_transition_gain`, coupled with subject variation in that parameter rather than greater cue salience, is warranted because Experiment 6 remains at 0.16 versus 0.44. Keep saturation beyond three intact. Conversely, narrow `salience_strength` or its profile dispersion slightly, since Experiments 1 and 22 have substantially excessive between-subject variance and exchangeable salience cannot explain the missing cardinality heterogeneity. These are targeted in-family adjustments that extend the direction validated by the accept gate without repeating the rejected global intervention.

**Outcome of this advice:** iter 6 candidate loss=0.2269 -> iter 7 candidate loss=0.2338 -> the gate marked it REJECTED.

### Iteration 7 (most recent — address this)
Verdict: regenerate
Interpretation: The current candidate is generally faithful to Coherence-Constrained Source Clustering, but it was rejected by the accept gate: loss increased from 0.2269 to 0.2338. The new coverage-weighted strongest-link correction did not deliver its intended selective improvement. Experiment 18 worsened from 0.623 to 0.651 against 0.498, Experiment 23 deteriorated sharply from -0.097 to -0.041 against -0.354, Experiment 21 remained at 0.90 against 0.64, and Experiment 20 improved only trivially from 0.663 to 0.654 against 0.504. Thus weighting one selected link by its pair coverage still allows an overly favorable representation of chains and dispersed tetrads. The widened triad transition produced only a small Experiment 6 gain, from 0.16 to 0.18 versus 0.44, and cannot justify the aggregate regression. Narrower salience did reduce variance in Experiments 1 and 22, but both remain more heterogeneous than the real data. Major cross-experiment failures persist: Experiments 2 and 3 have the wrong sign; Experiment 10 remains near indifference rather than strongly rejecting exact dyads; Experiments 12–14 miss the distributed-coalition effects; Experiments 16–17 understate redundancy/cardinality structure; and wide or chain-like coalitions remain overvalued in Experiments 18–20 and insufficiently rejected in Experiment 23. The dyadic near-over-exact effect in Experiment 22 and positional nulls in Experiments 4, 5, and 7–9 remain good.
Rationale: Build from the unchanged accepted iteration-6 base. The most recent coverage-weighted maximum-link recommendation was rejected, so do not repeat it, strengthen the n>=4 transition toward it, or further widen triad_transition_gain. Instead, revise stratum-level corroboration aggregation in a genuinely different way: score an entire posterior partition rather than selecting one privileged pair. For each occupied stratum, compute its best compact, supported link to another stratum, then aggregate these per-stratum values using posterior mass and a soft-min or low-temperature generalized mean. Unsupported residual strata should therefore contribute low values rather than being ignored, while a genuine two-stratum 2+2 partition retains its single coherent relation for all coalition mass. This remains a local consensus-coherence calculation and avoids the previously rejected global-span gate.

There is also an implementation-level faithfulness issue: coherence_width is loaded but never used; broad-dispersion onset is instead derived entirely from corroboration_peak and corroboration_width. This conflates the narrow independence peak with the separate between-cluster coherence parameter required by the arbiter. Use coherence_width to control a distinct, smooth dyadic tail decline, calibrated between the .004–.006 near-gap regime and the .014 broad regime. Keep Experiment 22's direct resolution gate unchanged. Apply the existing contextual-quality moderation only to this coherence penalty—without adding the rejected post-clustering mean-quality bonus—so low-quality broad dyads in Experiments 19 and 24 are penalized while high-quality unequal dyads in Experiment 10 are not indiscriminately suppressed.

Before changing redundancy cost, verify deterministic coalition scores for the exact stimuli in Experiments 1, 16, and 17. The current compact_excess machinery still does not generate the required cardinality profile, but the previously attempted separate duplicate-stratum surcharge was rejected. Any adjustment should therefore modify how existing supported excess mass is inferred or normalized, not add another surcharge. Prioritize the partition-wide link aggregation and the currently missing independent coherence scale as isolated changes; retain the accepted dyadic Goldilocks mechanism and positional exchangeability.

**Outcome of this advice:** iter 7 candidate loss=0.2338 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Coherence-Constrained Reliability-Strata Inference proposes that observers interpret an agreeing coalition by inferring a compact posterior distribution over latent reliability strata. Experts are assigned continuously rather than categorically to one through four possible source strata. Exact or virtually exact experts have a high probability of sharing a source and therefore incur accumulating redundancy. A small but nonzero reliability difference instead provides maximal corroboration because it supports distinguishable sources that remain mutually coherent. Larger separations progressively lose consensus coherence, producing a Goldilocks-shaped relation between reliability gap and coalition value. Coalition structure also matters: a small, balanced collection of compact strata supported by multiple experts is easier to compress and receives greater confidence than either one undifferentiated duplicate mass or many weakly supported singleton levels. Replication protection is granted only when posterior cluster mass is concentrated tightly around a stratum center; diffuse chains cannot obtain this protection merely through broad coassignment. For coalitions beyond triadic capacity, independence credit is evaluated across the entire posterior partition: every occupied stratum must have a compact, supported relation to another stratum, and weak residual strata lower partition coherence. A genuine balanced two-stratum structure can therefore retain corroboration, whereas a chain cannot obtain full credit from one privileged local link. Between-stratum coherence has a separate scale from the narrow independence peak, so a dyad can progress from redundancy at equality, to coherent corroboration at a narrow gap, and then to declining consensus at a broader gap. A replicated two-stratum tetrad can outperform four unsupported reliability levels, although each compact stratum's excess membership adds redundancy. Instructed reliability has a weak but consequential bounded influence after compression. Coalition magnitude saturates smoothly under finite capacity, with a modest corroborative release at the two-to-three transition that decays beyond three. Exchangeable idiosyncratic cue salience introduces individual variation without population-level position, onset, closure, adjacency, or recency effects.",
  "predict": "def predict(parameters, state, history) -> np.ndarray:\n    stim = np.asarray(state, dtype=np.float64)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"Coherence-Constrained Reliability-Strata Inference expects \"\n            f\"shape (2, n_features); got {stim.shape}.\"\n        )\n\n    a, b = stim[0], stim[1]\n    n_features = int(stim.shape[1])\n\n    v = np.asarray(parameters[\"validities\"], dtype=np.float64)\n    if v.ndim != 1 or v.size != n_features:\n        raise ValueError(\n            f\"validities length {v.size} != n_features {n_features}.\"\n        )\n    v = np.clip(v, 0.500001, 0.999999)\n\n    raw_salience = np.asarray(\n        parameters[\"cue_salience_profile\"], dtype=np.float64\n    )\n    if raw_salience.ndim != 1 or raw_salience.size != n_features:\n        raise ValueError(\n            f\"cue_salience_profile length {raw_salience.size} != \"\n            f\"n_features {n_features}.\"\n        )\n\n    validity_compression = float(parameters[\"validity_compression\"])\n    validity_reliance = float(parameters[\"validity_reliance\"])\n    salience_strength = float(parameters[\"salience_strength\"])\n    redundancy_width = float(parameters[\"redundancy_width\"])\n    redundancy_cost = float(parameters[\"redundancy_cost\"])\n    compact_excess_cost = float(parameters[\"compact_excess_cost\"])\n    corroboration_peak = float(parameters[\"corroboration_peak\"])\n    corroboration_width = float(parameters[\"corroboration_width\"])\n    independence_benefit = float(parameters[\"independence_benefit\"])\n    coherence_width = float(parameters[\"coherence_width\"])\n    coherence_cost = float(parameters[\"coherence_cost\"])\n    cluster_width = float(parameters[\"cluster_width\"])\n    cluster_complexity = float(parameters[\"cluster_complexity\"])\n    cluster_temperature = float(parameters[\"cluster_temperature\"])\n    support_threshold = float(parameters[\"support_threshold\"])\n    support_slope = float(parameters[\"support_slope\"])\n    cluster_support_bonus = float(parameters[\"cluster_support_bonus\"])\n    capacity_saturation = float(parameters[\"capacity_saturation\"])\n    count_curvature = float(parameters[\"count_curvature\"])\n    triad_transition_gain = float(parameters[\"triad_transition_gain\"])\n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n\n    # Communicated reliability is used, but diagnosticity differences are\n    # strongly compressed so that source organization can dominate isolated\n    # numerical reliability advantages.\n    diagnosticity = np.log(v / (1.0 - v))\n    represented = np.power(\n        np.maximum(diagnosticity, 1e-12), validity_compression\n    )\n    represented /= max(float(np.mean(represented)), 1e-12)\n    reliability_weight = (\n        (1.0 - validity_reliance) * np.ones(n_features, dtype=np.float64)\n        + validity_reliance * represented\n    )\n\n    # The salience profile is exchangeable across expert identities and is\n    # centered within subject. It therefore adds stable heterogeneity without\n    # imposing a common serial-position or spatial-organization gradient.\n    centered_salience = raw_salience - float(np.mean(raw_salience))\n    attention = np.exp(\n        np.clip(salience_strength * centered_salience, -10.0, 10.0)\n    )\n    attention /= max(float(np.mean(attention)), 1e-12)\n    cue_quality = reliability_weight * attention\n    cue_quality /= max(float(np.mean(cue_quality)), 1e-12)\n\n    difference = a - b\n    a_support = np.flatnonzero(difference > 0.0)\n    b_support = np.flatnonzero(difference < 0.0)\n\n    if a_support.size == 0 and b_support.size == 0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n\n    def soft_partition(x, k):\n        \"\"\"Deterministic soft clustering for one candidate stratum count.\"\"\"\n        n = int(x.size)\n        if k == 1:\n            assignments = np.ones((n, 1), dtype=np.float64)\n            centers = np.array([float(np.mean(x))], dtype=np.float64)\n            return assignments, centers\n\n        xmin = float(np.min(x))\n        xmax = float(np.max(x))\n        if xmax - xmin <= 1e-15:\n            centers = np.full(k, xmin, dtype=np.float64)\n        else:\n            quantiles = np.linspace(0.0, 1.0, k)\n            centers = np.quantile(x, quantiles).astype(np.float64)\n\n        assignments = np.full((n, k), 1.0 / float(k), dtype=np.float64)\n        for _ in range(16):\n            squared = (x[:, None] - centers[None, :]) ** 2\n            logits = -0.5 * squared / max(cluster_width ** 2, 1e-12)\n            logits -= np.max(logits, axis=1, keepdims=True)\n            assignments = np.exp(logits)\n            assignments /= np.maximum(\n                np.sum(assignments, axis=1, keepdims=True), 1e-12\n            )\n            masses = np.sum(assignments, axis=0)\n            updated = np.sum(assignments * x[:, None], axis=0) / np.maximum(\n                masses, 1e-12\n            )\n            centers = 0.65 * centers + 0.35 * updated\n\n        order = np.argsort(centers)\n        return assignments[:, order], centers[order]\n\n    def coalition_signal(indices):\n        idx = np.asarray(indices, dtype=np.int64)\n        n = int(idx.size)\n        if n == 0:\n            return 0.0\n        if n == 1:\n            return float(cue_quality[idx[0]])\n\n        x = v[idx]\n        qualities = cue_quality[idx]\n\n        # Smooth Bayesian model comparison over a small number of reliability\n        # strata. The likelihood rewards fit, while the complexity term makes\n        # unsupported singleton strata expensive. When all values coincide,\n        # the resolution gate strongly favors a single latent source.\n        max_k = int(min(4, n))\n        models = []\n        model_scores = []\n        observed_range = float(np.max(x) - np.min(x))\n        resolution_gate = 1.0 - np.exp(\n            -0.5 * (observed_range / max(cluster_width, 1e-12)) ** 2\n        )\n\n        for k in range(1, max_k + 1):\n            assignments, centers = soft_partition(x, k)\n            masses = np.sum(assignments, axis=0)\n            residual = np.sum(\n                assignments * (x[:, None] - centers[None, :]) ** 2\n            )\n            fit_cost = residual / max(\n                float(n) * cluster_width ** 2, 1e-12\n            )\n\n            support = 1.0 / (\n                1.0\n                + np.exp(\n                    -np.clip(\n                        support_slope * (masses - support_threshold),\n                        -50.0,\n                        50.0,\n                    )\n                )\n            )\n            unsupported_cost = float(np.mean(1.0 - support))\n            extra_strata = float(k - 1)\n            resolution_penalty = (\n                cluster_complexity\n                * extra_strata\n                / max(0.12 + resolution_gate, 1e-12)\n            )\n            score = -fit_cost - resolution_penalty - 0.35 * unsupported_cost\n            models.append((assignments, centers, masses, support))\n            model_scores.append(score)\n\n        model_scores = np.asarray(model_scores, dtype=np.float64)\n        posterior_logits = model_scores / max(cluster_temperature, 1e-12)\n        posterior_logits -= np.max(posterior_logits)\n        posterior = np.exp(posterior_logits)\n        posterior /= max(float(np.sum(posterior)), 1e-12)\n\n        # Posterior coassignment is a continuous estimate that two experts\n        # derive from the same latent source.\n        coassignment = np.zeros((n, n), dtype=np.float64)\n        compressibility = 0.0\n        expected_strata = 0.0\n        compact_excess = 0.0\n        stratum_link_credit = 0.0\n        for probability, model in zip(posterior, models):\n            assignments, centers, masses, support = model\n            k = int(assignments.shape[1])\n            coassignment += probability * (assignments @ assignments.T)\n            expected_strata += probability * float(k)\n\n            # Replication requires both excess posterior mass and low\n            # within-stratum variance. Broad assignments over a smooth chain\n            # therefore do not count as replicated support.\n            within_variance = np.sum(\n                assignments * (x[:, None] - centers[None, :]) ** 2,\n                axis=0,\n            ) / np.maximum(masses, 1e-12)\n            concentration = np.exp(\n                -0.5\n                * within_variance\n                / max(redundancy_width ** 2, 1e-12)\n            )\n            model_compact_excess = float(\n                np.sum(\n                    np.maximum(masses - 1.0, 0.0)\n                    * support\n                    * concentration\n                )\n                / float(n)\n            )\n            compact_excess += probability * model_compact_excess\n\n            if k > 1:\n                proportions = masses / max(float(np.sum(masses)), 1e-12)\n                entropy = -float(\n                    np.sum(proportions * np.log(np.maximum(proportions, 1e-12)))\n                ) / max(np.log(float(k)), 1e-12)\n                supported_mass = float(\n                    np.sum(proportions * support * concentration)\n                )\n                compactness = 1.0 - float(k - 1) / float(max(n, 2))\n                model_compressibility = (\n                    entropy\n                    * supported_mass\n                    * max(compactness, 0.0)\n                    * (float(k - 1) / float(k))\n                )\n                compressibility += probability * model_compressibility\n\n                # Partition-wide consensus: each occupied stratum receives\n                # its best coherent relation to another compact, supported\n                # stratum. The mass-weighted geometric mean acts as a soft\n                # minimum, so unsupported residual strata cannot be ignored.\n                link_matrix = np.zeros((k, k), dtype=np.float64)\n                for ci in range(k):\n                    for cj in range(ci + 1, k):\n                        center_gap = abs(float(centers[ci] - centers[cj]))\n                        center_resolution = 1.0 - np.exp(\n                            -0.5\n                            * (\n                                center_gap\n                                / max(redundancy_width, 1e-12)\n                            ) ** 2\n                        )\n                        center_goldilocks = np.exp(\n                            -0.5\n                            * (\n                                (center_gap - corroboration_peak)\n                                / max(corroboration_width, 1e-12)\n                            ) ** 2\n                        )\n                        balanced_mass = (\n                            2.0 * min(float(masses[ci]), float(masses[cj]))\n                            / max(float(n), 1e-12)\n                        )\n                        compact_support = np.sqrt(\n                            max(float(support[ci] * support[cj]), 0.0)\n                            * max(\n                                float(concentration[ci] * concentration[cj]),\n                                0.0,\n                            )\n                        )\n                        link_value = (\n                            center_resolution\n                            * center_goldilocks\n                            * np.clip(balanced_mass, 0.0, 1.0)\n                            * compact_support\n                        )\n                        link_matrix[ci, cj] = link_value\n                        link_matrix[cj, ci] = link_value\n\n                best_by_stratum = np.max(link_matrix, axis=1)\n                partition_consensus = float(\n                    np.exp(\n                        np.sum(\n                            proportions\n                            * np.log(np.clip(best_by_stratum, 1e-4, 1.0))\n                        )\n                    )\n                )\n                stratum_link_credit += probability * partition_consensus\n\n        # Pairwise source evidence has a Goldilocks shape. Equality implies\n        # common-source redundancy; a resolvable narrow gap is corroborative;\n        # and a separate coherence scale controls the broader declining tail.\n        redundant_load = 0.0\n        corroborative_load = 0.0\n        broad_dispersion = 0.0\n        pair_count = 0.0\n\n        tail_scale = max(0.20 * coherence_width, 1e-3)\n        tail_onset = coherence_width\n        tail_floor = 1.0 / (\n            1.0 + np.exp(np.clip(tail_onset / tail_scale, -50.0, 50.0))\n        )\n\n        for i in range(n):\n            for j in range(i + 1, n):\n                gap = abs(float(x[i] - x[j]))\n                exact_similarity = np.exp(\n                    -0.5 * (gap / max(redundancy_width, 1e-12)) ** 2\n                )\n                redundant_load += coassignment[i, j] * exact_similarity\n\n                goldilocks = np.exp(\n                    -0.5\n                    * (\n                        (gap - corroboration_peak)\n                        / max(corroboration_width, 1e-12)\n                    ) ** 2\n                )\n                direct_resolution = 1.0 - np.exp(\n                    -0.5 * (gap / max(redundancy_width, 1e-12)) ** 2\n                )\n                corroborative_load += direct_resolution * goldilocks\n\n                tail = 1.0 / (\n                    1.0\n                    + np.exp(\n                        -np.clip(\n                            (gap - tail_onset) / tail_scale,\n                            -50.0,\n                            50.0,\n                        )\n                    )\n                )\n                tail = np.clip(\n                    (tail - tail_floor) / max(1.0 - tail_floor, 1e-12),\n                    0.0,\n                    1.0,\n                )\n                broad_dispersion += tail\n                pair_count += 1.0\n\n        pair_count = max(pair_count, 1.0)\n        mean_corroboration = corroborative_load / pair_count\n        mean_dispersion = broad_dispersion / pair_count\n\n        # Dyads and triads retain the accepted pairwise Goldilocks relation.\n        # Beyond triadic capacity, corroboration is constrained by consensus\n        # across the complete posterior partition rather than one best link.\n        higher_order_load = float(max(n - 3, 0))\n        stratum_pooling = 1.0 - np.exp(-higher_order_load)\n        partition_limited_corroboration = min(\n            mean_corroboration, stratum_link_credit\n        )\n        mean_corroboration = (\n            (1.0 - stratum_pooling) * mean_corroboration\n            + stratum_pooling * partition_limited_corroboration\n        )\n\n        # Only compact, genuinely replicated strata protect a coalition from\n        # dispersion. Posterior overlap by itself cannot shield a smooth chain.\n        replication_guard = 1.0 / (1.0 + 1.5 * compact_excess)\n        contextual_advantage = max(\n            float(np.mean(represented[idx])) - 1.0, 0.0\n        )\n        quality_guard = np.exp(-2.0 * contextual_advantage)\n        mean_dispersion *= replication_guard * quality_guard\n\n        # Redundancy accumulates with the number of close within-source pairs.\n        # Compact excess membership adds a smaller incremental cost, allowing\n        # a second supported duplicate group to remain consequential.\n        redundancy_per_member = redundant_load / float(n)\n\n        additional = capacity_saturation * (\n            1.0\n            - np.exp(\n                -float(n - 1) / max(capacity_saturation, 1e-12)\n            )\n        )\n        represented_count = (1.0 + additional) ** count_curvature\n\n        # A smooth two-to-three corroborative release supports heterogeneous\n        # three-versus-two judgments but decays beyond three under capacity.\n        triad_transition = 1.0 / (\n            1.0 + np.exp(-np.clip((float(n) - 2.5) / 0.18, -50.0, 50.0))\n        )\n        beyond_three_decay = np.exp(\n            -float(max(n - 3, 0)) / max(capacity_saturation, 1e-12)\n        )\n        represented_count *= np.exp(\n            triad_transition_gain * triad_transition * beyond_three_decay\n        )\n\n        # Source-credit weighting gives less influence to cues heavily\n        # coassigned with coalition partners without introducing cue position.\n        dependence_load = np.sum(coassignment, axis=1) - 1.0\n        uniqueness = 1.0 / np.maximum(1.0 + dependence_load, 1e-12)\n        quality_weights = 0.55 + 0.45 * uniqueness / max(\n            float(np.mean(uniqueness)), 1e-12\n        )\n        coalition_quality = float(\n            np.sum(quality_weights * qualities)\n            / max(float(np.sum(quality_weights)), 1e-12)\n        )\n\n        structural_log_value = (\n            -redundancy_cost * redundancy_per_member\n            -compact_excess_cost * compact_excess\n            + independence_benefit * mean_corroboration\n            - coherence_cost * mean_dispersion\n            + cluster_support_bonus * compressibility\n        )\n        structural_factor = np.exp(\n            np.clip(structural_log_value, -8.0, 8.0)\n        )\n\n        # Expected strata mildly calibrate confidence but are capacity-limited;\n        # they do not create an unbounded second count term.\n        source_confidence = 1.0 + 0.06 * (\n            1.0\n            - np.exp(\n                -(expected_strata - 1.0)\n                / max(capacity_saturation, 1e-12)\n            )\n        )\n\n        signal = (\n            represented_count\n            * coalition_quality\n            * structural_factor\n            * source_confidence\n        )\n        return float(max(signal, 0.0))\n\n    evidence_a = coalition_signal(a_support)\n    evidence_b = coalition_signal(b_support)\n    net_a = evidence_a - evidence_b\n\n    logits = np.array(\n        [0.5 * beta * net_a, -0.5 * beta * net_a], dtype=np.float64\n    )\n    logits -= np.max(logits)\n    probs = np.exp(logits)\n    probs /= max(float(np.sum(probs)), 1e-12)\n\n    probs = (1.0 - epsilon) * probs + epsilon * 0.5\n    probs = np.maximum(probs, 0.0)\n    total = float(np.sum(probs))\n    if not np.isfinite(total) or total <= 0.0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n    return probs / total\n",
  "policy": "def policy(probs) -> int:\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.maximum(probs, 0.0)\n    total = float(np.sum(probs))\n    if not np.isfinite(total) or total <= 0.0:\n        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))\n    else:\n        probs /= total\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "validities": "validities",
    "cue_salience_profile": "[(-1.0, 1.0)] * n_features",
    "validity_compression": "[0.08, 0.18]",
    "validity_reliance": "[0.020, 0.065]",
    "salience_strength": "[0.10, 0.28]",
    "redundancy_width": "[0.0015, 0.0045]",
    "redundancy_cost": "[0.70, 1.35]",
    "compact_excess_cost": "[0.06, 0.20]",
    "corroboration_peak": "[0.004, 0.007]",
    "corroboration_width": "[0.003, 0.007]",
    "independence_benefit": "[0.18, 0.55]",
    "coherence_width": "[0.007, 0.011]",
    "coherence_cost": "[0.28, 0.72]",
    "cluster_width": "[0.004, 0.014]",
    "cluster_complexity": "[0.10, 0.28]",
    "cluster_temperature": "[0.45, 0.90]",
    "support_threshold": "[1.35, 1.75]",
    "support_slope": "[4.0, 8.0]",
    "cluster_support_bonus": "[0.30, 0.82]",
    "capacity_saturation": "[0.90, 3.10]",
    "count_curvature": "[0.34, 0.74]",
    "triad_transition_gain": "[0.04, 0.24]",
    "beta": "[1.25, 1.75]",
    "epsilon": "[0.025, 0.070]"
  },
  "rationale": "This is a targeted edit to the accepted iteration-6 model. The direct dyadic resolution gate, compact-replication statistic, redundancy normalization, count saturation, triad transition, contextual-quality moderation, cue salience, and all other validated mechanisms are unchanged. First, the rejected coverage-weighted maximum-link correction is replaced by partition-wide stratum consensus. Each occupied stratum receives its best supported link, but these values are combined with a posterior-mass-weighted geometric mean. Consequently, a weak or unsupported residual stratum cannot be ignored, while a genuine two-stratum 2+2 partition assigns the same coherent relation to essentially all coalition mass. This targets chains and dispersed tetrads without imposing the rejected global-span multiplier. Second, the previously unused coherence_width now independently controls the onset of broad-gap dispersion. Its narrow range places the decline above the successful .004–.006 corroboration regime but below the .014 broad regime. The existing quality guard moderates only this tail penalty, preserving high-quality unequal dyads without adding the rejected coalition-level validity bonus. No redundancy or cardinality ranges were changed because the most recent broadened triad intervention was rejected."
}
```

## Usage

```json
{
  "input_tokens": 67173,
  "output_tokens": 6484
}
```
