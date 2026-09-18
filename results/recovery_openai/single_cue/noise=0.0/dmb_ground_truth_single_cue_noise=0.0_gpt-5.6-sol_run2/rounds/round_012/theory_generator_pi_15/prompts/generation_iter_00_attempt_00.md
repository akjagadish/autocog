# generation_iter_00_attempt_00

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
- THEORY 2 = `pi_14`
- The recommendation below acts on THEORY 2 (= `pi_14`).

Replace pi_14 with a theory of Capacity-Gated Local Source Individuation. Its core claim should be that source dependence is inferred locally among the closest reliability neighbors, while larger coalitions are evaluated through bounded, nontransitive evidence accounts rather than through a globally coherent latent partition. Exact equality should strongly activate a common-source tag and discount duplicate evidence. A small nonzero gap should release that tag and provide local independence credit, producing the near-over-exact effect in Experiment 24. For dyads, that credit should decline once the gap becomes too broad, capturing Experiment 26. Crucially, this dyadic Goldilocks function must not become a coalition-wide span penalty: in tetrads and larger coalitions, local links should be capacity-limited and nonaccumulating, while distinct reliability regions can still contribute separate bounded accounts. This should allow diffuse coalitions to beat compact duplicate-heavy coalitions in Experiment 1 and Experiments 14–16 and 18, while preventing chains from accumulating a large advantage in Experiments 20 and 22.

The new theory should include an explicit cardinality gate: dyads permit fine metric comparison of reliability gaps; at three or more cues, observers increasingly compress reliability into a few contextual codes and use bounded account coverage plus duplicate penalties. Any two-to-three release should be weak and empirically constrained, because Experiment 17 is near zero, and the model must reproduce the cardinality profile in Experiment 19 rather than imposing a generic triadic bonus. Reliability weighting should remain compressed but consequential, and duplicate penalties should accumulate by locally tagged group as in the successful portion of pi_11. Do not add a generic global coherence, total-span, or strongest-stratum-link term.

The model should also distinguish choice sensitivity from structural valuation so that near-chance means and subject-threshold metrics are not forced to move together. Stable subject heterogeneity may vary duplicate tolerance, local-gap resolution, reliability compression, and response sensitivity, but these distributions must be shared across experiments rather than selected per design. Retain exchangeable cue salience and avoid population-level recency, onset, adjacency, or closure gradients, given the repeated null results for those mechanisms. During development, jointly calibrate against all experiments, with hard qualitative checks that the model simultaneously produces: dispersed preference in Experiment 1; approximately chance compact choice in Experiment 2; near-over-exact preference in Experiment 24; broad-dyad avoidance in Experiment 26; approximately chance chain choice in Experiments 20 and 22; duplicate avoidance in Experiments 12 and 18; and distributed-coalition preference in Experiments 14–16. It should additionally incorporate a coalition-level normalization or contrast process capable of addressing the shared failures in Experiments 4, 5, 8, and 25, without invoking unsupported serial-position effects.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_7` (overall score: 0.668)

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
- Experiment 25: real=0.9200 (var=0.0736) vs this=1.0000 (var=0.0000)
- Experiment 26: real=0.4956 (var=0.0045) vs this=0.4556 (var=0.0044)


---

### `pi_13` (overall score: 0.661)

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
- Experiment 25: real=0.9200 (var=0.0736) vs this=0.7900 (var=0.1509)
- Experiment 26: real=0.4956 (var=0.0045) vs this=0.4075 (var=0.0084)


---

### `pi_12` (overall score: 0.611)

**Description**
Opponent-Relative Stratified Sampling Theory proposes that people compress ordered expert reliabilities into a small contextual vocabulary of reliability strata and jointly compare opposing coalitions by sampled evidence, stratum coverage, balance, and relative concentration. Each stratum has a limited number of representational slots, while a second capacity limit constrains total evidence maintained across strata. Coalition mass is partially normalized, allowing heterogeneous dyad, triad, and tetrad effects without a single global coalition-size power law. Repetition is not intrinsically penalized: its effect depends on the rival coalition's concentration, coverage, and size. This opponent-relative contrast is strongest for diagnostic dyads. When equal-sized dyads have matched duplicate status, their individual mean qualities are partially assimilated toward a pooled comparison standard, attenuating arbitrary ordinal-composition contrasts without eliminating reliability information. For equally large coalitions of at least four experts, a one-sided duplicate-versus-chain contrast is pooled through the same capacity-limited representation, preventing generic diversity from producing a chain advantage. Concentration comparisons remain available when both large coalitions contain repetition but differ in its degree. Weak ordinal reliability weighting and exchangeable centered cue salience preserve sensitivity to instructed reliability without metric-distance, chain, onset, recency, closure, or adjacency effects.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Opponent-Relative Stratified Sampling expects shape "
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

    stratum_resolution = int(parameters["stratum_resolution"])
    sampling_capacity = int(parameters["sampling_capacity"])
    global_capacity = float(parameters["global_capacity"])
    occupancy_normalization = float(parameters["occupancy_normalization"])
    ordinal_reliance = float(parameters["ordinal_reliance"])
    salience_strength = float(parameters["salience_strength"])
    coverage_preference = float(parameters["coverage_preference"])
    balance_preference = float(parameters["balance_preference"])
    redundancy_sensitivity = float(parameters["redundancy_sensitivity"])
    rival_size_weight = float(parameters["rival_size_weight"])
    context_curvature = float(parameters["context_curvature"])
    matched_dyad_quality_shrinkage = float(
        parameters["matched_dyad_quality_shrinkage"]
    )
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    v = np.clip(validities, 0.500001, 0.999999)

    # Contextual strata are formed from the ordering of distinct instructed
    # reliability levels. No metric bandwidth, pairwise similarity kernel, or
    # transitive similarity graph is used. Equal levels necessarily share a
    # stratum, while small and large nonzero gaps are treated alike whenever
    # they occupy the same ordinal portion of the reliability vocabulary.
    unique_v, inverse_levels = np.unique(v, return_inverse=True)
    n_levels = int(unique_v.size)
    actual_strata = max(1, min(stratum_resolution, n_levels))

    if n_levels == 1:
        level_strata = np.zeros(1, dtype=np.int64)
        ordinal_contrast = np.zeros(1, dtype=np.float64)
    else:
        level_strata = np.floor(
            np.arange(n_levels, dtype=np.float64)
            * float(actual_strata)
            / float(n_levels)
        ).astype(np.int64)
        level_strata = np.minimum(level_strata, actual_strata - 1)
        ordinal_position = (
            np.arange(n_levels, dtype=np.float64) / float(n_levels - 1)
        )
        ordinal_contrast = 2.0 * ordinal_position - 1.0

    cue_strata = level_strata[inverse_levels]

    # Reliability affects cue strength only through weak ordinal contrast.
    # Consequently, changing a gap from .003 to .0058 has no direct effect.
    reliability_weight = np.exp(
        np.clip(ordinal_reliance * ordinal_contrast[inverse_levels], -10.0, 10.0)
    )
    reliability_weight /= max(float(np.mean(reliability_weight)), 1e-12)

    # Stable cue salience is centered and exchangeable over cue identity.
    centered_salience = raw_salience - float(np.mean(raw_salience))
    attention = np.exp(
        np.clip(salience_strength * centered_salience, -10.0, 10.0)
    )
    attention /= max(float(np.mean(attention)), 1e-12)

    cue_weights = reliability_weight * attention
    cue_weights /= max(float(np.mean(cue_weights)), 1e-12)

    difference = a - b
    a_support = np.flatnonzero(difference > 0.0)
    b_support = np.flatnonzero(difference < 0.0)

    if a_support.size == 0 and b_support.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    def summarize(indices):
        idx = np.asarray(indices, dtype=np.int64)
        n = int(idx.size)
        if n == 0:
            return {
                "n": 0,
                "coverage": 0,
                "balance": 0.0,
                "duplicate_fraction": 0.0,
                "sampled_signal": 0.0,
                "mean_quality": 0.0,
            }

        strata = cue_strata[idx]
        counts = np.bincount(strata, minlength=actual_strata).astype(np.float64)
        occupied = counts > 0.0
        coverage = int(np.sum(occupied))

        # Each reliability stratum has a finite set of representational slots.
        # This is the expected number of occupied slots after placing the
        # stratum's members into those slots. It creates discrete diminishing
        # returns without a global coalition-size power law.
        cap = float(sampling_capacity)
        expected_slots = cap * (
            1.0 - np.power(1.0 - 1.0 / cap, counts)
        )

        stratum_signal = 0.0
        for s in range(actual_strata):
            members = idx[strata == s]
            if members.size == 0:
                continue
            quality = float(np.mean(cue_weights[members]))
            stratum_signal += quality * float(expected_slots[s])

        # Partial normalization lies between summing occupied evidence and
        # fully averaging it. Sampling-capacity heterogeneity can therefore
        # preserve different dyad, triad, and tetrad profiles.
        expected_total = max(float(np.sum(expected_slots)), 1e-12)
        normalized_signal = stratum_signal / (
            expected_total ** occupancy_normalization
        )

        # A second capacity limit reflects how much sampled evidence can be
        # maintained across all strata on a trial.
        sampled_signal = global_capacity * (
            1.0 - np.exp(-normalized_signal / max(global_capacity, 1e-12))
        )

        if coverage <= 1:
            balance = 0.0
        else:
            proportions = counts[occupied] / float(n)
            entropy = -float(np.sum(proportions * np.log(proportions)))
            balance = entropy / max(float(np.log(coverage)), 1e-12)

        # Repetition is represented categorically at exact instructed levels.
        # It is only a vulnerability passed to the joint comparison below and
        # is not deducted from this coalition's evidence intrinsically.
        level_counts = np.bincount(
            inverse_levels[idx], minlength=n_levels
        ).astype(np.float64)
        duplicate_excess = float(np.sum(np.maximum(level_counts - 1.0, 0.0)))
        duplicate_fraction = duplicate_excess / float(max(n - 1, 1))

        return {
            "n": n,
            "coverage": coverage,
            "balance": balance,
            "duplicate_fraction": duplicate_fraction,
            "sampled_signal": sampled_signal,
            "mean_quality": float(np.mean(cue_weights[idx])),
        }

    sa = summarize(a_support)
    sb = summarize(b_support)

    # Coverage is evaluated relative to the contextual strata that are
    # represented by either opposing coalition on this trial.
    union_indices = np.concatenate((a_support, b_support)).astype(np.int64)
    if union_indices.size == 0:
        union_coverage = 1
    else:
        union_coverage = max(
            int(np.unique(cue_strata[union_indices]).size), 1
        )

    def structural_value(summary):
        if summary["n"] == 0:
            return 0.0
        if union_coverage <= 1:
            coverage_fraction = 0.0
        else:
            coverage_fraction = float(summary["coverage"] - 1) / float(
                union_coverage - 1
            )
        multiplier = (
            1.0
            + coverage_preference * coverage_fraction
            + balance_preference * summary["balance"]
        )
        return float(summary["sampled_signal"] * multiplier)

    value_a = structural_value(sa)
    value_b = structural_value(sb)

    # When equal-sized dyads have the same categorical duplicate status,
    # ordinal-bin coverage is not treated as diagnostic. Both receive the
    # same pooled two-member occupancy scale. Their mean qualities are also
    # assimilated toward their joint mean, leaving only a reduced residual
    # ordinal/salience contrast rather than an arbitrary broad-gap effect.
    matched_dyads = (
        sa["n"] == 2
        and sb["n"] == 2
        and ((sa["duplicate_fraction"] > 0.0)
             == (sb["duplicate_fraction"] > 0.0))
    )
    if matched_dyads:
        cap = float(sampling_capacity)
        pooled_slots = cap * (
            1.0 - (1.0 - 1.0 / cap) ** 2.0
        )
        pooled_scale = pooled_slots / (
            max(pooled_slots, 1e-12) ** occupancy_normalization
        )
        pooled_quality = 0.5 * (
            sa["mean_quality"] + sb["mean_quality"]
        )
        quality_a = pooled_quality + matched_dyad_quality_shrinkage * (
            sa["mean_quality"] - pooled_quality
        )
        quality_b = pooled_quality + matched_dyad_quality_shrinkage * (
            sb["mean_quality"] - pooled_quality
        )
        value_a = global_capacity * (
            1.0 - np.exp(
                -quality_a * pooled_scale
                / max(global_capacity, 1e-12)
            )
        )
        value_b = global_capacity * (
            1.0 - np.exp(
                -quality_b * pooled_scale
                / max(global_capacity, 1e-12)
            )
        )

    # A diverse chain and an exact block are not treated as intrinsically
    # different sources when both coalitions exceed working-memory capacity.
    # Pooling only the one-sided duplicate contrast leaves comparisons between
    # two differently concentrated repeated coalitions available.
    large_one_sided_contrast = (
        sa["n"] == sb["n"]
        and sa["n"] >= 4
        and ((sa["duplicate_fraction"] > 0.0)
             != (sb["duplicate_fraction"] > 0.0))
    )
    if large_one_sided_contrast:
        cap = float(sampling_capacity)
        pooled_slots = cap * (
            1.0 - (1.0 - 1.0 / cap) ** float(sa["n"])
        )
        pooled_scale = pooled_slots / (
            max(pooled_slots, 1e-12) ** occupancy_normalization
        )
        value_a = global_capacity * (
            1.0 - np.exp(
                -sa["mean_quality"] * pooled_scale
                / max(global_capacity, 1e-12)
            )
        )
        value_b = global_capacity * (
            1.0 - np.exp(
                -sb["mean_quality"] * pooled_scale
                / max(global_capacity, 1e-12)
            )
        )

    def contextual_vulnerability(focal, rival):
        if focal["n"] == 0 or focal["duplicate_fraction"] <= 0.0:
            return 0.0

        # One-sided duplicate-versus-chain contrasts are pooled at large
        # cardinalities, so they do not receive an additional vulnerability.
        if large_one_sided_contrast:
            return 0.0

        coverage_advantage = float(
            rival["coverage"] - focal["coverage"]
        ) / float(max(union_coverage, 1))
        size_advantage = float(rival["n"] - focal["n"]) / float(
            max(rival["n"], focal["n"], 1)
        )

        # A bounded direct comparison allows equality avoidance when an
        # equally large rival is less concentrated. It is moderately stronger
        # for dyads, where exact repetition is especially diagnostic.
        concentration_advantage = max(
            0.0,
            float(focal["duplicate_fraction"] - rival["duplicate_fraction"]),
        )
        bounded_concentration = 1.0 - np.exp(
            -3.0 * concentration_advantage
        )
        concentration_weight = (
            1.10 if focal["n"] == 2 and rival["n"] == 2 else 0.75
        )
        opponent_context = max(
            0.0,
            coverage_advantage
            + rival_size_weight * size_advantage
            + concentration_weight * bounded_concentration,
        )
        opponent_context = opponent_context ** context_curvature
        return float(focal["duplicate_fraction"] * opponent_context)

    vulnerability_a = contextual_vulnerability(sa, sb)
    vulnerability_b = contextual_vulnerability(sb, sa)

    adjusted_a = value_a * np.exp(
        -redundancy_sensitivity * vulnerability_a
    )
    adjusted_b = value_b * np.exp(
        -redundancy_sensitivity * vulnerability_b
    )
    net_a = float(adjusted_a - adjusted_b)

    logits = np.array(
        [0.5 * beta * net_a, -0.5 * beta * net_a], dtype=np.float64
    )
    logits = logits - np.max(logits)
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
- stratum_resolution: {2, 3, 4}
- sampling_capacity: {1, 2, 3}
- global_capacity: [2.2, 4.8]
- occupancy_normalization: [0.48, 0.80]
- ordinal_reliance: [0.005, 0.035]
- salience_strength: [0.08, 0.24]
- coverage_preference: [0.14, 0.38]
- balance_preference: [0.08, 0.30]
- redundancy_sensitivity: [0.65, 1.45]
- rival_size_weight: [0.20, 0.55]
- context_curvature: [0.70, 1.30]
- matched_dyad_quality_shrinkage: [0.18, 0.38]
- beta: [1.25, 1.70]
- epsilon: [0.03, 0.08]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.1217 (var=0.0117) vs this=-0.3588 (var=0.0139)
- Experiment 2: real=-0.2642 (var=0.0087) vs this=0.0117 (var=0.0092)
- Experiment 3: real=-0.0410 (var=0.0003) vs this=0.0144 (var=0.0003)
- Experiment 4: real=0.5087 (var=0.0038) vs this=0.5132 (var=0.0055)
- Experiment 5: real=0.5021 (var=0.0024) vs this=0.4983 (var=0.0020)
- Experiment 6: real=0.4400 (var=0.2464) vs this=0.8400 (var=0.1344)
- Experiment 7: real=0.0219 (var=0.0094) vs this=-0.0353 (var=0.0117)
- Experiment 8: real=0.0117 (var=0.0101) vs this=0.0358 (var=0.0069)
- Experiment 9: real=0.4958 (var=0.0041) vs this=0.4890 (var=0.0018)
- Experiment 10: real=0.1723 (var=0.0081) vs this=0.3006 (var=0.0042)
- Experiment 11: real=0.0317 (var=0.0030) vs this=0.1071 (var=0.0100)
- Experiment 12: real=0.6692 (var=0.0068) vs this=0.6571 (var=0.0061)
- Experiment 13: real=0.3242 (var=0.0194) vs this=0.3475 (var=0.0228)
- Experiment 14: real=0.5869 (var=0.0047) vs this=0.5878 (var=0.0035)
- Experiment 15: real=0.0142 (var=0.0045) vs this=-0.1379 (var=0.0071)
- Experiment 16: real=0.6779 (var=0.0032) vs this=0.7210 (var=0.0079)
- Experiment 17: real=-0.3486 (var=0.0131) vs this=-0.2260 (var=0.0128)
- Experiment 18: real=0.4983 (var=0.0018) vs this=0.4948 (var=0.0035)
- Experiment 19: real=-0.0283 (var=0.0129) vs this=-0.0354 (var=0.0045)
- Experiment 20: real=0.5042 (var=0.0027) vs this=0.5085 (var=0.0024)
- Experiment 21: real=0.6400 (var=0.2304) vs this=0.0000 (var=0.0000)
- Experiment 22: real=0.6817 (var=0.0026) vs this=0.7017 (var=0.0030)
- Experiment 23: real=-0.3542 (var=0.0111) vs this=0.0108 (var=0.0081)
- Experiment 24: real=0.3208 (var=0.0038) vs this=0.4994 (var=0.0025)
- Experiment 25: real=0.9200 (var=0.0736) vs this=0.6400 (var=0.2104)
- Experiment 26: real=0.4956 (var=0.0045) vs this=0.5041 (var=0.0040)


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
- pi_14: -0.1487 (var=0.0343)

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
- pi_14: 0.0617 (var=0.0113)

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
- pi_14: 0.0059 (var=0.0003)

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
- pi_14: 0.4938 (var=0.0036)

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
- pi_14: 0.4821 (var=0.0030)

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
- pi_14: 0.1600 (var=0.1344)

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
- pi_14: -0.0001 (var=0.0100)

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
- pi_14: 0.0179 (var=0.0084)

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
- pi_14: 0.5110 (var=0.0027)

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
- pi_14: 0.4606 (var=0.0076)

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
- pi_14: 0.0250 (var=0.0090)

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
- pi_14: 0.5206 (var=0.0047)

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
- pi_14: -0.0279 (var=0.0164)

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
- pi_14: 0.4959 (var=0.0038)

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
- pi_14: -0.0237 (var=0.0095)

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
- pi_14: 0.5260 (var=0.0041)

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
- pi_14: -0.0326 (var=0.0166)

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
- pi_14: 0.6452 (var=0.0049)

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
- pi_14: 0.1217 (var=0.0113)

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
- pi_14: 0.6625 (var=0.0051)

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
- pi_14: 0.9000 (var=0.0900)

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
- pi_14: 0.6856 (var=0.0112)

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
- pi_14: -0.0804 (var=0.0180)

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
- pi_14: 0.3846 (var=0.0102)

### Experiment 25
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
    def subject_score(df):
        compact_sides = []
        compact_choices = []

        for _, row in df.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float)
            b = np.asarray(row["option_b_ratings"], dtype=float)
            a_support = np.flatnonzero(a > b)
            b_support = np.flatnonzero(b > a)

            a_is_compact = a_support.size > 0 and np.all(a_support < 8)
            b_is_compact = b_support.size > 0 and np.all(b_support < 8)
            if a_is_compact == b_is_compact:
                continue

            compact_side = 0 if a_is_compact else 1
            compact_sides.append(compact_side)
            compact_choices.append(int(row["response"]) == compact_side)

        if len(compact_choices) == 0:
            return 0.5

        compact_sides = np.asarray(compact_sides, dtype=int)
        compact_choices = np.asarray(compact_choices, dtype=float)

        side_rates = []
        for side in (0, 1):
            mask = compact_sides == side
            if np.any(mask):
                side_rates.append(float(np.mean(compact_choices[mask])))

        compact_rate = float(np.mean(side_rates))
        dispersed_rate = 1.0 - compact_rate

        if dispersed_rate > 0.5:
            return 1.0
        if dispersed_rate < 0.5:
            return 0.0
        return 0.5

    if len(data) == 0:
        return 0.5

    scores = [subject_score(df) for _, df in data.groupby("subject_id", sort=False)]
    return float(np.mean(scores))
```

**Observed (real) value:** 0.9200 (var=0.0736)
**Other theories' values on this metric (for reference):**
- pi_11: 1.0000 (var=0.0000)
- pi_14: 0.4100 (var=0.2369)
- pi_1: 1.0000 (var=0.0000)
- pi_2: 0.5700 (var=0.2301)
- pi_3: 1.0000 (var=0.0000)
- pi_4: 0.9800 (var=0.0196)
- pi_5: 0.9800 (var=0.0196)
- pi_6: 0.5600 (var=0.2264)
- pi_7: 1.0000 (var=0.0000)
- pi_8: 1.0000 (var=0.0000)
- pi_9: 1.0000 (var=0.0000)
- pi_10: 1.0000 (var=0.0000)
- pi_12: 0.6400 (var=0.2104)
- pi_13: 0.7900 (var=0.1509)

### Experiment 26
**Design**
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    scores = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        a_support = np.flatnonzero(diff > 0)
        b_support = np.flatnonzero(diff < 0)

        # Retain comparisons involving either of the two diffuse chains:
        # features 12:16 have spacing .014 and features 16:20 spacing .020.
        all_support = np.concatenate((a_support, b_support))
        diffuse_trial = np.any((all_support >= 12) & (all_support < 20))
        if not diffuse_trial:
            continue

        # The compact coalitions occupy features 0:8. Determine whether the
        # compact coalition supports A or B, making the score side-invariant.
        a_is_compact = (
            a_support.size > 0
            and np.all((a_support >= 0) & (a_support < 8))
        )
        b_is_compact = (
            b_support.size > 0
            and np.all((b_support >= 0) & (b_support < 8))
        )

        if a_is_compact == b_is_compact:
            continue

        response = int(row["response"])
        chose_compact = (response == 0) if a_is_compact else (response == 1)
        scores.append(float(chose_compact))

    if len(scores) == 0:
        return 0.5
    return float(np.mean(np.asarray(scores, dtype=float)))
```

**Observed (real) value:** 0.4956 (var=0.0045)
**Other theories' values on this metric (for reference):**
- pi_14: 0.4944 (var=0.0099)
- pi_11: 0.1512 (var=0.0035)
- pi_1: 0.1559 (var=0.0102)
- pi_2: 0.5059 (var=0.0033)
- pi_3: 0.1231 (var=0.0025)
- pi_4: 0.3738 (var=0.0032)
- pi_5: 0.3950 (var=0.0033)
- pi_6: 0.4984 (var=0.0036)
- pi_7: 0.4556 (var=0.0044)
- pi_8: 0.6169 (var=0.0076)
- pi_9: 0.1922 (var=0.0028)
- pi_10: 0.2525 (var=0.0091)
- pi_12: 0.5041 (var=0.0040)
- pi_13: 0.4075 (var=0.0084)


## IMPLEMENTATION GUARDRAILS
- The parameters should be within the specified ranges.
- The model's predictions should be valid probability distributions (non-negative and sum to 1).
- When converting logits to probabilities via softmax, always use the numerically stable form: subtract the max before exponentiating (`x = x - np.max(x); p = np.exp(x); p /= p.sum()`). A naive `np.exp(x) / np.sum(np.exp(x))` overflows to Inf/NaN for large logits. Alternatively, use `scipy.special.softmax`.


## PROPOSAL
Propose a novel theory from scratch based on all the information available, faithfully implementing the mechanism family the arbiter prescribed above. Do NOT simply reuse anything generated in past rounds.

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
  "theory": "Capacity-Gated Local Source Individuation proposes that observers do not infer a single globally coherent partition of experts into latent sources. Instead, they first inspect only a capacity-limited set of nearest reliability neighbors. Exact reliability equality activates a categorical common-source tag, and every locally tagged duplicate group receives its own bounded redundancy penalty. A small positive gap releases the equality tag and earns independence credit, but this credit follows a Goldilocks function: it is greatest at a narrowly resolvable gap and declines when the gap becomes broad. Fine metric comparison is used primarily for dyads. At cardinalities of three or more, reliabilities are compressed into a few contextual quality codes that serve as bounded evidence accounts, not as transitive source clusters. Separate occupied codes can contribute distinct accounts, but account coverage saturates quickly, nearest-neighbor credits are sampled rather than accumulated, and both mechanisms are attenuated as coalition load grows. Consequently, diffuse coalitions can outperform compact duplicate-heavy coalitions without allowing a chain of pairwise separations to create an indefinitely increasing independence advantage. Duplicate penalties remain group-local and additive across tagged groups, preserving sensitivity to single- versus multiple-duplicate structures. Coalition values are finally compared through opponent-relative normalization and a bounded contrast transform. This separates structural valuation from response sensitivity: subjects can represent a structural difference while nevertheless producing near-chance choices, and stable heterogeneity in duplicate tolerance, gap resolution, reliability compression, account capacity, and response sensitivity can affect mean and threshold-based measures differently. Cue salience is exchangeable across identities, and the theory contains no population-level recency, onset, adjacency, closure, or display-position gradient.",
  "predict": "def predict(parameters, state, history) -> np.ndarray:\n    stim = np.asarray(state, dtype=np.float64)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"Capacity-Gated Local Source Individuation expects shape \"\n            f\"(2, n_features); got {stim.shape}.\"\n        )\n\n    a, b = stim[0], stim[1]\n    n_features = int(stim.shape[1])\n\n    v = np.asarray(parameters[\"validities\"], dtype=np.float64)\n    if v.ndim != 1 or v.size != n_features:\n        raise ValueError(\n            f\"validities length {v.size} != n_features {n_features}.\"\n        )\n    v = np.clip(v, 0.500001, 0.999999)\n\n    raw_salience = np.asarray(\n        parameters[\"cue_salience_profile\"], dtype=np.float64\n    )\n    if raw_salience.ndim != 1 or raw_salience.size != n_features:\n        raise ValueError(\n            f\"cue_salience_profile length {raw_salience.size} != \"\n            f\"n_features {n_features}.\"\n        )\n\n    validity_compression = float(parameters[\"validity_compression\"])\n    validity_reliance = float(parameters[\"validity_reliance\"])\n    salience_strength = float(parameters[\"salience_strength\"])\n    equality_resolution = float(parameters[\"equality_resolution\"])\n    dyad_gap_resolution = float(parameters[\"dyad_gap_resolution\"])\n    dyad_broad_scale = float(parameters[\"dyad_broad_scale\"])\n    dyad_independence_gain = float(parameters[\"dyad_independence_gain\"])\n    dyad_duplicate_penalty = float(parameters[\"dyad_duplicate_penalty\"])\n    code_resolution = int(parameters[\"code_resolution\"])\n    account_capacity = float(parameters[\"account_capacity\"])\n    account_normalization = float(parameters[\"account_normalization\"])\n    coverage_gain = float(parameters[\"coverage_gain\"])\n    large_coalition_gate = float(parameters[\"large_coalition_gate\"])\n    local_link_gain = float(parameters[\"local_link_gain\"])\n    local_link_capacity = int(parameters[\"local_link_capacity\"])\n    coalition_duplicate_penalty = float(\n        parameters[\"coalition_duplicate_penalty\"]\n    )\n    duplicate_group_saturation = float(\n        parameters[\"duplicate_group_saturation\"]\n    )\n    count_curvature = float(parameters[\"count_curvature\"])\n    contrast_normalization = float(parameters[\"contrast_normalization\"])\n    contrast_scale = float(parameters[\"contrast_scale\"])\n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n\n    # Communicated reliability remains consequential, but metric differences\n    # are compressed before entering evidence quality.\n    log_odds = np.log(v / (1.0 - v))\n    compressed = np.power(np.maximum(log_odds, 1e-12), validity_compression)\n    compressed /= max(float(np.mean(compressed)), 1e-12)\n    reliability_quality = (\n        (1.0 - validity_reliance)\n        + validity_reliance * compressed\n    )\n\n    # Subject-specific cue salience is centered and exchangeable. There is no\n    # systematic relation between salience and serial or display position.\n    centered_salience = raw_salience - float(np.mean(raw_salience))\n    attention = np.exp(\n        np.clip(salience_strength * centered_salience, -10.0, 10.0)\n    )\n    attention /= max(float(np.mean(attention)), 1e-12)\n\n    cue_quality = reliability_quality * attention\n    cue_quality /= max(float(np.mean(cue_quality)), 1e-12)\n\n    # Contextual reliability codes are coarse quality labels used only when\n    # coalition cardinality exceeds dyadic comparison capacity. They are not\n    # interpreted as globally coherent or transitive latent sources.\n    unique_v, inverse_levels = np.unique(v, return_inverse=True)\n    n_levels = int(unique_v.size)\n    actual_codes = max(1, min(code_resolution, n_levels))\n    if n_levels == 1:\n        level_codes = np.zeros(1, dtype=np.int64)\n    else:\n        level_codes = np.floor(\n            np.arange(n_levels, dtype=np.float64)\n            * float(actual_codes)\n            / float(n_levels)\n        ).astype(np.int64)\n        level_codes = np.minimum(level_codes, actual_codes - 1)\n    cue_codes = level_codes[inverse_levels]\n\n    difference = a - b\n    a_support = np.flatnonzero(difference > 0.0)\n    b_support = np.flatnonzero(difference < 0.0)\n\n    if a_support.size == 0 and b_support.size == 0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n\n    def goldilocks_credit(gap):\n        gap = float(max(gap, 0.0))\n        if gap <= equality_resolution:\n            return 0.0\n\n        # Release from equality rises rapidly around the observer's resolution\n        # threshold. A separate broad-gap decay prevents metric distance from\n        # becoming a monotonic independence signal.\n        release = 1.0 - np.exp(\n            -gap / max(dyad_gap_resolution, 1e-12)\n        )\n        broad_decay = np.exp(\n            -gap / max(dyad_broad_scale, 1e-12)\n        )\n        return float(np.clip(release * broad_decay, 0.0, 1.0))\n\n    def duplicate_load(indices):\n        idx = np.asarray(indices, dtype=np.int64)\n        if idx.size <= 1:\n            return 0.0\n\n        # Exact tags are evaluated separately by local validity group. Their\n        # penalties add across groups but saturate within each group.\n        levels = inverse_levels[idx]\n        counts = np.bincount(levels, minlength=n_levels).astype(np.float64)\n        load = 0.0\n        for count in counts[counts > 1.0]:\n            excess = float(count - 1.0)\n            load += excess / (\n                1.0 + duplicate_group_saturation * excess\n            )\n        return float(load)\n\n    def coalition_value(indices):\n        idx = np.asarray(indices, dtype=np.int64)\n        n = int(idx.size)\n        if n == 0:\n            return 0.0\n        if n == 1:\n            return float(cue_quality[idx[0]])\n\n        qualities = cue_quality[idx]\n        coalition_v = v[idx]\n\n        if n == 2:\n            # Dyads permit fine metric comparison. Equality activates a strong\n            # source tag; a narrow nonzero gap receives independence credit;\n            # broad gaps lose that credit.\n            gap = abs(float(coalition_v[0] - coalition_v[1]))\n            base = float(np.sum(qualities)) / (\n                2.0 ** account_normalization\n            )\n\n            if gap <= equality_resolution:\n                structural_factor = np.exp(-dyad_duplicate_penalty)\n            else:\n                credit = goldilocks_credit(gap)\n                structural_factor = 1.0 + dyad_independence_gain * credit\n\n            return float(\n                np.power(max(base, 1e-12), count_curvature)\n                * structural_factor\n            )\n\n        # At n >= 3, each occupied contextual code receives one bounded\n        # evidence account. Extra members increase an account only partially.\n        codes = cue_codes[idx]\n        occupied_codes = np.unique(codes)\n        account_values = []\n        for code in occupied_codes:\n            members = np.flatnonzero(codes == code)\n            m = int(members.size)\n            mean_quality = float(np.mean(qualities[members]))\n            if m == 1:\n                represented_mass = 1.0\n            else:\n                represented_mass = 1.0 + account_capacity * (\n                    1.0\n                    - np.exp(\n                        -float(m - 1) / max(account_capacity, 1e-12)\n                    )\n                )\n            account_values.append(mean_quality * represented_mass)\n\n        coverage = int(len(account_values))\n        account_total = float(np.sum(account_values))\n        normalized_accounts = account_total / (\n            float(max(coverage, 1)) ** account_normalization\n        )\n\n        # Only the first additional reliability region receives substantial\n        # coverage credit. The credit is progressively gated at larger loads,\n        # so a chain cannot accumulate a global-span advantage.\n        bounded_coverage = float(min(max(coverage - 1, 0), 1))\n        load_gate = 1.0 / (\n            1.0 + large_coalition_gate * float(max(n - 3, 0))\n        )\n        coverage_factor = 1.0 + (\n            coverage_gain * bounded_coverage * load_gate\n        )\n\n        # Inspect only a few nearest positive reliability links. Link credits\n        # are averaged, not summed, and are further capacity-gated with load.\n        order = np.argsort(coalition_v)\n        sorted_v = coalition_v[order]\n        adjacent_gaps = np.diff(sorted_v)\n        positive_gaps = adjacent_gaps[adjacent_gaps > equality_resolution]\n        if positive_gaps.size == 0:\n            local_credit = 0.0\n        else:\n            nearest = np.sort(positive_gaps)[:local_link_capacity]\n            credits = np.asarray(\n                [goldilocks_credit(g) for g in nearest],\n                dtype=np.float64,\n            )\n            local_credit = float(np.mean(credits)) if credits.size else 0.0\n\n        local_factor = 1.0 + local_link_gain * local_credit * load_gate\n\n        # Duplicate penalties accumulate by exact tagged group rather than by\n        # a transitive similarity graph. Their cardinality effect is bounded.\n        dup_load = duplicate_load(idx)\n        duplicate_factor = np.exp(\n            -coalition_duplicate_penalty * dup_load\n        )\n\n        represented = np.power(\n            max(normalized_accounts, 1e-12), count_curvature\n        )\n        return float(\n            represented\n            * coverage_factor\n            * local_factor\n            * duplicate_factor\n        )\n\n    value_a = coalition_value(a_support)\n    value_b = coalition_value(b_support)\n\n    # Opponent-relative normalization makes structural differences depend on\n    # the strength of the competing coalitions rather than on an unbounded\n    # absolute evidence scale.\n    comparison_scale = (\n        1.0 + 0.5 * (abs(value_a) + abs(value_b))\n    ) ** contrast_normalization\n    normalized_difference = (value_a - value_b) / max(\n        comparison_scale, 1e-12\n    )\n\n    # Structural valuation is bounded before response sensitivity is applied.\n    # Thus beta can alter choice consistency and threshold metrics without\n    # changing the represented ordering of coalitions.\n    structural_contrast = contrast_scale * np.tanh(\n        normalized_difference / max(contrast_scale, 1e-12)\n    )\n\n    logits = np.array(\n        [0.5 * beta * structural_contrast,\n         -0.5 * beta * structural_contrast],\n        dtype=np.float64,\n    )\n    logits -= np.max(logits)\n    probs = np.exp(logits)\n    probs /= probs.sum()\n\n    probs = (1.0 - epsilon) * probs + epsilon * 0.5\n    probs = np.maximum(probs, 0.0)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n    return probs / total\n",
  "policy": "def policy(probs) -> int:\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.maximum(probs, 0.0)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))\n    else:\n        probs /= total\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "validities": "validities",
    "cue_salience_profile": "[(-1.0, 1.0)] * n_features",
    "validity_compression": "[0.16, 0.34]",
    "validity_reliance": "[0.035, 0.11]",
    "salience_strength": "[0.04, 0.18]",
    "equality_resolution": "[0.0000001, 0.00001]",
    "dyad_gap_resolution": "[0.0025, 0.0055]",
    "dyad_broad_scale": "[0.0045, 0.0090]",
    "dyad_independence_gain": "[0.45, 1.05]",
    "dyad_duplicate_penalty": "[0.65, 1.35]",
    "code_resolution": "{2, 3}",
    "account_capacity": "[0.85, 1.65]",
    "account_normalization": "[0.50, 0.72]",
    "coverage_gain": "[0.10, 0.42]",
    "large_coalition_gate": "[0.85, 1.80]",
    "local_link_gain": "[0.08, 0.30]",
    "local_link_capacity": "{1, 2}",
    "coalition_duplicate_penalty": "[0.18, 0.48]",
    "duplicate_group_saturation": "[0.45, 1.05]",
    "count_curvature": "[0.72, 0.90]",
    "contrast_normalization": "[0.35, 0.70]",
    "contrast_scale": "[0.55, 1.15]",
    "beta": "[0.85, 2.20]",
    "epsilon": "[0.025, 0.085]"
  },
  "rationale": "The model directly implements the arbiter's requested cardinality gate rather than extending a dyadic distance function to entire coalitions. Dyads use an equality tag plus a nonmonotonic gap function. This gives exact duplicates a strong discount, rewards narrowly distinguishable pairs, and withdraws that reward for broad pairs, addressing the exact-avoidance and narrow-over-broad patterns in Experiments 10, 24, and 26. At three or more cues, evidence is represented through coarse contextual accounts. Coverage is capped after one additional account, nearest-neighbor independence credit is averaged over at most one or two links, and both are attenuated above three cues. Chains therefore cannot collect a reward from every successive link, which targets the near-chance chain comparisons in Experiments 18 and 20 while retaining limited sensitivity to locally distributed evidence. Exact duplicate groups are penalized separately and their penalties add across groups, supporting duplicate avoidance and the single- versus multiple-duplicate contrast in Experiments 12, 16, and 18. The account mechanism can still favor coalitions that cover distinct reliability regions over compact duplicate-heavy coalitions in Experiments 1 and 14–16, but it does not impose a global range, span, coherence, or strongest-stratum term. The weak, continuous transition into account coding avoids a generic triadic bonus, helping preserve the near-zero two-to-three release in Experiment 15 and the nonmonotonic cardinality profile in Experiments 17 and 19. Finally, opponent-relative normalization and bounded structural contrast reduce excessive coalition differences in the balanced designs, while the independently heterogeneous beta parameter permits population means and supermajority subject thresholds to diverge. Exchangeable salience allows stable individual variation without predicting the unsupported positional gradients tested in Experiments 4, 5, 7–9, and 25."
}
```

## Usage

```json
{
  "input_tokens": 56146,
  "output_tokens": 6471
}
```
