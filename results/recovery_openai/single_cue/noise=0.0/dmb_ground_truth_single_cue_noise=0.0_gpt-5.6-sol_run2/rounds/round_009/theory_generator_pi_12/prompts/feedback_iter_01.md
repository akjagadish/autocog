# feedback_iter_01

## System Prompt

You are a renowned cognitive scientist critiquing a freshly proposed candidate theory and model in the Decision Making (Binary Features) domain.

The candidate has been simulated on every previously run experiment. For each experiment you are shown the design, the metric, the value the metric takes on real data, and the value it takes on the candidate's simulated data.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the feedback is to SURFACE theories that are EXPERIMENT-INVARIANT: that is,theories that explain data across multiple experiments. 
Your task is to determine whether the candidate captures the human/real behavior well enough across these experiments. Return a verdict:
  * "continue"   — the candidate is good enough; carry on.
  * "regenerate" — the candidate fails to capture the empirical pattern; the proposing agent must produce a new candidate, taking your rationale into account.

Justify the verdict with a concrete diagnosis (which experiments fail, in what direction, what mechanism is likely missing or miscalibrated).

## SCOPE OF YOUR CRITIQUE — STAY INSIDE THE ARBITER'S MECHANISM FAMILY
When an "## ARBITER RECOMMENDATION" block is present below, the proposer was explicitly instructed to implement the mechanism family the arbiter prescribed. Your job is to grade FIT QUALITY *within that prescribed family*, not to relitigate which family should be used — that is the arbiter's call, made one level above this loop.

Concretely:
  * If the candidate misses the data, you may push for MINOR ADJUSTMENTS that keep the prescribed mechanism intact: tightening / widening parameter ranges, adding a temperature, swapping a normalization scheme, fixing a softmax / distance metric, re-balancing attention weights, fixing a learning-rate sign, correcting a bug in the gating or recurrence, etc.
  * You MUST NOT recommend switching to a different mechanism family. Such a switch is the arbiter's prerogative; recommending it here will mislead the proposer into oscillating between families across iterations.
  * Also grade FAITHFULNESS to the recommendation explicitly: if the candidate has clearly drifted into a different family than the one prescribed, say so in the rationale and ask for a return to the prescribed family — again, with minor adjustments, not a re-design.

## ACCEPT GATE — HOW THE LOOP DECIDES WHAT TO BUILD ON NEXT
This propose-loop has a programmatic accept gate. After every iteration the candidate's `aggregate_loss` is compared against the running-best loss (`accepted_loss`):
  * `loss < accepted_loss` → ACCEPTED. The candidate becomes the new running-best base; the next iteration's proposer will build on THIS candidate.
  * `loss >= accepted_loss` → REJECTED. The base is unchanged; the next iteration's proposer will build on the SAME `accepted` candidate again, with your new feedback on top. Rejected candidates are discarded — the loop guarantees the base never regresses, so you do NOT need to ask the proposer to "revert" anything; that already happens for free.

Two consequences for your verdict:
  * If the candidate you are grading was REJECTED by the gate, returning `"continue"` is silently downgraded to `"regenerate"` (returning a worse candidate would defeat the gate). Spend your rationale on a NEW direction the proposer should try on top of the unchanged accepted base, not on defending the rejected attempt.
  * If the candidate was ACCEPTED, you can return `"continue"` to stop the loop and ship this candidate, or `"regenerate"` to keep tuning further.

## LEARN FROM YOUR OWN PAST ADVICE
When a "## YOUR PRIOR CRITIQUES" block is present below, each prior iteration ends with an "Outcome of your advice" line that says whether the next candidate the proposer produced was ACCEPTED (your advice helped — its loss strictly beat the running best) or REJECTED (your advice didn't help — the proposer discarded the result and reset to the previous accepted base). This is the loop's ground-truth signal on whether *your own previous critique was good*. Use it explicitly:
  * If a previous piece of advice was ACCEPTED, it is OK to repeat / extend it. Reinforce in the same direction.
  * If a previous piece of advice was REJECTED, do NOT repeat the same recommendation; in your new rationale, briefly acknowledge that the previous push in that direction was rejected by the gate and try a different in-family knob (or a smaller step in the same direction) instead.
  * If you find yourself oscillating (e.g. iter 1 said "increase α", iter 2 said "decrease α", iter 3 about to say "increase α" again), STOP and recommend a value between the two flanking iterations instead.
  * The "## LOSS TRAJECTORY" block at the top of the user prompt summarises the same information at the loop level — consult it before issuing a new regenerate-with-direction recommendation.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## ARBITER RECOMMENDATION (mechanism family the proposer was told to implement)
The arbiter labelled this round's two theories in its recommendation as follows:
- THEORY 1 = `pi_11`
- THEORY 2 = `pi_9`
- The recommendation below acts on THEORY 2 (= `pi_9`).

Replace pi_9 with an Opponent-Relative Reliability-Coverage theory. The new theory should deny that pairwise metric proximity in reliability space directly induces dependence. Instead, on each trial people should compress experts into a few contextual reliability strata and compare the two opposing coalitions in terms of coverage, concentration, and balance across those strata. Limited-capacity sampling should allow approximately one or a few representatives from each stratum to enter the decision, so exact repetition and heavy within-stratum concentration can show diminishing returns without implying a smooth 0.003-versus-0.0058 distance slope or a chain advantage. Crucially, redundancy should be opponent-relative rather than an intrinsic penalty attached independently to each coalition: whether repeated experts are discounted should depend on the rival coalition's stratum coverage and cardinality. This can yield equality avoidance in Experiments 12 and 18 while preserving chance-level chain comparisons in Experiments 2 and 20. Add a reliability-balance or coverage objective under which a coalition spanning several moderate strata can defeat a coalition dominated by the uniquely highest-validity cue, addressing Experiments 3 and 4 without introducing serial-position effects. Coalition size should enter through a capacity-limited sampling or occupancy process rather than one global power law, permitting the nonlinear dyad/triad/tetrad profile in Experiments 17 and 19. Subject heterogeneity should primarily concern sampling capacity, stratum resolution, and preference for balanced versus concentrated evidence; cue identity salience may remain exchangeable and centered. The new theory should make the following qualitative commitments: near-zero narrow-to-wide slope in Experiment 1, chance-level chain preference in Experiments 2 and 20, no systematic onset/recency/closure effect, strong but context-sensitive penalties for repeated evidence, distributed-coalition preferences where reliability coverage improves, and weaker broad metric-gap effects than either current implementation predicts. It must be tested with shared parameter ranges across experiments and should not recover the rejected pi_9 mechanism through a hidden sharply localized similarity kernel.


## CANDIDATE THEORY
Opponent-Relative Stratified Sampling Theory proposes that people do not infer dependence from metric distances between expert validities. Instead, they compress the experiment's ordered reliability levels into a small contextual vocabulary of reliability strata, with subject-specific partition offsets that prevent fixed ordinal boundaries from creating population-level gap effects. On each trial, the two opposing coalitions are represented jointly by their stratum coverage, balance across occupied strata, and concentration within repeated reliability levels. Evidence enters a limited-capacity occupancy process, but coalition strength is based primarily on average sampled quality rather than summed coalition mass. A small bounded occupancy term preserves nonlinear effects of sampling capacity without allowing larger coalitions to dominate automatically. Redundancy remains opponent-relative: repetition is discounted when the rival is less concentrated, has broader coverage, or has a contextual cardinality advantage. Direct concentration comparison is available for dyads and triads, where exact repetition is readily represented; larger coalitions depend mainly on relative coverage and occupancy because exact member-level repetition exceeds focal capacity. Weak ordinal reliability weighting preserves sensitivity to instructed validity without creating metric-gap slopes. Coverage and balance can favor distributed moderate evidence over concentrated high-validity evidence. Subject heterogeneity concerns stratum resolution, partition offset, sampling capacity, balance preference, and opponent-relative redundancy sensitivity; exchangeable cue salience introduces individual differences without population-level onset, recency, closure, or adjacency effects.

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
    stratum_partition_offset = float(parameters["stratum_partition_offset"])
    sampling_capacity = int(parameters["sampling_capacity"])
    global_capacity = float(parameters["global_capacity"])
    ordinal_reliance = float(parameters["ordinal_reliance"])
    salience_strength = float(parameters["salience_strength"])
    coverage_preference = float(parameters["coverage_preference"])
    balance_preference = float(parameters["balance_preference"])
    redundancy_sensitivity = float(parameters["redundancy_sensitivity"])
    rival_size_weight = float(parameters["rival_size_weight"])
    context_curvature = float(parameters["context_curvature"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    v = np.clip(validities, 0.500001, 0.999999)

    # Contextual strata are formed from the ordering of distinct instructed
    # reliability levels. A subject-specific offset shifts ordinal boundaries,
    # so boundary crossings average out across people without introducing a
    # metric bandwidth or a pairwise similarity relation.
    unique_v, inverse_levels = np.unique(v, return_inverse=True)
    n_levels = int(unique_v.size)
    actual_strata = max(1, min(stratum_resolution, n_levels))

    if n_levels == 1:
        level_strata = np.zeros(1, dtype=np.int64)
        ordinal_contrast = np.zeros(1, dtype=np.float64)
    else:
        shifted_level_rank = (
            np.arange(n_levels, dtype=np.float64)
            + stratum_partition_offset
        )
        level_strata = np.floor(
            shifted_level_rank
            * float(actual_strata)
            / float(n_levels)
        ).astype(np.int64)
        level_strata = np.clip(level_strata, 0, actual_strata - 1)
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
                "duplicate_load": 0.0,
                "sampled_signal": 0.0,
            }

        strata = cue_strata[idx]
        counts = np.bincount(strata, minlength=actual_strata).astype(np.float64)
        occupied = counts > 0.0
        coverage = int(np.sum(occupied))

        # Each stratum has a finite set of representational slots. Expected
        # slot occupancy supplies diminishing returns within a stratum.
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

        # Normalize by expected sampled representatives so raw coalition mass
        # cannot dominate. A small bounded occupancy factor preserves limited
        # nonlinear cardinality effects without reinstating a global power law.
        expected_total = float(np.sum(expected_slots))
        average_sampled_quality = stratum_signal / max(expected_total, 1e-12)
        bounded_occupancy = 1.0 - np.exp(
            -expected_total / max(global_capacity, 1e-12)
        )
        sampled_signal = average_sampled_quality * (
            1.0 + 0.12 * bounded_occupancy
        )

        if coverage <= 1:
            balance = 0.0
        else:
            proportions = counts[occupied] / float(n)
            entropy = -float(np.sum(proportions * np.log(proportions)))
            balance = entropy / max(float(np.log(coverage)), 1e-12)

        # Exact repetition is represented as an opponent-comparison variable,
        # not subtracted intrinsically. Using excess count rather than a
        # cardinality-normalized fraction keeps the same duplicate pair stable
        # when a matched third representative is added to both coalitions.
        level_counts = np.bincount(
            inverse_levels[idx], minlength=n_levels
        ).astype(np.float64)
        duplicate_load = float(
            np.sum(np.maximum(level_counts - 1.0, 0.0))
        )

        return {
            "n": n,
            "coverage": coverage,
            "balance": balance,
            "duplicate_load": duplicate_load,
            "sampled_signal": sampled_signal,
        }

    sa = summarize(a_support)
    sb = summarize(b_support)

    # Coverage is evaluated relative to the contextual strata represented by
    # either opposing coalition on this trial.
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

    def contextual_vulnerability(focal, rival):
        if focal["n"] == 0 or focal["duplicate_load"] <= 0.0:
            return 0.0

        coverage_advantage = float(
            rival["coverage"] - focal["coverage"]
        ) / float(max(union_coverage, 1))
        size_advantage = float(rival["n"] - focal["n"]) / float(
            max(rival["n"], focal["n"], 1)
        )

        # For small coalitions, exact concentration can be compared directly
        # with the opponent even when cardinality and coarse coverage tie.
        # Beyond triads, limited capacity shifts comparison to stratum-level
        # coverage and occupancy, preserving chance-level chain comparisons.
        concentration_advantage = max(
            0.0,
            float(focal["duplicate_load"] - rival["duplicate_load"]),
        )
        if max(focal["n"], rival["n"]) > 3:
            concentration_advantage = 0.0

        opponent_context = max(
            0.0,
            concentration_advantage
            + coverage_advantage
            + rival_size_weight * size_advantage,
        )
        opponent_context = opponent_context ** context_curvature
        return float(opponent_context)

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
- stratum_partition_offset: [-0.49, 0.49]
- sampling_capacity: {1, 2, 3}
- global_capacity: [2.2, 4.8]
- ordinal_reliance: [0.012, 0.045]
- salience_strength: [0.08, 0.24]
- coverage_preference: [0.28, 0.52]
- balance_preference: [0.16, 0.38]
- redundancy_sensitivity: [1.00, 1.80]
- rival_size_weight: [0.20, 0.55]
- context_curvature: [0.70, 1.30]
- beta: [1.25, 1.70]
- epsilon: [0.03, 0.08]

`rationale`:
This is a minimal mechanism-preserving revision of the accepted opponent-relative model. First, summed occupied-slot evidence is replaced by average sampled quality with only a small bounded occupancy increment. This directly targets excessive coalition-mass effects in Experiments 1 and 6 and should reduce the spurious triad decrement in Experiment 15 while retaining sampling-capacity nonlinearities. Second, vulnerability now includes the focal coalition's excess repetition relative to its opponent, repairing the missing equality avoidance in Experiment 10 even when cardinality and coarse coverage tie. Excess duplicate count is not normalized by coalition size, so adding a matched third cue does not automatically weaken the duplicate contrast. Direct member-level concentration comparison is limited to dyads and triads; larger coalitions continue to be judged through opponent-relative stratum coverage and occupancy, protecting the near-chance chain results in Experiments 18 and 20 and retaining the successful Experiment 16 mechanism. Third, a subject-specific ordinal partition offset removes fixed rank-bin boundary artifacts that likely produced the excessive Experiment 11 gap contrast. Finally, coverage and balance ranges are strengthened while ordinal reliability is increased only modestly. This should amplify distributed-coalition preferences in Experiments 3 and 12–14 without restoring a metric similarity kernel, positional mechanism, or dominant highest-validity cue.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.1838 -> ACCEPTED
- iter 2 (current candidate you are grading): loss=0.1842 -> REJECTED
Running-best (last accepted) base: iter 1 at loss=0.1838.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
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
**Candidate trajectory (this loop):**
  - iter 1: -0.5933 (var=0.0154) (Δ vs real -0.4717)
  - iter 2 (current): -0.2988 (var=0.0227) (Δ vs real -0.1771)
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
**Candidate trajectory (this loop):**
  - iter 1: 0.0029 (var=0.0111) (Δ vs real +0.2671)
  - iter 2 (current): 0.0654 (var=0.0153) (Δ vs real +0.3296)
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
**Candidate trajectory (this loop):**
  - iter 1: 0.0220 (var=0.0006) (Δ vs real +0.0630)
  - iter 2 (current): 0.0172 (var=0.0004) (Δ vs real +0.0582)
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
**Candidate trajectory (this loop):**
  - iter 1: 0.5018 (var=0.0050) (Δ vs real -0.0069)
  - iter 2 (current): 0.4967 (var=0.0068) (Δ vs real -0.0119)
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
**Candidate trajectory (this loop):**
  - iter 1: 0.5015 (var=0.0033) (Δ vs real -0.0006)
  - iter 2 (current): 0.5035 (var=0.0028) (Δ vs real +0.0015)
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
**Candidate trajectory (this loop):**
  - iter 1: 0.9000 (var=0.0900) (Δ vs real +0.4600)
  - iter 2 (current): 0.0000 (var=0.0000) (Δ vs real -0.4400)
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
**Candidate trajectory (this loop):**
  - iter 1: -0.0054 (var=0.0112) (Δ vs real -0.0272)
  - iter 2 (current): 0.0197 (var=0.0128) (Δ vs real -0.0022)
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
**Candidate trajectory (this loop):**
  - iter 1: 0.0221 (var=0.0109) (Δ vs real +0.0104)
  - iter 2 (current): 0.0229 (var=0.0125) (Δ vs real +0.0113)
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
**Candidate trajectory (this loop):**
  - iter 1: 0.5098 (var=0.0024) (Δ vs real +0.0140)
  - iter 2 (current): 0.4929 (var=0.0027) (Δ vs real -0.0029)
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
**Candidate trajectory (this loop):**
  - iter 1: 0.5113 (var=0.0040) (Δ vs real +0.3390)
  - iter 2 (current): 0.2777 (var=0.0034) (Δ vs real +0.1054)
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
**Candidate trajectory (this loop):**
  - iter 1: 0.1600 (var=0.0055) (Δ vs real +0.1283)
  - iter 2 (current): 0.2500 (var=0.0139) (Δ vs real +0.2183)
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
**Candidate trajectory (this loop):**
  - iter 1: 0.6033 (var=0.0162) (Δ vs real -0.0658)
  - iter 2 (current): 0.8421 (var=0.0020) (Δ vs real +0.1729)
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
**Candidate trajectory (this loop):**
  - iter 1: 0.1950 (var=0.0586) (Δ vs real -0.1292)
  - iter 2 (current): 0.6858 (var=0.0063) (Δ vs real +0.3617)
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
**Candidate trajectory (this loop):**
  - iter 1: 0.5425 (var=0.0065) (Δ vs real -0.0444)
  - iter 2 (current): 0.6941 (var=0.0030) (Δ vs real +0.1072)
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
**Candidate trajectory (this loop):**
  - iter 1: -0.1996 (var=0.0282) (Δ vs real -0.2137)
  - iter 2 (current): -0.0104 (var=0.0065) (Δ vs real -0.0246)
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
**Candidate trajectory (this loop):**
  - iter 1: 0.7204 (var=0.0249) (Δ vs real +0.0425)
  - iter 2 (current): 0.6350 (var=0.0193) (Δ vs real -0.0429)
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
**Candidate trajectory (this loop):**
  - iter 1: -0.3010 (var=0.0157) (Δ vs real +0.0475)
  - iter 2 (current): -0.1910 (var=0.0089) (Δ vs real +0.1576)
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
**Candidate trajectory (this loop):**
  - iter 1: 0.4904 (var=0.0037) (Δ vs real -0.0079)
  - iter 2 (current): 0.4717 (var=0.0039) (Δ vs real -0.0267)
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
**Candidate trajectory (this loop):**
  - iter 1: -0.0500 (var=0.0050) (Δ vs real -0.0217)
  - iter 2 (current): -0.0133 (var=0.0041) (Δ vs real +0.0150)
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
**Candidate trajectory (this loop):**
  - iter 1: 0.4704 (var=0.0056) (Δ vs real -0.0338)
  - iter 2 (current): 0.5175 (var=0.0132) (Δ vs real +0.0133)
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

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1 (most recent)
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed Opponent-Relative Reliability-Coverage family and successfully removes the major chain and serial-position artifacts: Experiments 4, 5, 7, 8, 9, 18, 19, and 20 are close to the human values, and the nonlinear cardinality profile in Experiment 17 is captured reasonably well. It also produces distributed-coalition preferences in Experiments 12–14, though too weakly. However, several large, systematic failures show that the current occupancy and contextual-redundancy implementation is not yet experiment-invariant. It greatly overvalues coalition mass in Experiments 1 (-0.593 versus -0.122) and 6 (0.900 versus 0.440), creates a large spurious triad decrement in Experiment 15 (-0.200 versus 0.014), and gets the diagnostic pattern in Experiment 3 in the wrong direction (0.022 versus -0.041). Most importantly, it produces virtually no equality avoidance in Experiment 10 (0.511 versus 0.172). The current vulnerability gate only discounts repetition when the rival has greater coverage or cardinality, so it is inactive when equal-sized coalitions have similar coarse-stratum coverage even though one coalition is exactly repeated and the other is not. Experiment 11 also shows an excessive broad-gap effect (0.160 versus 0.032), apparently caused by deterministic rank-bin boundaries rather than metric distance. Experiment 2 remains at chance (0.003) rather than reproducing the observed anti-highest-cue contrast (-0.264), while the distributed effects in Experiments 12–14 and 13 in particular are underpowered.
Rationale: Keep the prescribed mechanism family, but revise three implementation details. First, make redundancy genuinely opponent-relative by comparing the two coalitions' concentration or duplicate fractions directly: a focal duplicate coalition should become vulnerable when its rival is less concentrated, even when coverage and cardinality are tied. This should repair Experiment 10 without introducing an intrinsic duplicate penalty; retain coverage and size as additional contextual moderators rather than necessary activation conditions. Second, reduce the dominance of raw sampled coalition mass. The current summed occupied-slot signal makes extra representatives too valuable despite saturation. Use a capacity-limited average sampled quality, or normalize occupied signal by expected sampled representatives and let bounded coverage/balance terms carry the structural advantage. This should pull Experiments 1 and 6 toward chance and remove the artificial triad decrement in Experiment 15 while preserving occupancy-based nonlinearities for Experiment 17. Coverage and balance weights can then be modestly strengthened to recover the underpredicted distributed preferences in Experiments 12–14 and the sign in Experiment 3. Third, eliminate deterministic ordinal-bin boundary artifacts by averaging occupancy over exchangeable random offsets or subject-specific ordinal partitions. Exact equalities must still share a stratum, but adjacent rank pairs should not acquire a population-level high-gap contrast merely because they cross a fixed bin edge; this targets Experiment 11 while remaining free of any metric-distance kernel. After reducing coalition-mass dominance, only a modest increase in ordinal reliability should be considered to soften the excessive anti-top-cue result in Experiment 1; increasing reliability alone would likely worsen Experiment 6. The next fit should preserve the already successful near-chance chain results in Experiments 18 and 20 and the absence of serial-position effects.

**Outcome of your advice:** iter 1 candidate loss=0.1838 -> CURRENT candidate loss=0.1842 -> the gate marked it REJECTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate remains faithful to the prescribed opponent-relative reliability-coverage family, but it was rejected by the accept gate and should not replace the iteration-1 base. The revision produced useful local improvements: Experiment 1 moved from -0.593 to -0.299, Experiment 10 from 0.511 to 0.278, and Experiment 15 from -0.200 to -0.010. It also preserved the absence of positional effects and remained close on Experiments 4, 5, 7-9, 18-20. However, the combined revision overcorrected elsewhere. Average-quality normalization flipped Experiment 6 from near-universal preference for the three-cue coalition (0.900) to universal rejection (0.000), eliminating the observed between-subject heterogeneity. Strengthened coverage and balance grossly overshot the distributed-coalition effects in Experiments 12-14, especially Experiment 13 (0.686 versus 0.324), and degraded the cardinality profile in Experiment 17 (-0.191 versus -0.349). The partition-offset change also worsened rather than reduced the broad-gap artifact in Experiment 11 (0.250 versus 0.032). Experiment 3 still has the wrong sign, and Experiment 2 remains empirically wrong, although its near-chance prediction is faithful to the arbiter's qualitative commitment.",
  "verdict": "regenerate",
  "rationale": "Because the previous combined recommendation was rejected, do not repeat the wholesale shift to average sampled quality, stronger coverage/balance, and partition offsets. Starting from the unchanged iteration-1 base, make a more surgical revision. Retain the promising direct opponent-relative duplicate-load comparison for equal-sized dyads and triads, since it substantially improved Experiment 10, but use a bounded or saturating concentration contrast so it does not dominate other configurations. For coalition mass, use an intermediate occupancy normalization rather than either summed slots or fully averaged quality\u2014for example, divide occupied signal by expected_total raised to an exponent strictly between 0 and 1. This should place Experiment 6 between the two extreme predictions while retaining subject variation and avoiding the deterioration in Experiment 17. Keep coverage and balance near the iteration-1 ranges rather than strengthening them; the original underprediction in Experiments 12-14 was much smaller than the current overshoot. To address Experiment 11, do not reuse the rejected partition-offset intervention. Instead, suppress generic stratum-coverage bonuses in equal-cardinality dyad-versus-dyad comparisons when the coalitions have matched duplicate status, while allowing the explicit opponent-relative equality contrast to operate when only one coalition contains exact repetition. This separates Experiment 10's equality avoidance from Experiment 11's unwanted ordinal boundary effect without introducing a metric-distance kernel. Preserve the successful exchangeability and large-coalition concentration cutoff that keep Experiments 4-9, 18, and 20 near chance and free of serial-position or chain effects."
}
```

## Usage

```json
{
  "input_tokens": 37931,
  "output_tokens": 1583
}
```
