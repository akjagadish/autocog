# feedback_iter_09

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
- THEORY 1 = `pi_8`
- THEORY 2 = `pi_6`
- The recommendation below acts on THEORY 2 (= `pi_6`).

Replace pi_6 with a new theory of inferred dependence among experts in represented reliability space. The central claim should be that people treat similarly reliable experts as partially dependent sources, so corroboration depends on both cue count and reliability diversity. Implement coalition evidence using a smooth similarity kernel over compressed diagnosticity—for example, an effective-evidence or precision calculation based on pairwise reliability-space similarities—rather than pi_8's special triadic duplicate-pair-in-context detector. Similar cues should share informational credit, while reliability-distant cues contribute more independent evidence. The mechanism should operate at every coalition size but accumulate nonlinearly, yielding weak dyadic gap effects and much stronger three-cue clustered-versus-distributed effects without an explicit n>=3 switch. Include sublinear count growth, a stable subject-level dependence/similarity sensitivity, compressed validity representation, exchangeable zero-centered cue salience, and lapse/response sensitivity. It should predict the robust distributed-coalition preferences in Experiments 1 and 14 and the graded early-gap release in Experiment 2, but with more moderate magnitudes than the current pi_8 instantiation. It must also preserve the established cross-task constraints: the count/majority patterns in Experiments 3–5, near-zero population positional and serial-organization effects in Experiments 6–11, the strong dyadic composition result in Experiment 12, and only a small dyadic reliability-gap contrast in Experiment 13. This would provide a substantive alternative explanation—general inferred source dependence—rather than copying pi_8's cardinality-specific contextual topology operation.


## CANDIDATE THEORY
Reliability-Space Credit Sharing proposes that people interpret agreement among experts through an inferred dependence network defined entirely in represented reliability space. Advertised validities have two related but distinct cognitive representations: a strongly compressed representation determines each cue's direct evidential weight, while a separately calibrated fixed-anchor representation determines inferred dependence. Within any option-supporting coalition, every expert shares informational credit with similarly represented experts according to a smooth, sharply localized similarity kernel. The amount of credit shared with a reliability-near neighbor is proportional to that neighbor's mean-normalized represented evidential strength, so corroboration by several similar strong experts implies more common-source redundancy than corroboration by equally similar weak experts. Reliability-distant experts contribute nearly undiluted evidence. The steep but continuous kernel concentrates redundancy around extremely similar reliability representations, producing a strong equality penalty without a broad dyadic reliability-gap gradient. Credit sharing applies at every coalition size, without an equality detector, dyadic/triadic switch, spatial grouping rule, or duplicate-pair-in-context operation. Because additional coalition members create multiple possible similarity relations, local dependence can accumulate nonlinearly in larger coalitions even when its effect on a single nonidentical pair is weak. Effective evidence is additionally subject to generic sublinear count growth. Stable subject-level dependence sensitivity, overlap curvature, compressed validity reliance, and exchangeable zero-centered cue salience generate individual differences without population-level serial-position effects. Choices arise from response sensitivity plus an independent lapse process, and representations remain fixed because no outcome feedback is provided.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Reliability-Space Credit Sharing expects shape (2, n_features); got {stim.shape}."
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
            f"cue_salience_profile length {raw_salience.size} != n_features {n_features}."
        )

    validity_compression = float(parameters["validity_compression"])
    direct_validity_compression = float(
        parameters["direct_validity_compression"]
    )
    validity_reliance = float(parameters["validity_reliance"])
    salience_strength = float(parameters["salience_strength"])
    coalition_saturation = float(parameters["coalition_saturation"])
    dependence_sensitivity = float(parameters["dependence_sensitivity"])
    similarity_bandwidth = float(parameters["similarity_bandwidth"])
    overlap_curvature = float(parameters["overlap_curvature"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Dependence and direct cue weighting use separately compressed versions
    # of the same instructed validity. Both retain the fixed 0.75 reference,
    # so their scales remain invariant across experiments.
    v = np.clip(validities, 0.500001, 0.999999)
    log_diagnosticity = np.log(v / (1.0 - v))
    reference_diagnosticity = np.log(0.75 / 0.25)

    represented = np.power(
        np.maximum(log_diagnosticity, 1e-12), validity_compression
    )
    represented /= max(
        float(reference_diagnosticity ** validity_compression), 1e-12
    )

    direct_represented = np.power(
        np.maximum(log_diagnosticity, 1e-12),
        direct_validity_compression,
    )
    direct_represented /= max(
        float(reference_diagnosticity ** direct_validity_compression),
        1e-12,
    )

    reliability_weight = (
        (1.0 - validity_reliance) * np.ones(n_features, dtype=np.float64)
        + validity_reliance * direct_represented
    )

    # Subject-specific salience is exchangeable over cue identities and is
    # centered within the display, precluding a population onset or recency
    # advantage while allowing stable individual configuration preferences.
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

    def coalition_signal(indices):
        idx = np.asarray(indices, dtype=int)
        n = int(idx.size)
        if n == 0:
            return 0.0
        if n == 1:
            return float(cue_weights[idx[0]])

        coordinates = represented[idx]
        gaps = coordinates[:, None] - coordinates[None, :]

        # A smooth super-Gaussian concentrates inferred dependence around
        # extremely close reliability representations. It remains continuous
        # and cardinality-general, but decays rapidly outside the local
        # overlap neighborhood.
        scaled_gaps = gaps / max(similarity_bandwidth, 1e-12)
        similarity = np.exp(-0.5 * np.power(scaled_gaps, 4.0))
        np.fill_diagonal(similarity, 0.0)

        # Credit shared with a similar neighbor is proportional to that
        # neighbor's represented evidential strength. Mean normalization
        # preserves the accepted global dependence scale while selectively
        # increasing redundancy among mutually similar strong experts.
        neighbor_strength = represented[idx] / max(
            float(np.mean(represented)), 1e-12
        )
        local_overlap = similarity @ neighbor_strength

        # Divisive credit allocation approximates precision loss under
        # correlated sources. In a pair, each cue has only one possible
        # neighbor. In larger coalitions, multiple weighted overlaps converge
        # on each member and create nonlinear redundancy without a size gate.
        independence_fraction = np.power(
            1.0 + dependence_sensitivity * local_overlap,
            -overlap_curvature,
        )
        effective_total = float(
            np.sum(cue_weights[idx] * independence_fraction)
        )

        # Generic count normalization makes corroboration sublinear even for
        # reliability-diverse coalitions. Dependence can further reduce the
        # marginal value of clustered additions but never deletes a cue.
        signal = effective_total / (float(n) ** coalition_saturation)
        return float(max(signal, 0.0))

    evidence_a = coalition_signal(a_support)
    evidence_b = coalition_signal(b_support)
    net_a = evidence_a - evidence_b

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
- validity_compression: [0.30, 0.34]
- direct_validity_compression: [0.08, 0.14]
- validity_reliance: [0.008, 0.025]
- salience_strength: [0.12, 0.28]
- coalition_saturation: [0.64, 0.72]
- dependence_sensitivity: [2.90, 3.30]
- similarity_bandwidth: [0.0045, 0.0065]
- overlap_curvature: [0.88, 0.96]
- beta: [1.12, 1.38]
- epsilon: [0.03, 0.08]

`rationale`:
This is a minimal structural edit to the accepted iteration-8 candidate. The rejected midpoint-dependent bandwidth is not retained, and all accepted parameter ranges, the constant narrow super-Gaussian kernel, additive overlap accumulation, divisive credit rule, count saturation, response sensitivity, lapse, fixed anchor, and exchangeable salience are preserved. The sole computational change replaces the unweighted neighbor count `sum_j similarity_ij` with `sum_j similarity_ij * represented_strength_j`, where represented strength is normalized by its experiment-wide mean. Thus a similar strong expert contributes more inferred common-source overlap than a similar weak expert, while the mean normalization avoids a global increase in dependence. This can selectively discount coalitions of mutually similar strong experts in Experiment 2 and strengthen the smooth equal-reliability penalty in Experiment 10 without widening the kernel or mechanically increasing the nonzero-gap gradient in Experiment 11. Accumulation remains additive and operates identically at every coalition size, so the edit introduces no equality detector, cardinality switch, positional mechanism, broad kernel tail, inverse-precision calculation, or superlinear overlap exponent.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.1428 -> ACCEPTED
- iter 2: loss=0.1163 -> ACCEPTED
- iter 3: loss=0.1133 -> ACCEPTED
- iter 4: loss=0.1157 -> REJECTED
- iter 5: loss=0.1137 -> REJECTED
- iter 6: loss=0.1100 -> ACCEPTED
- iter 7: loss=0.1154 -> REJECTED
- iter 8: loss=0.1005 -> ACCEPTED
- iter 9: loss=0.1100 -> REJECTED
- iter 10 (current candidate you are grading): loss=0.1115 -> REJECTED
Running-best (last accepted) base: iter 8 at loss=0.1005.

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
  - iter 1: -0.0533 (var=0.0157) (Δ vs real +0.0683)
  - iter 2: -0.3067 (var=0.0123) (Δ vs real -0.1850)
  - iter 3: -0.2508 (var=0.0222) (Δ vs real -0.1292)
  - iter 4: -0.1729 (var=0.0127) (Δ vs real -0.0512)
  - iter 5: -0.2546 (var=0.0148) (Δ vs real -0.1329)
  - iter 6: -0.2146 (var=0.0128) (Δ vs real -0.0929)
  - iter 7: -0.2767 (var=0.0119) (Δ vs real -0.1550)
  - iter 8: -0.2196 (var=0.0106) (Δ vs real -0.0979)
  - iter 9: -0.2525 (var=0.0133) (Δ vs real -0.1308)
  - iter 10 (current): -0.2729 (var=0.0114) (Δ vs real -0.1512)
**Other theories' values on this metric (for reference):**
- pi_1: 0.7129 (var=0.0389)
- pi_2: -0.6354 (var=0.0266)
- pi_3: -0.1033 (var=0.0092)
- pi_4: -0.1246 (var=0.0114)
- pi_5: -0.1179 (var=0.0105)
- pi_6: -0.0779 (var=0.0176)
- pi_7: -0.3083 (var=0.0280)
- pi_8: -0.1358 (var=0.0111)

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
  - iter 1: -0.0129 (var=0.0094) (Δ vs real +0.2512)
  - iter 2: 0.0317 (var=0.0120) (Δ vs real +0.2958)
  - iter 3: 0.0358 (var=0.0120) (Δ vs real +0.3000)
  - iter 4: -0.0096 (var=0.0094) (Δ vs real +0.2546)
  - iter 5: 0.0108 (var=0.0147) (Δ vs real +0.2750)
  - iter 6: 0.0125 (var=0.0103) (Δ vs real +0.2767)
  - iter 7: 0.0292 (var=0.0065) (Δ vs real +0.2933)
  - iter 8: -0.0029 (var=0.0103) (Δ vs real +0.2612)
  - iter 9: -0.0208 (var=0.0069) (Δ vs real +0.2433)
  - iter 10 (current): 0.0096 (var=0.0134) (Δ vs real +0.2737)
**Other theories' values on this metric (for reference):**
- pi_2: -0.0008 (var=0.0102)
- pi_1: 0.7221 (var=0.0226)
- pi_3: -0.1892 (var=0.0123)
- pi_4: -0.2667 (var=0.0120)
- pi_5: -0.2217 (var=0.0089)
- pi_6: -0.2321 (var=0.0159)
- pi_7: -0.0892 (var=0.0075)
- pi_8: -0.2221 (var=0.0114)

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
  - iter 1: -0.0003 (var=0.0003) (Δ vs real +0.0407)
  - iter 2: 0.0030 (var=0.0003) (Δ vs real +0.0440)
  - iter 3: -0.0001 (var=0.0002) (Δ vs real +0.0409)
  - iter 4: -0.0005 (var=0.0003) (Δ vs real +0.0405)
  - iter 5: 0.0016 (var=0.0003) (Δ vs real +0.0426)
  - iter 6: -0.0002 (var=0.0003) (Δ vs real +0.0408)
  - iter 7: -0.0017 (var=0.0002) (Δ vs real +0.0393)
  - iter 8: 0.0009 (var=0.0003) (Δ vs real +0.0419)
  - iter 9: 0.0027 (var=0.0003) (Δ vs real +0.0437)
  - iter 10 (current): 0.0035 (var=0.0003) (Δ vs real +0.0445)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0860 (var=0.0002)
- pi_2: -0.0017 (var=0.0001)
- pi_1: 0.0899 (var=0.0009)
- pi_4: -0.0260 (var=0.0002)
- pi_5: -0.0237 (var=0.0003)
- pi_6: -0.0366 (var=0.0003)
- pi_7: -0.0224 (var=0.0002)
- pi_8: -0.0376 (var=0.0003)

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
  - iter 1: 0.5109 (var=0.0065) (Δ vs real +0.0022)
  - iter 2: 0.4825 (var=0.0071) (Δ vs real -0.0262)
  - iter 3: 0.4965 (var=0.0082) (Δ vs real -0.0122)
  - iter 4: 0.5000 (var=0.0053) (Δ vs real -0.0086)
  - iter 5: 0.5087 (var=0.0043) (Δ vs real +0.0000)
  - iter 6: 0.5131 (var=0.0045) (Δ vs real +0.0044)
  - iter 7: 0.5016 (var=0.0074) (Δ vs real -0.0071)
  - iter 8: 0.5122 (var=0.0030) (Δ vs real +0.0036)
  - iter 9: 0.4995 (var=0.0040) (Δ vs real -0.0092)
  - iter 10 (current): 0.5091 (var=0.0047) (Δ vs real +0.0004)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5017 (var=0.0041)
- pi_3: 0.6281 (var=0.0049)
- pi_1: 0.1476 (var=0.0116)
- pi_4: 0.5061 (var=0.0029)
- pi_5: 0.4957 (var=0.0036)
- pi_6: 0.5046 (var=0.0065)
- pi_7: 0.4932 (var=0.0039)
- pi_8: 0.4759 (var=0.0052)

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
  - iter 1: 0.4971 (var=0.0031) (Δ vs real -0.0050)
  - iter 2: 0.5127 (var=0.0016) (Δ vs real +0.0106)
  - iter 3: 0.5056 (var=0.0022) (Δ vs real +0.0035)
  - iter 4: 0.4983 (var=0.0023) (Δ vs real -0.0037)
  - iter 5: 0.5094 (var=0.0020) (Δ vs real +0.0073)
  - iter 6: 0.5025 (var=0.0028) (Δ vs real +0.0004)
  - iter 7: 0.4908 (var=0.0025) (Δ vs real -0.0112)
  - iter 8: 0.4910 (var=0.0021) (Δ vs real -0.0110)
  - iter 9: 0.5088 (var=0.0021) (Δ vs real +0.0067)
  - iter 10 (current): 0.4973 (var=0.0027) (Δ vs real -0.0048)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5692 (var=0.0021)
- pi_4: 0.3798 (var=0.0029)
- pi_1: 0.1575 (var=0.0119)
- pi_2: 0.4992 (var=0.0023)
- pi_5: 0.5098 (var=0.0033)
- pi_6: 0.5123 (var=0.0025)
- pi_7: 0.5052 (var=0.0023)
- pi_8: 0.4940 (var=0.0029)

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
  - iter 1: 0.2600 (var=0.1924) (Δ vs real -0.1800)
  - iter 2: 0.2200 (var=0.1716) (Δ vs real -0.2200)
  - iter 3: 0.2200 (var=0.1716) (Δ vs real -0.2200)
  - iter 4: 0.3000 (var=0.2100) (Δ vs real -0.1400)
  - iter 5: 0.2400 (var=0.1824) (Δ vs real -0.2000)
  - iter 6: 0.2600 (var=0.1924) (Δ vs real -0.1800)
  - iter 7: 0.2000 (var=0.1600) (Δ vs real -0.2400)
  - iter 8: 0.2600 (var=0.1924) (Δ vs real -0.1800)
  - iter 9: 0.1800 (var=0.1476) (Δ vs real -0.2600)
  - iter 10 (current): 0.1800 (var=0.1476) (Δ vs real -0.2600)
**Other theories' values on this metric (for reference):**
- pi_4: 0.9800 (var=0.0196)
- pi_3: 0.0000 (var=0.0000)
- pi_1: 1.0000 (var=0.0000)
- pi_2: 1.0000 (var=0.0000)
- pi_5: 0.3200 (var=0.2176)
- pi_6: 0.3600 (var=0.2304)
- pi_7: 0.3200 (var=0.2176)
- pi_8: 0.5000 (var=0.2500)

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
  - iter 1: -0.0170 (var=0.0102) (Δ vs real -0.0389)
  - iter 2: 0.0075 (var=0.0094) (Δ vs real -0.0144)
  - iter 3: 0.0056 (var=0.0097) (Δ vs real -0.0162)
  - iter 4: -0.0007 (var=0.0082) (Δ vs real -0.0226)
  - iter 5: 0.0091 (var=0.0070) (Δ vs real -0.0128)
  - iter 6: -0.0090 (var=0.0083) (Δ vs real -0.0308)
  - iter 7: -0.0051 (var=0.0102) (Δ vs real -0.0270)
  - iter 8: -0.0032 (var=0.0076) (Δ vs real -0.0251)
  - iter 9: 0.0003 (var=0.0115) (Δ vs real -0.0215)
  - iter 10 (current): -0.0003 (var=0.0121) (Δ vs real -0.0221)
**Other theories' values on this metric (for reference):**
- pi_5: 0.2360 (var=0.0123)
- pi_4: 0.0009 (var=0.0095)
- pi_1: 0.0069 (var=0.0048)
- pi_2: -0.0035 (var=0.0050)
- pi_3: -0.0376 (var=0.0081)
- pi_6: 0.0257 (var=0.0065)
- pi_7: -0.0108 (var=0.0126)
- pi_8: 0.0088 (var=0.0097)

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
  - iter 1: 0.0137 (var=0.0116) (Δ vs real +0.0021)
  - iter 2: -0.0050 (var=0.0083) (Δ vs real -0.0167)
  - iter 3: -0.0017 (var=0.0117) (Δ vs real -0.0133)
  - iter 4: 0.0008 (var=0.0100) (Δ vs real -0.0108)
  - iter 5: -0.0092 (var=0.0107) (Δ vs real -0.0208)
  - iter 6: -0.0025 (var=0.0119) (Δ vs real -0.0142)
  - iter 7: 0.0054 (var=0.0089) (Δ vs real -0.0062)
  - iter 8: -0.0071 (var=0.0125) (Δ vs real -0.0187)
  - iter 9: 0.0029 (var=0.0116) (Δ vs real -0.0087)
  - iter 10 (current): 0.0304 (var=0.0103) (Δ vs real +0.0188)
**Other theories' values on this metric (for reference):**
- pi_4: 0.2729 (var=0.0146)
- pi_5: -0.0100 (var=0.0101)
- pi_1: 0.7325 (var=0.0409)
- pi_2: -0.0042 (var=0.0038)
- pi_3: -0.4775 (var=0.0097)
- pi_6: -0.0154 (var=0.0165)
- pi_7: -0.0021 (var=0.0101)
- pi_8: -0.0042 (var=0.0116)

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
  - iter 1: 0.5073 (var=0.0022) (Δ vs real +0.0115)
  - iter 2: 0.5015 (var=0.0025) (Δ vs real +0.0056)
  - iter 3: 0.4946 (var=0.0029) (Δ vs real -0.0013)
  - iter 4: 0.5071 (var=0.0025) (Δ vs real +0.0112)
  - iter 5: 0.4958 (var=0.0020) (Δ vs real +0.0000)
  - iter 6: 0.5008 (var=0.0023) (Δ vs real +0.0050)
  - iter 7: 0.4933 (var=0.0014) (Δ vs real -0.0025)
  - iter 8: 0.4942 (var=0.0037) (Δ vs real -0.0017)
  - iter 9: 0.4983 (var=0.0016) (Δ vs real +0.0025)
  - iter 10 (current): 0.5015 (var=0.0023) (Δ vs real +0.0056)
**Other theories' values on this metric (for reference):**
- pi_5: 0.6108 (var=0.0033)
- pi_6: 0.4996 (var=0.0028)
- pi_1: 0.4104 (var=0.0015)
- pi_2: 0.4985 (var=0.0025)
- pi_3: 0.6115 (var=0.0020)
- pi_4: 0.6271 (var=0.0029)
- pi_7: 0.4946 (var=0.0027)
- pi_8: 0.5056 (var=0.0026)

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
  - iter 1: 0.4007 (var=0.0041) (Δ vs real +0.2283)
  - iter 2: 0.2723 (var=0.0020) (Δ vs real +0.1000)
  - iter 3: 0.2460 (var=0.0034) (Δ vs real +0.0736)
  - iter 4: 0.2399 (var=0.0031) (Δ vs real +0.0676)
  - iter 5: 0.2754 (var=0.0033) (Δ vs real +0.1031)
  - iter 6: 0.2748 (var=0.0031) (Δ vs real +0.1025)
  - iter 7: 0.2640 (var=0.0038) (Δ vs real +0.0917)
  - iter 8: 0.2552 (var=0.0030) (Δ vs real +0.0829)
  - iter 9: 0.2691 (var=0.0029) (Δ vs real +0.0967)
  - iter 10 (current): 0.2604 (var=0.0038) (Δ vs real +0.0881)
**Other theories' values on this metric (for reference):**
- pi_6: 0.2057 (var=0.0035)
- pi_5: 0.5023 (var=0.0031)
- pi_1: 0.8796 (var=0.0067)
- pi_2: 0.4978 (var=0.0039)
- pi_3: 0.5414 (var=0.0029)
- pi_4: 0.3753 (var=0.0039)
- pi_7: 0.1814 (var=0.0023)
- pi_8: 0.1646 (var=0.0019)

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
  - iter 1: 0.0813 (var=0.0113) (Δ vs real +0.0496)
  - iter 2: 0.1129 (var=0.0103) (Δ vs real +0.0813)
  - iter 3: 0.1367 (var=0.0085) (Δ vs real +0.1050)
  - iter 4: 0.1771 (var=0.0088) (Δ vs real +0.1454)
  - iter 5: 0.1238 (var=0.0070) (Δ vs real +0.0921)
  - iter 6: 0.1300 (var=0.0056) (Δ vs real +0.0983)
  - iter 7: 0.1167 (var=0.0090) (Δ vs real +0.0850)
  - iter 8: 0.0958 (var=0.0112) (Δ vs real +0.0642)
  - iter 9: 0.1146 (var=0.0080) (Δ vs real +0.0829)
  - iter 10 (current): 0.1062 (var=0.0116) (Δ vs real +0.0746)
**Other theories' values on this metric (for reference):**
- pi_7: 0.2729 (var=0.0090)
- pi_6: 0.0217 (var=0.0086)
- pi_1: 0.0167 (var=0.0064)
- pi_2: -0.0292 (var=0.0133)
- pi_3: 0.1604 (var=0.0061)
- pi_4: 0.0054 (var=0.0079)
- pi_5: 0.0146 (var=0.0117)
- pi_8: 0.0125 (var=0.0063)

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
  - iter 1: 0.5158 (var=0.0034) (Δ vs real -0.1533)
  - iter 2: 0.6660 (var=0.0027) (Δ vs real -0.0031)
  - iter 3: 0.6835 (var=0.0024) (Δ vs real +0.0144)
  - iter 4: 0.6600 (var=0.0021) (Δ vs real -0.0092)
  - iter 5: 0.7094 (var=0.0031) (Δ vs real +0.0402)
  - iter 6: 0.6763 (var=0.0026) (Δ vs real +0.0071)
  - iter 7: 0.6973 (var=0.0019) (Δ vs real +0.0281)
  - iter 8: 0.6950 (var=0.0029) (Δ vs real +0.0258)
  - iter 9: 0.6813 (var=0.0020) (Δ vs real +0.0121)
  - iter 10 (current): 0.6840 (var=0.0031) (Δ vs real +0.0148)
**Other theories' values on this metric (for reference):**
- pi_6: 0.4883 (var=0.0028)
- pi_7: 0.5575 (var=0.0028)
- pi_1: 0.1550 (var=0.0096)
- pi_2: 0.5073 (var=0.0027)
- pi_3: 0.6663 (var=0.0014)
- pi_4: 0.5715 (var=0.0020)
- pi_5: 0.5602 (var=0.0027)
- pi_8: 0.7031 (var=0.0082)

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
  - iter 1: 0.1679 (var=0.0096) (Δ vs real -0.1563)
  - iter 2: 0.3754 (var=0.0128) (Δ vs real +0.0512)
  - iter 3: 0.4000 (var=0.0099) (Δ vs real +0.0758)
  - iter 4: 0.4321 (var=0.0096) (Δ vs real +0.1079)
  - iter 5: 0.4167 (var=0.0102) (Δ vs real +0.0925)
  - iter 6: 0.3613 (var=0.0122) (Δ vs real +0.0371)
  - iter 7: 0.3629 (var=0.0124) (Δ vs real +0.0387)
  - iter 8: 0.3642 (var=0.0119) (Δ vs real +0.0400)
  - iter 9: 0.3708 (var=0.0101) (Δ vs real +0.0467)
  - iter 10 (current): 0.3600 (var=0.0101) (Δ vs real +0.0358)
**Other theories' values on this metric (for reference):**
- pi_8: 0.5346 (var=0.0171)
- pi_6: 0.0158 (var=0.0163)
- pi_1: -0.6938 (var=0.0472)
- pi_2: 0.0129 (var=0.0129)
- pi_3: 0.6592 (var=0.0081)
- pi_4: 0.2250 (var=0.0097)
- pi_5: 0.2188 (var=0.0122)
- pi_7: 0.3412 (var=0.0162)

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
  - iter 1: 0.5253 (var=0.0038) (Δ vs real -0.0616)
  - iter 2: 0.5931 (var=0.0046) (Δ vs real +0.0062)
  - iter 3: 0.5931 (var=0.0039) (Δ vs real +0.0062)
  - iter 4: 0.6047 (var=0.0031) (Δ vs real +0.0178)
  - iter 5: 0.5969 (var=0.0025) (Δ vs real +0.0100)
  - iter 6: 0.6053 (var=0.0036) (Δ vs real +0.0184)
  - iter 7: 0.5778 (var=0.0035) (Δ vs real -0.0091)
  - iter 8: 0.5953 (var=0.0037) (Δ vs real +0.0084)
  - iter 9: 0.5891 (var=0.0046) (Δ vs real +0.0022)
  - iter 10 (current): 0.5941 (var=0.0035) (Δ vs real +0.0072)
**Other theories' values on this metric (for reference):**
- pi_6: 0.5012 (var=0.0028)
- pi_8: 0.6616 (var=0.0045)
- pi_1: 0.1222 (var=0.0090)
- pi_2: 0.5122 (var=0.0043)
- pi_3: 0.6784 (var=0.0035)
- pi_4: 0.6269 (var=0.0040)
- pi_5: 0.6112 (var=0.0040)
- pi_7: 0.5816 (var=0.0036)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed inferred-dependence family and successfully preserves the null positional/serial patterns: Experiments 4, 5, 7, 8, and 9 are close to observed values. However, its reliability-space dependence signal is much too weak on the central composition tests. Distributed-coalition preference is underestimated in Experiment 12 (0.5158 versus 0.6692), Experiment 13 (0.1679 versus 0.3242), and Experiment 14 (0.5253 versus 0.5869). The equal-versus-diverse composition effect in Experiment 10 is also far too weak: the candidate chooses the equal-reliability coalition at 0.4007 rather than the observed 0.1723. At the same time, it produces too broad a dyadic gap gradient in Experiment 11 (0.0813 versus 0.0317), with excessive between-subject variance. Thus the kernel is paradoxically insufficiently punitive at exact or very close reliability overlap but too responsive across wider reliability gaps. Generic count evidence is also too suppressed: the candidate underpredicts majority/count effects in Experiments 1, 2, 3, and 6, most dramatically in Experiment 2 (-0.0129 versus -0.2642) and Experiment 3 (-0.0003 versus -0.0410).
Rationale: Keep the smooth reliability-space credit-sharing mechanism, but recalibrate its shape and count normalization. Use a narrower similarity bandwidth together with stronger local dependence sensitivity or curvature. This should sharply discount exact and near-equal coalitions, improving Experiments 10 and 12–14, while allowing similarity to decay sooner so that the broad dyadic gap contrast in Experiment 11 becomes smaller rather than larger. Prefer a smooth, sharply peaked kernel or adjusted distance normalization—not an equality detector or cardinality switch. Separately, reduce coalition_saturation moderately so that three-cue and larger majorities retain more evidence; the present 0.70–0.82 range over-normalizes count and contributes to the misses in Experiments 1–3 and 6. Lowering saturation must be paired with the sharper near-overlap discount so clustered triads do not become indiscriminately stronger. A modest increase in beta can amplify composition contrasts only after those structural changes; beta alone would not fix the incorrect E10/E11 profile. Finally, narrow the between-subject dependence/bandwidth ranges after calibration, because Experiment 11's simulated variance is much too large. The null positional results should be protected by retaining centered exchangeable salience and avoiding any serial-position term.

**Outcome of your advice:** iter 1 candidate loss=0.1428 -> iter 2 candidate loss=0.1163 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed smooth reliability-space dependence family, and the accepted revision made substantial progress: Experiments 12 and 14 are now essentially exact, Experiment 13 is reasonably close, and the population-level positional and serial nulls in Experiments 4, 5, and 7–9 remain well preserved. Nevertheless, the fit is not yet experiment-invariant. The largest failure is Experiment 2, where the model predicts slightly positive agreement with the highest-validity discriminating cue (0.0317) while humans show a strong reversal (-0.2642). Count behavior is also miscalibrated in opposite directions across tasks: the model overstates majority opposition to the top cue in Experiment 1 (-0.3067 versus -0.1217), yet underpredicts the early three-cue coalition in Experiment 6 (0.2200 versus 0.4400) and misses Experiment 3's negative pattern. The dyadic dependence profile remains wrong as well: the exact/equal-reliability penalty is still too weak in Experiment 10 (0.2723 versus 0.1723), while the broader reliability-gap contrast is too large in Experiment 11 (0.1129 versus 0.0317), with excessive subject variance. Thus the previous sharper-kernel adjustment helped the central triadic composition tests but did not produce a transferable reliability metric across experiments.
Rationale: Build on the accepted sharper credit-sharing model, but do not repeat the previous broad push toward lower coalition saturation: although that revision improved aggregate loss and Experiments 12–14, it drove Experiment 1 past the target and did not repair Experiments 2, 3, or 6. Use an intermediate or slightly higher saturation range between the two iterations to moderate Experiment 1, rather than lowering it again. More importantly, revise the reliability-coordinate normalization within the same mechanism family. Dividing represented diagnosticity by each experiment's mean makes a single bandwidth denote different absolute reliability gaps in different validity sets; this likely explains why the kernel fits Experiments 12–14 but fails to transfer to Experiments 1–3 and produces the wrong E10/E11 profile. Anchor compressed log-diagnosticity to a fixed scale or fixed reference interval across experiments, then retune a sharply local smooth bandwidth. The target is stronger discounting at equality or extremely small gaps for Experiment 10, but faster decay outside that neighborhood to reduce Experiment 11, without increasing the already accurate Experiment 12 and 14 effects. Narrow subject-level bandwidth/compression variation as well, because Experiment 11 variance remains far above observed. Finally, diagnose Experiment 2 at the deterministic evidence level before changing beta: its sign is wrong, so response sensitivity cannot fix it. Check how the anchored coordinates and divisive overlap rank the opposing three-cue coalitions, and adjust validity reliance or kernel distance scaling so collective diverse evidence can overturn the first discriminating cue. Retain centered exchangeable salience and avoid positional terms, equality detectors, or cardinality switches.

**Outcome of your advice:** iter 2 candidate loss=0.1163 -> iter 3 candidate loss=0.1133 -> the gate marked it ACCEPTED.

### Iteration 3
Verdict: regenerate
Interpretation: The accepted fixed-anchor revision modestly improves aggregate loss and remains faithful to the prescribed reliability-space credit-sharing family. It preserves the near-null positional and serial effects in Experiments 4, 5, and 7–9, fits Experiment 12 closely, and is essentially exact in Experiment 14. Experiment 10 also moved in the correct direction. However, the theory is not yet experiment-invariant. Experiment 2 remains the decisive failure: predicted agreement with the first discriminating cue has the wrong sign and is near zero (0.0358) rather than the strong human reversal (-0.2642). Experiment 6 also substantially underestimates endorsement of the three-cue coalition (0.2200 versus 0.4400), and Experiment 3 misses its negative majority pattern. Meanwhile, Experiment 1 still overstates majority opposition to the top-validity cue (-0.2508 versus -0.1217). The dyadic dependence profile remains miscalibrated: the equal-reliability coalition is still chosen too often in Experiment 10 (0.2460 versus 0.1723), whereas the broader gap effect in Experiment 11 is now much too large (0.1367 versus 0.0317). Experiment 13 is also somewhat excessive (0.4000 versus 0.3242). Thus the fixed anchor helped transfer overall, but the current single-scale Gaussian does not jointly deliver a sharp equality penalty, weak dyadic gap effects, and strong graded triadic diversity effects.
Rationale: Retain the accepted fixed reliability anchor and the general credit-sharing mechanism, but change the shape of the smooth similarity kernel rather than repeating another global bandwidth or saturation shift. The previous normalization advice was accepted and improved loss, Experiment 1, and Experiment 10, but it simultaneously increased the Experiment 11 gap contrast and did not repair Experiment 2; another move in the same single-bandwidth direction is therefore unlikely to solve the tradeoff. Within the prescribed family, try a smooth sharply peaked core plus a weak broader component, or an equivalent super-Gaussian/shape-controlled kernel. The narrow core should impose a stronger penalty at equality for Experiment 10 and decay rapidly enough to suppress Experiment 11, while a weak tail can accumulate over the three pairwise relations in triads and produce the graded release needed in Experiment 2 without creating a large one-pair dyadic effect. Couple this with slightly stronger nonlinear overlap curvature so weak pairwise similarities matter collectively in triads, still without any coalition-size switch. Do not broadly lower coalition saturation again: that earlier combined push overshot Experiment 1 and failed to improve Experiment 6. Instead, modestly restore validity reliance to give the uniquely best cue more leverage in Experiment 1, while using the revised triadic overlap accumulation—not reduced validity weighting—to generate Experiment 2's reversal. Check deterministic trial rankings in Experiment 2 to ensure the diverse coalition actually wins before tuning beta. Finally, narrow dependence-shape heterogeneity after calibration, because Experiment 11's simulated between-subject variance remains far above the observed variance. Preserve centered exchangeable salience and all positional null constraints.

**Outcome of your advice:** iter 3 candidate loss=0.1133 -> iter 4 candidate loss=0.1157 -> the gate marked it REJECTED.

### Iteration 4
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed inferred-dependence family and retains several important successes: Experiment 12 is close (0.6600 versus 0.6692), Experiment 14 is reasonably close (0.6047 versus 0.5869), and the positional/serial controls in Experiments 4, 5, and 7–9 remain near zero. It also improves Experiment 1 and, modestly, Experiments 6 and 10 relative to the accepted iteration 3 base. Nevertheless, the two-scale-kernel revision was rejected by the loss gate and does not resolve the central cross-experiment conflict. Experiment 2 still has essentially the wrong qualitative pattern: -0.0096 instead of the strong human reversal of -0.2642. Experiment 3 remains near zero rather than -0.0410, and Experiment 6 still underpredicts three-cue endorsement (0.3000 versus 0.4400). Most importantly, the weak broad tail makes the supposedly small dyadic gap effect dramatically worse in Experiment 11: 0.1771 versus 0.0317, up from 0.1367 in the accepted base. Experiment 13 similarly rises farther above its target (0.4321 versus 0.3242). Experiment 10 remains insufficiently punitive at equality (0.2399 versus 0.1723). Thus the added tail produces broad reliability-gap sensitivity in dyads without generating the required triadic reversal in Experiment 2—the opposite of its intended selective accumulation.
Rationale: Because the current candidate was rejected, build again from the accepted iteration 3 fixed-anchor model rather than extending the rejected core-plus-tail change. My previous recommendation to add a broad tail and stronger overlap curvature did not help, so do not repeat it or increase dependence sensitivity further. Try a different in-family evidence calculation: retain one compact smooth reliability-similarity kernel, but replace the current additive nodewise-overlap/divisive heuristic with a regularized coalition-level effective-precision calculation (for example, a similarity/correlation matrix with an eigenvalue floor or ridge, followed by a bounded effective-information calculation). This remains exactly the prescribed inferred-source-dependence mechanism, but can discount a nearly singular equal-reliability coalition strongly while preventing many moderate pairwise similarities from creating an unbounded dyadic gap gradient. Normalize or saturate coalition-level redundancy rather than adding a broad pairwise tail.

Calibrate this equation against deterministic trial rankings before adjusting beta: it must make the non-first-cue coalition win substantially more often in Experiment 2; the present near-zero metric shows that response noise is not the limiting factor. At the same time, explicitly constrain the high-minus-low dyadic contrast in Experiment 11 near 0.03 and the Experiment 13 contrast near 0.32. Preserve the accepted fixed reliability anchor, centered exchangeable salience, and current count normalization initially; previous broad saturation changes did not solve Experiments 2, 3, or 6. Once the dependence equation has the correct signs, only a small intermediate adjustment to coalition saturation or beta should be used to recover Experiment 6 without again overshooting Experiment 1. Narrow salience/dependence heterogeneity after calibration, since Experiment 11 variance remains about three times the observed value. Protect the already good Experiment 12 and the positional nulls throughout.

**Outcome of your advice:** iter 4 candidate loss=0.1157 -> iter 5 candidate loss=0.1137 -> the gate marked it REJECTED.

### Iteration 5
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed inferred-dependence family, but the effective-precision substitution was rejected by the gate and slightly regresses from the accepted iteration-3 base. It preserves the positional and serial nulls in Experiments 4, 5, and 7–9 and remains close on Experiment 14. However, the central pattern is still not experiment-invariant. Experiment 2 remains qualitatively wrong: agreement with the first discriminating cue is near zero and positive (0.0108), whereas humans strongly reverse it (-0.2642). Count and majority behavior is also inconsistent: Experiment 1 shows excessive opposition to the top cue (-0.2546 versus -0.1217), Experiments 3 and 6 understate the relevant majority effects, and Experiment 6 reaches only 0.2400 versus 0.4400. The dependence profile remains too broad but insufficiently sharp at equality: the equal-coalition choice in Experiment 10 is too high (0.2754 versus 0.1723), while Experiment 11's gap contrast is almost four times too large (0.1238 versus 0.0317) and its subject variance is over twice the observed value. Experiments 12 and 13 also overshoot (0.7094 versus 0.6692 and 0.4167 versus 0.3242). Relative to the accepted base, the precision calculation worsens Experiments 1, 10, 12, and 13 without repairing Experiment 2.
Rationale: Build from the unchanged accepted iteration-3 base. My previous recommendation to replace its redundancy calculation with inverse-matrix effective precision produced a rejected candidate, so do not repeat or further strengthen that substitution. Try a different, smaller in-family adjustment: decouple the reliability representation used for direct cue weighting from the representation used to infer dependence. At present the same compressed coordinate controls both reliability_weight and kernel distances, forcing a tradeoff: changes intended to sharpen coalition redundancy also create excessive direct gap sensitivity in Experiment 11. Retain the fixed anchor, but give direct validity weighting its own stronger compression and lower, narrower reliance range, while independently calibrating the smooth dependence coordinate and bandwidth. This should suppress the broad E11 gradient without eliminating the local redundancy needed for Experiments 10 and 12–14. Use the accepted bounded coalition-redundancy rule initially rather than another global evidence-form replacement. Calibrate deterministic rankings first: the alternative coalition must win substantially more often in Experiment 2, equality must be more strongly discounted in Experiment 10, and nonzero dyadic gaps must contribute much less in Experiment 11. A smooth shape-controlled kernel with a steep local falloff is permissible, but do not reintroduce the previously rejected broad tail or increase global dependence strength. Hold coalition saturation and beta near the accepted values during this test; earlier broad count and response-sensitivity changes did not solve Experiments 2, 3, or 6 and risked further overshooting Experiment 1. After obtaining the correct deterministic signs, only a small intermediate saturation adjustment should be considered. Preserve centered exchangeable salience and narrow dependence/representation heterogeneity to reduce the excessive Experiment-11 variance.

**Outcome of your advice:** iter 5 candidate loss=0.1137 -> iter 6 candidate loss=0.1100 -> the gate marked it ACCEPTED.

### Iteration 6
Verdict: regenerate
Interpretation: The accepted decoupling of direct validity weighting from dependence coordinates improved the running-best loss and brought Experiment 13 closer while preserving the strong fit in Experiment 12 and the near-null positional/serial effects in Experiments 4, 5, and 7–9. The candidate is therefore faithful to the prescribed reliability-space dependence family and represents genuine progress. It is not yet experiment-invariant, however. Experiment 2 remains the clearest qualitative failure: the candidate predicts near-zero, slightly positive agreement with the first discriminating cue (0.0125), whereas humans show a strong reversal (-0.2642). Count-related behavior also remains inconsistent: Experiment 1 overpredicts opposition to the highest-validity cue (-0.2146 versus -0.1217), Experiment 3 is effectively null rather than negative, and Experiment 6 substantially underpredicts endorsement of the three-cue coalition (0.2600 versus 0.4400). The dependence profile still fails its key dyadic calibration: Experiment 10 does not penalize equal-reliability corroboration enough (0.2748 versus 0.1723), while Experiment 11 produces a reliability-gap contrast about four times too large (0.1300 versus 0.0317), with excessive between-subject variance. Experiments 12–14 are comparatively successful, although Experiment 14 is slightly high. Thus the current sharply localized kernel fits several triadic composition tests but still cannot combine a strong equality penalty, weak nonzero dyadic-gap effects, and the required graded multi-cue reversal.
Rationale: Build on this accepted candidate: the previous advice to separate direct and dependence representations helped, so retain that separation, the fixed anchor, centered exchangeable salience, and the current general credit-sharing family. The next adjustment should target how pairwise similarities combine within a coalition, rather than again widening the kernel, adding a broad tail, replacing the rule with inverse precision, or globally changing saturation—all of those directions were previously ineffective or rejected. Introduce a modest smooth superlinear transform of aggregate local overlap, such as using (sum_j similarity_ij)^q with q slightly above 1 before divisive credit allocation. This remains cardinality-general and has no n>=3 switch: for a dyad, a nonexact similarity below one is attenuated, reducing Experiment 11, while exact similarity remains one and can still receive the strong penalty needed in Experiment 10; in a triad, two moderate similarities can jointly exceed one and accumulate nonlinearly, potentially producing the missing Experiment 2 reversal. Pair this with a small reduction in baseline dependence sensitivity if necessary so Experiments 12–14, already near target, do not become too strong. Calibrate deterministic trial rankings before changing beta: the alternative coalition must systematically beat the first-cue coalition in Experiment 2, not merely become noisier. Keep direct validity reliance low and strongly compressed, but consider slightly less compression in the dependence coordinate so genuinely nonzero gaps separate more rapidly; compensate bandwidth only enough to preserve exact/near-equality discounting. Do not use a broad saturation shift to address Experiments 1 and 6, since their opposite errors indicate coalition-composition misranking rather than a single count-strength problem. After the structural profile is correct, only a small intermediate adjustment to saturation or response sensitivity should be attempted. Finally, narrow dependence-coordinate and kernel heterogeneity to bring Experiment 11 variance down toward 0.003 without sacrificing the positional nulls.

**Outcome of your advice:** iter 6 candidate loss=0.1100 -> iter 7 candidate loss=0.1154 -> the gate marked it REJECTED.

### Iteration 7
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed inferred-dependence family, but the current superlinear-overlap revision was rejected by the accept gate and worsened aggregate loss from 0.1100 to 0.1154. Its main successes remain the near-null positional and serial effects in Experiments 4, 5, and 7–9, a reasonably strong distributed-coalition effect in Experiment 12, and close fits in Experiments 13–14. However, it is not experiment-invariant. Experiment 2 is still qualitatively wrong and deteriorated: predicted agreement with the first discriminating cue is +0.0292 rather than the observed reversal of -0.2642. Experiment 1 now overstates opposition to the best cue (-0.2767 versus -0.1217), Experiment 6 further underpredicts endorsement of the three-cue coalition (0.2000 versus 0.4400), and Experiment 3 remains effectively null. The dyadic profile is also still incorrect: Experiment 10 requires a stronger equality penalty (0.2640 versus 0.1723), while Experiment 11 requires a much smaller nonzero-gap contrast (0.1167 versus 0.0317) and substantially less between-subject variance. Experiment 12 now overshoots as well (0.6973 versus 0.6692). Thus raising aggregate local overlap to a power above one slightly reduced the excessive Experiment-11 contrast, but amplified the wrong coalition structures in Experiments 1, 2, 6, and 12.
Rationale: Build from the unchanged accepted iteration-6 base. My previous recommendation to add a superlinear local-overlap exponent was rejected, so do not retain or further increase q; return overlap_exponent to 1 rather than repeating that direction. The next revision should target the dependence-coordinate geometry, which was not changed in this attempt. Keep the accepted separation between direct and dependence representations, but modestly reduce compression in the dependence coordinate—e.g. test a somewhat higher validity_compression range—while retuning a single narrow, sharply bounded smooth kernel on that transformed scale. This can change which Experiment-2 triads are judged clustered versus diverse, while allowing nonzero dyadic gaps in Experiment 11 to leave the narrow similarity neighborhood more rapidly. Pair a narrower core with only a small increase in local dependence curvature if needed to strengthen the exact/near-exact penalty in Experiment 10; do not add a broad tail or increase global dependence sensitivity, since those earlier directions were rejected and tended to overshoot Experiments 11–13.

Before stochastic simulation, explicitly inspect deterministic coalition signals for every Experiment-2 trial. The dependence mapping must make the coalition opposing the first discriminating cue win often enough to produce a negative metric; the persistent near-zero or positive result across all iterations shows that beta, lapse, and sampling noise cannot repair the current ranking. Use Experiments 10 and 11 jointly as calibration constraints: preserve a strong smooth penalty at equality while forcing similarities at the low-but-nonzero gaps to decay sharply. Hold coalition_saturation and beta near the accepted iteration-6 values during this structural test, because broad count changes previously traded Experiment 1 against Experiment 6 without fixing Experiment 2. Once the coalition rankings are correct, consider only a small intermediate count adjustment. Finally, narrow dependence-compression, bandwidth, and salience ranges to reduce Experiment 11's simulated variance from 0.0090 toward 0.0030, while retaining centered exchangeable salience and the successful positional nulls.

**Outcome of your advice:** iter 7 candidate loss=0.1154 -> iter 8 candidate loss=0.1005 -> the gate marked it ACCEPTED.

### Iteration 8
Verdict: regenerate
Interpretation: The accepted iteration is a meaningful aggregate improvement and remains faithful to the prescribed smooth reliability-space dependence family. The revised dependence geometry moves both dyadic diagnostics in the right direction: Experiment 10 improves from 0.2748 to 0.2552 and Experiment 11 from 0.1300 to 0.0958. It also preserves the near-null positional and serial effects in Experiments 4, 5, and 7–9, while Experiment 14 is close to target. However, the model is still not experiment-invariant. Experiment 2 remains the decisive qualitative failure: the prediction is essentially zero (-0.0029), while humans strongly choose against the highest-validity discriminating cue (-0.2642). Experiment 3 is likewise null rather than negative, and Experiment 6 substantially underpredicts endorsement of the three-cue coalition (0.26 versus 0.44). Experiment 1 still overstates opposition to the top cue (-0.2196 versus -0.1217). The dependence profile is improved but not calibrated: Experiment 10 still chooses the equal-reliability coalition too often, and Experiment 11 remains about three times too large with substantially excessive between-subject variance (0.0112 versus 0.0030). Experiments 12 and 13 now modestly overshoot their composition effects. Thus the candidate captures the broad distributed-coalition phenomenon but not the required graded early-gap reversal or the cross-task count constraints.
Rationale: Build on this accepted iteration. The most recent advice to use a less-compressed dependence coordinate and narrower single kernel was successful, so retain the fixed anchor, separate direct and dependence representations, additive local-overlap rule, and centered exchangeable salience. However, do not merely continue narrowing the same constant bandwidth: the large persistent Experiment-2 error indicates that global gap scaling alone is unlikely to repair coalition rankings. Instead, make one small in-family distance-metric refinement by allowing the smooth kernel bandwidth to vary continuously with the pair's reliability-space midpoint, such as a tightly bounded Weber-like local scale. This remains a general reliability-space similarity mechanism with stable subject parameters, but can distinguish near-neighbor relations in the validity region used by Experiment 2 from the nonzero dyadic gaps driving Experiment 11. Calibrate this using deterministic signals: the coalition opposing the first discriminating cue must win on substantially more Experiment-2 trials, while the Experiment-11 high-minus-low contrast remains near 0.03. Exact matches should remain at similarity one, so a slight increase in overlap curvature can strengthen Experiment 10 without broadening the kernel; offset any resulting overshoot in Experiments 12–14 with a small reduction in beta rather than weaker dependence. After correcting those structural rankings, test only a modest coordinated reduction in coalition saturation and beta: lower saturation can raise the three-versus-two effect in Experiment 6, while lower beta can prevent Experiment 1 and the already-strong composition effects from becoming more extreme. This should be a small joint adjustment, not the previously unsuccessful broad saturation push. Finally, narrow bandwidth, midpoint-scaling, and dependence ranges to reduce Experiment-11 heterogeneity. Do not reintroduce the rejected broad-tail kernel, inverse-precision replacement, or superlinear overlap exponent.

**Outcome of your advice:** iter 8 candidate loss=0.1005 -> iter 9 candidate loss=0.1100 -> the gate marked it REJECTED.

### Iteration 9 (most recent)
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed inferred-dependence family, but it was rejected by the accept gate: loss worsened from the accepted iteration 8 value of 0.1005 to 0.1100. The midpoint-dependent bandwidth provides only a tiny improvement on the central Experiment 2 failure (-0.0208 versus -0.2642) and improves Experiment 14, but it regresses several cross-task constraints. Experiment 1 becomes too negative (-0.2525 versus -0.1217), Experiment 6 falls from 0.26 to 0.18 versus the observed 0.44, and Experiment 3 remains essentially null rather than negative. The dyadic profile also worsens: Experiment 10 rises to 0.2691 versus 0.1723, while Experiment 11 rises to 0.1146 versus 0.0317 and still has excessive between-subject variance. Experiment 13 remains too strong, although Experiment 12 is close and Experiments 4–5 and 7–9 continue to preserve the positional and serial nulls. Overall, the local Weber scaling changes response magnitudes without producing the required Experiment 2 coalition rankings and sacrifices the accepted base's better aggregate calibration.
Rationale: Build from the unchanged accepted iteration-8 base. My previous recommendation to add midpoint-dependent bandwidth was rejected, so do not retain, enlarge, or further tune midpoint_scaling; restore the constant narrow kernel rather than repeating that direction. Also restore the accepted beta and overlap-curvature settings initially, because their coordinated change confounds the failed geometry test and worsened Experiments 1, 10, 11, and 13.

Try a different small adjustment within the same credit-sharing mechanism: change the normalization of neighbor overlap so that informational credit shared with cue j is smoothly weighted by j's represented evidential strength, rather than treating every similar neighbor as contributing the same unit of redundancy. For example, replace the unweighted sum of similarities with a bounded or mean-normalized sum of similarity times the neighbor's cue/reliability weight. Keep accumulation additive across neighbors and the same divisive independence rule—no cardinality switch, positional term, equality detector, broad tail, inverse-precision replacement, or superlinear overlap exponent. This reliability-weighted sharing can selectively penalize coalitions containing several mutually similar strong experts, potentially changing Experiment 2's deterministic rankings, while avoiding another global widening or narrowing that inflates Experiment 11.

Before stochastic simulation, verify trial-level deterministic rankings in Experiment 2; unless the coalition opposing the first discriminating cue wins on many more trials, beta, lapse, and parameter-range changes cannot generate the observed -0.2642. Jointly constrain Experiment 10 near 0.17 and Experiment 11 near 0.03 when selecting the overlap normalization. Hold coalition saturation near the accepted iteration-8 range during this structural test: repeated broad count adjustments have traded Experiment 1 against Experiment 6 without solving either jointly. Only after the ranking profile improves should a very small intermediate saturation/temperature adjustment be considered. Retain the accepted fixed anchor, separate direct and dependence representations, centered exchangeable salience, and narrow subject-level kernel parameters to preserve the positional nulls and reduce Experiment-11 heterogeneity.

**Outcome of your advice:** iter 9 candidate loss=0.1100 -> CURRENT candidate loss=0.1115 -> the gate marked it REJECTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The reliability-weighted credit-sharing edit is faithful to the prescribed inferred-dependence family, but it was rejected by the gate and worsened loss from the accepted iteration-8 value of 0.1005 to 0.1115. The proposed weighting does not repair Experiment 2: agreement with the first discriminating cue remains near zero and has the wrong sign (0.0096 versus -0.2642). It also worsens Experiment 1 (-0.2729 versus -0.1217) and Experiment 6 (0.18 versus 0.44), while Experiment 3 remains essentially null instead of negative. The dyadic calibration remains inadequate: the equality penalty in Experiment 10 is too weak (0.2604 versus 0.1723), whereas the nonzero reliability-gap contrast in Experiment 11 is over three times too large (0.1062 versus 0.0317), with excessive subject variance. By contrast, Experiments 12\u201314 remain reasonably close, and the positional and serial controls in Experiments 4\u20135 and 7\u20139 remain near their observed nulls. Thus the model captures distributed-coalition preferences in selected tasks but still lacks a transferable way for weak pairwise dependence to accumulate selectively in multi-cue coalitions.",
  "verdict": "regenerate",
  "rationale": "Build again from the unchanged accepted iteration-8 base. My most recent recommendation to weight overlap by neighbor reliability was rejected, so remove that weighting rather than further tuning or bounding it. Retain the accepted fixed anchor, separate direct and dependence representations, constant narrow kernel, centered exchangeable salience, and sublinear count term.\n\nTry a different in-family combination rule: bounded smooth diffusion over the reliability-similarity graph. For example, construct the same pairwise similarity matrix S and use a normalized first- plus second-order overlap, S + \u03bbS\u00b2, excluding diagonal/self-credit and controlling its scale by a fixed bound or spectral normalization before divisive credit allocation. This is a general inferred-dependence operation at every coalition size, not an explicit triadic detector. A dyad has no distinct two-step path, so its nonzero-gap effect is governed mainly by a deliberately small first-order coefficient, which can reduce Experiment 11. Coalitions of three or more cues possess genuine indirect paths, allowing several locally similar sources to accumulate dependence smoothly and potentially change the deterministic rankings in Experiment 2. Keep \u03bb modest and bounded so Experiments 12\u201314 do not overshoot. Do not reintroduce the rejected broad kernel tail, inverse-precision calculation, superlinear overlap power, midpoint-dependent bandwidth, or reliability-strength weighting.\n\nCalibrate deterministic signals before simulation: the coalition opposing the first discriminating cue must win on many more Experiment-2 trials; otherwise no temperature adjustment can produce the observed reversal. Simultaneously require a strong exact-overlap penalty for Experiment 10 and a weak first-order nonzero-gap contrast for Experiment 11. Once those rankings are correct, a small reduction in beta is worth testing: it would moderate the excessive Experiment-1 reversal and may move Experiment 6's overly deterministic subject-level opposition toward its observed 0.44. Such a temperature change should occur only after the new diffusion term supplies stronger structural effects in Experiments 2 and 10. Hold coalition saturation near the accepted range, since repeated saturation changes have not resolved the opposing errors in Experiments 1 and 6. Finally, narrow diffusion and bandwidth heterogeneity to reduce Experiment-11 variance while preserving the successful positional nulls."
}
```

## Usage

```json
{
  "input_tokens": 34926,
  "output_tokens": 1774
}
```
