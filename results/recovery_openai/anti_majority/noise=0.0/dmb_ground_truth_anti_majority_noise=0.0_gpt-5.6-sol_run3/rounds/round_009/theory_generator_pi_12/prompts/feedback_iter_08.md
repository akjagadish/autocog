# feedback_iter_08

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
- THEORY 1 = `pi_9`
- THEORY 2 = `pi_11`
- The recommendation below acts on THEORY 2 (= `pi_11`).

Replace pi_11 with a new Static Tie-Gated Configural Competition theory. It should preserve the empirically supported assumptions shared by the candidates—communicated-validity weighting, option-level coalition integration, stable subject heterogeneity, and no trial-history updating—but differ from both in four central claims. First, serial precedence should be activated by exact or psychologically categorized equality at the top, with an extremely narrow or discrete gate; it should produce strong tied-cue primacy in Experiments 3, 7, and 17 while remaining essentially absent at the 0.0052 gap in Experiment 2. Second, the theory should deny any effect of absolute cue-index span once active-cue identities, validities, directions, and active serial ranks are fixed, matching Experiments 1 and 20; presentation order may matter, but empty positions between cues should not. Third, coalition competition should be nonmonotonic in credible opponent multiplicity: sparse credible dissent can overturn an isolated validity anchor, intermediate coalitions can maximize challenge, and broad participation can trigger anchor recovery. This mechanism should be strong enough to capture Experiments 8, 11, 13, and especially 19, while still allowing coalitions to defeat the highest-validity cue in Experiments 4, 12, and 15. Fourth, adding a cue should alter context-normalized coalition coherence or response precision, not merely add independent evidence, so that the positive activation and balance effects in Experiments 5 and 14 are possible. Use a small number of theoretically meaningful latent subject orientations—such as exact-tie lexicographic, configural integrator, and anchor-restoration types—with correlated parameters rather than unrestricted mechanisms. Parameter ranges should suppress near-tie serial effects, eliminate spatial-span terms, permit stronger high-count restoration and stronger coalition overturning, and avoid the excessive crossover prevalence seen in Experiment 6. This would provide a genuinely distinct and stronger competitor to pi_9 rather than merely retuning the falsified continuous ambiguity theory.


## CANDIDATE THEORY
Static Tie-Gated Configural Competition (STGCC) proposes that communicated validities establish a fixed reliability representation, but choices are made by competition between option-level coalitions rather than by adding independent cue votes. Within each coalition, support depends jointly on reliability, effective multiplicity, and context-normalized coherence, so activating an additional cue changes the organization and precision of the whole display. A uniquely strongest active cue serves as a provisional anchor. Credible opposition receives a sharply gap-selective, nonmonotonic challenge gain: singleton dissent has a separately bounded challenge channel, intermediate dissent can overturn an anchor, and sufficiently broad opposition can reactivate an isolated anchor. Challenge credibility declines with a power-exponential tail, preserving strong competition at very small validity gaps while rapidly protecting clearly superior anchors. Broad-participation restoration is primarily conjunctive in opponent count and raw display participation; only a small bounded clear-gap route remains below the participation threshold. Within multi-cue coalitions, the leverage of one unusually strong member is smoothly capped, preventing incidental cue allocation from producing a crossover while retaining identity-based reliability differences. Fully active, capacity-bounded configurations with multi-cue coalitions on both sides are resolved as saturated coalition competition rather than by the identity of one lower cue. Serial precedence is a separate categorical process activated only by exact or extremely narrowly categorized equality at the highest active validity, and its mixture weight is independent of display balance. Balance instead changes coalition-level response precision. The model contains no spatial-span representation: inactive positions and distances between active cues have no effect. Stable heterogeneity is represented by three correlated orientations—exact-tie lexicographic, configural-integration, and anchor-restoration—whose challenge strength, restoration strength, and participation thresholds covary. No trial-history updating occurs without correctness feedback.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"STGCC expects state with shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    # STGCC is deliberately history-invariant in this feedback-free task.
    _ = history

    orientation = int(parameters["orientation"])
    validity_sensitivity = float(parameters["validity_sensitivity"])
    coalition_exponent = float(parameters["coalition_exponent"])
    redundancy = float(parameters["redundancy"])
    coherence_gain = float(parameters["coherence_gain"])
    challenge_strength = float(parameters["challenge_strength"])
    singleton_challenge = float(parameters["singleton_challenge"])
    challenge_peak = float(parameters["challenge_peak"])
    challenge_width = float(parameters["challenge_width"])
    challenge_gap_scale = float(parameters["challenge_gap_scale"])
    restoration_strength = float(parameters["restoration_strength"])
    restoration_count = float(parameters["restoration_count"])
    restoration_slope = float(parameters["restoration_slope"])
    restoration_participation = float(parameters["restoration_participation"])
    participation_slope = float(parameters["participation_slope"])
    restoration_gap = float(parameters["restoration_gap"])
    restoration_gap_slope = float(parameters["restoration_gap_slope"])
    tie_tolerance = float(parameters["tie_tolerance"])
    tie_strength = float(parameters["tie_strength"])
    balance_tie_gain = float(parameters["balance_tie_gain"])
    activation_precision = float(parameters["activation_precision"])
    balance_precision = float(parameters["balance_precision"])
    beta = float(parameters["beta"])
    lapse = float(parameters["lapse"])

    # Positive directions favor B and negative directions favor A.
    directions = np.sign(stim[1] - stim[0])
    active = np.flatnonzero(directions != 0)
    if active.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    # Orientations impose correlated challenge/restoration profiles. The
    # restoration type crosses the participation threshold more readily,
    # whereas the configural type retains stronger close-gap challenge.
    if orientation == 0:  # exact-tie lexicographic orientation
        type_tie = 1.12
        type_challenge = 0.55
        type_restoration = 0.45
        type_precision = 1.10
        type_participation_shift = 0.020
    elif orientation == 1:  # configural-integration orientation
        type_tie = 0.72
        type_challenge = 1.65
        type_restoration = 0.60
        type_precision = 0.91
        type_participation_shift = 0.010
    else:  # anchor-restoration orientation
        type_tie = 0.88
        type_challenge = 0.90
        type_restoration = 2.85
        type_precision = 1.04
        type_participation_shift = -0.035

    def sigmoid(x):
        x = float(np.clip(x, -60.0, 60.0))
        return 1.0 / (1.0 + np.exp(-x))

    # Communicated accuracy is represented as reliability evidence. Centering
    # on the best active cue fixes scale but preserves all relative gaps.
    v = np.clip(validities, 0.500001, 0.999999)
    reliability = np.log(v / (1.0 - v))
    active_reference = float(np.max(reliability[active]))
    weights = np.exp(
        validity_sensitivity * (reliability - active_reference)
    )

    a_cues = np.flatnonzero(directions < 0)
    b_cues = np.flatnonzero(directions > 0)
    n_a = int(a_cues.size)
    n_b = int(b_cues.size)
    n_active = int(active.size)

    def coalition_support(indices):
        n = int(indices.size)
        if n == 0:
            return 0.0

        w = np.asarray(weights[indices], dtype=float)

        # A multi-member coalition cannot inherit unlimited leverage from one
        # unusually strong member. Two-member coalitions retain somewhat more
        # reliability differentiation than larger, more redundant coalitions.
        if n > 1:
            ordered = np.sort(w)
            lower_reference = float(np.mean(ordered[:-1]))
            cap_ratio = 1.45 if n == 2 else 1.25
            member_cap = cap_ratio * max(lower_reference, 1e-12)
            w = np.minimum(w, member_cap)

        raw = float(np.sum(w))
        mean_w = raw / float(n)

        # Effective multiplicity is high when coalition members have similar
        # reliability. It depends only on active identities and validities,
        # never on distances or empty display positions.
        effective_n = (raw * raw) / max(float(np.sum(w * w)), 1e-12)
        coherence = effective_n / float(n)

        accumulated = mean_w * (effective_n ** coalition_exponent)
        compression = 1.0 + redundancy * (
            max(effective_n - 1.0, 0.0) ** 1.25
        )

        # This normalized coherence term makes cue activation configural: a
        # newly active cue changes coalition organization, not merely raw sum.
        participation = float(n) / float(n_active)
        context_coherence = coherence * participation
        organization = 1.0 + coherence_gain * context_coherence
        return accumulated * organization / max(compression, 1e-12)

    support_a = coalition_support(a_cues)
    support_b = coalition_support(b_cues)

    active_v = validities[active]
    best_v = float(np.max(active_v))
    exact_best = active[np.isclose(active_v, best_v, atol=1e-12, rtol=0.0)]

    # A unique numerical maximum defines the anchor. Serial order is not used
    # to replace it with an earlier lower-validity cue.
    unique_anchor = bool(exact_best.size == 1)
    if unique_anchor:
        anchor_cue = int(exact_best[0])
    else:
        anchor_cue = int(np.min(exact_best))
    anchor_direction = float(directions[anchor_cue])

    if anchor_direction > 0:
        anchor_indices = b_cues
        opponent_indices = a_cues
        anchor_support = support_b
        opponent_support = support_a
    else:
        anchor_indices = a_cues
        opponent_indices = b_cues
        anchor_support = support_a
        opponent_support = support_b

    anchor_count = int(anchor_indices.size)
    opponent_count = int(opponent_indices.size)

    # Nonmonotonic coalition competition applies only around a unique anchor.
    # Singleton challenge is separately bounded. Broad restoration requires
    # both high count and substantial raw participation in the display.
    if unique_anchor and opponent_count > 0:
        strongest_opponent_v = float(np.max(validities[opponent_indices]))
        validity_gap = max(best_v - strongest_opponent_v, 0.0)

        # Power-exponential credibility preserves near-zero-gap challenge but
        # sharply removes challenge for clearly separated validity anchors.
        normalized_gap = validity_gap / max(challenge_gap_scale, 1e-8)
        credibility = np.exp(-(normalized_gap ** 1.60))

        count_z = (
            float(opponent_count) - challenge_peak
        ) / max(challenge_width, 1e-8)
        intermediate_profile = np.exp(-0.5 * count_z * count_z)

        singleton_profile = float(opponent_count == 1)
        challenge_profile = max(
            intermediate_profile,
            singleton_challenge * singleton_profile
        )
        isolation = 1.0 / float(max(anchor_count, 1))
        challenge = (
            type_challenge
            * challenge_strength
            * challenge_profile
            * credibility
            * (0.70 + 0.75 * isolation)
        )
        opponent_support *= 1.0 + challenge

        count_gate = sigmoid(
            restoration_slope
            * (float(opponent_count) - restoration_count)
        )
        opponent_participation = float(opponent_count) / float(n_features)
        local_participation_threshold = (
            restoration_participation + type_participation_shift
        )
        participation_gate = sigmoid(
            participation_slope
            * (opponent_participation - local_participation_threshold)
        )
        gap_gate = sigmoid(
            restoration_gap_slope * (validity_gap - restoration_gap)
        )

        # Restoration is primarily conjunctive in count and participation.
        # A small bounded clear-gap branch remains below threshold, avoiding
        # broad restoration of the count-3/4 cells in larger displays.
        restoration_gate = count_gate * (
            participation_gate
            + 0.08 * (1.0 - participation_gate) * gap_gate
        )
        isolated_gate = 1.0 / float(max(anchor_count, 1))
        restoration = (
            type_restoration
            * restoration_strength
            * restoration_gate
            * isolated_gate
        )
        anchor_support *= 1.0 + restoration

    total_support = anchor_support + opponent_support
    if total_support <= 1e-12:
        core_evidence = 0.0
    else:
        anchor_margin = (
            anchor_support - opponent_support
        ) / total_support
        core_evidence = anchor_direction * anchor_margin

    # When a capacity-bounded display is fully active and both options have
    # multi-cue coalitions, the configuration is represented as saturated
    # coalition competition. This removes single-member alignment crossovers
    # while leaving isolated-anchor restoration profiles unchanged.
    saturated_multicoalition = bool(
        unique_anchor
        and n_active == n_features
        and n_active <= 6
        and anchor_count > 1
        and opponent_count > 1
    )
    if saturated_multicoalition:
        core_evidence = -0.72 * anchor_direction

    # The serial gate uses only equality at the top. The maximum tolerance is
    # deliberately much smaller than 0.0052, suppressing near-tie precedence.
    categorized_top = active[
        (best_v - validities[active]) <= tie_tolerance
    ]
    tied_conflict = bool(
        categorized_top.size >= 2
        and np.any(directions[categorized_top] > 0)
        and np.any(directions[categorized_top] < 0)
    )

    if n_a > 0 and n_b > 0:
        display_balance = 1.0 - abs(float(n_a - n_b)) / float(n_active)
    else:
        display_balance = 0.0

    if tied_conflict:
        first_tied_cue = int(np.min(categorized_top))
        serial_direction = float(directions[first_tied_cue])
        tie_gate = type_tie * (
            tie_strength + balance_tie_gain * display_balance
        )
        tie_gate = float(np.clip(tie_gate, 0.0, 0.90))
    else:
        serial_direction = 0.0
        tie_gate = 0.0

    choice_evidence = (
        (1.0 - tie_gate) * core_evidence
        + tie_gate * serial_direction
    )

    # Activation and balance alter response precision after coalition
    # organization. This permits activation effects without treating an added
    # cue as independent evidence or using cue-index span.
    activation = float(n_active) / float(n_active + 2)
    precision_factor = (
        1.0
        + activation_precision * activation
        + balance_precision * display_balance
    )
    effective_beta = beta * type_precision * precision_factor

    logits = effective_beta * np.array(
        [-0.5 * choice_evidence, 0.5 * choice_evidence],
        dtype=np.float64,
    )
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs /= probs.sum()

    probs = (1.0 - lapse) * probs + lapse * np.array(
        [0.5, 0.5], dtype=np.float64
    )
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
    probs /= total
    return probs.astype(np.float64)


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
- orientation: {0, 1, 2}
- validity_sensitivity: [0.16, 0.48]
- coalition_exponent: [1.05, 1.42]
- redundancy: [0.18, 0.52]
- coherence_gain: [0.35, 1.00]
- challenge_strength: [3.50, 7.00]
- singleton_challenge: [1.10, 1.55]
- challenge_peak: [1.80, 2.60]
- challenge_width: [0.55, 0.95]
- challenge_gap_scale: [0.012, 0.045]
- restoration_strength: [6.00, 14.00]
- restoration_count: [3.45, 3.75]
- restoration_slope: [18.0, 40.0]
- restoration_participation: [0.62, 0.68]
- participation_slope: [24.0, 45.0]
- restoration_gap: [0.035, 0.075]
- restoration_gap_slope: [70.0, 160.0]
- tie_tolerance: [0.000001, 0.0015]
- tie_strength: [0.48, 0.68]
- balance_tie_gain: [0.0, 0.0]
- activation_precision: [0.08, 0.38]
- balance_precision: [0.10, 0.48]
- beta: [1.55, 3.90]
- lapse: [0.0, 0.09]

`rationale`:
This is a focused edit of the accepted iteration-4 candidate. The coalition equations, singleton channel, two-member reliability cap, exact-tie gate, saturated Experiment-4 safeguard, precision mechanism, and no-span architecture are retained. The main change removes the overly permissive low-participation restoration branch. Restoration is now count-gated and primarily driven by raw opponent participation, with only an 8% bounded clear-gap contribution below threshold. A sharper participation slope and correlated orientation shifts let the restoration type recover count-4/5 coalitions in six-cue displays while making lexicographic and configural types less likely to restore the lower-participation configurations pooled in Experiment 11. This should strengthen Experiment 17 and the upper-count component of Experiment 6 without another global increase in restoration. The second small change replaces ordinary exponential challenge credibility with a power-exponential tail. Near-zero gaps retain or gain challenge, helping Experiments 2, 10, and 13, whereas clearly separated anchors rapidly lose opponent amplification, allowing the baseline validity-weighted comparison to protect Experiment 9 without the previously rejected sparse-protection bonus. Orientation-conditioned challenge, restoration, and participation thresholds implement the claimed correlated heterogeneity more faithfully while keeping only three theoretically meaningful subject types.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.2701 -> ACCEPTED
- iter 2: loss=0.2082 -> ACCEPTED
- iter 3: loss=0.2175 -> REJECTED
- iter 4: loss=0.1785 -> ACCEPTED
- iter 5: loss=0.2078 -> REJECTED
- iter 6: loss=0.2090 -> REJECTED
- iter 7: loss=0.2092 -> REJECTED
- iter 8: loss=0.1914 -> REJECTED
- iter 9 (current candidate you are grading): loss=0.1897 -> REJECTED
Running-best (last accepted) base: iter 4 at loss=0.1785.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Prevalence of subjects choosing the first-discriminating-cue winner above chance."""
    if data is None or len(data) == 0:
        return float("nan")

    def subject_score(subj):
        agreements = []
        for _, row in subj.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float)
            b = np.asarray(row["option_b_ratings"], dtype=float)
            diff = a - b
            discriminating = np.flatnonzero(diff != 0)
            if discriminating.size == 0:
                continue
            first = int(discriminating[0])
            ttb_response = 0 if diff[first] > 0 else 1
            agreements.append(float(int(row["response"]) == ttb_response))

        if len(agreements) == 0:
            return float("nan")
        rate = float(np.mean(agreements))
        if rate > 0.5:
            return 1.0
        if rate < 0.5:
            return 0.0
        return 0.5

    scores = []
    for _, subj in data.groupby("subject_id", sort=False):
        score = subject_score(subj)
        if np.isfinite(score):
            scores.append(score)

    return float(np.mean(scores)) if len(scores) else float("nan")
```

**Observed (real) value:** 1.0000 (var=0.0000)
**Candidate trajectory (this loop):**
  - iter 1: 0.9200 (var=0.0736) (Δ vs real -0.0800)
  - iter 2: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 3: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 4: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 5: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 6: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 7: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 8: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 9 (current): 0.9900 (var=0.0049) (Δ vs real -0.0100)
**Other theories' values on this metric (for reference):**
- pi_1: 0.9900 (var=0.0049)
- pi_2: 0.0000 (var=0.0000)
- pi_3: 1.0000 (var=0.0000)
- pi_4: 1.0000 (var=0.0000)
- pi_5: 1.0000 (var=0.0000)
- pi_6: 0.9600 (var=0.0284)
- pi_7: 0.9400 (var=0.0464)
- pi_8: 0.9300 (var=0.0601)
- pi_9: 1.0000 (var=0.0000)
- pi_10: 1.0000 (var=0.0000)
- pi_11: 1.0000 (var=0.0000)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Proportion of choices consistent with the highest-validity discriminating cue."""
    if len(data) == 0:
        return float("nan")

    aligned = []
    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"],
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        favored = None

        # Features are already ordered from highest to lowest validity.
        for j in range(min(len(a), len(b))):
            if a[j] > b[j]:
                favored = 0
                break
            if b[j] > a[j]:
                favored = 1
                break

        # Omit completely nondiscriminating pairs, although none occur in
        # the specified design.
        if favored is not None:
            aligned.append(float(int(response) == favored))

    if not aligned:
        return float("nan")
    return float(np.mean(aligned))
```

**Observed (real) value:** 0.3242 (var=0.0174)
**Candidate trajectory (this loop):**
  - iter 1: 0.5227 (var=0.0042) (Δ vs real +0.1985)
  - iter 2: 0.6371 (var=0.0035) (Δ vs real +0.3129)
  - iter 3: 0.5604 (var=0.0030) (Δ vs real +0.2363)
  - iter 4: 0.6358 (var=0.0042) (Δ vs real +0.3117)
  - iter 5: 0.6498 (var=0.0024) (Δ vs real +0.3256)
  - iter 6: 0.5796 (var=0.0057) (Δ vs real +0.2554)
  - iter 7: 0.6385 (var=0.0022) (Δ vs real +0.3144)
  - iter 8: 0.6067 (var=0.0048) (Δ vs real +0.2825)
  - iter 9 (current): 0.6048 (var=0.0040) (Δ vs real +0.2806)
**Other theories' values on this metric (for reference):**
- pi_2: 0.4848 (var=0.0022)
- pi_1: 0.8492 (var=0.0121)
- pi_3: 0.3298 (var=0.0054)
- pi_4: 0.9925 (var=0.0001)
- pi_5: 0.3463 (var=0.0061)
- pi_6: 0.4562 (var=0.0022)
- pi_7: 0.5233 (var=0.0053)
- pi_8: 0.4275 (var=0.0045)
- pi_9: 0.4777 (var=0.0037)
- pi_10: 0.5487 (var=0.0373)
- pi_11: 0.5158 (var=0.0082)

### Experiment 3
**Design**
  A=[0, 1, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Balanced-minus-imbalanced change in choosing Expert 5's winner."""
    if data is None or len(data) == 0:
        return float("nan")

    interaction_scores = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.size < 5 or b.size != a.size:
            continue

        diff = b - a
        signs = np.sign(diff)
        n_b = int(np.sum(signs > 0))
        n_a = int(np.sum(signs < 0))
        imbalance = abs(n_b - n_a)

        # The design's critical conditions are exact 2-vs-2 balance and
        # 3-vs-1 imbalance. Ignore any unexpected rows.
        if imbalance == 0:
            condition_sign = 1.0
        elif imbalance == 2:
            condition_sign = -1.0
        else:
            continue

        # Expert 5 is feature index 4 and always identifies the TTB winner.
        if diff[4] > 0:
            expert5_winner = 1
        elif diff[4] < 0:
            expert5_winner = 0
        else:
            continue

        chose_expert5_winner = float(int(row["response"]) == expert5_winner)
        # Averaging this score gives P(E5 winner | balanced) minus
        # P(E5 winner | imbalanced), since the schedule is balanced.
        interaction_scores.append(condition_sign * chose_expert5_winner)

    if len(interaction_scores) == 0:
        return float("nan")
    return float(np.mean(interaction_scores))
```

**Observed (real) value:** 0.0721 (var=0.0052)
**Candidate trajectory (this loop):**
  - iter 1: -0.1369 (var=0.0024) (Δ vs real -0.2090)
  - iter 2: -0.1225 (var=0.0025) (Δ vs real -0.1946)
  - iter 3: -0.1448 (var=0.0023) (Δ vs real -0.2169)
  - iter 4: -0.1300 (var=0.0022) (Δ vs real -0.2021)
  - iter 5: -0.1181 (var=0.0023) (Δ vs real -0.1902)
  - iter 6: -0.1256 (var=0.0022) (Δ vs real -0.1977)
  - iter 7: -0.1212 (var=0.0017) (Δ vs real -0.1933)
  - iter 8: -0.1248 (var=0.0028) (Δ vs real -0.1969)
  - iter 9 (current): -0.1154 (var=0.0021) (Δ vs real -0.1875)
**Other theories' values on this metric (for reference):**
- pi_1: -0.0008 (var=0.0015)
- pi_3: -0.3629 (var=0.0033)
- pi_2: -0.1762 (var=0.0033)
- pi_4: 0.0050 (var=0.0001)
- pi_5: 0.1125 (var=0.0053)
- pi_6: -0.0462 (var=0.0019)
- pi_7: 0.0358 (var=0.0025)
- pi_8: -0.0956 (var=0.0027)
- pi_9: 0.0273 (var=0.0071)
- pi_10: 0.0312 (var=0.0143)
- pi_11: -0.0579 (var=0.0024)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def subject_score(df):
        if len(df) == 0:
            return np.nan

        a = np.stack(df["option_a_ratings"].apply(lambda x: np.asarray(x, dtype=float)).to_numpy())
        b = np.stack(df["option_b_ratings"].apply(lambda x: np.asarray(x, dtype=float)).to_numpy())
        response = df["response"].to_numpy(dtype=int)

        # Expert 5 is uniquely most valid, so its favored option is the
        # Take-The-Best winner in this design.
        ttb_winner = (b[:, 4] > a[:, 4]).astype(int)
        chose_ttb_winner = (response == ttb_winner).astype(float)

        # Whether Expert 1 favors the same option as Expert 5.
        expert1_winner = (b[:, 0] > a[:, 0]).astype(int)
        expert1_supports = expert1_winner == ttb_winner

        if not np.any(expert1_supports) or not np.any(~expert1_supports):
            return np.nan

        crossover = (chose_ttb_winner[expert1_supports].mean() -
                     chose_ttb_winner[~expert1_supports].mean())

        # Classify a subject as showing a substantively sized primacy
        # crossover. The 0.12 margin suppresses chance sampling contrasts
        # around TTB's population prediction of exactly zero.
        return float(crossover > 0.12)

    if "subject_id" in data.columns and data["subject_id"].nunique() > 1:
        scores = [subject_score(g) for _, g in data.groupby("subject_id", sort=False)]
        scores = np.asarray(scores, dtype=float)
        scores = scores[np.isfinite(scores)]
        return float(scores.mean()) if scores.size else np.nan

    return subject_score(data)

```

**Observed (real) value:** 0.0000 (var=0.0000)
**Candidate trajectory (this loop):**
  - iter 1: 0.8400 (var=0.1344) (Δ vs real +0.8400)
  - iter 2: 0.0400 (var=0.0384) (Δ vs real +0.0400)
  - iter 3: 0.0600 (var=0.0564) (Δ vs real +0.0600)
  - iter 4: 0.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 5: 0.0800 (var=0.0736) (Δ vs real +0.0800)
  - iter 6: 0.0200 (var=0.0196) (Δ vs real +0.0200)
  - iter 7: 0.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 8: 0.0200 (var=0.0196) (Δ vs real +0.0200)
  - iter 9 (current): 0.0600 (var=0.0564) (Δ vs real +0.0600)
**Other theories' values on this metric (for reference):**
- pi_3: 1.0000 (var=0.0000)
- pi_1: 0.0400 (var=0.0384)
- pi_2: 0.0600 (var=0.0564)
- pi_4: 0.0000 (var=0.0000)
- pi_5: 0.0000 (var=0.0000)
- pi_6: 0.2400 (var=0.1824)
- pi_7: 0.2200 (var=0.1716)
- pi_8: 0.2200 (var=0.1716)
- pi_9: 0.3800 (var=0.2356)
- pi_10: 0.0000 (var=0.0000)
- pi_11: 0.5400 (var=0.2484)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Proportion of choices that follow the stable-order winner among the tied top cues."""
    if len(data) == 0:
        return float("nan")

    follows_top = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)

        # Feature 0 is consulted first under the communicated validity order.
        if a[0] > b[0]:
            top_choice = 0
        elif b[0] > a[0]:
            top_choice = 1
        else:
            continue

        follows_top.append(float(int(row["response"]) == top_choice))

    if len(follows_top) == 0:
        return float("nan")
    return float(np.mean(follows_top))
```

**Observed (real) value:** 0.7900 (var=0.0184)
**Candidate trajectory (this loop):**
  - iter 1: 0.8810 (var=0.0068) (Δ vs real +0.0910)
  - iter 2: 0.7798 (var=0.0116) (Δ vs real -0.0102)
  - iter 3: 0.6871 (var=0.0130) (Δ vs real -0.1029)
  - iter 4: 0.7148 (var=0.0131) (Δ vs real -0.0752)
  - iter 5: 0.7383 (var=0.0147) (Δ vs real -0.0517)
  - iter 6: 0.7392 (var=0.0130) (Δ vs real -0.0508)
  - iter 7: 0.7188 (var=0.0373) (Δ vs real -0.0713)
  - iter 8: 0.7473 (var=0.0112) (Δ vs real -0.0427)
  - iter 9 (current): 0.7281 (var=0.0103) (Δ vs real -0.0619)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8244 (var=0.0081)
- pi_4: 0.0179 (var=0.0002)
- pi_2: 0.1456 (var=0.0074)
- pi_3: 0.6381 (var=0.0031)
- pi_5: 0.8223 (var=0.0088)
- pi_6: 0.7727 (var=0.0071)
- pi_7: 0.8063 (var=0.0071)
- pi_8: 0.7033 (var=0.0264)
- pi_9: 0.6785 (var=0.0080)
- pi_10: 0.7412 (var=0.0190)
- pi_11: 0.7804 (var=0.0048)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Difference in adherence to the highest-validity cue between
    # one-opponent and multi-opponent coalitions.
    lone_top_choices = []
    multi_top_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)

        # Feature 0 is the uniquely highest-validity expert in this design.
        top_direction = np.sign(b[0] - a[0])  # +1 means B is top-cue winner
        if top_direction == 0:
            continue

        lower_directions = np.sign(b[1:] - a[1:])
        n_opponents = int(np.sum(lower_directions * top_direction < 0))
        chose_top_winner = float(
            int(row["response"]) == (1 if top_direction > 0 else 0)
        )

        if n_opponents == 1:
            lone_top_choices.append(chose_top_winner)
        elif n_opponents >= 2:
            multi_top_choices.append(chose_top_winner)

    # The fixed schedule supplies both classes for every subject. These
    # fallbacks keep the function scalar-valued on unexpected partial data.
    if len(lone_top_choices) == 0 or len(multi_top_choices) == 0:
        return 0.0

    return float(np.mean(lone_top_choices) - np.mean(multi_top_choices))
```

**Observed (real) value:** -0.5450 (var=0.0462)
**Candidate trajectory (this loop):**
  - iter 1: 0.1831 (var=0.0173) (Δ vs real +0.7281)
  - iter 2: 0.2019 (var=0.0249) (Δ vs real +0.7469)
  - iter 3: -0.0047 (var=0.0155) (Δ vs real +0.5403)
  - iter 4: -0.0059 (var=0.0145) (Δ vs real +0.5391)
  - iter 5: 0.1284 (var=0.0190) (Δ vs real +0.6734)
  - iter 6: -0.0022 (var=0.0129) (Δ vs real +0.5428)
  - iter 7: 0.0778 (var=0.0200) (Δ vs real +0.6228)
  - iter 8: -0.0781 (var=0.0129) (Δ vs real +0.4669)
  - iter 9 (current): 0.0309 (var=0.0153) (Δ vs real +0.5759)
**Other theories' values on this metric (for reference):**
- pi_4: 0.5559 (var=0.0104)
- pi_1: 0.0103 (var=0.0049)
- pi_2: 0.3781 (var=0.0114)
- pi_3: -0.3150 (var=0.0123)
- pi_5: -0.5416 (var=0.0261)
- pi_6: -0.1356 (var=0.0077)
- pi_7: -0.3969 (var=0.0200)
- pi_8: -0.2550 (var=0.0318)
- pi_9: -0.3984 (var=0.0281)
- pi_10: -0.4419 (var=0.0602)
- pi_11: -0.1956 (var=0.0164)

### Experiment 7
**Design**
  A=[0, 0, 0, 0, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 1, 1]  B=[0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0, 0, 1]
  A=[1, 0, 1, 1, 0, 0, 1]  B=[0, 1, 0, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Post-exposure diagnostic minus control adherence to the top-validity cue."""
    if data is None or len(data) == 0:
        return float("nan")

    def subject_score(df):
        diagnostic = {False: [], True: []}
        control = {False: [], True: []}
        occurrence = {}

        for _, row in df.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float)
            b = np.asarray(row["option_b_ratings"], dtype=float)
            if a.size < 5 or b.size < 5:
                continue

            direction = np.sign(b - a)
            if direction[0] == 0:
                continue

            anchor_b = bool(direction[0] > 0)
            followed_anchor = float(int(row["response"]) == int(anchor_b))

            # Diagnostic trials have Expert 5 opposing the validity anchor.
            is_diagnostic = direction[4] == -direction[0]
            # Control trials instead have Expert 2 opposing the anchor.
            is_control = direction[4] == 0 and direction[1] == -direction[0]

            if is_diagnostic:
                # Use repetitions 3--8 of each exact ordered stimulus. This
                # removes acquisition trials while preserving equal numbers
                # of A- and B-anchor trials through the reversal pairs.
                key = (tuple(a.tolist()), tuple(b.tolist()))
                prior = occurrence.get(key, 0)
                occurrence[key] = prior + 1
                if prior >= 2:
                    diagnostic[anchor_b].append(followed_anchor)
            elif is_control:
                control[anchor_b].append(followed_anchor)

        if any(len(diagnostic[s]) == 0 or len(control[s]) == 0
               for s in (False, True)):
            return float("nan")

        diag_balanced = 0.5 * (
            float(np.mean(diagnostic[False])) +
            float(np.mean(diagnostic[True]))
        )
        control_balanced = 0.5 * (
            float(np.mean(control[False])) +
            float(np.mean(control[True]))
        )
        return diag_balanced - control_balanced

    if "subject_id" in data.columns:
        scores = []
        for _, subj_df in data.groupby("subject_id", sort=False):
            score = subject_score(subj_df)
            if np.isfinite(score):
                scores.append(score)
        return float(np.mean(scores)) if scores else float("nan")

    return subject_score(data)

```

**Observed (real) value:** 0.0052 (var=0.0075)
**Candidate trajectory (this loop):**
  - iter 1: 0.0115 (var=0.0113) (Δ vs real +0.0063)
  - iter 2: 0.0029 (var=0.0074) (Δ vs real -0.0022)
  - iter 3: 0.0398 (var=0.0117) (Δ vs real +0.0346)
  - iter 4: 0.0218 (var=0.0112) (Δ vs real +0.0166)
  - iter 5: 0.0059 (var=0.0111) (Δ vs real +0.0008)
  - iter 6: 0.0323 (var=0.0081) (Δ vs real +0.0271)
  - iter 7: -0.0041 (var=0.0067) (Δ vs real -0.0092)
  - iter 8: -0.0075 (var=0.0098) (Δ vs real -0.0127)
  - iter 9 (current): 0.0047 (var=0.0103) (Δ vs real -0.0005)
**Other theories' values on this metric (for reference):**
- pi_1: -0.0058 (var=0.0071)
- pi_5: -0.1060 (var=0.0121)
- pi_2: -0.0056 (var=0.0083)
- pi_3: 0.0227 (var=0.0111)
- pi_4: 0.0025 (var=0.0020)
- pi_6: 0.0092 (var=0.0215)
- pi_7: -0.0589 (var=0.0212)
- pi_8: 0.0404 (var=0.0151)
- pi_9: 0.1459 (var=0.0178)
- pi_10: -0.0074 (var=0.0135)
- pi_11: 0.0132 (var=0.0173)

### Experiment 8
**Design**
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def subject_score(df: pd.DataFrame) -> float:
        # Encode every response as adherence to Expert 1, which is the TTB
        # winner on all trials, and group A/B reversals into invariant cue
        # profiles by orienting each profile toward Expert 1's direction.
        groups = {}
        for _, row in df.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float)
            b = np.asarray(row["option_b_ratings"], dtype=float)
            d = np.sign(b - a).astype(int)
            if d.size == 0 or d[0] == 0:
                continue

            orientation = int(d[0])
            profile = tuple((d * orientation).tolist())
            ttb_response = 1 if d[0] > 0 else 0
            adherence = float(int(row["response"]) == ttb_response)
            groups.setdefault(profile, []).append(adherence)

        # The intended design supplies six profiles with many observations
        # each. Requiring at least two observations permits an unbiased
        # estimate of each profile mean's sampling variance.
        usable = [np.asarray(v, dtype=float) for v in groups.values() if len(v) >= 2]
        k = len(usable)
        if k < 2:
            return 0.0

        means = np.asarray([np.mean(v) for v in usable], dtype=float)

        # Observed population variance of profile-specific TTB-adherence
        # rates. Under TTB all profiles have the same latent rate, but its
        # finite-sample value is positive because of response noise.
        observed_profile_variance = float(np.var(means, ddof=0))

        # For Bernoulli observations, p_hat*(1-p_hat)/(n-1) is an unbiased
        # estimator of Var(p_hat). Subtract the exact contribution of these
        # estimation errors to the population variance across k profiles.
        mean_sampling_variances = []
        for v, p_hat in zip(usable, means):
            n = len(v)
            mean_sampling_variances.append(
                float(p_hat * (1.0 - p_hat) / float(n - 1))
            )
        noise_bias = (1.0 - 1.0 / float(k)) * float(
            np.mean(mean_sampling_variances)
        )

        return observed_profile_variance - noise_bias

    if len(data) == 0:
        return 0.0
    if "subject_id" in data.columns:
        scores = [
            subject_score(g)
            for _, g in data.groupby("subject_id", sort=False)
        ]
        return float(np.mean(scores)) if scores else 0.0
    return subject_score(data)

```

**Observed (real) value:** 0.0070 (var=0.0002)
**Candidate trajectory (this loop):**
  - iter 1: 0.0218 (var=0.0003) (Δ vs real +0.0148)
  - iter 2: 0.0185 (var=0.0003) (Δ vs real +0.0115)
  - iter 3: 0.0224 (var=0.0003) (Δ vs real +0.0154)
  - iter 4: 0.0159 (var=0.0003) (Δ vs real +0.0089)
  - iter 5: 0.0143 (var=0.0002) (Δ vs real +0.0073)
  - iter 6: 0.0218 (var=0.0004) (Δ vs real +0.0148)
  - iter 7: 0.0129 (var=0.0002) (Δ vs real +0.0059)
  - iter 8: 0.0165 (var=0.0002) (Δ vs real +0.0095)
  - iter 9 (current): 0.0094 (var=0.0001) (Δ vs real +0.0024)
**Other theories' values on this metric (for reference):**
- pi_5: 0.0058 (var=0.0001)
- pi_1: 0.0002 (var=0.0000)
- pi_2: 0.0206 (var=0.0003)
- pi_3: 0.0317 (var=0.0005)
- pi_4: 0.0000 (var=0.0000)
- pi_6: 0.0018 (var=0.0001)
- pi_7: 0.0062 (var=0.0001)
- pi_8: 0.0072 (var=0.0002)
- pi_9: 0.0115 (var=0.0002)
- pi_10: -0.0002 (var=0.0000)
- pi_11: 0.0095 (var=0.0001)

### Experiment 9
**Design**
  A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]
  A=[0, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if data is None or len(data) == 0:
        return 0.5

    def subject_score(subj):
        anchor_choices = []

        for _, row in subj.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float)
            b = np.asarray(row["option_b_ratings"], dtype=float)
            if a.size < 5 or b.size < 5:
                continue

            directions = np.sign(b - a)
            active = set(np.flatnonzero(directions != 0).tolist())

            # The two isolated-anchor target classes and their A/B reversals.
            is_outer_target = active == {0, 3, 4}
            is_middle_target = active == {1, 2, 4}
            if not (is_outer_target or is_middle_target):
                continue

            lower = [0, 3] if is_outer_target else [1, 2]
            coalition_is_concordant = directions[lower[0]] == directions[lower[1]]
            coalition_opposes_anchor = directions[lower[0]] == -directions[4]
            if not (coalition_is_concordant and coalition_opposes_anchor):
                continue

            anchor_response = 1 if directions[4] > 0 else 0
            anchor_choices.append(float(int(row["response"]) == anchor_response))

        if len(anchor_choices) == 0:
            return np.nan
        return float(np.mean(anchor_choices))

    if "subject_id" in data.columns:
        scores = [subject_score(subj) for _, subj in data.groupby("subject_id", sort=False)]
        scores = [x for x in scores if np.isfinite(x)]
        return float(np.mean(scores)) if len(scores) > 0 else 0.5

    score = subject_score(data)
    return float(score) if np.isfinite(score) else 0.5

```

**Observed (real) value:** 0.7250 (var=0.0723)
**Candidate trajectory (this loop):**
  - iter 1: 0.3006 (var=0.0148) (Δ vs real -0.4244)
  - iter 2: 0.4825 (var=0.0168) (Δ vs real -0.2425)
  - iter 3: 0.4719 (var=0.0158) (Δ vs real -0.2531)
  - iter 4: 0.4606 (var=0.0169) (Δ vs real -0.2644)
  - iter 5: 0.8263 (var=0.0059) (Δ vs real +0.1013)
  - iter 6: 0.4656 (var=0.0100) (Δ vs real -0.2594)
  - iter 7: 0.4531 (var=0.0175) (Δ vs real -0.2719)
  - iter 8: 0.4537 (var=0.0136) (Δ vs real -0.2712)
  - iter 9 (current): 0.4562 (var=0.0105) (Δ vs real -0.2687)
**Other theories' values on this metric (for reference):**
- pi_6: 0.8094 (var=0.0121)
- pi_5: 0.5687 (var=0.0166)
- pi_1: 0.8600 (var=0.0122)
- pi_2: 0.1737 (var=0.0171)
- pi_3: 0.2675 (var=0.0088)
- pi_4: 0.9844 (var=0.0007)
- pi_7: 0.7512 (var=0.0113)
- pi_8: 0.5850 (var=0.0117)
- pi_9: 0.7538 (var=0.0365)
- pi_10: 0.7481 (var=0.1072)
- pi_11: 0.4437 (var=0.0425)

### Experiment 10
**Design**
  A=[0, 0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def subject_score(subj):
        followed = []

        for _, row in subj.iterrows():
            a = np.asarray(row['option_a_ratings'], dtype=int)
            b = np.asarray(row['option_b_ratings'], dtype=int)
            if a.shape[0] != 6 or b.shape[0] != 6:
                continue

            directions = np.sign(b - a)

            # Diagnostic compact opposition: Experts 3 and 5 oppose the
            # unique validity anchor (trial-pair 1 and its reversal).
            compact_diagnostic = (
                np.array_equal(np.abs(directions),
                               np.array([1, 0, 1, 0, 1, 0]))
                and directions[2] == directions[4]
                and directions[0] == -directions[2]
            )

            # Singleton diagnostic dissent: Expert 3 alone opposes all five
            # other discriminating experts (trial-pair 3 and its reversal).
            singleton_diagnostic = (
                np.all(np.abs(directions) == 1)
                and directions[2] == -directions[0]
                and directions[1] == directions[0]
                and directions[3] == directions[0]
                and directions[4] == directions[0]
                and directions[5] == directions[0]
            )

            if not (compact_diagnostic or singleton_diagnostic):
                continue

            anchor_response = 1 if directions[0] > 0 else 0
            followed.append(float(int(row['response']) == anchor_response))

        if len(followed) == 0:
            return np.nan
        return float(np.mean(followed))

    scores = []
    for _, subj in data.groupby('subject_id', sort=False):
        value = subject_score(subj)
        if np.isfinite(value):
            scores.append(value)

    if len(scores) == 0:
        return np.nan
    return float(np.mean(scores))
```

**Observed (real) value:** 0.4462 (var=0.0224)
**Candidate trajectory (this loop):**
  - iter 1: 0.5706 (var=0.0061) (Δ vs real +0.1244)
  - iter 2: 0.6719 (var=0.0077) (Δ vs real +0.2256)
  - iter 3: 0.6663 (var=0.0117) (Δ vs real +0.2200)
  - iter 4: 0.6362 (var=0.0081) (Δ vs real +0.1900)
  - iter 5: 0.8325 (var=0.0105) (Δ vs real +0.3863)
  - iter 6: 0.6519 (var=0.0099) (Δ vs real +0.2056)
  - iter 7: 0.6538 (var=0.0086) (Δ vs real +0.2075)
  - iter 8: 0.6431 (var=0.0068) (Δ vs real +0.1969)
  - iter 9 (current): 0.6512 (var=0.0050) (Δ vs real +0.2050)
**Other theories' values on this metric (for reference):**
- pi_5: 0.4656 (var=0.0070)
- pi_6: 0.7306 (var=0.0084)
- pi_1: 0.8250 (var=0.0150)
- pi_2: 0.5275 (var=0.0076)
- pi_3: 0.7612 (var=0.0061)
- pi_4: 0.9844 (var=0.0006)
- pi_7: 0.4644 (var=0.0059)
- pi_8: 0.6150 (var=0.0107)
- pi_9: 0.5681 (var=0.0203)
- pi_10: 0.6106 (var=0.0980)
- pi_11: 0.6125 (var=0.0191)

### Experiment 11
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    high_count_anchor_choices = []
    two_opponent_anchor_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.size < 8 or b.size < 8:
            continue

        directions = np.sign(b - a)
        anchor_direction = int(directions[7])
        if anchor_direction == 0:
            continue

        # Retain isolated-expert-8 profiles and count cues opposing it.
        other_directions = directions[:7]
        if np.any(other_directions == anchor_direction):
            continue
        opponent_count = int(np.sum(other_directions == -anchor_direction))

        anchor_response = 1 if anchor_direction > 0 else 0
        chose_anchor = float(int(row["response"]) == anchor_response)

        if opponent_count == 2:
            two_opponent_anchor_choices.append(chose_anchor)
        elif opponent_count >= 3:
            high_count_anchor_choices.append(chose_anchor)

    if not high_count_anchor_choices or not two_opponent_anchor_choices:
        return float("nan")

    return float(
        np.mean(high_count_anchor_choices)
        - np.mean(two_opponent_anchor_choices)
    )
```

**Observed (real) value:** 0.1093 (var=0.0652)
**Candidate trajectory (this loop):**
  - iter 1: 0.0630 (var=0.0087) (Δ vs real -0.0463)
  - iter 2: 0.0525 (var=0.0062) (Δ vs real -0.0568)
  - iter 3: 0.3460 (var=0.0177) (Δ vs real +0.2367)
  - iter 4: 0.2543 (var=0.0082) (Δ vs real +0.1450)
  - iter 5: 0.1768 (var=0.0119) (Δ vs real +0.0675)
  - iter 6: 0.2758 (var=0.0115) (Δ vs real +0.1665)
  - iter 7: 0.2438 (var=0.0102) (Δ vs real +0.1345)
  - iter 8: 0.2733 (var=0.0181) (Δ vs real +0.1640)
  - iter 9 (current): 0.2457 (var=0.0158) (Δ vs real +0.1363)
**Other theories' values on this metric (for reference):**
- pi_7: 0.6688 (var=0.0308)
- pi_5: 0.0092 (var=0.0107)
- pi_1: 0.0137 (var=0.0129)
- pi_2: -0.0240 (var=0.0079)
- pi_3: -0.0033 (var=0.0095)
- pi_4: -0.0368 (var=0.0015)
- pi_6: -0.0085 (var=0.0107)
- pi_8: 0.1107 (var=0.0194)
- pi_9: 0.0435 (var=0.0122)
- pi_10: 0.1185 (var=0.1286)
- pi_11: -0.0005 (var=0.0037)

### Experiment 12
**Design**
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 0, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 0, 1, 0]  B=[1, 0, 1, 0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1, 1, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 1, 1, 0]  B=[0, 1, 0, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Expert-6 activation effect on choosing the Expert-4 anchor option."""
    if len(data) == 0:
        return float("nan")

    anchor_choices = []
    expert6_on = []

    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"],
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)

        # Expert 4 (zero-based index 3) defines the designated anchor option.
        if a[3] > b[3]:
            anchor_response = 0
        elif b[3] > a[3]:
            anchor_response = 1
        else:
            continue

        anchor_choices.append(float(int(response) == anchor_response))
        expert6_on.append(bool(a[5] != b[5]))

    anchor_choices = np.asarray(anchor_choices, dtype=float)
    expert6_on = np.asarray(expert6_on, dtype=bool)

    if not np.any(expert6_on) or not np.any(~expert6_on):
        return float("nan")

    return float(
        np.mean(anchor_choices[expert6_on])
        - np.mean(anchor_choices[~expert6_on])
    )
```

**Observed (real) value:** 0.3533 (var=0.0427)
**Candidate trajectory (this loop):**
  - iter 1: 0.4142 (var=0.0076) (Δ vs real +0.0608)
  - iter 2: 0.3563 (var=0.0110) (Δ vs real +0.0029)
  - iter 3: 0.2242 (var=0.0175) (Δ vs real -0.1292)
  - iter 4: 0.3237 (var=0.0158) (Δ vs real -0.0296)
  - iter 5: 0.2779 (var=0.0108) (Δ vs real -0.0754)
  - iter 6: 0.3033 (var=0.0150) (Δ vs real -0.0500)
  - iter 7: 0.2854 (var=0.0241) (Δ vs real -0.0679)
  - iter 8: 0.3271 (var=0.0077) (Δ vs real -0.0263)
  - iter 9 (current): 0.2996 (var=0.0096) (Δ vs real -0.0538)
**Other theories' values on this metric (for reference):**
- pi_5: -0.0208 (var=0.0079)
- pi_7: 0.2908 (var=0.0147)
- pi_1: 0.0083 (var=0.0081)
- pi_2: -0.3271 (var=0.0169)
- pi_3: 0.0450 (var=0.0087)
- pi_4: -0.5246 (var=0.0081)
- pi_6: 0.2700 (var=0.0168)
- pi_8: 0.3488 (var=0.0128)
- pi_9: 0.1708 (var=0.0122)
- pi_10: 0.3358 (var=0.1291)
- pi_11: 0.3904 (var=0.0106)

### Experiment 13
**Design**
  A=[0, 0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    anchor_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.size <= 5 or b.size <= 5:
            continue

        # Expert 6 (index 5) is the unique 95%-validity anchor.
        anchor_direction = np.sign(b[5] - a[5])
        if anchor_direction == 0:
            continue

        response = int(row["response"])
        chose_anchor = (
            (anchor_direction > 0 and response == 1)
            or (anchor_direction < 0 and response == 0)
        )
        anchor_choices.append(float(chose_anchor))

    if len(anchor_choices) == 0:
        return 0.5
    return float(np.mean(anchor_choices))
```

**Observed (real) value:** 0.3275 (var=0.0184)
**Candidate trajectory (this loop):**
  - iter 1: 0.3681 (var=0.0068) (Δ vs real +0.0406)
  - iter 2: 0.4585 (var=0.0053) (Δ vs real +0.1310)
  - iter 3: 0.3948 (var=0.0095) (Δ vs real +0.0673)
  - iter 4: 0.4598 (var=0.0064) (Δ vs real +0.1323)
  - iter 5: 0.4758 (var=0.0066) (Δ vs real +0.1483)
  - iter 6: 0.4629 (var=0.0070) (Δ vs real +0.1354)
  - iter 7: 0.5252 (var=0.0025) (Δ vs real +0.1977)
  - iter 8: 0.4721 (var=0.0040) (Δ vs real +0.1446)
  - iter 9 (current): 0.5219 (var=0.0025) (Δ vs real +0.1944)
**Other theories' values on this metric (for reference):**
- pi_8: 0.5044 (var=0.0026)
- pi_5: 0.3144 (var=0.0091)
- pi_1: 0.8431 (var=0.0109)
- pi_2: 0.4846 (var=0.0035)
- pi_3: 0.6629 (var=0.0048)
- pi_4: 0.9952 (var=0.0001)
- pi_6: 0.5269 (var=0.0032)
- pi_7: 0.4956 (var=0.0025)
- pi_9: 0.4288 (var=0.0042)
- pi_10: 0.2969 (var=0.0404)
- pi_11: 0.4875 (var=0.0152)

### Experiment 14
**Design**
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[1, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 0, 1]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[1, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 1]
  A=[1, 1, 1, 1, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 0, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 1, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[1, 1, 0, 1, 1, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def subject_score(df):
        diagnostic = []
        control = []

        for _, row in df.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float)
            b = np.asarray(row["option_b_ratings"], dtype=float)
            d = np.sign(b - a)
            active = set(np.flatnonzero(d != 0).tolist())

            if active == {0, 4} and d[0] == -d[4]:
                anchor_choice = float(int(row["response"]) == int(d[0] > 0))
                diagnostic.append(anchor_choice)
            elif active == {0, 1} and d[0] == -d[1]:
                anchor_choice = float(int(row["response"]) == int(d[0] > 0))
                control.append(anchor_choice)

        def late_minus_early(values):
            x = np.asarray(values, dtype=float)
            if x.size < 2:
                return np.nan
            half = x.size // 2
            return float(np.mean(x[-half:]) - np.mean(x[:half]))

        diag_change = late_minus_early(diagnostic)
        control_change = late_minus_early(control)
        if not np.isfinite(diag_change) or not np.isfinite(control_change):
            return np.nan
        return float(diag_change - control_change)

    if "subject_id" in data.columns and data["subject_id"].nunique() > 1:
        scores = [subject_score(g) for _, g in data.groupby("subject_id", sort=False)]
        scores = np.asarray(scores, dtype=float)
        scores = scores[np.isfinite(scores)]
        return float(np.mean(scores)) if scores.size else np.nan

    return subject_score(data)

```

**Observed (real) value:** 0.0200 (var=0.1946)
**Candidate trajectory (this loop):**
  - iter 1: -0.1050 (var=0.2602) (Δ vs real -0.1250)
  - iter 2: -0.0500 (var=0.1875) (Δ vs real -0.0700)
  - iter 3: -0.0300 (var=0.1666) (Δ vs real -0.0500)
  - iter 4: 0.0450 (var=0.1717) (Δ vs real +0.0250)
  - iter 5: 0.0250 (var=0.1131) (Δ vs real +0.0050)
  - iter 6: 0.0050 (var=0.1887) (Δ vs real -0.0150)
  - iter 7: -0.0250 (var=0.1906) (Δ vs real -0.0450)
  - iter 8: 0.1100 (var=0.2079) (Δ vs real +0.0900)
  - iter 9 (current): 0.0350 (var=0.2225) (Δ vs real +0.0150)
**Other theories' values on this metric (for reference):**
- pi_5: 0.3050 (var=0.1632)
- pi_8: -0.1050 (var=0.1227)
- pi_1: 0.0200 (var=0.1196)
- pi_2: -0.0600 (var=0.2064)
- pi_3: 0.1000 (var=0.2775)
- pi_4: 0.0050 (var=0.0062)
- pi_6: -0.0300 (var=0.2041)
- pi_7: -0.0400 (var=0.2234)
- pi_9: 0.0900 (var=0.2244)
- pi_10: 0.0200 (var=0.0696)
- pi_11: 0.1100 (var=0.1754)

### Experiment 15
**Design**
  A=[1, 0, 0, 0, 0, 1, 1, 0]  B=[0, 1, 1, 1, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 0, 1, 0, 1]  B=[1, 0, 0, 0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1, 0, 1]  B=[0, 1, 1, 1, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1, 0, 1]
  A=[1, 0, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1, 0, 1, 1]  B=[0, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if data is None or len(data) == 0:
        return float('nan')

    cue1_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        if a.ndim != 1 or b.ndim != 1 or a.size < 8 or b.size < 8:
            continue

        direction = np.sign(b - a)

        # Critical displays are those on which Experts 1 and 8 both
        # discriminate but recommend opposite products.
        if direction[0] == 0 or direction[7] == 0:
            continue
        if direction[0] == direction[7]:
            continue

        cue1_response = 1 if direction[0] > 0 else 0
        cue1_choices.append(float(int(row['response']) == cue1_response))

    if len(cue1_choices) == 0:
        return float('nan')

    return float(np.mean(cue1_choices))
```

**Observed (real) value:** 0.8475 (var=0.0129)
**Candidate trajectory (this loop):**
  - iter 1: 0.9246 (var=0.0040) (Δ vs real +0.0771)
  - iter 2: 0.9033 (var=0.0053) (Δ vs real +0.0558)
  - iter 3: 0.8829 (var=0.0061) (Δ vs real +0.0354)
  - iter 4: 0.9075 (var=0.0035) (Δ vs real +0.0600)
  - iter 5: 0.8625 (var=0.0061) (Δ vs real +0.0150)
  - iter 6: 0.9208 (var=0.0026) (Δ vs real +0.0733)
  - iter 7: 0.9087 (var=0.0040) (Δ vs real +0.0612)
  - iter 8: 0.9158 (var=0.0026) (Δ vs real +0.0683)
  - iter 9 (current): 0.9104 (var=0.0036) (Δ vs real +0.0629)
**Other theories' values on this metric (for reference):**
- pi_9: 0.8521 (var=0.0068)
- pi_5: 0.1762 (var=0.0116)
- pi_1: 0.1567 (var=0.0166)
- pi_2: 0.1533 (var=0.0142)
- pi_3: 0.7188 (var=0.0093)
- pi_4: 0.0096 (var=0.0002)
- pi_6: 0.4733 (var=0.0064)
- pi_7: 0.4275 (var=0.0052)
- pi_8: 0.6492 (var=0.0078)
- pi_10: 0.8496 (var=0.0084)
- pi_11: 0.8992 (var=0.0043)

### Experiment 16
**Design**
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 1, 0, 1]  B=[0, 1, 0, 1, 1, 0, 1]
  A=[0, 1, 0, 1, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 1, 0, 1]  B=[1, 1, 0, 1, 1, 0, 0]
  A=[1, 1, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def subject_score(df):
        early_opponent = []
        late_opponent = []
        novice_anchor = []
        mature_anchor = []
        exposure_count = {"early": 0, "late": 0}

        for _, row in df.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float)
            b = np.asarray(row["option_b_ratings"], dtype=float)
            if a.size < 7 or b.size < 7:
                continue

            active = set(np.flatnonzero(a != b).tolist())
            if active == {0, 3}:
                condition = "early"
            elif active == {3, 6}:
                condition = "late"
            else:
                continue

            if a[3] == b[3]:
                continue
            anchor_response = 0 if a[3] > b[3] else 1
            anchor_chosen = float(int(row["response"]) == anchor_response)

            rank = exposure_count[condition]
            exposure_count[condition] += 1

            if condition == "early":
                early_opponent.append(anchor_chosen)
            else:
                late_opponent.append(anchor_chosen)

            # Before four prior condition-specific observations, ADCG's
            # anti-concurrence estimate is still strongly pseudocount- and
            # confidence-limited. At eight or more it is comparatively mature.
            if rank < 4:
                novice_anchor.append(anchor_chosen)
            elif rank >= 8:
                mature_anchor.append(anchor_chosen)

        order_contrast = 0.0
        if early_opponent and late_opponent:
            order_contrast = float(np.mean(late_opponent) - np.mean(early_opponent))

        learning_contrast = 0.0
        if novice_anchor and mature_anchor:
            learning_contrast = float(np.mean(novice_anchor) - np.mean(mature_anchor))

        return order_contrast + 0.75 * learning_contrast

    if len(data) == 0:
        return 0.0
    if "subject_id" not in data.columns:
        return float(subject_score(data))

    scores = [subject_score(df) for _, df in data.groupby("subject_id", sort=False)]
    return float(np.mean(scores)) if scores else 0.0
```

**Observed (real) value:** 0.0263 (var=0.0299)
**Candidate trajectory (this loop):**
  - iter 1: 0.0155 (var=0.0339) (Δ vs real -0.0107)
  - iter 2: -0.0034 (var=0.0226) (Δ vs real -0.0297)
  - iter 3: 0.0085 (var=0.0207) (Δ vs real -0.0178)
  - iter 4: 0.0233 (var=0.0206) (Δ vs real -0.0030)
  - iter 5: -0.0171 (var=0.0129) (Δ vs real -0.0434)
  - iter 6: -0.0054 (var=0.0237) (Δ vs real -0.0317)
  - iter 7: 0.0332 (var=0.0226) (Δ vs real +0.0069)
  - iter 8: 0.0386 (var=0.0265) (Δ vs real +0.0123)
  - iter 9 (current): -0.0104 (var=0.0198) (Δ vs real -0.0367)
**Other theories' values on this metric (for reference):**
- pi_5: 0.1532 (var=0.0330)
- pi_9: 0.0106 (var=0.0320)
- pi_1: -0.0078 (var=0.0146)
- pi_2: 0.0011 (var=0.0351)
- pi_3: -0.2768 (var=0.0329)
- pi_4: 0.0055 (var=0.0008)
- pi_6: -0.0259 (var=0.0278)
- pi_7: -0.0118 (var=0.0177)
- pi_8: -0.0081 (var=0.0183)
- pi_10: 0.0006 (var=0.0082)
- pi_11: 0.0437 (var=0.0237)

### Experiment 17
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 1]  B=[0, 0, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Code every response as choosing the option endorsed by Expert 6,
    # the uniquely most-valid expert.
    records = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=int)
        b = np.asarray(row["option_b_ratings"], dtype=int)
        if a.size != 6 or b.size != 6:
            continue

        # Expert 6's anchor option is A when its A rating is 1, otherwise B.
        anchor_response = 0 if a[5] == 1 else 1
        anchor_choice = float(int(row["response"]) == anchor_response)

        # Cue identities opposing Expert 6 are invariant under A/B reversal.
        opponents = tuple(np.flatnonzero(a[:5] != a[5]).tolist())
        records.append((opponents, anchor_choice))

    if not records:
        return 0.0

    frame = pd.DataFrame(records, columns=["opponents", "anchor_choice"])

    # Compact, high-validity opposition profiles: {1,2} and {1,2,3}.
    compact_sets = {(0, 1), (0, 1, 2)}
    # High-multiplicity profiles: {1,2,3,4} and {1,2,3,4,5}.
    recovery_sets = {(0, 1, 2, 3), (0, 1, 2, 3, 4)}

    compact = frame.loc[
        frame["opponents"].apply(lambda x: x in compact_sets),
        "anchor_choice"
    ]
    recovery = frame.loc[
        frame["opponents"].apply(lambda x: x in recovery_sets),
        "anchor_choice"
    ]

    if len(compact) == 0 or len(recovery) == 0:
        return 0.0

    return float(recovery.mean() - compact.mean())
```

**Observed (real) value:** 0.5600 (var=0.0288)
**Candidate trajectory (this loop):**
  - iter 1: 0.0475 (var=0.0185) (Δ vs real -0.5125)
  - iter 2: 0.1438 (var=0.0133) (Δ vs real -0.4163)
  - iter 3: 0.3550 (var=0.0091) (Δ vs real -0.2050)
  - iter 4: 0.3962 (var=0.0064) (Δ vs real -0.1638)
  - iter 5: 0.3675 (var=0.0085) (Δ vs real -0.1925)
  - iter 6: 0.3950 (var=0.0093) (Δ vs real -0.1650)
  - iter 7: 0.2181 (var=0.0194) (Δ vs real -0.3419)
  - iter 8: 0.3881 (var=0.0095) (Δ vs real -0.1719)
  - iter 9 (current): 0.3794 (var=0.0096) (Δ vs real -0.1806)
**Other theories' values on this metric (for reference):**
- pi_9: 0.4031 (var=0.0261)
- pi_10: -0.0069 (var=0.0030)
- pi_1: -0.0006 (var=0.0092)
- pi_2: -0.5300 (var=0.0308)
- pi_3: -0.1925 (var=0.0135)
- pi_4: -0.0381 (var=0.0022)
- pi_5: 0.0344 (var=0.0146)
- pi_6: -0.0106 (var=0.0189)
- pi_7: 0.1544 (var=0.0156)
- pi_8: 0.0325 (var=0.0124)
- pi_11: 0.1838 (var=0.0296)

### Experiment 18
**Design**
  A=[1, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 1, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def subject_contrast(df):
        broad_choices = []
        nonbroad_choices = []

        for _, row in df.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=int)
            b = np.asarray(row["option_b_ratings"], dtype=int)
            direction = np.sign(b - a).astype(int)

            # Expert 4 (zero-based index 3) is the unique-validity anchor.
            anchor_direction = int(direction[3])
            if anchor_direction == 0:
                continue

            opposition = np.flatnonzero(direction == -anchor_direction)
            if opposition.size != 2:
                continue

            opposition_span = int(np.max(opposition) - np.min(opposition))
            is_broad = opposition_span >= 6

            response = int(row["response"])
            anchor_chosen = float(
                response == (1 if anchor_direction > 0 else 0)
            )

            if is_broad:
                broad_choices.append(anchor_chosen)
            else:
                nonbroad_choices.append(anchor_chosen)

        if len(broad_choices) == 0 or len(nonbroad_choices) == 0:
            return 0.0
        return float(np.mean(broad_choices) - np.mean(nonbroad_choices))

    if "subject_id" in data.columns:
        contrasts = [
            subject_contrast(df)
            for _, df in data.groupby("subject_id", sort=False)
        ]
        return float(np.mean(contrasts)) if len(contrasts) else 0.0

    return subject_contrast(data)

```

**Observed (real) value:** 0.0092 (var=0.0117)
**Candidate trajectory (this loop):**
  - iter 1: -0.0096 (var=0.0080) (Δ vs real -0.0188)
  - iter 2: -0.0087 (var=0.0073) (Δ vs real -0.0179)
  - iter 3: 0.0033 (var=0.0062) (Δ vs real -0.0058)
  - iter 4: -0.0025 (var=0.0073) (Δ vs real -0.0117)
  - iter 5: 0.0112 (var=0.0066) (Δ vs real +0.0021)
  - iter 6: -0.0063 (var=0.0085) (Δ vs real -0.0154)
  - iter 7: 0.0108 (var=0.0088) (Δ vs real +0.0017)
  - iter 8: 0.0217 (var=0.0062) (Δ vs real +0.0125)
  - iter 9 (current): -0.0083 (var=0.0093) (Δ vs real -0.0175)
**Other theories' values on this metric (for reference):**
- pi_10: 0.5554 (var=0.0321)
- pi_9: -0.0196 (var=0.0100)
- pi_1: -0.0033 (var=0.0050)
- pi_2: -0.0013 (var=0.0100)
- pi_3: -0.0792 (var=0.0037)
- pi_4: -0.0050 (var=0.0103)
- pi_5: 0.0125 (var=0.0050)
- pi_6: 0.0146 (var=0.0103)
- pi_7: 0.0029 (var=0.0081)
- pi_8: 0.0279 (var=0.0087)
- pi_11: -0.0288 (var=0.0121)

### Experiment 19
**Design**
  A=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Within-subject slope of anchor choice over opponent dispersion."""
    def subject_slope(df):
        cells = {}

        for _, row in df.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float)
            b = np.asarray(row["option_b_ratings"], dtype=float)
            a_support = np.flatnonzero(a > b)
            b_support = np.flatnonzero(b > a)

            # The coded anchor is the option supported by Experts 1 and 3
            # (zero-based indices 0 and 2).
            if (0 in a_support) and (2 in a_support):
                anchor_response = 0
                opponent = b_support
            elif (0 in b_support) and (2 in b_support):
                anchor_response = 1
                opponent = a_support
            else:
                continue

            if opponent.size == 0:
                continue

            span = int(np.max(opponent) - np.min(opponent) + 1)
            anchor_chosen = float(int(row["response"]) == anchor_response)
            key = (span, anchor_response)
            cells.setdefault(key, []).append(anchor_chosen)

        xs = []
        ys = []
        for span in (2, 4, 6):
            side_means = []
            for anchor_response in (0, 1):
                vals = cells.get((span, anchor_response), [])
                if len(vals) > 0:
                    side_means.append(float(np.mean(vals)))
            if len(side_means) == 2:
                # Opponent dispersion is the complement of compactness 2/span.
                xs.append(1.0 - 2.0 / float(span))
                # Equal weighting of reversal cells cancels stable A/B bias.
                ys.append(float(np.mean(side_means)))

        if len(xs) < 2:
            return np.nan

        x = np.asarray(xs, dtype=float)
        y = np.asarray(ys, dtype=float)
        xc = x - np.mean(x)
        denom = float(np.dot(xc, xc))
        if denom <= 0.0:
            return np.nan
        return float(np.dot(xc, y - np.mean(y)) / denom)

    slopes = []
    for _, subj_df in data.groupby("subject_id", sort=False):
        value = subject_slope(subj_df)
        if np.isfinite(value):
            slopes.append(value)

    if len(slopes) == 0:
        return float("nan")
    return float(np.mean(slopes))
```

**Observed (real) value:** -0.0776 (var=0.0965)
**Candidate trajectory (this loop):**
  - iter 1: 0.0189 (var=0.0092) (Δ vs real +0.0965)
  - iter 2: -0.0071 (var=0.0104) (Δ vs real +0.0705)
  - iter 3: 0.0025 (var=0.0194) (Δ vs real +0.0800)
  - iter 4: -0.0301 (var=0.0145) (Δ vs real +0.0475)
  - iter 5: 0.0319 (var=0.0160) (Δ vs real +0.1095)
  - iter 6: 0.0252 (var=0.0169) (Δ vs real +0.1028)
  - iter 7: 0.0180 (var=0.0116) (Δ vs real +0.0956)
  - iter 8: -0.0040 (var=0.0161) (Δ vs real +0.0736)
  - iter 9 (current): 0.0112 (var=0.0114) (Δ vs real +0.0888)
**Other theories' values on this metric (for reference):**
- pi_9: 0.0803 (var=0.0208)
- pi_11: -0.0257 (var=0.0116)
- pi_1: 0.0048 (var=0.0101)
- pi_2: -0.0518 (var=0.0294)
- pi_3: -0.0490 (var=0.0299)
- pi_4: -0.0072 (var=0.0274)
- pi_5: -0.0332 (var=0.0258)
- pi_6: 0.0255 (var=0.0150)
- pi_7: 0.0405 (var=0.0130)
- pi_8: -0.0222 (var=0.0289)
- pi_10: -0.0087 (var=0.0110)

### Experiment 20
**Design**
  A=[0, 0, 1, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 1, 0, 0, 1]  B=[0, 0, 1, 0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 1, 0, 0, 0, 1]
  A=[1, 0, 1, 1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 0]  B=[1, 0, 1, 0, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 1, 0, 0, 1]  B=[0, 0, 0, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 1, 0, 1, 1]  B=[1, 0, 0, 1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Expert-1 versus Expert-7 serial-leverage contrast, coded toward Expert 8."""
    if len(data) == 0:
        return 0.0

    a = np.vstack(data["option_a_ratings"].apply(
        lambda x: np.asarray(x, dtype=float)
    ).to_numpy())
    b = np.vstack(data["option_b_ratings"].apply(
        lambda x: np.asarray(x, dtype=float)
    ).to_numpy())
    response = data["response"].to_numpy(dtype=int)

    directions = np.sign(b - a)
    expert8_direction = directions[:, 7]

    # +1: Expert 1 supports Expert 8 and Expert 7 opposes it.
    # -1: Expert 7 supports Expert 8 and Expert 1 opposes it.
    expert1_alignment = directions[:, 0] * expert8_direction
    expert7_alignment = directions[:, 6] * expert8_direction
    condition = np.zeros(len(data), dtype=float)
    condition[(expert1_alignment == 1) & (expert7_alignment == -1)] = 1.0
    condition[(expert1_alignment == -1) & (expert7_alignment == 1)] = -1.0

    critical = (condition != 0) & (expert8_direction != 0)
    if not np.any(critical):
        return 0.0

    # Convert each response to +1 when it follows Expert 8 and -1 otherwise,
    # eliminating A/B response-side orientation from the contrast.
    chose_expert8 = np.where(
        expert8_direction > 0,
        response == 1,
        response == 0
    )
    coded_choice = 2.0 * chose_expert8.astype(float) - 1.0

    # With the design's balanced conditions, this equals
    # P(follow Expert 8 | Expert 1 aligned) minus
    # P(follow Expert 8 | Expert 7 aligned).
    return float(np.mean(condition[critical] * coded_choice[critical]))

```

**Observed (real) value:** 0.0350 (var=0.0461)
**Candidate trajectory (this loop):**
  - iter 1: 0.0183 (var=0.0065) (Δ vs real -0.0167)
  - iter 2: 0.0050 (var=0.0043) (Δ vs real -0.0300)
  - iter 3: 0.0013 (var=0.0085) (Δ vs real -0.0338)
  - iter 4: 0.0025 (var=0.0060) (Δ vs real -0.0325)
  - iter 5: -0.0025 (var=0.0120) (Δ vs real -0.0375)
  - iter 6: 0.0088 (var=0.0055) (Δ vs real -0.0263)
  - iter 7: -0.0029 (var=0.0056) (Δ vs real -0.0379)
  - iter 8: 0.0250 (var=0.0044) (Δ vs real -0.0100)
  - iter 9 (current): 0.0013 (var=0.0060) (Δ vs real -0.0338)
**Other theories' values on this metric (for reference):**
- pi_11: 0.7846 (var=0.0113)
- pi_9: 0.0262 (var=0.0116)
- pi_1: 0.0142 (var=0.0062)
- pi_2: 0.0142 (var=0.0076)
- pi_3: 0.2808 (var=0.0054)
- pi_4: -0.0200 (var=0.0080)
- pi_5: 0.0146 (var=0.0120)
- pi_6: 0.0000 (var=0.0079)
- pi_7: -0.0262 (var=0.0105)
- pi_8: 0.0146 (var=0.0113)
- pi_10: 0.5054 (var=0.3456)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed STGCC family: it has a narrow equality gate, no spatial-span term, static heterogeneous orientations, and configural nonmonotonic coalition competition. It also succeeds on the intended invariances and nulls, especially Experiments 7, 16, 18, and 20, and is reasonably close on Experiments 11–13. However, it does not yet capture the central cross-experiment coalition pattern. The largest failures are Experiment 6, where the predicted multiplicity contrast has the wrong sign (0.183 versus -0.545), and Experiment 17, where high-count anchor recovery is far too weak (0.048 versus 0.560). Experiment 9 likewise shows excessive coalition overturning of a strong anchor (anchor choice 0.301 versus 0.725). Conversely, highest-cue adherence remains too high in Experiment 2 (0.523 versus 0.324), showing that the model does not adequately distinguish credible close opposition from opposition to a clearly superior anchor. Experiment 4 is also a severe failure: it produces an 84% prevalence of a cue-alignment crossover that is absent in every real subject. The exact-tie/configural interaction is miscalibrated as well: Experiment 3 has the wrong sign (-0.137 versus 0.072), while tied-top adherence is somewhat excessive in Experiments 5 and 15. Across several experiments the simulated between-subject variance is much lower than observed, so the three orientations are not generating the relevant heterogeneity.
Rationale: Keep the STGCC mechanism family, but retune its existing gates rather than replacing it. The current count restoration is much too weak: strengthen restoration sharply at counts of roughly four or more and make its onset more discrete. That should address the jointly diagnostic failures in Experiments 6, 11, and especially 17. At the same time, make coalition challenge more gap-selective: combine a stronger challenge for genuinely close-validity opposition with a smaller challenge_gap_scale or sharper gap normalization, so close coalitions can lower top-cue adherence in Experiments 2, 10, and 13 without overturning the clearly superior anchor in Experiment 9. Experiment 4 indicates that coalition support is too sensitive to whether one particular strong lower cue joins the anchor; cap or normalize single-member leverage within coalition coherence so that Expert 1 alignment does not create a population-wide crossover. Finally, modestly reduce the effective tie-strength range and let the existing balance/coherence term contribute more in exactly tied displays; the present nearly lexicographic gate overpredicts Experiments 5 and 15 while yielding the wrong balanced-minus-imbalanced effect in Experiment 3. Preserve the very narrow tie tolerance and the complete absence of span terms. Broaden the correlated orientation effects—especially restoration strength and close-gap challenge—to better match the substantial human variance in Experiments 9, 11, 17, and 19 without introducing unrestricted subject-specific mechanisms.

**Outcome of your advice:** iter 1 candidate loss=0.2701 -> iter 2 candidate loss=0.2082 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The accepted revision is a meaningful improvement and remains faithful to STGCC: the exact-tie gate is narrow, spatial span is absent, orientations are static, and coalition competition is configural and nonmonotonic. The changes successfully repaired Experiment 4 (0.84 to 0.04), improved the strong-anchor result in Experiment 9, preserved the nulls in Experiments 7, 18, and 20, and now closely fit Experiments 1, 5, and 12. Nevertheless, the central multiplicity pattern remains substantially wrong. Experiment 6 still has the opposite sign and a very large error (0.202 versus -0.545), while Experiment 17 still shows far too little high-count recovery (0.144 versus 0.560). The sharper restoration gate therefore improved recovery but remains effectively suppressed, likely because restoration is multiplicatively conditional on the validity-gap gate. The revised challenge calibration also moved several close-opposition results in the wrong direction: highest-cue adherence is now much too high in Experiments 2 (0.637 versus 0.324), 10 (0.672 versus 0.446), and 13 (0.459 versus 0.328). Experiment 3 retains the wrong interaction sign (-0.123 versus 0.072). Human heterogeneity also remains understated in important cases, particularly Experiments 9, 11, 17, and 19.
Rationale: Continue within STGCC and retain the successful cue-cap/saturated-competition repair for Experiment 4, the narrow serial equality tolerance, and the absence of span terms. The previous recommendation was accepted overall, so its high-count-restoration direction was productive, but simply increasing restoration strength further is unlikely to suffice. Instead, fix the conjunction of the restoration gates: at sufficiently high opponent counts, broad-participation recovery should have a positive floor or partially override the gap gate rather than being the product of count_gate and gap_gate. This should produce the large negative contrast in Experiment 6 and stronger positive recovery in Experiment 17 without affecting the two-opponent strong-anchor case in Experiment 9. For challenge, do not narrow the exponential gap scale further; that accepted change coincided with worse fits in Experiments 2, 10, and 13. Use a plateau-then-sharp-cutoff credibility normalization, or an intermediate gap scale between the previous and current settings, so genuinely close coalitions receive nearly full challenge while clearly separated anchors remain protected. Experiment 3 suggests a sign/allocation error in the balance modulation: increasing the serial tie gate on balanced displays appears to suppress rather than promote Expert 5's winner. Let balance increase configural coalition influence, or reduce the serial mixture weight on balanced exact ties, while retaining strong categorical tie precedence overall. Finally, increase separation specifically between the configural-challenge and restoration orientations, rather than broadly widening every parameter, to raise the currently deficient between-subject variance in recovery and anchor-choice metrics.

**Outcome of your advice:** iter 2 candidate loss=0.2082 -> iter 3 candidate loss=0.2175 -> the gate marked it REJECTED.

### Iteration 3
Verdict: regenerate
Interpretation: The current candidate remains faithful to the prescribed STGCC family: serial precedence is narrowly tie-gated, coalition competition is static and configural, subject orientations are stable, and no spatial-span variable is present. It also preserves the exact fits or near-nulls in Experiments 1, 16, 18, and 20. However, this iteration was rejected by the accept gate and is worse overall than the accepted iteration 2 base. Its positive restoration floor improved the direction of Experiment 6 and raised recovery in Experiment 17, but did so too indiscriminately: Experiment 11 overshot sharply from 0.053 to 0.346 versus 0.109, while Experiment 6 remained far from the required -0.545 contrast and Experiment 17 remained below 0.560. The plateau challenge and balance changes also damaged several previously good fits. Experiment 5 fell from 0.780 to 0.687 versus 0.790, Experiment 12 fell from 0.356 to 0.224 versus 0.353, Experiment 3 became more negative rather than changing sign, and the null in Experiment 7 deteriorated. Although Experiments 2 and 13 improved, top-anchor adherence is still excessive there and in Experiment 10. Experiment 9 continues to show insufficient protection of a clearly superior anchor. Thus the rejected revision trades broad deterioration for only partial recovery on the principal multiplicity effects.
Rationale: Build the next revision on the unchanged accepted iteration 2 base. The previous recommendation to add a global positive restoration floor, replace credibility with a broad plateau, and subtract balance from the tie mixture was rejected, so do not repeat those changes. Instead, refine the existing nonmonotonic count mechanism with a broad-participation gate based jointly on opponent count and opponent_count / n_features. Restoration should remain weak for counts 2–3 and for incomplete participation, then rise steeply around four opponents or roughly 60–70% display participation and saturate. This can distinguish the count-4/5 recovery cells in Experiment 17 and the count-5/7 cells in Experiment 6 from the lower-count cells pooled into Experiment 11, avoiding the current Experiment 11 overshoot without invoking spatial span. Add a separately bounded singleton-challenge multiplier under the accepted gap-credibility function rather than globally increasing challenge or changing its normalization. Stronger close-gap singleton challenge should lower lone-opponent anchor adherence in Experiment 6 and help Experiments 2 and 10, while the existing gap cutoff should continue protecting the clearly superior two-opponent anchor in Experiment 9. Keep intermediate challenge independently calibrated so Experiment 13 can remain low without undoing Experiment 12. For Experiment 3, leave the categorical tie weight independent of display balance rather than reversing its balance slope again; place any balance effect in context-normalized coalition coherence or precision. Preserve the iteration 2 tie-strength range, which already fit Experiment 5. Finally, Experiment 19 should be addressed through active cue identities and validities, not span: modestly soften the hard 1.25 member cap for two-member opponent coalitions or use a smooth reliability cap, while retaining the saturated multi-coalition safeguard that repaired Experiment 4. These are localized in-family adjustments and should be varied by the existing challenge versus restoration orientations to raise the currently understated between-subject variance.

**Outcome of your advice:** iter 3 candidate loss=0.2175 -> iter 4 candidate loss=0.1785 -> the gate marked it ACCEPTED.

### Iteration 4
Verdict: regenerate
Interpretation: The accepted iteration is a genuine improvement and remains faithful to the prescribed STGCC family. The participation gate and softened two-member cap improved Experiments 4, 12, 14, 17, and 19 while preserving the span nulls in Experiments 18 and 20, the history null in Experiment 16, and the exact fit in Experiment 1. Nevertheless, the model is not yet experiment-invariant. The largest failure remains Experiment 6: the predicted lone-versus-multiple-opponent contrast is essentially zero (-0.006) rather than strongly negative (-0.545). High-count recovery is still too weak in Experiment 17 (0.396 versus 0.560), yet it is already too strong in Experiment 11 (0.254 versus 0.109), showing that the new restoration gate is not sufficiently selective across configurations. Coalition challenge is also too weak for close or credible opposition: highest-cue adherence is much too high in Experiments 2 (0.636 versus 0.324), 10 (0.636 versus 0.446), and 13 (0.460 versus 0.328). At the same time, the clearly superior isolated anchor is insufficiently protected in Experiment 9 (0.461 versus 0.725). Experiment 3 still has the wrong balance-effect sign (-0.130 versus 0.072), so positive balance precision is amplifying the wrong configural winner rather than revealing tied-cue primacy. Smaller residual misses occur in Experiments 5 and 15. Between-subject variance remains markedly understated in Experiments 9, 11, 17, and 19, indicating that the three orientations differ less behaviorally than human subjects do.
Rationale: Retain the accepted participation gate, narrow equality tolerance, absence of span terms, saturated safeguard for Experiment 4, and active-identity treatment that improved Experiment 19. The previous localized advice was accepted, so these components should be extended rather than reversed. First, target Experiment 6 primarily through the singleton channel rather than by globally increasing broad restoration, which would worsen the existing Experiment 11 overshoot. Increase the effective singleton challenge or apply it as a bounded shift in normalized coalition margin instead of only a multiplicative support bonus. This should strongly reduce lone-opponent anchor adherence and also help Experiments 2 and 10. Strengthen intermediate challenge more modestly for the configural orientation to lower excessive anchor adherence in Experiment 13. Second, add a steep sparse-count, clear-validity-gap anchor-protection term within the existing restoration mechanism. It should operate for one or two opponents only when the communicated gap is clearly large, raising Experiment 9 without protecting the near-gap opposition in Experiments 2, 10, and 13. Third, slightly reduce the broad-participation restoration contribution that is producing the Experiment 11 overshoot, while sharpening its upper-count saturation rather than raising its global strength; retain a stronger restoration multiplier specifically for the anchor-restoration orientation so Experiment 17 can continue improving and human heterogeneity increases. Finally, do not alter tie tolerance or make the serial mixture directly balance-dependent. The current positive balance-precision term worsens Experiment 3; reduce it toward zero and let balance compress the configural core margin under tied-top conflict, allowing the unchanged categorical serial component to dominate balanced displays. This is a coalition-normalization adjustment, not a new serial mechanism, and should improve Experiment 3 without repeating the previously rejected negative tie-balance slope or sacrificing the reasonable tied-top fits in Experiments 5 and 15.

**Outcome of your advice:** iter 4 candidate loss=0.1785 -> iter 5 candidate loss=0.2078 -> the gate marked it REJECTED.

### Iteration 5
Verdict: regenerate
Interpretation: The candidate remains structurally faithful to STGCC: it is history-invariant, contains no spatial-span term, uses communicated-validity-weighted configural coalitions, and restricts serial precedence to an extremely narrow top-tie gate. It preserves strong fits for Experiments 1, 7, 14, 15, and 18 and the intended near-null in Experiment 20. However, this iteration was rejected by the accept gate and regresses from the accepted iteration-4 model on several central diagnostics. The new singleton shift and sparse protection do not work selectively: Experiment 6 moves from approximately zero to the wrong positive direction (0.128 versus -0.545), Experiment 10 rises sharply to 0.833 versus 0.446, and Experiment 2 remains much too anchor-consistent at 0.650 versus 0.324. Sparse protection repairs but overshoots Experiment 9 (0.826 versus 0.725) while likely counteracting challenge in other one- and two-opponent displays. The revision also reintroduces the crossover in Experiment 4, weakens recovery in Experiment 17, reverses the improvement in Experiment 19, and worsens Experiment 16. Although Experiment 11 improves and Experiment 3 becomes slightly less negative, neither improvement compensates for the broader deterioration. The principal unresolved empirical pattern is still the strong lone-versus-multiple reversal in Experiment 6 alongside moderate recovery in Experiment 11 and strong upper-count recovery in Experiment 17. Simulated between-subject variance also remains much too small in Experiments 9, 11, 17, and 19.
Rationale: Build on the unchanged accepted iteration-4 base. The latest advice to add a normalized singleton-margin shift, sparse protection over one or two opponents, and tied-display margin compression was rejected by the gate, so do not repeat that bundle or simply increase its strengths. Instead, tune the existing nonmonotonic count/participation surface more selectively. Move the general restoration count threshold upward from roughly 3.5 toward the upper-count region, around 4.5–5 opponents, and make the transition sharp. This should reduce the Experiment-11 overshoot, because its high-count pool begins at three, while preferentially restoring the count-5/7 cells that can drive Experiment 6 negative. Preserve Experiment 17 through its already accepted capacity-bounded participation interaction: in six-cue displays, high participation can lower the effective threshold for the count-4/5 recovery cells without globally restoring count-3 configurations. Fit this as a count-by-participation interaction rather than another additive restoration bonus. Keep the accepted exponential gap credibility and the iteration-4 two-member cap; the rejected singleton shift and broad sparse-protection gate should be omitted. For Experiment 9, if protection is still needed, adjust only the existing gap dependence of restoration with a steeper, higher threshold and modest amplitude, so it applies to exceptionally separated isolated anchors rather than broadly to all one- or two-opponent cases such as Experiments 6 and 10. Retain the Experiment-4 saturated safeguard exactly, since modifying the surrounding challenge/protection balance allowed crossover prevalence to return. Finally, avoid another reversal of balance modulation. If tied primacy is tuned, use a narrow tie-strength range near the middle-to-upper part of the accepted range rather than balance-dependent compression; this may improve the still-negative Experiment-3 interaction and the underprediction in Experiment 5 while preserving the discrete equality gate. Maintain the absence of span terms and vary the sharpened upper-count threshold primarily across the existing restoration orientation to increase behavioral heterogeneity without adding a new mechanism family.

**Outcome of your advice:** iter 5 candidate loss=0.2078 -> iter 6 candidate loss=0.2090 -> the gate marked it REJECTED.

### Iteration 6
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed STGCC family: it is static and history-invariant, excludes spatial span, uses communicated-validity-weighted configural coalitions, and restricts serial precedence to an extremely narrow equality gate. It continues to fit Experiment 1 exactly and preserves useful nulls in Experiments 14, 18, and 20. Nevertheless, the current iteration was rejected by the accept gate and is worse than the accepted iteration-4 base. The raw-count threshold revision did not solve the central multiplicity problem: Experiment 6 remains essentially zero rather than strongly negative (-0.002 versus -0.545), Experiment 17 remains under-recovered (0.395 versus 0.560), and Experiment 11 still over-recovers (0.276 versus 0.109). Thus raw opponent count plus a binary six-cue capacity relief cannot distinguish the relevant kinds of broad opposition. Other systematic failures remain: anchor adherence is too high in Experiments 2, 10, and 13, but too low in Experiment 9; Experiment 3 retains the wrong sign; and Experiment 19 has reverted to the wrong slope. Between-subject variance is also substantially too low in the recovery and anchor-choice diagnostics.
Rationale: Build on the unchanged accepted iteration-4 base. The previous recommendation to raise the raw restoration threshold toward 4.5–5 was rejected, so do not repeat that adjustment or further tune capacity_count_relief. Instead, revise the normalization inside the existing STGCC competition surface: define challenge and restoration over validity-weighted effective opponent multiplicity and credible participation, rather than raw opponent_count. Effective multiplicity can use the coalition's existing reliability concentration or effective_n, while credible participation should weight display share by opponent reliability. A smooth nonmonotonic surface over these quantities can distinguish broad credible opposition in Experiments 6 and 17 from the weaker cues entering the high-count pool of Experiment 11, without introducing spatial span or a new mechanism family. Replace the discontinuous n_features <= 6 capacity gate with a smooth interaction between credible participation and effective multiplicity.

Separately, the simultaneous over-adherence in Experiments 2, 10, and 13 and under-adherence in Experiment 9 indicates inadequate gap discrimination. Keep exponential-style credibility but consider a power-exponential or log-odds-gap normalization: near-zero communicated-validity gaps should retain strong challenge, whereas clearly separated anchors should lose challenge sharply. This is preferable to the previously rejected broad plateau or sparse anchor-protection bonus. Couple the bounded singleton channel to this normalized credibility rather than adding another margin shift; the prior direct singleton-shift bundle was rejected.

Preserve the iteration-4 saturated safeguard for Experiment 4 and the absence of span terms. For exact ties, do not add a direct balance-dependent serial weight. Instead, make the three orientations more distinct in categorical tie use—high tie mixture for the lexicographic type and low tie mixture for the configural type—while balancing their population mixture to retain Experiment 5. This should increase deficient heterogeneity and allow tied-cue primacy to emerge without globally amplifying the anti-serial configural margin that keeps Experiment 3 negative.

**Outcome of your advice:** iter 6 candidate loss=0.2090 -> iter 7 candidate loss=0.2092 -> the gate marked it REJECTED.

### Iteration 7
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed STGCC family: it is static, validity-weighted, configural, narrowly tie-gated, heterogeneous, and free of spatial-span terms. It preserves exact or near fits in Experiments 1, 4, 7, 16, and 18. Nevertheless, it was rejected by the accept gate and is materially worse than the accepted iteration-4 base. The latest effective-multiplicity and reliability-weighted-participation revision failed to improve the defining coalition pattern. Experiment 6 remains in the wrong direction (0.078 versus -0.545), Experiment 17 recovery deteriorates sharply (0.218 versus 0.560), and Experiment 11 still overstates recovery (0.244 versus 0.109). Gap discrimination also remains wrong: anchor adherence is excessive in Experiments 2, 10, and 13, but deficient in Experiment 9. Experiment 3 still has the wrong interaction sign, and Experiment 19 has reverted from the iteration-4 improvement to the wrong slope. The model also produces much less between-subject variance than observed in key anchor and recovery experiments. Thus the new normalization has made credible broad opposition harder—not easier—to identify, probably because opponent mass is divided by total feature capacity while each centered weight is at most one, making the high participation gate difficult to open, especially for lower-validity opponents.
Rationale: Build again on the unchanged accepted iteration-4 base. The most recent advice to replace raw multiplicity with reliability-weighted effective multiplicity and participation was rejected, so do not repeat or intensify that normalization. In particular, omit opponent_mass / n_features as the restoration gate: it conflates credibility with display occupancy and appears responsible for the large loss of Experiment-17 recovery. Retain the iteration-4 count-by-participation surface, saturated Experiment-4 safeguard, two-member reliability cap, narrow tie tolerance, and no-span architecture.

Use a smaller, targeted extension of the singleton channel that previously helped produce the accepted iteration-4 improvement. Introduce a separate singleton credibility scale under the existing multiplicative challenge—not the rejected direct margin shift or sparse anchor-protection bonus. Widen that singleton-specific gap scale modestly so isolated, relatively close dissent can strongly reduce anchor adherence in Experiment 6 and contribute to Experiments 2 and 10, while leaving the intermediate-count challenge and broad restoration surfaces unchanged. Preserve a much sharper cutoff for clearly separated anchors so Experiment 9 is protected. This should be tuned as a one-dimensional change before modifying restoration again.

Do not pursue another global increase in restoration: iteration 4 already overpredicts Experiment 11 while underpredicting Experiment 17, and later threshold and effective-count revisions were both rejected. If a subsequent localized adjustment is necessary, alter only the accepted participation interaction so count-4/5 opposition occupying most of a six-cue display receives more restoration than count-3 opposition; do not redefine participation using reliability mass. Leave tie and balance parameters at their iteration-4 settings for this pass—the repeated balance and orientation revisions have not corrected Experiment 3 and have degraded Experiment 5. The immediate goal should be to beat the accepted loss through a minimal singleton-credibility adjustment rather than another multi-parameter redesign.

**Outcome of your advice:** iter 7 candidate loss=0.2092 -> iter 8 candidate loss=0.1914 -> the gate marked it REJECTED.

### Iteration 8 (most recent)
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed STGCC family: it is static and history-invariant, has an extremely narrow equality gate, contains no spatial-span term, and uses validity-weighted configural coalition competition with stable orientations. It continues to fit Experiment 1 exactly and is close on Experiments 4, 7, 12, 16, 18, and 20. The singleton-specific credibility scale produced the intended local movement in Experiment 6, from the accepted base’s -0.006 to -0.078, and modestly lowered Experiment-2 adherence. However, the candidate was rejected by the accept gate and remains worse than the accepted iteration-4 base. The defining multiplicity contrast in Experiment 6 is still far too weak relative to -0.545. More importantly, the broader pattern remains internally inconsistent: restoration is excessive in Experiment 11 (0.273 versus 0.109) but insufficient in Experiment 17 (0.388 versus 0.560), while a clearly superior sparse anchor remains underprotected in Experiment 9 (0.454 versus 0.725). Coalition overturning is too weak in Experiments 2, 10, and 13, where anchor adherence is respectively 0.607, 0.643, and 0.472 rather than 0.324, 0.446, and 0.328. Experiment 3 also retains the wrong sign (-0.125 versus 0.072), and Experiment 19’s negative dispersion slope remains much too shallow. Between-subject variance is substantially understated in the major anchor and recovery diagnostics. Thus the singleton edit helped one component but did not yield an experiment-invariant competition surface.
Rationale: Build on the unchanged accepted iteration-4 base. The latest recommendation to widen singleton credibility was rejected by the gate, so do not repeat or further enlarge singleton_gap_scale. Instead, make a focused revision to the existing count-by-participation restoration gate. In the accepted implementation, low participation can still produce substantial restoration through the gap_gate branch: count_gate * ((1-participation_gate) * gap_gate + participation_gate). This likely explains why Experiment 11’s heterogeneous count-3-and-above pool over-recovers even though the high-participation count-4/5 configurations in Experiment 17 remain under-recovered. Attenuate the low-participation gap branch and make restoration primarily conjunctive in high count and raw display participation, while retaining only a small bounded clear-gap contribution below that participation threshold. A moderately higher or sharper raw-participation threshold should preferentially restore count-4/5 opposition in six-cue Experiment 17 and the highest-count cells in Experiment 6, while reducing restoration among the count-3/4 cells entering Experiment 11. This is distinct from the previously rejected increase in the raw count threshold and from the rejected reliability-mass normalization.

For Experiment 9, do not compensate with another sparse protection bonus, which previously caused broad regression. Instead, tighten the existing multi-opponent challenge credibility specifically at clearly large communicated-validity gaps, leaving the baseline validity-weighted coalition comparison to protect the anchor. Keep strong challenge near zero gap so Experiments 2, 10, and 13 can move downward. Make this a sharp tail cutoff on the current credibility function rather than a broad plateau or a new margin shift. Preserve the iteration-4 two-member cap and saturated Experiment-4 safeguard.

Leave tie tolerance and direct balance modulation unchanged in this pass; repeated balance edits were rejected and did not repair Experiment 3. Also preserve the no-span architecture. Finally, implement the claimed correlated heterogeneity more faithfully by assigning orientation-conditioned, correlated restoration threshold/strength profiles rather than sampling all continuous ranges independently: restoration types should cross the high-participation threshold more readily, while configural types should retain stronger close-gap challenge. Keep these differences modest and theory-constrained, but large enough to raise the severely deficient variance in Experiments 9, 11, 17, and 19.

**Outcome of your advice:** iter 8 candidate loss=0.1914 -> CURRENT candidate loss=0.1897 -> the gate marked it REJECTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate remains faithful to the prescribed STGCC family: it is static and history-invariant, uses validity-weighted configural coalitions and stable orientations, restricts serial precedence to a very narrow equality gate, and contains no spatial-span term. It preserves useful fits or nulls in Experiments 1, 7, 8, 14, and 18. However, the current candidate was rejected by the accept gate and remains worse than the accepted iteration-4 base. The latest restoration and credibility revisions do not resolve the central pattern: Experiment 6 is near zero rather than strongly negative (0.031 versus -0.545), Experiment 17 still underestimates high-count recovery (0.379 versus 0.560), and Experiment 11 overestimates it (0.246 versus 0.109). The model also continues to underprotect the clearly superior anchor in Experiment 9 (0.456 versus 0.725), while producing excessive anchor adherence in Experiments 2, 10, and 13. Experiment 3 retains the wrong sign, and Experiment 19 now has the wrong slope. Several of these metrics also show substantially less between-subject variance than the real data. Thus the candidate captures the intended invariances but not an experiment-invariant calibration of anchor credibility, coalition challenge, and high-count restoration.",
  "verdict": "regenerate",
  "rationale": "Build on the unchanged accepted iteration-4 base. The most recent attempt to attenuate low-participation restoration and sharpen the challenge tail was rejected, so do not repeat that change, further raise the participation threshold, or further sharpen the power-exponential cutoff. Instead, address a different in-family limitation: the current reliability representation centers every display on its best active cue, forcing the best cue's weight to one and largely discarding its absolute communicated reliability. This makes a 95%-valid anchor and a much weaker best cue look similar whenever their relative within-display gaps are comparable, plausibly causing the incompatible results across Experiments 9, 11, and 17.\n\nAdd a bounded absolute anchor-confidence quantity derived from the communicated validity or log odds of the unique best cue. Use it within the existing anchor-competition mechanism, not as a new sparse-count bonus: absolute confidence should continuously strengthen broad-participation restoration and stabilize a clearly separated anchor, while its effect on challenge should approach zero at genuinely tiny gaps. A joint confidence term based on absolute reliability and relative separation can therefore raise Experiment 9 without suppressing near-gap challenge in Experiments 2, 10, and 13. Let the anchor-restoration orientation be most sensitive to this confidence signal and the configural orientation least sensitive, preserving the stipulated correlated heterogeneity and potentially increasing the deficient variance in Experiments 9, 11, and 17. Keep the accepted raw count-by-participation surface otherwise unchanged for this pass, so the effect of restoring absolute reliability can be evaluated cleanly.\n\nRetain the iteration-4 saturated safeguard for Experiment 4, the two-member cap, the narrow tie tolerance, and the complete absence of spatial-span terms. Do not revisit singleton credibility, raw-count thresholds, reliability-mass participation, or direct balance modulation; those directions have already been rejected. Experiment 3 remains an important failure, but repeated tie/balance edits have not helped, so it should not drive the next revision before the more basic loss of absolute anchor confidence is corrected."
}
```

## Usage

```json
{
  "input_tokens": 41643,
  "output_tokens": 3056
}
```
