# feedback_iter_00

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
Static Tie-Gated Configural Competition (STGCC) proposes that communicated validities establish a fixed reliability representation, but choices are made by competition between option-level coalitions rather than by adding independent cue votes. Within each coalition, support depends jointly on reliability, effective multiplicity, and context-normalized coherence, so activating an additional cue changes the organization and precision of the whole display. A uniquely strongest active cue serves as a provisional anchor. Credible opposition receives a nonmonotonic challenge gain: singleton dissent can overturn a weak or isolated anchor, intermediate-sized coalitions are maximally challenging, and sufficiently broad opposition can reactivate a clearly superior anchor through multiplicity-based restoration. Serial precedence is a separate categorical process. It is activated only when at least two active, conflicting cues are exactly equal or fall inside an extremely narrow subjective equality category at the highest validity level; a validity gap as large as 0.0052 is therefore outside the gate. Once activated, the earliest active member of the tied-top set supplies bounded lexicographic evidence. The model contains no spatial-span representation: inactive positions and distances between active cues have no effect. Stable heterogeneity is represented by three orientations—exact-tie lexicographic, configural-integration, and anchor-restoration—which jointly alter tie use, coalition challenge, restoration, and precision. Because subjects receive no correctness feedback, these orientations and validity representations remain fixed throughout the experiment.

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
    challenge_peak = float(parameters["challenge_peak"])
    challenge_width = float(parameters["challenge_width"])
    challenge_gap_scale = float(parameters["challenge_gap_scale"])
    restoration_strength = float(parameters["restoration_strength"])
    restoration_count = float(parameters["restoration_count"])
    restoration_slope = float(parameters["restoration_slope"])
    restoration_gap = float(parameters["restoration_gap"])
    restoration_gap_slope = float(parameters["restoration_gap_slope"])
    tie_tolerance = float(parameters["tie_tolerance"])
    tie_strength = float(parameters["tie_strength"])
    activation_precision = float(parameters["activation_precision"])
    balance_precision = float(parameters["balance_precision"])
    beta = float(parameters["beta"])
    lapse = float(parameters["lapse"])

    # Positive directions favor B and negative directions favor A.
    directions = np.sign(stim[1] - stim[0])
    active = np.flatnonzero(directions != 0)
    if active.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    # The three orientations induce correlated mechanistic differences rather
    # than independently enabling a collection of unrelated special cases.
    if orientation == 0:  # exact-tie lexicographic orientation
        type_tie = 1.18
        type_challenge = 0.88
        type_restoration = 0.90
        type_precision = 1.08
    elif orientation == 1:  # configural-integration orientation
        type_tie = 0.78
        type_challenge = 1.28
        type_restoration = 0.72
        type_precision = 0.94
    else:  # anchor-restoration orientation
        type_tie = 0.92
        type_challenge = 1.02
        type_restoration = 1.55
        type_precision = 1.02

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
    # Intermediate opposition is maximally diagnostic; very broad opposition
    # can instead restore a clearly superior isolated anchor.
    if unique_anchor and opponent_count > 0:
        strongest_opponent_v = float(np.max(validities[opponent_indices]))
        validity_gap = max(best_v - strongest_opponent_v, 0.0)
        credibility = np.exp(
            -validity_gap / max(challenge_gap_scale, 1e-8)
        )

        count_z = (
            float(opponent_count) - challenge_peak
        ) / max(challenge_width, 1e-8)
        intermediate_profile = np.exp(-0.5 * count_z * count_z)

        # Singleton dissent remains consequential, while the Gaussian term
        # allows two- or three-member coalitions to maximize the challenge.
        singleton_profile = float(opponent_count == 1)
        challenge_profile = max(
            intermediate_profile,
            0.58 * singleton_profile
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
        gap_gate = sigmoid(
            restoration_gap_slope * (validity_gap - restoration_gap)
        )
        isolated_gate = 1.0 / float(max(anchor_count, 1))
        restoration = (
            type_restoration
            * restoration_strength
            * count_gate
            * gap_gate
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

    if tied_conflict:
        first_tied_cue = int(np.min(categorized_top))
        serial_direction = float(directions[first_tied_cue])
        tie_gate = float(np.clip(
            type_tie * tie_strength,
            0.0,
            0.96
        ))
    else:
        serial_direction = 0.0
        tie_gate = 0.0

    choice_evidence = (
        (1.0 - tie_gate) * core_evidence
        + tie_gate * serial_direction
    )

    # Activation and balance alter response precision after coalition
    # organization. This permits positive activation effects without treating
    # an added cue as independent evidence or using cue-index span.
    activation = float(n_active) / float(n_active + 2)
    if n_a > 0 and n_b > 0:
        balance = 1.0 - abs(float(n_a - n_b)) / float(n_active)
    else:
        balance = 0.0
    precision_factor = (
        1.0
        + activation_precision * activation
        + balance_precision * balance
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
- validity_sensitivity: [0.05, 0.24]
- coalition_exponent: [1.05, 1.42]
- redundancy: [0.18, 0.52]
- coherence_gain: [0.20, 0.75]
- challenge_strength: [1.80, 4.20]
- challenge_peak: [1.80, 2.60]
- challenge_width: [0.65, 1.15]
- challenge_gap_scale: [0.025, 0.095]
- restoration_strength: [1.20, 3.80]
- restoration_count: [3.20, 4.10]
- restoration_slope: [5.0, 12.0]
- restoration_gap: [0.025, 0.065]
- restoration_gap_slope: [55.0, 130.0]
- tie_tolerance: [0.000001, 0.0015]
- tie_strength: [0.72, 0.91]
- activation_precision: [0.05, 0.32]
- balance_precision: [0.05, 0.38]
- beta: [1.55, 3.90]
- lapse: [0.0, 0.09]

`rationale`:
STGCC directly implements the arbiter's prescribed replacement family while avoiding the principal failures of continuous ambiguity and span-sensitive models. Its top-equality gate is genuinely narrow and categorical, allowing strong precedence for exact tied maxima without extending that effect to the 0.0052 gap in Experiment 2. No calculation uses cue-index distances, adjacency, display span, or inactive positions; order enters only through the earliest member of an actively conflicting tied-top set. Consequently, matched active configurations remain invariant under insertion of empty positions, as required by Experiments 1, 18, and 20. The anchor competition mechanism is explicitly nonmonotonic. Singleton opposition remains credible, intermediate opposition receives maximal challenge amplification, and high-count recovery requires both broad participation and a meaningful validity advantage. This separates coalition defeat in Experiments 4, 12, 13, and 15 from anchor recovery in Experiments 11 and 17, instead of forcing one monotonic count rule across them. Coalition support is configural because effective multiplicity and context-normalized coherence are recomputed whenever a cue is activated. Added cues can therefore reorganize support and increase response precision, providing mechanisms for the activation and balance effects in Experiments 5 and 14. Finally, the three stable orientations create correlated, interpretable heterogeneity: lexicographic subjects have stronger exact-tie precedence but weaker coalition challenge, integrators show the reverse pattern, and restoration-oriented subjects selectively recover clear anchors under high multiplicity. This constrained heterogeneity should reduce spurious unique-anchor crossover prevalence in Experiments 4 and 6 while preserving substantial between-subject variation where the human data show it.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1 (current candidate you are grading): loss=0.2701 -> ACCEPTED
Running-best (last accepted) base: iter 1 at loss=0.2701.

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
**Candidate (simulated) value:** 0.9200 (var=0.0736)
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
**Candidate (simulated) value:** 0.5227 (var=0.0042)
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
**Candidate (simulated) value:** -0.1369 (var=0.0024)
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
**Candidate (simulated) value:** 0.8400 (var=0.1344)
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
**Candidate (simulated) value:** 0.8810 (var=0.0068)
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
**Candidate (simulated) value:** 0.1831 (var=0.0173)
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
**Candidate (simulated) value:** 0.0115 (var=0.0113)
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
**Candidate (simulated) value:** 0.0218 (var=0.0003)
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
**Candidate (simulated) value:** 0.3006 (var=0.0148)
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
**Candidate (simulated) value:** 0.5706 (var=0.0061)
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
**Candidate (simulated) value:** 0.0630 (var=0.0087)
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
**Candidate (simulated) value:** 0.4142 (var=0.0076)
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
**Candidate (simulated) value:** 0.3681 (var=0.0068)
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
**Candidate (simulated) value:** -0.1050 (var=0.2602)
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
**Candidate (simulated) value:** 0.9246 (var=0.0040)
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
**Candidate (simulated) value:** 0.0155 (var=0.0339)
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
**Candidate (simulated) value:** 0.0475 (var=0.0185)
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
**Candidate (simulated) value:** -0.0096 (var=0.0080)
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
**Candidate (simulated) value:** 0.0189 (var=0.0092)
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
**Candidate (simulated) value:** 0.0183 (var=0.0065)
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

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate is faithful to the prescribed STGCC family: it has a narrow equality gate, no spatial-span term, static heterogeneous orientations, and configural nonmonotonic coalition competition. It also succeeds on the intended invariances and nulls, especially Experiments 7, 16, 18, and 20, and is reasonably close on Experiments 11\u201313. However, it does not yet capture the central cross-experiment coalition pattern. The largest failures are Experiment 6, where the predicted multiplicity contrast has the wrong sign (0.183 versus -0.545), and Experiment 17, where high-count anchor recovery is far too weak (0.048 versus 0.560). Experiment 9 likewise shows excessive coalition overturning of a strong anchor (anchor choice 0.301 versus 0.725). Conversely, highest-cue adherence remains too high in Experiment 2 (0.523 versus 0.324), showing that the model does not adequately distinguish credible close opposition from opposition to a clearly superior anchor. Experiment 4 is also a severe failure: it produces an 84% prevalence of a cue-alignment crossover that is absent in every real subject. The exact-tie/configural interaction is miscalibrated as well: Experiment 3 has the wrong sign (-0.137 versus 0.072), while tied-top adherence is somewhat excessive in Experiments 5 and 15. Across several experiments the simulated between-subject variance is much lower than observed, so the three orientations are not generating the relevant heterogeneity.",
  "verdict": "regenerate",
  "rationale": "Keep the STGCC mechanism family, but retune its existing gates rather than replacing it. The current count restoration is much too weak: strengthen restoration sharply at counts of roughly four or more and make its onset more discrete. That should address the jointly diagnostic failures in Experiments 6, 11, and especially 17. At the same time, make coalition challenge more gap-selective: combine a stronger challenge for genuinely close-validity opposition with a smaller challenge_gap_scale or sharper gap normalization, so close coalitions can lower top-cue adherence in Experiments 2, 10, and 13 without overturning the clearly superior anchor in Experiment 9. Experiment 4 indicates that coalition support is too sensitive to whether one particular strong lower cue joins the anchor; cap or normalize single-member leverage within coalition coherence so that Expert 1 alignment does not create a population-wide crossover. Finally, modestly reduce the effective tie-strength range and let the existing balance/coherence term contribute more in exactly tied displays; the present nearly lexicographic gate overpredicts Experiments 5 and 15 while yielding the wrong balanced-minus-imbalanced effect in Experiment 3. Preserve the very narrow tie tolerance and the complete absence of span terms. Broaden the correlated orientation effects\u2014especially restoration strength and close-gap challenge\u2014to better match the substantial human variance in Experiments 9, 11, 17, and 19 without introducing unrestricted subject-specific mechanisms."
}
```

## Usage

```json
{
  "input_tokens": 29922,
  "output_tokens": 1852
}
```
