# feedback_iter_03

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
- THEORY 1 = `pi_13`
- THEORY 2 = `pi_14`
- The recommendation below acts on THEORY 2 (= `pi_14`).

Replace pi_14 with a Diversity-Legitimacy Attribution theory. The new theory should propose that people infer not only endorsement strength but also the likely informational independence or rhetorical coordination of each coalition. Homogeneous, high-validity coalitions are compressed into coordinated option-level claims and can become targets of skepticism in unresolved bilateral conflicts. In contrast, a coalition combining weak or neutral endorsers with an exceptional expert can be construed as diverse corroboration: heterogeneity makes the exceptional endorsement look independently diagnostic rather than like a stronger advocacy claim.

The theory should use a continuous, option-symmetric arbitration among: (1) a default nonlinear raw tally that dominates large absolute and proportional margins; (2) coordinated-claim discounting or pragmatic reversal for coherent coalitions in unresolved bilateral conflicts; and (3) a diversity-legitimacy signal combining breadth, dispersion, and upper-tail surprise. The third signal must be an interaction rather than a generic soft maximum: an extreme expert should gain positive influence when embedded in a sufficiently broad heterogeneous coalition, while an equally extreme but isolated or homogeneous endorsement may still be discounted as an overclaim. Coalition heterogeneity should therefore be capable of increasing evidential credibility, not only reducing advocacy salience.

This account should predict a negative trend in Experiment 1 because replacing neutral fillers with moderately valid fillers removes diversity and upper-tail distinctiveness; high capture in Experiment 2 because the perfect expert is legitimized by a broad heterogeneous coalition; the negative dispersion gradient in Experiment 19; strong heterogeneous-coalition choices in Experiment 21; modest following of unilateral coalitions in Experiment 26; and majority dominance in Experiments 6, 8, and 22. All gates should vary smoothly with breadth, dispersion, validity contrast, and tally decisiveness, without endpoint-specific rules. Stable subject differences should primarily govern sensitivity to coordination versus diversity, with constrained choice precision and lapse parameters. Calibration should be joint across all experiments and should reproduce both pooled metrics and the generally modest subject-level variances.


## CANDIDATE THEORY
Diversity-Legitimacy Attribution theory proposes that people infer both evidential strength and the social origin of expert agreement. Choice continuously arbitrates among a nonlinear raw tally, skeptical reversal of coordinated option-level claims, and a diversity-legitimacy signal in which an exceptional expert becomes credible only through broad, heterogeneous, weak-background corroboration. Homogeneous repetition is compressed, but a residual raw coordination contrast preserves meaningful differences between coherent and dispersed coalitions. Coordination recruitment is itself mostly compressed, preventing coalition breadth or perfect validity from producing categorical increases in skepticism. Decisive tallies recruit counting strongly even though count evidence has a constrained plateau, while unresolved conflicts admit coordination and diversity construals. Stable attribution style governs sensitivity to coordination versus diversity; precision and lapse differences remain narrow.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Diversity-Legitimacy Attribution model. With no outcome feedback, past
    # choices provide no basis for learning expert validity and are ignored.
    stimulus = np.asarray(state, dtype=np.float64)
    if stimulus.ndim != 2 or stimulus.shape[0] != 2:
        raise ValueError(
            f"Expected state with shape (2, n_features); got {stimulus.shape}."
        )

    validities = np.asarray(parameters["validities"], dtype=np.float64)
    n_features = stimulus.shape[1]
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} does not match n_features {n_features}."
        )

    # Positive votes favor A and negative votes favor B. Equal ratings are
    # nondiagnostic and do not enter either option's inferred coalition.
    signed_votes = np.sign(stimulus[0] - stimulus[1])
    active = signed_votes != 0.0
    active_count = int(np.count_nonzero(active))
    tally_margin = float(np.sum(signed_votes))
    margin_magnitude = abs(tally_margin)

    # Smooth coding of communicated validity. No validity receives a special
    # categorical status, including the endpoints of the scale.
    validity = np.clip(validities, 0.5, 1.0)
    curvature = float(parameters["validity_curvature"])
    endpoint = float(np.tanh(0.5 * curvature))
    authority = np.tanh(curvature * (validity - 0.5)) / max(endpoint, 1e-12)

    def summarize(side):
        members = authority[active & (signed_votes == side)]
        breadth = int(members.size)
        if breadth == 0:
            return {
                "breadth": 0,
                "formation": 0.0,
                "coordination": 0.0,
                "legitimacy": 0.0,
                "dispersion": 0.0,
            }

        mean_strength = float(np.mean(members))
        dispersion = float(np.sqrt(np.mean((members - mean_strength) ** 2)))
        maximum_index = int(np.argmax(members))
        maximum = float(members[maximum_index])

        # Coalition formation saturates smoothly: repetition establishes that
        # there is a coalition but does not add independent evidence linearly.
        breadth_scale = float(parameters["breadth_scale"])
        breadth_power = float(parameters["breadth_power"])
        formation = 1.0 - np.exp(
            -(float(breadth) / breadth_scale) ** breadth_power
        )

        # Homogeneous, authoritative endorsers are compressed into a coherent
        # option-level claim. Dispersion weakens coordination attribution.
        coordination_homogeneity = np.exp(
            -(dispersion / float(parameters["coordination_dispersion_scale"])) ** 2
        )
        coordination = (
            formation
            * coordination_homogeneity
            * mean_strength ** float(parameters["coordination_extremity"])
        )

        # Upper-tail surprise is assessed relative to the remaining coalition,
        # not relative to a mean containing the exceptional expert itself.
        # The exceptional endorsement is legitimate only when a nontrivial,
        # weak background exists to provide heterogeneous corroboration.
        if breadth <= 1:
            background_breadth = 0
            background_mean = maximum
            weak_anchor = 0.0
        else:
            background = np.delete(members, maximum_index)
            background_breadth = int(background.size)
            background_mean = float(np.mean(background))
            weak_anchor = float(np.mean(np.exp(
                -background / float(parameters["weak_anchor_scale"])
            )))
        upper_surprise = max(maximum - background_mean, 0.0)

        # Diversity legitimacy is a genuine interaction. It is exactly weak
        # when any constituent is absent: background breadth, dispersion,
        # upper-tail surprise, exceptional authority, or weak-background
        # corroboration. Thus this is not a generic soft maximum.
        dispersion_gate = dispersion / (
            dispersion + float(parameters["diversity_dispersion_scale"])
        )
        surprise_gate = upper_surprise / (
            upper_surprise + float(parameters["surprise_scale"])
        )
        background_formation = 1.0 - np.exp(
            -(float(background_breadth) / breadth_scale) ** breadth_power
        ) if background_breadth > 0 else 0.0
        breadth_gate = background_formation ** float(
            parameters["diversity_breadth_power"]
        )
        legitimacy = (
            breadth_gate
            * dispersion_gate
            * surprise_gate
            * maximum
            * weak_anchor
        )

        return {
            "breadth": breadth,
            "formation": float(formation),
            "coordination": float(coordination),
            "legitimacy": float(legitimacy),
            "dispersion": dispersion,
        }

    side_a = summarize(1.0)
    side_b = summarize(-1.0)

    # Low-threshold compression gives weak but coherent coalitions pragmatic
    # salience. A residual raw contrast preserves coherent-versus-dispersed
    # distinctions that would disappear if both sides saturated completely.
    coordination_scale = float(parameters["coordination_authority_scale"])
    coordination_a = 1.0 - np.exp(
        -side_a["coordination"] / coordination_scale
    )
    coordination_b = 1.0 - np.exp(
        -side_b["coordination"] / coordination_scale
    )
    raw_coordination_contrast = (
        side_a["coordination"] - side_b["coordination"]
    )
    coordination_raw_residual = float(
        parameters["coordination_raw_residual"]
    )
    coordination_contrast = (
        (1.0 - coordination_raw_residual) * (coordination_a - coordination_b)
        + coordination_raw_residual * raw_coordination_contrast
    )
    legitimacy_contrast = side_a["legitimacy"] - side_b["legitimacy"]

    # Decisiveness combines absolute margin and proportional margin. A sparse
    # one-vote advantage is therefore not equated with a broad majority.
    if active_count == 0:
        proportional_margin = 0.0
        decisiveness = 0.0
    else:
        proportional_margin = margin_magnitude / float(active_count)
        absolute_certainty = 1.0 - np.exp(
            -(margin_magnitude / float(parameters["margin_scale"]))
            ** float(parameters["margin_power"])
        )
        decisiveness = absolute_certainty * (
            proportional_margin ** float(parameters["proportion_power"])
        )
    decisiveness = float(np.clip(decisiveness, 0.0, 1.0))
    unresolved = 1.0 - decisiveness

    # Stable attribution style reallocates interpretive attention between
    # coordination skepticism and diversity legitimacy without altering the
    # communicated validities or creating endpoint-specific rules.
    attribution_style = float(parameters["attribution_style"]) - 0.5
    style_shift = float(parameters["style_span"]) * attribution_style

    # Recruitment mostly uses compressed coordination salience, with the same
    # small raw residual used in the directional contrast. This flattens the
    # otherwise excessive breadth and endpoint dependence of claim recruitment.
    strongest_raw_coordination = max(
        side_a["coordination"], side_b["coordination"]
    )
    strongest_compressed_coordination = max(coordination_a, coordination_b)
    strongest_coordination = (
        (1.0 - coordination_raw_residual)
        * strongest_compressed_coordination
        + coordination_raw_residual * strongest_raw_coordination
    )
    strongest_legitimacy = max(side_a["legitimacy"], side_b["legitimacy"])

    # Smooth arbitration utilities. The raw tally receives increasing priority
    # with decisiveness; attributional construals receive priority in unresolved
    # conflicts according to their stimulus-specific support.
    count_utility = (
        float(parameters["count_bias"])
        + float(parameters["count_gate_gain"]) * decisiveness
    )
    coordination_utility = (
        float(parameters["coordination_bias"])
        + float(parameters["unresolved_gain"]) * unresolved
        + float(parameters["coordination_salience"]) * strongest_coordination
        + style_shift
    )
    diversity_utility = (
        float(parameters["diversity_bias"])
        + float(parameters["unresolved_gain"]) * unresolved
        + float(parameters["diversity_salience"]) * strongest_legitimacy
        - style_shift
    )

    utilities = np.asarray(
        [count_utility, coordination_utility, diversity_utility],
        dtype=np.float64,
    )
    utilities /= float(parameters["arbitration_temperature"])
    utilities -= np.max(utilities)
    arbitration = np.exp(utilities)
    arbitration /= np.sum(arbitration)

    # Default tally evidence follows a Hill curve: one-vote margins remain
    # weak, evidence rises rapidly at intermediate margins, and large margins
    # plateau instead of becoming increasingly diagnostic of margin counting.
    if margin_magnitude < 1e-12:
        count_evidence = 0.0
    else:
        count_power = float(parameters["count_exponent"])
        count_half_margin = float(parameters["count_half_margin"])
        margin_term = margin_magnitude ** count_power
        count_curve = margin_term / (
            count_half_margin ** count_power + margin_term
        )
        count_evidence = (
            float(parameters["count_strength"])
            * np.sign(tally_margin)
            * count_curve
            * (0.5 + 0.5 * proportional_margin)
        )

    # Pragmatic reversal requires two represented sides. Its strength falls
    # continuously as the tally resolves the conflict, preventing skepticism
    # from defeating clear majorities or reversing unilateral endorsement.
    bilateral_scale = float(parameters["bilateral_scale"])
    bilateral = np.sqrt(
        max(side_a["formation"] * side_b["formation"], 0.0)
    )
    bilateral = bilateral / (bilateral + bilateral_scale)
    conflict_focus = np.exp(
        -(margin_magnitude / float(parameters["conflict_margin_scale"])) ** 2
    )
    coordination_evidence = -float(parameters["coordination_strength"]) * (
        bilateral
        * conflict_focus
        * np.tanh(
            float(parameters["coordination_slope"])
            * coordination_contrast
        )
    )

    # Legitimate diversity supports, rather than reverses, the coalition with
    # broad heterogeneous corroboration and a surprising upper-tail expert.
    diversity_evidence = float(parameters["diversity_strength"]) * (
        unresolved
        * np.tanh(
            float(parameters["diversity_slope"])
            * legitimacy_contrast
        )
    )

    precision_trait = float(parameters["precision_trait"]) - 0.5
    precision = float(parameters["choice_precision"]) * (
        1.0 + float(parameters["precision_span"]) * precision_trait
    )
    precision = max(precision, 0.10)

    # Each construal supplies a complete conditional choice distribution. We
    # marginalize over construal recruitment instead of averaging incompatible
    # evidence representations into a single latent score.
    evidences = precision * np.asarray(
        [count_evidence, coordination_evidence, diversity_evidence],
        dtype=np.float64,
    )
    channel_p_a = np.empty(3, dtype=np.float64)
    for i, evidence in enumerate(evidences):
        logits = np.asarray([0.5 * evidence, -0.5 * evidence], dtype=np.float64)
        logits -= np.max(logits)
        channel_probs = np.exp(logits)
        channel_probs /= np.sum(channel_probs)
        channel_p_a[i] = channel_probs[0]

    p_a = float(np.dot(arbitration, channel_p_a))
    probabilities = np.asarray([p_a, 1.0 - p_a], dtype=np.float64)

    lapse = float(parameters["lapse_rate"])
    probabilities = (1.0 - lapse) * probabilities + 0.5 * lapse
    probabilities = np.clip(probabilities, 0.0, 1.0)
    probabilities /= np.sum(probabilities)
    return probabilities


`policy(probs) -> int`:
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = np.clip(probabilities, 0.0, None)
    total = float(np.sum(probabilities))
    if not np.isfinite(total) or total <= 0.0:
        probabilities = np.full(
            probabilities.size,
            1.0 / probabilities.size,
            dtype=np.float64,
        )
    else:
        probabilities /= total
    return int(np.random.choice(probabilities.size, p=probabilities))


`parameters`:
- validity_curvature: [1.6, 2.6]
- breadth_scale: [0.85, 1.25]
- breadth_power: [0.80, 1.20]
- coordination_dispersion_scale: [0.16, 0.25]
- coordination_extremity: [1.15, 1.70]
- coordination_authority_scale: [0.035, 0.070]
- coordination_raw_residual: [0.35, 0.55]
- diversity_dispersion_scale: [0.15, 0.24]
- surprise_scale: [0.14, 0.24]
- weak_anchor_scale: [0.28, 0.45]
- diversity_breadth_power: [1.15, 1.75]
- margin_scale: [2.2, 3.2]
- margin_power: [1.25, 1.75]
- proportion_power: [0.65, 1.05]
- count_bias: [-0.10, 0.15]
- count_gate_gain: [3.8, 5.2]
- coordination_bias: [-0.15, 0.20]
- diversity_bias: [-1.00, -0.55]
- unresolved_gain: [0.75, 1.20]
- coordination_salience: [1.35, 2.10]
- diversity_salience: [5.0, 7.5]
- attribution_style: [0, 1]
- style_span: [0.35, 0.65]
- arbitration_temperature: [0.72, 1.00]
- count_strength: [1.75, 2.25]
- count_exponent: [1.8, 2.4]
- count_half_margin: [2.0, 2.8]
- coordination_strength: [4.6, 6.2]
- coordination_slope: [2.4, 3.6]
- bilateral_scale: [0.20, 0.34]
- conflict_margin_scale: [2.4, 3.4]
- diversity_strength: [4.0, 5.6]
- diversity_slope: [3.0, 4.8]
- choice_precision: [0.92, 1.12]
- precision_trait: [0, 1]
- precision_span: [0.08, 0.20]
- lapse_rate: [0.01, 0.055]
- validities: validities

`rationale`:
This is a localized edit of the accepted iteration. The successful Hill-shaped tally and nonmaximum-background legitimacy interaction are retained unchanged. First, the saturated coordination contrast is replaced by a convex blend of compressed and raw contrasts. The compressed component preserves sensitivity to weak 51–55% coalitions in Experiment 21, while the raw residual restores coherent-versus-dispersed distinctions needed in Experiment 16 and reduces instability from subtracting two nearly saturated claims in Experiments 8, 9, 15, and 22. Second, coordination recruitment now uses the same compressed/raw blend rather than only raw coordination. This modestly flattens breadth and endpoint dependence, targeting the excessive Experiment 14 gradient and ordinary-versus-endpoint difference in Experiment 22 without changing coalition formation or bilateral gating. Third, count evidence has a slightly lower plateau, while count recruitment increases more strongly only with decisiveness. This should reduce the excessive margin-sensitive likelihood ratio in Experiment 2 while preserving majority dominance in Experiments 4, 6, and 20. Finally, diversity strength is reduced only mildly to move Experiments 19 and 26 away from ceiling while preserving the representation responsible for the accurately negative Experiment 25 trend. Trait spans, precision, lapse, authority coding, and all other mechanisms are unchanged.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.1842 -> ACCEPTED
- iter 2: loss=0.1868 -> REJECTED
- iter 3: loss=0.1792 -> ACCEPTED
- iter 4 (current candidate you are grading): loss=0.1577 -> ACCEPTED
Running-best (last accepted) base: iter 4 at loss=0.1577.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1]
  A=[0, 1, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 1, 0, 1]
  A=[1, 0, 1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 1, 0]  B=[1, 0, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Choice rate for the Take-The-Best winner on TTB-vs-tally conflict trials."""
    aligned = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b

        # Validities are strictly descending in feature-index order, so the
        # first nonzero difference identifies the Take-The-Best winner.
        discriminating = np.flatnonzero(diff != 0)
        if discriminating.size == 0:
            continue
        ttb_winner = 0 if diff[discriminating[0]] > 0 else 1

        a_wins = int(np.sum(diff > 0))
        b_wins = int(np.sum(diff < 0))
        if a_wins == b_wins:
            continue
        tally_winner = 0 if a_wins > b_wins else 1

        # Retain only directional conflicts between the two heuristics.
        if ttb_winner != tally_winner:
            aligned.append(float(int(row["response"]) == ttb_winner))

    # The selected design always contains conflict trials; the fallback keeps
    # the function defined for any unexpected slice.
    return float(np.mean(aligned)) if aligned else 0.5
```

**Observed (real) value:** 0.4033 (var=0.0042)
**Candidate trajectory (this loop):**
  - iter 1: 0.2608 (var=0.0038) (Δ vs real -0.1425)
  - iter 2: 0.3113 (var=0.0060) (Δ vs real -0.0921)
  - iter 3: 0.3625 (var=0.0051) (Δ vs real -0.0408)
  - iter 4 (current): 0.3167 (var=0.0045) (Δ vs real -0.0867)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8450 (var=0.0143)
- pi_2: 0.1350 (var=0.0104)
- pi_3: 0.3775 (var=0.0056)
- pi_4: 0.3392 (var=0.0040)
- pi_5: 0.3108 (var=0.0052)
- pi_6: 0.4721 (var=0.0071)
- pi_7: 0.0896 (var=0.0023)
- pi_8: 0.3383 (var=0.0039)
- pi_9: 0.3208 (var=0.0107)
- pi_10: 0.3250 (var=0.0112)
- pi_11: 0.4133 (var=0.0100)
- pi_12: 0.3004 (var=0.0051)
- pi_13: 0.2571 (var=0.0056)
- pi_14: 0.3292 (var=0.0051)

### Experiment 2
**Design**
  A=[1, 1, 1, 0, 1, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 1, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 1, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Total log marginal-likelihood ratio favoring Tallying over TTB.
    # The parameter grids approximate uniform priors over the stated ranges.
    beta_grid = np.linspace(0.1, 20.0, 120)
    epsilon_grid = np.linspace(0.0, 0.5, 61)
    beta = beta_grid[:, None]
    epsilon = epsilon_grid[None, :]

    def logmeanexp(x):
        x = np.asarray(x, dtype=float)
        m = float(np.max(x))
        return float(m + np.log(np.mean(np.exp(x - m))))

    def subject_log_bf(df):
        margins = []
        correct = []

        for _, row in df.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float)
            b = np.asarray(row["option_b_ratings"], dtype=float)
            a_wins = int(np.sum(a > b))
            b_wins = int(np.sum(b > a))
            margin = abs(a_wins - b_wins)

            if margin == 0:
                continue

            winner = 0 if a_wins > b_wins else 1
            margins.append(float(margin))
            correct.append(float(int(row["response"]) == winner))

        if len(margins) == 0:
            return np.nan

        margins = np.asarray(margins, dtype=float)
        correct = np.asarray(correct, dtype=float)
        tiny = 1e-12

        # TTB assigns the same winner probability to every non-tied pair.
        core_ttb = 1.0 / (1.0 + np.exp(-beta))
        p_ttb = (1.0 - epsilon) * core_ttb + epsilon * 0.5
        p_ttb = np.clip(p_ttb, tiny, 1.0 - tiny)
        n_correct = float(np.sum(correct))
        n_error = float(correct.size - n_correct)
        ll_ttb = n_correct * np.log(p_ttb) + n_error * np.log1p(-p_ttb)

        # Tallying's winner probability depends on the observed tally margin.
        ll_tally = np.zeros_like(p_ttb, dtype=float)
        for d, y in zip(margins, correct):
            x = beta * d
            core = np.where(x >= 0.0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))
            p = (1.0 - epsilon) * core + epsilon * 0.5
            p = np.clip(p, tiny, 1.0 - tiny)
            ll_tally += y * np.log(p) + (1.0 - y) * np.log1p(-p)

        log_ml_tally = logmeanexp(ll_tally)
        log_ml_ttb = logmeanexp(ll_ttb)
        return float(log_ml_tally - log_ml_ttb)

    if "subject_id" in data.columns:
        values = [subject_log_bf(g) for _, g in data.groupby("subject_id", sort=False)]
    else:
        values = [subject_log_bf(data)]

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")

    # Independent participants contribute additive evidence. On a one-subject
    # slice this is simply that participant's log Bayes factor.
    return float(np.sum(values))

```

**Observed (real) value:** 89.1422 (var=7.5896)
**Candidate trajectory (this loop):**
  - iter 1: 226.2049 (var=2.7761) (Δ vs real +137.0627)
  - iter 2: 121.1837 (var=1.9413) (Δ vs real +32.0415)
  - iter 3: 201.9244 (var=4.0862) (Δ vs real +112.7822)
  - iter 4 (current): 163.4468 (var=2.5954) (Δ vs real +74.3045)
**Other theories' values on this metric (for reference):**
- pi_2: 1.9009 (var=0.0916)
- pi_1: 0.5503 (var=0.0488)
- pi_3: 49.6494 (var=1.7369)
- pi_4: 100.3141 (var=2.8389)
- pi_5: 59.0660 (var=1.4267)
- pi_6: 84.3674 (var=2.5041)
- pi_7: 83.3362 (var=1.6438)
- pi_8: 56.1425 (var=1.2063)
- pi_9: 59.9467 (var=3.7118)
- pi_10: 89.8046 (var=9.1438)
- pi_11: 51.5068 (var=3.0531)
- pi_12: 156.3801 (var=2.2245)
- pi_13: 101.4886 (var=2.8840)
- pi_14: 187.4611 (var=2.5841)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Proportion of choices favoring the option with fewer positive ratings."""
    if data is None or len(data) == 0:
        return float("nan")

    minority_choices = []
    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"],
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        tally_margin_a = float(np.sum(a > b) - np.sum(b > a))
        r = int(response)

        if tally_margin_a < 0:       # A has fewer cue wins
            minority_choices.append(float(r == 0))
        elif tally_margin_a > 0:     # B has fewer cue wins
            minority_choices.append(float(r == 1))
        # Tied tallies have no minority option and are omitted.

    if len(minority_choices) == 0:
        return float("nan")
    return float(np.mean(minority_choices))
```

**Observed (real) value:** 0.4508 (var=0.0031)
**Candidate trajectory (this loop):**
  - iter 1: 0.2362 (var=0.0033) (Δ vs real -0.2146)
  - iter 2: 0.2335 (var=0.0038) (Δ vs real -0.2173)
  - iter 3: 0.3810 (var=0.0029) (Δ vs real -0.0698)
  - iter 4 (current): 0.2977 (var=0.0030) (Δ vs real -0.1531)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5394 (var=0.0025)
- pi_2: 0.1190 (var=0.0098)
- pi_1: 0.7531 (var=0.0085)
- pi_4: 0.4698 (var=0.0021)
- pi_5: 0.4573 (var=0.0018)
- pi_6: 0.3958 (var=0.0029)
- pi_7: 0.1542 (var=0.0036)
- pi_8: 0.4106 (var=0.0035)
- pi_9: 0.3440 (var=0.0028)
- pi_10: 0.3654 (var=0.0036)
- pi_11: 0.3771 (var=0.0040)
- pi_12: 0.3973 (var=0.0029)
- pi_13: 0.2502 (var=0.0050)
- pi_14: 0.3169 (var=0.0017)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if len(data) == 0:
        return float("nan")

    a = np.vstack([np.asarray(x, dtype=float) for x in data["option_a_ratings"]])
    b = np.vstack([np.asarray(x, dtype=float) for x in data["option_b_ratings"]])
    responses = np.asarray(data["response"], dtype=int)

    # Positive margin means that A wins more binary feature comparisons;
    # negative margin means that B does.
    delta = a - b
    tally_margin = np.sum(delta > 0, axis=1) - np.sum(delta < 0, axis=1)

    # Select the two orientations of the diagnostic 6-versus-1 conflict:
    # the tally winner has a five-cue margin but is opposed by expert 1.
    strongest_cue = delta[:, 0]
    critical = (np.abs(tally_margin) == 5) & (strongest_cue * tally_margin < 0)
    if not np.any(critical):
        return float("nan")

    chose_a = responses == 0
    tally_winner_is_a = tally_margin > 0

    # Rate at which the numerous weak experts override the strongest expert.
    return float(np.mean(chose_a[critical] == tally_winner_is_a[critical]))
```

**Observed (real) value:** 0.8213 (var=0.0169)
**Candidate trajectory (this loop):**
  - iter 1: 0.7725 (var=0.0079) (Δ vs real -0.0488)
  - iter 2: 0.6737 (var=0.0119) (Δ vs real -0.1475)
  - iter 3: 0.7656 (var=0.0048) (Δ vs real -0.0556)
  - iter 4 (current): 0.7469 (var=0.0080) (Δ vs real -0.0744)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8606 (var=0.0060)
- pi_3: 0.3869 (var=0.0089)
- pi_1: 0.1325 (var=0.0094)
- pi_4: 0.7775 (var=0.0083)
- pi_5: 0.8337 (var=0.0061)
- pi_6: 0.7837 (var=0.0060)
- pi_7: 0.9413 (var=0.0016)
- pi_8: 0.8544 (var=0.0058)
- pi_9: 0.9294 (var=0.0023)
- pi_10: 0.9350 (var=0.0031)
- pi_11: 0.8800 (var=0.0067)
- pi_12: 0.8381 (var=0.0062)
- pi_13: 0.8313 (var=0.0084)
- pi_14: 0.8056 (var=0.0058)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Recode each response as choosing the option endorsed by the 95%-valid
    # expert (feature 0), irrespective of whether that option is A or B.
    by_size = {1: [], 2: [], 4: []}

    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        difference = a - b

        # In this design, coalition size is the number of experts endorsing A
        # over B (equal to the number endorsing B over A).
        coalition_size = int(np.sum(difference > 0.0))
        if coalition_size not in by_size:
            continue

        high_expert_option = 0 if difference[0] > 0.0 else 1
        chose_high_expert = float(int(row['response']) == high_expert_option)
        by_size[coalition_size].append(chose_high_expert)

    if any(len(by_size[k]) == 0 for k in (1, 2, 4)):
        return float('nan')

    p1 = float(np.mean(by_size[1]))
    p2 = float(np.mean(by_size[2]))
    p4 = float(np.mean(by_size[4]))

    # Approximate inverse-noise weighting of the conditions with the largest
    # predicted between-theory probability differences. The size-3 condition
    # is omitted because both theories predict it close to chance.
    return float(1.8 * (p1 - 0.5) + 0.9 * (p2 - 0.5) - (p4 - 0.5))
```

**Observed (real) value:** -0.0382 (var=0.0352)
**Candidate trajectory (this loop):**
  - iter 1: -0.6993 (var=0.0176) (Δ vs real -0.6611)
  - iter 2: -0.3346 (var=0.0515) (Δ vs real -0.2964)
  - iter 3: -0.0839 (var=0.0351) (Δ vs real -0.0458)
  - iter 4 (current): -0.5192 (var=0.0351) (Δ vs real -0.4810)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5887 (var=0.0712)
- pi_4: 0.0071 (var=0.0478)
- pi_1: 0.5484 (var=0.0538)
- pi_2: -0.0256 (var=0.0579)
- pi_5: -0.0126 (var=0.0452)
- pi_6: -0.0866 (var=0.0563)
- pi_7: -0.5556 (var=0.0918)
- pi_8: 0.0104 (var=0.0390)
- pi_9: -0.4666 (var=0.0908)
- pi_10: -0.4282 (var=0.0521)
- pi_11: -0.3760 (var=0.0522)
- pi_12: -0.4507 (var=0.0378)
- pi_13: -0.3333 (var=0.0486)
- pi_14: -0.1537 (var=0.0452)

### Experiment 6
**Design**
  A=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Proportion of choices favoring the option with the larger raw vote tally."""
    scores = []
    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"],
    ):
        a = np.asarray(a_cell, dtype=np.float64)
        b = np.asarray(b_cell, dtype=np.float64)
        margin = float(np.sum(a - b))
        r = int(response)

        if margin > 0.0:
            scores.append(1.0 if r == 0 else 0.0)
        elif margin < 0.0:
            scores.append(1.0 if r == 1 else 0.0)
        else:
            scores.append(0.5)

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** 0.8462 (var=0.0122)
**Candidate trajectory (this loop):**
  - iter 1: 0.7844 (var=0.0023) (Δ vs real -0.0619)
  - iter 2: 0.7202 (var=0.0029) (Δ vs real -0.1260)
  - iter 3: 0.7925 (var=0.0028) (Δ vs real -0.0537)
  - iter 4 (current): 0.7444 (var=0.0022) (Δ vs real -0.1019)
**Other theories' values on this metric (for reference):**
- pi_4: 0.8037 (var=0.0033)
- pi_3: 0.5665 (var=0.0017)
- pi_1: 0.8565 (var=0.0102)
- pi_2: 0.8612 (var=0.0065)
- pi_5: 0.8506 (var=0.0014)
- pi_6: 0.8273 (var=0.0015)
- pi_7: 0.8325 (var=0.0029)
- pi_8: 0.8592 (var=0.0031)
- pi_9: 0.9035 (var=0.0023)
- pi_10: 0.9150 (var=0.0039)
- pi_11: 0.8948 (var=0.0037)
- pi_12: 0.8283 (var=0.0029)
- pi_13: 0.8612 (var=0.0028)
- pi_14: 0.8171 (var=0.0013)

### Experiment 7
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Score whether each response selected the option endorsed by the
    # high-validity experts (features 0 through 6), irrespective of screen side.
    weighted_success = 0.0
    total_weight = 0.0

    # Increasing weights emphasize the larger-coalition conditions, where the
    # theories' predicted choice probabilities diverge most, while retaining
    # lower-m conditions to keep the subject-level estimate stable.
    weights = np.asarray([0.30, 0.55, 0.72, 0.85, 0.94, 1.00], dtype=float)

    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"],
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        if a.size < 7 or b.size < 7:
            continue

        high_margin = float(np.sum(a[:7] - b[:7]))
        m = int(round(abs(high_margin)))
        if high_margin == 0.0 or m < 2 or m > 7:
            continue

        chose_a = int(response) == 0
        high_option_is_a = high_margin > 0.0
        chose_high_coalition = float(chose_a == high_option_is_a)

        weight = float(weights[m - 2])
        weighted_success += weight * chose_high_coalition
        total_weight += weight

    if total_weight == 0.0:
        return float("nan")
    return float(weighted_success / total_weight)
```

**Observed (real) value:** 0.1161 (var=0.0079)
**Candidate trajectory (this loop):**
  - iter 1: 0.0989 (var=0.0012) (Δ vs real -0.0173)
  - iter 2: 0.1043 (var=0.0020) (Δ vs real -0.0118)
  - iter 3: 0.1010 (var=0.0018) (Δ vs real -0.0151)
  - iter 4 (current): 0.0966 (var=0.0017) (Δ vs real -0.0195)
**Other theories' values on this metric (for reference):**
- pi_5: 0.7494 (var=0.0030)
- pi_4: 0.5919 (var=0.0025)
- pi_1: 0.8622 (var=0.0079)
- pi_2: 0.8381 (var=0.0106)
- pi_3: 0.9864 (var=0.0001)
- pi_6: 0.1972 (var=0.0033)
- pi_7: 0.2755 (var=0.0544)
- pi_8: 0.2347 (var=0.0051)
- pi_9: 0.3269 (var=0.0321)
- pi_10: 0.0811 (var=0.0040)
- pi_11: 0.1813 (var=0.0070)
- pi_12: 0.2380 (var=0.0051)
- pi_13: 0.0835 (var=0.0026)
- pi_14: 0.0942 (var=0.0009)

### Experiment 8
**Design**
  A=[0, 1, 0, 0]  B=[1, 0, 0, 0]
  A=[0, 1, 1, 1]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[0, 1, 1, 1]  B=[1, 1, 1, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 0, 1, 0]
  A=[1, 1, 0, 1]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 0, 0, 1]
  A=[1, 1, 1, 0]  B=[0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if data is None or len(data) == 0:
        return 0.0

    # Bounded reliability code specified by Compressed-Reliability Consensus.
    validities = np.array([0.5, 0.6, 0.8, 1.0], dtype=np.float64)
    reliability_code = np.tanh(3.0 * (validities - 0.5)) / np.tanh(1.5)

    numerators = []
    weights = []
    subjects = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=np.float64)
        b = np.asarray(row["option_b_ratings"], dtype=np.float64)
        contrast = float(np.dot(a - b, reliability_code))
        weight = abs(contrast)
        if not np.isfinite(weight) or weight <= 0.0:
            continue

        # +1 means the response followed the more reliable endorser, and -1
        # means it followed the less reliable endorser. This is invariant to
        # whether that endorser appeared on option A or option B.
        choice_sign = 1.0 - 2.0 * float(row["response"])
        aligned_choice = np.sign(contrast) * choice_sign
        numerators.append(weight * aligned_choice)
        weights.append(weight)
        subjects.append(row["subject_id"])

    if len(weights) == 0:
        return 0.0

    work = pd.DataFrame({
        "subject_id": subjects,
        "numerator": numerators,
        "weight": weights,
    })
    grouped = work.groupby("subject_id", sort=False)[["numerator", "weight"]].sum()
    valid = grouped["weight"] > 0.0
    if not bool(valid.any()):
        return 0.0

    subject_scores = (
        grouped.loc[valid, "numerator"] / grouped.loc[valid, "weight"]
    ).to_numpy(dtype=np.float64)

    # Normalized cumulative directional evidence: for one subject this is the
    # subject's weighted alignment score; pooled evidence grows with sqrt(N).
    return float(np.sqrt(subject_scores.size) * np.mean(subject_scores))

```

**Observed (real) value:** -3.5372 (var=0.0438)
**Candidate trajectory (this loop):**
  - iter 1: -4.5034 (var=0.0092) (Δ vs real -0.9662)
  - iter 2: -5.3952 (var=0.0071) (Δ vs real -1.8580)
  - iter 3: -5.0527 (var=0.0080) (Δ vs real -1.5156)
  - iter 4 (current): -5.5641 (var=0.0064) (Δ vs real -2.0269)
**Other theories' values on this metric (for reference):**
- pi_4: 0.4221 (var=0.0108)
- pi_5: -0.0028 (var=0.0102)
- pi_1: 4.5627 (var=0.0649)
- pi_2: 0.0176 (var=0.0099)
- pi_3: 3.9747 (var=0.0054)
- pi_6: -3.4648 (var=0.0108)
- pi_7: -1.4013 (var=0.0925)
- pi_8: -4.4717 (var=0.0102)
- pi_9: -2.4502 (var=0.0542)
- pi_10: -3.8304 (var=0.0211)
- pi_11: -4.5864 (var=0.0269)
- pi_12: -5.1299 (var=0.0137)
- pi_13: -4.4524 (var=0.0353)
- pi_14: -4.6563 (var=0.0108)

### Experiment 9
**Design**
  A=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if data is None or len(data) == 0:
        return float('nan')

    weighted_success = 0.0
    total_weight = 0.0

    # Conditions are represented in a canonical orientation using the sign of
    # the overall tally. This makes exact A/B-reversed pairs contribute to the
    # same planned condition.
    condition_weights = {
        7: 2.00,
        3: 1.45,
        -3: 0.80,
        -6: 1.10,
    }

    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=np.float64)
        b = np.asarray(row['option_b_ratings'], dtype=np.float64)
        if a.size < 7 or b.size != a.size:
            continue

        signed_votes = np.sign(a - b)
        high_validity_margin = float(np.sum(signed_votes[:7]))
        tally_margin = float(np.sum(signed_votes))
        if high_validity_margin == 0.0 or tally_margin == 0.0:
            continue

        canonical_margin = int(round(high_validity_margin * np.sign(tally_margin)))
        weight = condition_weights.get(canonical_margin, 1.0)

        response = int(row['response'])
        chose_high_validity_side = (
            (high_validity_margin > 0.0 and response == 0) or
            (high_validity_margin < 0.0 and response == 1)
        )
        weighted_success += weight * float(chose_high_validity_side)
        total_weight += weight

    if total_weight == 0.0:
        return float('nan')
    return float(weighted_success / total_weight)
```

**Observed (real) value:** 0.3248 (var=0.0018)
**Candidate trajectory (this loop):**
  - iter 1: 0.1081 (var=0.0014) (Δ vs real -0.2167)
  - iter 2: 0.0973 (var=0.0014) (Δ vs real -0.2275)
  - iter 3: 0.0860 (var=0.0011) (Δ vs real -0.2388)
  - iter 4 (current): 0.0834 (var=0.0006) (Δ vs real -0.2415)
**Other theories' values on this metric (for reference):**
- pi_5: 0.6632 (var=0.0032)
- pi_6: 0.1697 (var=0.0029)
- pi_1: 0.7324 (var=0.0054)
- pi_2: 0.5977 (var=0.0022)
- pi_3: 0.9801 (var=0.0004)
- pi_4: 0.5348 (var=0.0021)
- pi_7: 0.3215 (var=0.0354)
- pi_8: 0.2814 (var=0.0029)
- pi_9: 0.3376 (var=0.0157)
- pi_10: 0.1905 (var=0.0047)
- pi_11: 0.2755 (var=0.0039)
- pi_12: 0.3307 (var=0.0044)
- pi_13: 0.2713 (var=0.0034)
- pi_14: 0.1516 (var=0.0017)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 0, 0, 1]
  A=[1, 1, 0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 0, 0, 1]  B=[1, 1, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 0, 1, 1]  B=[1, 1, 0, 1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    scores = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b

        # The concentrated endpoint condition has exactly two discriminating
        # experts: one 100%-valid expert and the final 50%-valid expert.
        if np.count_nonzero(diff) != 2 or diff.size < 8 or diff[7] == 0:
            continue

        response = int(row["response"])
        low_validity_side = 0 if diff[7] > 0 else 1
        scores.append(float(response == low_validity_side))

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** 0.8617 (var=0.0104)
**Candidate trajectory (this loop):**
  - iter 1: 0.8842 (var=0.0050) (Δ vs real +0.0225)
  - iter 2: 0.8925 (var=0.0057) (Δ vs real +0.0308)
  - iter 3: 0.8792 (var=0.0067) (Δ vs real +0.0175)
  - iter 4 (current): 0.9175 (var=0.0027) (Δ vs real +0.0558)
**Other theories' values on this metric (for reference):**
- pi_6: 0.7625 (var=0.0137)
- pi_5: 0.5375 (var=0.0072)
- pi_1: 0.1683 (var=0.0142)
- pi_2: 0.4767 (var=0.0123)
- pi_3: 0.0008 (var=0.0000)
- pi_4: 0.4383 (var=0.0129)
- pi_7: 0.8467 (var=0.0208)
- pi_8: 0.8408 (var=0.0059)
- pi_9: 0.7475 (var=0.0177)
- pi_10: 0.8658 (var=0.0100)
- pi_11: 0.8533 (var=0.0093)
- pi_12: 0.8767 (var=0.0047)
- pi_13: 0.8358 (var=0.0117)
- pi_14: 0.8267 (var=0.0060)

### Experiment 11
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if len(data) == 0:
        return float('nan')

    scores = []
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        diff = a - b

        # The first five experts have 70% validity. Restrict attention to the
        # diffuse conflicts (3-vs-2, 4-vs-3, and 5-vs-4), where at least five
        # experts discriminate between the products.
        discriminating_count = int(np.count_nonzero(diff))
        if discriminating_count < 5:
            continue

        high_validity_margin = float(np.sum(diff[:5]))
        if high_validity_margin > 0:
            chose_high_validity_coalition = int(row['response']) == 0
        elif high_validity_margin < 0:
            chose_high_validity_coalition = int(row['response']) == 1
        else:
            continue

        scores.append(float(chose_high_validity_coalition))

    if not scores:
        return float('nan')
    return float(np.mean(scores))
```

**Observed (real) value:** 0.1572 (var=0.0077)
**Candidate trajectory (this loop):**
  - iter 1: 0.1911 (var=0.0032) (Δ vs real +0.0339)
  - iter 2: 0.1175 (var=0.0034) (Δ vs real -0.0397)
  - iter 3: 0.1781 (var=0.0031) (Δ vs real +0.0208)
  - iter 4 (current): 0.1258 (var=0.0023) (Δ vs real -0.0314)
**Other theories' values on this metric (for reference):**
- pi_7: 0.9125 (var=0.0016)
- pi_6: 0.2006 (var=0.0082)
- pi_1: 0.8514 (var=0.0098)
- pi_2: 0.8394 (var=0.0083)
- pi_3: 0.7772 (var=0.0054)
- pi_4: 0.5706 (var=0.0031)
- pi_5: 0.6192 (var=0.0027)
- pi_8: 0.2039 (var=0.0054)
- pi_9: 0.4628 (var=0.0171)
- pi_10: 0.1292 (var=0.0058)
- pi_11: 0.1917 (var=0.0067)
- pi_12: 0.2394 (var=0.0066)
- pi_13: 0.0964 (var=0.0043)
- pi_14: 0.0842 (var=0.0012)

### Experiment 12
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    conventional_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        d = a - b

        # Select the 12 mirrored conditions comparing equal-sized coalitions
        # of 70%-valid experts (indices 0:6) and 50%-valid experts (6:12),
        # excluding all conditions involving 95%-valid experts.
        d70 = d[:6]
        d50 = d[6:12]
        d95 = d[12:15]
        n70 = int(np.count_nonzero(d70))
        n50 = int(np.count_nonzero(d50))
        s70 = float(np.sum(d70))
        s50 = float(np.sum(d50))

        is_target = (
            n70 > 0
            and n70 == n50
            and abs(s70) == n70
            and abs(s50) == n50
            and np.sign(s70) == -np.sign(s50)
            and np.count_nonzero(d95) == 0
        )
        if not is_target:
            continue

        response = int(row["response"])
        # Score 1 when the chosen product is endorsed by the 70%-valid
        # coalition, regardless of whether that product appears as A or B.
        chose_70_option = ((s70 > 0 and response == 0) or
                           (s70 < 0 and response == 1))
        conventional_choices.append(float(chose_70_option))

    if len(conventional_choices) == 0:
        return float("nan")
    return float(np.mean(conventional_choices))
```

**Observed (real) value:** 0.1261 (var=0.0089)
**Candidate trajectory (this loop):**
  - iter 1: 0.1847 (var=0.0039) (Δ vs real +0.0586)
  - iter 2: 0.1078 (var=0.0023) (Δ vs real -0.0183)
  - iter 3: 0.1511 (var=0.0018) (Δ vs real +0.0250)
  - iter 4 (current): 0.1069 (var=0.0020) (Δ vs real -0.0192)
**Other theories' values on this metric (for reference):**
- pi_6: 0.2356 (var=0.0041)
- pi_7: 0.8364 (var=0.0032)
- pi_1: 0.8367 (var=0.0093)
- pi_2: 0.4964 (var=0.0036)
- pi_3: 0.7497 (var=0.0047)
- pi_4: 0.5364 (var=0.0028)
- pi_5: 0.4892 (var=0.0026)
- pi_8: 0.1583 (var=0.0037)
- pi_9: 0.3469 (var=0.0105)
- pi_10: 0.1528 (var=0.0051)
- pi_11: 0.1503 (var=0.0048)
- pi_12: 0.1225 (var=0.0047)
- pi_13: 0.1078 (var=0.0040)
- pi_14: 0.0944 (var=0.0011)

### Experiment 13
**Design**
  A=[1, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def subject_contrast(df):
        successes = np.zeros(3, dtype=np.float64)
        totals = np.zeros(3, dtype=np.float64)

        for _, row in df.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=np.float64)
            b = np.asarray(row["option_b_ratings"], dtype=np.float64)
            signed = np.sign(a - b)

            # Exactly one of the first three (90%, 95%, 100%) experts is
            # discriminating in each designed trial.
            focal = np.flatnonzero(signed[:3] != 0.0)
            if focal.size != 1:
                continue
            condition = int(focal[0])
            focal_sign = float(signed[condition])
            response = int(row["response"])

            # Score choosing the option endorsed by the focal high-validity
            # expert, making exact A/B reversals contribute on the same scale.
            chose_focal_option = (
                (focal_sign > 0.0 and response == 0)
                or (focal_sign < 0.0 and response == 1)
            )
            successes[condition] += float(chose_focal_option)
            totals[condition] += 1.0

        if np.any(totals == 0.0):
            return np.nan
        rates = successes / totals
        return float(rates[1] - 0.5 * (rates[0] + rates[2]))

    if "subject_id" in data.columns and data["subject_id"].nunique() > 1:
        values = []
        for _, subj_df in data.groupby("subject_id", sort=False):
            value = subject_contrast(subj_df)
            if np.isfinite(value):
                values.append(value)
        return float(np.mean(values)) if values else np.nan

    return subject_contrast(data)

```

**Observed (real) value:** 0.0094 (var=0.0050)
**Candidate trajectory (this loop):**
  - iter 1: -0.0003 (var=0.0061) (Δ vs real -0.0097)
  - iter 2: 0.1300 (var=0.0059) (Δ vs real +0.1206)
  - iter 3: 0.0369 (var=0.0076) (Δ vs real +0.0275)
  - iter 4 (current): 0.0187 (var=0.0101) (Δ vs real +0.0094)
**Other theories' values on this metric (for reference):**
- pi_8: 0.0569 (var=0.0085)
- pi_6: -0.0147 (var=0.0080)
- pi_1: -0.0147 (var=0.0061)
- pi_2: 0.0028 (var=0.0045)
- pi_3: -0.0878 (var=0.0068)
- pi_4: -0.0187 (var=0.0085)
- pi_5: 0.0069 (var=0.0102)
- pi_7: -0.0338 (var=0.0098)
- pi_9: -0.0147 (var=0.0077)
- pi_10: 0.0109 (var=0.0179)
- pi_11: 0.0403 (var=0.0095)
- pi_12: 0.0016 (var=0.0084)
- pi_13: 0.0481 (var=0.0062)
- pi_14: -0.0197 (var=0.0084)

### Experiment 14
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Preference for the option not endorsed by the perfectly valid expert,
    # scored separately at each coalition breadth and display orientation.
    cell_values = {(m, side): [] for m in range(1, 7) for side in (-1, 1)}

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.size == 0 or b.size != a.size:
            continue

        endpoint_side = int(np.sign(a[0] - b[0]))
        if endpoint_side == 0:
            continue

        active_count = int(np.count_nonzero(a - b))
        m = active_count // 2
        if m < 1 or m > 6 or active_count != 2 * m:
            continue

        response = int(row["response"])
        # endpoint_side=+1 means the certainty expert endorses A, so choosing
        # B is the anti-endpoint choice. For endpoint_side=-1, choosing A is.
        anti_endpoint_choice = float(
            (endpoint_side == 1 and response == 1)
            or (endpoint_side == -1 and response == 0)
        )
        cell_values[(m, endpoint_side)].append(anti_endpoint_choice)

    breadth_means = []
    for m in range(1, 7):
        orientation_means = []
        for side in (-1, 1):
            values = cell_values[(m, side)]
            if len(values) == 0:
                return float("nan")
            orientation_means.append(float(np.mean(values)))
        breadth_means.append(float(np.mean(orientation_means)))

    # A normalized linear low-to-high coalition-breadth contrast. Positive
    # values indicate that anti-endpoint preference attenuates as m increases.
    weights = np.array([2.5, 1.5, 0.5, -0.5, -1.5, -2.5], dtype=float)
    return float(np.dot(weights, np.asarray(breadth_means)) / 4.5)

```

**Observed (real) value:** 0.1781 (var=0.0129)
**Candidate trajectory (this loop):**
  - iter 1: 0.4692 (var=0.0065) (Δ vs real +0.2911)
  - iter 2: 0.4539 (var=0.0085) (Δ vs real +0.2758)
  - iter 3: 0.4928 (var=0.0067) (Δ vs real +0.3147)
  - iter 4 (current): 0.5125 (var=0.0049) (Δ vs real +0.3344)
**Other theories' values on this metric (for reference):**
- pi_6: -0.0096 (var=0.0069)
- pi_8: 0.1467 (var=0.0083)
- pi_1: 0.0031 (var=0.0052)
- pi_2: 0.0064 (var=0.0127)
- pi_3: 0.0007 (var=0.0000)
- pi_4: -0.0000 (var=0.0117)
- pi_5: -0.0346 (var=0.0125)
- pi_7: -0.0029 (var=0.0065)
- pi_9: 0.1044 (var=0.0094)
- pi_10: 0.1275 (var=0.0095)
- pi_11: 0.1763 (var=0.0121)
- pi_12: 0.1631 (var=0.0099)
- pi_13: 0.3596 (var=0.0395)
- pi_14: -0.0294 (var=0.0060)

### Experiment 15
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    low_endpoint_choices = []
    high_endpoint_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.size < 5 or b.size != a.size:
            continue

        diff = a - b
        active = np.flatnonzero(diff != 0)
        if active.size != 2 or 0 not in active:
            continue

        opponents = active[active != 0]
        if opponents.size != 1:
            continue
        opponent = int(opponents[0])
        if diff[0] * diff[opponent] >= 0:
            continue

        response = int(row["response"])
        endpoint_chosen = (
            (diff[0] > 0 and response == 0)
            or (diff[0] < 0 and response == 1)
        )

        if opponent in (1, 2):
            low_endpoint_choices.append(float(endpoint_chosen))
        elif opponent == 4:
            high_endpoint_choices.append(float(endpoint_chosen))

    if len(low_endpoint_choices) == 0 or len(high_endpoint_choices) == 0:
        return float("nan")

    low_rate = float(np.mean(low_endpoint_choices))
    high_rate = float(np.mean(high_endpoint_choices))

    return float((low_rate < 0.5) and (high_rate > 0.5))

```

**Observed (real) value:** 0.0000 (var=0.2400)
**Candidate trajectory (this loop):**
  - iter 1: 0.0000 (var=0.0196) (Δ vs real +0.0000)
  - iter 2: 0.0000 (var=0.1824) (Δ vs real +0.0000)
  - iter 3: 1.0000 (var=0.2496) (Δ vs real +1.0000)
  - iter 4 (current): 0.0000 (var=0.1344) (Δ vs real +0.0000)
**Other theories' values on this metric (for reference):**
- pi_8: 1.0000 (var=0.1600)
- pi_9: 0.0000 (var=0.1204)
- pi_1: 0.0000 (var=0.0000)
- pi_2: 0.0000 (var=0.1600)
- pi_3: 0.0000 (var=0.0000)
- pi_4: 0.0000 (var=0.1204)
- pi_5: 0.0000 (var=0.0900)
- pi_6: 1.0000 (var=0.2484)
- pi_7: 0.0000 (var=0.0000)
- pi_10: 0.0000 (var=0.1716)
- pi_11: 0.0000 (var=0.1476)
- pi_12: 0.0000 (var=0.2100)
- pi_13: 0.0000 (var=0.2176)
- pi_14: 1.0000 (var=0.2400)

### Experiment 16
**Design**
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    selected_choices = []

    # Coalitions opposed to the four 75%-valid experts. The maximally
    # dispersed coalition is intentionally excluded because the competing
    # theory predicts that its anti-reliability signal is strongly diluted.
    homogeneous = (4, 5, 6, 7)
    moderate_dispersion = (8, 9, 10, 11)
    selected = {homogeneous, moderate_dispersion}

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=int)
        b = np.asarray(row["option_b_ratings"], dtype=int)
        response = int(row["response"])

        # Locate the option endorsed by the four 75%-valid experts.
        a_has_low = int(np.sum(a[:4])) > int(np.sum(b[:4]))
        low_option = a if a_has_low else b
        opposing_option = b if a_has_low else a

        # Require the expected four-expert low-validity coalition and select
        # the two conditions where the theories' absolute predictions diverge.
        low_set = tuple(np.flatnonzero(low_option).tolist())
        opposing_set = tuple(np.flatnonzero(opposing_option).tolist())
        if low_set != (0, 1, 2, 3) or opposing_set not in selected:
            continue

        chose_low = (response == 0) if a_has_low else (response == 1)
        selected_choices.append(float(chose_low))

    if len(selected_choices) == 0:
        return 0.0
    return float(np.mean(selected_choices))

```

**Observed (real) value:** 0.8819 (var=0.0059)
**Candidate trajectory (this loop):**
  - iter 1: 0.8134 (var=0.0027) (Δ vs real -0.0684)
  - iter 2: 0.4988 (var=0.0049) (Δ vs real -0.3831)
  - iter 3: 0.5044 (var=0.0048) (Δ vs real -0.3775)
  - iter 4 (current): 0.7006 (var=0.0051) (Δ vs real -0.1812)
**Other theories' values on this metric (for reference):**
- pi_9: 0.5850 (var=0.0057)
- pi_8: 0.7128 (var=0.0076)
- pi_1: 0.1353 (var=0.0104)
- pi_2: 0.5084 (var=0.0042)
- pi_3: 0.2400 (var=0.0050)
- pi_4: 0.5028 (var=0.0039)
- pi_5: 0.5072 (var=0.0053)
- pi_6: 0.4859 (var=0.0035)
- pi_7: 0.7762 (var=0.0258)
- pi_10: 0.7981 (var=0.0069)
- pi_11: 0.6425 (var=0.0062)
- pi_12: 0.5591 (var=0.0047)
- pi_13: 0.6663 (var=0.0061)
- pi_14: 0.5331 (var=0.0030)

### Experiment 17
**Design**
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Estimate the dispersion gradient after orienting every trial toward the
    # homogeneous coalition of four 75%-valid experts (features 0--3).
    cells = {("low", "A"): [], ("low", "B"): [],
             ("high", "A"): [], ("high", "B"): []}

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=np.int64)
        b = np.asarray(row["option_b_ratings"], dtype=np.int64)
        if a.size < 16 or b.size < 16:
            continue

        a_is_reference = bool(np.all(a[:4] == 1) and np.all(a[4:] == 0))
        b_is_reference = bool(np.all(b[:4] == 1) and np.all(b[4:] == 0))

        if a_is_reference == b_is_reference:
            continue

        if a_is_reference:
            reference_side = "A"
            competitor = b
            chose_reference = float(int(row["response"]) == 0)
        else:
            reference_side = "B"
            competitor = a
            chose_reference = float(int(row["response"]) == 1)

        if np.all(competitor[4:8] == 1) and int(np.sum(competitor)) == 4:
            level = "low"
        elif np.all(competitor[12:16] == 1) and int(np.sum(competitor)) == 4:
            level = "high"
        else:
            # The moderately dispersed coalition is not needed for the
            # endpoint contrast.
            continue

        cells[(level, reference_side)].append(chose_reference)

    means = {}
    for key, values in cells.items():
        if len(values) == 0:
            return float("nan")
        means[key] = float(np.mean(np.asarray(values, dtype=np.float64)))

    # Average mirrored, side-specific contrasts so fixed A/B response bias
    # cannot masquerade as a dispersion effect.
    gradient_a = means[("high", "A")] - means[("low", "A")]
    gradient_b = means[("high", "B")] - means[("low", "B")]
    return float(0.5 * (gradient_a + gradient_b))
```

**Observed (real) value:** -0.3250 (var=0.0394)
**Candidate trajectory (this loop):**
  - iter 1: -0.1856 (var=0.0077) (Δ vs real +0.1394)
  - iter 2: -0.3619 (var=0.0092) (Δ vs real -0.0369)
  - iter 3: -0.4219 (var=0.0103) (Δ vs real -0.0969)
  - iter 4 (current): -0.2706 (var=0.0060) (Δ vs real +0.0544)
**Other theories' values on this metric (for reference):**
- pi_8: 0.0656 (var=0.0171)
- pi_10: -0.0869 (var=0.0165)
- pi_1: -0.6887 (var=0.0481)
- pi_2: -0.0075 (var=0.0180)
- pi_3: -0.0756 (var=0.0152)
- pi_4: -0.0106 (var=0.0148)
- pi_5: 0.0225 (var=0.0162)
- pi_6: -0.0081 (var=0.0173)
- pi_7: 0.3625 (var=0.0685)
- pi_9: 0.0256 (var=0.0200)
- pi_11: -0.1862 (var=0.0175)
- pi_12: -0.3481 (var=0.0127)
- pi_13: -0.1138 (var=0.0119)
- pi_14: -0.3506 (var=0.0116)

### Experiment 18
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    singleton_choices = []
    broad_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        response = int(row["response"])

        # In this design, the first six experts have validity .80 and the
        # final six have validity .55. Determine which displayed option is
        # endorsed by the higher-validity coalition.
        high_direction = float(np.sum(a[:6] - b[:6]))
        m = int(np.sum(np.abs(a[:6] - b[:6])))
        if high_direction == 0.0:
            continue

        # Code 1 when the lower-validity coalition's option was selected,
        # invariant to whether that option was displayed as A or B.
        if high_direction > 0.0:
            lower_validity_choice = float(response == 1)
        else:
            lower_validity_choice = float(response == 0)

        if m == 1:
            singleton_choices.append(lower_validity_choice)
        elif m in (4, 6):
            broad_choices.append(lower_validity_choice)

    if len(singleton_choices) == 0 or len(broad_choices) == 0:
        return float("nan")

    return float(np.mean(broad_choices) - np.mean(singleton_choices))
```

**Observed (real) value:** 0.0058 (var=0.0053)
**Candidate trajectory (this loop):**
  - iter 1: 0.0479 (var=0.0079) (Δ vs real +0.0421)
  - iter 2: 0.0171 (var=0.0077) (Δ vs real +0.0113)
  - iter 3: 0.0267 (var=0.0071) (Δ vs real +0.0208)
  - iter 4 (current): 0.0208 (var=0.0045) (Δ vs real +0.0150)
**Other theories' values on this metric (for reference):**
- pi_10: 0.1996 (var=0.0174)
- pi_8: 0.0329 (var=0.0084)
- pi_1: 0.0217 (var=0.0047)
- pi_2: -0.0258 (var=0.0192)
- pi_3: -0.2842 (var=0.0112)
- pi_4: 0.0117 (var=0.0128)
- pi_5: -0.0238 (var=0.0143)
- pi_6: 0.0196 (var=0.0149)
- pi_7: 0.0146 (var=0.0242)
- pi_9: 0.0392 (var=0.0136)
- pi_11: -0.0058 (var=0.0109)
- pi_12: 0.0112 (var=0.0045)
- pi_13: 0.0700 (var=0.0088)
- pi_14: 0.0896 (var=0.0063)

### Experiment 19
**Design**
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    scores = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.size < 20 or b.size < 20:
            continue

        active = np.abs(a - b) > 0.5
        # Select the maximally dispersed 50/50/100/100 condition and its
        # option-reversed counterpart: four conflicts at indices 0:4 and four
        # at indices 16:20, with no other active features.
        is_extreme = (
            int(np.sum(active[:4])) == 4
            and int(np.sum(active[16:20])) == 4
            and int(np.sum(active)) == 8
        )
        if not is_extreme:
            continue

        # The heterogeneous coalition is the option endorsed at indices 16:20.
        hetero_is_a = float(np.sum(a[16:20])) > float(np.sum(b[16:20]))
        response = int(row["response"])
        chose_heterogeneous = (response == 0) if hetero_is_a else (response == 1)
        scores.append(float(chose_heterogeneous))

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** 0.8617 (var=0.0116)
**Candidate trajectory (this loop):**
  - iter 1: 0.9117 (var=0.0029) (Δ vs real +0.0500)
  - iter 2: 0.9467 (var=0.0018) (Δ vs real +0.0850)
  - iter 3: 0.9525 (var=0.0017) (Δ vs real +0.0908)
  - iter 4 (current): 0.9533 (var=0.0019) (Δ vs real +0.0917)
**Other theories' values on this metric (for reference):**
- pi_11: 0.6850 (var=0.0125)
- pi_10: 0.4383 (var=0.0097)
- pi_1: 0.8400 (var=0.0148)
- pi_2: 0.4925 (var=0.0099)
- pi_3: 0.9967 (var=0.0001)
- pi_4: 0.5492 (var=0.0114)
- pi_5: 0.4933 (var=0.0092)
- pi_6: 0.5200 (var=0.0113)
- pi_7: 0.1133 (var=0.0090)
- pi_8: 0.3575 (var=0.0132)
- pi_9: 0.3425 (var=0.0165)
- pi_12: 0.8750 (var=0.0092)
- pi_13: 0.8958 (var=0.0071)
- pi_14: 0.8000 (var=0.0069)

### Experiment 20
**Design**
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Majority-consistent choice rate on the three largest tally margins."""
    if len(data) == 0:
        return float("nan")

    margins = np.asarray([
        float(np.sum(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))
        for a, b in zip(data["option_a_ratings"], data["option_b_ratings"])
    ])
    responses = data["response"].to_numpy(dtype=int)

    # In this design, this selects margins 5, 7, and 9 and their reversals.
    selected = np.abs(margins) >= 5.0
    if not np.any(selected):
        return float("nan")

    # A is majority-supported for positive margins; B for negative margins.
    majority_choice = np.where(margins > 0.0, responses == 0, responses == 1)
    return float(np.mean(majority_choice[selected]))
```

**Observed (real) value:** 0.8567 (var=0.0052)
**Candidate trajectory (this loop):**
  - iter 1: 0.9025 (var=0.0027) (Δ vs real +0.0458)
  - iter 2: 0.8163 (var=0.0037) (Δ vs real -0.0404)
  - iter 3: 0.8771 (var=0.0028) (Δ vs real +0.0204)
  - iter 4 (current): 0.8467 (var=0.0024) (Δ vs real -0.0100)
**Other theories' values on this metric (for reference):**
- pi_10: 0.9071 (var=0.0048)
- pi_11: 0.7683 (var=0.0169)
- pi_1: 0.8667 (var=0.0078)
- pi_2: 0.8796 (var=0.0088)
- pi_3: 0.5863 (var=0.0044)
- pi_4: 0.8721 (var=0.0046)
- pi_5: 0.8692 (var=0.0022)
- pi_6: 0.8175 (var=0.0031)
- pi_7: 0.8971 (var=0.0021)
- pi_8: 0.8579 (var=0.0046)
- pi_9: 0.9187 (var=0.0024)
- pi_12: 0.8512 (var=0.0024)
- pi_13: 0.8075 (var=0.0051)
- pi_14: 0.8633 (var=0.0030)

### Experiment 21
**Design**
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Estimate the probability of choosing the lower-validity (50%) coalition
    # in the five intermediate conditions: 51.25%, 52%, 52.75%, 53.5%, 55%.
    lower_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        difference = a - b

        # The nonbaseline coalition occupies one adjacent feature pair. The
        # selected range corresponds to feature indices 4 through 13.
        active_later = np.flatnonzero(np.abs(difference[2:]) > 0.5) + 2
        if active_later.size == 0:
            continue
        pair_start = int(np.min(active_later))
        if pair_start < 4 or pair_start > 12:
            continue

        # Positive means the more-valid coalition endorses A; negative means
        # it endorses B. Align responses so 1 always denotes selection of the
        # opposing, 50%-valid coalition.
        high_side = float(np.sum(difference[2:]))
        response = int(row["response"])
        if high_side > 0:
            lower_choices.append(float(response == 1))
        elif high_side < 0:
            lower_choices.append(float(response == 0))

    if len(lower_choices) == 0:
        return float("nan")
    return float(np.mean(lower_choices))
```

**Observed (real) value:** 0.8667 (var=0.0085)
**Candidate trajectory (this loop):**
  - iter 1: 0.5563 (var=0.0047) (Δ vs real -0.3103)
  - iter 2: 0.8193 (var=0.0038) (Δ vs real -0.0473)
  - iter 3: 0.7297 (var=0.0057) (Δ vs real -0.1370)
  - iter 4 (current): 0.7393 (var=0.0060) (Δ vs real -0.1273)
**Other theories' values on this metric (for reference):**
- pi_11: 0.6383 (var=0.0061)
- pi_12: 0.7713 (var=0.0051)
- pi_1: 0.1560 (var=0.0099)
- pi_2: 0.5060 (var=0.0040)
- pi_3: 0.4937 (var=0.0044)
- pi_4: 0.4840 (var=0.0040)
- pi_5: 0.4897 (var=0.0034)
- pi_6: 0.6213 (var=0.0050)
- pi_7: 0.3313 (var=0.0056)
- pi_8: 0.6583 (var=0.0050)
- pi_9: 0.5213 (var=0.0048)
- pi_10: 0.5533 (var=0.0037)
- pi_13: 0.8703 (var=0.0098)
- pi_14: 0.8800 (var=0.0026)

### Experiment 22
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Score whether each response is skeptical: choosing the option opposite
    # the coalition containing the informative (> .50-valid) experts.
    ordinary_scores = []
    endpoint_scores = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        response = int(row["response"])
        signed = a - b

        ordinary_direction = float(np.sum(signed[:6]))   # 55%-valid experts
        endpoint_direction = float(np.sum(signed[6:12])) # 100%-valid experts

        if abs(ordinary_direction) > 1e-12:
            # Positive means the stronger coalition supports A, so choosing B
            # is the skeptical response. Negative reverses that coding.
            skeptical = response if ordinary_direction > 0 else 1 - response
            ordinary_scores.append(float(skeptical))
        elif abs(endpoint_direction) > 1e-12:
            skeptical = response if endpoint_direction > 0 else 1 - response
            endpoint_scores.append(float(skeptical))

    if len(ordinary_scores) == 0 or len(endpoint_scores) == 0:
        return float("nan")

    return float(np.mean(ordinary_scores) - np.mean(endpoint_scores))
```

**Observed (real) value:** -0.0058 (var=0.0033)
**Candidate trajectory (this loop):**
  - iter 1: -0.3125 (var=0.0089) (Δ vs real -0.3067)
  - iter 2: -0.0529 (var=0.0049) (Δ vs real -0.0471)
  - iter 3: -0.1338 (var=0.0055) (Δ vs real -0.1279)
  - iter 4 (current): -0.0925 (var=0.0055) (Δ vs real -0.0867)
**Other theories' values on this metric (for reference):**
- pi_12: -0.0321 (var=0.0055)
- pi_11: -0.1446 (var=0.0057)
- pi_1: -0.0183 (var=0.0028)
- pi_2: 0.0004 (var=0.0094)
- pi_3: 0.4383 (var=0.0047)
- pi_4: 0.0229 (var=0.0117)
- pi_5: 0.0138 (var=0.0150)
- pi_6: -0.0417 (var=0.0067)
- pi_7: -0.6117 (var=0.0473)
- pi_8: -0.0883 (var=0.0095)
- pi_9: -0.1883 (var=0.0185)
- pi_10: -0.2504 (var=0.0061)
- pi_13: -0.0554 (var=0.0066)
- pi_14: -0.0254 (var=0.0050)

### Experiment 23
**Design**
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Reversal-balanced choice rate for the high-spread skewed coalition."""
    y_choice_by_orientation = {True: [], False: []}
    target_active = np.arange(18, 26, dtype=int)

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=int)
        b = np.asarray(row["option_b_ratings"], dtype=int)
        if a.size < 26 or b.size < 26:
            continue

        active = np.flatnonzero(a != b)
        if not np.array_equal(active, target_active):
            continue

        # In the largest-spread pair, expert 25 is the unique upper-tail
        # member of the positively skewed coalition Y.
        y_is_a = bool(a[25] > b[25])
        response = int(row["response"])
        chose_y = float(response == (0 if y_is_a else 1))
        y_choice_by_orientation[y_is_a].append(chose_y)

    orientation_means = []
    for y_is_a in (True, False):
        values = y_choice_by_orientation[y_is_a]
        if len(values) > 0:
            orientation_means.append(float(np.mean(values)))

    if len(orientation_means) == 0:
        return 0.5
    return float(np.mean(orientation_means))
```

**Observed (real) value:** 0.5000 (var=0.0088)
**Candidate trajectory (this loop):**
  - iter 1: 0.5225 (var=0.0066) (Δ vs real +0.0225)
  - iter 2: 0.3106 (var=0.0110) (Δ vs real -0.1894)
  - iter 3: 0.4669 (var=0.0114) (Δ vs real -0.0331)
  - iter 4 (current): 0.4556 (var=0.0092) (Δ vs real -0.0444)
**Other theories' values on this metric (for reference):**
- pi_13: 0.5337 (var=0.0069)
- pi_12: 0.4281 (var=0.0107)
- pi_1: 0.8306 (var=0.0151)
- pi_2: 0.4850 (var=0.0078)
- pi_3: 0.5300 (var=0.0079)
- pi_4: 0.4838 (var=0.0101)
- pi_5: 0.5100 (var=0.0067)
- pi_6: 0.5144 (var=0.0109)
- pi_7: 0.4450 (var=0.0091)
- pi_8: 0.4925 (var=0.0088)
- pi_9: 0.4850 (var=0.0116)
- pi_10: 0.4881 (var=0.0077)
- pi_11: 0.4625 (var=0.0112)
- pi_14: 0.4625 (var=0.0068)

### Experiment 24
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Rate of choosing the option endorsed by the unilateral coalition."""
    if data is None or len(data) == 0:
        return float("nan")

    scores = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        margin = float(np.sum(a - b))
        if not np.isfinite(margin) or abs(margin) < 1e-12:
            continue

        response = int(row["response"])
        chose_endorsed = ((margin > 0.0 and response == 0) or
                          (margin < 0.0 and response == 1))
        scores.append(float(chose_endorsed))

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** 0.5763 (var=0.0023)
**Candidate trajectory (this loop):**
  - iter 1: 0.6146 (var=0.0025) (Δ vs real +0.0383)
  - iter 2: 0.5494 (var=0.0026) (Δ vs real -0.0269)
  - iter 3: 0.6217 (var=0.0035) (Δ vs real +0.0454)
  - iter 4 (current): 0.5837 (var=0.0048) (Δ vs real +0.0075)
**Other theories' values on this metric (for reference):**
- pi_12: 0.1106 (var=0.0028)
- pi_13: 0.5363 (var=0.0187)
- pi_1: 0.8290 (var=0.0100)
- pi_2: 0.8840 (var=0.0070)
- pi_3: 0.8248 (var=0.0030)
- pi_4: 0.6258 (var=0.0040)
- pi_5: 0.7140 (var=0.0028)
- pi_6: 0.4058 (var=0.0090)
- pi_7: 0.6960 (var=0.0175)
- pi_8: 0.4792 (var=0.0113)
- pi_9: 0.4990 (var=0.0291)
- pi_10: 0.2062 (var=0.0081)
- pi_11: 0.2412 (var=0.0072)
- pi_14: 0.5962 (var=0.0031)

### Experiment 25
**Design**
  A=[1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]  B=[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Estimate the linear change in preference for the 70%-expert coalition as
    # its three fillers are progressively upgraded from 50% to 60% validity.
    records = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.size < 7 or b.size < 7:
            continue

        # Feature 6 is the fixed 70%-valid expert and uniquely identifies the
        # focal coalition. Its display side is reversed in half the trials.
        if a[6] > b[6]:
            focal_is_a = True
            focal = a
        elif b[6] > a[6]:
            focal_is_a = False
            focal = b
        else:
            continue

        # Features 3--5 are the three possible 60%-valid fillers. Their sum is
        # the designed progression level k in {0, 1, 2, 3}.
        k = int(np.rint(np.sum(focal[3:6])))
        if k < 0 or k > 3:
            continue

        response = int(row["response"])
        chose_focal = float((response == 0) if focal_is_a else (response == 1))
        records.append((focal_is_a, k, chose_focal))

    if len(records) == 0:
        return 0.0

    # Compute the trend separately for the two display orientations, then
    # average. This cancels stable A/B response biases while giving all four
    # experimental conditions equal weight.
    orientation_slopes = []
    for focal_is_a in (False, True):
        condition_means = []
        levels = []
        for k in range(4):
            vals = [y for side, level, y in records
                    if side == focal_is_a and level == k]
            if len(vals) > 0:
                levels.append(float(k))
                condition_means.append(float(np.mean(vals)))

        if len(levels) >= 2:
            x = np.asarray(levels, dtype=float)
            y = np.asarray(condition_means, dtype=float)
            xc = x - np.mean(x)
            denom = float(np.sum(xc ** 2))
            if denom > 0.0:
                orientation_slopes.append(float(np.sum(xc * y) / denom))

    if len(orientation_slopes) == 0:
        return 0.0

    # Multiplication by three expresses the fitted trend as the predicted
    # preference change from k=0 to k=3.
    return float(3.0 * np.mean(orientation_slopes))
```

**Observed (real) value:** -0.3065 (var=0.0176)
**Candidate trajectory (this loop):**
  - iter 1: -0.0350 (var=0.0029) (Δ vs real +0.2715)
  - iter 2: -0.0510 (var=0.0049) (Δ vs real +0.2555)
  - iter 3: -0.3022 (var=0.0186) (Δ vs real +0.0043)
  - iter 4 (current): -0.0475 (var=0.0044) (Δ vs real +0.2590)
**Other theories' values on this metric (for reference):**
- pi_13: 0.4533 (var=0.0337)
- pi_14: -0.0902 (var=0.0067)
- pi_1: 0.0125 (var=0.0113)
- pi_2: -0.0430 (var=0.0145)
- pi_3: 0.0315 (var=0.0064)
- pi_4: -0.0068 (var=0.0180)
- pi_5: 0.0112 (var=0.0203)
- pi_6: -0.2332 (var=0.0153)
- pi_7: 0.0670 (var=0.0139)
- pi_8: -0.0320 (var=0.0077)
- pi_9: -0.0730 (var=0.0151)
- pi_10: 0.0152 (var=0.0045)
- pi_11: -0.0285 (var=0.0096)
- pi_12: -0.0690 (var=0.0064)

### Experiment 26
**Design**
  A=[1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    captures = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)

        # Features 0:8 define the heterogeneous coalition; features 8:12
        # define the homogeneous 51%-valid coalition. Determine which option
        # is supported by the former without relying on display side.
        mixed_margin = float(np.sum(a[:8]) - np.sum(b[:8]))
        if mixed_margin == 0.0:
            continue

        mixed_is_a = mixed_margin > 0.0
        mixed_ratings = a if mixed_is_a else b
        n_perfect_endorsers = int(np.sum(mixed_ratings[:4]))

        # The maximally heterogeneous condition has one 100%-valid endorser
        # and three 50%-valid endorsers.
        if n_perfect_endorsers != 1:
            continue

        response = int(row["response"])
        chose_mixed = (mixed_is_a and response == 0) or (
            (not mixed_is_a) and response == 1
        )
        captures.append(float(chose_mixed))

    if len(captures) == 0:
        return float("nan")
    return float(np.mean(captures))
```

**Observed (real) value:** 0.8683 (var=0.0070)
**Candidate trajectory (this loop):**
  - iter 1: 0.9150 (var=0.0030) (Δ vs real +0.0467)
  - iter 2: 0.9508 (var=0.0025) (Δ vs real +0.0825)
  - iter 3: 0.9542 (var=0.0028) (Δ vs real +0.0858)
  - iter 4 (current): 0.9283 (var=0.0033) (Δ vs real +0.0600)
**Other theories' values on this metric (for reference):**
- pi_14: 0.2225 (var=0.0071)
- pi_13: 0.7600 (var=0.0287)
- pi_1: 0.8550 (var=0.0183)
- pi_2: 0.4992 (var=0.0105)
- pi_3: 0.9992 (var=0.0000)
- pi_4: 0.5467 (var=0.0118)
- pi_5: 0.4808 (var=0.0090)
- pi_6: 0.2417 (var=0.0074)
- pi_7: 0.1750 (var=0.0274)
- pi_8: 0.3258 (var=0.0111)
- pi_9: 0.3617 (var=0.0152)
- pi_10: 0.3475 (var=0.0110)
- pi_11: 0.4842 (var=0.0171)
- pi_12: 0.5133 (var=0.0104)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed Diversity-Legitimacy Attribution family and captures several important qualitative effects: anti-validity choices in Experiments 7, 10–12 and 16; heterogeneous-coalition preference in Experiments 19 and 26; near-null results in Experiments 13, 15, 18, and 23; and modest unilateral following in Experiment 24. Nevertheless, the fit is not experiment-invariant. Large failures occur in Experiment 21 (0.556 versus 0.867), Experiment 25 (-0.035 versus -0.307), Experiment 5 (-0.699 versus -0.038), Experiment 22 (-0.313 versus -0.006), Experiment 3 (0.236 versus 0.451), and Experiment 2 (226 versus 89). Experiment 1 also follows the intended negative direction but is too tally-dominant (TTB-conflict choice 0.261 versus 0.403). The model exaggerates breadth effects in Experiment 14 (0.469 versus 0.178) and understates minority/high-validity choices in Experiment 9 (0.108 versus 0.325). Most simulated subject variances are also substantially below the observed values, indicating overly uniform or saturated channel recruitment.
Rationale: The accepted candidate is a promising base, but its arbitration and coalition scaling need in-family recalibration. First, coordination is too strongly coupled to coalition breadth and absolute authority through formation times mean_strength^extremity. This creates an excessive size-dependent reversal in Experiment 5 and an overly steep breadth gradient in Experiment 14, while making slightly-above-neutral homogeneous coalitions too weak to reverse in Experiment 21. Replace that product with a smoother, rapidly saturating breadth normalization and a low-threshold or contrast-normalized authority gate. The coordination channel should distinguish a 51–55% coalition from a neutral coalition without making a 100% coalition categorically more reversible; this should also remove the spurious ordinary-versus-endpoint difference in Experiment 22. Second, tally arbitration is too sharply margin-dependent: Experiment 2 shows far too much evidence for margin-sensitive tallying, even while majority rates in Experiments 4 and 6 are slightly low. Raise baseline count recruitment but flatten its dependence on margin, for example by reducing count_gate_gain/count curvature while modestly increasing count_bias or count-channel precision. Let stronger bilateral coordination handle small-margin conflicts; that can raise minority choices in Experiments 1, 3, and 9 without sacrificing clear-majority performance. Third, the diversity channel correctly produces high capture in Experiments 19 and 26 but barely responds when neutral fillers are upgraded in Experiment 25. Increase the local sensitivity of the weak-background/upper-surprise interaction across the 50–60% range, rather than globally increasing diversity strength, since heterogeneous capture is already slightly too high. Finally, widen attribution-style effects modestly or avoid saturated channel probabilities so the model can reproduce observed between-subject variance. These are calibration and normalization changes within the prescribed mechanism, not a change of theory family.

**Outcome of your advice:** iter 1 candidate loss=0.1842 -> iter 2 candidate loss=0.1868 -> the gate marked it REJECTED.

### Iteration 2
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed Diversity-Legitimacy Attribution family, and the rejected recalibration achieved substantial local improvements in Experiments 2, 5, 21, and 22. However, its aggregate loss increased from 0.1842 to 0.1868 because those gains came with severe cross-experiment regressions. Majority performance deteriorated in Experiments 4 (0.674 versus 0.821) and 6 (0.720 versus 0.846), the successful anti-reliability pattern in Experiment 8 became too extreme (-5.395 versus -3.537), and Experiment 16 collapsed from 0.813 to 0.499 against an observed 0.882. Previously near-null effects also became spurious in Experiments 13 (0.130 versus 0.009) and 23 (0.311 versus 0.500). Meanwhile, central failures were barely changed: minority choice remains far too low in Experiments 3 and 9, the breadth gradient remains excessive in Experiment 14, and the required negative filler-upgrade trend remains much too weak in Experiment 25. Heterogeneous-coalition capture is now somewhat excessive in Experiments 19 and 26. Thus the rapid formation saturation, relative coordination normalization, and globally flatter count arbitration overcorrected the accepted model rather than yielding experiment-invariant improvement.
Rationale: Because this candidate was rejected by the accept gate, the next proposal should be a new, smaller modification of the unchanged iter-1 accepted base. The previous recommendation to jointly flatten count recruitment, rapidly saturate coalition formation, and use relative coordination normalization was empirically rejected and should not be repeated wholesale. Instead, reshape only the tally evidence into a smooth Hill-like curve: weak evidence at a one-vote margin, a rapid rise over intermediate margins, and a plateau for large margins. This can raise minority choices in Experiments 1, 3, and 9 and reduce Experiment 2's excessive margin-sensitive Bayes factor without lowering the clear-majority rates in Experiments 4, 6, and 20, as the current broad reduction in count recruitment did. For Experiments 21 and 22, retain the accepted model's breadth formation and arbitration but compress authority only inside the coordination contrast with a low-threshold saturating transform; avoid the current ratio normalization by total coordination, which appears responsible for distortions in Experiments 8, 13, 16, and 23. This is a localized way to make 51–55% homogeneous claims sufficiently salient while reducing the ordinary-versus-endpoint discrepancy. Finally, address Experiment 25 through the representation of upper-tail distinctiveness rather than another global increase in diversity sensitivity: compute surprise relative to the nonmaximum coalition background and interact it with the breadth of that background. This should make upgrading fillers from 50% to 60% sharply reduce the exceptional expert's legitimacy while allowing diversity strength to be reduced enough to preserve the observed, non-ceiling capture in Experiments 19 and 26 and the null in Experiment 23. Do not widen trait spans further until pooled effects are stable; the current widening did not solve the general variance mismatch and produced unstable condition-level behavior.

**Outcome of your advice:** iter 2 candidate loss=0.1868 -> iter 3 candidate loss=0.1792 -> the gate marked it ACCEPTED.

### Iteration 3 (most recent)
Verdict: regenerate
Interpretation: The accepted iteration is a meaningful improvement and remains faithful to the prescribed Diversity-Legitimacy Attribution family. The localized changes successfully moved Experiment 1 from 0.261 to 0.363, Experiment 3 from 0.236 to 0.381, Experiment 5 from -0.699 to -0.084, and—most importantly—Experiment 25 from -0.035 to -0.302 against -0.307 observed. It also preserved reasonable majority behavior in Experiments 4, 6, and 20 and retained the required heterogeneous-coalition preferences in Experiments 19 and 26. Thus the prior accepted advice concerning a Hill-shaped tally and nonmaximum-background surprise was productive. Nevertheless, the model is not yet experiment-invariant. Experiment 2 still shows much too much margin-sensitive tally evidence (201.9 versus 89.1). Experiment 16 has collapsed to chance (0.504 versus 0.882), Experiment 14 has an excessive breadth gradient (0.493 versus 0.178), and Experiment 9 remains far too strongly anti-high-validity (0.086 versus 0.325). Coordination skepticism is also too extreme in Experiment 8 (-5.053 versus -3.537), while Experiment 21 still underpredicts choices of the lower-validity coalition (0.730 versus 0.867) and Experiment 22 retains an unwanted ordinary-versus-endpoint difference (-0.134 versus -0.006). Experiment 15 additionally flips its discontinuous point metric from the correct 0 to 1. Diversity effects are directionally correct but somewhat saturated in Experiments 19 and 26 (about 0.95 versus 0.86–0.87), with variances correspondingly too small.
Rationale: Continue from this accepted candidate, but make another localized in-family adjustment rather than undoing the successful Hill tally or the new background-relative legitimacy interaction. The clearest defect is the sidewise low-threshold saturation of coordination: mapping both coalitions through 1-exp(-coordination/scale) nearly saturates them before taking their difference, erasing useful contrasts in Experiment 16 while producing unstable reversals in Experiments 8, 9, 15, and 22. Replace the saturated-only contrast with a convex blend of raw and compressed coordination contrasts, or retain a small raw residual after compression. Give weak 51–55% coalitions salience through a smooth low-authority floor inside claim formation or authority coding rather than by saturating each side's final coordination score. This should preserve the accepted improvement in Experiment 21 while restoring the strong homogeneous-versus-dispersed distinction in Experiment 16 and reducing endpoint artifacts. The breadth dependence of coordination should also saturate only modestly earlier beyond coalition breadth two; avoid the wholesale rapid-formation recalibration previously rejected by the gate. That targeted flattening is needed for Experiment 14 without disturbing unilateral endorsement or large-majority cases. For Experiment 2, retain the Hill form but rebalance evidence amplitude against recruitment: slightly lower the count channel's plateau strength while modestly increasing count recruitment only at high decisiveness. This can reduce ceiling-level, margin-dependent choice probabilities in Experiment 2 while protecting Experiments 4, 6, and 20 and leaving weak one-vote evidence in Experiments 1 and 3 intact. Finally, preserve the successful Experiment 25 legitimacy representation; address the mild ceilings in Experiments 19 and 26 with only a small saturation of diversity-channel recruitment or slope, not a global reduction of the weak-background interaction. Do not widen trait spans yet, because prior variance-focused widening was rejected and pooled coordination calibration remains the primary problem.

**Outcome of your advice:** iter 3 candidate loss=0.1792 -> CURRENT candidate loss=0.1577 -> the gate marked it ACCEPTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate is faithful to the prescribed Diversity-Legitimacy Attribution family, and the latest accepted adjustment materially improves aggregate loss. In particular, retaining a raw coordination residual helps Experiment 16 recover from 0.504 to 0.701, reduces Experiment 2's excessive tally Bayes factor from 201.9 to 163.4, restores the correct binary result in Experiment 15, and preserves the qualitative heterogeneous-coalition effects in Experiments 19 and 26. Experiment 20, Experiment 24, and the near-null Experiments 13, 18, and 23 are also well captured. Nevertheless, the theory is not yet experiment-invariant. The current recruitment changes reverse several of the previous accepted candidate's strongest successes: Experiment 25 collapses from -0.302 to -0.048 versus -0.307 observed, Experiment 5 deteriorates from -0.084 to -0.519 versus -0.038, and minority/high-validity choices fall in Experiments 1 and 3. Majority dominance is now too weak in Experiments 4 and 6, while Experiment 9 remains extremely low at 0.083 versus 0.325. Coordination reversal is excessive in Experiment 8, but still insufficient in the breadth-supported conflicts of Experiments 16 and 21. Most conspicuously, Experiment 14's breadth gradient grows to 0.513 versus 0.178. Experiment 2 remains substantially too margin-sensitive, and the heterogeneous effects in Experiments 19 and 26 remain too close to ceiling. Subject variances are generally too small, especially where channel probabilities saturate.",
  "verdict": "regenerate",
  "rationale": "The previous recommendation was accepted by the gate, so preserve its successful core: the Hill-shaped tally, background-relative diversity interaction, and a residual unsaturated coordination contrast. The new problem is that raw coordination magnitude is being used both directionally and to recruit the coordination channel. Recruitment through the strongest absolute coordination appears to suppress count and diversity channels even when the relevant between-option contrast is small. This likely explains the simultaneous loss of Experiment 25's diversity trend, weaker majorities in Experiments 4 and 6, and the exaggerated breadth effect in Experiment 14. Decouple these roles: retain a small raw residual in the directional coordination contrast, but recruit coordination using bilateral conflict, unresolvedness, and bounded contrast magnitude rather than the strongest coalition's absolute raw coordination. Normalize the raw directional residual by coalition formation, or compare the homogeneity-authority component after removing shared breadth. Breadth can remain in the bilateral gate. This should preserve the recovered coherent-versus-dispersed distinction in Experiment 16 without allowing equal-breadth coalitions to generate the excessive size gradient in Experiment 14 or the endpoint discrepancy in Experiment 22. Add a smooth background-breadth factor specifically to pragmatic reversal so singleton-versus-singleton conflicts are less extreme; that targets Experiments 8 and 10 while preserving the breadth-two reversal needed in Experiment 21 and breadth-four effect in Experiment 16. Do not globally weaken coordination, because Experiments 16 and 21 still require more reversal than currently predicted. The latest global reduction of diversity strength was also poorly targeted: it only modestly reduced Experiment 26 but destroyed the previously accurate Experiment 25 trend. Restore the prior diversity evidence range and control ceilings by lowering its evidence slope or cap while increasing recruitment sensitivity in the moderate-legitimacy region. That separates local sensitivity to filler upgrades in Experiment 25 from asymptotic choice strength in Experiments 19 and 26. Finally, retain the count Hill form but slightly compress only its upper-margin differences; Experiment 2 remains too tally-margin-sensitive, whereas broad majority underprediction should primarily be repaired by preventing irrelevant coordination recruitment, not by globally raising count strength. Defer widening trait spans until these pooled channel-coupling errors are corrected."
}
```

## Usage

```json
{
  "input_tokens": 47900,
  "output_tokens": 2117
}
```
