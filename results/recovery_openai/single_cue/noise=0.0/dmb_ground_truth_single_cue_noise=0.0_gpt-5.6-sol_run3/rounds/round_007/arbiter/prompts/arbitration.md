# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_8" and "pi_9") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_8" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_9" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_8
Contextual Anti-Reliability Arbitration theory proposes that communicated validity numbers are interpreted partly as pragmatic claims whose assertiveness can undermine their credibility. Every above-chance claim therefore generates continuous evidence against the option endorsed by that expert, rather than being categorically reversed by a special subgroup. The magnitude of this anti-reliability response varies continuously across people. Near-perfect claims additionally evoke endpoint distrust because unusually strong assurances are perceived as especially suspicious. When an endpoint cue faces a substantive opposing claim, both its ordinary and endpoint-specific distrust contributions are attenuated; against a genuinely neutral cue, both remain intact. Claims around 95% are more strongly compressed in the ordinary channel and do not receive the endpoint bonus, separating near-endpoint skepticism from distrust of literal certainty. Anti-reliability evidence is accumulated separately from the raw vote tally and divisively normalized by coalition breadth, within-side compositional heterogeneity, and sparse across-side claim mismatch. The latter detects heterogeneous singleton conflicts that within-side dispersion cannot detect, while exempting an isolated certainty claim opposed by a neutral cue. A small smooth boost preserves moderate-claim distrust in broad but not maximally dense coalitions. Its influence is strongest when the validity contrast is concentrated and the tally is tied or weak, while it decays smoothly as the absolute tally margin grows. Raw tally evidence additionally receives a continuously heterogeneous, intermediate-margin decision-temperature adjustment. The mechanism remains continuous, option-swap symmetric, and history independent because choices are followed by no accuracy feedback.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Contextual Anti-Reliability Arbitration. History is intentionally ignored:
    # without outcome feedback, prior choices cannot reveal expert accuracy.
    stimulus = np.asarray(state, dtype=np.float64)
    if stimulus.ndim != 2 or stimulus.shape[0] != 2:
        raise ValueError(
            f"Expected state with shape (2, n_features); got {stimulus.shape}."
        )

    n_features = stimulus.shape[1]
    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} does not match "
            f"n_features {n_features}."
        )

    # Positive endorsements favor A and negative endorsements favor B.
    signed_votes = np.sign(stimulus[0] - stimulus[1])
    active = signed_votes != 0.0
    tally_margin = float(np.sum(signed_votes))
    active_count = int(np.count_nonzero(active))

    # Encode the assertiveness of every communicated above-chance claim.
    # Moderate claims already carry pragmatic suspicion. The ordinary channel
    # is strongly compressed near 95%, preventing near-endpoint claims from
    # behaving like literal-certainty claims.
    v = np.clip(validities, 0.5, 1.0)
    above_chance = np.clip(2.0 * (v - 0.5), 0.0, 1.0)
    ordinary_base = np.power(above_chance, 0.82)

    def logistic(x):
        if x >= 0.0:
            return 1.0 / (1.0 + np.exp(-x))
        ex = np.exp(x)
        return ex / (1.0 + ex)

    upper_compression = np.array(
        [logistic(50.0 * (float(x) - 0.86)) for x in above_chance],
        dtype=np.float64,
    )
    ordinary_claim = ordinary_base * (1.0 - 0.55 * upper_compression)

    # Endpoint distrust is reserved for claims extremely close to certainty.
    endpoint_low = logistic(100.0 * (0.0 - 0.98))
    endpoint_high = logistic(100.0 * (1.0 - 0.98))
    endpoint_curve = np.array(
        [
            (logistic(100.0 * (float(x) - 0.98)) - endpoint_low)
            / (endpoint_high - endpoint_low)
            for x in above_chance
        ],
        dtype=np.float64,
    )
    endpoint_curve = np.clip(endpoint_curve, 0.0, 1.0)

    if active_count == 0:
        anti_reliability = 0.0
    else:
        active_signs = signed_votes[active]
        active_ordinary = ordinary_claim[active]
        active_endpoint = endpoint_curve[active]
        active_above_chance = above_chance[active]

        # Endpoint distrust is preserved against a neutral 50% opponent, but
        # attenuated by the smooth maximum ordinary strength on the opposing
        # side. The attenuation applies to the endpoint cue's total distrust,
        # not merely to its endpoint bonus.
        nonendpoint_ordinary = active_ordinary * (1.0 - active_endpoint)
        endpoint_focus = np.ones(active_count, dtype=np.float64)
        for i in range(active_count):
            opposing = nonendpoint_ordinary[active_signs != active_signs[i]]
            if opposing.size > 0:
                attention_logits = 6.0 * opposing
                attention_logits -= np.max(attention_logits)
                attention = np.exp(attention_logits)
                attention /= attention.sum()
                opposing_peak = float(np.dot(attention, opposing))
                endpoint_focus[i] = 1.0 / (
                    1.0 + 3.0 * opposing_peak ** 1.4
                )

        focused_endpoint = active_endpoint * endpoint_focus
        active_claims = (
            active_ordinary * (1.0 - active_endpoint)
            + (
                active_ordinary
                + float(parameters["endpoint_distrust"])
            ) * focused_endpoint
        )
        signed_claim_contrast = float(np.dot(active_signs, active_claims))
        total_claim_mass = float(np.sum(active_claims))

        # Relative contrast measures whether distrust is directionally
        # concentrated rather than spread across two similarly assertive sides.
        concentration = abs(signed_claim_contrast) / (total_claim_mass + 1e-12)
        concentration = float(np.clip(concentration, 0.0, 1.0))
        focus_floor = float(parameters["concentration_floor"])
        focus = focus_floor + (1.0 - focus_floor) * concentration ** 1.35

        # Estimate heterogeneity within each endorsing coalition. Mixing claims
        # of substantially different strength on the same side makes their
        # pragmatic interpretation less coherent and hence more strongly diluted.
        positive_claims = active_claims[active_signs > 0.0]
        negative_claims = active_claims[active_signs < 0.0]

        def side_dispersion(values):
            if values.size <= 1:
                return 0.0
            mean_value = float(np.mean(values))
            return float(np.mean((values - mean_value) ** 2))

        positive_dispersion = side_dispersion(positive_claims)
        negative_dispersion = side_dispersion(negative_claims)
        within_dispersion = (
            positive_claims.size * positive_dispersion
            + negative_claims.size * negative_dispersion
        ) / float(active_count)

        breadth_exponent = float(parameters["coalition_dilution_exponent"])
        breadth_normalizer = float(active_count) ** breadth_exponent
        heterogeneity_normalizer = 1.0 + (
            float(parameters["composition_dilution"])
            * float(active_count)
            * within_dispersion
        )

        # Within-side dispersion is exactly zero when each option has only one
        # active endorser. Across-active dispersion therefore supplies a second,
        # smoothly sparse-weighted composition diagnostic. It is switched off
        # for a literal endpoint claim against a neutral claim, preserving the
        # isolated 100%-versus-50% distrust response.
        across_dispersion = float(np.mean((active_claims - np.mean(active_claims)) ** 2))
        sparse_weight = (2.0 / float(active_count)) ** 2
        endpoint_neutral_exemption = float(np.max(active_endpoint)) * (
            1.0 - float(np.max(nonendpoint_ordinary))
        )
        across_gate = 1.0 - np.clip(endpoint_neutral_exemption, 0.0, 1.0)
        across_normalizer = 1.0 + (
            float(parameters["across_composition_dilution"])
            * sparse_weight
            * across_gate
            * across_dispersion
        )

        contextual_contrast = (
            signed_claim_contrast * focus
            / (
                breadth_normalizer
                * heterogeneity_normalizer
                * across_normalizer
                + 1e-12
            )
        )

        # Moderate pragmatic claims receive a small accumulation boost in broad
        # but not maximally dense coalitions. The smooth validity and breadth
        # gates avoid strengthening sparse 95%-claim conflicts or dense diffuse
        # conflicts indiscriminately.
        strongest_assertiveness = float(np.max(active_above_chance))
        moderate_gate = logistic(12.0 * (0.72 - strongest_assertiveness))
        breadth_peak = (
            (1.0 - np.exp(-max(active_count - 2, 0) / 2.0))
            * np.exp(-((active_count - 7.0) / 4.0) ** 2)
        )
        contextual_contrast *= float(
            1.0 + 0.15 * moderate_gate * breadth_peak
        )

        # The anti-reliability signal is bounded and continuously heterogeneous
        # across subjects. Its sign opposes the communicated validity contrast.
        accumulated = np.tanh(
            float(parameters["anti_accumulation"]) * contextual_contrast
        )
        anti_reliability = -float(parameters["anti_reliability_strength"]) * float(accumulated)

        # A smooth arbitration function cedes control to the tally as its margin
        # becomes decisive; there are no exact-tie or one-margin special cases.
        margin_scale = float(parameters["margin_protection_scale"])
        margin_power = float(parameters["margin_protection_power"])
        margin_gate = 1.0 / (
            1.0 + (abs(tally_margin) / margin_scale) ** margin_power
        )
        anti_reliability *= float(margin_gate)

    # Raw votes retain their categorical status but saturate gradually, limiting
    # overconfidence while preserving dominance by large vote coalitions.
    tally_evidence = 4.5 * np.tanh(tally_margin / 4.5)
    total_evidence = tally_evidence + anti_reliability

    evidence_scale = float(parameters["evidence_scale"])
    normalized_evidence = float(np.tanh(total_evidence / evidence_scale))
    decision_variable = float(parameters["choice_sensitivity"]) * normalized_evidence

    # Confidence separates most strongly at intermediate margins. The bounded
    # gain is zero through margin one and decreases again for very large margins,
    # increasing heterogeneous margin sensitivity without indiscriminately
    # sharpening every decisive coalition.
    margin_excess = max(abs(tally_margin) - 1.0, 0.0)
    intermediate_margin_shape = (
        0.5 * margin_excess * np.exp(1.0 - margin_excess / 2.0)
        if margin_excess > 0.0 else 0.0
    )
    margin_temperature = 1.0 + float(parameters["margin_temperature_gain"]) * intermediate_margin_shape
    decision_variable *= float(margin_temperature)

    # Positive evidence favors A. Symmetric logits guarantee that swapping the
    # displayed options swaps their probabilities exactly.
    logits = np.array(
        [0.5 * decision_variable, -0.5 * decision_variable],
        dtype=np.float64,
    )
    logits -= np.max(logits)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()

    lapse = float(parameters["lapse_rate"])
    probabilities = (1.0 - lapse) * probabilities + 0.5 * lapse
    probabilities = np.clip(probabilities, 0.0, 1.0)
    probabilities /= probabilities.sum()
    return probabilities

`policy(probs) -> int`:
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = np.clip(probabilities, 0.0, None)
    total = float(np.sum(probabilities))
    if not np.isfinite(total) or total <= 0.0:
        probabilities = np.full(
            probabilities.size, 1.0 / probabilities.size, dtype=np.float64
        )
    else:
        probabilities /= total
    return int(np.random.choice(probabilities.size, p=probabilities))

## THEORY 2 — pi_9
Finite-Pool Pragmatic Claim Competition theory proposes that binary expert endorsements generate two separable signals. A categorical vote channel counts every discriminating expert equally and therefore represents coalition consensus independently of communicated validity. A pragmatic claim channel interprets an expert's stated validity as the assertiveness of the expert's claim. Claim assertiveness is encoded by a smooth linear function of the communicated scale, including from 90% through 100%. Most people respond skeptically to assertive claims, although the polarity and magnitude of this interpretation vary continuously across individuals and can include weakly conventional interpretations. Claim processing draws on a finite attentional pool. Every discriminating expert, including a nominally 50%-valid expert whose own claim carries no directional strength, competes for that pool. Competition increases both with active-set size and with similarity among simultaneously active claims. Consequently, an isolated high-validity claim can receive substantial attention, whereas its influence declines when embedded in a broad or compositionally redundant coalition. This single allocation rule produces breadth and composition effects without endpoint bonuses, validity notches, or condition-specific exemptions. Pragmatic claim evidence is bounded, while categorical tally evidence continues to increase with vote margin. Skepticism can thus reverse tied or weak tallies but normally yields to a decisive raw-vote coalition. In the absence of outcome feedback, neither signal is recalibrated from choice history.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Finite-Pool Pragmatic Claim Competition model. History is deliberately
    # ignored because previous choices provide no feedback about cue accuracy.
    stimulus = np.asarray(state, dtype=np.float64)
    if stimulus.ndim != 2 or stimulus.shape[0] != 2:
        raise ValueError(
            f"Expected state with shape (2, n_features); got {stimulus.shape}."
        )

    n_features = stimulus.shape[1]
    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} does not match "
            f"n_features {n_features}."
        )

    # +1 denotes a unique endorsement of A, -1 a unique endorsement of B.
    signed_votes = np.sign(stimulus[0] - stimulus[1])
    active = signed_votes != 0.0
    tally_margin = float(np.sum(signed_votes))

    # Categorical votes and interpreted numerical claims are represented in
    # separate channels. This exactly linear assertiveness code has no special
    # status for 95% or 100% and therefore has zero intrinsic 90/95/100
    # curvature.
    v = np.clip(validities, 0.5, 1.0)
    claim_assertiveness = 2.0 * (v - 0.5)

    if not np.any(active):
        allocated_claim_contrast = 0.0
    else:
        active_signs = signed_votes[active]
        active_claims = claim_assertiveness[active]
        active_count = active_claims.size

        # All discriminating experts enter the competition matrix. Thus a 50%
        # expert has zero directional claim strength but still consumes claim-
        # interpretation capacity. Competition has one general form: a global
        # component plus a similarity-dependent component.
        width = float(parameters["similarity_width"])
        competition_floor = float(parameters["competition_floor"])
        distances = np.abs(active_claims[:, None] - active_claims[None, :])
        similarity = np.exp(-distances / width)
        pairwise_competition = (
            competition_floor + (1.0 - competition_floor) * similarity
        )
        np.fill_diagonal(pairwise_competition, 0.0)
        competition_load = np.sum(pairwise_competition, axis=1)

        # Finite capacity determines the attention retained by each claim.
        # Similar claims in broad coalitions strongly compete, while isolated
        # claims receive most of the available processing weight.
        capacity = float(parameters["attentional_capacity"])
        attention = capacity / (capacity + competition_load)
        allocated_claim_contrast = float(
            np.dot(active_signs, attention * active_claims)
        )

    # Pragmatic interpretation is continuous across subjects. Positive values
    # encode skepticism/reactance and reverse the communicated claim contrast;
    # zero ignores it; negative values produce a conventional interpretation.
    # The declared range makes skepticism predominant without introducing a
    # categorical latent class.
    accumulation = float(parameters["claim_accumulation"])
    bounded_claim = float(np.tanh(accumulation * allocated_claim_contrast))
    claim_evidence = -float(parameters["claim_construal"]) * bounded_claim

    # The categorical vote channel grows smoothly with margin. Its saturation
    # level exceeds the maximum typical claim contribution, so large coalitions
    # reliably dominate even though claim evidence can reverse weak margins.
    tally_scale = float(parameters["tally_saturation"])
    tally_evidence = (
        float(parameters["tally_sensitivity"])
        * tally_scale
        * np.tanh(tally_margin / tally_scale)
    )

    total_evidence = tally_evidence + claim_evidence
    decision_variable = float(parameters["choice_sensitivity"]) * total_evidence

    # Positive evidence favors A. Symmetric logits ensure exact option-swap
    # invariance.
    logits = np.array(
        [0.5 * decision_variable, -0.5 * decision_variable],
        dtype=np.float64,
    )
    logits -= np.max(logits)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()

    lapse = float(parameters["lapse_rate"])
    probabilities = (1.0 - lapse) * probabilities + 0.5 * lapse
    probabilities = np.clip(probabilities, 0.0, 1.0)
    probabilities /= probabilities.sum()
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

## EXPERIMENT 1 (proposed by pi_8)

### DESIGN
**Validities (n_features=5):** [1.0, 0.5, 0.7, 0.8, 0.9]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 5: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 6: A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]

**Rationale:** This eight-pair design tests the advocated theory's condition-specific attenuation of endpoint distrust. In each base pair, the 100%-valid expert endorses A and exactly one opposing expert endorses B; the opponent's validity increases across 50%, 70%, 80%, and 90%. The categorical tally is always tied 1-1. Contextual Anti-Reliability Arbitration predicts a qualitative crossover. Against the 50% cue, the endpoint-versus-neutral exemption preserves the 100% expert's full ordinary and endpoint-specific distrust, producing a strong preference for B. A 70% opponent should still leave the endpoint claim more distrusted, whereas the 80% condition should approach the transition. Against the substantive 90% opponent, attentional focusing attenuates the endpoint cue's total distrust enough that the 90% claim becomes more distrusted than the 100% claim; the model therefore predicts a preference for A. Thus, paradoxically, strengthening the expert opposed to the endpoint cue should move choices from rejecting A to favoring A, crossing chance between the low- and high-validity conditions. Finite-Pool Pragmatic Claim Competition cannot produce this within-person crossover. With one endorser per side, both claims receive the same attention because their pairwise competition loads are identical, so allocated claim contrast is attention times [1-(2v_B-1)], which remains positive whenever v_B<1. A skeptical construal consequently favors B in every condition, merely declining toward chance as the opponent approaches 100%; a weak conventional construal favors A throughout and also cannot switch signs. The critical planned contrast is therefore endpoint-endorsed choice below chance for the 50%/70% opponents but above chance for the 90% opponent, rather than only a monotonic confidence reduction. The last four pairs are exact A/B reversals, balancing response side while preserving this interaction. Eight unique pairs are each repeated 12 times for approximately 96 trials.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This is a qualitative crossover-phenotype indicator. It equals one only when the certainty-endorsed option is rejected below chance against the pooled 50%/70% opponents but selected above chance against the 90% opponent. Choices are recoded relative to the endpoint endorser, so the exact A/B reversals remove response-side effects. Contextual Anti-Reliability Arbitration predicts both parts of this conjunction: intact endpoint distrust in the low-validity contexts and attenuated endpoint distrust against the substantive 90% claim. Finite-Pool Pragmatic Claim Competition predicts increasing endpoint-choice rates as the opponent approaches certainty, but its skeptical claim contrast remains directed against the endpoint option at 90%, so it should fail the above-chance component. The binary sign criterion also deemphasizes heterogeneous effect magnitudes and is bounded in per-subject variance.

Source:
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


### RESULTS
- Predicted under pi_8 (simulated): 1.0000 (var=0.1600)
- Predicted under pi_9 (simulated): 0.0000 (var=0.1204)
- Observed on real data: 0.0000 (var=0.2400)

## EXPERIMENT 2 (proposed by pi_9)

### DESIGN
**Validities (n_features=14):** [0.75, 0.75, 0.75, 0.75, 0.85, 0.85, 0.85, 0.85, 0.8, 0.825, 0.875, 0.9, 0.775, 0.925]

**Trial pairs (n=6):**
  trial 1: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  trial 2: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  trial 3: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1]
  trial 4: A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** This six-pair design holds the categorical tally exactly tied at 4-4 while manipulating only the dispersion of B's claim validities. In every base pair, A is endorsed by four homogeneous 75%-valid experts (assertiveness 0.50 each). B's four endorsements have the same mean assertiveness of 0.70 in all three conditions: four 85% experts; 80%/82.5%/87.5%/90% experts; or 77.5%/82.5%/87.5%/92.5% experts. Finite-Pool Pragmatic Claim Competition predicts increasing A choice as B's coalition becomes more dispersed. Homogeneous B claims strongly compete with one another, whereas dispersion releases especially assertive B claims from similarity-based competition, making the allocated contrast increasingly favor B; the predominant skeptical construal reverses that contrast into stronger evidence for A. Contextual Anti-Reliability Arbitration predicts the opposite gradient. B has greater ordinary claim mass than A throughout, so anti-reliability initially favors A, but increasing within-B dispersion divisively dilutes that signal; the most dispersed condition also includes a smoothly compressed 92.5% claim, further reducing A-favoring anti-reliability evidence. It therefore predicts decreasing A choice toward chance. The critical test is the sign of the within-subject dispersion slope under an invariant 4-4 tally and invariant mean linear validity contrast. The last three pairs are exact option reversals. Six unique pairs are each repeated 16 times for 96 trials.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric measures the absolute choice share of the four-75% coalition in the homogeneous and moderately dispersed conditions, pooling exact option reversals. It is qualitatively different from the failed endpoint slope. Finite-Pool Pragmatic Claim Competition substantially attenuates each claim through eight-way attentional competition, so its bounded skeptical signal should produce only a modest preference for the 75% coalition. Contextual Anti-Reliability Arbitration instead produces a strong anti-reliability contrast when the opposing higher-validity coalition remains coherent, predicting a markedly larger 75%-coalition choice share. The maximally dispersed condition is excluded because heterogeneity and upper-range compression dilute the competing theory's signal there and make its prediction converge toward the advocated theory. Pooling the remaining 64 trials per subject reduces binomial variance while retaining the conditions with the largest expected difference in absolute choice level.

Source:
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


### RESULTS
- Predicted under pi_8 (simulated): 0.7128 (var=0.0076)
- Predicted under pi_9 (simulated): 0.5850 (var=0.0057)
- Observed on real data: 0.8819 (var=0.0059)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
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
**Predicted under pi_8:** 0.3383 (var=0.0039)
**Predicted under pi_9:** 0.3208 (var=0.0107)

### Experiment 4
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
**Predicted under pi_8:** 56.1425 (var=1.2063)
**Predicted under pi_9:** 59.9467 (var=3.7118)

### Experiment 5
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
**Predicted under pi_8:** 0.4106 (var=0.0035)
**Predicted under pi_9:** 0.3440 (var=0.0028)

### Experiment 6
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
**Predicted under pi_8:** 0.8544 (var=0.0058)
**Predicted under pi_9:** 0.9294 (var=0.0023)

### Experiment 7
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
**Predicted under pi_8:** 0.0104 (var=0.0390)
**Predicted under pi_9:** -0.4666 (var=0.0908)

### Experiment 8
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
**Predicted under pi_8:** 0.8592 (var=0.0031)
**Predicted under pi_9:** 0.9035 (var=0.0023)

### Experiment 9
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
**Predicted under pi_8:** 0.2347 (var=0.0051)
**Predicted under pi_9:** 0.3269 (var=0.0321)

### Experiment 10
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
**Predicted under pi_8:** -4.4717 (var=0.0102)
**Predicted under pi_9:** -2.4502 (var=0.0542)

### Experiment 11
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
**Predicted under pi_8:** 0.2814 (var=0.0029)
**Predicted under pi_9:** 0.3376 (var=0.0157)

### Experiment 12
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
**Predicted under pi_8:** 0.8408 (var=0.0059)
**Predicted under pi_9:** 0.7475 (var=0.0177)

### Experiment 13
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
**Predicted under pi_8:** 0.2039 (var=0.0054)
**Predicted under pi_9:** 0.4628 (var=0.0171)

### Experiment 14
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
**Predicted under pi_8:** 0.1583 (var=0.0037)
**Predicted under pi_9:** 0.3469 (var=0.0105)

### Experiment 15
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
**Predicted under pi_8:** 0.0569 (var=0.0085)
**Predicted under pi_9:** -0.0147 (var=0.0077)

### Experiment 16
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
**Predicted under pi_8:** 0.1467 (var=0.0083)
**Predicted under pi_9:** 0.1044 (var=0.0094)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all 16 experiments, pi_8 is the substantially stronger task-invariant account, although neither theory is adequate. By absolute point-estimate error, pi_8 is closer in Experiments 2, 3, 5\u201310, 12\u201314, and 16; pi_9 is closer mainly in Experiments 1, 4, 11, and 15. The most diagnostic broad pattern is that participants often choose against the higher-validity coalition while still allowing sufficiently large raw-vote margins to dominate. Pi_8 captures this combination much better: it is close on the concentrated 100%-versus-50% reversal in Experiment 12, diffuse anti-validity effects in Experiments 13\u201314, coalition-breadth attenuation in Experiment 16, and tally dominance in Experiments 6 and 8. Pi_9's finite-pool allocation attenuates skeptical claim evidence too strongly, producing choices that are systematically too conventional or too tally-dominated in Experiments 5, 7, 9, 12, 13, and 14. Experiment 2 reinforces this problem: the observed 75%-coalition choice rate is 0.8819, above even pi_8's 0.7128 and far above pi_9's 0.5850. However, pi_8's distinctive endpoint-attentuation crossover is directly disconfirmed in Experiment 1: the observed phenotype is 0, exactly matching pi_9 rather than pi_8's prediction of 1. The observed subject variance of 0.2400 also indicates substantial phenotype heterogeneity, but it does not rescue the incorrect pooled crossover. Experiment 15 likewise provides little evidence for pi_8's proposed special 95% curvature: the observed contrast is near zero and closer to pi_9. Thus the supported part of pi_8 is its general separation of skeptical validity evidence from categorical tally evidence, not its endpoint bonus, 95% compression, opponent-specific exemption, or other condition-tailored machinery. Variance predictions tell a similar story: both accounts miss heterogeneity in several experiments; pi_8 often underpredicts it, whereas pi_9 sometimes produces excessive variance while still missing the mean. Since pi_8 wins broadly but its most specialized claim fails, the best next step is not to tune pi_9's current finite-pool model. It should be replaced by a genuinely different, stronger competitor that explains robust anti-validity choice without endpoint notches or symmetric pairwise attentional competition.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace pi_9 with a monotone contextual reactance-and-consensus theory. Communicated validities should generate smoothly increasing skeptical evidence against endorsed options, with no intrinsic discontinuity or notch at 95% or 100% and no predicted endpoint crossover. The strength of this evidence should depend on a coalition-level representation rather than pi_9's claim-by-claim symmetric competition: observers first summarize each side's mean assertiveness, breadth, and coherence, then compare the two coalition summaries. Repetition of similarly valid experts should reinforce a coalition summary sublinearly rather than merely causing mutual attentional suppression; this is needed to produce the very strong preference for the lower-validity coalition in Experiment 2. Coalition breadth and mixed composition should then normalize or compress the skeptical contrast, accounting for attenuation across broad conflicts without condition-specific exemptions. Add a separate raw-tally channel whose influence increases nonlinearly with absolute vote margin, allowing the tally to dominate in Experiments 4, 6, and 8 while leaving tied and one-vote-margin conflicts highly susceptible to anti-validity reversals. Subject-level parameters should continuously vary in reactance strength, tally sensitivity, and decision noise, with correlated heterogeneity permitted so that strong skeptics can also differ in margin protection; avoid categorical skeptic/conventional classes. The theory should predict persistent rejection of the 100% option against 50%, 70%, and 90% opponents, with the effect shrinking toward chance but not crossing it, thereby matching Experiment 1 while retaining the concentrated endpoint reversal in Experiment 12. It should also predict approximately smooth 90%\u201395%\u2013100% behavior, as indicated by Experiment 15. Preserve option-swap symmetry and history independence. This creates a principled competitor to pi_8: both posit tally and pragmatic channels, but the new theory derives context effects from coalition-level sublinear consensus and normalization rather than endpoint distrust bonuses, validity compression notches, sparse exemptions, or pi_9's finite pairwise attentional pool."
}
```

## Usage

```json
{
  "input_tokens": 23936,
  "output_tokens": 1735
}
```
