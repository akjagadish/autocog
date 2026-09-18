# metric_exp00_attempt_00

## System Prompt

You are a psychology researcher proposing a metric in the Decision Making (Binary Features) domain.

Your goal is adversarial: propose a metric that DISCRIMINATES the two theories — i.e., its value, computed on data simulated under your advocated theory, should be as far as possible from its value computed on data simulated under the competing theory. The direction of the gap does not matter; what matters is that the two theories produce visibly different numbers on this metric. The metric is computed on the data collected from the experimental design provided in the prompt. Produce a metric where you're prediction will be much more accurate than the competing theory's prediction on human data.

Your metric is a Python function

    metric(data: pd.DataFrame) -> float

Available imports inside `metric`:
- numpy as np
- pandas as pd

The system evaluates your metric in two ways and reports the pair as `point_estimate (var=between_subject_variance)` everywhere downstream:
- `point_estimate` is `metric(data)` applied to the FULL pooled DataFrame (all subjects together) — the canonical scalar;
- `between_subject_variance` is the population variance (`ddof=0`) of `metric(subj_df)` re-applied per `subject_id`, summarising how stable the metric is across subjects. If your metric only makes sense on multi-subject data this will fall back to `n/a` and the metric is rejected (the acceptance test below cannot run without it). Prefer metrics that work both on the pooled DataFrame and on a single subject's slice.

Acceptance rule: the system simulates each theory and runs Welch's two-sample t-test on `(point_estimate_self, between_subject_variance_self, N)` vs. `(point_estimate_adv, between_subject_variance_adv, N)`, where N is the number of HUMAN subjects the experiment will actually be run with (a fixed small number, currently 25). Your metric is admitted iff the two-sided p-value is below the significance level (currently alpha=0.01). Implication: a large between-theory gap is NOT enough — if either theory's metric is also highly variable across subjects, N humans won't reliably distinguish them and the metric will be rejected. Aim for contrasts that are both large in mean AND tight per subject.

Do NOT propose metrics that are trivially true for your theory.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

Each subject completes ~96 trials in a single block, with order randomized independently per subject. On every trial the subject sees two options A and B, each described by `n_features` binary expert ratings (each 0 or 1). The per-feature validities and n_features are fixed per experiment (design-time choices). Validities are communicated to the subject in the instructions. Both `n_features` and `validities` are exposed to your `predict` via the `parameters` dict. The subject chooses A or B; no correctness feedback is provided after the choice.

## CHOSEN EXPERIMENTAL DESIGN
**Validities (n_features=16):** [0.75, 0.75, 0.75, 0.75, 0.7, 0.7, 0.7, 0.7, 0.6, 0.65, 0.75, 0.813, 0.5, 0.7, 0.8, 0.856]

**Trial pairs (n=6):**
  trial 1: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  trial 3: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  trial 4: A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** This six-pair design holds the categorical tally, coalition breadth, and aggregate anti-reliability contrast approximately constant while manipulating only the internal dispersion of B's coalition. In every base pair, A is endorsed by four homogeneous 75%-valid experts. B is also endorsed by four experts, but its composition is either homogeneous (70%, 70%, 70%, 70%), moderately dispersed (60%, 65%, 75%, 81.3%), or strongly dispersed (50%, 70%, 80%, 85.6%). These compositions were chosen so that B's summed ordinary claim strength under Contextual Anti-Reliability Arbitration is approximately constant: each coalition sums to about four times 0.4^0.82. A's claim sum, the signed claim contrast, the 4-4 tally, and the eight-cue active breadth therefore remain fixed. The advocated theory predicts that increasing B-side claim dispersion enlarges its compositional normalizer and progressively dilutes anti-reliability evidence against the stronger A coalition. It consequently predicts an increasing A-choice profile toward chance from homogeneous to strongly dispersed B. Coalition-Level Skeptical Consensus predicts the opposite slope. Its B coalitions have nearly matched mean linear assertiveness, but their coherence falls sharply with dispersion. B's coalition summary therefore decreases, enlarging A's coalition-level advantage; skeptical reactance against A becomes stronger, so A choice should decrease as B becomes more dispersed. Thus the critical result is an oppositely signed within-subject dispersion gradient under an invariant tied tally and nearly invariant aggregate claim mass. The final three pairs are exact A/B reversals, balancing response side and requiring mirrored gradients. Six unique pairs are each repeated 16 times for 96 trials.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Contextual Anti-Reliability Arbitration theory proposes that communicated validity numbers are interpreted partly as pragmatic claims whose assertiveness can undermine their credibility. Every above-chance claim therefore generates continuous evidence against the option endorsed by that expert, rather than being categorically reversed by a special subgroup. The magnitude of this anti-reliability response varies continuously across people. Near-perfect claims additionally evoke endpoint distrust because unusually strong assurances are perceived as especially suspicious. When an endpoint cue faces a substantive opposing claim, both its ordinary and endpoint-specific distrust contributions are attenuated; against a genuinely neutral cue, both remain intact. Claims around 95% are more strongly compressed in the ordinary channel and do not receive the endpoint bonus, separating near-endpoint skepticism from distrust of literal certainty. Anti-reliability evidence is accumulated separately from the raw vote tally and divisively normalized by coalition breadth, within-side compositional heterogeneity, and sparse across-side claim mismatch. The latter detects heterogeneous singleton conflicts that within-side dispersion cannot detect, while exempting an isolated certainty claim opposed by a neutral cue. A small smooth boost preserves moderate-claim distrust in broad but not maximally dense coalitions. Its influence is strongest when the validity contrast is concentrated and the tally is tied or weak, while it decays smoothly as the absolute tally margin grows. Raw tally evidence additionally receives a continuously heterogeneous, intermediate-margin decision-temperature adjustment. The mechanism remains continuous, option-swap symmetric, and history independent because choices are followed by no accuracy feedback.

**Parameters:**
- anti_reliability_strength: [3.0, 5.6]
- anti_accumulation: [2.0, 3.5]
- endpoint_distrust: [1.10, 2.05]
- coalition_dilution_exponent: [0.46, 0.72]
- composition_dilution: [0.8, 2.4]
- across_composition_dilution: [1.0, 3.5]
- concentration_floor: [0.38, 0.62]
- margin_protection_scale: [1.55, 2.25]
- margin_protection_power: [2.2, 3.4]
- evidence_scale: [2.2, 3.0]
- choice_sensitivity: [1.45, 2.65]
- margin_temperature_gain: [0.0, 0.32]
- lapse_rate: [0.0, 0.10]
- validities: validities

**`predict source code`:**
```python
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
```

**`policy source code`:**
```python
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
```


## COMPETING THEORY
**Description:** Coalition-Level Skeptical Consensus theory proposes that people do not compare communicated validities expert by expert. They first construct a representation of each option's endorsing coalition from three attributes: its mean assertiveness, its breadth, and its internal coherence. Mean assertiveness increases smoothly with communicated validity from 50% through 100%, without endpoint bonuses, notches, or categorical boundaries. Repeated similarly assertive endorsements strengthen the coalition representation according to a sublinear consensus function, whereas heterogeneous coalitions are represented less coherently. The contrast between the two coalition representations is additionally normalized by total conflict breadth and compositional mixture. In narrow conflicts, reactance is further normalized when both coalitions make substantive claims or when neither coalition approaches the maximum of the same smooth assertiveness continuum. This leaves a maximal 100%-versus-50% contrast intact while attenuating moderate and bilateral narrow conflicts without creating an endpoint bonus or threshold. People pragmatically react against the coalition making the stronger aggregate claim, so the coalition contrast enters choice with a skeptical sign. This reactance is continuous across individuals and can be weak rather than defining discrete skeptic classes. A smooth breadth-dependent gain makes isolated conflicts less reactance-provoking than repeated consensus, while preserving compression in very broad conflicts. A separate categorical tally channel ignores validity and increases nonlinearly with the absolute vote margin. It is weak enough at ties and one-vote margins to permit anti-validity reversals, but gradually protects decisive majorities without forcing ceiling-level tally choices. A shared continuous reactance trait modulates margin curvature. This modulation is centered and analytically mean-corrected, so it preserves the population-average tally function while producing increasingly large individual differences at large margins. Independent continuous variation in tally sensitivity and decision noise supplies further subject heterogeneity. Because choices receive no correctness feedback, coalition construction is history independent; because all evidence is expressed as signed option contrasts, the model is exactly invariant to swapping A and B.

**Parameters:**
- consensus_exponent: [0.68, 0.84]
- coherence_scale: [0.025, 0.090]
- coherence_floor: [0.42, 0.68]
- breadth_normalization: [0.05, 0.14]
- composition_normalization: [0.8, 2.2]
- bilateral_narrow_normalization: [2.0, 4.0]
- submaximal_narrow_normalization: [1.0, 2.0]
- breadth_reinforcement_floor: [0.62, 0.76]
- coalition_accumulation: [1.15, 1.75]
- reactance_strength: [2.4, 3.7]
- reactance_trait: [0, 1]
- tally_sensitivity: [0.20, 0.38]
- tally_trait: [0, 1]
- margin_exponent: [1.15, 1.40]
- reactance_margin_coupling: [-0.50, 0.60]
- choice_sensitivity: [0.78, 1.24]
- noise_trait: [0, 1]
- lapse_rate: [0.0, 0.08]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Coalition-Level Skeptical Consensus model. Choice history is deliberately
    # ignored because no outcome feedback is available for learning validity.
    stimulus = np.asarray(state, dtype=np.float64)
    if stimulus.ndim != 2 or stimulus.shape[0] != 2:
        raise ValueError(
            f"Expected state with shape (2, n_features); got {stimulus.shape}."
        )

    validities = np.asarray(parameters["validities"], dtype=np.float64)
    n_features = stimulus.shape[1]
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} does not match "
            f"n_features {n_features}."
        )

    # +1 denotes a discriminating endorsement of A and -1 an endorsement of B.
    signed_votes = np.sign(stimulus[0] - stimulus[1])
    active = signed_votes != 0.0
    tally_margin = float(np.sum(signed_votes))

    # Assertiveness is completely smooth and monotone on the communicated
    # validity scale. In particular, 90%, 95%, and 100% receive neighboring
    # values on the same linear continuum.
    validity = np.clip(validities, 0.5, 1.0)
    assertiveness = 2.0 * (validity - 0.5)

    def coalition_summary(side):
        members = assertiveness[active & (signed_votes == side)]
        breadth = int(members.size)
        if breadth == 0:
            return 0.0, 0.0, 0

        mean_assertiveness = float(np.mean(members))
        dispersion = float(np.mean((members - mean_assertiveness) ** 2))

        # Similar members form a coherent aggregate claim. Mixed validity
        # compositions reduce coherence continuously rather than triggering an
        # exception for any particular validity or coalition size.
        coherence_scale = float(parameters["coherence_scale"])
        coherence_floor = float(parameters["coherence_floor"])
        coherence = coherence_floor + (1.0 - coherence_floor) * np.exp(
            -dispersion / coherence_scale
        )

        # Repetition reinforces consensus, but with diminishing returns. This
        # differs from allocating a fixed attentional pool among individual
        # claims: adding a similar endorser always increases the summary.
        consensus_exponent = float(parameters["consensus_exponent"])
        consensus = float(breadth) ** consensus_exponent
        summary = mean_assertiveness * consensus * coherence
        return float(summary), dispersion, breadth

    summary_a, dispersion_a, breadth_a = coalition_summary(1.0)
    summary_b, dispersion_b, breadth_b = coalition_summary(-1.0)
    raw_coalition_contrast = summary_a - summary_b
    total_breadth = breadth_a + breadth_b

    if total_breadth == 0:
        contextual_contrast = 0.0
    else:
        # Broad conflicts compress pragmatic contrast. Internal mixture adds a
        # second coalition-level normalizer, weighted by coalition breadth.
        excess_breadth = max(total_breadth - 2, 0)
        breadth_normalizer = 1.0 + float(
            parameters["breadth_normalization"]
        ) * np.log1p(float(excess_breadth))

        pooled_dispersion = (
            breadth_a * dispersion_a + breadth_b * dispersion_b
        ) / float(total_breadth)
        composition_normalizer = 1.0 + float(
            parameters["composition_normalization"]
        ) * float(total_breadth) * pooled_dispersion

        contextual_contrast = raw_coalition_contrast / (
            breadth_normalizer * composition_normalizer + 1e-12
        )

        # Narrow conflicts are additionally normalized when both coalition
        # summaries are substantive or when the strongest summary remains
        # below the top of the smooth assertiveness scale. Both effects decay
        # continuously with conflict breadth. A maximal singleton contrast of
        # 100% against 50% has zero added normalization.
        narrow_weight = np.exp(-float(excess_breadth) / 2.0)
        bilateral_mass = min(abs(summary_a), abs(summary_b))
        strongest_summary = min(max(abs(summary_a), abs(summary_b)), 1.0)
        submaximal_gap = 1.0 - strongest_summary
        narrow_normalizer = 1.0 + narrow_weight * (
            float(parameters["bilateral_narrow_normalization"])
            * bilateral_mass
            + float(parameters["submaximal_narrow_normalization"])
            * submaximal_gap ** 2
        )
        contextual_contrast /= float(narrow_normalizer)

        # A singleton conflict evokes less coalition-level reactance than a
        # repeated consensus. The gain rises smoothly with breadth and remains
        # distinct from the broad-conflict normalization above.
        reinforcement_floor = float(parameters["breadth_reinforcement_floor"])
        reinforcement_gain = reinforcement_floor + (
            1.0 - reinforcement_floor
        ) * (1.0 - np.exp(-float(excess_breadth) / 2.0))
        contextual_contrast *= float(reinforcement_gain)

    # A continuous trait controls skeptical construal. It also contributes to
    # margin curvature below, permitting correlated heterogeneity without a
    # categorical skeptic/conventional mixture.
    reactance_trait = float(parameters["reactance_trait"])
    reactance_strength = float(parameters["reactance_strength"]) * (
        0.65 + 0.70 * reactance_trait
    )
    accumulated_contrast = np.tanh(
        float(parameters["coalition_accumulation"]) * contextual_contrast
    )
    skeptical_evidence = -reactance_strength * float(accumulated_contrast)

    # The raw tally is represented independently of communicated validity. Its
    # mildly superlinear margin function leaves a one-vote lead vulnerable but
    # lets decisive margins gradually dominate bounded reactance without
    # producing near-deterministic tally choices.
    tally_trait = float(parameters["tally_trait"])
    tally_sensitivity = float(parameters["tally_sensitivity"]) * (
        0.70 + 0.60 * tally_trait
    )
    base_margin_exponent = float(parameters["margin_exponent"])
    margin_coupling = float(parameters["reactance_margin_coupling"])
    exponent_deviation = margin_coupling * (reactance_trait - 0.5)
    margin_exponent = base_margin_exponent + exponent_deviation

    if abs(tally_margin) < 1e-12:
        tally_evidence = 0.0
    else:
        margin_magnitude = abs(tally_margin)

        # Centering the exponent deviation preserves one-vote evidence exactly.
        # For larger margins, divide by its analytic population mean under the
        # continuous uniform reactance trait. Thus widening the coupling adds
        # heterogeneity that grows with margin without changing the accepted
        # population-average tally curve.
        mean_span = 0.5 * margin_coupling * np.log(margin_magnitude)
        if abs(mean_span) < 1e-8:
            heterogeneity_correction = 1.0
        else:
            heterogeneity_correction = float(np.sinh(mean_span) / mean_span)

        tally_evidence = (
            tally_sensitivity
            * np.sign(tally_margin)
            * margin_magnitude ** margin_exponent
            / heterogeneity_correction
        )

    total_evidence = tally_evidence + skeptical_evidence

    # Decision noise varies continuously and independently across subjects.
    noise_trait = float(parameters["noise_trait"])
    choice_sensitivity = float(parameters["choice_sensitivity"]) * (
        0.65 + 0.70 * noise_trait
    )
    decision_variable = choice_sensitivity * total_evidence

    # Positive evidence favors A. Symmetric logits guarantee exact option-swap
    # symmetry, including on tied and mirrored trials.
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
```

**`policy source code`:**
```python
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
```


## DATA SCHEMA
Your metric receives a tidy per-trial pandas DataFrame stacking all subjects (rows grouped by `subject_id`, in trial order). Columns:
- subject_id: Subject identifier (one row per trial per subject).
- option_a_ratings: List of n_features binary expert ratings (each 0 or 1) for option A on this trial.
- option_b_ratings: List of n_features binary expert ratings (each 0 or 1) for option B on this trial.
- response: 0 if subject chose A, 1 if subject chose B.

## IMPLEMENTATION GUARDRAILS
Any column in the schema above whose description names a list / tuple / np.ndarray (i.e. a per-trial sequence of values) holds non-scalar cells. Those cells are NOT hashable, so operations that hash row values fail with `TypeError: unhashable type: 'list'`. Treating `<seq_col>` as a placeholder for any such sequence-valued column:
- Avoid: `data.groupby('<seq_col>')`, `data['<seq_col>'].value_counts()`,     `data['<seq_col>'].nunique()`, `data['<seq_col>'].unique()` (returns     an object array but downstream `set()` / `in dict` will crash),     `set(data['<seq_col>'])`, `data['<seq_col>'].isin([...])` against list     values, or using a list cell as a dict key.
- If you need a hashable surrogate, project to one first, e.g.:
    - `data['<seq_col>_key'] = data['<seq_col>'].apply(tuple)` then group by `<seq_col>_key`
    - `data['<seq_col>_str'] = data['<seq_col>'].apply(lambda x: ''.join(map(str, x)))`
    Scalar columns (ints, floats, strings like `subject_id`, integer     responses, etc.) hash fine and can be used directly.
- Generator expressions inside function calls like `map()` or `join()` MUST be     parenthesized. For example:
    - WRONG: `map(str, int(v) for v in x)` → SyntaxError
    - RIGHT: `map(str, (int(v) for v in x))` or use a list comp: `[str(int(v)) for v in x]`
- Always verify your code is syntactically valid Python before returning it.

## METRICS YOU ALREADY TRIED AND FAILED ON
Each entry below is a metric you previously proposed in this round that did NOT discriminate the two theories at the human sample size — either it errored, its between-subject variance was unavailable, or Welch's t-test on `(self mean, self var, N)` vs. `(adv mean, adv var, N)` returned p ≥ alpha. The `outcome` line is the simulation result (means, between-subject variances, t-statistic and p-value at the human N) on the same `data_self` / `data_adv` your next metric will be evaluated on. Use the numbers to see where your hypothesised contrast collapsed — small mean gap, large per-subject variance, or both — and propose something qualitatively different. Don't repeat the same idea with cosmetic tweaks.
(none yet)

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Estimate the dispersion gradient after orienting every trial toward the\n    # homogeneous coalition of four 75%-valid experts (features 0--3).\n    cells = {(\"low\", \"A\"): [], (\"low\", \"B\"): [],\n             (\"high\", \"A\"): [], (\"high\", \"B\"): []}\n\n    for _, row in data.iterrows():\n        a = np.asarray(row[\"option_a_ratings\"], dtype=np.int64)\n        b = np.asarray(row[\"option_b_ratings\"], dtype=np.int64)\n        if a.size < 16 or b.size < 16:\n            continue\n\n        a_is_reference = bool(np.all(a[:4] == 1) and np.all(a[4:] == 0))\n        b_is_reference = bool(np.all(b[:4] == 1) and np.all(b[4:] == 0))\n\n        if a_is_reference == b_is_reference:\n            continue\n\n        if a_is_reference:\n            reference_side = \"A\"\n            competitor = b\n            chose_reference = float(int(row[\"response\"]) == 0)\n        else:\n            reference_side = \"B\"\n            competitor = a\n            chose_reference = float(int(row[\"response\"]) == 1)\n\n        if np.all(competitor[4:8] == 1) and int(np.sum(competitor)) == 4:\n            level = \"low\"\n        elif np.all(competitor[12:16] == 1) and int(np.sum(competitor)) == 4:\n            level = \"high\"\n        else:\n            # The moderately dispersed coalition is not needed for the\n            # endpoint contrast.\n            continue\n\n        cells[(level, reference_side)].append(chose_reference)\n\n    means = {}\n    for key, values in cells.items():\n        if len(values) == 0:\n            return float(\"nan\")\n        means[key] = float(np.mean(np.asarray(values, dtype=np.float64)))\n\n    # Average mirrored, side-specific contrasts so fixed A/B response bias\n    # cannot masquerade as a dispersion effect.\n    gradient_a = means[(\"high\", \"A\")] - means[(\"low\", \"A\")]\n    gradient_b = means[(\"high\", \"B\")] - means[(\"low\", \"B\")]\n    return float(0.5 * (gradient_a + gradient_b))",
  "rationale": "This is the within-subject endpoint dispersion gradient in choosing the homogeneous four-expert 75% coalition. Trials are first reoriented so that choosing that coalition has the same meaning in the original and option-swapped displays. The high-dispersion minus low-dispersion contrast is then computed separately for reference-on-A and reference-on-B trials and averaged, cancelling stable response-side bias. Contextual Anti-Reliability Arbitration predicts a positive gradient because dispersion dilutes the opposing coalition's anti-reliability contribution, whereas Coalition-Level Skeptical Consensus predicts a negative gradient because dispersion reduces the opposing coalition's coherence and thereby strengthens skeptical reaction against the homogeneous reference coalition. Using the extreme dispersion conditions maximizes the predicted sign reversal, while the repeated mirrored trials provide a relatively stable subject-level estimate."
}
```

## Usage

```json
{
  "input_tokens": 8261,
  "output_tokens": 2805
}
```
