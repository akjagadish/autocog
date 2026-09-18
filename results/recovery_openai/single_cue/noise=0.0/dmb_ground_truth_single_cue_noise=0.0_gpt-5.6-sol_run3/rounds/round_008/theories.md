# Round 8 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_8` — KILLED ✗

**Description:** Contextual Anti-Reliability Arbitration theory proposes that communicated validity numbers are interpreted partly as pragmatic claims whose assertiveness can undermine their credibility. Every above-chance claim therefore generates continuous evidence against the option endorsed by that expert, rather than being categorically reversed by a special subgroup. The magnitude of this anti-reliability response varies continuously across people. Near-perfect claims additionally evoke endpoint distrust because unusually strong assurances are perceived as especially suspicious. When an endpoint cue faces a substantive opposing claim, both its ordinary and endpoint-specific distrust contributions are attenuated; against a genuinely neutral cue, both remain intact. Claims around 95% are more strongly compressed in the ordinary channel and do not receive the endpoint bonus, separating near-endpoint skepticism from distrust of literal certainty. Anti-reliability evidence is accumulated separately from the raw vote tally and divisively normalized by coalition breadth, within-side compositional heterogeneity, and sparse across-side claim mismatch. The latter detects heterogeneous singleton conflicts that within-side dispersion cannot detect, while exempting an isolated certainty claim opposed by a neutral cue. A small smooth boost preserves moderate-claim distrust in broad but not maximally dense coalitions. Its influence is strongest when the validity contrast is concentrated and the tally is tied or weak, while it decays smoothly as the absolute tally margin grows. Raw tally evidence additionally receives a continuously heterogeneous, intermediate-margin decision-temperature adjustment. The mechanism remains continuous, option-swap symmetric, and history independent because choices are followed by no accuracy feedback.

**Rationale:** This is a minimal edit to the accepted iteration-7 candidate. The ordinary validity mapping, endpoint range, intermediate-margin temperature, tally mapping, and response transformation are all retained. The main addition is an across-active compositional normalizer. Unlike within-side variance, it remains informative when each side has only one endorser, so it can dilute sparse mismatched claim contrasts in Experiment 8 and related heterogeneous profiles. Its effect decreases quadratically with coalition size, and a continuous endpoint-neutral exemption preserves the successful isolated 100%-versus-50% reversal in Experiment 10. The concentration-floor range is shifted modestly upward to strengthen anti-reliability in diffuse but coherent moderate coalitions such as Experiments 7 and 11. The within-side composition-dilution range is widened upward to offset that increase in heterogeneous broad profiles such as Experiments 1 and 9. These changes express heterogeneity through contextual composition parameters rather than global choice sensitivity, while preserving option-swap symmetry, smooth arbitration, and history independence.

**Parameters:**
  - `anti_reliability_strength`: `[3.0, 5.6]`
  - `anti_accumulation`: `[2.0, 3.5]`
  - `endpoint_distrust`: `[1.10, 2.05]`
  - `coalition_dilution_exponent`: `[0.46, 0.72]`
  - `composition_dilution`: `[0.8, 2.4]`
  - `across_composition_dilution`: `[1.0, 3.5]`
  - `concentration_floor`: `[0.38, 0.62]`
  - `margin_protection_scale`: `[1.55, 2.25]`
  - `margin_protection_power`: `[2.2, 3.4]`
  - `evidence_scale`: `[2.2, 3.0]`
  - `choice_sensitivity`: `[1.45, 2.65]`
  - `margin_temperature_gain`: `[0.0, 0.32]`
  - `lapse_rate`: `[0.0, 0.10]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
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

**`policy(probs)`:**
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


### slot 2 — `pi_10` — SURVIVED ✓

**Description:** Coalition-Level Skeptical Consensus theory proposes that people do not compare communicated validities expert by expert. They first construct a representation of each option's endorsing coalition from three attributes: its mean assertiveness, its breadth, and its internal coherence. Mean assertiveness increases smoothly with communicated validity from 50% through 100%, without endpoint bonuses, notches, or categorical boundaries. Repeated similarly assertive endorsements strengthen the coalition representation according to a sublinear consensus function, whereas heterogeneous coalitions are represented less coherently. The contrast between the two coalition representations is additionally normalized by total conflict breadth and compositional mixture. In narrow conflicts, reactance is further normalized when both coalitions make substantive claims or when neither coalition approaches the maximum of the same smooth assertiveness continuum. This leaves a maximal 100%-versus-50% contrast intact while attenuating moderate and bilateral narrow conflicts without creating an endpoint bonus or threshold. People pragmatically react against the coalition making the stronger aggregate claim, so the coalition contrast enters choice with a skeptical sign. This reactance is continuous across individuals and can be weak rather than defining discrete skeptic classes. A smooth breadth-dependent gain makes isolated conflicts less reactance-provoking than repeated consensus, while preserving compression in very broad conflicts. A separate categorical tally channel ignores validity and increases nonlinearly with the absolute vote margin. It is weak enough at ties and one-vote margins to permit anti-validity reversals, but gradually protects decisive majorities without forcing ceiling-level tally choices. A shared continuous reactance trait modulates margin curvature. This modulation is centered and analytically mean-corrected, so it preserves the population-average tally function while producing increasingly large individual differences at large margins. Independent continuous variation in tally sensitivity and decision noise supplies further subject heterogeneity. Because choices receive no correctness feedback, coalition construction is history independent; because all evidence is expressed as signed option contrasts, the model is exactly invariant to swapping A and B.

**Rationale:** This is an isolated edit to the accepted iteration-4 tally heterogeneity mechanism. All coalition summaries, contextual normalizers, skeptical accumulation, tally sensitivity, and decision equations are retained. The reactance–margin coupling range is widened, and the resulting exponent variation is divided by its analytic mean under the continuous reactance trait. Consequently, the average raw-tally evidence remains approximately equal to the accepted model at each margin, protecting Experiment 2's successful mean margin profile, while individual differences grow with log margin. One-vote evidence is unchanged exactly. At margins five and above, the added spread should reduce pooled ceiling probabilities through logistic curvature and increase between-subject variance in Experiments 4 and 6. The edit avoids the rejected explicit saturation, common-mode cancellation, concave coalition readout, and additional bilateral normalizers, thereby minimizing risk to Experiments 7, 10, 12, 14, and 16.

**Parameters:**
  - `consensus_exponent`: `[0.68, 0.84]`
  - `coherence_scale`: `[0.025, 0.090]`
  - `coherence_floor`: `[0.42, 0.68]`
  - `breadth_normalization`: `[0.05, 0.14]`
  - `composition_normalization`: `[0.8, 2.2]`
  - `bilateral_narrow_normalization`: `[2.0, 4.0]`
  - `submaximal_narrow_normalization`: `[1.0, 2.0]`
  - `breadth_reinforcement_floor`: `[0.62, 0.76]`
  - `coalition_accumulation`: `[1.15, 1.75]`
  - `reactance_strength`: `[2.4, 3.7]`
  - `reactance_trait`: `[0, 1]`
  - `tally_sensitivity`: `[0.20, 0.38]`
  - `tally_trait`: `[0, 1]`
  - `margin_exponent`: `[1.15, 1.40]`
  - `reactance_margin_coupling`: `[-0.50, 0.60]`
  - `choice_sensitivity`: `[0.78, 1.24]`
  - `noise_trait`: `[0, 1]`
  - `lapse_rate`: `[0.0, 0.08]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
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

**`policy(probs)`:**
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


## Replacement

### `pi_11` → slot 1 (via `new_theory`)

**Description:** Coherence-Gated Relative Coalition Comparison theory proposes that people compress endorsements into option-level coalition representations before making a choice. Each representation combines smoothly encoded mean assertiveness, sublinear consensus, and a coherence gate. The coherence gate is threshold-shaped: ordinary variation among coalition members has little effect, whereas sufficiently extreme dispersion sharply weakens the coalition. Consequently, a maximally dispersed opposing coalition makes a homogeneous coalition relatively prominent and thus a stronger target of skeptical interpretation, without making routine coalition expansion intrinsically reactance-producing. Sublinear consensus is offset by breadth and active-evidence normalization, yielding approximately breadth-invariant comparison of equal-sized homogeneous coalitions. A narrow-conflict focus rule further distinguishes maximal relative contrasts from ordinary singleton disagreements: a smooth 100%-versus-50% contrast remains strongly skepticism-provoking, while less extreme singleton contrasts are attenuated without an endpoint-specific bonus. A separate validity-blind tally channel grows concavely with vote margin. Its gain is reduced while its curvature is increased, pivoting tally evidence away from one-vote margins and toward moderate and large margins. Continuous correlation between tally orientation and response noise prevents high-tally subjects from becoming deterministically majority-following. Skeptical orientation, tally sensitivity, and decision noise vary continuously across people, with composition-dependent heterogeneity concentrated in genuinely dispersed conflicts. With no correctness feedback, processing remains history independent and exactly option-swap symmetric.

**Rationale:** This is an isolated calibration of the accepted iteration-2 tally channel; the accepted validity code, coalition representations, coherence gate, breadth normalization, skeptical focus, and conflict-noise equations are unchanged. The tally exponent is raised from [0.58, 0.80] to [0.78, 0.96], while tally sensitivity is lowered from [0.32, 0.58] to [0.25, 0.43]. These jointly pivot the existing power curve: one-vote evidence changes little or decreases slightly, while evidence becomes more differentiated across moderate and large margins. This should raise minority choices in Experiment 3 and strengthen the margin-dependent tally signature in Experiment 2 without simply restoring the excessive global tally confidence of iteration 1. The only functional addition is a modest continuous correlation between tally orientation and response noise. Subjects with larger tally gain receive lower decision precision, retaining substantial majority reliance while limiting ceiling behavior in Experiments 4 and 6. This correlation is subject-level and monotonic rather than a rejected margin-specific temperature profile. No roster-dependent dispersion baseline, activation gate, Hill tally replacement, or per-coalition consensus division is introduced.

**Parameters:**
  - `validity_curvature`: `[1.6, 3.0]`
  - `consensus_exponent`: `[0.58, 0.76]`
  - `coherence_scale`: `[0.035, 0.075]`
  - `coherence_floor`: `[0.05, 0.25]`
  - `coherence_power`: `[1.8, 3.0]`
  - `coherence_trait_coupling`: `[-0.70, 0.70]`
  - `attention_floor`: `[0.28, 0.46]`
  - `evidence_normalization`: `[0.48, 0.76]`
  - `contrast_accumulation`: `[1.15, 1.75]`
  - `singleton_focus_floor`: `[0.45, 0.70]`
  - `skeptical_strength`: `[1.85, 2.85]`
  - `skeptical_orientation`: `[0, 1]`
  - `composition_trait_gain`: `[0.30, 0.90]`
  - `skeptic_margin_coupling`: `[-0.30, 0.45]`
  - `tally_sensitivity`: `[0.25, 0.43]`
  - `tally_trait`: `[0, 1]`
  - `tally_exponent`: `[0.78, 0.96]`
  - `choice_sensitivity`: `[0.82, 1.18]`
  - `noise_trait`: `[0, 1]`
  - `tally_noise_correlation`: `[0.15, 0.35]`
  - `conflict_noise`: `[0.10, 0.55]`
  - `lapse_rate`: `[0.0, 0.08]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # Coherence-Gated Relative Coalition Comparison. Choice history is ignored
    # because previous choices are not accompanied by correctness feedback.
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

    # Positive signs endorse A and negative signs endorse B. Features on which
    # the options agree carry neither a categorical vote nor a directional claim.
    signed_votes = np.sign(stimulus[0] - stimulus[1])
    active = signed_votes != 0.0
    tally_margin = float(np.sum(signed_votes))
    total_breadth = int(np.count_nonzero(active))

    # A single smooth, monotone, saturating validity code. The normalization
    # maps 50% to zero and 100% to one without assigning either 95% or 100% a
    # qualitatively distinct representation.
    validity = np.clip(validities, 0.5, 1.0)
    curvature = float(parameters["validity_curvature"])
    endpoint = np.tanh(0.5 * curvature)
    assertiveness = np.tanh(curvature * (validity - 0.5)) / max(endpoint, 1e-12)

    skeptic_trait = float(parameters["skeptical_orientation"])

    def summarize(side):
        members = assertiveness[active & (signed_votes == side)]
        breadth = int(members.size)
        if breadth == 0:
            return 0.0, 0.0, 0

        mean_strength = float(np.mean(members))
        dispersion = float(np.mean((members - mean_strength) ** 2))

        # The same continuous skeptical trait changes sensitivity to internal
        # inconsistency. The superlinear dispersion ratio leaves ordinary
        # coalition composition largely intact but sharply gates sufficiently
        # dispersed coalitions.
        scale = float(parameters["coherence_scale"]) * np.exp(
            float(parameters["coherence_trait_coupling"])
            * (skeptic_trait - 0.5)
        )
        floor = float(parameters["coherence_floor"])
        dispersion_ratio = dispersion / max(scale, 1e-12)
        coherence = floor + (1.0 - floor) * np.exp(
            -(dispersion_ratio ** float(parameters["coherence_power"]))
        )

        consensus = float(breadth) ** float(parameters["consensus_exponent"])
        representation = mean_strength * consensus * coherence
        return float(representation), dispersion, breadth

    rep_a, dispersion_a, breadth_a = summarize(1.0)
    rep_b, dispersion_b, breadth_b = summarize(-1.0)

    if total_breadth == 0:
        contextual_contrast = 0.0
        pooled_dispersion = 0.0
    else:
        pooled_dispersion = (
            breadth_a * dispersion_a + breadth_b * dispersion_b
        ) / float(total_breadth)

        # For equal breadth n on both sides, homogeneous representations grow
        # as n**alpha while this breadth unit grows by the same factor. Thus
        # repetition alone produces almost no change in validity contrast.
        alpha = float(parameters["consensus_exponent"])
        breadth_unit = (float(total_breadth) / 2.0) ** alpha

        # Total active assertiveness consumes limited comparison capacity. This
        # is a relative, composition-sensitive normalization and therefore
        # preserves a maximal 100%-versus-50% singleton contrast.
        active_mean = float(np.mean(assertiveness[active]))
        denominator = breadth_unit * (
            float(parameters["attention_floor"])
            + float(parameters["evidence_normalization"]) * active_mean
        )
        raw_relative_contrast = (rep_a - rep_b) / max(denominator, 1e-12)
        contextual_contrast = float(np.tanh(
            float(parameters["contrast_accumulation"])
            * raw_relative_contrast
        ))

    # Stronger relative coalition claims provoke skepticism. Trait effects are
    # selectively coupled to composition and margin, so individual differences
    # grow where construal is genuinely ambiguous rather than on every trial.
    dispersion_index = pooled_dispersion / (pooled_dispersion + 0.04)
    margin_index = abs(tally_margin) / (abs(tally_margin) + 2.0)
    skeptical_multiplier = 0.70 + 0.60 * skeptic_trait
    skeptical_multiplier *= (
        1.0
        + float(parameters["composition_trait_gain"])
        * (skeptic_trait - 0.5)
        * dispersion_index
    )
    skeptical_multiplier *= (
        1.0
        + float(parameters["skeptic_margin_coupling"])
        * (skeptic_trait - 0.5)
        * margin_index
    )
    skeptical_multiplier = max(float(skeptical_multiplier), 0.05)

    # In a narrow bilateral conflict, weak relative contrasts receive less
    # pragmatic focus than maximal contrasts. The adjustment vanishes smoothly
    # with breadth and therefore does not create a general repetition benefit.
    narrow_weight = np.exp(-max(float(total_breadth) - 2.0, 0.0) / 2.0)
    contrast_focus = 1.0 - narrow_weight * (
        1.0 - float(parameters["singleton_focus_floor"])
    ) * (1.0 - abs(contextual_contrast))
    skeptical_evidence = -float(parameters["skeptical_strength"]) * (
        skeptical_multiplier * contrast_focus * contextual_contrast
    )

    # The independent tally channel is validity blind. Its concave nonlinear
    # growth differentiates margins and protects large majorities, but is less
    # steep and less confident than a superlinear margin rule.
    tally_trait = float(parameters["tally_trait"])
    tally_gain = float(parameters["tally_sensitivity"]) * (
        0.20 + 1.60 * tally_trait
    )
    if abs(tally_margin) < 1e-12:
        tally_evidence = 0.0
    else:
        magnitude = abs(tally_margin)
        tally_curve = (1.0 + magnitude) ** float(parameters["tally_exponent"]) - 1.0
        tally_evidence = tally_gain * np.sign(tally_margin) * tally_curve

    total_evidence = tally_evidence + skeptical_evidence

    # Noise varies continuously, but conflict-dependent noise is concentrated
    # in dispersed, low-margin comparisons instead of being a blanket variance
    # adjustment. High tally orientation is also coupled to lower response
    # precision, preventing high-gain subjects from becoming deterministic.
    noise_trait = float(parameters["noise_trait"])
    base_gain = float(parameters["choice_sensitivity"]) * (
        0.72 + 0.56 * noise_trait
    )
    base_gain /= (
        1.0
        + float(parameters["tally_noise_correlation"])
        * tally_trait
    )
    low_margin_weight = 1.0 / (1.0 + abs(tally_margin) / 2.0)
    conflict_load = dispersion_index * low_margin_weight
    effective_gain = base_gain / (
        1.0
        + float(parameters["conflict_noise"])
        * (1.20 - 0.40 * noise_trait)
        * conflict_load
    )
    decision_variable = float(effective_gain * total_evidence)

    # Positive evidence favors A. Symmetric logits guarantee exact invariance
    # under swapping the two displayed options.
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

**`policy(probs)`:**
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
