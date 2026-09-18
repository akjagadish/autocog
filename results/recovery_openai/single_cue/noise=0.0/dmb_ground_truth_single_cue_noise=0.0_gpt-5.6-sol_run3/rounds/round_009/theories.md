# Round 9 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_11` — SURVIVED ✓

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


### slot 2 — `pi_10` — KILLED ✗

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

### `pi_12` → slot 2 (via `new_theory`)

**Description:** Uncertainty-Arbitrated Overclaim Construal theory proposes that people form two competing, stimulus-dependent interpretations of binary expert advice. The reliability construal compresses each option's endorsers into a coalition claim defined by mean communicated reliability and an inferred trustworthiness estimate. Coalition breadth does not directly amplify this claim, so repeating an otherwise homogeneous coalition is approximately representationally redundant. Trustworthiness remains high under ordinary heterogeneity but collapses after dispersion crosses a subject-specific uncertainty threshold. This makes an extremely split coalition a poor bearer of a unified claim and can render the opposing homogeneous coalition the uniquely salient overclaim. A skeptical pragmatic response then favors the option opposite that salient claim. In parallel, a social-count construal ignores validity and represents only the raw endorsement margin. Its evidence follows a sigmoid power curve with convex initial growth and a confidence-limited asymptote. A localized confidence-control window attenuates moderate tally margins while leaving decisive margins nearly unchanged. A metacognitive gate arbitrates between the two construals according to claim certainty, coalition dispersion, tally decisiveness, and representational disagreement. Moderate tied claims receive additional scrutiny only inside a narrow reliability-contrast band, preventing the intervention from spreading to endpoint-like equal-coalition conflicts. Precision is selectively reduced for one-vote conflicts and for endpoint-sized bilateral claims supported by two trustworthy coalitions. Consequently, ambiguous conflicts move toward chance, moderate overclaims can evoke skepticism, and extremely dispersed claims remain discounted. With no correctness feedback, the process is history independent. Every computation uses signed option contrasts, guaranteeing exact option-swap symmetry.

**Rationale:** This is a minimal extension of the accepted candidate. The extreme-dispersion trust transition, breadth-invariant claim code, arbitration equation, and large-margin tally asymptote are unchanged. Four localized calibrations address the latest critique. First, the existing one-vote precision limiter is modestly strengthened, moving Experiments 1 and 3 toward chance without globally flattening evidence. Second, a Gaussian tally-confidence window now attenuates margins around 3–5 but vanishes at decisive margins, targeting Experiment 2's excessive margin differentiation and the slight overprediction in Experiments 4 and 6 while protecting margins 7 and above in Experiment 20. Third, the moderate tied-claim boost is made substantially taller but much narrower, concentrating scrutiny near the Experiment-16 reliability contrast while excluding more endpoint-like equal-coalition conflicts implicated in Experiment 5. Fourth, the trusted-extreme limiter is strengthened and sharpened to moderate excessive anti-reliability responding in Experiments 8 and 10 only when both endpoint-sized claims remain trustworthy. It remains inactive for Experiment 19's sharply discounted split coalition, protecting the accepted dispersion crossover.

**Parameters:**
  - `reliability_curvature`: `[2.0, 3.2]`
  - `dispersion_threshold`: `[0.050, 0.082]`
  - `dispersion_steepness`: `[32.0, 52.0]`
  - `trust_tail`: `[0.0, 0.07]`
  - `sampling_scale`: `[0.70, 1.30]`
  - `sampling_floor`: `[0.72, 0.90]`
  - `claim_floor`: `[0.16, 0.30]`
  - `skeptical_orientation`: `[0, 1]`
  - `skeptic_trait_spread`: `[0.50, 0.90]`
  - `skeptical_gain`: `[2.6, 3.6]`
  - `claim_accumulation`: `[1.25, 1.75]`
  - `ambiguity_margin_scale`: `[2.2, 3.8]`
  - `tally_power`: `[1.75, 2.35]`
  - `tally_half_margin`: `[2.5, 3.4]`
  - `tally_gain_trait`: `[0, 1]`
  - `tally_trait_spread`: `[0.22, 0.48]`
  - `tally_gain`: `[1.85, 2.35]`
  - `mid_margin_center`: `[3.6, 4.4]`
  - `mid_margin_width`: `[0.9, 1.3]`
  - `mid_margin_tally_damping`: `[0.14, 0.24]`
  - `arbitration_threshold`: `[0.28, 0.62]`
  - `certainty_weight`: `[1.45, 2.10]`
  - `disagreement_weight`: `[0.55, 1.05]`
  - `dispersion_uncertainty_weight`: `[0.12, 0.38]`
  - `margin_resolution_weight`: `[1.10, 1.65]`
  - `dispersion_reference`: `[0.035, 0.065]`
  - `gate_temperature`: `[0.16, 0.28]`
  - `moderate_tie_gate_boost`: `[1.8, 2.8]`
  - `moderate_contrast_center`: `[0.20, 0.27]`
  - `moderate_contrast_width`: `[0.035, 0.070]`
  - `precision_trait`: `[0, 1]`
  - `precision_trait_spread`: `[0.28, 0.55]`
  - `base_precision`: `[1.05, 1.35]`
  - `small_margin_precision_damping`: `[0.50, 0.66]`
  - `small_margin_decay`: `[0.75, 1.15]`
  - `trusted_extreme_precision_damping`: `[0.35, 0.52]`
  - `extreme_contrast_center`: `[0.62, 0.72]`
  - `extreme_contrast_width`: `[0.045, 0.080]`
  - `lapse_rate`: `[0.0, 0.055]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # Uncertainty-Arbitrated Overclaim Construal model. History is intentionally
    # ignored because choices are never followed by correctness feedback.
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

    # Positive signs endorse A and negative signs endorse B. Agreements are
    # non-discriminating and enter neither representation.
    signed_votes = np.sign(stimulus[0] - stimulus[1])
    active = signed_votes != 0.0
    tally_margin = float(np.sum(signed_votes))
    margin_magnitude = abs(tally_margin)
    active_count = int(np.count_nonzero(active))

    # A smooth bounded reliability code avoids endpoint categories while
    # retaining sensitivity throughout the communicated validity scale.
    validity = np.clip(validities, 0.5, 1.0)
    reliability_curvature = float(parameters["reliability_curvature"])
    endpoint = np.tanh(0.5 * reliability_curvature)
    reliability = np.tanh(
        reliability_curvature * (validity - 0.5)
    ) / max(float(endpoint), 1e-12)

    # The arbitration threshold is a stable continuous individual difference.
    arbitration_threshold = float(parameters["arbitration_threshold"])
    dispersion_threshold = float(parameters["dispersion_threshold"]) * (
        0.82 + 0.36 * arbitration_threshold
    )

    def coalition(side):
        members = reliability[active & (signed_votes == side)]
        breadth = int(members.size)
        if breadth == 0:
            return 0.0, 0.0, 0.0, 0

        # Mean reliability is the coalition's location estimate. Breadth is not
        # an unconditional multiplier: homogeneous repetitions communicate the
        # same option-level claim rather than successively stronger claims.
        location = float(np.mean(members))
        dispersion = float(np.mean((members - location) ** 2))

        # Inferred coalition trustworthiness has a threshold-shaped collapse.
        # Ordinary heterogeneity is tolerated, whereas a maximally split
        # coalition is treated as unable to sustain a unified assertion.
        steepness = float(parameters["dispersion_steepness"])
        z = np.clip(steepness * (dispersion - dispersion_threshold), -60.0, 60.0)
        core_trust = 1.0 / (1.0 + np.exp(z))
        tail = float(parameters["trust_tail"])
        trust = tail + (1.0 - tail) * core_trust

        # Very small coalitions carry somewhat less metacognitive certainty,
        # but this saturates rapidly and cannot generate repeated-consensus
        # reactance in broad homogeneous coalitions.
        sampling_confidence = 1.0 - np.exp(
            -float(breadth) / float(parameters["sampling_scale"])
        )
        metacognitive_trust = trust * (
            float(parameters["sampling_floor"])
            + (1.0 - float(parameters["sampling_floor"])) * sampling_confidence
        )
        claim = location * trust
        return float(claim), float(metacognitive_trust), dispersion, breadth

    claim_a, trust_a, dispersion_a, breadth_a = coalition(1.0)
    claim_b, trust_b, dispersion_b, breadth_b = coalition(-1.0)

    if active_count == 0:
        reliability_contrast = 0.0
        pooled_dispersion = 0.0
        claim_certainty = 0.0
    else:
        # Relative normalization makes claim comparison breadth invariant for
        # equal homogeneous coalitions and bounds it under arbitrary designs.
        claim_mass = abs(claim_a) + abs(claim_b)
        reliability_contrast = (claim_a - claim_b) / (
            float(parameters["claim_floor"]) + claim_mass
        )
        reliability_contrast = float(np.clip(reliability_contrast, -1.0, 1.0))

        pooled_dispersion = (
            breadth_a * dispersion_a + breadth_b * dispersion_b
        ) / float(active_count)

        # A contrast can be scrutinized when at least one side supports a
        # trustworthy claim. Thus discounting a split coalition can expose the
        # opposing homogeneous coalition as the salient overclaim.
        strongest_trust = max(trust_a, trust_b)
        claim_certainty = strongest_trust * abs(reliability_contrast)

    # Skeptical interpretation reverses a salient reliability-weighted claim.
    # Individual differences are amplified in low-margin, representation-level
    # conflicts rather than uniformly across all trials.
    skeptical_trait = float(parameters["skeptical_orientation"])
    ambiguity_index = 1.0 / (
        1.0 + (margin_magnitude / float(parameters["ambiguity_margin_scale"])) ** 2
    )
    skeptical_multiplier = 1.0 + float(parameters["skeptic_trait_spread"]) * (
        skeptical_trait - 0.5
    ) * (0.35 + 0.65 * ambiguity_index)
    skeptical_gain = float(parameters["skeptical_gain"]) * max(
        skeptical_multiplier, 0.10
    )
    bounded_claim = np.tanh(
        float(parameters["claim_accumulation"]) * reliability_contrast
    )
    skeptical_evidence = -skeptical_gain * float(bounded_claim)

    # The independent social-count representation has convex initial growth,
    # an intermediate steep region, and a confidence-limited asymptote.
    if margin_magnitude < 1e-12:
        tally_shape = 0.0
    else:
        tally_power = float(parameters["tally_power"])
        half_margin = float(parameters["tally_half_margin"])
        numerator = margin_magnitude ** tally_power
        tally_shape = numerator / (half_margin ** tally_power + numerator)

    # Tally-gain heterogeneity is attenuated as count evidence becomes decisive,
    # helping preserve the low between-subject variance of clear margin series.
    tally_trait = float(parameters["tally_gain_trait"])
    tally_uncertainty = 1.0 - tally_shape
    tally_multiplier = 1.0 + float(parameters["tally_trait_spread"]) * (
        tally_trait - 0.5
    ) * (0.25 + 0.75 * tally_uncertainty)
    tally_gain = float(parameters["tally_gain"]) * max(tally_multiplier, 0.10)
    tally_evidence = (
        tally_gain * np.sign(tally_margin) * tally_shape
        if margin_magnitude > 0.0 else 0.0
    )

    # Confidence control selectively attenuates moderate margins, where raw
    # counting is informative but not yet decisive. The Gaussian window has
    # little effect on one-vote trials or margins seven and above.
    mid_margin_window = np.exp(
        -0.5 * (
            (margin_magnitude - float(parameters["mid_margin_center"]))
            / float(parameters["mid_margin_width"])
        ) ** 2
    )
    tally_evidence *= 1.0 - float(
        parameters["mid_margin_tally_damping"]
    ) * mid_margin_window

    # Arbitration depends jointly on claim reliability, coalition dispersion,
    # tally decisiveness, and disagreement between the two construals.
    normalized_claim_direction = float(np.tanh(skeptical_evidence))
    normalized_tally_direction = float(np.tanh(tally_evidence))
    disagreement = abs(
        normalized_claim_direction - normalized_tally_direction
    ) / 2.0
    dispersion_index = pooled_dispersion / (
        pooled_dispersion + float(parameters["dispersion_reference"])
    ) if active_count > 0 else 0.0

    # A localized boost admits skeptical scrutiny of moderate reliability
    # contrasts only when the tally is unresolved and both coalitions remain
    # trustworthy. It therefore does not revive a maximally split coalition.
    both_trustworthy = min(trust_a, trust_b)
    contrast_magnitude = abs(reliability_contrast)
    moderate_center = float(parameters["moderate_contrast_center"])
    moderate_width = float(parameters["moderate_contrast_width"])
    moderate_band = np.exp(
        -0.5 * ((contrast_magnitude - moderate_center) / moderate_width) ** 2
    )
    tie_index = np.exp(-(margin_magnitude / 0.35) ** 2)
    moderate_tie_boost = (
        float(parameters["moderate_tie_gate_boost"])
        * tie_index
        * both_trustworthy
        * moderate_band
    )

    gate_drive = (
        float(parameters["certainty_weight"]) * claim_certainty
        + float(parameters["disagreement_weight"]) * disagreement
        - float(parameters["dispersion_uncertainty_weight"]) * dispersion_index
        - float(parameters["margin_resolution_weight"]) * tally_shape
        - arbitration_threshold
        + moderate_tie_boost
    )
    gate_temperature = float(parameters["gate_temperature"])
    gate_logit = np.clip(gate_drive / gate_temperature, -60.0, 60.0)
    claim_weight = 1.0 / (1.0 + np.exp(-gate_logit))

    # Arbitration is a reliability-weighted competition between construals,
    # not an additive fixed coalition-plus-tally score.
    total_evidence = (
        claim_weight * skeptical_evidence
        + (1.0 - claim_weight) * tally_evidence
    )

    # Precision differences matter chiefly when the stimulus remains ambiguous;
    # their expression contracts on decisive tallies.
    precision_trait = float(parameters["precision_trait"])
    unresolved = 1.0 - tally_shape * (1.0 - claim_weight)
    precision_multiplier = 1.0 + float(parameters["precision_trait_spread"]) * (
        precision_trait - 0.5
    ) * (0.20 + 0.80 * unresolved)
    precision = float(parameters["base_precision"]) * max(
        precision_multiplier, 0.10
    )

    # Selective confidence limiting pulls one-vote conflicts toward chance
    # without changing the moderate-to-large tally asymptote. A second limiter
    # applies to endpoint-sized bilateral claims only when both coalitions are
    # trustworthy, preserving the extreme-dispersion crossover.
    nonzero_margin_index = 1.0 - np.exp(-(margin_magnitude / 0.35) ** 2)
    small_margin_load = nonzero_margin_index * np.exp(
        -max(margin_magnitude - 1.0, 0.0)
        / float(parameters["small_margin_decay"])
    )
    precision *= 1.0 - float(
        parameters["small_margin_precision_damping"]
    ) * small_margin_load

    extreme_center = float(parameters["extreme_contrast_center"])
    extreme_width = float(parameters["extreme_contrast_width"])
    extreme_z = np.clip(
        (contrast_magnitude - extreme_center) / extreme_width, -60.0, 60.0
    )
    extreme_contrast_index = 1.0 / (1.0 + np.exp(-extreme_z))
    trusted_extreme_load = tie_index * both_trustworthy * extreme_contrast_index
    precision *= 1.0 - float(
        parameters["trusted_extreme_precision_damping"]
    ) * trusted_extreme_load

    decision_variable = float(precision * total_evidence)

    # Symmetric logits ensure exact invariance when A and B are exchanged.
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
