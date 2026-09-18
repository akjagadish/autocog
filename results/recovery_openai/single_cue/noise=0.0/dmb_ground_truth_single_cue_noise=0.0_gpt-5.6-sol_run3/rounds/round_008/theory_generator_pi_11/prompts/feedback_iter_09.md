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
- THEORY 2 = `pi_10`
- The recommendation below acts on THEORY 1 (= `pi_8`).

Replace pi_8 with a new theory of coherence-gated, breadth-normalized skeptical comparison. The new theory should retain pi_10's empirically supported idea that internal heterogeneity weakens a coalition representation: when the opposing coalition becomes dispersed, its representation falls, increasing the relative prominence of the homogeneous coalition and therefore increasing skeptical reaction against that homogeneous coalition, producing the negative gradient in Experiment 1. Unlike pi_10, however, coalition repetition should not intrinsically produce a large increase in reactance. Add two approximately offsetting breadth processes: sublinear consensus formation and divisive normalization by total active evidence or limited attention. For equal-sized homogeneous coalitions, these should yield an almost breadth-invariant validity contrast, matching Experiment 2. Use a separate nonlinear tally channel that protects large margins but is less steep and less confident than pi_10's current channel, thereby matching Experiments 5-8 and 11 while retaining the strong tally evidence in Experiments 4 and 6. Validity should be encoded on a smooth saturating continuum, without pi_8's 95% compression notch or a literal-certainty bonus. Strong 100%-versus-50% reversals should emerge from maximal relative contrast and contextual normalization, not from an endpoint-specific mechanism. Include continuous subject variation in skeptical orientation, tally sensitivity, and noise, with heterogeneity allowed to depend on conflict composition and margin; avoid both discrete skeptic classes and blanket variance inflation. The resulting theory should jointly target the strong negative dispersion effect, negligible homogeneous breadth effect, substantial but non-ceiling majority reliance, and smooth endpoint behavior.


## CANDIDATE THEORY
Coherence-Gated Relative Coalition Comparison theory proposes that people compress endorsements into option-level coalition representations before making a choice. Each representation combines smoothly encoded mean assertiveness, sublinear consensus, and a coherence gate. Coalition heterogeneity is evaluated relative to the maximum variance feasible at that coalition's own mean assertiveness, yielding a roster-independent index based only on active coalition members. The coherence gate remains near one for homogeneous, adjacent-validity, and moderately heterogeneous coalitions, but falls steeply near maximal normalized heterogeneity. Consequently, a maximally dispersed opposing coalition makes a homogeneous coalition relatively prominent and thus a stronger target of skeptical interpretation, without making routine coalition expansion intrinsically reactance-producing. Sublinear consensus is offset by breadth and active-evidence normalization, yielding approximately breadth-invariant comparison of equal-sized homogeneous coalitions. A narrow-conflict focus rule distinguishes maximal relative contrasts from ordinary singleton disagreements: a smooth 100%-versus-50% contrast remains strongly skepticism-provoking, while less extreme singleton contrasts are attenuated without an endpoint-specific bonus. A separate validity-blind tally channel grows concavely with vote margin. Its gain is reduced while its curvature is increased, pivoting tally evidence away from one-vote margins and toward moderate and large margins. Continuous correlation between tally orientation and response noise prevents high-tally subjects from becoming deterministically majority-following. Skeptical orientation, tally sensitivity, and decision noise vary continuously across people, with composition-dependent heterogeneity concentrated in genuinely dispersed conflicts. With no correctness feedback, processing remains history independent and exactly option-swap symmetric.

`predict(parameters, state, history) -> np.ndarray`:
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

        # Normalize observed variance by the maximum variance feasible for
        # bounded [0, 1] strengths at this coalition's own mean. This depends
        # only on active coalition members and is invariant to unused experts.
        feasible_variance = max(mean_strength * (1.0 - mean_strength), 1e-12)
        normalized_heterogeneity = float(np.clip(
            dispersion / feasible_variance, 0.0, 1.0
        ))

        # Coherence remains intact below a high normalized-heterogeneity
        # threshold and then falls steeply in the upper tail. The continuous
        # skeptical trait shifts that threshold only modestly.
        threshold = float(parameters["coherence_scale"]) + 0.15 * float(
            parameters["coherence_trait_coupling"]
        ) * (skeptic_trait - 0.5)
        threshold = float(np.clip(threshold, 0.35, 0.82))
        upper_tail = np.clip(
            (normalized_heterogeneity - threshold) / max(1.0 - threshold, 1e-12),
            0.0,
            1.0,
        )
        floor = float(parameters["coherence_floor"])
        coherence = floor + (1.0 - floor) * np.exp(
            -4.0 * upper_tail ** float(parameters["coherence_power"])
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
- validity_curvature: [1.6, 3.0]
- consensus_exponent: [0.58, 0.76]
- coherence_scale: [0.50, 0.68]
- coherence_floor: [0.05, 0.25]
- coherence_power: [1.8, 3.0]
- coherence_trait_coupling: [-0.70, 0.70]
- attention_floor: [0.28, 0.46]
- evidence_normalization: [0.48, 0.76]
- contrast_accumulation: [1.15, 1.75]
- singleton_focus_floor: [0.45, 0.70]
- skeptical_strength: [1.85, 2.85]
- skeptical_orientation: [0, 1]
- composition_trait_gain: [0.30, 0.90]
- skeptic_margin_coupling: [-0.30, 0.45]
- tally_sensitivity: [0.25, 0.43]
- tally_trait: [0, 1]
- tally_exponent: [0.78, 0.96]
- choice_sensitivity: [0.82, 1.18]
- noise_trait: [0, 1]
- tally_noise_correlation: [0.15, 0.35]
- conflict_noise: [0.10, 0.55]
- lapse_rate: [0.0, 0.08]
- validities: validities

`rationale`:
This is a coherence-only edit to the accepted iteration-7 candidate. The accepted tally equation, tally–noise interaction, contextual normalization, skeptical focus, smooth validity code, endpoint treatment, and all other parameter ranges are unchanged. Within summarize, raw dispersion divided by a fixed scale is replaced by dispersion divided by mean_strength * (1 - mean_strength), the maximum variance feasible for bounded assertiveness values conditional on the coalition's own mean. The resulting index is clipped to [0, 1] and depends only on active members, avoiding the rejected roster-wide baseline. The former scale parameter is narrowly repurposed as a high heterogeneity threshold. Coherence stays exactly near one below that threshold and collapses steeply only in its upper tail. This should preserve representations for routine adjacent-validity expansion and moderately heterogeneous coalitions, improving Experiments 5 and 16, while selectively weakening nearly maximally dispersed coalitions and strengthening the negative gradient in Experiment 17. Because singleton coalitions have zero internal heterogeneity, the edit does not globally increase skepticism or worsen the already excessive singleton response in Experiment 8. Continuous trait variation is retained through a modest shift of the upper-tail threshold.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.1643 -> ACCEPTED
- iter 2: loss=0.1297 -> ACCEPTED
- iter 3: loss=0.1461 -> REJECTED
- iter 4: loss=0.1663 -> REJECTED
- iter 5: loss=0.1321 -> REJECTED
- iter 6: loss=0.1524 -> REJECTED
- iter 7: loss=0.1180 -> ACCEPTED
- iter 8: loss=0.1257 -> REJECTED
- iter 9: loss=0.1333 -> REJECTED
- iter 10 (current candidate you are grading): loss=0.1591 -> REJECTED
Running-best (last accepted) base: iter 7 at loss=0.1180.

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
  - iter 1: 0.1804 (var=0.0041) (Δ vs real -0.2229)
  - iter 2: 0.4325 (var=0.0125) (Δ vs real +0.0292)
  - iter 3: 0.3642 (var=0.0089) (Δ vs real -0.0392)
  - iter 4: 0.1921 (var=0.0055) (Δ vs real -0.2112)
  - iter 5: 0.3912 (var=0.0088) (Δ vs real -0.0121)
  - iter 6: 0.4925 (var=0.0098) (Δ vs real +0.0892)
  - iter 7: 0.4146 (var=0.0072) (Δ vs real +0.0113)
  - iter 8: 0.4233 (var=0.0085) (Δ vs real +0.0200)
  - iter 9: 0.4242 (var=0.0091) (Δ vs real +0.0208)
  - iter 10 (current): 0.5121 (var=0.0201) (Δ vs real +0.1088)
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
  - iter 1: 142.3953 (var=3.4889) (Δ vs real +53.2531)
  - iter 2: 32.2968 (var=3.7975) (Δ vs real -56.8455)
  - iter 3: -10.6193 (var=2.4866) (Δ vs real -99.7616)
  - iter 4: 32.0319 (var=5.2483) (Δ vs real -57.1103)
  - iter 5: 14.3210 (var=5.5998) (Δ vs real -74.8212)
  - iter 6: -19.4052 (var=7.1342) (Δ vs real -108.5475)
  - iter 7: 37.8180 (var=3.6767) (Δ vs real -51.3243)
  - iter 8: 17.4412 (var=3.1956) (Δ vs real -71.7011)
  - iter 9: 21.9599 (var=1.9867) (Δ vs real -67.1824)
  - iter 10 (current): -158.9834 (var=7.2886) (Δ vs real -248.1256)
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
  - iter 1: 0.2225 (var=0.0036) (Δ vs real -0.2283)
  - iter 2: 0.3765 (var=0.0046) (Δ vs real -0.0744)
  - iter 3: 0.3821 (var=0.0034) (Δ vs real -0.0687)
  - iter 4: 0.2685 (var=0.0041) (Δ vs real -0.1823)
  - iter 5: 0.3656 (var=0.0040) (Δ vs real -0.0852)
  - iter 6: 0.3769 (var=0.0039) (Δ vs real -0.0740)
  - iter 7: 0.3860 (var=0.0039) (Δ vs real -0.0648)
  - iter 8: 0.3871 (var=0.0026) (Δ vs real -0.0637)
  - iter 9: 0.3733 (var=0.0037) (Δ vs real -0.0775)
  - iter 10 (current): 0.3969 (var=0.0033) (Δ vs real -0.0540)
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
  - iter 1: 0.9531 (var=0.0016) (Δ vs real +0.1319)
  - iter 2: 0.8769 (var=0.0048) (Δ vs real +0.0556)
  - iter 3: 0.8569 (var=0.0084) (Δ vs real +0.0356)
  - iter 4: 0.9275 (var=0.0038) (Δ vs real +0.1062)
  - iter 5: 0.8894 (var=0.0089) (Δ vs real +0.0681)
  - iter 6: 0.8881 (var=0.0056) (Δ vs real +0.0669)
  - iter 7: 0.8656 (var=0.0058) (Δ vs real +0.0444)
  - iter 8: 0.8606 (var=0.0077) (Δ vs real +0.0394)
  - iter 9: 0.8581 (var=0.0065) (Δ vs real +0.0369)
  - iter 10 (current): 0.8544 (var=0.0093) (Δ vs real +0.0331)
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
  - iter 1: -0.5277 (var=0.0478) (Δ vs real -0.4896)
  - iter 2: -0.4070 (var=0.0585) (Δ vs real -0.3688)
  - iter 3: -0.4819 (var=0.0444) (Δ vs real -0.4438)
  - iter 4: -0.3874 (var=0.0675) (Δ vs real -0.3493)
  - iter 5: -0.4203 (var=0.0736) (Δ vs real -0.3821)
  - iter 6: -0.4368 (var=0.0422) (Δ vs real -0.3987)
  - iter 7: -0.4099 (var=0.0525) (Δ vs real -0.3717)
  - iter 8: -0.4126 (var=0.0623) (Δ vs real -0.3744)
  - iter 9: -0.4078 (var=0.0574) (Δ vs real -0.3697)
  - iter 10 (current): -0.3614 (var=0.0461) (Δ vs real -0.3233)
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
  - iter 1: 0.9544 (var=0.0014) (Δ vs real +0.1081)
  - iter 2: 0.9004 (var=0.0037) (Δ vs real +0.0542)
  - iter 3: 0.8998 (var=0.0041) (Δ vs real +0.0535)
  - iter 4: 0.9073 (var=0.0039) (Δ vs real +0.0610)
  - iter 5: 0.8950 (var=0.0042) (Δ vs real +0.0488)
  - iter 6: 0.9104 (var=0.0044) (Δ vs real +0.0642)
  - iter 7: 0.8883 (var=0.0034) (Δ vs real +0.0421)
  - iter 8: 0.9004 (var=0.0036) (Δ vs real +0.0542)
  - iter 9: 0.8831 (var=0.0036) (Δ vs real +0.0369)
  - iter 10 (current): 0.8527 (var=0.0063) (Δ vs real +0.0065)
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
  - iter 1: 0.1893 (var=0.0053) (Δ vs real +0.0732)
  - iter 2: 0.1619 (var=0.0045) (Δ vs real +0.0457)
  - iter 3: 0.1513 (var=0.0062) (Δ vs real +0.0352)
  - iter 4: 0.2077 (var=0.0067) (Δ vs real +0.0915)
  - iter 5: 0.1457 (var=0.0051) (Δ vs real +0.0296)
  - iter 6: 0.1558 (var=0.0049) (Δ vs real +0.0397)
  - iter 7: 0.1882 (var=0.0066) (Δ vs real +0.0721)
  - iter 8: 0.1573 (var=0.0062) (Δ vs real +0.0412)
  - iter 9: 0.2025 (var=0.0056) (Δ vs real +0.0863)
  - iter 10 (current): 0.1582 (var=0.0066) (Δ vs real +0.0421)
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
  - iter 1: -5.2294 (var=0.0196) (Δ vs real -1.6922)
  - iter 2: -5.1204 (var=0.0155) (Δ vs real -1.5832)
  - iter 3: -5.2145 (var=0.0134) (Δ vs real -1.6773)
  - iter 4: -4.9237 (var=0.0160) (Δ vs real -1.3866)
  - iter 5: -4.9730 (var=0.0217) (Δ vs real -1.4358)
  - iter 6: -5.1610 (var=0.0167) (Δ vs real -1.6238)
  - iter 7: -4.8005 (var=0.0247) (Δ vs real -1.2633)
  - iter 8: -4.7108 (var=0.0191) (Δ vs real -1.1737)
  - iter 9: -4.7589 (var=0.0161) (Δ vs real -1.2217)
  - iter 10 (current): -4.6152 (var=0.0207) (Δ vs real -1.0780)
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
  - iter 1: 0.2789 (var=0.0047) (Δ vs real -0.0459)
  - iter 2: 0.2628 (var=0.0043) (Δ vs real -0.0621)
  - iter 3: 0.2643 (var=0.0032) (Δ vs real -0.0605)
  - iter 4: 0.2769 (var=0.0039) (Δ vs real -0.0480)
  - iter 5: 0.2720 (var=0.0048) (Δ vs real -0.0528)
  - iter 6: 0.2649 (var=0.0051) (Δ vs real -0.0599)
  - iter 7: 0.2758 (var=0.0047) (Δ vs real -0.0490)
  - iter 8: 0.2764 (var=0.0042) (Δ vs real -0.0485)
  - iter 9: 0.2688 (var=0.0031) (Δ vs real -0.0561)
  - iter 10 (current): 0.2024 (var=0.0066) (Δ vs real -0.1224)
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
  - iter 1: 0.8817 (var=0.0061) (Δ vs real +0.0200)
  - iter 2: 0.8908 (var=0.0058) (Δ vs real +0.0292)
  - iter 3: 0.8542 (var=0.0104) (Δ vs real -0.0075)
  - iter 4: 0.8600 (var=0.0082) (Δ vs real -0.0017)
  - iter 5: 0.8825 (var=0.0057) (Δ vs real +0.0208)
  - iter 6: 0.8767 (var=0.0078) (Δ vs real +0.0150)
  - iter 7: 0.8683 (var=0.0072) (Δ vs real +0.0067)
  - iter 8: 0.8650 (var=0.0055) (Δ vs real +0.0033)
  - iter 9: 0.8717 (var=0.0054) (Δ vs real +0.0100)
  - iter 10 (current): 0.8483 (var=0.0069) (Δ vs real -0.0133)
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
  - iter 1: 0.2169 (var=0.0118) (Δ vs real +0.0597)
  - iter 2: 0.1603 (var=0.0065) (Δ vs real +0.0031)
  - iter 3: 0.1714 (var=0.0044) (Δ vs real +0.0142)
  - iter 4: 0.2128 (var=0.0058) (Δ vs real +0.0556)
  - iter 5: 0.1650 (var=0.0062) (Δ vs real +0.0078)
  - iter 6: 0.1694 (var=0.0057) (Δ vs real +0.0122)
  - iter 7: 0.1861 (var=0.0059) (Δ vs real +0.0289)
  - iter 8: 0.1897 (var=0.0076) (Δ vs real +0.0325)
  - iter 9: 0.1831 (var=0.0079) (Δ vs real +0.0258)
  - iter 10 (current): 0.2111 (var=0.0095) (Δ vs real +0.0539)
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
  - iter 1: 0.1422 (var=0.0041) (Δ vs real +0.0161)
  - iter 2: 0.1472 (var=0.0039) (Δ vs real +0.0211)
  - iter 3: 0.1414 (var=0.0043) (Δ vs real +0.0153)
  - iter 4: 0.1581 (var=0.0067) (Δ vs real +0.0319)
  - iter 5: 0.1456 (var=0.0042) (Δ vs real +0.0194)
  - iter 6: 0.1436 (var=0.0051) (Δ vs real +0.0175)
  - iter 7: 0.1653 (var=0.0043) (Δ vs real +0.0392)
  - iter 8: 0.1444 (var=0.0063) (Δ vs real +0.0183)
  - iter 9: 0.1650 (var=0.0043) (Δ vs real +0.0389)
  - iter 10 (current): 0.1531 (var=0.0041) (Δ vs real +0.0269)
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
  - iter 1: 0.0063 (var=0.0103) (Δ vs real -0.0031)
  - iter 2: 0.0253 (var=0.0118) (Δ vs real +0.0159)
  - iter 3: -0.0003 (var=0.0152) (Δ vs real -0.0097)
  - iter 4: 0.0369 (var=0.0131) (Δ vs real +0.0275)
  - iter 5: -0.0056 (var=0.0085) (Δ vs real -0.0150)
  - iter 6: -0.0091 (var=0.0106) (Δ vs real -0.0184)
  - iter 7: -0.0069 (var=0.0147) (Δ vs real -0.0163)
  - iter 8: -0.0091 (var=0.0123) (Δ vs real -0.0184)
  - iter 9: -0.0066 (var=0.0081) (Δ vs real -0.0159)
  - iter 10 (current): 0.0350 (var=0.0082) (Δ vs real +0.0256)
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
  - iter 1: 0.1946 (var=0.0088) (Δ vs real +0.0165)
  - iter 2: 0.2083 (var=0.0116) (Δ vs real +0.0303)
  - iter 3: 0.2111 (var=0.0116) (Δ vs real +0.0331)
  - iter 4: 0.2196 (var=0.0111) (Δ vs real +0.0415)
  - iter 5: 0.1911 (var=0.0141) (Δ vs real +0.0131)
  - iter 6: 0.1974 (var=0.0115) (Δ vs real +0.0193)
  - iter 7: 0.1924 (var=0.0099) (Δ vs real +0.0143)
  - iter 8: 0.1822 (var=0.0120) (Δ vs real +0.0042)
  - iter 9: 0.2094 (var=0.0114) (Δ vs real +0.0314)
  - iter 10 (current): 0.1883 (var=0.0124) (Δ vs real +0.0103)
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
  - iter 1: 0.0000 (var=0.0900) (Δ vs real +0.0000)
  - iter 2: 0.0000 (var=0.1924) (Δ vs real +0.0000)
  - iter 3: 0.0000 (var=0.1824) (Δ vs real +0.0000)
  - iter 4: 0.0000 (var=0.1924) (Δ vs real +0.0000)
  - iter 5: 0.0000 (var=0.2016) (Δ vs real +0.0000)
  - iter 6: 0.0000 (var=0.1056) (Δ vs real +0.0000)
  - iter 7: 0.0000 (var=0.1476) (Δ vs real +0.0000)
  - iter 8: 0.0000 (var=0.2016) (Δ vs real +0.0000)
  - iter 9: 0.0000 (var=0.1204) (Δ vs real +0.0000)
  - iter 10 (current): 0.0000 (var=0.1344) (Δ vs real +0.0000)
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
  - iter 1: 0.6166 (var=0.0054) (Δ vs real -0.2653)
  - iter 2: 0.6541 (var=0.0076) (Δ vs real -0.2278)
  - iter 3: 0.6197 (var=0.0057) (Δ vs real -0.2622)
  - iter 4: 0.6547 (var=0.0055) (Δ vs real -0.2272)
  - iter 5: 0.6200 (var=0.0046) (Δ vs real -0.2619)
  - iter 6: 0.6650 (var=0.0054) (Δ vs real -0.2169)
  - iter 7: 0.6578 (var=0.0065) (Δ vs real -0.2241)
  - iter 8: 0.6559 (var=0.0075) (Δ vs real -0.2259)
  - iter 9: 0.6434 (var=0.0082) (Δ vs real -0.2384)
  - iter 10 (current): 0.6359 (var=0.0061) (Δ vs real -0.2459)
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
  - iter 1: -0.1569 (var=0.0126) (Δ vs real +0.1681)
  - iter 2: -0.1844 (var=0.0135) (Δ vs real +0.1406)
  - iter 3: -0.2281 (var=0.0159) (Δ vs real +0.0969)
  - iter 4: -0.2075 (var=0.0130) (Δ vs real +0.1175)
  - iter 5: -0.2481 (var=0.0151) (Δ vs real +0.0769)
  - iter 6: -0.2025 (var=0.0132) (Δ vs real +0.1225)
  - iter 7: -0.1881 (var=0.0178) (Δ vs real +0.1369)
  - iter 8: -0.1875 (var=0.0163) (Δ vs real +0.1375)
  - iter 9: -0.1787 (var=0.0154) (Δ vs real +0.1462)
  - iter 10 (current): 0.0031 (var=0.0124) (Δ vs real +0.3281)
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
  - iter 1: -0.0033 (var=0.0092) (Δ vs real -0.0092)
  - iter 2: -0.0096 (var=0.0085) (Δ vs real -0.0154)
  - iter 3: 0.0025 (var=0.0068) (Δ vs real -0.0033)
  - iter 4: 0.0292 (var=0.0099) (Δ vs real +0.0233)
  - iter 5: 0.0179 (var=0.0080) (Δ vs real +0.0121)
  - iter 6: 0.0188 (var=0.0078) (Δ vs real +0.0129)
  - iter 7: 0.0004 (var=0.0101) (Δ vs real -0.0054)
  - iter 8: 0.0050 (var=0.0111) (Δ vs real -0.0008)
  - iter 9: 0.0108 (var=0.0063) (Δ vs real +0.0050)
  - iter 10 (current): 0.0221 (var=0.0105) (Δ vs real +0.0163)
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

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed coherence-gated, breadth-normalized skeptical-comparison family and improves several key qualitative targets: endpoint behavior is smooth (Experiments 10 and 13), the homogeneous breadth effect is negligible in Experiment 18, the endpoint breadth gradient is close in Experiment 14, and the dispersion effect has the correct sign in Experiment 17. However, the joint fit is not yet adequate. The tally channel is substantially too dominant and too deterministic: TTB/minority choices are far too rare in Experiments 1 and 3 (0.180 versus 0.403; 0.223 versus 0.451), evidence for tallying is excessive in Experiment 2 (142 versus 89), and large-majority reliance is near ceiling in Experiments 4 and 6 (0.953 versus 0.821; 0.954 versus 0.846). The same imbalance likely contributes to excessive majority influence in Experiments 7 and 11. Subject variance is also much too low in Experiments 2, 4, and 6, showing that the continuous tally/noise traits do not create enough heterogeneity on homogeneous high-margin trials. Separately, the model still does not produce enough coherence-dependent skeptical behavior: Experiment 16 is much too low (0.617 versus 0.882), and the negative dispersion gradient in Experiment 17 is only about half the observed magnitude (-0.157 versus -0.325). Yet skepticism is already excessive in the small-coalition reliability conflicts of Experiment 8 (-5.229 versus -3.537). Finally, Experiment 5 shows a major failure of the intended breadth cancellation (-0.528 versus -0.038). Matching the consensus exponent in the denominator cancels pure replication, but it does not cancel changes in coalition mean assertiveness, active-mean normalization, and coherence as differently valid experts are added.
Rationale: Keep the prescribed mechanism, but recalibrate its channels. First, reduce the effective tally contribution—principally tally_sensitivity and/or choice gain, with a slightly more concave tally curve—so that margins remain influential without generating ceiling choices. Add stronger continuous subject variation specifically in tally sensitivity or gain at moderate and large margins; the current conflict-noise term operates mainly on dispersed low-margin trials and therefore cannot reproduce the observed variance in Experiments 2, 4, and 6. A modestly broader lapse or temperature range can help, but should not substitute for correcting the tally-to-skepticism balance. Second, strengthen the differential coherence penalty while avoiding a global increase in skeptical_strength: a global increase would worsen Experiment 8. Use a smoother but steeper composition-sensitive coherence gate, or stronger coherence-trait coupling, that responds to the dispersion manipulation in Experiments 16-17 but is weak for singleton conflicts. Third, repair breadth normalization for Experiment 5. The cancellation must cover composition-induced changes in mean assertiveness and coherence, not only the shared n^alpha factor. A symmetric normalization based on the two coalition strength scales, or coherence computed relative to a breadth/composition baseline, could preserve the strong dispersion contrast while preventing ordinary coalition expansion from creating a large breadth trend. Preserve the current smooth validity code and endpoint treatment, which fit Experiments 10, 13, and 14 well.

**Outcome of your advice:** iter 1 candidate loss=0.1643 -> iter 2 candidate loss=0.1297 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The accepted revision substantially improves the aggregate fit and validates much of the previous tuning direction. Experiments 1, 4, 6, and 11 move markedly toward the data, while the model retains good smooth-endpoint behavior in Experiments 10, 13, and 15 and negligible breadth dependence in Experiment 18. It remains faithful to the prescribed coherence-gated, breadth-normalized skeptical-comparison family. Nevertheless, several diagnostic failures remain too large to ship. The tally recalibration overshot in Experiment 2: evidence favoring tallying fell from 142.4 to 32.3, well below the observed 89.1. At the same time, the margin-one conflicts in Experiment 3 still show too few minority choices (0.377 versus 0.451), whereas majority choice remains somewhat excessive at larger margins in Experiments 4 and 6. Thus a single concave power curve is not producing the required margin profile. Experiment 5 remains the largest structural failure of breadth cancellation (-0.407 versus -0.038). The explicit narrow_weight makes skeptical focus increase with breadth, counteracting the intended normalization. Skepticism is also too strong in ordinary singleton reliability conflicts in Experiment 8 (-5.12 versus -3.54), but too weak for broad coalition conflicts in Experiment 16 (0.654 versus 0.882) and for the dispersion manipulation in Experiment 17 (-0.184 versus -0.325). The latter pattern indicates that skeptical strength is still allocated to the wrong compositions rather than simply being globally too weak or strong.
Rationale: Build on this accepted candidate, but make targeted in-family adjustments. First, do not continue the previous global tally reduction; it was helpful overall but overshot Experiment 2. Replace or augment the current pure power tally curve with a smooth thresholded or Hill-shaped margin function that contributes very little at margin one, rises across moderate margins, and saturates before becoming deterministic. This should raise minority choice in Experiment 3 and restore the margin-dependent Bayes factor in Experiment 2 without returning Experiments 4 and 6 to ceiling. Tune the new curve between the two iterations rather than simply restoring the former tally gain. Second, remove the breadth dependence from contrast_focus. Its current exp(-(total_breadth-2)/2) term directly makes equally balanced broad coalitions more skepticism-provoking than singleton coalitions and is the likely source of the Experiment 5 failure. Let focus depend smoothly on normalized contrast extremity instead of breadth, so ordinary singleton contrasts can remain attenuated in Experiment 8 while the maximal 100%-versus-50% contrast in Experiment 10 stays strong and equal-sized replication remains neutral. Third, sharpen and slightly lower the threshold of the coherence gate for moderate-to-maximal within-coalition dispersion, while keeping homogeneous coalitions unaffected. Couple any additional skeptical gain to dispersed broad coalitions rather than increasing skeptical_strength globally. This should continue the accepted improvement in Experiments 16-17 and also reduce the residual conventional choices in Experiments 7 and 12 without worsening Experiment 8. Preserve the smooth validity code and endpoint treatment, which are already performing well.

**Outcome of your advice:** iter 2 candidate loss=0.1297 -> iter 3 candidate loss=0.1461 -> the gate marked it REJECTED.

### Iteration 3
Verdict: regenerate
Interpretation: The current candidate remains faithful to the prescribed coherence-gated, breadth-normalized skeptical-comparison family, and it retains several strong fits: Experiments 10, 12, 13, 15, and 18 are close, Experiment 17 moves toward the required negative dispersion gradient, and Experiments 4, 6, 7, and 11 are directionally reasonable. However, this iteration was rejected by the accept gate because aggregate loss increased from 0.1297 to 0.1461. The attempted Hill tally channel is the clearest regression. Experiment 2 falls from an already-low 32.3 to -10.6 against an observed 89.1, indicating that predicted choice probability no longer increases sufficiently across the experiment's tally margins. It only slightly improves the margin-one minority rate in Experiment 3 and leaves large-margin reliance too high in Experiment 6. Removing breadth dependence from skeptical focus also fails to repair Experiment 5 and instead worsens it from -0.407 to -0.482, far from -0.038. The stronger broad-dispersion adjustment improves Experiment 17 to -0.228, but reduces Experiment 16 from 0.654 to 0.620 versus 0.882. This implies that the revised coherence gate is suppressing moderately dispersed opposing coalitions too early: the model needs high baseline representation for homogeneous and moderately dispersed coalitions, followed by a sharper loss only near maximal dispersion. Ordinary singleton skepticism remains excessive in Experiment 8 (-5.21 versus -3.54), while the maximal endpoint reversal in Experiment 10 is correctly preserved.
Rationale: Build again from the unchanged accepted iteration-2 base, not from this rejected candidate. Do not repeat the wholesale Hill substitution or simple removal of breadth-dependent focus; the gate shows that bundle was counterproductive. For the tally channel, retain the accepted base's basic margin function and adjust effective decision temperature or continuous tally heterogeneity by margin instead. A smooth U-shaped noise profile—more noise at margin one, less at moderate margins, and somewhat more again at very large margins—can raise minority choices in Experiment 3, generate the strong margin-dependent evidence required by Experiment 2, and prevent Experiments 4 and 6 from becoming too confident. This uses the recommendation's permitted margin-dependent heterogeneity without flattening the tally curve through early saturation. For Experiment 5, restore the accepted focus term and repair the normalization itself: cancel equal-coalition breadth and active-evidence changes at the coalition-representation level, rather than relying on a common denominator containing active_mean. For example, compare coalition means after separately removing each coalition's n^alpha consensus scale, with a symmetric bounded normalization; keep coherence as the composition-sensitive deviation from this breadth baseline. Finally, move the coherence transition upward rather than lowering it further, while making its upper tail steeper. Homogeneous and moderately dispersed opponents should retain substantial representation, supporting the high baseline in Experiment 16, whereas maximally dispersed opponents should collapse sharply enough to preserve or strengthen Experiment 17's negative gradient. Reduce singleton skepticism with a more nonlinear contrast-extremity focus or a lower singleton floor, while retaining near-full gain only at maximal normalized contrast so Experiment 8 weakens without sacrificing Experiment 10. Preserve the smooth validity continuum and endpoint treatment.

**Outcome of your advice:** iter 3 candidate loss=0.1461 -> iter 4 candidate loss=0.1663 -> the gate marked it REJECTED.

### Iteration 4
Verdict: regenerate
Interpretation: The current candidate is faithful to the prescribed coherence-gated, breadth-normalized skeptical-comparison family, but it is a substantial regression and was correctly rejected by the gate: loss rose from the accepted 0.1297 to 0.1663. The new margin-precision profile moved behavior in the wrong direction. Relative to iteration 2, TTB/minority choices collapse in Experiments 1 and 3 (0.4325 to 0.1921 and 0.3765 to 0.2685), while majority reliance rises in Experiment 4 (0.8769 to 0.9275). It also damages Experiments 7 and 11. Experiment 2 remains far below the observed tally-gradient evidence (32.0 versus 89.1), so the added precision profile neither restores the required margin differentiation nor controls confidence. The separate removal of each coalition's consensus scale produces only a slight improvement in Experiment 5, which remains severely wrong (-0.387 versus -0.038), while degrading several previously good fits. The revised upper-tail coherence gate does not solve the composition problem: Experiment 16 remains much too low (0.655 versus 0.882), and Experiment 17 still has only about two-thirds of the observed negative gradient (-0.208 versus -0.325). Smooth endpoint behavior remains a strength—especially Experiment 10—and Experiments 12, 13, 15, and 18 remain reasonably close, but these successes do not compensate for the broad regressions. Between-subject variance also remains generally too low in high-margin conditions.
Rationale: Build from the unchanged accepted iteration-2 base. My previous recommendation to introduce a U-shaped margin-dependent precision profile was rejected by the gate, so do not repeat it, and do not retain the current moderate-margin gain or per-coalition consensus division. Freeze the accepted base's tally curve and coherence transition for the next attempt rather than changing several interacting components again. Make one localized in-family adjustment: gate skeptical evidence downward for coherent, equal-sized bilateral coalitions with nonmaximal validity contrast, while smoothly restoring full skeptical gain for either genuinely dispersed coalitions or near-maximal normalized contrasts. This directly targets the excessive anti-validity response in Experiment 5 and should also weaken the excessive ordinary singleton reversal in Experiment 8, while an extremity exemption can preserve the accurate 100%-versus-50% reversal in Experiment 10. Implement this as a bounded multiplicative activation based on coalition coherence, size balance, and absolute normalized contrast—not as another breadth-dependent focus term or wholesale normalization change. After establishing that this isolated change improves loss, address the residual Experiment 2/3/4 margin profile separately through narrower parameter-range calibration or continuous subject-level correlation between tally sensitivity and noise, rather than another nonmonotonic margin-temperature function. Preserve the smooth validity continuum and exact option-swap symmetry.

**Outcome of your advice:** iter 4 candidate loss=0.1663 -> iter 5 candidate loss=0.1321 -> the gate marked it REJECTED.

### Iteration 5
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed coherence-gated, breadth-normalized skeptical-comparison family, and the localized activation gate produced some useful movements: Experiment 1 is now close to observed (0.391 versus 0.403), Experiment 7 improves to 0.146 versus 0.116, Experiment 14 is close, and the dispersion gradient in Experiment 17 strengthens from the accepted base's -0.184 to -0.248 versus -0.325. Smooth endpoint behavior also remains good in Experiments 10, 13, and 15. Nevertheless, the candidate was rejected because loss increased from the accepted 0.1297 to 0.1321. The proposed activation gate does not solve its primary target: Experiment 5 remains severely wrong and is slightly worse than the accepted base (-0.420 versus -0.407, observed -0.038). It also lowers Experiment 16 from 0.654 to 0.620, moving farther from 0.882. This combination indicates that the gate suppresses skeptical evidence in the moderate, balanced conflicts where a high baseline response is needed, while failing to suppress the breadth-induced response in Experiment 5. The likely reason is that Experiment 5's expanding coalitions acquire raw validity dispersion, causing coherence_balance to decline and switching off the very suppression intended to remove the breadth pattern. The tally profile also remains inadequate: Experiment 2 shows far too little margin-dependent evidence for tallying (14.3 versus 89.1), Experiment 3 has too few minority choices (0.366 versus 0.451), and Experiments 4 and 6 remain too tally-confident. Variance is still too low in several high-margin conditions, especially Experiments 4 and 6, while being excessive in Experiment 5.
Rationale: Because this candidate was rejected, build from the unchanged iteration-2 base. The previous localized activation-gate recommendation nearly recovered the best loss but was still rejected, so do not strengthen or repeat that gate. Instead, make one different in-family adjustment aimed at the common Experiment 5/16/17 diagnosis: replace raw within-coalition validity dispersion in the coherence gate with breadth- and composition-baseline-adjusted dispersion. Routine dispersion expected from adding adjacent experts as coalition breadth grows should be subtracted or ratio-normalized against a smooth expected-dispersion baseline, with coherence responding mainly to excess dispersion relative to that baseline. Keep the accepted representation, consensus, normalization, and tally equations otherwise fixed. This should prevent ordinary expansion in Experiment 5 from progressively disabling coherence or skeptical suppression, allow homogeneous and moderately dispersed coalitions in Experiment 16 to retain strong representations, and still let the maximally dispersed coalition in Experiment 17 collapse sharply. Use a smooth nonnegative residual or bounded ratio rather than dividing each coalition's whole representation by its consensus scale; that earlier per-coalition division was part of a rejected revision. Also freeze the tally curve for this iteration: wholesale Hill substitution and nonmonotonic margin-temperature changes were both rejected previously. Once the composition correction is tested in isolation, the residual Experiment 2/3/4/6 profile can be addressed through narrower parameter-range calibration or continuous correlation between tally sensitivity and noise, rather than another change in functional family. Preserve the smooth endpoint validity code and option-swap symmetry.

**Outcome of your advice:** iter 5 candidate loss=0.1321 -> iter 6 candidate loss=0.1524 -> the gate marked it REJECTED.

### Iteration 6
Verdict: regenerate
Interpretation: The candidate remains within the prescribed coherence-gated, breadth-normalized skeptical-comparison family, but the current revision is not an improvement: it was rejected with loss 0.1524 versus the accepted iteration-2 loss of 0.1297. My previous recommendation to baseline-adjust coalition dispersion was therefore unsuccessful and should not be repeated. The new roster-dependent correction damages several central fits. Experiment 1 moves from 0.4325 to 0.4925 against 0.4033, and Experiment 2 collapses from 32.3 to -19.4 against 89.1. Experiment 5 remains severely wrong (-0.437 versus -0.038), so the adjustment fails its main breadth-normalization target. It produces only a small improvement in Experiment 16 (0.665 versus 0.654, still far below 0.882) and does not strengthen the required Experiment 17 gradient enough (-0.203 versus -0.325). The likely problem is that expected dispersion depends on variance across the entire communicated roster and on n_features, including inactive experts. Consequently, an otherwise identical active coalition receives different coherence depending on unused roster composition, creating experiment-specific distortions rather than an invariant notion of excess heterogeneity. Other persistent failures remain: minority choices are too rare in Experiment 3, majority reliance is too high and insufficiently heterogeneous in Experiments 4 and 6, and skepticism is excessive in Experiment 8. Smooth endpoint behavior remains good in Experiments 10, 13, and 15, while Experiments 11, 12, 14, and 18 are reasonably close.
Rationale: Build from the unchanged accepted iteration-2 base. Do not retain or further tune the roster-variance dispersion baseline; the gate shows that this direction worsened the joint fit, and earlier activation gates, per-coalition consensus division, Hill tally replacement, and nonmonotonic margin temperatures were also rejected. For the next attempt, leave the accepted coherence, breadth normalization, focus rule, and endpoint code fixed and make an isolated calibration of the existing power tally channel. Increase the tally exponent modestly while lowering its gain so the curve is pivoted around a moderate-to-large margin: margin-one evidence should decrease, while choice probability should rise more distinctly across the margin levels. This can raise the Experiment 3 minority rate and restore the strong margin-dependent evidence in Experiment 2 without increasing the already excessive average confidence in Experiments 4 and 6. Add continuous correlation between tally gain and response noise or lapse—rather than a margin-specific temperature—so high-gain subjects are also less deterministic. That should increase the deficient between-subject variance in Experiments 4 and 6 and keep large-majority reliance substantial but below ceiling. Use narrower, jointly constrained ranges rather than another functional replacement. The unresolved Experiment 5 and 16–17 composition effects remain important, but repeated structural edits to coherence have regressed the aggregate fit; first establish whether this simpler tally-profile calibration can beat the accepted base while preserving its smooth endpoint and breadth-invariance successes.

**Outcome of your advice:** iter 6 candidate loss=0.1524 -> iter 7 candidate loss=0.1180 -> the gate marked it ACCEPTED.

### Iteration 7
Verdict: regenerate
Interpretation: Iteration 7 is a genuine improvement and validates the most recent advice: pivoting the existing power tally curve with a higher exponent, lower gain, and continuous tally–noise correlation reduced aggregate loss from 0.1297 to 0.1180. Experiment 1 is now excellent (0.415 versus 0.403), Experiment 3 improves modestly, and excessive majority confidence is reduced in Experiments 4 and 6. The candidate also preserves the prescribed family and its established successes: the endpoint reversal in Experiment 10, smooth 90%–95%–100% behavior in Experiment 13, the null categorical result in Experiment 15, negligible homogeneous breadth effect in Experiment 18, and the breadth gradient in Experiment 14. However, the theory is not yet experiment-invariant enough to ship. Experiment 5 remains a large structural failure (-0.410 versus -0.038), showing that equal-sized coalition expansion still changes the skeptical validity contrast substantially despite the claimed breadth cancellation. Experiment 16 remains much too weak (0.658 versus 0.882), and Experiment 17 captures only about 58% of the observed negative dispersion gradient (-0.188 versus -0.325). Thus the coherence mechanism does not create a sufficiently sharp relative-representation change between ordinary and maximally dispersed coalitions. Skepticism is simultaneously too strong in Experiment 8 (-4.801 versus -3.537). The tally profile also remains under-differentiated: Experiment 2 supplies much less evidence for margin-sensitive tallying than observed (37.8 versus 89.1), although large-margin reliance is still somewhat excessive in Experiments 4 and 6. Experiment 3 likewise retains too few minority choices (0.386 versus 0.451). Finally, high-margin between-subject variance remains far below the data in Experiments 2, 4, and 6, so the new tally–noise correlation helps point estimates more than heterogeneity.
Rationale: Build on this accepted iteration-7 candidate. The latest tally calibration was accepted, so it is reasonable to extend it cautiously rather than reverting to previously rejected Hill curves or margin-specific temperatures. A small further pivot—slightly higher tally exponent with correspondingly lower gain—may strengthen differentiation across the margins in Experiment 2 and reduce Experiment 7 and 11 conventional choices without restoring ceiling performance in Experiments 4 and 6. Widen the continuous tally-trait/noise correlation modestly to address their deficient subject variance, but keep this subject-level rather than margin-gated.

The more important next adjustment should be an isolated change to contextual normalization, not another roster-dependent dispersion baseline, activation gate, per-coalition consensus division, or shifted coherence threshold; those directions were rejected previously. Replace the denominator’s pooled active_mean term with a smooth symmetric normalization based on the represented evidence itself, such as a bounded function of |rep_a|+|rep_b| alongside the breadth floor. This remains within coherence-gated, breadth-normalized relative comparison. It should absorb composition-induced changes in coalition mean strength and coherence that currently produce the severe Experiment 5 breadth pattern. Because represented evidence falls when a coalition becomes genuinely dispersed, the same relative normalization can amplify the contrast between ordinary and maximally dispersed opponents, potentially strengthening Experiment 17 without globally increasing skeptical_strength. Preserve a nonzero floor so homogeneous and moderately dispersed coalitions retain enough absolute representation for Experiment 16, and preserve maximal relative contrast when one side is near 50% validity so Experiment 10 remains intact. Do not globally strengthen skepticism, which would worsen Experiment 8. The smooth validity code, endpoint treatment, option-swap symmetry, and current basic tally family should otherwise remain fixed.

**Outcome of your advice:** iter 7 candidate loss=0.1180 -> iter 8 candidate loss=0.1257 -> the gate marked it REJECTED.

### Iteration 8
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed coherence-gated, breadth-normalized skeptical-comparison family, but the represented-evidence normalization was not successful. The gate rejected this iteration: aggregate loss rose from 0.1180 to 0.1257. Its intended targets barely change or worsen: Experiment 5 remains severely wrong (-0.413 versus -0.038), Experiment 16 remains much too low (0.656 versus 0.882), and Experiment 17 remains far too shallow (-0.188 versus -0.325). Thus normalizing by |rep_a|+|rep_b| does not remove the composition-driven breadth pattern or amplify maximal dispersion as intended. It also creates collateral regressions: Experiment 2 falls from 37.8 to 17.4 versus 89.1, Experiment 6 rises from 0.888 to 0.900 versus 0.846, and Experiment 11 moves farther from its target. Improvements in Experiments 8, 10, 12, 14, and 18 are modest and do not compensate. Persistent cross-experiment failures remain too large: excessive coherent-conflict skepticism in Experiments 5 and 8, insufficient skeptical discrimination in Experiments 16-17, too few minority choices in Experiment 3, and insufficient between-subject variance in Experiments 2, 4, and 6. Smooth endpoint behavior and option-swap symmetry remain strengths.
Rationale: Build from the unchanged accepted iteration-7 base. My previous recommendation to replace active-mean normalization with represented-density normalization was rejected, so do not repeat, interpolate toward, or further tune that denominator. Freeze the iteration-7 coherence, focus, contextual normalization, validity, and endpoint equations for the next attempt.

Use the accepted tally-calibration direction in a more controlled, isolated way. Retain the same power tally family, but parameterize it around a fixed moderate-large pivot, approximately margin five: normalize the power curve by its value at that pivot and give the pivot amplitude its own narrow range. Then increase the exponent only slightly while holding evidence near margin five approximately constant. This should reduce evidence at margins one and three, increasing minority choices in Experiment 3 and lowering conventional choices in Experiments 7 and 11, while increasing separation between moderate and very large margins in Experiment 2. Holding margin-five evidence fixed should avoid worsening Experiments 4 and 6, whose diagnostic conflicts are near that margin. This is preferable to another Hill curve or margin-specific temperature, both previously rejected.

Address deficient subject variance through a centered continuous trait interaction rather than simply widening the current divisor `1 + correlation*tally_trait`, which can compress trait differences. Let tally gain and precision covary around their population means so high-gain subjects remain noisier without shifting the mean margin-five response or eliminating the tails of the subject distribution. Keep ranges narrow and test this tally-only revision before making another structural coherence change. The unresolved Experiments 5 and 16-17 remain important, but repeated normalization, activation, and dispersion-baseline edits have all been rejected; another bundled structural intervention would obscure whether the accepted tally improvement can be extended.

**Outcome of your advice:** iter 8 candidate loss=0.1257 -> iter 9 candidate loss=0.1333 -> the gate marked it REJECTED.

### Iteration 9 (most recent)
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed coherence-gated, breadth-normalized skeptical-comparison family, but this iteration is a clear regression and was rejected by the gate: loss increased from the accepted iteration-7 value of 0.1180 to 0.1333. The margin-five pivot and centered tally–precision interaction did not produce the intended margin profile. Relative to iteration 7, Experiment 2 remains far below the observed margin-sensitive tally evidence and worsens from 37.8 to 22.0 versus 89.1; Experiment 3 moves away from the required minority-choice rate, from 0.386 to 0.373 versus 0.451; and Experiment 7 deteriorates sharply from 0.188 to 0.203 versus 0.116. Experiments 14, 16, and 17 also regress. Although Experiments 4 and 6 improve slightly, both still show excessive majority reliance and much too little between-subject variance. The persistent structural failures are unchanged: Experiment 5 shows a large spurious breadth/composition effect (-0.408 versus -0.038), Experiment 16 has far too little preference for the homogeneous 75%-valid coalition (0.643 versus 0.882), and Experiment 17 produces only about half the observed negative dispersion gradient (-0.179 versus -0.325). Together, Experiments 16-17 indicate that the coherence gate does not adequately distinguish ordinary or moderate heterogeneity from maximal dispersion. Meanwhile, skepticism remains excessive in Experiment 8 (-4.759 versus -3.537). Smooth endpoint behavior remains successful in Experiments 10, 13, and 15, and homogeneous breadth dependence remains negligible in Experiment 18.
Rationale: Build from the unchanged accepted iteration-7 base. My previous recommendation to pivot the tally curve at margin five and replace the precision divisor with a centered covariance correction was rejected, so do not repeat or further tune that parameterization. Restore iteration 7's tally equation, tally–noise interaction, and parameter ranges, and freeze the tally, contextual-normalization, focus, validity, and endpoint components for the next test.

Make one isolated in-family revision to the coherence gate. The current gate applies a fixed raw-variance scale to assertiveness dispersion, so the same coherence_scale can classify adjacent-validity mixtures in Experiment 5, moderate dispersion in Experiment 16, and maximal dispersion in Experiment 17 inconsistently across validity rosters. Replace raw variance divided by a fixed scale with a roster-independent, locally normalized heterogeneity index—for example, variance divided by the maximum feasible variance conditional on the coalition's own mean assertiveness, with numerical floors and clipping. Then apply a high-threshold, steep upper-tail gate to that bounded index. Keep coherence near one throughout homogeneous and moderately heterogeneous coalitions, and make it collapse only near maximal normalized heterogeneity. This differs from the previously rejected roster-wide dispersion baseline: it must depend only on active members of the coalition, not unused experts or total feature count.

Such a scale-free upper-tail gate should preserve representation for the homogeneous and moderate conditions needed to raise Experiment 16, avoid treating routine adjacent-expert expansion as incoherence in Experiment 5, and selectively deepen the maximal-dispersion effect required by Experiment 17. Tune the threshold and steepness narrowly rather than globally increasing skeptical_strength or lowering the coherence threshold; those changes would worsen Experiment 8. Test this coherence-only change before revisiting tally heterogeneity.

**Outcome of your advice:** iter 9 candidate loss=0.1333 -> CURRENT candidate loss=0.1591 -> the gate marked it REJECTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate is faithful to the prescribed coherence-gated, breadth-normalized skeptical-comparison family, but the current coherence-only revision is a severe regression and was rejected by the gate: loss rose from the accepted iteration-7 value of 0.1180 to 0.1591. My previous recommendation to normalize variance by the maximum feasible variance conditional on coalition mean was therefore unsuccessful. Most importantly, the intended dispersion effect disappears in Experiment 17: the gradient changes from -0.188 in iteration 7 to 0.003, versus an observed -0.325. Experiment 16 also worsens to 0.636 versus 0.882. The normalization creates broad collateral damage: Experiment 2 collapses from 37.8 to -159.0 versus 89.1, Experiment 1 rises from 0.415 to 0.512 versus 0.403, and Experiments 9 and 11 move substantially away from their targets. Experiment 5 improves only slightly and remains structurally wrong (-0.361 versus -0.038). Improvements in Experiments 6 and 8 are useful but insufficient, while the established smooth endpoint fits in Experiments 10, 13, and 15 remain broadly intact. The likely problem is that variance divided by mean_strength*(1-mean_strength) changes scale strongly with coalition mean, while the imposed 0.50\u20130.68 upper-tail threshold evidently does not isolate the experimentally maximal dispersion. It therefore leaves the critical Experiment-17 coalition effectively coherent while perturbing other validity compositions enough to destroy the tally-margin pattern in Experiment 2.",
  "verdict": "regenerate",
  "rationale": "Build from the unchanged accepted iteration-7 base. Do not repeat or further tune the conditional-maximum-variance normalization; the gate decisively rejected my previous advice. Also freeze iteration 7's tally equation, tally\u2013noise interaction, contextual normalization, focus rule, smooth validity code, and endpoint treatment. The accepted tally calibration should not be disturbed after the rejected margin-five pivot and the present coherence-induced collapse in Experiment 2.\n\nTry a different, isolated in-family coherence statistic: an active-member-only upper-tail pairwise-dispersion index. For example, average a high power of pairwise assertiveness distances, mean(|s_i-s_j|^p) with p roughly 3\u20135, using a mean rather than a sum so coalition breadth does not intrinsically increase incoherence. Feed this bounded statistic through a smooth steep gate whose threshold lies between moderate and genuinely extreme pairwise separation. This differs from both the rejected roster-wide baseline and the rejected mean-conditional variance ratio. It should be exactly zero for homogeneous coalitions, remain very small when adjacent-validity experts are added in Experiment 5 or under moderate dispersion in Experiment 16, and respond strongly when a substantial fraction of coalition members are widely separated in Experiment 17. Keep coherence essentially one below the threshold and avoid coupling the threshold strongly to skeptical orientation until the pooled gradient is correct.\n\nThe immediate calibration criterion should be preservation of iteration-7 behavior on Experiments 1, 2, 9, and 11 while deepening Experiment 17 and raising Experiment 16. Do not globally increase skeptical strength, because Experiment 8 is already too negative, and do not bundle another normalization or tally revision with this test. Experiment 5 remains unresolved, but the present edit demonstrates that mean-conditional variance is not the route to fixing it."
}
```

## Usage

```json
{
  "input_tokens": 39364,
  "output_tokens": 2094
}
```
