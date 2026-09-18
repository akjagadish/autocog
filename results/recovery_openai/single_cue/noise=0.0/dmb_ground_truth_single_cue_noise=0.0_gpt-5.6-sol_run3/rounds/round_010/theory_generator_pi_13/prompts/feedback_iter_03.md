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
- THEORY 1 = `pi_11`
- THEORY 2 = `pi_12`
- The recommendation below acts on THEORY 1 (= `pi_11`).

Replace pi_11 with a new Conflict-Conditional Evidence-Construal theory. The central claim should be that people recruit among three psychologically distinct representations according to the geometry of the current conflict: (1) a raw-count representation that dominates when the normalized vote margin is decisive; (2) a coalition-claim representation that can trigger pragmatic reversal when tallies are tied or weak and one option is portrayed as more authoritatively supported; and (3) an exemplar or diagnostic-cue representation that preserves influential extreme experts inside heterogeneous coalitions rather than treating dispersion as global incoherence. Recruitment weights should vary smoothly with normalized tally margin, signed validity contrast, coalition entropy, and sparsity, without hand-coded windows for particular margins, validity bands, or endpoints. The skeptical coalition response should rise rapidly even for small communicated-validity advantages and then saturate, explaining Experiment 1 and the strong anti-validity rates in Experiments 9, 13, 14, and 18. Equal-sized homogeneous repetition should normally be redundant, explaining Experiments 2 and 20 and the near-null result in Experiment 7. Large tally margins should increasingly suppress pragmatic reversal, reproducing Experiments 4, 6, 8, and 22. Crucially, heterogeneous coalitions should be decomposable into salient experts or subcoalitions rather than discounted solely by variance; this can retain pi_12's success in Experiments 19 and 21 while addressing its major failure in Experiment 18. Stable subject-level propensities for counting, pragmatic skepticism, and exemplar attention should be correlated and hierarchically distributed, with trial-level stochastic recruitment, so the theory can reproduce both pooled means and the elevated variance in selected composition-sensitive experiments. Preserve history independence and exact option-swap symmetry, but do not preserve pi_11's narrow-conflict focus rule, additive fixed channels, or threshold-only coherence interpretation.


## CANDIDATE THEORY
Conflict-Geometry Recruitment theory proposes that people probabilistically recruit one of three representations according to current conflict geometry. Raw counting dominates increasingly decisive vote imbalances. Coalition-claim construal compresses homogeneous repetition into redundant option-level claims and produces a rapidly saturating skeptical response to communicated-authority contrasts. Exemplar construal preserves diagnostic experts within heterogeneous coalitions through sidewise upper-tail concentration, but is recruited chiefly when tally conflict remains unresolved. Its suppression grows convexly with tally decisiveness, allowing diagnostic cues to influence weak conflicts without overriding decisive majorities. Stable correlated subject propensities influence recruitment, while trial-level stochastic construal is marginalized into predicted choice probabilities. The process is history independent and exactly invariant under exchanging options A and B.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Conflict-Geometry Recruitment model. History is intentionally ignored:
    # choices are not followed by information that could update expert quality.
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

    # Positive votes favor A and negative votes favor B. Agreements are not
    # evidence for either displayed option.
    signed_votes = np.sign(stimulus[0] - stimulus[1])
    active = signed_votes != 0.0
    active_count = int(np.count_nonzero(active))
    tally_margin = float(np.sum(signed_votes))
    margin_magnitude = abs(tally_margin)

    # Smooth authority coding on the full communicated-validity continuum.
    validity = np.clip(validities, 0.5, 1.0)
    raw_authority = 2.0 * (validity - 0.5)
    curvature = float(parameters["validity_curvature"])
    authority = np.tanh(curvature * raw_authority) / max(
        float(np.tanh(curvature)), 1e-12
    )

    exemplar_temperature = float(parameters["exemplar_temperature"])

    def represent(side):
        members = authority[active & (signed_votes == side)]
        breadth = int(members.size)
        if breadth == 0:
            return 0.0, 0.0, 0.0, 0.0, 0

        # Coalition claims encode mean authority. Consequently, homogeneous
        # repetition is representationally redundant rather than cumulative.
        claim = float(np.mean(members))
        variance = float(np.mean((members - claim) ** 2))
        heterogeneity = float(np.clip(4.0 * variance, 0.0, 1.0))

        # A normalized soft maximum retains diagnostic members without using a
        # hard winner or an endpoint category. It equals the common authority
        # in a homogeneous coalition and approaches its upper tail when mixed.
        centered = exemplar_temperature * (members - np.max(members))
        weights = np.exp(centered)
        weights /= np.sum(weights)
        exemplar = float(np.dot(weights, members))

        # Upper-tail concentration is computed separately within each side, so
        # diagnostic substructure cannot cancel across opposing coalitions.
        upper_tail = max(exemplar - claim, 0.0)
        return claim, exemplar, heterogeneity, upper_tail, breadth

    claim_a, exemplar_a, hetero_a, tail_a, breadth_a = represent(1.0)
    claim_b, exemplar_b, hetero_b, tail_b, breadth_b = represent(-1.0)

    if active_count == 0:
        claim_contrast = 0.0
        exemplar_contrast = 0.0
        pooled_heterogeneity = 0.0
        pooled_upper_tail = 0.0
        sparsity = 1.0
    else:
        claim_contrast = (claim_a - claim_b) / (
            float(parameters["claim_floor"]) + abs(claim_a) + abs(claim_b)
        )
        claim_contrast = float(np.clip(claim_contrast, -1.0, 1.0))

        exemplar_contrast = (exemplar_a - exemplar_b) / (
            float(parameters["exemplar_floor"])
            + abs(exemplar_a)
            + abs(exemplar_b)
        )
        exemplar_contrast = float(np.clip(exemplar_contrast, -1.0, 1.0))

        pooled_heterogeneity = (
            breadth_a * hetero_a + breadth_b * hetero_b
        ) / float(active_count)
        pooled_upper_tail = (
            breadth_a * tail_a + breadth_b * tail_b
        ) / float(active_count)
        sparsity = 1.0 - float(active_count) / float(max(n_features, 1))

    # Decisiveness requires both a large proportional imbalance and enough
    # absolute votes. This distinguishes a sparse one-vote conflict from a
    # substantial majority without introducing windows for particular margins.
    if active_count == 0:
        normalized_margin = 0.0
    else:
        normalized_margin = margin_magnitude / float(active_count)
    size_certainty = 1.0 - np.exp(
        -margin_magnitude / float(parameters["margin_scale"])
    )
    decisiveness = float(normalized_margin * size_certainty)

    # Signed validity geometry relative to the tally is invariant to swapping
    # A and B: both factors reverse under a swap. Positive values mean that
    # authority and the raw majority point in the same direction.
    if margin_magnitude < 1e-12:
        validity_tally_alignment = 0.0
    else:
        validity_tally_alignment = float(
            np.sign(tally_margin) * claim_contrast
        )

    diagnostic_concentration = abs(exemplar_contrast - claim_contrast)
    diagnostic_scale = float(parameters["diagnostic_scale"])
    diagnostic_extremity = (
        diagnostic_concentration ** 2
        / (diagnostic_scale ** 2 + diagnostic_concentration ** 2 + 1e-12)
    )

    # Sidewise upper-tail concentration only exists when a coalition contains
    # diagnostic internal structure. Gating by heterogeneity prevents a merely
    # homogeneous authority contrast from duplicating coalition-claim evidence.
    upper_tail_salience = (
        pooled_upper_tail ** 2
        / (diagnostic_scale ** 2 + pooled_upper_tail ** 2 + 1e-12)
    ) * np.sqrt(max(pooled_heterogeneity, 0.0))

    # Correlated hierarchical propensities. One shared style variable loads in
    # opposite directions on counting versus interpretive construals, while
    # representation-specific deviations retain individual flexibility.
    shared_style = float(parameters["shared_style"]) - 0.5
    loading = float(parameters["trait_loading"])
    count_trait = (
        -loading * shared_style + float(parameters["count_offset"])
    )
    skeptic_trait = (
        loading * shared_style + float(parameters["skeptic_offset"])
    )
    exemplar_trait = (
        0.55 * loading * shared_style + float(parameters["exemplar_offset"])
    )

    # Claim recruitment responds rapidly to authority contrast independently
    # of coalition breadth. Only an additional, smaller smooth term depends on
    # sparsity, and it is strongest for highly authoritative claims.
    absolute_claim = abs(claim_contrast)
    saturated_claim = 1.0 - np.exp(
        -float(parameters["claim_saturation"]) * absolute_claim
    )
    authority_extremity = max(abs(claim_a), abs(claim_b)) ** 4

    # Smooth recruitment utilities. Convex decisiveness suppression leaves
    # weak conflicts accessible to exemplars while strongly protecting large
    # majorities. Upper-tail recruitment is restricted to unresolved conflict.
    count_utility = (
        float(parameters["count_bias"])
        + count_trait
        + float(parameters["margin_recruitment"]) * decisiveness
        + float(parameters["alignment_recruitment"])
        * validity_tally_alignment
    )
    claim_utility = (
        float(parameters["claim_bias"])
        + skeptic_trait
        + float(parameters["claim_salience"]) * saturated_claim
        + float(parameters["sparsity_claim_gain"])
        * sparsity * authority_extremity
        + float(parameters["margin_recruitment"]) * (1.0 - decisiveness)
        - float(parameters["heterogeneity_escape"])
        * pooled_heterogeneity
    )
    exemplar_utility = (
        float(parameters["exemplar_bias"])
        + exemplar_trait
        + float(parameters["exemplar_heterogeneity"])
        * pooled_heterogeneity
        + (1.0 - decisiveness)
        * (
            float(parameters["exemplar_diagnostic"])
            * diagnostic_extremity
            + float(parameters["exemplar_extremity_gain"])
            * upper_tail_salience
        )
        - float(parameters["exemplar_margin_suppression"])
        * decisiveness ** 2
    )

    utilities = np.asarray(
        [count_utility, claim_utility, exemplar_utility], dtype=np.float64
    )
    utilities /= float(parameters["recruitment_temperature"])
    utilities -= np.max(utilities)
    recruitment = np.exp(utilities)
    recruitment /= np.sum(recruitment)

    # Each representation supplies a complete conditional choice distribution.
    # The count channel grows with margin, the coalition channel reverses a
    # rapidly saturating authority contrast, and the exemplar channel follows
    # the diagnostic upper-tail contrast.
    if margin_magnitude < 1e-12:
        count_evidence = 0.0
    else:
        count_evidence = (
            float(parameters["count_strength"])
            * np.sign(tally_margin)
            * margin_magnitude ** float(parameters["count_power"])
        )

    skeptical_multiplier = max(0.20, 1.0 + 0.55 * skeptic_trait)
    claim_evidence = -float(parameters["skeptic_strength"]) * (
        skeptical_multiplier
        * np.tanh(float(parameters["claim_slope"]) * claim_contrast)
    )

    exemplar_multiplier = max(0.20, 1.0 + 0.45 * exemplar_trait)
    exemplar_evidence = float(parameters["exemplar_strength"]) * (
        exemplar_multiplier
        * np.tanh(
            float(parameters["exemplar_slope"]) * exemplar_contrast
        )
    )

    precision = float(parameters["choice_precision"])
    evidences = precision * np.asarray(
        [count_evidence, claim_evidence, exemplar_evidence],
        dtype=np.float64,
    )

    # Marginalize stochastic representation recruitment. Computing a mixture
    # of channel-specific choices, rather than averaging evidence, preserves
    # psychologically distinct construals and their trial-level competition.
    channel_p_a = np.empty(3, dtype=np.float64)
    for i, evidence in enumerate(evidences):
        channel_logits = np.asarray(
            [0.5 * evidence, -0.5 * evidence], dtype=np.float64
        )
        channel_logits -= np.max(channel_logits)
        channel_probs = np.exp(channel_logits)
        channel_probs /= np.sum(channel_probs)
        channel_p_a[i] = channel_probs[0]

    p_a = float(np.dot(recruitment, channel_p_a))
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
- validity_curvature: [1.2, 2.4]
- claim_floor: [0.055, 0.14]
- claim_slope: [2.2, 4.0]
- claim_saturation: [3.5, 6.0]
- exemplar_temperature: [7.0, 12.0]
- exemplar_floor: [0.08, 0.18]
- exemplar_slope: [2.0, 3.6]
- diagnostic_scale: [0.12, 0.22]
- margin_scale: [1.8, 3.2]
- count_power: [0.42, 0.60]
- count_strength: [0.52, 0.72]
- skeptic_strength: [2.5, 3.8]
- exemplar_strength: [5.0, 8.0]
- shared_style: [0, 1]
- count_offset: [-0.55, 0.55]
- skeptic_offset: [-0.55, 0.55]
- exemplar_offset: [-0.25, 0.25]
- trait_loading: [0.75, 1.30]
- count_bias: [-0.15, 0.30]
- claim_bias: [0.05, 0.45]
- exemplar_bias: [-1.25, -0.65]
- margin_recruitment: [2.2, 3.5]
- alignment_recruitment: [-0.35, 0.35]
- claim_salience: [0.30, 0.65]
- sparsity_claim_gain: [0.02, 0.15]
- heterogeneity_escape: [1.4, 2.8]
- exemplar_heterogeneity: [1.2, 2.5]
- exemplar_diagnostic: [3.5, 5.5]
- exemplar_extremity_gain: [1.4, 2.8]
- exemplar_margin_suppression: [10.0, 15.0]
- recruitment_temperature: [0.65, 1.15]
- choice_precision: [0.90, 1.20]
- lapse_rate: [0.0, 0.055]
- validities: validities

`rationale`:
This is a minimal gating and calibration edit to the accepted model. The linear exemplar penalty is replaced by a convex decisiveness-squared penalty, with its coefficient increased so suppression remains strong for genuinely decisive majorities while becoming much weaker on one-vote conflicts. Count evidence is flattened and modestly weakened, and decisive count recruitment is partially backed off, targeting the excessive tally differentiation in Experiment 2 and the majority overshoots in Experiments 4, 6, and 20 while restoring diagnostic choices in Experiments 1 and 3. Generic sparse claim amplification is sharply narrowed to reduce the erroneous homogeneous breadth effects in Experiments 5, 8, and 14. Finally, the homogeneous absolute-exemplar bonus is replaced by a sidewise upper-tail-minus-mean statistic gated by coalition heterogeneity and unresolved conflict. This prevents exemplar processing from duplicating claim processing in homogeneous ties, which should restore skeptical responding in Experiments 16 and 21, while retaining non-cancelling diagnostic substructure for Experiments 17 and 19. All edits preserve three-channel stochastic recruitment, hierarchical subject propensities, history independence, and exact option-swap symmetry.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.1981 -> ACCEPTED
- iter 2: loss=0.1738 -> ACCEPTED
- iter 3: loss=0.1712 -> ACCEPTED
- iter 4 (current candidate you are grading): loss=0.1437 -> ACCEPTED
Running-best (last accepted) base: iter 4 at loss=0.1437.

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
  - iter 1: 0.2608 (var=0.0035) (Δ vs real -0.1425)
  - iter 2: 0.3304 (var=0.0074) (Δ vs real -0.0729)
  - iter 3: 0.2117 (var=0.0052) (Δ vs real -0.1917)
  - iter 4 (current): 0.2546 (var=0.0071) (Δ vs real -0.1487)
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
  - iter 1: 169.4025 (var=2.6558) (Δ vs real +80.2603)
  - iter 2: 129.7761 (var=3.8907) (Δ vs real +40.6338)
  - iter 3: 225.7233 (var=3.6807) (Δ vs real +136.5810)
  - iter 4 (current): 108.2181 (var=3.0251) (Δ vs real +19.0759)
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
  - iter 1: 0.2275 (var=0.0035) (Δ vs real -0.2233)
  - iter 2: 0.2323 (var=0.0039) (Δ vs real -0.2185)
  - iter 3: 0.2035 (var=0.0061) (Δ vs real -0.2473)
  - iter 4 (current): 0.2373 (var=0.0041) (Δ vs real -0.2135)
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
  - iter 1: 0.8375 (var=0.0053) (Δ vs real +0.0162)
  - iter 2: 0.3069 (var=0.0323) (Δ vs real -0.5144)
  - iter 3: 0.8906 (var=0.0050) (Δ vs real +0.0694)
  - iter 4 (current): 0.8313 (var=0.0056) (Δ vs real +0.0100)
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
  - iter 1: -0.6201 (var=0.0501) (Δ vs real -0.5819)
  - iter 2: -0.5422 (var=0.0622) (Δ vs real -0.5041)
  - iter 3: -0.5494 (var=0.0545) (Δ vs real -0.5112)
  - iter 4 (current): -0.5385 (var=0.0551) (Δ vs real -0.5003)
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
  - iter 1: 0.7721 (var=0.0055) (Δ vs real -0.0742)
  - iter 2: 0.7679 (var=0.0053) (Δ vs real -0.0783)
  - iter 3: 0.9019 (var=0.0019) (Δ vs real +0.0556)
  - iter 4 (current): 0.8767 (var=0.0027) (Δ vs real +0.0304)
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
  - iter 1: 0.0774 (var=0.0021) (Δ vs real -0.0387)
  - iter 2: 0.0894 (var=0.0023) (Δ vs real -0.0267)
  - iter 3: 0.1075 (var=0.0039) (Δ vs real -0.0086)
  - iter 4 (current): 0.0942 (var=0.0057) (Δ vs real -0.0220)
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
  - iter 1: -6.0028 (var=0.0088) (Δ vs real -2.4656)
  - iter 2: -5.9765 (var=0.0113) (Δ vs real -2.4393)
  - iter 3: -5.4364 (var=0.0249) (Δ vs real -1.8992)
  - iter 4 (current): -5.8258 (var=0.0108) (Δ vs real -2.2886)
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
  - iter 1: 0.2132 (var=0.0020) (Δ vs real -0.1117)
  - iter 2: 0.2413 (var=0.0022) (Δ vs real -0.0835)
  - iter 3: 0.2513 (var=0.0039) (Δ vs real -0.0736)
  - iter 4 (current): 0.2588 (var=0.0029) (Δ vs real -0.0661)
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
  - iter 1: 0.9208 (var=0.0047) (Δ vs real +0.0592)
  - iter 2: 0.9158 (var=0.0051) (Δ vs real +0.0542)
  - iter 3: 0.8717 (var=0.0105) (Δ vs real +0.0100)
  - iter 4 (current): 0.9158 (var=0.0050) (Δ vs real +0.0542)
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
  - iter 1: 0.0806 (var=0.0021) (Δ vs real -0.0767)
  - iter 2: 0.0867 (var=0.0031) (Δ vs real -0.0706)
  - iter 3: 0.1331 (var=0.0089) (Δ vs real -0.0242)
  - iter 4 (current): 0.0925 (var=0.0033) (Δ vs real -0.0647)
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
  - iter 1: 0.0764 (var=0.0022) (Δ vs real -0.0497)
  - iter 2: 0.0958 (var=0.0028) (Δ vs real -0.0303)
  - iter 3: 0.1544 (var=0.0087) (Δ vs real +0.0283)
  - iter 4 (current): 0.0831 (var=0.0031) (Δ vs real -0.0431)
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
  - iter 1: 0.0222 (var=0.0132) (Δ vs real +0.0128)
  - iter 2: 0.0228 (var=0.0127) (Δ vs real +0.0134)
  - iter 3: 0.0216 (var=0.0095) (Δ vs real +0.0122)
  - iter 4 (current): 0.0187 (var=0.0094) (Δ vs real +0.0094)
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
  - iter 1: -0.0149 (var=0.0052) (Δ vs real -0.1929)
  - iter 2: 0.3019 (var=0.0319) (Δ vs real +0.1239)
  - iter 3: 0.4510 (var=0.0242) (Δ vs real +0.2729)
  - iter 4 (current): 0.4165 (var=0.0366) (Δ vs real +0.2385)
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
  - iter 1: 0.0000 (var=0.0736) (Δ vs real +0.0000)
  - iter 2: 0.0000 (var=0.1476) (Δ vs real +0.0000)
  - iter 3: 0.0000 (var=0.1600) (Δ vs real +0.0000)
  - iter 4 (current): 0.0000 (var=0.2176) (Δ vs real +0.0000)
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
  - iter 1: 0.6684 (var=0.0056) (Δ vs real -0.2134)
  - iter 2: 0.6897 (var=0.0062) (Δ vs real -0.1922)
  - iter 3: 0.6884 (var=0.0077) (Δ vs real -0.1934)
  - iter 4 (current): 0.6656 (var=0.0102) (Δ vs real -0.2162)
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
  - iter 1: -0.0819 (var=0.0112) (Δ vs real +0.2431)
  - iter 2: -0.0600 (var=0.0157) (Δ vs real +0.2650)
  - iter 3: -0.0731 (var=0.0126) (Δ vs real +0.2519)
  - iter 4 (current): -0.0919 (var=0.0156) (Δ vs real +0.2331)
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
  - iter 1: -0.0117 (var=0.0039) (Δ vs real -0.0175)
  - iter 2: 0.0046 (var=0.0046) (Δ vs real -0.0012)
  - iter 3: 0.0071 (var=0.0074) (Δ vs real +0.0013)
  - iter 4 (current): -0.0112 (var=0.0053) (Δ vs real -0.0171)
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
  - iter 1: 0.5383 (var=0.0318) (Δ vs real -0.3233)
  - iter 2: 0.6833 (var=0.0326) (Δ vs real -0.1783)
  - iter 3: 0.8900 (var=0.0125) (Δ vs real +0.0283)
  - iter 4 (current): 0.8875 (var=0.0134) (Δ vs real +0.0258)
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
  - iter 1: 0.8658 (var=0.0042) (Δ vs real +0.0092)
  - iter 2: 0.8800 (var=0.0036) (Δ vs real +0.0233)
  - iter 3: 0.8925 (var=0.0030) (Δ vs real +0.0358)
  - iter 4 (current): 0.7825 (var=0.0068) (Δ vs real -0.0742)
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
  - iter 1: 0.8970 (var=0.0023) (Δ vs real +0.0303)
  - iter 2: 0.8537 (var=0.0117) (Δ vs real -0.0130)
  - iter 3: 0.7853 (var=0.0247) (Δ vs real -0.0813)
  - iter 4 (current): 0.8503 (var=0.0096) (Δ vs real -0.0163)
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
  - iter 1: 0.0029 (var=0.0032) (Δ vs real +0.0088)
  - iter 2: -0.0146 (var=0.0038) (Δ vs real -0.0087)
  - iter 3: -0.0592 (var=0.0080) (Δ vs real -0.0533)
  - iter 4 (current): -0.0400 (var=0.0078) (Δ vs real -0.0342)
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

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed three-representation family and improves several central invariances: it closely fits large-margin counting in Experiments 4 and 20, the near-null breadth effects in Experiments 18 and 22, the endpoint nonlinearity diagnostic in Experiment 13, and the strong skeptical choice in Experiment 21. However, the aggregate fit is not yet experiment-invariant. Skeptical claim construal is recruited too broadly or too deterministically: TTB/high-validity choices are too low in Experiments 1 (0.261 vs 0.403), 7, 9, 11, and 12; anti-validity alignment is too extreme in Experiment 8 (-6.003 vs -3.537) and Experiment 10 (0.921 vs 0.862). The very large miss in Experiment 5 (-0.620 vs -0.038) shows that the unconditional sparsity term creates an erroneous coalition-size dependence where equal-sized repetition should be largely redundant. At the same time, the exemplar mechanism does not actually deliver the recommended heterogeneous-coalition behavior: Experiment 19 is near chance (0.538 vs 0.862), and the dispersion gradient in Experiment 17 is much too weak (-0.082 vs -0.325). Experiment 16 is also low (0.668 vs 0.882). Thus the model currently overuses skepticism in ordinary homogeneous conflicts while under-recruiting diagnostic exemplars in the specifically heterogeneous conflicts where they are needed. Experiment 14 is a separate structural failure: its breadth contrast is essentially absent and slightly reversed (-0.015 vs 0.178). Finally, margin behavior is too steep in Experiment 2 (log BF 169 vs 89) even though majority choice is somewhat too weak in Experiment 6 (0.772 vs 0.846), indicating that count-channel margin sensitivity and level are confounded. Between-subject variance is also generally too low, notably in Experiments 2, 4, 6-8, 10-12, 14, and 17.
Rationale: Retain this accepted candidate as the in-family base, but recalibrate recruitment geometry rather than globally weakening or strengthening skepticism. First, remove or sharply reduce the unconditional positive exemplar_sparsity effect, which violates homogeneous repetition redundancy and appears responsible for the severe Experiment 5 size pattern. If sparsity is retained, interact it smoothly with absolute claim contrast or diagnostic concentration so that sparse, highly authoritative conflicts can differ without imposing a generic breadth gradient. A smooth sparsity-by-claim-contrast term in claim recruitment could recover the positive Experiment 14 attenuation while leaving moderate homogeneous conflicts closer to redundant. Second, make exemplar recruitment much more selective and effective at extreme diagnostic composition: increase or sharpen exemplar_temperature and exemplar_diagnostic, and/or use a smooth saturating nonlinear transform of diagnostic concentration. This should raise Experiment 19 and strengthen Experiment 17 without globally boosting exemplars in the moderate conditions of Experiment 16. Third, reduce the count channel's margin exponent while modestly increasing its baseline strength or recruitment level; this can flatten the excessive margin differentiation in Experiment 2 while raising majority choice in Experiment 6 and preserving Experiments 4 and 20. Do not simply reduce skeptic_strength globally, because strong skepticism is correctly needed in Experiments 21 and parts of 16; instead lower claim recruitment for ordinary homogeneous conflicts and reserve it for weak, high-contrast coalition claims. Finally, widen subject-level recruitment offsets or add a subject-specific recruitment temperature/strength scale to increase stable between-subject variance, while keeping exact swap symmetry and history independence.

**Outcome of your advice:** iter 1 candidate loss=0.1981 -> iter 2 candidate loss=0.1738 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The accepted revision improves aggregate loss and several central targets: Experiment 1 moves toward the observed TTB rate, Experiment 2's excessive margin differentiation is reduced, Experiment 18 is essentially exact, Experiment 21 remains strong, and Experiment 19 improves substantially. These gains support retaining the prescribed conflict-conditional three-representation family. Nevertheless, the model is not yet experiment-invariant. The most serious new failure is Experiment 4: the 6-versus-1 tally choice collapses from 0.838 to 0.307 against an observed 0.821. The selective-exemplar revision is therefore being recruited even when a large tally margin should suppress diagnostic-cue control. Experiment 19 also remains low (0.683 versus 0.862), while Experiment 17's dispersion gradient became slightly worse and remains far too weak (-0.060 versus -0.325). Thus exemplar recruitment is simultaneously too strong in decisive asymmetric conflicts and insufficiently sensitive to the relevant composition geometry in tied conflicts. The severe Experiment 5 miss persists (-0.542 versus -0.038), indicating that the sparsity-by-claim interaction still creates an unwanted coalition-breadth pattern among equal-sized homogeneous coalitions. Skepticism also remains too deterministic in Experiment 8 (-5.977 versus -3.537), whereas minority or high-validity choices remain too rare in Experiments 3, 9, 11, and 16. Experiment 14 now overshoots its breadth contrast (0.302 versus 0.178) and has excessive variance. Count calibration improved but remains imperfect: Experiment 2 is still too tally-like, while Experiment 6 remains below the observed majority rate. Variance is no longer uniformly deficient; it is now excessive in exemplar-sensitive Experiments 4, 14, and 19 but still too low in several ordinary counting and claim conditions.
Rationale: Build on this accepted candidate, preserving the three representations, smooth recruitment, swap symmetry, and history independence. The previously accepted push toward selective exemplars helped Experiment 19 and should not be reversed globally, but its gating must be corrected. Add a strong smooth decisiveness penalty to exemplar utility, such as a negative coefficient on decisiveness or an interaction with 1-decisisiveness, so large tally margins suppress exemplar recruitment; this directly targets the catastrophic Experiment 4 regression and should also raise Experiment 6 without weakening tied-conflict exemplar effects. For Experiments 17 and 19, do not merely increase exemplar strength again: that previous change failed to improve Experiment 17. Instead add a tied/weak-margin diagnostic-cue recruitment term based on a convex or saturating transform of absolute exemplar contrast or sidewise upper-tail extremity. The current diagnostic_concentration only measures exemplar-minus-mean discrepancy and can be zero for homogeneous but extremely diagnostic coalitions, making it unable to express Experiment 17's gradient.

Rework claim recruitment to separate authority sensitivity from breadth sensitivity. Use a rapidly saturating absolute claim-contrast term that is independent of coalition breadth, preserving strong responses to small validity advantages in Experiment 21, plus only a smaller sparsity modulation gated smoothly by very high authority extremity. The present linear abs(claim_contrast) * sparsity term is still producing the erroneous Experiment 5 size pattern and the excessive Experiment 14 gradient. This change should make ordinary homogeneous repetition redundant while retaining limited breadth attenuation for extreme authority conflicts. Avoid globally reducing skeptic strength, because Experiments 21 and 18 are well fit; reduce its recruitment specifically in lower-dimensional ordinary conflicts to soften Experiment 8.

Continue the successful count-channel separation only cautiously: a slightly lower count power paired with slightly higher count strength or decisive-margin recruitment may further reduce Experiment 2's Bayes factor while raising Experiment 6, but the urgent large-margin failure is exemplar gating rather than inadequate count evidence. Finally, do not widen heterogeneity globally again. Reduce exemplar-specific trait-offset/loading ranges, which are producing excessive variance in Experiments 4, 14, and 19, while allowing somewhat more count- or claim-specific subject variation where empirical variance remains underestimated.

**Outcome of your advice:** iter 2 candidate loss=0.1738 -> iter 3 candidate loss=0.1712 -> the gate marked it ACCEPTED.

### Iteration 3 (most recent)
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed conflict-conditional three-representation family, and the last accepted revision successfully repaired its most acute exemplar-gating failure. Experiment 4 recovered from 0.307 to 0.891 against 0.821, Experiment 19 improved from 0.683 to 0.890 against 0.862, and Experiment 18 remains essentially exact. Experiments 7, 10, 12, 13, 15, and 20 are also reasonably captured. Thus the accepted advice to suppress exemplars under decisive majorities while retaining them in unresolved heterogeneous conflict was directionally correct. However, the small aggregate-loss improvement masks severe regressions elsewhere. Weak and moderate conflicts now favor the tally much too strongly: Experiment 1 falls to 0.212 versus 0.403, Experiment 3 to 0.204 versus 0.451, and Experiment 2's tally-over-TTB evidence rises dramatically to 225.7 versus 89.1. Large-majority choice also overshoots in Experiments 6 and 20. The model therefore suppresses diagnostic-cue control too early and makes the count channel excessively margin-sensitive. Claim recruitment remains miscalibrated across coalition breadth: Experiment 5 is still badly wrong (-0.549 versus -0.038), Experiment 8 remains too skeptically polarized (-5.436 versus -3.537), and Experiment 14 now greatly overshoots the breadth gradient (0.451 versus 0.178). Meanwhile, Experiment 21 has deteriorated from near target to 0.785 versus 0.867, indicating excessive conventional exemplar competition in tied homogeneous conflicts. The dispersion gradient in Experiment 17 remains far too weak (-0.073 versus -0.325), and Experiment 16 remains low (0.688 versus 0.882). The model is consequently not experiment-invariant despite its improved aggregate loss.
Rationale: Build on the accepted candidate, but make the decisiveness gate nonlinear rather than reversing the successful large-margin suppression. The current linear term `-exemplar_margin_suppression * decisiveness` penalizes exemplar recruitment substantially even for weak one-vote conflicts. Replace it with a smooth convex penalty such as a coefficient on `decisiveness**2` (or another threshold-free power greater than one), recalibrated to remain strong near decisive margins. This should restore diagnostic-cue choices in Experiments 1 and 3 while preserving the repairs in Experiments 4 and 19. Separately, partially back off the latest increase in count dominance: use count strength and decisive recruitment between the iteration-2 and iteration-3 settings, and further flatten conditional count evidence with a modestly lower count power if needed. The goal is to reduce Experiment 2's excessive margin differentiation without undoing adequate majority choice in Experiments 4, 6, and 20.

The previous reductions to generic sparsity modulation were accepted overall but did not repair Experiment 5 and now overproduce Experiment 14. Do not repeat another broad change to skeptic strength. Instead, sharply narrow `sparsity_claim_gain` and remove the ungated homogeneous-coalition contribution of `diagnostic_cue_salience`. That absolute-exemplar term recruits a conventional, authority-following exemplar even when exemplar and claim are merely duplicate homogeneous summaries, plausibly explaining the deterioration in Experiment 21 and the unstable breadth effects in Experiments 5 and 14. To retain the successful heterogeneous behavior, replace that term with a sidewise upper-tail concentration statistic—such as a smooth breadth-weighted upper-tail-minus-mean quantity for each coalition—gated by pooled heterogeneity and unresolved conflict. Unlike the current contrast-of-contrasts, this statistic should not cancel when both coalitions contain diagnostic substructure. It can strengthen the missing Experiment 17 dispersion gradient and preserve Experiment 19 without activating exemplars in ordinary homogeneous ties.

Finally, reducing sparse claim amplification should soften the excessive skepticism in Experiment 8 while leaving broad tied claim recruitment available for Experiment 16. Preserve exact option-swap symmetry, history independence, hierarchical subject propensities, and the three-channel stochastic-recruitment architecture. Variance tuning should be secondary until these pooled directional failures are corrected; current variance is already close in Experiments 10, 12, 13, 18, and 19 but remains too low in several count conditions and Experiment 17.

**Outcome of your advice:** iter 3 candidate loss=0.1712 -> CURRENT candidate loss=0.1437 -> the gate marked it ACCEPTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The current candidate is faithful to the prescribed conflict-conditional three-representation family, and the latest accepted revision materially improves aggregate fit. In particular, the convex exemplar gate preserves the repaired decisive-majority result in Experiment 4 (0.831 versus 0.821), keeps the extreme heterogeneous-coalition result in Experiment 19 close (0.888 versus 0.862), and improves Experiment 2 substantially (108.2 versus 89.1). Experiments 6, 7, 13, 15, 18, and 21 are also reasonably captured. Thus the previously accepted advice\u2014convex margin gating, weaker count differentiation, and a heterogeneity-gated upper-tail statistic\u2014was productive and should not be reversed.\n\nNevertheless, the theory is not yet experiment-invariant. Diagnostic or minority choices remain much too rare in Experiments 1 (0.255 versus 0.403) and 3 (0.237 versus 0.451). The largest structural miss remains Experiment 5 (-0.539 versus -0.038), while Experiment 14's breadth contrast is much too large (0.417 versus 0.178). These paired failures indicate that coalition authority is still excessively altered by adding weaker same-side members. The arithmetic mean used for `claim` makes homogeneous or nested repetition dilute an option-level claim, despite the theory's stated redundancy principle. Merely narrowing `sparsity_claim_gain` could not repair this because the unwanted breadth dependence is built into the claim representation itself.\n\nThe candidate also remains too skeptically polarized in Experiment 8 (-5.826 versus -3.537) and at the concentrated endpoint in Experiment 10 (0.916 versus 0.862), yet it is insufficiently skeptical in the broad tied coalitions of Experiment 16 (0.666 versus 0.882). Experiment 17's composition gradient is still far too weak (-0.092 versus -0.325). The present exemplar recruitment statistics vanish for internally homogeneous coalitions, so the model cannot strongly recruit diagnostic-cue processing when a homogeneous opposing coalition becomes extremely authoritative across conditions. Finally, Experiment 20 has regressed to too little majority choice (0.783 versus 0.857): pragmatic claim reversal is not explicitly suppressed enough at near-maximal decisiveness. Between-subject variance also remains too low in several count and composition-sensitive experiments, although variance calibration is secondary to these pooled directional errors.",
  "verdict": "regenerate",
  "rationale": "Retain the accepted three-channel architecture and the successful convex exemplar suppression, but revise three in-family components.\n\nFirst, fix coalition-claim normalization rather than making another broad sparsity or skeptic-strength adjustment. Replace the arithmetic coalition mean with a smooth top-weighted or saturating option-level authority summary\u2014for example, a normalized log-sum-exp/softmax mean with only a small residual dilution component. The representation should approach the most authoritative member while remaining continuous and symmetric. This should make nested same-side repetition substantially more redundant, directly reducing the severe Experiment 5 effect and the oversized Experiment 14 breadth gradient. Do not globally weaken skepticism, because Experiment 21 is close and Experiment 16 requires more rather than less pragmatic reversal.\n\nSecond, add back only a narrowly targeted diagnostic-extremity contribution to exemplar recruitment. The prior removal of a generic homogeneous absolute-exemplar bonus improved aggregate loss, so do not restore that broad term. Instead use a steep but smooth interaction between absolute authority contrast and endpoint/extreme-authority salience, gated by unresolved tally conflict. It should be negligible for the small validity contrasts in Experiment 21 but increase for highly diagnostic sparse conflicts and the extreme conditions of Experiment 17. This can raise diagnostic/minority choices in Experiments 1 and 3, soften excessive anti-validity polarization in Experiments 8 and 10, and strengthen Experiment 17 without sacrificing the already good heterogeneous result in Experiment 19.\n\nThird, add an explicit convex decisiveness penalty to coalition-claim recruitment, not just an indirect loss of the `(1-decisiveness)` bonus. A high-power term such as `-claim_margin_suppression * decisiveness**q` with `q > 2` can suppress pragmatic reversal only near maximal tally decisiveness. This specifically targets Experiment 20 while leaving weaker 6-versus-1 conflicts such as Experiment 4 and tied claim effects such as Experiments 16 and 21 largely intact. If Experiment 16 remains low after claim normalization, allow only a modest, rapidly saturating support-certainty factor based on bilateral coalition breadth; avoid a linear breadth effect that would recreate Experiment 5.\n\nKeep count calibration near the current accepted setting: Experiment 2 improved markedly and Experiments 4 and 6 are close, so another broad reduction in count strength would be counterproductive. After the pooled means improve, modestly widen claim/count trait variation to address underestimated variance, while preserving exact option-swap symmetry, history independence, and stochastic representation recruitment."
}
```

## Usage

```json
{
  "input_tokens": 41534,
  "output_tokens": 2586
}
```
