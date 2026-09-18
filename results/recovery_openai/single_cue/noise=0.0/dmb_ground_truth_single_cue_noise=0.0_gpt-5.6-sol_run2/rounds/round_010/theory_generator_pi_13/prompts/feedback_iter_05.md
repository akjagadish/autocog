# feedback_iter_05

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
- The recommendation below acts on THEORY 2 (= `pi_12`).

Replace pi_12 with a theory of probabilistic source individuation under adaptive compression. The central claim should be that observers infer whether agreeing experts are independent sources, using exact equality as a strong—but not absolute—common-source cue and evaluating nonzero reliability differences relative to the reliability distribution in the current task. In small coalitions, especially dyads, exact equality should sharply increase inferred dependence, whereas even a tiny nonzero difference can preserve separate-source status; this captures Experiment 2 without a fixed metric tolerance. Coalition evidence should then combine expected independent-source count, ordinal reliability quality, and reliability-range or stratum coverage through smooth capacity-limited compression. Broadly distributed tetrads can retain a diversity advantage, but its magnitude should vary across subjects and decline gradually with load rather than being forcibly pooled at cardinality four; this permits Experiment 1's 0.64 prevalence and high heterogeneity while allowing near-chance chain-versus-duplicate outcomes in Experiments 20 and 22 when the represented range is narrow or compression is strong. Repetition costs should emerge from inferred dependence and the opponent's effective source count, not from cardinality-specific dyad and tetrad clauses. Include stable subject-level heterogeneity in source-independence priors, compression capacity, and diversity sensitivity so threshold metrics need not collapse to zero variance. Reliability weighting should remain weak and contextual, and cue identity should remain exchangeable, consistent with the largely null positional, onset, spacing, and closure results. Finally, add an opponent-relative conflict mechanism in which a coalition's effective source count can outweigh its highest individual cue; this mechanism must accommodate the anti-highest-cue effects in Experiments 3-4 and the stronger-than-predicted cardinality profile in Experiment 19 without introducing trial-specific exceptions. The new theory should make continuous predictions across dyads, triads, tetrads, and larger coalitions and avoid hard gates such as 'matched dyad' or 'n >= 4 one-sided contrast.'


## CANDIDATE THEORY
Adaptive Source-Resolution Compression proposes that observers infer latent informational sources behind agreeing experts. Exact reliability equality is a strong but probabilistic common-source cue, whereas any nonzero difference sharply supports individuation and only weakly varies with the task-relative size of that difference. Pairwise dependence beliefs yield a continuous effective-source count, and additional sources are encoded through subject-specific capacity-limited compression. Decisions use a single opponent-relative decomposition of coalition evidence: a bounded relative source-code contrast, continuously attenuated by coalition load and cardinality imbalance, is combined with separate bounded diversity and weak reliability-quality modifiers. Thus genuine source-configuration differences can matter in equal-cardinality conflicts without allowing raw coalition size to dominate unequal-cardinality choices. Reliability coverage is represented by a rapidly saturating, zero-preserving transform, so the first nonzero separation supplies most of the diversity signal while larger metric gaps add little. Stable heterogeneity in dependence priors, compression capacity, diversity sensitivity, conflict weighting, and exchangeable cue salience produces graded population prevalence without positional or cardinality-specific rules.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Adaptive Source-Resolution Compression expects shape "
            f"(2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = int(stim.shape[1])

    v = np.asarray(parameters["validities"], dtype=np.float64)
    if v.ndim != 1 or v.size != n_features:
        raise ValueError(
            f"validities length {v.size} != n_features {n_features}."
        )
    v = np.clip(v, 0.500001, 0.999999)

    raw_salience = np.asarray(
        parameters["cue_salience_profile"], dtype=np.float64
    )
    if raw_salience.ndim != 1 or raw_salience.size != n_features:
        raise ValueError(
            f"cue_salience_profile length {raw_salience.size} != "
            f"n_features {n_features}."
        )

    source_dependence_prior = float(parameters["source_dependence_prior"])
    equality_evidence = float(parameters["equality_evidence"])
    nonzero_separation = float(parameters["nonzero_separation"])
    separation_floor = float(parameters["separation_floor"])
    positive_gap_rank_gain = float(parameters["positive_gap_rank_gain"])
    compression_capacity = float(parameters["compression_capacity"])
    load_individuation = float(parameters["load_individuation"])
    source_signal_curvature = float(parameters["source_signal_curvature"])
    rank_reliance = float(parameters["rank_reliance"])
    diversity_sensitivity = float(parameters["diversity_sensitivity"])
    coverage_mix = float(parameters["coverage_mix"])
    salience_strength = float(parameters["salience_strength"])
    conflict_gain = float(parameters["conflict_gain"])
    conflict_scale = float(parameters["conflict_scale"])
    best_cue_tradeoff = float(parameters["best_cue_tradeoff"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Contextual ordinal reliability representation. Ties share a rank, and
    # cue identity does not enter the population-level weighting rule.
    unique_v, inverse, counts = np.unique(
        v, return_inverse=True, return_counts=True
    )
    n_levels = int(unique_v.size)
    if n_levels == 1:
        level_rank = np.full(1, 0.5, dtype=np.float64)
    else:
        cumulative = np.cumsum(counts).astype(np.float64)
        midranks = cumulative - 0.5 * counts
        level_rank = midranks / float(n_features)

    cue_rank = level_rank[inverse]
    rank_contrast = 2.0 * (cue_rank - float(np.mean(cue_rank)))
    reliability_quality = np.exp(
        np.clip(rank_reliance * rank_contrast, -10.0, 10.0)
    )

    # Stable salience is centered and exchangeable across identities. It can
    # create individual configuration preferences without a common onset,
    # spacing, closure, or recency gradient.
    centered_salience = raw_salience - float(np.mean(raw_salience))
    attention = np.exp(
        np.clip(salience_strength * centered_salience, -10.0, 10.0)
    )
    attention /= max(float(np.mean(attention)), 1e-12)

    cue_quality = reliability_quality * attention
    cue_quality /= max(float(np.mean(cue_quality)), 1e-12)

    # Build the task-relative distribution of strictly positive reliability
    # differences. Equality remains a separate probabilistic cue.
    positive_gaps = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            gap = abs(float(v[i] - v[j]))
            if gap > 0.0:
                positive_gaps.append(gap)
    positive_gaps = np.sort(np.asarray(positive_gaps, dtype=np.float64))

    task_range = float(np.max(v) - np.min(v))
    task_rank_sd = float(np.std(cue_rank))

    difference = a - b
    a_support = np.flatnonzero(difference > 0.0)
    b_support = np.flatnonzero(difference < 0.0)

    if a_support.size == 0 and b_support.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    def coalition_representation(indices):
        idx = np.asarray(indices, dtype=np.int64)
        n = int(idx.size)
        if n == 0:
            return 0.0, 0.0, 0.0

        if n == 1:
            effective_sources = 1.0
        else:
            dependence_sum = 0.0

            for ii in range(n):
                for jj in range(ii + 1, n):
                    gap = abs(float(v[idx[ii]] - v[idx[jj]]))

                    if gap == 0.0:
                        # Equality is strong but nonabsolute, with broad
                        # subject-level variation in its interpretation.
                        logit_dependence = (
                            source_dependence_prior + equality_evidence
                        )
                    else:
                        # Positive gaps primarily share one separate-source
                        # category. Gap rank supplies only a small contextual
                        # modulation, avoiding a broad metric gradient.
                        if positive_gaps.size == 0:
                            gap_percentile = 1.0
                        else:
                            gap_percentile = float(
                                np.searchsorted(
                                    positive_gaps, gap, side="right"
                                )
                            ) / float(positive_gaps.size)
                        separation = np.clip(
                            separation_floor
                            + positive_gap_rank_gain * gap_percentile,
                            0.0,
                            1.0,
                        )
                        logit_dependence = (
                            source_dependence_prior
                            - nonzero_separation * separation
                        )

                    dependence = 1.0 / (
                        1.0 + np.exp(-np.clip(logit_dependence, -50.0, 50.0))
                    )

                    # Dependence information declines gradually with load.
                    # The reduced decline preserves equality effects when a
                    # dyad becomes part of a triad without introducing a gate.
                    load = float(max(n - 2, 0))
                    retained_dependence = dependence * np.exp(
                        -load_individuation
                        * load
                        / max(compression_capacity, 1e-12)
                    )
                    dependence_sum += retained_dependence

            mean_dependence_load = 2.0 * dependence_sum / float(n)
            effective_sources = float(n) / (1.0 + mean_dependence_load)
            effective_sources = float(
                np.clip(effective_sources, 1.0, float(n))
            )

        # The first inferred source is fully represented. Additional sources
        # are capacity-limited and then compressed by a common smooth
        # curvature. This sharply reduces raw-count leverage without a
        # cardinality-specific rule.
        additional_sources = compression_capacity * (
            1.0
            - np.exp(
                -(effective_sources - 1.0)
                / max(compression_capacity, 1e-12)
            )
        )
        compressed_sources = 1.0 + additional_sources
        source_code = compressed_sources ** source_signal_curvature

        coalition_v = v[idx]
        coalition_ranks = cue_rank[idx]

        if n <= 1 or task_range <= 1e-12:
            range_coverage = 0.0
        else:
            raw_range_coverage = float(
                (np.max(coalition_v) - np.min(coalition_v)) / task_range
            )
            range_coverage = raw_range_coverage / (
                0.08 + 0.92 * raw_range_coverage
            )

        if n <= 1 or task_rank_sd <= 1e-12:
            stratum_coverage = 0.0
        else:
            raw_stratum_coverage = float(
                np.clip(
                    np.std(coalition_ranks) / task_rank_sd,
                    0.0,
                    1.0,
                )
            )
            stratum_coverage = raw_stratum_coverage / (
                0.08 + 0.92 * raw_stratum_coverage
            )

        coverage = (
            coverage_mix * range_coverage
            + (1.0 - coverage_mix) * stratum_coverage
        )
        coverage = float(np.clip(coverage, 0.0, 1.0))

        # Diversity is available at low load but decays increasingly strongly
        # for tetrads and larger coalitions through one continuous function.
        load_retention = np.exp(
            -float(max(n - 1, 0))
            / max(compression_capacity, 1e-12)
        )
        diversity_factor = 1.0 + (
            diversity_sensitivity * coverage * load_retention
        )

        mean_quality = float(np.mean(cue_quality[idx]))
        best_quality = float(np.max(cue_quality[idx]))
        signal = source_code * diversity_factor * mean_quality
        return float(signal), float(source_code), best_quality

    signal_a, sources_a, best_a = coalition_representation(a_support)
    signal_b, sources_b, best_b = coalition_representation(b_support)

    # Decompose each coalition into its source code and its diversity/quality
    # modifier. This prevents source magnitude from entering once through the
    # raw signal difference and again through a separate conflict term.
    modifier_a = signal_a / max(sources_a, 1e-12)
    modifier_b = signal_b / max(sources_b, 1e-12)
    mean_modifier = 0.5 * (modifier_a + modifier_b)
    modifier_component = 0.5 * (sources_a + sources_b) * (
        modifier_a - modifier_b
    )

    source_total = max(sources_a + sources_b, 1e-12)
    relative_source_contrast = (sources_a - sources_b) / source_total

    nominal_a = float(a_support.size)
    nominal_b = float(b_support.size)
    nominal_total = max(nominal_a + nominal_b, 1.0)
    mean_nominal_load = 0.5 * nominal_total
    cardinality_imbalance = abs(nominal_a - nominal_b) / nominal_total

    # The entire source-code contrast is represented once as a bounded,
    # opponent-relative quantity. Equal-cardinality source configurations
    # retain leverage, while unequal-cardinality and high-load contrasts are
    # attenuated continuously rather than by cardinality-specific gates.
    load_retention = compression_capacity / (
        compression_capacity + 0.35 * mean_nominal_load
    )
    imbalance_retention = np.exp(
        -mean_nominal_load
        * cardinality_imbalance
        / max(compression_capacity, 1e-12)
    )
    source_component = (
        mean_modifier
        * (2.0 + conflict_gain)
        * np.tanh(relative_source_contrast / max(conflict_scale, 1e-12))
        * load_retention
        * imbalance_retention
    )

    # Reliability remains a weak contextual tie-breaker and cannot duplicate
    # the unbounded source-count pathway removed above.
    best_component = best_cue_tradeoff * np.tanh(
        (best_a - best_b) / max(conflict_scale, 1e-12)
    )

    net_a = source_component + modifier_component + best_component

    logits = np.array(
        [0.5 * beta * net_a, -0.5 * beta * net_a], dtype=np.float64
    )
    logits -= np.max(logits)
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
- source_dependence_prior: [-2.20, 0.10]
- equality_evidence: [1.80, 5.20]
- nonzero_separation: [3.20, 5.80]
- separation_floor: [0.58, 0.78]
- positive_gap_rank_gain: [0.03, 0.14]
- compression_capacity: [1.00, 2.80]
- load_individuation: [0.18, 0.55]
- source_signal_curvature: [0.40, 0.66]
- rank_reliance: [0.002, 0.015]
- diversity_sensitivity: [0.12, 0.85]
- coverage_mix: [0.45, 0.75]
- salience_strength: [0.12, 0.34]
- conflict_gain: [0.04, 0.16]
- conflict_scale: [0.55, 1.10]
- best_cue_tradeoff: [0.00, 0.08]
- beta: [1.35, 1.90]
- epsilon: [0.025, 0.070]

`rationale`:
This is a localized revision of the accepted iteration-2 model. The principal edit removes the duplicated source-count pathway: instead of using the unbounded raw coalition-signal difference and then adding another source conflict term, each signal is decomposed into source code and diversity/quality modifier. The source-code contribution enters exactly once through a bounded relative contrast, with smooth load and cardinality-imbalance attenuation. This should reduce excessive many-versus-one and three-versus-two leverage in Experiments 1 and 6 while preserving stronger source-configuration leverage for the equal-sized coalitions in Experiment 2. The accepted multiplicative diversity computation, dependence inference, compression function, and load-individuation mechanism are otherwise retained. A second narrow edit applies a rapidly saturating zero-preserving transform to range and rank-dispersion coverage. Consequently, the first nonzero reliability difference supplies most of the individuation cue, while larger gaps add little, targeting Experiment 11 without globally weakening diversity in Experiments 12–16. Finally, rank reliance and the best-cue offset are narrowed because both can recreate a metric reliability gradient or cancel an effective-source advantage. Equality evidence, heterogeneity ranges, exchangeable salience, and the successful large-load and positional invariances remain unchanged.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.1730 -> ACCEPTED
- iter 2: loss=0.1529 -> ACCEPTED
- iter 3: loss=0.1609 -> REJECTED
- iter 4: loss=0.1628 -> REJECTED
- iter 5: loss=0.1606 -> REJECTED
- iter 6 (current candidate you are grading): loss=0.1587 -> REJECTED
Running-best (last accepted) base: iter 2 at loss=0.1529.

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
  - iter 1: -0.5721 (var=0.0141) (Δ vs real -0.4504)
  - iter 2: -0.3571 (var=0.0238) (Δ vs real -0.2354)
  - iter 3: -0.2087 (var=0.0376) (Δ vs real -0.0871)
  - iter 4: -0.3833 (var=0.0326) (Δ vs real -0.2617)
  - iter 5: -0.4396 (var=0.0256) (Δ vs real -0.3179)
  - iter 6 (current): -0.2058 (var=0.0349) (Δ vs real -0.0842)
**Other theories' values on this metric (for reference):**
- pi_1: 0.7129 (var=0.0389)
- pi_2: -0.6354 (var=0.0266)
- pi_3: -0.1033 (var=0.0092)
- pi_4: -0.1246 (var=0.0114)
- pi_5: -0.1179 (var=0.0105)
- pi_6: -0.0779 (var=0.0176)
- pi_7: -0.3083 (var=0.0280)
- pi_8: -0.1358 (var=0.0111)
- pi_9: -0.2600 (var=0.0153)
- pi_10: -0.1767 (var=0.0260)
- pi_11: -0.2383 (var=0.0360)
- pi_12: -0.3588 (var=0.0139)

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
  - iter 1: 0.0317 (var=0.0104) (Δ vs real +0.2958)
  - iter 2: 0.0129 (var=0.0119) (Δ vs real +0.2771)
  - iter 3: 0.0046 (var=0.0121) (Δ vs real +0.2687)
  - iter 4: 0.0254 (var=0.0113) (Δ vs real +0.2896)
  - iter 5: 0.0000 (var=0.0112) (Δ vs real +0.2642)
  - iter 6 (current): 0.0063 (var=0.0129) (Δ vs real +0.2704)
**Other theories' values on this metric (for reference):**
- pi_2: -0.0008 (var=0.0102)
- pi_1: 0.7221 (var=0.0226)
- pi_3: -0.1892 (var=0.0123)
- pi_4: -0.2667 (var=0.0120)
- pi_5: -0.2217 (var=0.0089)
- pi_6: -0.2321 (var=0.0159)
- pi_7: -0.0892 (var=0.0075)
- pi_8: -0.2221 (var=0.0114)
- pi_9: -0.0008 (var=0.0090)
- pi_10: 0.0396 (var=0.0103)
- pi_11: 0.0096 (var=0.0132)
- pi_12: 0.0117 (var=0.0092)

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
  - iter 1: 0.0053 (var=0.0003) (Δ vs real +0.0463)
  - iter 2: 0.0068 (var=0.0004) (Δ vs real +0.0478)
  - iter 3: 0.0113 (var=0.0003) (Δ vs real +0.0523)
  - iter 4: 0.0071 (var=0.0002) (Δ vs real +0.0481)
  - iter 5: -0.0014 (var=0.0003) (Δ vs real +0.0396)
  - iter 6 (current): 0.0042 (var=0.0004) (Δ vs real +0.0452)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0860 (var=0.0002)
- pi_2: -0.0017 (var=0.0001)
- pi_1: 0.0899 (var=0.0009)
- pi_4: -0.0260 (var=0.0002)
- pi_5: -0.0237 (var=0.0003)
- pi_6: -0.0366 (var=0.0003)
- pi_7: -0.0224 (var=0.0002)
- pi_8: -0.0376 (var=0.0003)
- pi_9: 0.0012 (var=0.0003)
- pi_10: 0.0007 (var=0.0004)
- pi_11: 0.0039 (var=0.0005)
- pi_12: 0.0144 (var=0.0003)

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
  - iter 1: 0.5127 (var=0.0064) (Δ vs real +0.0041)
  - iter 2: 0.5193 (var=0.0057) (Δ vs real +0.0106)
  - iter 3: 0.4968 (var=0.0055) (Δ vs real -0.0118)
  - iter 4: 0.4959 (var=0.0055) (Δ vs real -0.0128)
  - iter 5: 0.5124 (var=0.0076) (Δ vs real +0.0037)
  - iter 6 (current): 0.4953 (var=0.0076) (Δ vs real -0.0134)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5017 (var=0.0041)
- pi_3: 0.6281 (var=0.0049)
- pi_1: 0.1476 (var=0.0116)
- pi_4: 0.5061 (var=0.0029)
- pi_5: 0.4957 (var=0.0036)
- pi_6: 0.5046 (var=0.0065)
- pi_7: 0.4932 (var=0.0039)
- pi_8: 0.4759 (var=0.0052)
- pi_9: 0.5033 (var=0.0044)
- pi_10: 0.4788 (var=0.0086)
- pi_11: 0.4771 (var=0.0066)
- pi_12: 0.5132 (var=0.0055)

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
  - iter 1: 0.4931 (var=0.0041) (Δ vs real -0.0090)
  - iter 2: 0.5000 (var=0.0041) (Δ vs real -0.0021)
  - iter 3: 0.5148 (var=0.0029) (Δ vs real +0.0127)
  - iter 4: 0.5044 (var=0.0044) (Δ vs real +0.0023)
  - iter 5: 0.4935 (var=0.0029) (Δ vs real -0.0085)
  - iter 6 (current): 0.4952 (var=0.0039) (Δ vs real -0.0069)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5692 (var=0.0021)
- pi_4: 0.3798 (var=0.0029)
- pi_1: 0.1575 (var=0.0119)
- pi_2: 0.4992 (var=0.0023)
- pi_5: 0.5098 (var=0.0033)
- pi_6: 0.5123 (var=0.0025)
- pi_7: 0.5052 (var=0.0023)
- pi_8: 0.4940 (var=0.0029)
- pi_9: 0.4973 (var=0.0028)
- pi_10: 0.4923 (var=0.0026)
- pi_11: 0.5060 (var=0.0029)
- pi_12: 0.4983 (var=0.0020)

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
  - iter 1: 0.9600 (var=0.0384) (Δ vs real +0.5200)
  - iter 2: 0.8200 (var=0.1476) (Δ vs real +0.3800)
  - iter 3: 0.3400 (var=0.2244) (Δ vs real -0.1000)
  - iter 4: 0.4000 (var=0.2400) (Δ vs real -0.0400)
  - iter 5: 0.5200 (var=0.2496) (Δ vs real +0.0800)
  - iter 6 (current): 0.5400 (var=0.2484) (Δ vs real +0.1000)
**Other theories' values on this metric (for reference):**
- pi_4: 0.9800 (var=0.0196)
- pi_3: 0.0000 (var=0.0000)
- pi_1: 1.0000 (var=0.0000)
- pi_2: 1.0000 (var=0.0000)
- pi_5: 0.3200 (var=0.2176)
- pi_6: 0.3600 (var=0.2304)
- pi_7: 0.3200 (var=0.2176)
- pi_8: 0.5000 (var=0.2500)
- pi_9: 0.1600 (var=0.1344)
- pi_10: 0.0600 (var=0.0564)
- pi_11: 0.1600 (var=0.1344)
- pi_12: 0.8400 (var=0.1344)

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
  - iter 1: 0.0045 (var=0.0094) (Δ vs real -0.0174)
  - iter 2: 0.0172 (var=0.0113) (Δ vs real -0.0047)
  - iter 3: 0.0105 (var=0.0105) (Δ vs real -0.0113)
  - iter 4: -0.0030 (var=0.0114) (Δ vs real -0.0249)
  - iter 5: -0.0092 (var=0.0121) (Δ vs real -0.0310)
  - iter 6 (current): 0.0117 (var=0.0111) (Δ vs real -0.0102)
**Other theories' values on this metric (for reference):**
- pi_5: 0.2360 (var=0.0123)
- pi_4: 0.0009 (var=0.0095)
- pi_1: 0.0069 (var=0.0048)
- pi_2: -0.0035 (var=0.0050)
- pi_3: -0.0376 (var=0.0081)
- pi_6: 0.0257 (var=0.0065)
- pi_7: -0.0108 (var=0.0126)
- pi_8: 0.0088 (var=0.0097)
- pi_9: -0.0034 (var=0.0107)
- pi_10: -0.0065 (var=0.0117)
- pi_11: 0.0143 (var=0.0064)
- pi_12: -0.0353 (var=0.0117)

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
  - iter 1: -0.0183 (var=0.0104) (Δ vs real -0.0300)
  - iter 2: 0.0229 (var=0.0127) (Δ vs real +0.0112)
  - iter 3: -0.0083 (var=0.0159) (Δ vs real -0.0200)
  - iter 4: 0.0088 (var=0.0151) (Δ vs real -0.0029)
  - iter 5: 0.0108 (var=0.0136) (Δ vs real -0.0008)
  - iter 6 (current): 0.0050 (var=0.0198) (Δ vs real -0.0067)
**Other theories' values on this metric (for reference):**
- pi_4: 0.2729 (var=0.0146)
- pi_5: -0.0100 (var=0.0101)
- pi_1: 0.7325 (var=0.0409)
- pi_2: -0.0042 (var=0.0038)
- pi_3: -0.4775 (var=0.0097)
- pi_6: -0.0154 (var=0.0165)
- pi_7: -0.0021 (var=0.0101)
- pi_8: -0.0042 (var=0.0116)
- pi_9: 0.0033 (var=0.0081)
- pi_10: -0.0167 (var=0.0081)
- pi_11: -0.0054 (var=0.0118)
- pi_12: 0.0358 (var=0.0069)

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
  - iter 1: 0.5065 (var=0.0034) (Δ vs real +0.0106)
  - iter 2: 0.4913 (var=0.0026) (Δ vs real -0.0046)
  - iter 3: 0.4854 (var=0.0028) (Δ vs real -0.0104)
  - iter 4: 0.5117 (var=0.0039) (Δ vs real +0.0158)
  - iter 5: 0.4973 (var=0.0026) (Δ vs real +0.0015)
  - iter 6 (current): 0.5065 (var=0.0029) (Δ vs real +0.0106)
**Other theories' values on this metric (for reference):**
- pi_5: 0.6108 (var=0.0033)
- pi_6: 0.4996 (var=0.0028)
- pi_1: 0.4104 (var=0.0015)
- pi_2: 0.4985 (var=0.0025)
- pi_3: 0.6115 (var=0.0020)
- pi_4: 0.6271 (var=0.0029)
- pi_7: 0.4946 (var=0.0027)
- pi_8: 0.5056 (var=0.0026)
- pi_9: 0.4960 (var=0.0023)
- pi_10: 0.4925 (var=0.0029)
- pi_11: 0.5090 (var=0.0024)
- pi_12: 0.4890 (var=0.0018)

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
  - iter 1: 0.2903 (var=0.0067) (Δ vs real +0.1180)
  - iter 2: 0.3739 (var=0.0063) (Δ vs real +0.2016)
  - iter 3: 0.3352 (var=0.0069) (Δ vs real +0.1629)
  - iter 4: 0.1836 (var=0.0056) (Δ vs real +0.0113)
  - iter 5: 0.2766 (var=0.0085) (Δ vs real +0.1043)
  - iter 6 (current): 0.3189 (var=0.0085) (Δ vs real +0.1465)
**Other theories' values on this metric (for reference):**
- pi_6: 0.2057 (var=0.0035)
- pi_5: 0.5023 (var=0.0031)
- pi_1: 0.8796 (var=0.0067)
- pi_2: 0.4978 (var=0.0039)
- pi_3: 0.5414 (var=0.0029)
- pi_4: 0.3753 (var=0.0039)
- pi_7: 0.1814 (var=0.0023)
- pi_8: 0.1646 (var=0.0019)
- pi_9: 0.2645 (var=0.0031)
- pi_10: 0.3205 (var=0.0042)
- pi_11: 0.2817 (var=0.0060)
- pi_12: 0.3006 (var=0.0042)

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
  - iter 1: 0.1704 (var=0.0134) (Δ vs real +0.1388)
  - iter 2: 0.1404 (var=0.0122) (Δ vs real +0.1087)
  - iter 3: 0.1129 (var=0.0092) (Δ vs real +0.0813)
  - iter 4: 0.2250 (var=0.0123) (Δ vs real +0.1933)
  - iter 5: 0.1471 (var=0.0094) (Δ vs real +0.1154)
  - iter 6 (current): 0.1171 (var=0.0099) (Δ vs real +0.0854)
**Other theories' values on this metric (for reference):**
- pi_7: 0.2729 (var=0.0090)
- pi_6: 0.0217 (var=0.0086)
- pi_1: 0.0167 (var=0.0064)
- pi_2: -0.0292 (var=0.0133)
- pi_3: 0.1604 (var=0.0061)
- pi_4: 0.0054 (var=0.0079)
- pi_5: 0.0146 (var=0.0117)
- pi_8: 0.0125 (var=0.0063)
- pi_9: 0.1046 (var=0.0086)
- pi_10: 0.0933 (var=0.0083)
- pi_11: 0.1204 (var=0.0114)
- pi_12: 0.1071 (var=0.0100)

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
  - iter 1: 0.6535 (var=0.0061) (Δ vs real -0.0156)
  - iter 2: 0.6092 (var=0.0053) (Δ vs real -0.0600)
  - iter 3: 0.5721 (var=0.0056) (Δ vs real -0.0971)
  - iter 4: 0.6750 (var=0.0050) (Δ vs real +0.0058)
  - iter 5: 0.5604 (var=0.0062) (Δ vs real -0.1088)
  - iter 6 (current): 0.5635 (var=0.0073) (Δ vs real -0.1056)
**Other theories' values on this metric (for reference):**
- pi_6: 0.4883 (var=0.0028)
- pi_7: 0.5575 (var=0.0028)
- pi_1: 0.1550 (var=0.0096)
- pi_2: 0.5073 (var=0.0027)
- pi_3: 0.6663 (var=0.0014)
- pi_4: 0.5715 (var=0.0020)
- pi_5: 0.5602 (var=0.0027)
- pi_8: 0.7031 (var=0.0082)
- pi_9: 0.6831 (var=0.0036)
- pi_10: 0.7271 (var=0.0030)
- pi_11: 0.6735 (var=0.0042)
- pi_12: 0.6571 (var=0.0061)

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
  - iter 1: 0.2308 (var=0.0264) (Δ vs real -0.0933)
  - iter 2: 0.1817 (var=0.0252) (Δ vs real -0.1425)
  - iter 3: 0.1388 (var=0.0146) (Δ vs real -0.1854)
  - iter 4: 0.3775 (var=0.0234) (Δ vs real +0.0533)
  - iter 5: 0.1200 (var=0.0268) (Δ vs real -0.2042)
  - iter 6 (current): 0.0750 (var=0.0231) (Δ vs real -0.2492)
**Other theories' values on this metric (for reference):**
- pi_8: 0.5346 (var=0.0171)
- pi_6: 0.0158 (var=0.0163)
- pi_1: -0.6938 (var=0.0472)
- pi_2: 0.0129 (var=0.0129)
- pi_3: 0.6592 (var=0.0081)
- pi_4: 0.2250 (var=0.0097)
- pi_5: 0.2188 (var=0.0122)
- pi_7: 0.3412 (var=0.0162)
- pi_9: 0.3688 (var=0.0124)
- pi_10: 0.4242 (var=0.0096)
- pi_11: 0.3646 (var=0.0242)
- pi_12: 0.3475 (var=0.0228)

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
  - iter 1: 0.5813 (var=0.0053) (Δ vs real -0.0056)
  - iter 2: 0.5522 (var=0.0037) (Δ vs real -0.0347)
  - iter 3: 0.5266 (var=0.0047) (Δ vs real -0.0603)
  - iter 4: 0.5938 (var=0.0039) (Δ vs real +0.0069)
  - iter 5: 0.5234 (var=0.0050) (Δ vs real -0.0634)
  - iter 6 (current): 0.5141 (var=0.0034) (Δ vs real -0.0728)
**Other theories' values on this metric (for reference):**
- pi_6: 0.5012 (var=0.0028)
- pi_8: 0.6616 (var=0.0045)
- pi_1: 0.1222 (var=0.0090)
- pi_2: 0.5122 (var=0.0043)
- pi_3: 0.6784 (var=0.0035)
- pi_4: 0.6269 (var=0.0040)
- pi_5: 0.6112 (var=0.0040)
- pi_7: 0.5816 (var=0.0036)
- pi_9: 0.5988 (var=0.0038)
- pi_10: 0.6191 (var=0.0035)
- pi_11: 0.5844 (var=0.0036)
- pi_12: 0.5878 (var=0.0035)

### Experiment 15
**Design**
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Triadic onset in preference for reliability-separated coalitions."""
    dyad_choices = []
    triad_choices = []

    for row in data.itertuples(index=False):
        a = np.asarray(row.option_a_ratings, dtype=np.int8)
        b = np.asarray(row.option_b_ratings, dtype=np.int8)
        d = a - b

        a_support = np.flatnonzero(d > 0)
        b_support = np.flatnonzero(d < 0)
        if a_support.size != b_support.size or a_support.size not in (2, 3):
            continue

        # The reliability-separated pair uses cues 4:8 (validities .65 and
        # .7666667); the matched pair uses two cues 0:4 (validity .70).
        a_mid = int(np.sum((a_support >= 4) & (a_support < 8)))
        b_mid = int(np.sum((b_support >= 4) & (b_support < 8)))
        a_equal = int(np.sum(a_support < 4))
        b_equal = int(np.sum(b_support < 4))

        if a_mid == 2 and b_equal == 2:
            separated_is_a = True
        elif b_mid == 2 and a_equal == 2:
            separated_is_a = False
        else:
            continue

        response = int(row.response)
        chose_separated = float(
            (separated_is_a and response == 0)
            or ((not separated_is_a) and response == 1)
        )

        if a_support.size == 2:
            dyad_choices.append(chose_separated)
        else:
            triad_choices.append(chose_separated)

    if len(dyad_choices) == 0 or len(triad_choices) == 0:
        return float("nan")

    return float(np.mean(triad_choices) - np.mean(dyad_choices))
```

**Observed (real) value:** 0.0142 (var=0.0045)
**Candidate trajectory (this loop):**
  - iter 1: -0.1271 (var=0.0074) (Δ vs real -0.1412)
  - iter 2: -0.1113 (var=0.0115) (Δ vs real -0.1254)
  - iter 3: -0.1417 (var=0.0106) (Δ vs real -0.1558)
  - iter 4: -0.1479 (var=0.0064) (Δ vs real -0.1621)
  - iter 5: -0.2013 (var=0.0149) (Δ vs real -0.2154)
  - iter 6 (current): -0.1417 (var=0.0093) (Δ vs real -0.1558)
**Other theories' values on this metric (for reference):**
- pi_8: 0.2675 (var=0.0085)
- pi_9: -0.0350 (var=0.0068)
- pi_1: -0.3600 (var=0.0100)
- pi_2: -0.0133 (var=0.0099)
- pi_3: -0.0054 (var=0.0075)
- pi_4: -0.1288 (var=0.0113)
- pi_5: 0.0125 (var=0.0091)
- pi_6: -0.0029 (var=0.0108)
- pi_7: 0.0317 (var=0.0068)
- pi_10: 0.0337 (var=0.0098)
- pi_11: -0.0717 (var=0.0091)
- pi_12: -0.1379 (var=0.0071)

### Experiment 16
**Design**
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Mirror-corrected probability of choosing a single-duplicate coalition
    # (features 8:16) over a double-duplicate coalition (features 0:8).
    chosen_single = []
    single_on_a = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=np.float64)
        b = np.asarray(row["option_b_ratings"], dtype=np.float64)
        if a.size < 16 or b.size < 16:
            continue

        a_is_single = float(np.sum(a[8:16])) > float(np.sum(a[0:8]))
        b_is_single = float(np.sum(b[8:16])) > float(np.sum(b[0:8]))
        if a_is_single == b_is_single:
            continue

        response = int(row["response"])
        chose_single = ((response == 0 and a_is_single) or
                         (response == 1 and b_is_single))
        chosen_single.append(float(chose_single))
        single_on_a.append(bool(a_is_single))

    if not chosen_single:
        return float("nan")

    y = np.asarray(chosen_single, dtype=np.float64)
    side = np.asarray(single_on_a, dtype=bool)

    # Give the two mirror orientations equal weight, canceling stable A/B bias.
    if np.any(side) and np.any(~side):
        return float(0.5 * (np.mean(y[side]) + np.mean(y[~side])))
    return float(np.mean(y))

```

**Observed (real) value:** 0.6779 (var=0.0032)
**Candidate trajectory (this loop):**
  - iter 1: 0.6008 (var=0.0054) (Δ vs real -0.0771)
  - iter 2: 0.5585 (var=0.0042) (Δ vs real -0.1194)
  - iter 3: 0.5192 (var=0.0029) (Δ vs real -0.1588)
  - iter 4: 0.5819 (var=0.0056) (Δ vs real -0.0960)
  - iter 5: 0.5117 (var=0.0038) (Δ vs real -0.1663)
  - iter 6 (current): 0.5050 (var=0.0044) (Δ vs real -0.1729)
**Other theories' values on this metric (for reference):**
- pi_9: 0.6575 (var=0.0035)
- pi_8: 0.5031 (var=0.0031)
- pi_1: 0.8390 (var=0.0107)
- pi_2: 0.5010 (var=0.0023)
- pi_3: 0.8723 (var=0.0016)
- pi_4: 0.6344 (var=0.0024)
- pi_5: 0.6090 (var=0.0024)
- pi_6: 0.4815 (var=0.0033)
- pi_7: 0.6562 (var=0.0033)
- pi_10: 0.6777 (var=0.0028)
- pi_11: 0.6365 (var=0.0053)
- pi_12: 0.7210 (var=0.0079)

### Experiment 17
**Design**
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Signed, inverse-variance-weighted cardinality profile of choices for the
    # reliability-diverse coalition. The duplicate pair identifying each
    # cardinality is fixed by this experimental design.
    duplicate_pair = {2: (0, 1), 3: (4, 5), 4: (10, 11)}
    weights = {2: -1.00, 3: 0.53, 4: -0.51}

    def subject_score(df):
        chosen_diverse = {2: [], 3: [], 4: []}

        for _, row in df.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float)
            b = np.asarray(row["option_b_ratings"], dtype=float)
            d = a - b
            cardinality = int(np.sum(d > 0))
            if cardinality not in duplicate_pair:
                continue

            i, j = duplicate_pair[cardinality]
            if d[i] > 0 and d[j] > 0:
                # The duplicate coalition supports A, so B is diverse.
                y = float(int(row["response"]) == 1)
            elif d[i] < 0 and d[j] < 0:
                # The duplicate coalition supports B, so A is diverse.
                y = float(int(row["response"]) == 0)
            else:
                continue
            chosen_diverse[cardinality].append(y)

        if any(len(chosen_diverse[k]) == 0 for k in (2, 3, 4)):
            return np.nan

        rates = {k: float(np.mean(chosen_diverse[k])) for k in (2, 3, 4)}
        return float(sum(weights[k] * (rates[k] - 0.5) for k in (2, 3, 4)))

    scores = []
    for _, subj_df in data.groupby("subject_id", sort=False):
        value = subject_score(subj_df)
        if np.isfinite(value):
            scores.append(value)

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** -0.3486 (var=0.0131)
**Candidate trajectory (this loop):**
  - iter 1: -0.2517 (var=0.0139) (Δ vs real +0.0969)
  - iter 2: -0.2148 (var=0.0148) (Δ vs real +0.1338)
  - iter 3: -0.1947 (var=0.0162) (Δ vs real +0.1539)
  - iter 4: -0.3337 (var=0.0101) (Δ vs real +0.0149)
  - iter 5: -0.2449 (var=0.0163) (Δ vs real +0.1037)
  - iter 6 (current): -0.1758 (var=0.0236) (Δ vs real +0.1728)
**Other theories' values on this metric (for reference):**
- pi_10: -0.1210 (var=0.0132)
- pi_9: -0.2371 (var=0.0104)
- pi_1: 0.0290 (var=0.0066)
- pi_2: 0.0005 (var=0.0172)
- pi_3: -0.1762 (var=0.0147)
- pi_4: -0.1228 (var=0.0094)
- pi_5: 0.0260 (var=0.0189)
- pi_6: -0.0595 (var=0.0208)
- pi_7: -0.3008 (var=0.0082)
- pi_8: -0.0731 (var=0.0163)
- pi_11: -0.1886 (var=0.0173)
- pi_12: -0.2260 (var=0.0128)

### Experiment 18
**Design**
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Equal-weighted, mirror-corrected probability of choosing the chain coalition."""
    if data is None or len(data) == 0:
        return float("nan")

    records = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.size < 20 or b.size != a.size:
            continue

        d = a - b
        chain_direction = float(np.sum(d[:10]))
        if abs(chain_direction) < 1e-12:
            continue

        # Positive means that A is supported by one of the two chain blocks;
        # negative means that B is supported by a chain block.
        chain_is_a = chain_direction > 0.0
        response = int(row["response"])
        chain_chosen = float((response == 0 and chain_is_a) or
                             (response == 1 and not chain_is_a))

        # Identify the crossed chain/comparison configuration independently
        # of whether it was displayed in its original or mirrored orientation.
        chain_block = int(np.sum(np.abs(d[5:10])) > np.sum(np.abs(d[0:5])))
        comparison_block = int(
            np.sum(np.abs(d[15:20])) > np.sum(np.abs(d[10:15]))
        )
        family = 2 * chain_block + comparison_block
        records.append((row["subject_id"], family, chain_chosen))

    if not records:
        return float("nan")

    scored = pd.DataFrame(records, columns=["subject_id", "family", "score"])
    # Equal weighting of the four crossed configurations prevents stable cue
    # salience or an uneven number of usable repetitions from dominating.
    family_rates = scored.groupby(["subject_id", "family"])["score"].mean()
    subject_rates = family_rates.groupby(level=0).mean()
    return float(subject_rates.mean())
```

**Observed (real) value:** 0.4983 (var=0.0018)
**Candidate trajectory (this loop):**
  - iter 1: 0.5540 (var=0.0072) (Δ vs real +0.0556)
  - iter 2: 0.5060 (var=0.0045) (Δ vs real +0.0077)
  - iter 3: 0.5304 (var=0.0031) (Δ vs real +0.0321)
  - iter 4: 0.5663 (var=0.0045) (Δ vs real +0.0679)
  - iter 5: 0.5204 (var=0.0047) (Δ vs real +0.0221)
  - iter 6 (current): 0.5019 (var=0.0049) (Δ vs real +0.0035)
**Other theories' values on this metric (for reference):**
- pi_9: 0.7094 (var=0.0054)
- pi_10: 0.2913 (var=0.0022)
- pi_1: 0.1706 (var=0.0080)
- pi_2: 0.4950 (var=0.0026)
- pi_3: 0.1296 (var=0.0016)
- pi_4: 0.3533 (var=0.0025)
- pi_5: 0.3854 (var=0.0022)
- pi_6: 0.5079 (var=0.0037)
- pi_7: 0.4525 (var=0.0026)
- pi_8: 0.5229 (var=0.0036)
- pi_11: 0.4998 (var=0.0039)
- pi_12: 0.4948 (var=0.0035)

### Experiment 19
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    narrow_choices = []
    wide_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        response = int(row["response"])

        a_support = np.flatnonzero(a > b)
        b_support = np.flatnonzero(b > a)

        # Features 0--3 form the two narrow-gap target dyads;
        # features 4--7 form the two wide-gap target dyads.
        a_narrow = np.any(a_support < 4)
        b_narrow = np.any(b_support < 4)
        a_wide = np.any((a_support >= 4) & (a_support < 8))
        b_wide = np.any((b_support >= 4) & (b_support < 8))

        if a_narrow and not b_narrow:
            narrow_choices.append(float(response == 0))
        elif b_narrow and not a_narrow:
            narrow_choices.append(float(response == 1))
        elif a_wide and not b_wide:
            wide_choices.append(float(response == 0))
        elif b_wide and not a_wide:
            wide_choices.append(float(response == 1))

    if len(narrow_choices) == 0 or len(wide_choices) == 0:
        return float("nan")

    return float(np.mean(wide_choices) - np.mean(narrow_choices))
```

**Observed (real) value:** -0.0283 (var=0.0129)
**Candidate trajectory (this loop):**
  - iter 1: 0.0012 (var=0.0115) (Δ vs real +0.0296)
  - iter 2: -0.0058 (var=0.0114) (Δ vs real +0.0225)
  - iter 3: 0.0083 (var=0.0132) (Δ vs real +0.0367)
  - iter 4: 0.0158 (var=0.0050) (Δ vs real +0.0442)
  - iter 5: -0.0021 (var=0.0092) (Δ vs real +0.0263)
  - iter 6 (current): -0.0138 (var=0.0096) (Δ vs real +0.0146)
**Other theories' values on this metric (for reference):**
- pi_11: -0.0029 (var=0.0090)
- pi_9: 0.1900 (var=0.0096)
- pi_1: -0.0088 (var=0.0046)
- pi_2: 0.0033 (var=0.0099)
- pi_3: 0.0775 (var=0.0094)
- pi_4: 0.0067 (var=0.0102)
- pi_5: 0.0087 (var=0.0124)
- pi_6: 0.0029 (var=0.0113)
- pi_7: 0.0192 (var=0.0098)
- pi_8: -0.0146 (var=0.0165)
- pi_10: -0.0008 (var=0.0157)
- pi_12: -0.0354 (var=0.0045)

### Experiment 20
**Design**
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Score each response as 1 when the chain-validity coalition was chosen
    # and 0 when the exact-duplicate comparison coalition was chosen.
    chain_when_a = []
    chain_when_b = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=np.float64)
        b = np.asarray(row["option_b_ratings"], dtype=np.float64)
        d = a - b

        # In this design, chain experts occupy indices 0:8. Exactly one side
        # is supported by four of these experts on every experimental trial.
        a_chain_support = int(np.sum(d[:8] > 0.0))
        b_chain_support = int(np.sum(d[:8] < 0.0))
        response = int(row["response"])

        if a_chain_support > b_chain_support:
            chain_when_a.append(float(response == 0))
        elif b_chain_support > a_chain_support:
            chain_when_b.append(float(response == 1))

    # Average the two response-side orientations separately so the statistic
    # is explicitly mirror-corrected rather than sensitive to side imbalance.
    orientation_means = []
    if len(chain_when_a) > 0:
        orientation_means.append(float(np.mean(chain_when_a)))
    if len(chain_when_b) > 0:
        orientation_means.append(float(np.mean(chain_when_b)))

    if len(orientation_means) == 0:
        return float("nan")
    return float(np.mean(orientation_means))
```

**Observed (real) value:** 0.5042 (var=0.0027)
**Candidate trajectory (this loop):**
  - iter 1: 0.5844 (var=0.0040) (Δ vs real +0.0802)
  - iter 2: 0.5398 (var=0.0046) (Δ vs real +0.0356)
  - iter 3: 0.5585 (var=0.0031) (Δ vs real +0.0544)
  - iter 4: 0.6638 (var=0.0050) (Δ vs real +0.1596)
  - iter 5: 0.5331 (var=0.0074) (Δ vs real +0.0290)
  - iter 6 (current): 0.5285 (var=0.0040) (Δ vs real +0.0244)
**Other theories' values on this metric (for reference):**
- pi_9: 0.7292 (var=0.0039)
- pi_11: 0.5040 (var=0.0037)
- pi_1: 0.1427 (var=0.0067)
- pi_2: 0.5029 (var=0.0031)
- pi_3: 0.1819 (var=0.0027)
- pi_4: 0.3612 (var=0.0019)
- pi_5: 0.3902 (var=0.0029)
- pi_6: 0.5054 (var=0.0028)
- pi_7: 0.4792 (var=0.0023)
- pi_8: 0.6019 (var=0.0034)
- pi_10: 0.4062 (var=0.0025)
- pi_12: 0.5085 (var=0.0024)

### Experiment 21
**Design**
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Fraction of subjects showing a supermajority preference for diverse coalitions."""
    if data is None or len(data) == 0:
        return 0.0

    records = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.size != 16 or b.size != 16:
            continue

        diff = a - b
        a_support = np.flatnonzero(diff > 0)
        b_support = np.flatnonzero(diff < 0)

        # In this design, indices 0--7 form the two reliability-diverse
        # coalitions, whereas indices 8--15 form the exact-duplicate blocks.
        a_diverse = (a_support.size == 4 and np.all(a_support < 8))
        b_diverse = (b_support.size == 4 and np.all(b_support < 8))
        a_duplicate = (a_support.size == 4 and np.all(a_support >= 8))
        b_duplicate = (b_support.size == 4 and np.all(b_support >= 8))

        if a_diverse and b_duplicate:
            chose_diverse = int(row["response"] == 0)
        elif b_diverse and a_duplicate:
            chose_diverse = int(row["response"] == 1)
        else:
            continue

        records.append((row["subject_id"], chose_diverse))

    if len(records) == 0:
        return 0.0

    scored = pd.DataFrame(records, columns=["subject_id", "chose_diverse"])
    subject_rates = scored.groupby("subject_id")["chose_diverse"].mean()

    # A 2/3 threshold lies well above the competing theory's near-chance
    # prediction but below the advocated theory's predicted range.
    return float(np.mean(subject_rates >= (2.0 / 3.0)))
```

**Observed (real) value:** 0.6400 (var=0.2304)
**Candidate trajectory (this loop):**
  - iter 1: 0.9000 (var=0.0900) (Δ vs real +0.2600)
  - iter 2: 0.7400 (var=0.1924) (Δ vs real +0.1000)
  - iter 3: 0.4000 (var=0.2400) (Δ vs real -0.2400)
  - iter 4: 1.0000 (var=0.0000) (Δ vs real +0.3600)
  - iter 5: 0.7800 (var=0.1716) (Δ vs real +0.1400)
  - iter 6 (current): 0.3600 (var=0.2304) (Δ vs real -0.2800)
**Other theories' values on this metric (for reference):**
- pi_11: 1.0000 (var=0.0000)
- pi_12: 0.0000 (var=0.0000)
- pi_1: 0.9600 (var=0.0384)
- pi_2: 0.0000 (var=0.0000)
- pi_3: 0.0000 (var=0.0000)
- pi_4: 0.0000 (var=0.0000)
- pi_5: 0.0000 (var=0.0000)
- pi_6: 0.0000 (var=0.0000)
- pi_7: 1.0000 (var=0.0000)
- pi_8: 0.0000 (var=0.0000)
- pi_9: 1.0000 (var=0.0000)
- pi_10: 1.0000 (var=0.0000)

### Experiment 22
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # In this fixed design, features 0--3 form the two near-duplicate
    # coalitions, whereas features 4--7 form the two exact-duplicate
    # coalitions. Every trial pits one type against the other.
    near_choices = []

    for row in data.itertuples(index=False):
        a = np.asarray(row.option_a_ratings, dtype=np.float64)
        b = np.asarray(row.option_b_ratings, dtype=np.float64)

        a_near_mass = float(np.sum(a[:4]))
        a_exact_mass = float(np.sum(a[4:8]))
        b_near_mass = float(np.sum(b[:4]))
        b_exact_mass = float(np.sum(b[4:8]))

        a_is_near = a_near_mass > a_exact_mass
        b_is_near = b_near_mass > b_exact_mass

        # Retain only the intended near-versus-exact diagnostic trials.
        if a_is_near == b_is_near:
            continue

        response = int(row.response)
        chose_near = a_is_near if response == 0 else b_is_near
        near_choices.append(float(chose_near))

    if len(near_choices) == 0:
        return 0.5
    return float(np.mean(near_choices))
```

**Observed (real) value:** 0.6817 (var=0.0026)
**Candidate trajectory (this loop):**
  - iter 1: 0.7285 (var=0.0044) (Δ vs real +0.0469)
  - iter 2: 0.6621 (var=0.0078) (Δ vs real -0.0196)
  - iter 3: 0.6727 (var=0.0040) (Δ vs real -0.0090)
  - iter 4: 0.8181 (var=0.0054) (Δ vs real +0.1365)
  - iter 5: 0.7438 (var=0.0060) (Δ vs real +0.0621)
  - iter 6 (current): 0.6598 (var=0.0058) (Δ vs real -0.0219)
**Other theories' values on this metric (for reference):**
- pi_12: 0.7017 (var=0.0030)
- pi_11: 0.4998 (var=0.0029)
- pi_1: 0.8473 (var=0.0103)
- pi_2: 0.5046 (var=0.0017)
- pi_3: 0.4006 (var=0.0025)
- pi_4: 0.3654 (var=0.0023)
- pi_5: 0.4827 (var=0.0022)
- pi_6: 0.4890 (var=0.0027)
- pi_7: 0.5088 (var=0.0022)
- pi_8: 0.4983 (var=0.0026)
- pi_9: 0.7244 (var=0.0023)
- pi_10: 0.5156 (var=0.0026)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed probabilistic source-individuation and adaptive-compression family, and it captures several important invariances: positional and organizational effects are appropriately near null in Experiments 4, 5, 7, 8, and 9; distributed-coalition effects fit well in Experiments 12 and 14; Experiment 19 is near its null contrast; and Experiment 22 shows the required near-versus-exact equality discontinuity. However, the model is not yet experiment-invariant. Its largest failures indicate excessive coalition-count leverage: it is much too anti-highest-cue in Experiment 1 (-0.572 versus -0.122), predicts almost every subject to favor the early triad in Experiment 6 (0.96 versus 0.44), and misses the anti-highest-cue configuration in Experiment 2 in the opposite direction (0.032 versus -0.264). It also produces an excessive continuous reliability-gap gradient in Experiment 11 (0.170 versus 0.032), an unwanted dyad-to-triad decline in the separated-coalition advantage in Experiment 15 (-0.127 versus 0.014), and too much chain/diversity advantage at larger loads in Experiments 18 and 20. Subject heterogeneity remains inadequate in Experiment 21: prevalence is 0.90 rather than 0.64 and variance is correspondingly too low. Secondary misses occur in Experiments 10, 13, 16, and 17, although their directions are less diagnostic.
Rationale: The accepted loss of 0.173 is a useful improvement, but the remaining systematic errors are too large to ship. Revise within the same mechanism family. The most important implementation issue is that the opponent-relative conflict term uses the uncompressed effective-source difference, `sources_a - sources_b`. This bypasses the capacity bottleneck and lets raw cardinality dominate precisely where compression should matter, likely causing the extreme results in Experiments 1 and 6. Use the same smoothly compressed or capacity-normalized source representation in conflict arbitration, and reduce conflict gain or beta enough to avoid deterministic majority endorsement. Do not simply weaken source-count conflict globally, because Experiment 2 still needs a stronger configuration-sensitive anti-highest-cue effect; instead, let conflict depend on the compressed source advantage relative to the opponent's best-cue advantage.

Flatten sensitivity among nonzero gaps while preserving the sharp equality/nonzero distinction. The current formula forces separation to span from `separation_floor` to 1 across positive-gap percentiles, producing the excessive Experiment 11 gradient. Add a small positive-gap rank-gain parameter or narrow that span, while separately retaining or slightly strengthening equality evidence. Broaden the equality-evidence/prior distribution in both directions rather than raising its lower bound uniformly: some subjects need weak equality pooling and others strong pooling. This should preserve the pooled near-versus-exact effect in Experiment 22 while lowering Experiment 21 prevalence toward 0.64 and increasing its between-subject heterogeneity.

Finally, reduce the abrupt loss of dependence information between dyads and triads—e.g., modestly lower `load_individuation` or revise its smooth capacity normalization—so Experiment 15 approaches zero rather than showing a large negative triad-minus-dyad contrast. Compensate at tetrad and larger loads through stronger smooth saturation of the coalition signal and conflict term, not cardinality gates, to bring Experiments 18 and 20 toward chance. Preserve the currently successful exchangeability and coverage behavior in Experiments 4-5, 7-9, 12, 14, and 19.

**Outcome of your advice:** iter 1 candidate loss=0.1730 -> iter 2 candidate loss=0.1529 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The accepted revision is a meaningful improvement and remains faithful to probabilistic source individuation under adaptive compression. In particular, using compressed source codes in conflict reduced the excessive cardinality effects in Experiments 1 and 6, moved the large-coalition comparisons in Experiments 18 and 20 close to chance, lowered Experiment 21 prevalence while increasing its heterogeneity, and preserved the equality discontinuity in Experiment 22. Exchangeability is also well supported by the near-null results in Experiments 4, 5, 7, 8, and 9, with Experiment 19 remaining appropriately small. Nevertheless, the fit is not yet experiment-invariant. The largest residual failures are Experiment 6 (0.82 versus 0.44), Experiment 2 (0.013 versus -0.264), and Experiment 1 (-0.357 versus -0.122). Thus raw coalition size still has too much influence in unequal-cardinality conflicts, while configuration-sensitive source individuation has too little influence in the equal-cardinality conflict of Experiment 2. The revision also worsened several repetition/diversity results: exact duplicates remain too attractive in Experiment 10 (0.374 versus 0.172), and the model underestimates distributed or less-redundant coalition preferences in Experiments 12, 13, and 16. Experiment 17's cardinality profile is also too weak. Meanwhile, the nonzero-gap gradient remains excessive in Experiment 11 (0.140 versus 0.032), and Experiment 15 still shows an erroneous triad-minus-dyad decline (-0.111 versus 0.014). These joint errors indicate that the current compression adjustment suppresses count, dependence, and diversity through too much of the same pathway.
Rationale: Continue tuning within the accepted mechanism family; the previous advice was productive, so retain compressed conflict, broad subject heterogeneity, and the weak positive-gap rank term. Do not apply another global reduction of conflict gain, beta, or source curvature: that would likely repeat the pattern of improving Experiments 1, 6, 18, and 20 while further damaging Experiments 10, 12, 13, 16, and 17.

Instead, decouple smooth count leverage from configuration-sensitive individuation. A suitable minor normalization change is to represent conflict with two bounded components: a strongly compressed cardinality/source-magnitude difference and an independence-efficiency or individuation-residual difference, such as effective sources relative to the smooth cardinality-only expectation. Give the latter enough weight to distinguish equally sized coalitions in Experiment 2, while keeping the former weak enough that 3-versus-2 and many-versus-1 comparisons do not dominate Experiments 6 and 1. This preserves the arbiter's opponent-relative effective-source mechanism without introducing cardinality gates.

Likewise, separate the diversity contribution from the compressed count magnitude. An additive bounded coverage bonus, rather than only `source_code * diversity_factor`, would permit weaker count evidence alongside stronger discrimination between duplicate and distributed coalitions. Use a concave, zero-preserving transform of positive range/stratum coverage so that tiny nonzero separation quickly establishes distinct-source coverage but additional metric gap adds little. This should flatten Experiment 11, strengthen near-versus-exact and duplicate penalties in Experiments 10 and 16, and restore the diversity effects lost in Experiments 12 and 13. Preserve exact equality as a distinct probabilistic cue rather than raising equality evidence uniformly, since Experiment 21 prevalence is already slightly high at 0.74 and Experiment 22 is close.

Finally, modestly reduce the remaining load-dependent attenuation of inferred dependence. The accepted movement in this direction improved Experiment 15, but the contrast remains substantially negative. Dependence inference should persist when a dyad is embedded in a triad or tetrad; large-load attenuation should occur mainly in the subsequent smooth compression of the source code and diversity bonus. This can improve Experiments 15 and 16 without restoring the excessive five-source effects that were successfully removed in Experiments 18 and 20.

**Outcome of your advice:** iter 2 candidate loss=0.1529 -> iter 3 candidate loss=0.1609 -> the gate marked it REJECTED.

### Iteration 3
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed probabilistic source-individuation and adaptive-compression family, but it was rejected because the latest decoupling was not experiment-invariant. It substantially improved excessive cardinality leverage in Experiments 1 and 6 and modestly improved the nonzero-gap gradient in Experiment 11 and the equality contrast in Experiment 22. However, those gains came with systematic losses in the central diversity and redundancy phenomena: distributed or less-redundant coalitions are now too weak in Experiments 12 (0.572 versus 0.669), 13 (0.139 versus 0.324), 14 (0.527 versus 0.587), and 16 (0.519 versus 0.678); the cardinality profile in Experiment 17 also moved farther from the data. Experiment 21 was overcorrected from 0.74 to 0.40 rather than approaching 0.64. Large-load comparisons in Experiments 18 and 20 moved away from their near-chance fits, while Experiment 15 worsened despite the reduced load attenuation. Most importantly, Experiment 2 remains essentially at chance (0.005) rather than showing the observed anti-highest-cue effect (-0.264), and Experiment 3 retains the wrong sign. Thus the new independence-efficiency branch is not delivering the required opponent-relative configuration effect. Its exponential attenuation by total conflict load makes it negligible in the 3-versus-3 trials of Experiment 2, while the additive diversity bonus is itself strongly attenuated by coalition size and consequently suppresses the tetrad and distributed-coalition effects it was intended to preserve.
Rationale: Because the current candidate was rejected, the next revision should be made on top of the accepted iteration-2 base. My previous recommendation to combine a separate count-leverage parameter, an additive load-attenuated diversity bonus, and further reduced load-dependent individuation was unsuccessful and should not be repeated wholesale. In particular, do not reduce `load_individuation` further: Experiment 15 became more negative, not less.

Try a smaller, more localized in-family revision. First, attenuate additional-source magnitude only moderately—use an intermediate count-leverage range rather than the current strong 0.35–0.65 attenuation—while leaving the accepted diversity representation largely intact. The current step overshot Experiment 6 and Experiment 21 and erased effects in Experiments 12–17. If an additive coverage term is retained, remove the steep `exp(-(n-1)/capacity)` decay or replace it with a much gentler capacity normalization; that decay makes the nominally separate diversity channel almost vanish for triads and tetrads. Diversity should decline smoothly with represented load, not be effectively extinguished by cardinality.

Second, fix the ineffective conflict residual rather than merely increasing its gain. The current `individuation_retention = exp(-conflict_load/capacity)` nearly eliminates independence information in Experiment 2. Feed a bounded opponent-relative effective-source residual into conflict after source inference but before this severe total-load attenuation, normalized continuously by mean coalition size or compressed capacity. Its strength can instead be modulated by posterior individuation certainty or represented reliability coverage, so equal-cardinality coalitions with sharply different dependence posteriors remain distinguishable while narrow-range tetrads in Experiments 18 and 20 remain near chance. Check the orientation against the highest discriminating cue: the effective-source advantage must be able to outweigh the opponent's best cue, not simply add another weak near-zero term.

Third, flatten Experiment 11 directly through the contextual gap-rank component rather than through broad diversity suppression. Narrow `positive_gap_rank_gain` toward zero while preserving the discrete equality-versus-nonzero posterior contrast. A modest strengthening of equality dependence, with broad subject heterogeneity retained, could jointly reduce exact-duplicate choices in Experiment 10 and strengthen Experiments 16 and 22, but it should be balanced with only moderate count attenuation so Experiment 21 stays near 0.64 rather than returning to a homogeneous extreme. Preserve the successful exchangeability evidenced by Experiments 4, 5, 7, 8, and 9.

**Outcome of your advice:** iter 3 candidate loss=0.1609 -> iter 4 candidate loss=0.1628 -> the gate marked it REJECTED.

### Iteration 4
Verdict: regenerate
Interpretation: The candidate remains faithful to probabilistic source individuation under adaptive compression, but the new independence-efficiency conflict branch is not experiment-invariant and the candidate was correctly rejected. It improves several isolated metrics—Experiments 10, 12, 14, and 17 are close, and Experiment 6 reaches 0.40 versus 0.44—but substantially degrades the central cross-experiment pattern. Experiment 2 remains near chance (0.025 versus -0.264), so the added branch still does not generate the required anti-highest-cue choice. Experiment 3 likewise retains the wrong sign. At the same time, the branch greatly exaggerates individuation effects in equal-cardinality tetrads: Experiment 20 rises to 0.664 instead of 0.504, Experiment 22 rises to 0.818 instead of 0.682, and Experiment 21 becomes unanimous at 1.00 rather than showing the observed heterogeneous prevalence of 0.64. It also moves Experiment 18 away from chance. Thus the efficiency residual is weak where needed in Experiment 2 but dominant in several tetrad comparisons. Unequal-cardinality behavior is also inconsistent: Experiment 1 remains too anti-highest-cue (-0.383 versus -0.122), despite the good Experiment 6 result. Finally, narrowing positive_gap_rank_gain did not flatten Experiment 11; the contrast instead increased to 0.225 versus 0.032. This shows that the remaining metric-gap gradient is primarily entering through range/stratum coverage and ordinal quality, not through the explicit gap-rank term. Experiment 15 remains strongly negative (-0.148 versus 0.014), indicating excessive loss of the separated-coalition advantage from dyads to triads.
Rationale: Because this candidate was rejected, revise the accepted iteration-2 base rather than repeating the independence-efficiency addition. My previous recommendation to expose an independently weighted efficiency residual was rejected twice in practice: it failed to solve Experiment 2 and destabilized Experiments 18, 20, 21, and 22. Do not retain or simply retune that branch.

Use a more localized adjustment to the accepted opponent-relative conflict computation. Keep compressed effective-source magnitude as the relevant quantity, but separate its bounded contribution from the best-cue contribution and apply a smooth load/imbalance normalization. For example, attenuate source conflict continuously with mean coalition load and nominal cardinality imbalance, while retaining appreciable leverage for equal-sized triads. Reduce the positive best-cue offset in this arbitration so that a credible effective-source advantage can actually overturn the highest cue in Experiment 2. This should strengthen the required 3-versus-3 source configuration effect without restoring excessive 3-versus-2 leverage in Experiments 1 and 6 or the tetrad effects in Experiments 18 and 20. Verify the sign explicitly on Experiment 2 configurations; the current near-zero result suggests cancellation rather than insufficient response temperature.

Address Experiment 11 through the coverage representation, not by narrowing positive_gap_rank_gain again—the latter advice was rejected and the metric worsened. Replace metric-like dyadic range and standard-deviation coverage with a saturating ordinal-stratum occupancy measure: exact equality has zero distinct-stratum coverage, while small and large nonzero dyadic separations receive nearly similar coverage; larger coalitions can still benefit from occupying multiple strata. Preserve the accepted multiplicative diversity pathway. This is consistent with contextual ordinal coverage and should flatten Experiment 11 without erasing the successful distributed-coalition effects in Experiments 12–14.

Finally, remove the nominal-cardinality dependence of diversity retention that creates the persistent negative Experiment 15 contrast. A small in-family normalization change is to make retention depend on compressed represented load rather than directly on n, while preserving stronger smooth compression at tetrad and larger loads. Do not globally increase equality evidence or diversity sensitivity: the current extremes in Experiments 21 and 22 demonstrate that such changes would again collapse heterogeneity and overshoot tetrads. Preserve the accepted exchangeability behavior in Experiments 4, 5, 7, 8, and 9.

**Outcome of your advice:** iter 4 candidate loss=0.1628 -> iter 5 candidate loss=0.1606 -> the gate marked it REJECTED.

### Iteration 5 (most recent)
Verdict: regenerate
Interpretation: The current candidate is faithful to the prescribed source-individuation/compression family, but it was correctly rejected by the accept gate: loss 0.1606 remains worse than the accepted iteration-2 loss of 0.1529. The latest ordinal-occupancy and represented-load changes do not generalize. They preserve the desired exchangeability in Experiments 4, 5, 8, and 9 and keep Experiments 18–20 reasonably near chance. They also slightly improve Experiment 3, although its anti-highest-cue pattern is still too weak. However, the principal failures remain or worsen. Experiment 2 is still at chance (0.000 versus -0.264), so effective-source differences still cannot overturn the highest discriminating cue. Experiment 1 is again much too anti-highest-cue (-0.440 versus -0.122), showing excessive unequal-cardinality leverage, while Experiment 6 remains somewhat too favorable to the triad (0.52 versus 0.44). Exact duplicates are insufficiently penalized in Experiment 10 (0.277 versus 0.172), and the purportedly gap-invariant representation still produces an excessive Experiment 11 gradient (0.147 versus 0.032). Most damagingly, the new diversity-retention rule suppresses broad coalition effects across Experiments 12–16: Experiments 12, 13, 14, and 16 all substantially underestimate distributed or less-redundant preferences, and Experiment 15 becomes even more negative (-0.201 versus 0.014). Experiment 17 also remains too weak. Experiment 21 is still too homogeneous and high (0.78, variance 0.172 versus 0.64, variance 0.230), while Experiment 22 now overshoots (0.744 versus 0.682). Thus the model simultaneously overexpresses source count in unequal-cardinality choices and underexpresses configuration-sensitive individuation and diversity in equal-cardinality coalitions.
Rationale: Build the next revision on the unchanged accepted iteration-2 base. The last three rejected iterations show that neither an independently weighted efficiency-residual branch nor replacing the accepted coverage/retention machinery with coarse ordinal occupancy and cubic represented-load decay is productive. Do not repeat those changes.

A different localized issue is that the decision variable still contains the raw coalition contrast `signal_a - signal_b`, and then adds a separately normalized conflict term. Normalizing only the added conflict cannot control the source-count leverage already present in the raw signal. This likely explains why Experiment 1 remains excessively anti-highest-cue even when conflict is attenuated, while the separately bounded term has little effect in Experiment 2. Refactor the opponent-relative arbitration so the source-code part of the entire coalition contrast is bounded once rather than appearing both unnormalized in `signal_a - signal_b` and again in `relational_conflict`. A smooth relative contrast such as a scaled log ratio or `(source_a-source_b)/(source_a+source_b)` can preserve differences between equally sized coalitions; apply continuous load and cardinality-imbalance normalization to that unified source component. Combine diversity and weak ordinal quality as separate bounded modifiers. This remains the prescribed effective-source mechanism and is not the rejected independence-efficiency branch. Explicitly verify orientation on Experiment 2: the coalition with more inferred independent sources must receive a sufficiently negative highest-cue agreement score, whereas nominal 3-versus-2 or many-versus-1 differences should be attenuated.

Retain the accepted iteration-2 diversity pathway rather than the current three-stratum/cubic rule. The current rule jointly worsened Experiments 12–16 and made Experiment 15 more negative, so another represented-load-retention adjustment in the same direction is unwarranted. To address Experiment 11 more narrowly, reduce metric sensitivity inside the positive-gap coverage transform itself: make the first nonzero separation supply most of the coverage and let further range add only a small smooth increment. Also narrow `rank_reliance` toward zero, because ordinal quality remains another route by which high-gap dyads differ from low-gap dyads. Do not globally suppress diversity sensitivity; broad-coverage triads and tetrads already need stronger effects in Experiments 12–16.

Finally, preserve broad subject heterogeneity and avoid globally increasing equality evidence. The accepted base was already reasonably close on Experiment 21 and Experiment 22; the current overshoots show that stronger equality pooling would be counterproductive. The immediate priority is a single, unified opponent-relative normalization that reduces unequal-cardinality count dominance while amplifying genuine equal-cardinality source-configuration contrasts, without disturbing the successful positional, onset, spacing, closure, and large-load null results.

**Outcome of your advice:** iter 5 candidate loss=0.1606 -> CURRENT candidate loss=0.1587 -> the gate marked it REJECTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate remains faithful to probabilistic source individuation under adaptive compression, and it preserves the desired exchangeability: Experiments 4, 5, and 7\u20139 remain close to their positional, spacing, onset, and closure nulls. It also fits the large-load near-chance outcomes in Experiments 18 and 20 and retains the near-versus-exact effect in Experiment 22. The unified source normalization improved Experiment 1 and kept Experiment 6 much closer than the accepted iteration-2 base. Nevertheless, the candidate was rejected because its aggregate loss (0.1587) is still worse than the accepted base (0.1529). More importantly, the central source-configuration effect in Experiment 2 remains absent: 0.006 versus -0.264. This near-zero result has persisted across all six iterations, indicating that decision-level gain and normalization changes are not exposing the required source ordering. The current revision also suppresses broad diversity and redundancy effects too strongly: Experiments 12, 13, 14, and 16 are all substantially low, Experiment 17's cardinality profile is too weak, and Experiment 21 prevalence falls to 0.36 rather than 0.64. Experiment 15 retains a large erroneous decline from dyads to triads (-0.142 versus 0.014). Exact duplicates are still too attractive in Experiment 10, and the nonzero-gap gradient remains excessive in Experiment 11. Thus the model now compresses equal-cardinality configuration evidence along with the unwanted unequal-cardinality count evidence.",
  "verdict": "regenerate",
  "rationale": "Build on the unchanged accepted iteration-2 base. My previous recommendation to decompose and bound the entire source contrast produced a modest improvement over iterations 3\u20135 but was still rejected and severely weakened Experiments 12\u201317 and 21. Do not repeat that decision-level decomposition or further attenuate diversity/load signals globally.\n\nThe persistent Experiment 2 null suggests a different, earlier bottleneck: the current effective-source approximation, n/(1 + mean dependence), collapses the whole pairwise dependence structure into a single sum. Distinct dependence topologies can therefore receive nearly identical source counts, especially in equal-sized triads, even when exact-equality links are arranged differently. Replace this scalar pair-sum approximation with a smooth expected-cluster or effective-rank calculation over the same probabilistic common-source matrix. For example, a PSD dependence kernel can be converted to a continuous participation-ratio/effective-rank source count, calibrated to approach one for a fully common source and n for independent sources. This is an in-family source-count normalization, not a new mechanism or an independence-efficiency decision branch. It should preserve dyadic equality sensitivity while allowing the topology of equality and nonzero-separation links to distinguish the Experiment 2 coalitions. Before tuning response gains, explicitly verify that this inferred source count favors the anti-highest-cue option across the Experiment 2 configurations; otherwise no downstream temperature can recover the observed sign.\n\nRetain the accepted compression and multiplicative diversity machinery initially, allowing the revised source count to address Experiments 10 and 12\u201317 through inferred redundancy rather than a new additive bonus. Tune its spectral/cluster concentration moderately so equal-cardinality broad configurations remain discriminable while nominal 3-versus-2 and many-versus-one differences still saturate. Preserve the accepted large-load compression to protect Experiments 18 and 20.\n\nFinally, handle Experiment 21 through the distribution of existing subject-level parameters rather than another global equality increase. Widen equality-pooling heterogeneity around approximately the accepted median\u2014allowing both a weak-pooling tail and a strong-pooling tail\u2014and, if supported, correlate stronger equality pooling with stronger compression. That can increase duplicate penalties in Experiments 10 and 16 while avoiding unanimous tetrad preferences. Do not again narrow positive-gap rank gain alone or replace coverage with coarse ordinal occupancy; both directions were rejected previously and failed to resolve Experiment 11. Preserve the successful exchangeability parameters."
}
```

## Usage

```json
{
  "input_tokens": 47898,
  "output_tokens": 2778
}
```
