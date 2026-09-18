# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_5") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_5" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3
Prospective Capacity-Gated Comparison theory: before choosing, people perceptually register how many cues discriminate between the options and compare that comparison load with a stable cognitive capacity. If the discriminating set is manageable, they integrate all discriminating cues with an equal-weight majority rule. If it exceeds capacity, they compress the comparison into a validity-ranked lexicographic search and rely on the first discriminating cue. Strategy recruitment changes smoothly near the capacity boundary, reflecting variability in momentary load, but is determined by an observable property of the option pair rather than by experiment identity. Ordinary response noise is applied only after a strategy has produced its intended choice.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Capacity-Gated Comparison expects shape (2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.shape[0] != n_features:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {n_features}."
        )

    differences = a - b
    discriminating = np.flatnonzero(differences != 0)
    load = float(discriminating.size)
    if load == 0:
        return np.array([0.5, 0.5], dtype=float)

    capacity = float(parameters["capacity"])
    gate_slope = float(parameters["gate_slope"])

    # Probability that exhaustive equal-weight integration is recruited.
    # This stable logistic gate approaches one below capacity and zero above it.
    gate_logit = gate_slope * (capacity - load)
    if gate_logit >= 0.0:
        exp_neg = np.exp(-gate_logit)
        p_integrate = 1.0 / (1.0 + exp_neg)
    else:
        exp_pos = np.exp(gate_logit)
        p_integrate = exp_pos / (1.0 + exp_pos)

    # Manageable-load strategy: all discriminating cues receive equal weight.
    tally_margin = float(np.sum(np.sign(differences)))
    if tally_margin > 0.0:
        tally_scores = np.array([1.0, 0.0])
    elif tally_margin < 0.0:
        tally_scores = np.array([0.0, 1.0])
    else:
        tally_scores = np.array([0.0, 0.0])

    # Overload strategy: validity-ranked compression to the first diagnostic cue.
    cue_order = np.argsort(-validities, kind="stable")
    lex_scores = np.array([0.0, 0.0])
    for cue in cue_order:
        if differences[cue] > 0.0:
            lex_scores = np.array([1.0, 0.0])
            break
        if differences[cue] < 0.0:
            lex_scores = np.array([0.0, 1.0])
            break

    beta = float(parameters["beta"])

    def noisy_strategy_choice(scores):
        logits = beta * np.asarray(scores, dtype=float)
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs /= probs.sum()
        return probs

    # Response noise follows, rather than determines, strategy recruitment.
    p_tally = noisy_strategy_choice(tally_scores)
    p_lex = noisy_strategy_choice(lex_scores)
    p_core = p_integrate * p_tally + (1.0 - p_integrate) * p_lex

    epsilon = float(parameters["epsilon"])
    probs = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, 1.0)
    probs /= probs.sum()
    return probs

`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## THEORY 2 — pi_5
Load-Indexed Format Arbitration theory: people first register the observable number of cues that discriminate between the options and use it to recruit among three comparison formats. At low discriminating-cue load, they diffusely pool all diagnostic cues with equal weight, but the confidence attached to this normalized majority evidence varies stably across individuals. When the display contains a partially tied leading-validity tier, confidence in diffuse pooling is selectively attenuated at loads 3–4 because the tied priority structure interferes with forming a unitary low-load comparison. At intermediate load, people retrieve diagnostic cues sequentially from the display; retrieval accessibility combines communicated-validity tier with a recency gradient that is attenuated at very low loads and reaches full strength by overload loads. At high load, they compress the display to the highest-validity active tier, with a format-specific and person-specific evidence gain. The high-load transition distinguishes loads 6–7 from load 8, and its location varies coherently with recency strength. Observable validity-tier structure changes compressibility: a distinctive partially tied leading tier facilitates compression, whereas an entirely undifferentiated validity display delays it. Equal-validity tiers invoke a stable person-specific convention: some individuals use the first displayed diagnostic member of the tier, whereas others pool graded signed evidence across all diagnostic members of that tier. Under high load, whole-tier poolers obtain a modest coherence benefit when compressing a partially tied leading tier, strengthening—but not discretizing—the tier's graded mean evidence. These representations and their format-specific confidence generate signed evidence before ordinary response noise is applied.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Load-Indexed Format Arbitration expects shape (2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.shape[0] != n_features:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {n_features}."
        )

    signs = np.sign(a - b)
    active = np.flatnonzero(signs != 0.0)
    load = float(active.size)
    if active.size == 0:
        return np.array([0.5, 0.5], dtype=float)

    def stable_sigmoid(x):
        x = float(x)
        if x >= 0.0:
            z = np.exp(-x)
            return 1.0 / (1.0 + z)
        z = np.exp(x)
        return z / (1.0 + z)

    # Observable load gradually opens sequential retrieval and, later,
    # validity-tier compression. The low transition begins slightly earlier
    # so retrieval can contribute on three-cue conflicts.
    low_width = float(parameters["transition_width"])
    entered_retrieval = stable_sigmoid(
        (load - float(parameters["low_center"])) / low_width
    )

    global_best = float(np.max(validities))
    global_top_tier = np.flatnonzero(np.isclose(
        validities, global_best, rtol=1e-9, atol=1e-12
    ))
    partial_top_tie = 1.0 if 1 < global_top_tier.size < n_features else 0.0
    all_equal = 1.0 if global_top_tier.size == n_features else 0.0

    # Recency strength and compression timing are stably correlated within a
    # person. Strong-recency subjects retain retrieval longer; validity-led
    # subjects compress earlier. Distinct partial tiers facilitate chunking,
    # whereas an all-equal display supplies no validity boundary to compress.
    recency_strength = float(parameters["recency_strength"])
    recency_profile = (recency_strength - 5.8) / 1.0
    effective_high_center = (
        float(parameters["high_center"])
        + float(parameters["recency_compression_coupling"]) * recency_profile
        - partial_top_tie * float(parameters["partial_tie_acceleration"])
        + all_equal * float(parameters["all_equal_delay"])
    )
    compressed = stable_sigmoid(
        (load - effective_high_center)
        / float(parameters["high_transition_width"])
    )

    w_low = 1.0 - entered_retrieval
    w_high = entered_retrieval * compressed
    w_mid = entered_retrieval * (1.0 - compressed)

    # Low-load format: diffuse equal-weight comparison. Normalization makes
    # the evidence the diagnostic-cue majority margin rather than display size.
    low_evidence = float(np.mean(signs[active]))

    # A partially tied leading tier selectively lowers confidence in diffuse
    # pooling at loads 3-4, without changing strictly ordered low-load displays.
    low_gain_multiplier = 1.0
    if partial_top_tie > 0.0 and 3.0 <= load <= 4.0:
        low_gain_multiplier = float(parameters["partial_tie_low_attenuation"])

    # Construct communicated-validity tiers. Cues tied in communicated
    # validity have the same rank; display position remains separately visible.
    tier_rank = np.asarray(
        [np.sum(validities > validities[j] + 1e-12) for j in range(n_features)],
        dtype=float,
    )
    rank_scale = max(1.0, float(n_features - 1))
    normalized_rank = tier_rank / rank_scale
    if n_features > 1:
        display_position = np.arange(n_features, dtype=float) / float(n_features - 1)
    else:
        display_position = np.zeros(1, dtype=float)

    # Intermediate format: expected evidence from sequential retrieval.
    # Positional accessibility is mildly attenuated at three-cue load and
    # reaches its original strength by load six, preserving overload gradients.
    recency_load_scale = 0.85 + 0.15 * np.clip((load - 3.0) / 3.0, 0.0, 1.0)
    retrieval_log_access = (
        recency_load_scale * recency_strength * display_position[active]
        - float(parameters["validity_strength"]) * normalized_rank[active]
    )
    retrieval_log_access -= np.max(retrieval_log_access)
    retrieval_weights = np.exp(retrieval_log_access)
    retrieval_weights /= retrieval_weights.sum()
    mid_evidence = float(np.dot(retrieval_weights, signs[active]))

    # High-load format: compress to the best communicated-validity tier that
    # is diagnostic on this trial. Stable tie_rule=0 uses its first displayed
    # member; tie_rule=1 pools graded evidence over the complete active tier.
    best_validity = float(np.max(validities[active]))
    top_tier = active[np.isclose(
        validities[active], best_validity, rtol=1e-9, atol=1e-12
    )]

    tie_rule = int(parameters["tie_rule"])
    if tie_rule == 0:
        first_member = int(np.min(top_tier))
        high_evidence = float(signs[first_member])
    else:
        high_evidence = float(np.mean(signs[top_tier]))

    # At overload, coherent retrieval of an entire partially tied leading tier
    # gives whole-tier poolers a bounded confidence benefit. The underlying
    # evidence remains its graded mean rather than being collapsed to its sign.
    high_gain = float(parameters["high_evidence_gain"])
    if tie_rule == 1 and partial_top_tie > 0.0 and load >= 6.0:
        high_gain *= float(parameters["partial_tier_pooling_gain"])

    beta = float(parameters["beta"])

    def evidence_probabilities(evidence, gain=1.0):
        logits = np.array([
            0.5 * beta * float(gain) * float(evidence),
            -0.5 * beta * float(gain) * float(evidence),
        ], dtype=float)
        logits -= np.max(logits)
        p = np.exp(logits)
        p /= p.sum()
        return p

    # Format recruitment is part of representation, not response noise.
    # Stable format-specific gains calibrate confidence without weakening the
    # intermediate positional profile that is needed at overload loads.
    p_core = (
        w_low * evidence_probabilities(
            low_evidence,
            low_gain_multiplier * float(parameters["low_evidence_gain"])
        )
        + w_mid * evidence_probabilities(mid_evidence)
        + w_high * evidence_probabilities(high_evidence, high_gain)
    )

    # Ordinary lapses occur only after each format has generated evidence.
    epsilon = float(parameters["epsilon"])
    probs = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=float)
    return probs / total


`policy(probs) -> int`:
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / len(probs)
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=10):** [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 1, 1, 1]
  trial 2: A=[1, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 1, 0, 1, 1]
  trial 3: A=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  trial 4: A=[1, 0, 1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 1, 1, 1]
  trial 5: A=[1, 1, 1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  trial 6: A=[1, 0, 1, 1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  trial 7: A=[0, 0, 0, 0, 0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 1, 0, 1, 0, 1, 1]  B=[1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  trial 9: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0, 0, 0, 0, 0]
  trial 11: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 0, 1, 1, 0, 1, 0, 0, 0, 0]

**Rationale:** This design isolates the competing theory's late high-load format transition while placing all trials above the advocated theory's capacity boundary. There are two exemplars each at discriminating-cue loads 6, 7, and 8, followed by exact A/B reversals. In every unreversed pair, the highest-validity active expert favors A, whereas the most recent active experts favor B. At loads 6 and 7 the equal-weight tally also favors B; at load 8 it is tied. Prospective Capacity-Gated Comparison should already be predominantly lexicographic: it may show some residual B-majority influence at load 6, but should strongly favor A by load 7 and then change very little from load 7 to load 8. Across its parameter ranges, its approximate unreversed prediction is P(A)=0.65-0.89 at load 6, 0.77-0.89 at load 7, and 0.80-0.89 at load 8. Load-Indexed Format Arbitration instead assigns substantial weight to recency-dominated sequential retrieval at loads 6-7, yielding B-directed evidence because the late diagnostic cues all favor B. Its narrow high-load transition then increasingly compresses the display to the uniquely highest-validity active cue, which favors A. It therefore predicts a substantially steeper increase in A choice from load 6 through load 8, with especially diagnostic disagreement at load 6 and a larger load-7-to-load-8 change than the advocated theory. The two exemplars at each load alter active positions while preserving this early-validity-versus-late-recency conflict. Exact reversals require complementary choice profiles and control response-side bias. Twelve unique pairs repeated eight times yield 96 trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
The metric is the proportion of choices agreeing with the highest-validity discriminating cue on load-6 trials. Capacity-Gated Comparison predicts high agreement because six cues exceed its capacity boundary, making validity-ranked compression predominant. Format Arbitration predicts low agreement because load 6 is still dominated by sequential retrieval, and the late, highly accessible cues favor the opposite option. Restricting the metric to load 6 avoids the competitor's heterogeneous load-7 transition while retaining 32 observations per subject. Exact A/B reversals balance response-side preferences, making the subject-level proportion comparatively stable.

Source:
def metric(data: pd.DataFrame) -> float:
    agreements = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        active = np.flatnonzero(diff != 0.0)

        # Restrict to the six-discriminating-cue trials, where the theories'
        # predictions are maximally separated and the competitor has not yet
        # entered its person-variable high-load transition.
        if active.size != 6:
            continue

        # Validities strictly decrease with display position, so the first
        # active feature is the validity-ranked lexicographic cue.
        first = int(active[0])
        lex_response = 0 if diff[first] > 0.0 else 1
        agreements.append(float(int(row["response"]) == lex_response))

    if len(agreements) == 0:
        return float("nan")
    return float(np.mean(agreements))

### RESULTS
- Predicted under pi_3 (simulated): 0.7987 (var=0.0067)
- Predicted under pi_5 (simulated): 0.0850 (var=0.0021)
- Observed on real data: 0.1175 (var=0.0081)

## EXPERIMENT 2 (proposed by pi_5)

### DESIGN
**Validities (n_features=8):** [0.91, 0.91, 0.82, 0.75, 0.69, 0.63, 0.57, 0.52]

**Trial pairs (n=24):**
  trial 1: A=[1, 0, 0, 0, 1, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  trial 4: A=[0, 1, 0, 1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[0, 1, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  trial 6: A=[0, 1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0]
  trial 7: A=[1, 0, 0, 1, 0, 1, 0, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[1, 0, 0, 0, 0, 1, 0, 1]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  trial 9: A=[1, 0, 0, 1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  trial 10: A=[0, 1, 1, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[0, 1, 0, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  trial 12: A=[0, 1, 1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0]
  trial 13: A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 1]
  trial 14: A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 1]
  trial 15: A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 1, 0, 0, 0]
  trial 16: A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0, 1, 0]
  trial 17: A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 1, 0]
  trial 18: A=[0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0, 0, 0]
  trial 19: A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 1, 0, 1, 0, 1]
  trial 20: A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 0, 1]
  trial 21: A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 1, 0, 1, 0, 0]
  trial 22: A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 1, 1, 0]
  trial 23: A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1, 1, 0]
  trial 24: A=[0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0, 1, 0, 0]

**Rationale:** This design creates matched low-load confidence and recency dissociations while making Prospective Capacity-Gated Comparison exactly invariant. Experts 1 and 2 form a distinctive partially tied leading-validity tier. Each unreversed trial has either three or four discriminating cues, the highest-validity active cue favors A, and the equal-weight majority also favors A. Consequently, both branches of the competing theory select A on every unreversed trial, predicting the same P(A) throughout (approximately 0.80-0.89 over its parameter ranges), regardless of load, unanimity, or the losing cue's display position. Within each fixed active-cue set, Load-Indexed Format Arbitration instead predicts a graded profile. The unanimous condition has evidence +1 in every format and should produce the strongest A preference. Moving to a narrow A majority reduces normalized diffuse evidence to 1/3 at load 3 or 1/2 at load 4; moreover, the partially tied leading tier selectively attenuates low-format confidence at these loads. Among narrow-majority trials, placing the sole B-favoring cue late rather than at an intermediate position should further reduce A choice because sequential retrieval has a strong recency gradient. Thus the critical predicted ordering is unanimous > intermediate-position loser > latest-position loser, especially at load 4 where retrieval recruitment is greater. The competing theory predicts no such within-set ordering because tally direction and lexicographic direction are unchanged. Exact A/B reversals provide response-side balance. There are 24 unique pairs, yielding four repetitions each for 96 trials.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



### METRIC
Rationale:
The metric contrasts majority-winner choice rates on unanimous trials with otherwise majority-consistent trials whose sole dissenting cue is the latest displayed active cue. Exact A/B reversals are reoriented toward the majority winner, balancing response side and reducing sensitivity to stable A/B response biases. The advocated theory predicts a large positive contrast: unanimous evidence is strong in every format, whereas a late dissenting cue can dominate recency-weighted sequential retrieval, particularly at load four. The competing theory predicts approximately zero because both its equal-weight tally and validity-ranked lexicographic strategy select the same majority winner in both conditions. Pooling all load-3 and load-4 instances supplies 32 observations per condition per subject, keeping the subject-level contrast comparatively stable.

Source:
def metric(data: pd.DataFrame) -> float:
    unanimous = []
    late_loser = []

    for row in data.itertuples(index=False):
        a = np.asarray(row.option_a_ratings, dtype=float)
        b = np.asarray(row.option_b_ratings, dtype=float)
        signs = np.sign(a - b)
        active = np.flatnonzero(signs != 0.0)
        if active.size == 0:
            continue

        margin = float(np.sum(signs[active]))
        if margin == 0.0:
            continue

        # Reorient every trial toward its equal-weight-majority winner, thereby
        # combining A/B-reversed pairs and eliminating response-side direction.
        winner_sign = 1.0 if margin > 0.0 else -1.0
        relative_signs = signs[active] * winner_sign
        winner_response = 0 if winner_sign > 0.0 else 1
        followed_winner = float(int(row.response) == winner_response)

        if np.all(relative_signs > 0.0):
            unanimous.append(followed_winner)
        elif np.sum(relative_signs < 0.0) == 1:
            losing_active_position = int(active[np.flatnonzero(relative_signs < 0.0)[0]])
            if losing_active_position == int(np.max(active)):
                late_loser.append(followed_winner)

    if len(unanimous) == 0 or len(late_loser) == 0:
        return 0.0

    return float(np.mean(unanimous) - np.mean(late_loser))

### RESULTS
- Predicted under pi_3 (simulated): -0.0169 (var=0.0083)
- Predicted under pi_5 (simulated): 0.3750 (var=0.0124)
- Observed on real data: 0.7750 (var=0.0279)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Contrast top-cue adherence between tally-congruent and
    # tally-conflicting trials, matching on absolute tally margin.
    cells = {(d, c): [] for d in (2, 4, 6) for c in (-1, 1)}

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        response = int(row["response"])

        # Feature 0 is the highest-validity cue in this experiment.
        if a[0] > b[0]:
            top_choice = 0
            top_relative_margin = int(np.sum(a > b) - np.sum(b > a))
        elif b[0] > a[0]:
            top_choice = 1
            top_relative_margin = int(np.sum(b > a) - np.sum(a > b))
        else:
            continue

        d = abs(top_relative_margin)
        if d not in (2, 4, 6) or top_relative_margin == 0:
            continue

        congruence = 1 if top_relative_margin > 0 else -1
        top_adherence = 1.0 if response == top_choice else -1.0
        cells[(d, congruence)].append(top_adherence)

    matched_contrasts = []
    for d in (2, 4, 6):
        congruent = cells[(d, 1)]
        conflicting = cells[(d, -1)]
        if len(congruent) == 0 or len(conflicting) == 0:
            continue
        # Division by two keeps the metric in [-1, 1].
        matched_contrasts.append(
            0.5 * (float(np.mean(congruent)) - float(np.mean(conflicting)))
        )

    if len(matched_contrasts) == 0:
        return float("nan")
    return float(np.mean(matched_contrasts))
```

**Observed (real) value:** -0.0300 (var=0.0056)
**Predicted under pi_3:** -0.0028 (var=0.0068)
**Predicted under pi_5:** 0.0367 (var=0.0093)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    scores = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        d = a - b

        # +1 denotes an A prediction and -1 a B prediction.
        tally_margin = np.sum(np.sign(d))
        if tally_margin == 0:
            continue
        tally_direction = 1.0 if tally_margin > 0 else -1.0

        # Validities decrease with feature index in this experiment.
        discriminating = np.flatnonzero(d != 0)
        if discriminating.size == 0:
            continue
        first = int(discriminating[0])
        ttb_direction = 1.0 if d[first] > 0 else -1.0

        # Agreement trials cannot distinguish the theories.
        if tally_direction == ttb_direction:
            continue

        observed_direction = 1.0 if int(row["response"]) == 0 else -1.0
        scores.append(observed_direction * tally_direction)

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** 0.6683 (var=0.0367)
**Predicted under pi_3:** 0.6600 (var=0.0188)
**Predicted under pi_5:** 0.6467 (var=0.0132)

### Experiment 5
**Design**
  A=[0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 0, 1, 0, 1]  B=[1, 0, 0, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Return a parity-matched low-to-high load crossover index."""
    if data is None or len(data) == 0:
        return float("nan")

    loads = []
    majority_consistent = []

    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"],
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        diff = a - b
        loads.append(int(np.count_nonzero(diff)))

        tally_margin = float(np.sum(np.sign(diff)))
        if tally_margin > 0:
            majority_response = 0
        elif tally_margin < 0:
            majority_response = 1
        else:
            majority_response = -1

        majority_consistent.append(
            float(int(response) == majority_response)
            if majority_response >= 0 else np.nan
        )

    work = pd.DataFrame({
        "load": np.asarray(loads, dtype=int),
        "majority_consistent": np.asarray(majority_consistent, dtype=float),
    })

    p = {}
    for load in (3, 4, 6, 7, 8):
        values = work.loc[
            work["load"] == load, "majority_consistent"
        ].dropna()
        if len(values) == 0:
            return float("nan")
        p[load] = float(values.mean())

    odd_crossover = p[3] - p[7]
    even_crossover = p[4] - 0.5 * (p[6] + p[8])
    return float(0.5 * (odd_crossover + even_crossover))

```

**Observed (real) value:** 0.3612 (var=0.0106)
**Predicted under pi_3:** 0.6512 (var=0.0114)
**Predicted under pi_5:** 0.5350 (var=0.0071)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0, 0, 1, 0]  B=[1, 1, 0, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Rate of validity-leading-cue-consistent choices on overload trials."""
    scores = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        differences = a - b
        discriminating = np.flatnonzero(differences != 0)

        # Focus on the 6- and 8-cue overload conditions.
        if discriminating.size < 6:
            continue

        # Validities are strictly descending by feature index, so the first
        # discriminating index is the highest-validity diagnostic cue.
        leading_cue = int(discriminating[0])
        lex_response = 0 if differences[leading_cue] > 0 else 1
        scores.append(float(int(row["response"]) == lex_response))

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** 0.8756 (var=0.0078)
**Predicted under pi_3:** 0.8291 (var=0.0019)
**Predicted under pi_5:** 0.8884 (var=0.0019)

### Experiment 7
**Design**
  A=[1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 1]  B=[0, 0, 0, 1, 1, 1, 0]
  A=[1, 1, 0, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1, 1]  B=[1, 1, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0, 1]  B=[1, 0, 0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def subject_classification(df):
        aligned = []
        for _, row in df.iterrows():
            a = np.asarray(row['option_a_ratings'], dtype=float)
            b = np.asarray(row['option_b_ratings'], dtype=float)
            d = np.sign(a - b)

            # Select trials where Experts 2 and 3 agree with each other but
            # oppose Expert 1: a 2-to-1 conflict within the top-validity tier.
            if d.size >= 3 and d[0] != 0 and d[1] == d[2] and d[1] == -d[0]:
                response = int(row['response'])
                chose_a = 1.0 if response == 0 else -1.0
                aligned.append(float(chose_a * d[0] > 0))

        if len(aligned) == 0:
            return np.nan
        rate = float(np.mean(aligned))
        if rate > 0.5:
            return 1.0
        if rate < 0.5:
            return 0.0
        return 0.5

    values = []
    for _, subj_df in data.groupby('subject_id', sort=False):
        value = subject_classification(subj_df)
        if np.isfinite(value):
            values.append(value)

    if len(values) == 0:
        return float('nan')
    return float(np.mean(values))
```

**Observed (real) value:** 0.5000 (var=0.2000)
**Predicted under pi_3:** 1.0000 (var=0.0000)
**Predicted under pi_5:** 0.5800 (var=0.2436)

### Experiment 8
**Design**
  A=[0, 0, 1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 0, 0, 0]  B=[0, 1, 0, 0, 1, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0, 1, 1]  B=[0, 1, 0, 1, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Contrast majority-choice rates across the four positions of the
    # second cue opposing the majority. Coefficients sum to zero, so any
    # profile that is flat across positions has expected metric zero.
    weights = {1: -0.24, 3: -0.05, 5: 0.10, 7: 0.19}
    outcomes = {1: [], 3: [], 5: [], 7: []}

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        signs = np.sign(a - b)
        active = np.flatnonzero(signs != 0)
        if active.size == 0:
            continue

        majority_sign = np.sign(np.sum(signs[active]))
        if majority_sign == 0:
            continue

        # Expert 1 (array index 0) is the highest-validity cue. The target
        # position is the other active cue pointing in the same direction.
        top_sign = signs[0]
        opposing = [int(j) for j in active if j != 0 and signs[j] == top_sign]
        if len(opposing) != 1 or opposing[0] not in outcomes:
            continue

        response = int(row["response"])
        chose_majority = ((response == 0 and majority_sign > 0) or
                          (response == 1 and majority_sign < 0))
        outcomes[opposing[0]].append(float(chose_majority))

    if any(len(outcomes[j]) == 0 for j in outcomes):
        return float("nan")

    rates = {j: float(np.mean(outcomes[j])) for j in outcomes}
    return float(sum(weights[j] * rates[j] for j in weights))

```

**Observed (real) value:** -0.1132 (var=0.0031)
**Predicted under pi_3:** 0.0009 (var=0.0007)
**Predicted under pi_5:** -0.1079 (var=0.0009)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all eight experiments, pi_5 is decisively more task-invariant. Its mean absolute point-estimate error is about 0.099, versus about 0.307 for pi_3. Experiment 1 is especially diagnostic: observed leading-cue agreement at load 6 was 0.1175, close to pi_5's 0.0850 and radically inconsistent with pi_3's 0.7987. Experiment 2 likewise rules out pi_3 structurally: because its tally and lexicographic branches choose the same option in every condition, it predicts essentially no unanimous-versus-late-dissenter contrast (-0.0169), whereas the observed contrast is 0.7750. Pi_5 predicts the correct positive effect (0.3750), although it substantially underestimates its magnitude. The broader record reinforces this conclusion. Pi_5 is better in Experiments 5, 6, 7, and 8, including an excellent account of the positional effect in Experiment 8 (-0.1079 predicted versus -0.1132 observed) and the heterogeneous tied-tier behavior in Experiment 7 (0.5800 versus 0.5000). Pi_3 is only modestly closer in Experiment 3, where both are near the observed null, and Experiment 4, where both accurately predict majority preference. Pi_5 is not complete: it underpredicts the very large confidence disruption caused by a late dissenting cue in Experiment 2, overpredicts the load crossover in Experiment 5, and often produces less between-subject variance than observed, notably in Experiments 1, 2, and 4. Nevertheless, it captures the crucial validity, recency, load, position, and tie-heterogeneity phenomena that pi_3 cannot. Pi_3's failures cannot be repaired by parameter ranges alone. Raising its capacity could let tallying explain Experiment 1, but would undermine overload leading-cue adherence in Experiment 6; more decisively, no setting of its capacity gate can generate Experiment 2's contrast because both of its strategies make identical binary predictions there. Thus its core two-format theory lacks the representational sensitivity demanded by the data.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace pi_3 with a genuinely different competitor to pi_5: a Conflict-Triggered Attentional Sampling theory. Rather than selecting discrete tally, retrieval, or compression formats at fixed load transitions, subjects should use one continuous stochastic comparison process. Diagnostic cues enter an attentional race whose accessibility jointly reflects communicated validity, display position/recency, and local conflict structure. Sampled signed evidence accumulates toward a stopping bound; unanimous evidence promotes rapid, confident termination, while a newly encountered dissenting cue reopens or prolongs sampling and receives a transient attentional boost. Same-sign cues encountered close together should reinforce one another as a coherent coalition, allowing the late opposing block in Experiment 1 to overwhelm the leading cue, while leading validity can dominate high-load displays without such a coherent recency pattern, as in Experiment 6. Load should affect behavior indirectly by increasing sampling opportunities, conflict, and the probability of reaching the bound before exhaustive inspection\u2014not through hard-coded load-specific formats. Stable individual differences should govern validity sensitivity, recency bias, stopping threshold, conflict-trigger strength, coalition reinforcement, and treatment of equal-validity tiers; the latter can range continuously from first-member selection to whole-tier pooling, accounting for Experiment 7's heterogeneity. This theory should generate the strong unanimous-versus-late-dissenter contrast in Experiment 2, the gradual crossover in Experiment 5, and the positional gradient in Experiment 8 from the same sampling dynamics, while retaining ordinary post-decision lapse noise. It would be a stronger and more parsimonious challenger to pi_5 because it replaces pi_5's numerous load- and tie-specific clauses with a single process and makes novel predictions from cue order, conflict timing, and sign clustering."
}
```

## Usage

```json
{
  "input_tokens": 14547,
  "output_tokens": 2284
}
```
