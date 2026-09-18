# experiment_attempt_00

## System Prompt

You are a renowned cognitive scientist designing an experiment in the Decision Making (Binary Features) domain.

Your goal is to be an adversarial collaborator: propose a design whose outcomes would be predicted by your advocated theory but NOT by the competing theory. Both are provided below.

A useful proposal targets a *quantitative* dissociation between the two theories — how they respond differently to specific stimuli in addition to differences in overall performance.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

A multi-attribute decision-making experiment. On each trial the subject sees two options (A, B), each described by `n_features` integer expert ratings (`n_features` is set by the length of `validities` you propose). Choose `validities` — one per feature, each in [0.5, 1.0], order free — to fix each expert's advertised accuracy; subjects are told these values up front. Then choose `trial_a_ratings/trial_b_ratings` (each rating value in [0, 1]) so that the intended decision-making theories make distinguishable predictions: avoid degenerate pairs where every theory agrees, and prefer pairs that maximally dissociate the candidate theories. Validities and the trial ratings together define the design; they are fixed across all trials in this experiment. No trial-by-trial correctness feedback. The total number of trials per subject is held at roughly 96: each unique pair is repeated K = max(1, 96 // n_unique_pairs) times in an independently-randomized order per subject.

Subjects see the following instructions:
In this experiment you will repeatedly choose between two fictitious products, A and B. On every trial you will see `n_features` expert ratings for each product (the number of experts is fixed across all trials and is set by the length of `validities`).

Each rating is an integer in [0, 1]. The ratings are displayed as a horizontal filled bar with the numeric value (e.g. "0/1") shown next to it. Higher = more positive.

The same experts (in the same order) provide ratings for both products on every trial. Each expert's accuracy (their validity expressed as a percentage, e.g. "Expert 1 (80%)") is shown next to their rating on every trial AND is also listed up front in an "Expert accuracies" panel.

On each trial, decide which product is of higher quality and press A for product A or B for product B. There is no time limit and no feedback. Note that for the first ~`min_rt_ms` of each trial the answer prompt is hidden and the keys are locked, so subjects first see the full ratings and can answer once the A / B prompt appears — design pairs that actually require comparing the ratings.

Total trials per subject is roughly `MAX_TRIALS`: each unique pair you propose is repeated `K = max(1, MAX_TRIALS // n_unique_pairs)` times in an independently-randomized order per subject.

## ADVOCATED THEORY
**Description:** Categorical Source-Budget Inference proposes that people recode instructed expert reliabilities into a small number of latent information-source categories. Category membership is inferred locally: experts with exactly or nearly matching represented reliabilities are likely to be treated as exchangeable observations from one source, whereas this coassignment declines sharply outside a narrow subject-specific neighborhood. Within each option-supporting coalition, members assigned to one source draw from a shared evidence budget whose value grows as a strongly concave power of cluster membership. Exact reliability matches additionally invoke a categorical equality atom, producing an almost flat shared budget without changing how nonzero reliability gaps are partitioned. Budgets belonging to distinct inferred sources are added. A duplicate pair is therefore redundant in a dyad, an unrelated outsider adds a separate budget without changing that duplicate cost, and two duplicate clusters consume two separate budgets. The total is then subjected to generic nominal-coalition normalization and a separately calibrated saturating release that is concentrated between two and three effective sources. Instructed validity differences have little direct influence after strong compression. Stable, exchangeable, zero-centered cue salience permits individual configuration preferences but creates no population-level serial-position mechanism. With no correctness feedback, source assignments and cue salience remain fixed across trials.

**Parameters:**
- validities: validities
- cue_salience_profile: [(-1.0, 1.0)] * n_features
- direct_validity_compression: [0.08, 0.14]
- validity_reliance: [0.006, 0.024]
- salience_strength: [0.18, 0.40]
- category_width: [0.008, 0.018]
- category_boundary_shift: [-0.30, 0.30]
- within_source_curvature: [0.08, 0.22]
- equality_source_curvature: [0.00, 0.05]
- coalition_normalization: [0.88, 0.98]
- source_count_curvature: [0.18, 0.26]
- source_count_midpoint: [2.45, 2.65]
- source_count_slope: [7.0, 11.0]
- beta: [1.45, 1.85]
- epsilon: [0.03, 0.08]

**`predict source code`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Categorical Source-Budget Inference expects shape "
            f"(2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = int(stim.shape[1])

    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    raw_salience = np.asarray(
        parameters["cue_salience_profile"], dtype=np.float64
    )
    if raw_salience.ndim != 1 or raw_salience.size != n_features:
        raise ValueError(
            f"cue_salience_profile length {raw_salience.size} != "
            f"n_features {n_features}."
        )

    direct_validity_compression = float(
        parameters["direct_validity_compression"]
    )
    validity_reliance = float(parameters["validity_reliance"])
    salience_strength = float(parameters["salience_strength"])
    category_width = float(parameters["category_width"])
    category_boundary_shift = float(parameters["category_boundary_shift"])
    within_source_curvature = float(parameters["within_source_curvature"])
    equality_source_curvature = float(
        parameters["equality_source_curvature"]
    )
    coalition_normalization = float(parameters["coalition_normalization"])
    source_count_curvature = float(parameters["source_count_curvature"])
    source_count_midpoint = float(parameters["source_count_midpoint"])
    source_count_slope = float(parameters["source_count_slope"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    v = np.clip(validities, 0.500001, 0.999999)

    # Reliability has only a weak direct effect. Diagnosticity differences
    # are strongly compressed around a fixed .75 reference so the scale is
    # comparable across experiments.
    diagnosticity = np.log(v / (1.0 - v))
    reference = np.log(0.75 / 0.25)
    represented_direct = np.power(
        np.maximum(diagnosticity, 1e-12), direct_validity_compression
    )
    represented_direct /= max(
        float(reference ** direct_validity_compression), 1e-12
    )
    reliability_weight = (
        (1.0 - validity_reliance) * np.ones(n_features, dtype=np.float64)
        + validity_reliance * represented_direct
    )

    # Idiosyncratic salience is exchangeable across identities and explicitly
    # centered, so it cannot generate a common onset, recency, or closure rule.
    centered_salience = raw_salience - float(np.mean(raw_salience))
    attention = np.exp(
        np.clip(salience_strength * centered_salience, -10.0, 10.0)
    )
    attention /= max(float(np.mean(attention)), 1e-12)

    cue_weights = reliability_weight * attention
    cue_weights /= max(float(np.mean(cue_weights)), 1e-12)

    # Category uncertainty controls a narrow local coassignment radius.
    # Exact matches are always coassigned; nonidentical reliabilities are
    # coassigned only inside a sharply localized subject-specific region.
    coassignment_radius = category_width * np.exp(category_boundary_shift)

    difference = a - b
    a_support = np.flatnonzero(difference > 0.0)
    b_support = np.flatnonzero(difference < 0.0)

    if a_support.size == 0 and b_support.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    def coalition_signal(indices):
        idx = np.asarray(indices, dtype=int)
        n = int(idx.size)
        if n == 0:
            return 0.0

        # Infer a local source partition by joining only reliability-near
        # coalition members. Connected components are exchangeable in expert
        # identity and produce no repeated global boundary artifacts.
        parent = np.arange(n, dtype=np.int64)

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = int(parent[i])
            return i

        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_j] = root_i

        coalition_v = v[idx]
        for i in range(n):
            for j in range(i + 1, n):
                if abs(float(coalition_v[i] - coalition_v[j])) <= coassignment_radius:
                    union(i, j)

        labels = np.asarray([find(i) for i in range(n)], dtype=np.int64)
        unique_labels = np.unique(labels)
        source_budget_total = 0.0

        # Experts assigned to one latent source share a cluster-level budget.
        # Exact reliability matches invoke a categorical equality atom with
        # near-flat membership growth. Nonidentical local clusters retain the
        # accepted model's original pooling curvature.
        for label in unique_labels:
            local_members = np.flatnonzero(labels == label)
            members = idx[local_members]
            membership = float(members.size)
            source_quality = float(np.mean(cue_weights[members]))
            local_validities = coalition_v[local_members]
            exact_match = (
                members.size > 1
                and float(np.max(local_validities) - np.min(local_validities))
                <= 1e-12
            )
            cluster_curvature = (
                equality_source_curvature
                if exact_match
                else within_source_curvature
            )
            source_budget_total += source_quality * (
                membership ** cluster_curvature
            )

        effective_source_count = float(unique_labels.size)

        # Source budgets already add independent-source evidence. The extra
        # count calibration is therefore bounded and releases primarily when
        # effective source count crosses from two to three, rather than
        # multiplying evidence by an unbounded second power of source count.
        release = 1.0 / (
            1.0
            + np.exp(
                -np.clip(
                    source_count_slope
                    * (effective_source_count - source_count_midpoint),
                    -50.0,
                    50.0,
                )
            )
        )
        count_factor = np.exp(source_count_curvature * release)
        nominal_normalizer = float(n) ** coalition_normalization
        signal = source_budget_total * count_factor / max(
            nominal_normalizer, 1e-12
        )
        return float(max(signal, 0.0))

    evidence_a = coalition_signal(a_support)
    evidence_b = coalition_signal(b_support)
    net_a = evidence_a - evidence_b

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
```

**`policy source code`:**
```python
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.maximum(probs, 0.0)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** Reliability-Space Credit Sharing proposes that people interpret agreement among experts through an inferred dependence network defined entirely in represented reliability space. Advertised validities have two related but distinct cognitive representations: a strongly compressed representation determines each cue's direct evidential weight, while a separately calibrated fixed-anchor representation determines inferred dependence. Within any option-supporting coalition, every expert shares informational credit with similarly represented experts according to a smooth, sharply localized similarity kernel. An expert surrounded by reliability-near neighbors therefore contributes less independent evidence, whereas reliability-distant experts contribute nearly undiluted evidence. The steep but continuous kernel concentrates redundancy around extremely similar reliability representations, producing a strong equality penalty without a broad dyadic reliability-gap gradient. Credit sharing applies at every coalition size, without an equality detector, dyadic/triadic switch, spatial grouping rule, or duplicate-pair-in-context operation. Because additional coalition members create multiple possible similarity relations, local dependence can accumulate nonlinearly in larger coalitions even when its effect on a single nonidentical pair is weak. Effective evidence is additionally subject to generic sublinear count growth. Stable subject-level dependence sensitivity, overlap curvature, compressed validity reliance, and exchangeable zero-centered cue salience generate individual differences without population-level serial-position effects. Choices arise from response sensitivity plus an independent lapse process, and representations remain fixed because no outcome feedback is provided.

**Parameters:**
- validities: validities
- cue_salience_profile: [(-1.0, 1.0)] * n_features
- validity_compression: [0.30, 0.34]
- direct_validity_compression: [0.08, 0.14]
- validity_reliance: [0.008, 0.025]
- salience_strength: [0.12, 0.28]
- coalition_saturation: [0.64, 0.72]
- dependence_sensitivity: [2.90, 3.30]
- similarity_bandwidth: [0.0045, 0.0065]
- overlap_curvature: [0.88, 0.96]
- beta: [1.12, 1.38]
- epsilon: [0.03, 0.08]

**`predict source code`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Reliability-Space Credit Sharing expects shape (2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = int(stim.shape[1])

    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    raw_salience = np.asarray(
        parameters["cue_salience_profile"], dtype=np.float64
    )
    if raw_salience.ndim != 1 or raw_salience.size != n_features:
        raise ValueError(
            f"cue_salience_profile length {raw_salience.size} != n_features {n_features}."
        )

    validity_compression = float(parameters["validity_compression"])
    direct_validity_compression = float(
        parameters["direct_validity_compression"]
    )
    validity_reliance = float(parameters["validity_reliance"])
    salience_strength = float(parameters["salience_strength"])
    coalition_saturation = float(parameters["coalition_saturation"])
    dependence_sensitivity = float(parameters["dependence_sensitivity"])
    similarity_bandwidth = float(parameters["similarity_bandwidth"])
    overlap_curvature = float(parameters["overlap_curvature"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Dependence and direct cue weighting use separately compressed versions
    # of the same instructed validity. Both retain the fixed 0.75 reference,
    # so their scales remain invariant across experiments.
    v = np.clip(validities, 0.500001, 0.999999)
    log_diagnosticity = np.log(v / (1.0 - v))
    reference_diagnosticity = np.log(0.75 / 0.25)

    represented = np.power(
        np.maximum(log_diagnosticity, 1e-12), validity_compression
    )
    represented /= max(
        float(reference_diagnosticity ** validity_compression), 1e-12
    )

    direct_represented = np.power(
        np.maximum(log_diagnosticity, 1e-12),
        direct_validity_compression,
    )
    direct_represented /= max(
        float(reference_diagnosticity ** direct_validity_compression),
        1e-12,
    )

    reliability_weight = (
        (1.0 - validity_reliance) * np.ones(n_features, dtype=np.float64)
        + validity_reliance * direct_represented
    )

    # Subject-specific salience is exchangeable over cue identities and is
    # centered within the display, precluding a population onset or recency
    # advantage while allowing stable individual configuration preferences.
    centered_salience = raw_salience - float(np.mean(raw_salience))
    attention = np.exp(
        np.clip(salience_strength * centered_salience, -10.0, 10.0)
    )
    attention /= max(float(np.mean(attention)), 1e-12)

    cue_weights = reliability_weight * attention
    cue_weights /= max(float(np.mean(cue_weights)), 1e-12)

    difference = a - b
    a_support = np.flatnonzero(difference > 0.0)
    b_support = np.flatnonzero(difference < 0.0)

    if a_support.size == 0 and b_support.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    def coalition_signal(indices):
        idx = np.asarray(indices, dtype=int)
        n = int(idx.size)
        if n == 0:
            return 0.0
        if n == 1:
            return float(cue_weights[idx[0]])

        coordinates = represented[idx]
        gaps = coordinates[:, None] - coordinates[None, :]

        # A smooth super-Gaussian concentrates inferred dependence around
        # extremely close reliability representations. It remains continuous
        # and cardinality-general, but decays faster than the previous
        # Gaussian outside the local overlap neighborhood.
        scaled_gaps = gaps / max(similarity_bandwidth, 1e-12)
        similarity = np.exp(-0.5 * np.power(scaled_gaps, 4.0))
        np.fill_diagonal(similarity, 0.0)
        local_overlap = np.sum(similarity, axis=1)

        # Divisive credit allocation approximates precision loss under
        # correlated sources. In a pair, each cue has only one possible
        # neighbor. In larger coalitions, multiple pairwise overlaps converge
        # on each member and create nonlinear redundancy without a size gate.
        independence_fraction = np.power(
            1.0 + dependence_sensitivity * local_overlap,
            -overlap_curvature,
        )
        effective_total = float(
            np.sum(cue_weights[idx] * independence_fraction)
        )

        # Generic count normalization makes corroboration sublinear even for
        # reliability-diverse coalitions. Dependence can further reduce the
        # marginal value of clustered additions but never deletes a cue.
        signal = effective_total / (float(n) ** coalition_saturation)
        return float(max(signal, 0.0))

    evidence_a = coalition_signal(a_support)
    evidence_b = coalition_signal(b_support)
    net_a = evidence_a - evidence_b

    logits = np.array(
        [0.5 * beta * net_a, -0.5 * beta * net_a], dtype=np.float64
    )
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs /= probs.sum()

    probs = (1.0 - epsilon) * probs + epsilon * 0.5
    probs = np.maximum(probs, 0.0)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
    return probs / total
```

**`policy source code`:**
```python
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.maximum(probs, 0.0)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This 7-cue design creates a parameter-robust quantitative dissociation. On every trial, the highest-validity cue discriminates and favors one option, so Take The Best stops immediately and predicts the same choice probability for that option at every evidence level: (1-epsilon)*sigmoid(beta)+epsilon/2. In contrast, the lower-validity cues increasingly favor the other option. The tally margin opposing the top-cue winner ranges from 0 through 5. Tallying therefore predicts exactly 0.5 choice probability at margin 0 and an increasingly strong preference for the opposite option at margins 1-5, with probability (1-epsilon)*sigmoid(-beta*d)+epsilon/2 for the top-cue winner. Thus the critical signature is TTB's flat, above-chance endorsement of the top-cue winner versus Tallying's graded reversal as the opposing cue count grows. Mirrored A/B versions control for response-side bias. There are 12 unique pairs, yielding 8 repetitions each and exactly 96 trials.

[1] This design tests a validity-order invariance unique to Tallying. Every pair contains exactly three A-winning cues, three B-winning cues, and two tied cues. Tallying therefore predicts P(A)=0.5 on every trial, exactly and independently of beta and epsilon. In contrast, Take The Best finds a discriminating cue at validity rank 1, 2, or 3 and predicts the option favored by that cue with probability q=(1-epsilon)*sigmoid(beta)+epsilon/2. Thus TTB predicts a stable, above-chance directional preference even though the total cue tally is perfectly balanced, whereas Tallying predicts uniform guessing. The first discriminating cue favors A in half the pairs and B in the corresponding option-swapped mirrors, controlling response-side bias. There are two distinct evidence configurations at each stopping rank and their mirrors, yielding four trials per rank and 12 unique pairs total; each is repeated eight times for 96 trials. The critical advocated-theory signature is chance-level choice across all configurations despite large differences in which high-validity cue favors which option; TTB instead predicts complementary probabilities q and 1-q for the mirrored pairs.

[2] This 7-cue design holds the tally margin constant while systematically relocating the same 3-versus-2 cue coalition from early, low-validity positions to late, high-validity positions. In each of the first six pairs, A wins exactly three cues, B wins exactly two, and two cues tie. Tallying therefore predicts exactly the same A-choice probability on all six: (1-epsilon)*sigmoid(beta)+epsilon/2. The final six pairs are A/B-swapped mirrors, for which Tallying predicts the complementary constant probability. Accessibility-Balanced Evidence Integration instead assigns progressively greater weight to later cues; the ascending validities reinforce this positional gradient throughout the advocated parameter ranges. Consequently, its predicted A-choice probability should increase monotonically across the first six configurations. At the low end, A's three early wins should be overcome by B's two late wins, reversing the tally preference; at the high end, A's late coalition should produce a strong A preference. Coalition saturation further penalizes A's three-cue majority relative to B's two-cue coalition but is constant in form across the progression. The mirrored pairs should exhibit the complementary graded pattern, controlling response-side bias. Thus the decisive signature is a large configuration-dependent slope, including a preference reversal despite an invariant tally margin; Tallying permits neither. There are 12 unique pairs, each repeated eight times, yielding exactly 96 trials.

[3] This design isolates a positional-accessibility gradient while holding both cue tally and diagnosticity exactly constant. In each of the first six pairs, A wins one cue and B wins one cue, so Tallying predicts P(A)=0.5 exactly, independently of beta and epsilon. B's win is always at Expert 2 (index 1), whereas A's win moves successively from Expert 3 through Expert 8 (indices 2-7). All of these discriminating experts have identical advertised validity (0.70), so their diagnosticity components are equal. Accessibility-Balanced Evidence Integration therefore predicts a strictly increasing A-choice probability as A's supporting cue moves later: its evidence difference is proportional to accessibility(target)-accessibility(index 1). Coalition saturation is inactive because the coalitions are always one cue versus one cue. Across the competing theory's parameter ranges, the nearest-position contrast should be only slightly above chance (approximately 0.51-0.52), while the most separated contrast should produce approximately 0.62-0.79 A choices. The final six pairs swap A and B, yielding complementary predictions and controlling response-side bias. The high- and low-validity experts provide the required communicated validity spread but are tied in these trials, ensuring that the graded competing-theory prediction is specifically attributable to positional accessibility rather than validity. There are 12 unique pairs, each repeated eight times for 96 trials.

[4] This 10-cue design pits universal positional accessibility against switch-closure in balanced two-versus-two conflicts. Experts 1 and 4-10 all have validity 0.70; only these equal-validity experts discriminate. Experts 2 and 3 provide the required validity spread (0.95 and 0.55) but are tied, so validity differences cannot explain choices. In each of the first six pairs, A is supported by two cues including the final discriminating cue, whereas B is supported by two cues whose combined positions are later on average. Switch-Closure Coalition Integration enters its balanced-coalition branch: because A supplies the final discriminating cue and all four discriminating cues have identical validity, it predicts the same above-chance A-choice probability for all six configurations (approximately 0.58-0.66 across its parameter ranges), irrespective of spacing. Accessibility-Balanced Evidence Integration instead sums the cues' exponentially increasing positional weights. In every pair, B's two adjacent late cues have greater total accessibility than A's early-plus-terminal pair, so it robustly predicts P(A)<0.5. It additionally predicts graded changes as A's early cue moves later and the accessibility deficit shrinks, whereas the competing theory remains flat because coalition size and closure direction are unchanged. The final six pairs are exact A/B mirrors, producing complementary predictions and controlling response-side bias. Thus the decisive signature is a sign reversal—advocated-theory preference for B versus competing-theory preference for A—together with a graded spacing effect predicted only by the advocated theory. There are 12 unique pairs, each repeated eight times, for 96 trials.

[5] This 10-cue design targets the advocated theory's amplified coalition-primacy gate in unequal but nearly balanced three-versus-two conflicts. In each of the first four pairs, A receives a fixed, adjacent three-cue coalition at Experts 1-3, while B receives an adjacent two-cue coalition that moves progressively later, from Experts 4-5 through Experts 7-8. All discriminating experts have identical validity (0.70); the high- and low-validity Experts 9-10 are always tied and provide the required validity spread without affecting directional evidence. Switch-Closure Coalition Integration predicts A above chance throughout: its sublinear accumulation gives the three-cue coalition a modest numerical advantage, and its near-balance gate strongly favors A's earlier-onset coherent coalition. As B's coalition moves later, its onset-primacy score declines, so the advocated theory predicts a monotonic increase in P(A). Accessibility-Balanced Evidence Integration makes the opposite predictions. Its strong universal recency gradient gives B's two later cues more total weight than A's saturation-penalized three early cues, producing P(A)<0.5 even in the earliest configuration. Moving B later further increases its accessibility, so it predicts a monotonic decrease in P(A). Thus the theories differ both categorically in preferred option and quantitatively in the sign of the positional slope. The final four pairs are exact A/B mirrors, controlling response-side bias and yielding complementary predictions. Eight unique pairs are repeated 12 times each for exactly 96 trials.

[6] This 10-cue design isolates the advocated theory's translation-invariant run-asymmetry mechanism in nearly balanced three-versus-two conflicts. In each of the first six trials, A is supported by three equal-validity cues and B by the fixed adjacent pair at Experts 7-8. A's coalition always begins at Expert 1 and always contains exactly one adjacent link, but its internal ordering alternates between an entry-compact form [1,1,g] and an exit-compact form [g,1], for gaps g=2, 3, and 4. Thus each matched pair has identical coalition sizes, onset positions, total validity, and conventional adjacency coherence. Switch-Closure Coalition Integration consequently predicts exactly the same A-choice probability across all six configurations: its unequal-coalition gate only represents onset primacy and the proportion of adjacent links, both of which are held constant. Reliability-Compressed Parsing Mixtures instead assigns positive centered run coherence to entry-compact coalitions and negative coherence to exit-compact coalitions. It therefore predicts higher P(A) for every entry-compact configuration than for its matched exit-compact configuration. Moreover, the entry-minus-exit contrast should increase from gap 2 to gap 4 because the compact-versus-fragmented distinction becomes more pronounced. This contrast has the same sign for every integration style, although the stable size-calibration split can shift subjects' overall preference for the three-cue coalition. The final six trials are exact A/B mirrors, so the corresponding prediction is higher endorsement of the three-cue coalition when it is entry-compact, while controlling response-side bias. Experts 9-10 provide the required validity spread but are tied; every discriminating expert has validity 0.70. Twelve unique pairs repeated eight times yield exactly 96 trials.

[7] This 16-cue design tests absolute coalition-onset sensitivity against the competing theory's translation-invariant parsing rule in nearly balanced three-versus-two conflicts. In the first six pairs, A always has a three-cue coalition with the identical internal gap pattern [1,2] (one adjacent link), while B always has an adjacent two-cue coalition. Only their locations change: A's coalition begins progressively earlier than B's in the first three configurations and later than B's in the final three. All discriminating experts have validity 0.70; Experts 15 and 16 provide the validity spread but are tied. Switch-Closure Coalition Integration holds coalition size, accumulation, coherence, and validity evidence constant while its amplified near-balance gate increasingly disfavors A as A's coalition onset moves from before to after B's. It therefore predicts a marked monotonic decline in endorsement of the three-cue coalition, with a robust early-versus-late preference reversal. Reliability-Compressed Parsing Mixtures predicts exactly invariant endorsement of the three-cue coalition across these translations for every integration style: the size ratio is fixed, the three-cue coalition's centered run coherence is unchanged, the two-cue coalition's centered run coherence is always zero, and absolute chunk position is not used in its three-versus-two branch. The last six pairs are exact A/B mirrors, permitting the critical analysis to code choice of the three-cue coalition rather than response side. Twelve unique pairs repeated eight times yield 96 trials.

[8] This 8-cue design isolates the advocated theory's population-common completion term in balanced three-versus-three conflicts. Experts 1-6 all have validity 0.70 and discriminate on every trial; Experts 7-8 have validities 0.95 and 0.55 but give tied ratings, providing the required validity spread without creating reliability or composition differences between coalitions. In each of the first eight pairs, A is supported by exactly three cues, including Expert 6, the final discriminating cue, while B is supported by the complementary three cues. The eight configurations vary which two earlier cues join Expert 6, sampling compact, fragmented, and interleaved coalitions. Reliability-Compressed Parsing Mixtures adds a completion signal favoring A in every configuration because A supplies the final discriminating cue. Its completion increment ranges from 0.323 to 0.575 before response sensitivity, implying a common above-chance closure-coalition tendency, with smaller style-dependent parsing deviations around it. Exchangeable Corroboration with Idiosyncratic Salience has no completion mechanism: coalition sizes, validities, and order-free composition loads are exactly equal, so its population prediction for the closure coalition is 0.5. It permits stable individual configuration preferences from random salience profiles, but these preferences are zero-centered and therefore should produce approximately equal numbers of positive and negative subject-level closure effects rather than the advocated theory's common positive displacement. The final eight pairs are exact A/B mirrors, controlling response-side bias while preserving which spatial coalition contains the final discriminating cue. The decisive quantitative statistics are the closure-coded grand mean and the distribution of subject-level closure effects: reliably above 0.5 and predominantly positive under the advocated theory, versus population-centered at 0.5 with sign-symmetric heterogeneity under the competing theory. Sixteen unique pairs repeated six times yield exactly 96 trials.

[9] This 16-cue design isolates the advocated theory's order-free composition-redundancy mechanism in balanced two-versus-two conflicts. In the first six pairs, A is supported by two experts having the same advertised validity, which increases across conditions from 0.68 to 0.97; B is always supported by the low-validity Experts 7 and 15 (0.55 and 0.60). Every supporting pair is separated by eight positions. Consequently, both coalitions have zero adjacency and cannot place both members in the same width-2, width-3, or width-4 parsing chunk under any competing-theory integration style. Coalition size is also exactly balanced, and the two-versus-two format prevents completion. Reliability-Compressed Parsing Mixtures is therefore reduced to additive compressed reliability evidence and predicts a modest A preference that increases monotonically as A's two validities rise. Exchangeable Corroboration with Idiosyncratic Salience instead computes the pairwise composition load. A's load rises from (2*0.68-1)^2 = 0.1296 to (2*0.97-1)^2 = 0.8836, whereas B's load remains (2*0.55-1)(2*0.60-1) = 0.02. Its strong composition discount overwhelms its very weak reliability weighting, predicting preference for the low-validity B coalition and a steep monotonic decrease in P(A) as A's validity increases. Exchangeable salience can create stable individual deviations but supplies no population-level slope tied to these translated configurations. The final six pairs are exact A/B mirrors. The primary statistic should code selection of the equal-validity coalition: the advocated theory predicts a pronounced negative validity slope, while the competing theory predicts a shallow positive slope. Twelve unique pairs repeated eight times yield exactly 96 trials.

[10] This 18-cue design isolates the advocated theory's reliability-gap-dependent latent-source inference in balanced two-versus-two conflicts. In the first eight pairs, A is supported by a target pair whose mean advertised validity is always 0.90, while B is supported by the fixed heterogeneous reference pair (0.82, 0.68). The target pair's within-pair validity gap increases from 0.00 through 0.04 and 0.10 to 0.18; each level has two order-reversed or duplicate realizations to average over idiosyncratic cue salience. Reliability-Schema Source Inference predicts a large positive gap effect: the equal extreme pair (0.90, 0.90) receives both narrow shared-source discounting and the equality-specific extreme-pair penalty, whereas increasing the gap rapidly releases the target cues from inferred overlap. At the 0.18 gap, the (0.99, 0.81) pair is treated as nearly independent, so target-pair choice should approach approximately one half despite the reference coalition. Exchangeable Corroboration with Idiosyncratic Salience predicts a much flatter profile. Its target composition load changes only from 0.64 at equality to 0.6076 at the largest gap because the target mean remains fixed, and all target loads remain far above the reference load of 0.2304. It should therefore continue strongly discounting the target coalition, generally keeping target choice around the lower quarter of the response range with only a small gap-related increase. The decisive statistic is thus the slope or equality-to-0.18 contrast in target-coalition choice: large under the advocated theory but small under the competitor, particularly producing near-chance versus clearly below-chance target choice at the widest gap. The final eight pairs are exact A/B mirrors. Sixteen unique pairs repeated six times yield exactly 96 trials.

[11] This 12-cue design tests whether the internal reliability topology of a three-expert coalition matters after coalition size, mean validity, and the advocated theory's order-free composition load have been matched. Each clustered coalition has validities (0.87, 0.87, 0.96), whereas each distributed coalition has validities approximately (0.84804, 0.90, 0.95196). On the bounded-strength scale x=2v-1, these are respectively (0.74, 0.74, 0.92) and (0.69608, 0.80, 0.90392). Both triples have the same sum, 2.40, and exactly the same mean pairwise product, 0.6364. Thus Exchangeable Corroboration with Idiosyncratic Salience gives the coalitions equal composition discounts, equal size saturation, and identical three-member corroboration calibration. Their summed compressed-reliability weights are also nearly identical throughout the advocated parameter ranges. Its population prediction is therefore approximately P(distributed)=0.50, although stable individual salience profiles can produce subject-specific deviations. Reliability-Schema Source Inference instead predicts substantially greater choice of the distributed coalition. The clustered coalition contains an exactly matched, jointly reliable 0.87 pair, activating strong narrow shared-source overlap for two of its three members. The distributed coalition has no equal pair; its adjacent validity gaps are about 0.052 and its largest gap is about 0.104, yielding markedly less overlap despite some shoulder dependence. The four crossed comparisons use two duplicate instances of each coalition type, balancing how often every expert participates and reducing contamination from cue-specific salience. Four exact A/B mirrors control response-side bias. The decisive statistic is distributed-coalition choice pooled across mirrors: approximately chance under the advocated theory but robustly above chance under the competing theory. Eight unique pairs are repeated 12 times each for exactly 96 trials.

[12] This 12-cue design isolates the advocated theory's contextual reliability-region topology operation in equal-cardinality three-versus-three contests. There are two interchangeable clustered coalitions, each with validities (0.60, 0.60, 0.90), and two interchangeable distributed coalitions, each with validities (0.87320508, 0.70, 0.52679492). On the bounded diagnostic-strength scale x=2v-1, the clustered triple is (0.20, 0.20, 0.80), while the distributed triple is approximately (0.74641016, 0.40, 0.05358984). Both have strength sum 1.20 and sum of pairwise products 0.36, so their mean pair-composition load is exactly 0.12. Coalition cardinality, saturation, third-member calibration, and composition redundancy therefore cancel between options in both theories. The competing theory has no reliability-topology operation and should predict distributed-coalition choice very near 0.50 at the population level; its compressed reliability sums actually give the clustered coalition a tiny advantage, so it cannot explain a substantial distributed preference. In contrast, the advocated theory represents the two 0.60 clustered cues in effectively the same reliability region while the 0.90 cue occupies a distinct region. Its triadic operation consequently compresses the clustered coalition toward two informational units. The distributed triple has three widely separated represented diagnosticities and receives essentially no such compression. Equal cardinality also fully activates the advocated theory's relational topology contrast, producing a robust preference for the distributed coalition whose magnitude should increase with subject-level topology sensitivity. Four crossed comparisons make every coalition instance appear equally often, and four exact A/B mirrors control response-side bias. The primary statistic is mirror-corrected selection of the distributed coalition: clearly above chance under Contextual Reliability-Region Compression but approximately chance, or marginally below chance, under Exchangeable Corroboration. Eight unique pairs are repeated 12 times each for exactly 96 trials.

[13] This 18-cue design maps the competing theory's reliability-region clustering function while holding constant every order-free composition statistic used by Exchangeable Corroboration. The cues form six disjoint three-expert coalitions. Writing bounded diagnostic strength as x=2v-1, every coalition has exactly the same sum of strengths (1.20) and the same sum of pairwise products (0.36), hence the same mean pair-composition load (0.12). Coalition size, saturation, third-member calibration, mean advertised validity, and composition redundancy therefore cancel. The coalitions differ only in how far an initially duplicate low-validity pair is separated: the within-pair validity gaps are 0, 0.015, 0.03, 0.06, 0.10, and 0.14. Exchangeable Corroboration has no reliability-gap or topology operation and thus predicts an approximately flat, chance-level probability of choosing the less-clustered coalition, apart from tiny compressed-reliability residuals and zero-centered subject-specific salience. Contextual Reliability-Region Compression instead predicts a nonlinear release from duplicate-pair compression. Its largest effects should occur across the first few gap increments, where represented separation crosses the 0.018-0.040 cluster-bandwidth range; comparisons between the two widest-gap coalitions should approach chance because both are already distributed. The direct zero-gap versus 0.14-gap comparison should show the strongest preference for the distributed coalition, subject to the theory's concave topology contrast. The six comparisons form a cycle, so every coalition and every expert participates equally often. Exact A/B mirrors control response-side bias. The critical quantitative signature is therefore a localized, nonlinear gap-release profile under Contextual Reliability-Region Compression versus a nearly flat profile under Exchangeable Corroboration. Twelve unique pairs repeated eight times yield exactly 96 trials.

[14] This design tests a cardinality-specific contextual onset of redundancy. Four equal-reliability coalitions contain two 0.70 experts, whereas four reliability-separated coalitions contain experts with validities 0.65 and 0.7666667. On the bounded-strength scale x=2v-1, both pair types have exactly the same product: 0.40*0.40 = 0.30*0.5333334 = 0.16. Thus, in the 2-versus-2 trials, Contextual Reliability-Region Compression assigns them the same pair-composition load and has no triadic topology operation; apart from very small compressed-validity residuals, it predicts approximately chance-level selection of the separated pair. Reliability-Space Credit Sharing instead strongly discounts the exactly matched 0.70 pair, while the 0.65/0.7666667 pair lies far outside its narrow similarity kernel, predicting a large above-chance separated-pair preference. In matched 3-versus-3 trials, one distinct 0.92 outsider is added to each coalition. This activates the advocated theory's duplicate-pair-in-context operation only for the equal 0.70 pair, as well as its equal-cardinality relational topology contrast, so it now predicts a substantial separated-coalition preference. The competing theory continues to penalize the equal pair through the same cardinality-general credit-sharing rule and therefore predicts a strong separated preference in both dyadic and triadic contests, if anything slightly attenuated after adding an independent outsider. The decisive statistic is the coalition-size interaction: Contextual Reliability-Region Compression predicts an onset from approximately 0.50 in dyads to clearly above 0.50 in triads, whereas Reliability-Space Credit Sharing predicts an already-large dyadic effect with little or negative dyad-to-triad change. Crossed coalition instances, balanced outsider assignments, and exact A/B mirrors control cue salience and response-side bias. Sixteen unique pairs repeated six times yield exactly 96 trials.

[15] This 16-cue design tests whether redundancy continues accumulating across multiple duplicate reliability clusters or saturates after detecting a duplicate-pair-in-context relation. Two double-duplicate coalitions have validities (0.60, 0.60, 0.90, 0.90), while two single-duplicate coalitions have validities (0.75, 0.75, 0.9621320344, 0.5378679656). Writing bounded diagnostic strength as x=2v-1, these are respectively (0.20, 0.20, 0.80, 0.80) and (0.50, 0.50, 0.9242640688, 0.0757359312). Both coalition types have strength sum 2.00 and sum of pairwise products 1.32, so Contextual Reliability-Region Compression gives them exactly the same pair-composition load. Their sums of compressed diagnosticities are also nearly identical throughout that theory's validity-compression range. Both contain an exact duplicate pair accompanied by reliability-distinct outsiders, making their contextual topology scores close to saturation. Because the contextual theory's topology operation removes at most approximately one informational unit, it predicts only a small preference for the single-duplicate coalition, generally near chance. Reliability-Space Credit Sharing instead applies divisive sharing separately to every coalition member. In the double-duplicate coalition, all four experts have a reliability-identical neighbor and are discounted. In the single-duplicate coalition, only the two 0.75 experts are discounted, while the extreme outsiders contribute nearly undiluted evidence. It therefore predicts a substantially larger preference for the single-duplicate coalition, roughly in the 0.62-0.69 range before subject heterogeneity is integrated. The decisive statistic is mirror-corrected selection of the single-duplicate coalition: a large displacement above chance supports cardinality-general credit accumulation, whereas a small effect supports contextual one-unit topology compression. Four crossed comparisons balance coalition instances and stable cue salience; four exact A/B mirrors control response-side bias. Eight unique pairs are repeated 12 times each for exactly 96 trials.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "This design tests the advocated theory's sharply calibrated release when effective source count crosses from two to three, using duplicate-versus-independent coalitions at cardinalities 2, 3, and 4. At each cardinality, the duplicate coalition and the reliability-diverse coalition have the same mean advertised validity: dyads compare (0.75,0.75) with (0.60,0.90), triads compare (0.75,0.75,0.60) with (0.60,0.70,0.80), and tetrads compare (0.75,0.75,0.60,0.90) with (0.60,0.70,0.80,0.90). Thus the principal manipulation is source topology rather than coalition size or mean reliability. Categorical Source-Budget Inference assigns the duplicate and diverse coalitions respectively 1 versus 2 effective sources in dyads, 2 versus 3 in triads, and 3 versus 4 in tetrads. Its saturating source-count release is concentrated specifically between two and three sources. It therefore predicts a nonmonotonic cardinality profile: preference for the diverse coalition should increase from the dyadic contrast to a maximum in the triadic contrast, where only the diverse coalition crosses the release threshold, and then decrease in the tetradic contrast, where both coalitions are already beyond that threshold. Reliability-Space Credit Sharing has no two-to-three-source gate. It continuously discounts the two identical 0.75 experts at every cardinality, while all reliability-separated experts contribute nearly independently. Generic count normalization makes the duplicate penalty's effect on coalition signal largest in the dyad and progressively smaller after independent outsiders are added. It therefore predicts the opposite dyad-to-triad slope: diverse-coalition preference should decline monotonically, or at minimum decline from dyads to triads, rather than peak at three members. The decisive statistic is the cardinality-by-topology interaction, especially the change in mirror-corrected diverse-coalition choice from dyads to triads: positive under the advocated theory and negative under the competitor. Exact A/B mirrors control response-side bias. Six unique pairs are each repeated 16 times, yielding exactly 96 trials.",
  "validities": [
    0.75,
    0.75,
    0.6,
    0.9,
    0.75,
    0.75,
    0.6,
    0.6,
    0.7,
    0.8,
    0.75,
    0.75,
    0.6,
    0.9,
    0.6,
    0.7,
    0.8,
    0.9
  ],
  "trial_a_ratings": [
    [
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1
    ],
    [
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0
    ]
  ],
  "trial_b_ratings": [
    [
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 11225,
  "output_tokens": 5015
}
```
