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
**Validities (n_features=8):** [0.86, 0.86, 0.55, 0.95, 0.86, 0.86, 0.86, 0.86]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  trial 3: A=[0, 1, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  trial 4: A=[1, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  trial 5: A=[0, 1, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 1, 0, 1, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  trial 7: A=[0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 0]
  trial 8: A=[0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 1]
  trial 9: A=[0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 1]
  trial 10: A=[0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 0]
  trial 11: A=[0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1, 0, 0]
  trial 12: A=[0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 1, 0]

**Rationale:** This design isolates RBIS's exact-balanced coalition-geometry rule under a clear unique validity maximum. Expert 4 (95%) and the weak Expert 3 (55%) always support the coded anchor option, while exactly two 86%-valid experts support the opposing option. Thus every profile contains four active reasons in an exact 2-versus-2 balance and has the same validity multiset on each coalition. The critical manipulation changes only the serial span of the opposing pair. In the broad condition the opponent pairs are Experts {1,7}, {1,8}, and {2,8}; their index spans are at least 80% of the seven-step display span. In the nonbroad condition the pairs are {1,5}, {2,6}, and {5,7}, which fall below that threshold. RBIS therefore applies its full exact-balance resolution tendency to the broad pairs but only 54% of that tendency to the nonbroad pairs, shifting broad displays more strongly toward the unique Expert-4 anchor. All computations before this geometry rule are matched because opponent count, coalition validities, band ranks, anchor identity, and anchor-partner identity are identical. S3C predicts no broad-versus-nonbroad contrast: the 9-point validity gap disables its tied-order regime, and serial span and compactness are exactly ignored under a clear unique maximum. Its coalition support, diagnostic challenge, credibility, restoration, and precision calculations are therefore identical across all six profiles. The primary statistic is P(Expert-4 option | broad opposition) minus P(Expert-4 option | nonbroad opposition): RBIS predicts a geometry-dependent contrast, ordinarily positive because stronger balance resolution pulls choices toward Expert 4, whereas S3C predicts exactly zero for every parameter setting. Subject-level geometry contrasts are also informative because RBIS permits stable heterogeneity in the magnitude, and occasionally the sign, of this effect, while S3C permits no stimulus-geometry effect at all. Exact A/B reversals control response-side preference. The 12 unique pairs are repeated eight times for 96 trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Reliability-Band Interpretation Switching (RBIS) proposes that communicated cue validities are encoded as ordinal reliability bands rather than exact numerical weights. On each display, the decision maker identifies the highest occupied band and an initial anchor reason. If several active cues occupy that band, they are functionally tied and serial order selects the provisional anchor; if only one cue occupies it, serial order has no influence. A fully discriminating display can disambiguate a genuinely unique numerical maximum, preventing a lower-reliability early cue from spuriously replacing the designated anchor merely because it falls near a band boundary. The current display is classified into one of four mutually exclusive interpretation states: top-band resolution, diagnostic-coalition reframing, high-multiplicity anchor protection, or residual banded integration. Reframing is especially accessible for unique-top-band singleton and exactly two-cue opposition, with singleton dissent receiving the larger diagnostic advantage. State assignment for these canonical diagnostic shapes is less dependent on broad interpretation type, although stable subject-specific thresholds remain. Coalition credibility is represented coarsely by reliability-band proximity. An isolated exact maximum can be protected against redundant high-count opposition, whereas embedded balanced displays are not automatically protected. In the residual state, singleton and two-cue coalitions receive separate bounded credibility gains, while coalitions of three or more remain strongly saturated. When a nearby-band sparse coalition remains unresolved after the main state gate, it can become the selected diagnostic reason with modest commitment rather than defaulting to anchor-favoring integration. Those sparse mechanisms are suppressed on exact-balanced four-reason displays, where balance geometry instead determines interpretation. Balanced four-reason displays are resolved according to coalition geometry: opposition spanning the reliability display is more diagnostic of an anchor-resolving configuration than a compact opposing pair, although compact pairs receive an intermediate resolution tendency. High-count coherent opposition that narrowly misses the primary protection gate enters a low-commitment anchor-protection state, avoiding an all-or-none split between reframing and strong protection. Conflicting top-band displays slightly suppress generic compact reframing, allowing order-based top-band resolution to operate without imposing universal precedence. Displays with an early unique maximum embedded in a small coalition are assigned a common topological resolution commitment, preventing incidental opponent identity or span from producing large profile differences. Saturated unique-maximum displays with nonsingleton opposition enter a canonical top-band-resolution state whose classification is invariant to reallocating one supporting cue. Stable subject differences in band width, state thresholds, protection boundaries, sparse-coalition credibility, and state-specific response commitment produce persistent strategy heterogeneity. Because choices reveal no outcome information, RBIS never learns cue concurrence, opposition, or frequency from trial history.

**Parameters:**
- validities: validities
- band_width: [0.025, 0.085]
- band_decay: [0.55, 1.35]
- redundancy: [0.32, 0.78]
- singleton_opposition_gain: [1.30, 1.60]
- pair_opposition_gain: [1.22, 1.48]
- credible_sparse_commitment: [0.12, 0.30]
- reframe_bias: [0.45, 1.05]
- balance_reframe: [0.70, 1.60]
- embedded_reframe: [0.25, 0.90]
- singleton_reframe: [0.75, 1.55]
- diagnostic_reframe_offset: [0.35, 0.55]
- credibility_tolerance: [0.85, 2.20]
- state_threshold: [1.00, 1.75]
- compact_limit: [2.0, 3.2]
- isolated_protection_count: [2.80, 3.05]
- high_multiplicity_count: [5.5, 7.5]
- one_step_protection: [0.45, 0.95]
- fallback_protection_commitment: [0.12, 0.26]
- tied_reframe_scale: [0.74, 0.90]
- tie_commitment: [0.50, 0.95]
- integration_strength: [0.65, 1.25]
- reframe_strength: [1.30, 2.20]
- protection_strength: [1.15, 2.00]
- exact_balance_resolution: [0.82, 0.98]
- beta: [1.6, 4.5]
- lapse: [0.01, 0.12]
- interpretation_type: {0, 1, 2, 3, 4}

**`predict source code`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"RBIS expects state with shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    # There is deliberately no learning in this feedback-free task.
    _ = history

    band_width = float(parameters["band_width"])
    band_decay = float(parameters["band_decay"])
    redundancy = float(parameters["redundancy"])
    singleton_opposition_gain = float(parameters["singleton_opposition_gain"])
    pair_opposition_gain = float(parameters["pair_opposition_gain"])
    credible_sparse_commitment = float(parameters["credible_sparse_commitment"])
    reframe_bias = float(parameters["reframe_bias"])
    balance_reframe = float(parameters["balance_reframe"])
    embedded_reframe = float(parameters["embedded_reframe"])
    singleton_reframe = float(parameters["singleton_reframe"])
    diagnostic_reframe_offset = float(parameters["diagnostic_reframe_offset"])
    credibility_tolerance = float(parameters["credibility_tolerance"])
    state_threshold = float(parameters["state_threshold"])
    compact_limit = float(parameters["compact_limit"])
    isolated_protection_count = float(parameters["isolated_protection_count"])
    high_multiplicity_count = float(parameters["high_multiplicity_count"])
    one_step_protection = float(parameters["one_step_protection"])
    fallback_protection_commitment = float(parameters["fallback_protection_commitment"])
    tied_reframe_scale = float(parameters["tied_reframe_scale"])
    tie_commitment = float(parameters["tie_commitment"])
    integration_strength = float(parameters["integration_strength"])
    reframe_strength = float(parameters["reframe_strength"])
    protection_strength = float(parameters["protection_strength"])
    exact_balance_resolution = float(parameters["exact_balance_resolution"])
    beta = float(parameters["beta"])
    lapse = float(parameters["lapse"])
    interpretation_type = int(parameters["interpretation_type"])

    # Positive directions favor B; negative directions favor A.
    directions = np.sign(stim[1] - stim[0])
    active = np.flatnonzero(directions != 0)
    if active.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    # Stable interpretation types jointly alter entry into discrete states.
    # The wider separation generates durable between-subject differences
    # without trial-wise strategy resampling.
    if interpretation_type == 0:      # anchor-protective reason selector
        local_threshold = state_threshold + 1.20
        local_protection_count = isolated_protection_count - 0.10
        local_reframe_scale = 0.65
    elif interpretation_type == 1:    # coalition-sensitive reason selector
        local_threshold = state_threshold - 0.80
        local_protection_count = isolated_protection_count + 0.20
        local_reframe_scale = 1.35
    elif interpretation_type == 2:    # singleton-dissent specialist
        local_threshold = state_threshold - 0.45
        local_protection_count = isolated_protection_count
        singleton_reframe += 1.00
        local_reframe_scale = 1.20
    elif interpretation_type == 3:    # multiplicity-protection specialist
        local_threshold = state_threshold + 0.20
        local_protection_count = isolated_protection_count - 0.20
        high_multiplicity_count -= 0.50
        local_reframe_scale = 0.88
    else:                             # balanced configuration interpreter
        local_threshold = state_threshold - 0.55
        local_protection_count = isolated_protection_count
        balance_reframe += 0.85
        one_step_protection += 0.25
        local_reframe_scale = 1.12

    active_v = validities[active]
    best_v = float(np.max(active_v))

    # Validities are quantized relative to the best active validity. All cues
    # in ordinal band zero are treated as reliability-equivalent.
    width = max(band_width, 1e-8)
    band_rank = np.floor(
        np.maximum(best_v - validities, 0.0) / width + 1e-10
    ).astype(int)
    top_band = active[band_rank[active] == 0]

    # On a saturated six-cue display, simultaneous discrimination supplies
    # enough contrast to disambiguate a genuinely unique numerical maximum.
    # This narrowly prevents cue substitutions from changing the provisional
    # anchor in the Experiment-4 geometry without imposing a broad dense-
    # display protection rule.
    exact_max = active[np.isclose(active_v, best_v, atol=1e-10)]
    if n_features == 6 and active.size == 6 and exact_max.size == 1:
        top_band = exact_max

    # An isolated exact maximum facing at least three cues is the protected
    # reason even when subjective banding places an earlier cue in its band.
    if exact_max.size == 1:
        exact_direction = float(directions[int(exact_max[0])])
        exact_side_count = int(np.sum(directions[active] == exact_direction))
        exact_opposition_count = int(
            np.sum(directions[active] == -exact_direction)
        )
        if exact_side_count == 1 and exact_opposition_count >= 3:
            top_band = exact_max

    # Serial order is used solely inside the ambiguous top reliability band.
    anchor_cue = int(np.min(top_band))
    anchor_direction = float(directions[anchor_cue])
    top_band_conflict = bool(
        np.any(directions[top_band] > 0) and
        np.any(directions[top_band] < 0)
    )
    top_band_tied = bool(top_band.size > 1)
    unique_top_band = bool(top_band.size == 1)

    anchor_side = active[directions[active] == anchor_direction]
    opposition = active[directions[active] == -anchor_direction]
    anchor_count = int(anchor_side.size)
    opposition_count = int(opposition.size)
    total_count = int(active.size)

    if opposition_count == 0:
        chosen_evidence = protection_strength * anchor_direction
    else:
        # Opposition is grouped by coarse reliability-band proximity. Exact
        # opponent identity and small within-band differences cannot toggle a
        # state, which prevents incidental cue substitutions from generating
        # a primacy crossover.
        opponent_bands = band_rank[opposition].astype(float)
        opponent_best_band = int(np.min(band_rank[opposition]))
        coarse_distance = max(opponent_best_band - 1, 0)
        credibility = float(np.exp(
            -float(coarse_distance) / max(credibility_tolerance, 1e-8)
        ))

        count_balance = float(
            np.exp(-abs(opposition_count - anchor_count))
        )
        exact_balance = bool(opposition_count == anchor_count)
        compact = float(opposition_count <= compact_limit)
        singleton = float(opposition_count == 1)
        embedded = float(anchor_count > 1)
        isolated = float(anchor_count == 1)

        # Protection and reframing are categorical interpretation states.
        # Two opponents can still reframe an isolated anchor. Protection is
        # unavailable until there are at least three apparently redundant
        # opponents, producing the required nonmonotonic recovery.
        isolated_high_count = bool(
            isolated > 0.0 and
            opposition_count >= 3 and
            opposition_count >= local_protection_count
        )
        isolated_unique_high_multiplicity = bool(
            isolated > 0.0 and
            unique_top_band and
            total_count >= high_multiplicity_count and
            opposition_count >= 3
        )
        one_step_minority = bool(
            embedded > 0.0 and
            opposition_count == anchor_count + 1 and
            one_step_protection >= 0.45
        )
        protection_state = bool(
            isolated_high_count or
            isolated_unique_high_multiplicity or
            one_step_minority
        )

        base_reframe_score = credibility * (
            reframe_bias
            + 0.65 * compact
            + balance_reframe * count_balance
            + 0.35 * embedded_reframe * embedded
            + singleton_reframe * singleton
        )
        reframe_score = local_reframe_scale * base_reframe_score

        # Unique-top-band singleton and two-opponent coalitions receive a
        # configuration-specific threshold reduction. Tied-top displays are
        # excluded so their stable-order resolution is not weakened. For
        # these canonical shapes, broad interpretation-type offsets are
        # compressed, and singleton dissent receives the larger reduction.
        shape_specific_reframe = bool(
            unique_top_band and
            (singleton > 0.0 or opposition_count == 2)
        )
        if shape_specific_reframe:
            canonical_scale = 0.65 + 0.35 * local_reframe_scale
            reframe_score = canonical_scale * base_reframe_score
            effective_reframe_threshold = (
                0.65 * state_threshold
                + 0.35 * local_threshold
                - diagnostic_reframe_offset * (1.0 + 0.80 * singleton)
            )
        else:
            effective_reframe_threshold = local_threshold

        # Conflicting top-band reasons ordinarily remain eligible for order
        # resolution; generic compactness therefore has reduced ability to
        # preempt that state, without imposing unconditional tie precedence.
        if top_band_tied and top_band_conflict:
            reframe_score *= tied_reframe_scale

        # Compact and singleton opposition are diagnostic directly. Exact
        # balance can also trigger a categorical reinterpretation under a
        # unique top band, while conflicting tied-top displays retain their
        # separate order-resolution regime.
        diagnostic_shape = bool(
            compact > 0.0 or
            singleton > 0.0 or
            (exact_balance and not top_band_tied)
        )
        reframe_state = bool(
            diagnostic_shape and
            reframe_score > effective_reframe_threshold
        )

        if protection_state:
            # High multiplicity is interpreted as redundant opposition, so
            # the selected high-band reason is protected.
            chosen_evidence = protection_strength * anchor_direction
        elif reframe_state:
            # Reframing selects the opposing interpretation as a whole; it is
            # not a continuous bonus added to an evidence accumulator.
            chosen_evidence = -reframe_strength * anchor_direction
        elif top_band_tied and top_band_conflict:
            # Ambiguous highest-band reasons are resolved by stable display
            # order only after protection and diagnostic reframing are tested.
            chosen_evidence = tie_commitment * anchor_direction
        else:
            # Residual state: coarse ordinal integration. Exact numerical
            # validity differences within a band play no role.
            ordinal_weight = np.exp(-band_decay * band_rank.astype(float))

            def coalition_support(indices):
                n = int(indices.size)
                if n == 0:
                    return 0.0
                raw = float(np.sum(ordinal_weight[indices]))
                # Additional same-option reasons are treated as increasingly
                # redundant rather than as independent linear evidence.
                return raw / (1.0 + redundancy * float(max(n - 1, 0)))

            support_anchor = coalition_support(anchor_side)
            support_opposition = coalition_support(opposition)

            # Singleton and paired opposition receive separately calibrated
            # residual credibility. Exact-balanced four-reason displays are
            # excluded because their interpretation is governed by the
            # dedicated coalition-geometry resolution below.
            sparse_nonbalanced = bool(
                unique_top_band and
                not (total_count == 4 and exact_balance)
            )
            if sparse_nonbalanced and opposition_count == 1:
                support_opposition *= singleton_opposition_gain
            elif sparse_nonbalanced and opposition_count == 2:
                support_opposition *= pair_opposition_gain

            margin = (
                (support_anchor - support_opposition) /
                max(support_anchor + support_opposition, 1e-12)
            )
            chosen_evidence = integration_strength * anchor_direction * margin

            # A nearby-band sparse coalition that reaches the residual branch
            # is interpreted as a weak diagnostic reason rather than being
            # returned to anchor-favoring integration. This is a categorical,
            # low-commitment reason-selection state and is unavailable on the
            # balanced four-reason displays used by the geometry rule.
            credible_sparse_state = bool(
                sparse_nonbalanced and
                opposition_count <= 2 and
                coarse_distance == 0 and
                chosen_evidence * anchor_direction > 0.0
            )
            if credible_sparse_state:
                chosen_evidence = (
                    -credible_sparse_commitment * anchor_direction
                )

        # Balanced four-reason resolution depends on coalition geometry rather
        # than feature count alone. Compact opposing pairs receive an
        # intermediate resolution tendency, while edge-spanning opposition
        # continues to receive the full tendency.
        if total_count == 4 and exact_balance and unique_top_band:
            opposition_span = float(
                np.max(opposition) - np.min(opposition)
            ) if opposition_count > 1 else 0.0
            display_span = float(max(n_features - 1, 1))
            broad_opposition = float(
                opposition_span >= 0.80 * display_span
            )
            balance_resolution = exact_balance_resolution * (
                0.54 + 0.46 * broad_opposition
            )
            chosen_evidence = (
                (1.0 - balance_resolution) * chosen_evidence
                + balance_resolution * anchor_direction
            )

        # Coherent high-count opposition that narrowly misses the primary
        # protection gate enters a common low-commitment protection state.
        # This produces moderate recovery without polarizing subjects into
        # strong reframing and near-certain protection groups.
        opponent_band_span = int(
            np.max(band_rank[opposition]) - np.min(band_rank[opposition])
        )
        fallback_high_count_state = bool(
            not protection_state and
            isolated > 0.0 and
            unique_top_band and
            exact_max.size == 1 and
            opposition_count >= 3 and
            opponent_band_span <= 1
        )
        if fallback_high_count_state:
            chosen_evidence = (
                fallback_protection_commitment * anchor_direction
            )

        # Saturated unique-maximum displays with nonsingleton opposition enter
        # a canonical top-band-resolution state. Its selected reason and
        # commitment are invariant to moving one nonanchor cue between sides.
        # Singleton opposition remains eligible for diagnostic reframing.
        if (
            n_features == 6 and
            total_count == 6 and
            exact_max.size == 1 and
            opposition_count >= 2
        ):
            chosen_evidence = protection_strength * anchor_direction

        # Small embedded coalitions organized around the earliest unique
        # numerical maximum share one coarse topology. Pooling their response
        # commitment prevents incidental opponent identity and span from
        # producing brittle profile-to-profile state differences.
        early_embedded_topology = bool(
            exact_max.size == 1 and
            int(exact_max[0]) == 0 and
            anchor_cue == 0 and
            anchor_count > 1 and
            3 <= total_count <= 5
        )
        if early_embedded_topology:
            chosen_evidence = 0.55 * integration_strength * anchor_direction

    logits = beta * np.array(
        [-0.5 * chosen_evidence, 0.5 * chosen_evidence],
        dtype=np.float64,
    )
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs /= probs.sum()

    probs = (1.0 - lapse) * probs + lapse * np.array(
        [0.5, 0.5], dtype=np.float64
    )
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
    probs /= total
    return probs.astype(np.float64)
```

**`policy source code`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** Static Configural Coalition Competition (S3C) proposes that decisions are reconstructed from the current cue display without learning from trial history. Each discriminating expert retains an identity-specific weight derived from communicated validity, but validity evidence is bounded to prevent a uniquely most-valid cue from becoming arbitrarily dominant. Serial accessibility is a distinct regime that is exactly absent under a clear unique maximum and operates only among genuinely tied or extremely near-tied active cues. Option-specific evidence accumulates with smooth within-coalition redundancy. Compact one- and two-cue opposition receives a display-based diagnosticity bonus, while the credibility of a coalition is computed through a bounded aggregation of its members' validity gaps so that moving one cue cannot induce a quasi-discrete change in competition. The effects of credibility and anchor isolation are likewise softly bounded. At larger opposition multiplicities, compact diagnosticity decays and a saturating restoration process can return choice toward a strong isolated anchor. This restoration receives a smooth pre-sigmoid increment when an effectively isolated anchor faces more than two opponents, while embedded anchors remain protected from that increment. Exact balance versus one-step imbalance can modulate response precision without changing evidence direction, allowing activation of an additional cue to sharpen an already configured coalition preference. Stable subject differences jointly vary validity compression, tied-cue accessibility, compact-dissent sensitivity, restoration, balance sensitivity, and response precision. In the absence of outcome feedback, no aspect of the model updates from trial history.

**Parameters:**
- validities: validities
- validity_sensitivity: [0.08, 0.70]
- validity_floor: [0.58, 0.76]
- order_accessibility: [1.20, 3.80]
- accessibility_scale: [0.10, 0.75]
- order_span: [0.55, 2.50]
- balance_accessibility: [0.55, 1.0]
- balance_scale: [0.45, 1.80]
- redundancy_strength: [0.35, 1.40]
- redundancy_curvature: [1.05, 2.20]
- similarity_sensitivity: [2.0, 16.0]
- compact_dissent: [1.5, 6.5]
- compact_peak: [1.10, 1.60]
- compact_width: [0.45, 0.90]
- compactness_gain: [0.20, 1.50]
- credibility_scale: [0.025, 0.16]
- isolation_gain: [0.25, 1.50]
- restoration_strength: [0.65, 0.97]
- restoration_threshold: [2.25, 3.60]
- restoration_slope: [3.5, 11.0]
- balance_restoration: [0.45, 1.0]
- restoration_credibility: [0.30, 0.70]
- isolation_restoration: [0.15, 0.55]
- anchor_threshold: [0.66, 0.82]
- anchor_slope: [10.0, 35.0]
- balance_precision_gain: [0.75, 1.60]
- balance_precision_width: [0.45, 0.75]
- beta: [1.3, 5.0]
- lapse: [0.0, 0.13]
- configuration_type: {0, 1, 2}

**`predict source code`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"S3C expects state with shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    # S3C is deliberately history-invariant in this feedback-free task.
    _ = history

    directions = np.sign(stim[1] - stim[0])
    active = np.flatnonzero(directions != 0)
    if active.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    validity_sensitivity = float(parameters["validity_sensitivity"])
    validity_floor = float(parameters["validity_floor"])
    order_accessibility = float(parameters["order_accessibility"])
    accessibility_scale = float(parameters["accessibility_scale"])
    order_span = float(parameters["order_span"])
    balance_accessibility = float(parameters["balance_accessibility"])
    balance_scale = float(parameters["balance_scale"])
    redundancy_strength = float(parameters["redundancy_strength"])
    redundancy_curvature = float(parameters["redundancy_curvature"])
    similarity_sensitivity = float(parameters["similarity_sensitivity"])
    compact_dissent = float(parameters["compact_dissent"])
    compact_peak = float(parameters["compact_peak"])
    compact_width = float(parameters["compact_width"])
    compactness_gain = float(parameters["compactness_gain"])
    credibility_scale = float(parameters["credibility_scale"])
    isolation_gain = float(parameters["isolation_gain"])
    restoration_strength = float(parameters["restoration_strength"])
    restoration_threshold = float(parameters["restoration_threshold"])
    restoration_slope = float(parameters["restoration_slope"])
    balance_restoration = float(parameters["balance_restoration"])
    restoration_credibility = float(parameters["restoration_credibility"])
    isolation_restoration = float(parameters["isolation_restoration"])
    anchor_threshold = float(parameters["anchor_threshold"])
    anchor_slope = float(parameters["anchor_slope"])
    balance_precision_gain = float(parameters["balance_precision_gain"])
    balance_precision_width = float(parameters["balance_precision_width"])
    beta = float(parameters["beta"])
    lapse = float(parameters["lapse"])
    configuration_type = int(parameters["configuration_type"])

    # Correlated stable types alter several mechanisms together rather than
    # independently injecting trial-wise variability.
    if configuration_type == 0:       # coalition-sensitive integrator
        validity_floor = min(0.90, validity_floor + 0.12)
        order_accessibility *= 0.75
        restoration_strength *= 0.65
    elif configuration_type == 1:     # tied-order-sensitive integrator
        validity_floor = min(0.90, validity_floor + 0.00)
        order_accessibility *= 1.35
        restoration_strength *= 0.85
    else:                              # intermediate configural integrator
        validity_floor = min(0.90, validity_floor + 0.05)
        order_accessibility *= 0.90
        restoration_strength *= 1.00

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-float(np.clip(x, -60.0, 60.0))))

    v = np.clip(validities, 0.500001, 0.999999)
    reliability = np.log(v / (1.0 - v))
    best_active_reliability = float(np.max(reliability[active]))

    active_v = validities[active]
    best_v = float(np.max(active_v))
    # Only genuine or extremely close top ties enter the order regime.
    best_candidates = active[np.isclose(active_v, best_v, atol=0.005)]
    tied_order_regime = bool(best_candidates.size > 1)

    # Identity-specific instruction weights with a bounded strongest-to-weakest
    # ratio. Validity ordering is preserved, but a single anchor cannot become
    # arbitrarily dominant under the exponential reliability transform.
    unbounded_weight = np.exp(
        validity_sensitivity * (reliability - best_active_reliability)
    )
    instruction_weight = (
        validity_floor + (1.0 - validity_floor) * unbounded_weight
    )

    a_cues = np.flatnonzero(directions < 0)
    b_cues = np.flatnonzero(directions > 0)
    n_a = int(a_cues.size)
    n_b = int(b_cues.size)
    count_imbalance = abs(n_b - n_a)

    # Serial accessibility is exactly absent under a clear unique maximum.
    # This cleanly separates validity-based choice from tied-order choice.
    near_balance = np.exp(
        -float(count_imbalance) / max(balance_scale, 1e-8)
    )
    access = np.ones(n_features, dtype=float)
    if tied_order_regime:
        for serial_rank, cue in enumerate(active):
            validity_closeness = np.exp(
                -(best_active_reliability - reliability[cue])
                / max(accessibility_scale, 1e-8)
            )
            serial_salience = np.exp(
                -float(serial_rank) / max(order_span, 1e-8)
            )
            activation_gate = (
                (1.0 - balance_accessibility)
                + balance_accessibility * near_balance
            )
            access[cue] += (
                order_accessibility
                * validity_closeness
                * serial_salience
                * activation_gate
            )

    accessible_weight = instruction_weight * access

    def coalition_support(indices):
        n = int(indices.size)
        if n == 0:
            return 0.0

        weights = np.asarray(accessible_weight[indices], dtype=float)
        raw = float(np.sum(weights))
        dominant = max(float(np.max(weights)), 1e-12)
        effective_multiplicity = raw / dominant
        excess = max(effective_multiplicity - 1.0, 0.0)

        coalition_validities = validities[indices]
        spread = (
            float(np.std(coalition_validities)) if n > 1 else 0.0
        )
        similarity = np.exp(-similarity_sensitivity * spread)
        redundancy = 1.0 + redundancy_strength * similarity * (
            excess ** redundancy_curvature
        )
        return raw / max(redundancy, 1e-12)

    support_a = coalition_support(a_cues)
    support_b = coalition_support(b_cues)

    # A communicated-validity tie is resolved by bounded current accessibility,
    # not by an unconditional first-cue rule.
    candidate_access = accessible_weight[best_candidates]
    anchor_cue = int(best_candidates[int(np.argmax(candidate_access))])
    anchor_direction = float(directions[anchor_cue])

    if anchor_direction > 0:
        anchor_indices = b_cues
        opponent_indices = a_cues
        anchor_support = support_b
        opponent_support = support_a
    else:
        anchor_indices = a_cues
        opponent_indices = b_cues
        anchor_support = support_a
        opponent_support = support_b

    anchor_count = int(anchor_indices.size)
    opponent_count = int(opponent_indices.size)

    def effective_multiplicity(indices):
        if indices.size == 0:
            return 0.0
        weights = np.asarray(accessible_weight[indices], dtype=float)
        return float(np.sum(weights)) / max(float(np.max(weights)), 1e-12)

    anchor_effective_n = effective_multiplicity(anchor_indices)
    opponent_effective_n = effective_multiplicity(opponent_indices)

    if opponent_count > 0:
        opponent_v = validities[opponent_indices]

        # Each member's validity gap is smoothly bounded before aggregation.
        # Consequently, adding or moving one relatively strong cue cannot
        # multiply coalition credibility by an arbitrarily large amount.
        raw_validity_gaps = np.maximum(best_v - opponent_v, 0.0)
        bounded_gap = float(np.mean(np.tanh(
            raw_validity_gaps / max(credibility_scale, 1e-8)
        )))
        opponent_credibility_gate = np.exp(-bounded_gap)

        # Absolute serial position can affect compactness only inside the tied
        # order regime. It is exactly disabled for clear unique maxima.
        serial_span = int(np.max(opponent_indices) - np.min(opponent_indices) + 1)
        compactness = float(opponent_count) / float(max(serial_span, 1))
        effective_compactness_gain = (
            compactness_gain if tied_order_regime else 0.0
        )

        count_distance = (
            opponent_effective_n - compact_peak
        ) / max(compact_width, 1e-8)
        compact_count_profile = np.exp(-0.5 * count_distance ** 2)

        # Isolation remains strongest for a lone anchor, but its contribution
        # is bounded rather than multiplying challenge without limit.
        isolation = 1.0 / float(max(anchor_count, 1))
        raw_isolation_bonus = isolation_gain * isolation
        bounded_isolation_bonus = raw_isolation_bonus / (
            1.0 + raw_isolation_bonus
        )
        challenge_multiplier = 1.0 + (
            compact_dissent
            * compact_count_profile
            * opponent_credibility_gate
            * (1.0 + effective_compactness_gain * compactness)
            * (1.0 + bounded_isolation_bonus)
        )
        opponent_support *= challenge_multiplier

    total = anchor_support + opponent_support
    if total <= 1e-12:
        core_evidence = 0.0
    else:
        anchor_margin = (anchor_support - opponent_support) / total
        core_evidence = anchor_direction * anchor_margin

    if anchor_count > 0:
        anchor_identity_share = float(accessible_weight[anchor_cue]) / max(
            float(np.sum(accessible_weight[anchor_indices])), 1e-12
        )
    else:
        anchor_identity_share = 0.0

    # The isolated-anchor increment now enters the pre-sigmoid restoration
    # scale. It turns on smoothly above two effective opponents and saturates,
    # while the cubed identity share protects embedded anchors.
    isolated_transition = sigmoid(
        restoration_slope * (opponent_effective_n - 2.35)
    )
    restoration_latent = (
        restoration_slope
        * (opponent_effective_n - restoration_threshold)
        + 4.0
        * isolation_restoration
        * (np.clip(anchor_identity_share, 0.0, 1.0) ** 3)
        * isolated_transition
    )
    high_count_gate = sigmoid(restoration_latent)
    balance_crossing = sigmoid(
        restoration_slope
        * ((opponent_effective_n - anchor_effective_n) - 0.5)
    )
    count_gate = max(
        high_count_gate,
        balance_restoration * balance_crossing
    )

    absolute_anchor_gate = sigmoid(
        anchor_slope * (best_v - anchor_threshold)
    )
    credibility_gate = (
        restoration_credibility
        + (1.0 - restoration_credibility) * absolute_anchor_gate
    )

    # Embedded anchors can be restored, but isolated anchors express the
    # effect most strongly.
    embedding_factor = 0.35 + 0.65 * np.sqrt(
        np.clip(anchor_identity_share, 0.0, 1.0)
    )
    restoration_gate = (
        restoration_strength
        * count_gate
        * credibility_gate
        * embedding_factor
    )
    restoration_gate = float(np.clip(restoration_gate, 0.0, 0.97))

    choice_evidence = (
        (1.0 - restoration_gate) * core_evidence
        + restoration_gate * anchor_direction
    )

    # Moving one member away from exact balance can sharpen an already formed
    # directional preference. This display-only modulation changes precision,
    # not evidence direction, and is disabled in the tied-order regime.
    if tied_order_regime:
        effective_beta = beta
    else:
        width = max(balance_precision_width, 1e-8)
        one_step_profile = np.exp(
            -0.5 * ((float(count_imbalance) - 1.0) / width) ** 2
        )
        balanced_profile = np.exp(
            -0.5 * (float(count_imbalance) / width) ** 2
        )
        precision_shift = balance_precision_gain * (
            one_step_profile - balanced_profile
        )
        effective_beta = beta * np.exp(
            float(np.clip(precision_shift, -1.8, 1.8))
        )

    logits = effective_beta * np.array(
        [-0.5 * choice_evidence, 0.5 * choice_evidence],
        dtype=np.float64,
    )
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs /= probs.sum()

    probs = (1.0 - lapse) * probs + lapse * np.array(
        [0.5, 0.5], dtype=np.float64
    )
    probs = np.clip(probs, 0.0, None)
    normalizer = float(probs.sum())
    if not np.isfinite(normalizer) or normalizer <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
    probs /= normalizer
    return probs.astype(np.float64)
```

**`policy source code`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    def subject_contrast(df):\n        broad_choices = []\n        nonbroad_choices = []\n\n        for _, row in df.iterrows():\n            a = np.asarray(row[\"option_a_ratings\"], dtype=int)\n            b = np.asarray(row[\"option_b_ratings\"], dtype=int)\n            direction = np.sign(b - a).astype(int)\n\n            # Expert 4 (zero-based index 3) is the unique-validity anchor.\n            anchor_direction = int(direction[3])\n            if anchor_direction == 0:\n                continue\n\n            opposition = np.flatnonzero(direction == -anchor_direction)\n            if opposition.size != 2:\n                continue\n\n            opposition_span = int(np.max(opposition) - np.min(opposition))\n            is_broad = opposition_span >= 6\n\n            response = int(row[\"response\"])\n            anchor_chosen = float(\n                response == (1 if anchor_direction > 0 else 0)\n            )\n\n            if is_broad:\n                broad_choices.append(anchor_chosen)\n            else:\n                nonbroad_choices.append(anchor_chosen)\n\n        if len(broad_choices) == 0 or len(nonbroad_choices) == 0:\n            return 0.0\n        return float(np.mean(broad_choices) - np.mean(nonbroad_choices))\n\n    if \"subject_id\" in data.columns:\n        contrasts = [\n            subject_contrast(df)\n            for _, df in data.groupby(\"subject_id\", sort=False)\n        ]\n        return float(np.mean(contrasts)) if len(contrasts) else 0.0\n\n    return subject_contrast(data)\n",
  "rationale": "This is the subject-balanced broad-minus-nonbroad contrast in choosing the option supported by the uniquely most valid Expert 4. It codes choices relative to the anchor rather than as A/B responses, so the exact reversals cancel response-side tendencies. Broad opposition is identified by the preregistered span criterion: the two cues opposing Expert 4 are at least six index positions apart. RBIS predicts stronger anchor-directed balance resolution for these broad coalitions than for nonbroad coalitions. S3C disables serial-span and compactness effects under the clear 0.95 unique maximum, so its matched profiles predict a contrast of zero. All 96 trials contribute, and averaging subject-specific contrasts in pooled data keeps the canonical estimate aligned with the quantities used to assess between-subject stability."
}
```

## Usage

```json
{
  "input_tokens": 11228,
  "output_tokens": 1784
}
```
