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
**Validities (n_features=8):** [0.82, 0.78, 0.74, 0.7, 0.66, 0.6, 0.55, 0.51]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 0, 0, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 1, 0, 0, 1]  B=[0, 1, 1, 1, 0, 0, 0, 1]
  trial 3: A=[1, 0, 0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0, 0, 1, 0]
  trial 4: A=[1, 0, 0, 0, 1, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0, 1, 1]
  trial 5: A=[1, 0, 0, 0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 0, 1, 0, 0]
  trial 6: A=[1, 0, 0, 0, 1, 1, 0, 1]  B=[0, 1, 1, 1, 0, 1, 0, 1]
  trial 7: A=[1, 0, 0, 0, 1, 1, 1, 0]  B=[0, 1, 1, 1, 0, 1, 1, 0]
  trial 8: A=[1, 0, 0, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 0, 1, 1, 1]
  trial 9: A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 0]
  trial 10: A=[0, 1, 1, 1, 0, 0, 0, 1]  B=[1, 0, 0, 0, 1, 0, 0, 1]
  trial 11: A=[0, 1, 1, 1, 0, 0, 1, 0]  B=[1, 0, 0, 0, 1, 0, 1, 0]
  trial 12: A=[0, 1, 1, 1, 0, 0, 1, 1]  B=[1, 0, 0, 0, 1, 0, 1, 1]
  trial 13: A=[0, 1, 1, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1, 0, 0]
  trial 14: A=[0, 1, 1, 1, 0, 1, 0, 1]  B=[1, 0, 0, 0, 1, 1, 0, 1]
  trial 15: A=[0, 1, 1, 1, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1, 1, 1, 0]
  trial 16: A=[0, 1, 1, 1, 0, 1, 1, 1]  B=[1, 0, 0, 0, 1, 1, 1, 1]

**Rationale:** Use eight experts with an ordinary, moderately separated validity hierarchy and a sparse but perfectly recurrent relational structure. Exactly five of eight cues discriminate on every trial (density = 0.625): the highest-validity cue and Expert 5 favor the lexical winner, while Experts 2–4 form a stable three-cue coalition favoring the opposite option. Thus ordered search favors the leader, whereas the equal-weight majority favors the recurring coalition by 3–2. Experts 6–8 are ties whose common values vary factorially, creating eight visually distinct displays without changing density, cue directions, or coalition membership; exact A/B reversals balance response side, yielding 16 unique pairs and six repetitions each. Ecological Encoding Hysteresis sees a constant density of 0.625, below its 0.68–0.71 transition threshold. Its context estimate approaches 0.625 from an even sparser prior, so it predicts persistent ordered-search choices and little variation across serial position or tie-value variants. Relational Coalition Consolidation instead repeatedly observes the identical, internally coherent co-opposition of Experts 2–4. Their pairwise consistency and prototype recurrence should raise coalition accessibility despite the sparse ecology. Because the validity hierarchy is not unusually dominant, the safeguard is negligible; once consolidated, the validity-weighted lower-cue signal is strongly opposite the leader and should reverse choices toward the coalition winner. The key quantitative dissociation is therefore an acquisition-dependent crossover: leader-winner choices should remain high and approximately stationary under Ecological Encoding Hysteresis, whereas Relational Coalition Consolidation predicts declining leader-winner choice across encounters and eventual coalition-winner predominance. Conditional on matched trial position, all eight tie variants should form a common plateau under both accounts, confirming that any temporal shift is driven by learned cue relations rather than superficial display repetition.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Relational Coalition Consolidation theory proposes that people retain the advertised validity hierarchy while also learning recurring relationships among lower-ranked cues. On each encounter, the learner represents lower cues relative to the highest-validity discriminating cue and records which cues jointly oppose that leader. Repeated, internally coherent opposition patterns become relational chunks: their accessibility increases with recurrence, pairwise co-opposition, consensus, and repeated conflict with the leading cue. Accessibility changes gradually and exhibits hysteresis, so an established coalition remains available across occasional mismatching trials while an isolated coalition does not immediately displace rank-based processing. Decision evidence is a continuous competition between a graded rank signal, strengthened by advertised validity dominance, and a graded coalition signal reflecting the validity-weighted direction of lower cues. Coalition magnitude is compressed so that learned accessibility, rather than the raw size of the current tally margin, carries more of the influence. Cue count alone therefore does not cause pooling. Many cues can preserve lexical dominance when their coalitional membership rotates, whereas a smaller but recurrent and coherent anti-leader coalition can capture choice. An unusually dominant advertised leader additionally retains a bounded residual influence, preventing complete coalition capture without globally suppressing ordinary coalition learning. The account predicts order-dependent acquisition, persistence after coalition reversals, and different choices for identical displays following stable versus unstable coalition histories.

**Parameters:**
- n_features: n_features
- validities: validities
- learning_rate: [0.08, 0.14]
- relation_prior: [0.8, 1.6]
- recurrence_rate: [2.5, 4.5]
- coalition_threshold: [0.16, 0.25]
- gate_steepness: [8.0, 13.0]
- activation_retention: [0.78, 0.88]
- hysteresis_strength: [0.45, 0.75]
- initial_access: [0.18, 0.30]
- access_floor: [0.34, 0.44]
- access_ceiling: [0.82, 0.94]
- dominance_gain: [1.0, 2.2]
- coalition_gain: [3.2, 4.8]
- consensus_exponent: [0.40, 0.60]
- dominance_threshold: [0.58, 0.68]
- dominance_steepness: [25.0, 40.0]
- rank_safeguard: [0.78, 0.94]
- evidence_sensitivity: [1.0, 1.8]
- lapse: [0.0, 0.06]

**`predict source code`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Relational Coalition Consolidation expects shape (2, n_features); got {stim.shape}."
        )

    n_features = int(parameters["n_features"])
    if stim.shape[1] != n_features:
        raise ValueError(
            f"Stimulus has {stim.shape[1]} features but n_features={n_features}."
        )

    validities = np.asarray(parameters["validities"], dtype=float).reshape(-1)
    if validities.size != n_features:
        raise ValueError("validities must contain one value per feature.")

    learning_rate = float(parameters["learning_rate"])
    relation_prior = float(parameters["relation_prior"])
    recurrence_rate = float(parameters["recurrence_rate"])
    coalition_threshold = float(parameters["coalition_threshold"])
    gate_steepness = float(parameters["gate_steepness"])
    activation_retention = float(parameters["activation_retention"])
    hysteresis_strength = float(parameters["hysteresis_strength"])
    initial_access = float(parameters["initial_access"])
    access_floor = float(parameters["access_floor"])
    access_ceiling = float(parameters["access_ceiling"])
    dominance_gain = float(parameters["dominance_gain"])
    coalition_gain = float(parameters["coalition_gain"])
    consensus_exponent = float(parameters["consensus_exponent"])
    dominance_threshold = float(parameters["dominance_threshold"])
    dominance_steepness = float(parameters["dominance_steepness"])
    rank_safeguard = float(parameters["rank_safeguard"])
    sensitivity = float(parameters["evidence_sensitivity"])
    lapse = float(parameters["lapse"])

    cue_order = np.argsort(-validities, kind="stable")
    rank_position = np.empty(n_features, dtype=int)
    rank_position[cue_order] = np.arange(n_features)

    # The advertised validity reliability above chance supplies a graded,
    # bounded evidence scale. A small floor handles validity exactly 0.5.
    reliability = np.maximum(2.0 * validities - 1.0, 0.02)

    pair_opposition = np.zeros((n_features, n_features), dtype=float)
    pair_opportunity = np.zeros((n_features, n_features), dtype=float)
    prototypes = []
    activation = float(np.clip(initial_access, 0.0, 1.0))

    def describe(display):
        arr = np.asarray(display, dtype=float)
        if arr.ndim != 2 or arr.shape != (2, n_features):
            return None
        directions = np.sign(arr[0] - arr[1])
        leader = None
        for cue in cue_order:
            if directions[cue] != 0.0:
                leader = int(cue)
                break
        if leader is None:
            return {
                "directions": directions,
                "leader": None,
                "leader_sign": 0.0,
                "lower": np.array([], dtype=int),
                "opponents": np.array([], dtype=int),
                "mask": np.zeros(n_features, dtype=bool)
            }

        leader_sign = float(directions[leader])
        lower = np.asarray([
            j for j in range(n_features)
            if rank_position[j] > rank_position[leader] and directions[j] != 0.0
        ], dtype=int)
        opponents = np.asarray([
            j for j in lower if directions[j] == -leader_sign
        ], dtype=int)
        mask = np.zeros(n_features, dtype=bool)
        mask[opponents] = True
        return {
            "directions": directions,
            "leader": leader,
            "leader_sign": leader_sign,
            "lower": lower,
            "opponents": opponents,
            "mask": mask
        }

    def relational_strength(desc):
        lower = desc["lower"]
        opponents = desc["opponents"]
        mask = desc["mask"]
        if lower.size == 0 or opponents.size < 2:
            return 0.0

        n_opp = float(opponents.size)
        n_support = float(lower.size - opponents.size)
        # Net coherence is zero when lower cues split evenly and one when a
        # unanimous lower coalition opposes the leading cue.
        coherence = max(0.0, (n_opp - n_support) / float(lower.size))

        pair_values = []
        for u in range(opponents.size):
            for v in range(u + 1, opponents.size):
                j = int(opponents[u])
                k = int(opponents[v])
                numerator = pair_opposition[j, k] + 0.5 * relation_prior
                denominator = pair_opportunity[j, k] + relation_prior
                pair_values.append(numerator / max(denominator, 1e-12))
        pair_consistency = float(np.mean(pair_values)) if pair_values else 0.5

        # Similarity-based recurrence permits a learned coalition to tolerate
        # occasional missing members, rather than requiring exact repetition.
        recurrence_mass = 0.0
        current_size = int(np.sum(mask))
        for old_mask, old_weight in prototypes:
            intersection = int(np.sum(mask & old_mask))
            union = int(np.sum(mask | old_mask))
            similarity = (float(intersection) / float(union)) if union > 0 else 0.0
            recurrence_mass += old_weight * similarity ** 3
        recurrence = 1.0 - np.exp(-recurrence_rate * recurrence_mass)

        return float(coherence * pair_consistency * recurrence)

    def update_memory(desc):
        nonlocal prototypes
        if desc is None or desc["leader"] is None:
            return
        lower = desc["lower"]
        opponents = desc["opponents"]
        mask = desc["mask"]

        # Recency-weight all previously learned relations and prototypes.
        pair_opposition[:] *= (1.0 - learning_rate)
        pair_opportunity[:] *= (1.0 - learning_rate)
        prototypes = [
            (m, w * (1.0 - learning_rate))
            for m, w in prototypes
            if w * (1.0 - learning_rate) > 1e-6
        ]

        for u in range(lower.size):
            for v in range(u + 1, lower.size):
                j = int(lower[u])
                k = int(lower[v])
                pair_opportunity[j, k] += learning_rate
                pair_opportunity[k, j] += learning_rate

        for u in range(opponents.size):
            for v in range(u + 1, opponents.size):
                j = int(opponents[u])
                k = int(opponents[v])
                pair_opposition[j, k] += learning_rate
                pair_opposition[k, j] += learning_rate

        if opponents.size >= 2:
            prototypes.append((mask.copy(), learning_rate))

    past_a = history.get("option_a_ratings", [])
    past_b = history.get("option_b_ratings", [])
    n_past = min(len(past_a), len(past_b))

    # Reconstruct the subject's latent coalition memory in actual trial order.
    for i in range(n_past):
        a_i = np.asarray(past_a[i], dtype=float).reshape(-1)
        b_i = np.asarray(past_b[i], dtype=float).reshape(-1)
        if a_i.size != n_features or b_i.size != n_features:
            continue
        desc_i = describe(np.vstack([a_i, b_i]))
        structure_i = relational_strength(desc_i)
        drive = (
            gate_steepness * (structure_i - coalition_threshold)
            + hysteresis_strength * (activation - 0.5) * 4.0
        )
        drive = float(np.clip(drive, -60.0, 60.0))
        target = 1.0 / (1.0 + np.exp(-drive))
        activation = (
            activation_retention * activation
            + (1.0 - activation_retention) * target
        )
        activation = float(np.clip(activation, 0.0, 1.0))
        update_memory(desc_i)

    current = describe(stim)
    if current is None or current["leader"] is None:
        return np.array([0.5, 0.5], dtype=np.float64)

    # Viewing the current coalition retrieves matching relational chunks, but
    # activation changes only partially, preserving history and hysteresis.
    current_structure = relational_strength(current)
    current_drive = (
        gate_steepness * (current_structure - coalition_threshold)
        + hysteresis_strength * (activation - 0.5) * 4.0
    )
    current_drive = float(np.clip(current_drive, -60.0, 60.0))
    current_target = 1.0 / (1.0 + np.exp(-current_drive))
    activation = (
        activation_retention * activation
        + (1.0 - activation_retention) * current_target
    )
    activation = float(np.clip(activation, 0.0, 1.0))

    coalition_access = access_floor + (access_ceiling - access_floor) * activation
    coalition_access = float(np.clip(coalition_access, 0.0, 1.0))

    leader = int(current["leader"])
    leader_sign = float(current["leader_sign"])
    lower = current["lower"]

    # Rank evidence remains graded: a leader separated in advertised validity
    # from all currently discriminating lower cues receives a dominance bonus.
    if lower.size > 0:
        strongest_lower = float(np.max(reliability[lower]))
    else:
        strongest_lower = 0.0
    validity_dominance = max(
        0.0,
        (float(reliability[leader]) - strongest_lower)
        / max(float(reliability[leader]), 0.02)
    )
    rank_evidence = leader_sign * (1.0 + dominance_gain * validity_dominance)

    # A retrieved coalition pools lower directions without discarding their
    # advertised validities. A concave transform compresses differences among
    # current tally margins, leaving learned accessibility to carry more of
    # the coalition's magnitude while retaining graded directional evidence.
    if lower.size == 0:
        coalition_evidence = 0.0
    else:
        lower_weights = reliability[lower]
        raw_coalition_evidence = float(
            np.dot(lower_weights, current["directions"][lower])
            / max(float(np.sum(lower_weights)), 1e-12)
        )
        coalition_evidence = float(
            np.sign(raw_coalition_evidence)
            * np.abs(raw_coalition_evidence) ** consensus_exponent
        )

    decision_delta = (
        (1.0 - coalition_access) * rank_evidence
        + coalition_access * coalition_gain * coalition_evidence
    )

    # Only an unusually dominant advertised leader receives residual rank
    # protection. This bounded evidence-side safeguard is negligible under
    # ordinary validity hierarchies and does not suppress coalition learning.
    safeguard_logit = dominance_steepness * (
        validity_dominance - dominance_threshold
    )
    safeguard_logit = float(np.clip(safeguard_logit, -60.0, 60.0))
    safeguard_gate = 1.0 / (1.0 + np.exp(-safeguard_logit))
    safeguard_weight = float(np.clip(rank_safeguard * safeguard_gate, 0.0, 1.0))
    decision_delta = (
        (1.0 - safeguard_weight) * decision_delta
        + safeguard_weight * rank_evidence
    )

    logits = np.array([
        0.5 * sensitivity * decision_delta,
        -0.5 * sensitivity * decision_delta
    ], dtype=float)
    logits = logits - np.max(logits)
    core = np.exp(logits)
    core /= core.sum()

    probs = (1.0 - lapse) * core + lapse * np.array([0.5, 0.5])
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
    return probs / total
```

**`policy source code`:**
```python
def policy(probs) -> int:
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
**Description:** Ecological Encoding Hysteresis theory claims that decision makers learn how to encode an entire block of comparisons. They begin with a sparse-ecology prior favoring maintenance of the advertised validity hierarchy. After each comparison, they update a slowly changing estimate of the proportion of cues that discriminate. When discrimination remains sparse and ties are common, cue ranks remain cognitively useful and an ordered-search task set persists: the highest-validity discriminating cue controls choice, even on an occasional dense trial. When comparisons repeatedly contain many simultaneous discriminations, maintaining separate cue ranks becomes inefficient. People then compress cue directions into a relational gist and approximately pool them with equal weights. A steep contextual transition and a ceiling on compression implement hysteresis-like persistence rather than trial-by-trial load arbitration. Individual differences in the contextual threshold, prior strength, compression ceiling, sensitivity, and lapses produce heterogeneous behavior near the transition. Thus, identical diagnostic comparisons can elicit lexical choices in sparse filler blocks but majority choices in saturated filler blocks.

**Parameters:**
- n_features: n_features
- validities: validities
- context_prior: [0.50, 0.60]
- prior_strength: [1.0, 3.0]
- context_threshold: [0.68, 0.71]
- context_steepness: [35.0, 55.0]
- pooling_ceiling: [0.68, 0.76]
- evidence_sensitivity: [1.6, 3.0]
- lapse: [0.0, 0.08]

**`predict source code`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Ecological Encoding Hysteresis expects shape (2, n_features); got {stim.shape}."
        )

    n_features = int(parameters["n_features"])
    if stim.shape[1] != n_features:
        raise ValueError(
            f"Stimulus has {stim.shape[1]} features but n_features={n_features}."
        )

    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError("validities must contain one value per feature.")

    # Reconstruct the slowly accumulated block ecology. Each observation is
    # the fraction of cues that discriminate in a comparison; its complement
    # is tie prevalence. A prior prevents one unusual early trial from
    # immediately replacing the initial ordered-search task set.
    density_sum = 0.0
    n_observed = 0
    past_a = history.get("option_a_ratings", [])
    past_b = history.get("option_b_ratings", [])
    n_past = min(len(past_a), len(past_b))

    for i in range(n_past):
        a_i = np.asarray(past_a[i], dtype=float).reshape(-1)
        b_i = np.asarray(past_b[i], dtype=float).reshape(-1)
        if a_i.size != n_features or b_i.size != n_features:
            continue
        density_sum += float(np.mean(a_i != b_i))
        n_observed += 1

    # Viewing the current display supplies another ecological observation,
    # but its effect shrinks as evidence about the block accumulates.
    current_density = float(np.mean(stim[0] != stim[1]))
    density_sum += current_density
    n_observed += 1

    prior_density = float(parameters["context_prior"])
    prior_strength = float(parameters["prior_strength"])
    context_density = (
        prior_strength * prior_density + density_sum
    ) / (prior_strength + float(n_observed))

    # Saturated ecologies trigger a compressed relational encoding. The
    # ceiling allows some ordered processing to survive even in dense blocks.
    threshold = float(parameters["context_threshold"])
    steepness = float(parameters["context_steepness"])
    gate_logit = steepness * (context_density - threshold)
    gate_logit = float(np.clip(gate_logit, -60.0, 60.0))
    compression_activation = 1.0 / (1.0 + np.exp(-gate_logit))
    pooling_probability = (
        float(parameters["pooling_ceiling"]) * compression_activation
    )

    directions = np.sign(stim[0] - stim[1])

    # Ordered encoding: consult cues by advertised validity and stop at the
    # first discrimination.
    lexical_delta = 0.0
    cue_order = np.argsort(-validities, kind="stable")
    for cue in cue_order:
        if directions[cue] != 0.0:
            lexical_delta = float(directions[cue])
            break

    # Compressed encoding: retain all cue directions but discard fine-grained
    # rank and magnitude information. Sign normalization puts its response
    # scale on the same footing as the lexical representation.
    pooled_delta = float(np.sign(np.sum(directions)))

    sensitivity = float(parameters["evidence_sensitivity"])

    def choice_probs(delta):
        logits = np.array([
            0.5 * sensitivity * delta,
            -0.5 * sensitivity * delta
        ], dtype=float)
        logits = logits - np.max(logits)
        p = np.exp(logits)
        p /= p.sum()
        return p

    p_ordered = choice_probs(lexical_delta)
    p_compressed = choice_probs(pooled_delta)
    p_core = (
        (1.0 - pooling_probability) * p_ordered
        + pooling_probability * p_compressed
    )

    lapse = float(parameters["lapse"])
    probs = (1.0 - lapse) * p_core + lapse * np.array([0.5, 0.5])
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
    return probs / total
```

**`policy source code`:**
```python
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    \"\"\"Acquisition-weighted capture by the recurring anti-leader coalition.\"\"\"\n    if data is None or len(data) == 0:\n        return float(\"nan\")\n\n    scores = []\n    for _, subj in data.groupby(\"subject_id\", sort=False):\n        coalition_choices = []\n        for _, row in subj.iterrows():\n            a = np.asarray(row[\"option_a_ratings\"], dtype=float).reshape(-1)\n            b = np.asarray(row[\"option_b_ratings\"], dtype=float).reshape(-1)\n            if a.size == 0 or b.size == 0 or a[0] == b[0]:\n                continue\n\n            # Expert 1 is the highest-validity cue. The coalition winner is\n            # the option opposed to the option favored by that expert.\n            leader_response = 0 if a[0] > b[0] else 1\n            response = int(row[\"response\"])\n            coalition_choices.append(float(response != leader_response))\n\n        n = len(coalition_choices)\n        if n == 0:\n            continue\n\n        choices = np.asarray(coalition_choices, dtype=float)\n        early_n = min(8, n)\n        late_n = min(32, n)\n        early_rate = float(np.mean(choices[:early_n]))\n        late_rate = float(np.mean(choices[-late_n:]))\n\n        # Late coalition capture plus a modest acquisition bonus.\n        scores.append(late_rate + 0.25 * (late_rate - early_rate))\n\n    if not scores:\n        return float(\"nan\")\n    return float(np.mean(scores))",
  "rationale": "The metric is 1.25 times the coalition-choice rate over the final 32 trials minus 0.25 times its rate over the first 8 trials. It therefore primarily measures stable late-block coalition capture while adding a smaller, pre-specified reward for acquisition. Relational Coalition Consolidation predicts high late coalition choice and an early-to-late increase as the recurrent Experts 2\u20134 relation becomes accessible. Ecological Encoding Hysteresis predicts persistent highest-validity-cue choices because the constant discrimination density remains below its pooling threshold, yielding both a low late coalition rate and little temporal change. Coding choices relative to Expert 1 removes A/B side reversals, and averaging many late trials keeps the per-subject statistic stable."
}
```

## Usage

```json
{
  "input_tokens": 7743,
  "output_tokens": 3287
}
```
