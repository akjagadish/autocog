# Round 2 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_4` — SURVIVED ✓

**Description:** Ecological Encoding Hysteresis theory claims that decision makers learn how to encode an entire block of comparisons. They begin with a sparse-ecology prior favoring maintenance of the advertised validity hierarchy. After each comparison, they update a slowly changing estimate of the proportion of cues that discriminate. When discrimination remains sparse and ties are common, cue ranks remain cognitively useful and an ordered-search task set persists: the highest-validity discriminating cue controls choice, even on an occasional dense trial. When comparisons repeatedly contain many simultaneous discriminations, maintaining separate cue ranks becomes inefficient. People then compress cue directions into a relational gist and approximately pool them with equal weights. A steep contextual transition and a ceiling on compression implement hysteresis-like persistence rather than trial-by-trial load arbitration. Individual differences in the contextual threshold, prior strength, compression ceiling, sensitivity, and lapses produce heterogeneous behavior near the transition. Thus, identical diagnostic comparisons can elicit lexical choices in sparse filler blocks but majority choices in saturated filler blocks.

**Rationale:** The model addresses the central failure of invariant Take-The-Best by permitting genuine lower-cue influence, but only after the block ecology induces a compressed encoding. Experiment 1 has a relatively tie-rich average ecology, so its contextual estimate remains below threshold and even dense embedded trials are usually processed lexicographically, yielding negligible tally modulation. Experiment 2 repeatedly presents high discrimination density, rapidly moving the contextual state above threshold; the resulting mixture is majority-favoring but retains enough ordered processing to avoid the excessive tally contrast of a pure equal-weight rule. Experiment 3 averages below the transition despite containing comparisons with several discriminating cues, preserving a reliably lexical preference. Experiment 4 is maximally saturated, so compression approaches its subject-specific ceiling and produces the observed majority preference. Unlike a current-load gate, the decision rule for a stimulus depends on preceding filler trials. The model therefore predicts that placing identical conflict trials in sparse versus saturated blocks will reverse their dominant influence from the validity-leading cue toward the cue majority.

**Parameters:**
  - `n_features`: `n_features`
  - `validities`: `validities`
  - `context_prior`: `[0.50, 0.60]`
  - `prior_strength`: `[1.0, 3.0]`
  - `context_threshold`: `[0.68, 0.71]`
  - `context_steepness`: `[35.0, 55.0]`
  - `pooling_ceiling`: `[0.68, 0.76]`
  - `evidence_sensitivity`: `[1.6, 3.0]`
  - `lapse`: `[0.0, 0.08]`

**`predict(parameters, stimulus, history)`:**
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

**`policy(probs)`:**
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


### slot 2 — `pi_3` — KILLED ✗

**Description:** Rank-Budget Arbitration theory proposes that people initially represent advertised cue validities as an ordered hierarchy and try to make a lexicographic choice using the highest-ranked discriminating cue. Reliable access to that hierarchy is capacity-limited, however. The probability that rank retrieval fails increases smoothly with the observable representational load imposed by the number of available cues and the number of cues that must be compared on the current trial. When rank retrieval succeeds, choice approximates Take The Best and lower-ranked tally direction has little influence. When retrieval fails, cue ranks are compressed into a coarse common category, and the person pools the directions of all discriminating cues with approximately equal weight. Strategy arbitration occurs anew on every trial rather than assigning a person permanently to one heuristic. Individual capacity determines the transition location, gate steepness determines how abrupt it is, evidence sensitivity controls response reliability, and lapses produce occasional uninformed choices. Thus, lexicographic and tally-like rules are limiting regimes of a single adaptive representational mechanism.

**Rationale:** The two fixed heuristics fail in opposite directions: pure TTB correctly removes the tally slope in the seven-feature task but predicts strongly TTB-consistent choices in the ten-feature conflict task, whereas pure tallying produces excessive tally influence in both. The proposed model addresses this with a within-subject probabilistic gate driven by task complexity rather than experiment identity. With seven cues, observable load lies far below the sampled capacity range, so the pooling gate is nearly closed and lower-cue tally direction has almost no effect. With ten cues, load is near or above capacity, yielding a predominantly—but not exclusively—pooled regime. The selected gate range gives approximately the intermediate mixture needed to attenuate the opposing predictions of pure TTB and pure tallying toward the observed positive Experiment 2 contrast. Trial-level arbitration and response noise also avoid unrealistically classifying each subject as a permanent heuristic user. Because the same load equation applies for any feature count, the model predicts a smooth, testable transition at intermediate set sizes: increasing feature count or the fraction of simultaneously discriminating cues should progressively increase tally-consistent responding.

**Parameters:**
  - `n_features`: `n_features`
  - `validities`: `validities`
  - `capacity`: `[9.7, 9.9]`
  - `gate_steepness`: `[2.0, 2.4]`
  - `evidence_sensitivity`: `[1.0, 5.0]`
  - `lapse`: `[0.0, 0.15]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Rank-Budget Arbitration expects shape (2, n_features); got {stim.shape}."
        )

    n_features = int(parameters["n_features"])
    if stim.shape[1] != n_features:
        raise ValueError(
            f"Stimulus has {stim.shape[1]} features but n_features={n_features}."
        )

    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError("validities must contain one value per feature.")

    diff = stim[0] - stim[1]
    directions = np.sign(diff)
    discriminating = np.flatnonzero(directions != 0)

    # Successful rank retrieval: use only the most valid discriminating cue.
    if discriminating.size == 0:
        lexical_delta = 0.0
    else:
        cue_order = np.argsort(-validities, kind="stable")
        lexical_delta = 0.0
        for cue in cue_order:
            if directions[cue] != 0:
                lexical_delta = float(directions[cue])
                break

    # Failed rank retrieval: cue ranks are compressed and directions are pooled.
    # The sign transform prevents larger tallies from receiving mechanically
    # greater response precision than lexicographic evidence.
    pooled_delta = float(np.sign(np.sum(directions)))

    # Besides cue-set size, simultaneous discriminations add comparison load.
    comparison_fraction = float(discriminating.size) / float(max(n_features, 1))
    observable_load = float(n_features) + 0.25 * comparison_fraction

    capacity = float(parameters["capacity"])
    steepness = float(parameters["gate_steepness"])
    gate_logit = steepness * (observable_load - capacity)
    gate_logit = float(np.clip(gate_logit, -60.0, 60.0))
    pooling_probability = 1.0 / (1.0 + np.exp(-gate_logit))

    sensitivity = float(parameters["evidence_sensitivity"])

    def binary_softmax(delta):
        logits = np.array([
            0.5 * sensitivity * delta,
            -0.5 * sensitivity * delta
        ], dtype=float)
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs /= probs.sum()
        return probs

    p_lexical = binary_softmax(lexical_delta)
    p_pooled = binary_softmax(pooled_delta)
    p_core = ((1.0 - pooling_probability) * p_lexical
              + pooling_probability * p_pooled)

    lapse = float(parameters["lapse"])
    probs = (1.0 - lapse) * p_core + lapse * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, None)
    probs /= probs.sum()
    return probs
```

**`policy(probs)`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / len(probs)
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))
```


## Replacement

### `pi_5` → slot 2 (via `new_theory`)

**Description:** Relational Coalition Consolidation theory proposes that people retain the advertised validity hierarchy while also learning recurring relationships among lower-ranked cues. On each encounter, the learner represents lower cues relative to the highest-validity discriminating cue and records which cues jointly oppose that leader. Repeated, internally coherent opposition patterns become relational chunks: their accessibility increases with recurrence, pairwise co-opposition, consensus, and repeated conflict with the leading cue. Accessibility changes gradually and exhibits hysteresis, so an established coalition remains available across occasional mismatching trials while an isolated coalition does not immediately displace rank-based processing. Decision evidence is a continuous competition between a graded rank signal, strengthened by advertised validity dominance, and a graded coalition signal reflecting the validity-weighted direction of lower cues. Coalition magnitude is compressed so that learned accessibility, rather than the raw size of the current tally margin, carries more of the influence. Cue count alone therefore does not cause pooling. Many cues can preserve lexical dominance when their coalitional membership rotates, whereas a smaller but recurrent and coherent anti-leader coalition can capture choice. An unusually dominant advertised leader additionally retains a bounded residual influence, preventing complete coalition capture without globally suppressing ordinary coalition learning. The account predicts order-dependent acquisition, persistence after coalition reversals, and different choices for identical displays following stable versus unstable coalition histories.

**Rationale:** This is a minimal edit to the accepted iteration-1 model. Learning, prototype matching, activation, hysteresis, and the original convex arbitration are unchanged. First, the lower-cue weighted mean receives a concave signed transform. This preserves direction and graded evidence but compresses differences between weak and strong current tally margins, making expressed coalition influence depend relatively more on learned accessibility. It should reduce the spurious raw-tally modulation in Experiment 1 while strengthening weak but consolidated coalition signals in Experiments 2, 4, and 5; unanimous coalitions such as those prominent in Experiment 6 remain unchanged because magnitude one is fixed by the transform. Second, a sharply thresholded, bounded residual-rank safeguard is applied after the original arbitration. It is designed to activate only for an unusually large advertised validity gap, restoring lexical polarity in Experiment 3 without repeating the rejected global dominance suppression or universal additive log-odds rank term. Ordinary validity hierarchies retain the accepted base's strong coalition expression.

**Parameters:**
  - `n_features`: `n_features`
  - `validities`: `validities`
  - `learning_rate`: `[0.08, 0.14]`
  - `relation_prior`: `[0.8, 1.6]`
  - `recurrence_rate`: `[2.5, 4.5]`
  - `coalition_threshold`: `[0.16, 0.25]`
  - `gate_steepness`: `[8.0, 13.0]`
  - `activation_retention`: `[0.78, 0.88]`
  - `hysteresis_strength`: `[0.45, 0.75]`
  - `initial_access`: `[0.18, 0.30]`
  - `access_floor`: `[0.34, 0.44]`
  - `access_ceiling`: `[0.82, 0.94]`
  - `dominance_gain`: `[1.0, 2.2]`
  - `coalition_gain`: `[3.2, 4.8]`
  - `consensus_exponent`: `[0.40, 0.60]`
  - `dominance_threshold`: `[0.58, 0.68]`
  - `dominance_steepness`: `[25.0, 40.0]`
  - `rank_safeguard`: `[0.78, 0.94]`
  - `evidence_sensitivity`: `[1.0, 1.8]`
  - `lapse`: `[0.0, 0.06]`

**`predict(parameters, stimulus, history)`:**
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

**`policy(probs)`:**
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
