# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_1" and "pi_6") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_1" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_6" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_1
People compare two options by consulting cues one at a time in order of validity, stopping at the first cue that discriminates between the two options. That cue alone determines the choice: the option with the higher value on the discriminating cue wins, and no other cue is consulted. Cues with lower validity are never reached when a higher-validity cue already discriminates, so Take The Best is a "one-reason" decision rule — only a single feature is ever used on any given choice. Because only the sign of the comparison on the top discriminating cue matters, TTB ignores both the magnitude of that difference and all information on lower-validity cues, making it maximally frugal in the use of evidence. Cue validities are subjective and can be learned, inferred from instructions, or fixed by task structure. When no cue discriminates (all feature-wise comparisons tie) the learner has no basis for preference and must guess. Response noise enters through a softmax over the binary TTB score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    # Paper-faithful Take The Best (Gigerenzer & Goldstein 1996).
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B. Cue cascade: features are consulted in order
    # of descending validity; the first discriminating cue (strict
    # inequality) determines the winner; if no cue discriminates,
    # the model guesses uniformly. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )
    # Descending validity; argsort is stable so validity ties break
    # toward the earlier feature index.
    cue_order = np.argsort(-val, kind="stable").tolist()

    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    if winner is None:
        # No discriminating cue — pure guess.
        return np.ones(2) / 2.0

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. For the
    # binary TTB score this collapses to sigmoid(beta) for the winner,
    # giving a direct mapping from beta onto the paper's flip-noise
    # levels (beta=0 ↔ 50/50; beta ≫ 1 ↔ deterministic).
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


## THEORY 2 — pi_6
Competitive Source-Binding Cascade. Communicated validities establish an ordinal, usually lexicographic cue hierarchy, but successful use of a cue requires binding its validity label, display identity, and current recommendation into one retrievable representation. Binding fidelity is determined jointly rather than by independent cue accessibility. In particular, nearby discriminating cues with similar compressed validities and the same recommendation form a competitive cluster. Members of such a cluster mutually inhibit one another and can become collectively difficult to identify, so adding redundant support can reduce rather than increase use of the highest-validity cue. When the leading cue is not retrieved, search falls through the validity hierarchy, often allowing isolated, salient opposing cues to control choice. Validities are represented nonlinearly, with moderately diagnostic labels especially vulnerable to source-binding uncertainty. Display position contributes modest primacy and a terminally localized recency boost but no obligatory late-position cost. Thus, pure terminal position can be beneficial while dense terminal redundancy remains harmful because competition overwhelms recency. In unusually long displays, memory load additionally produces a zero-mean early-to-late binding contrast: early sources receive modest interference while late sources retain greater temporal distinctiveness. Stable individual differences govern validity compression, the location and width of maximal binding uncertainty, competitive interference, serial-position biases, long-display positional contrast, and response reversal.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Competitive Source-Binding Cascade expects shape (2, n_features); got {stim.shape}."
        )

    n_features = int(parameters["n_features"])
    if stim.shape[1] != n_features:
        raise ValueError(
            f"Stimulus has {stim.shape[1]} features but n_features={n_features}."
        )

    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities must have length {n_features}; got shape {validities.shape}."
        )

    compression = float(parameters["validity_compression"])
    baseline = float(parameters["baseline_binding"])
    binding_uncertainty = float(parameters["binding_uncertainty"])
    uncertainty_center = float(parameters["uncertainty_center"])
    uncertainty_width = float(parameters["uncertainty_width"])
    interference = float(parameters["interference_strength"])
    intermediate_competition = float(parameters["intermediate_competition"])
    crowd_exponent = float(parameters["crowding_exponent"])
    similarity_bandwidth = float(parameters["similarity_bandwidth"])
    neighborhood_width = float(parameters["neighborhood_width"])
    primacy = float(parameters["primacy_strength"])
    recency = float(parameters["recency_strength"])
    long_display_position = float(parameters["long_display_position_strength"])
    execution_flip = float(parameters["execution_flip"])

    differences = stim[0] - stim[1]
    directions = np.sign(differences)
    discriminating = directions != 0.0
    if not np.any(discriminating):
        return np.array([0.5, 0.5], dtype=np.float64)

    # Compress communicated validity onto a bounded subjective diagnosticity
    # scale. Ranking remains based on the communicated values, while distances
    # used in source competition are subjective and compressed.
    objective_strength = np.clip((validities - 0.5) / 0.5, 0.0, 1.0)
    subjective_strength = np.power(objective_strength, compression)

    positions = np.arange(n_features, dtype=np.float64)
    serial_scale = max(0.12 * max(n_features - 1, 1), 1.0)
    primacy_profile = np.exp(-positions / serial_scale)
    recency_profile = np.exp(-(n_features - 1.0 - positions) / serial_scale)

    # An additional zero-mean positional contrast emerges only under memory
    # loads exceeding 24 cues. It is therefore absent in the 18- and 24-cue
    # displays while reaching full strength at 32 cues. Centering preserves
    # average binding support rather than globally increasing accessibility.
    long_display_load = float(np.clip((n_features - 24.0) / 8.0, 0.0, 1.0))
    if n_features > 1:
        long_position_profile = 2.0 * positions / (n_features - 1.0) - 1.0
    else:
        long_position_profile = np.zeros(n_features, dtype=np.float64)

    # Binding uncertainty is concentrated around a subject-specific region of
    # moderately diagnostic labels. This permits the .8177 region to be more
    # confusable than near-equal .55 labels without changing validity rank.
    uncertainty_scale = max(uncertainty_width, 1e-9)
    uncertainty_profile = np.exp(
        -0.5 * np.square(
            (subjective_strength - uncertainty_center) / uncertainty_scale
        )
    )

    # Compute display-contingent competitive crowding. Only cues that actually
    # discriminate on this trial enter the competition. Same-direction cues
    # inhibit one another most when they are nearby and have similar compressed
    # validities. Consequently, redundant cues can jointly suppress retrieval.
    crowding = np.zeros(n_features, dtype=np.float64)
    active = np.flatnonzero(discriminating)
    sim_scale = max(similarity_bandwidth, 1e-9)
    pos_scale = max(neighborhood_width, 1e-9)

    for j in active:
        competitors = active[active != j]
        if competitors.size == 0:
            continue
        same_direction = directions[competitors] == directions[j]
        competitors = competitors[same_direction]
        if competitors.size == 0:
            continue

        position_similarity = np.exp(
            -np.abs(positions[competitors] - positions[j]) / pos_scale
        )
        validity_similarity = np.exp(
            -np.abs(
                subjective_strength[competitors] - subjective_strength[j]
            ) / sim_scale
        )
        crowding[j] = float(np.sum(position_similarity * validity_similarity))

    # Moderately diagnostic sources are especially vulnerable when they must
    # be distinguished from a cluster of similar sources. This selectively
    # strengthens competition among grouped 70% cues while allowing a slightly
    # shallower global density penalty for very high-validity terminal clusters.
    competition_cost = (
        interference
        * np.power(crowding, crowd_exponent)
        * (1.0 + intermediate_competition * uncertainty_profile)
    )
    binding_logits = (
        baseline
        - binding_uncertainty * uncertainty_profile
        - competition_cost
        + primacy * primacy_profile
        + recency * recency_profile
        + long_display_position * long_display_load * long_position_profile
    )

    # Stable logistic transformation to cue-binding probabilities.
    binding_prob = np.empty(n_features, dtype=np.float64)
    nonnegative = binding_logits >= 0.0
    binding_prob[nonnegative] = 1.0 / (
        1.0 + np.exp(-binding_logits[nonnegative])
    )
    exp_logits = np.exp(binding_logits[~nonnegative])
    binding_prob[~nonnegative] = exp_logits / (1.0 + exp_logits)
    binding_prob = np.clip(binding_prob, 1e-9, 1.0 - 1e-9)

    # Validity-guided cascade: the first successfully bound discriminating cue
    # controls the intended response. Stable sorting uses display order only to
    # resolve exact validity ties. We marginalize over residual binding outcomes
    # after the display-level competitive costs have been computed.
    cue_order = np.argsort(-validities, kind="stable")
    reach_probability = 1.0
    intended_a = 0.0
    intended_b = 0.0

    for j in cue_order:
        if not discriminating[j]:
            continue
        stop_probability = reach_probability * binding_prob[j]
        if differences[j] > 0.0:
            intended_a += stop_probability
        else:
            intended_b += stop_probability
        reach_probability *= 1.0 - binding_prob[j]

    # If no discriminating source is successfully bound, the person guesses.
    intended_a += 0.5 * reach_probability
    intended_b += 0.5 * reach_probability
    total = intended_a + intended_b
    if not np.isfinite(total) or total <= 0.0:
        intended_a = 0.5
    else:
        intended_a /= total

    # A separate execution process can reverse the intended binary response.
    p_a = (
        intended_a * (1.0 - execution_flip)
        + (1.0 - intended_a) * execution_flip
    )
    p_a = float(np.clip(p_a, 0.0, 1.0))
    probs = np.array([p_a, 1.0 - p_a], dtype=np.float64)
    probs /= probs.sum()
    return probs

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

## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
**Validities (n_features=32):** [0.55, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.92, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]

**Trial pairs (n=12):**
  trial 1: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 10: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]
  trial 12: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

**Rationale:** The six base conditions hold both the amount and direction of evidence constant while manipulating only the spatial density of a same-direction cue cluster. Expert 17 (index 16) has the uniquely highest validity, 92%, and always favors A. Six 90%-valid experts also favor A, while an isolated 55%-valid expert favors B. Across conditions, the six redundant A cues move symmetrically away from Expert 17: their offsets are ±{1,2,3}, ±{1,3,5}, ±{2,4,6}, ±{2,5,8}, ±{3,7,11}, and ±{5,10,15}. This symmetry approximately balances early-versus-late positional effects in the 32-cue display, and the number, validities, and recommendations of all discriminating cues remain fixed. Take The Best always stops at the 92% cue and therefore predicts one condition-invariant probability q=(1-epsilon)*sigmoid(beta)+epsilon/2 of choosing A. The Competitive Source-Binding Cascade instead predicts a pronounced spacing gradient. In the densest condition, the seven nearby, similarly valid A sources form a mutually inhibiting cluster; even the leading 92% source receives crowding from both sides, permitting the isolated B source to control many choices. As spacing increases, positional similarity and hence crowding decline exponentially, so successful retrieval of at least one A source should increase and A choices should rise, potentially crossing from below to above chance. This is an adversarial redundancy effect: distributing exactly the same supporting evidence improves its use, whereas TTB is indifferent to both redundancy and spacing. Exact A/B reversals balance response-side preferences and should invert the competing theory's spacing profile without changing its magnitude. The 12 unique pairs are repeated eight times, yielding 96 trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
This is the within-subject increase in following the uniquely highest-validity cue when its six redundant supporters are sparse (mean offsets 7 or 10) rather than dense/intermediate (offsets 2–5). Take The Best predicts the same probability of following Expert 17 in every arrangement, so its expected contrast is zero even when beta, lapse rate, or overall response accuracy differ across subjects. The competing cascade predicts substantially greater congruence in sparse displays because reduced source competition restores retrieval of Expert 17 or another same-direction high-validity cue. Pooling 32 sparse and 64 dense observations per subject yields a stable estimate, while coding choices relative to Expert 17 and including exact A/B reversals cancels fixed response-side preferences.

Source:
def metric(data: pd.DataFrame) -> float:
    # Choice congruence with the uniquely most valid cue (Expert 17, index 16),
    # contrasted between the two sparsest and four densest cue arrangements.
    congruent_sparse = []
    congruent_dense = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b

        if diff.size <= 16 or diff[16] == 0:
            continue

        leading_direction = np.sign(diff[16])
        leading_response = 0 if leading_direction > 0 else 1
        congruent = float(int(row["response"]) == leading_response)

        # Recover the arrangement from the six other cues supporting the same
        # option as Expert 17. Their mean distance from index 16 is exactly one
        # of 2, 3, 4, 5, 7, or 10 in this design.
        indices = np.arange(diff.size)
        supporters = indices[(np.sign(diff) == leading_direction) & (indices != 16)]
        if supporters.size == 0:
            continue
        spread = float(np.mean(np.abs(supporters - 16)))

        if spread > 6.0:
            congruent_sparse.append(congruent)
        else:
            congruent_dense.append(congruent)

    if len(congruent_sparse) == 0 or len(congruent_dense) == 0:
        return 0.0

    return float(np.mean(congruent_sparse) - np.mean(congruent_dense))

### RESULTS
- Predicted under pi_1 (simulated): -0.0003 (var=0.0059)
- Predicted under pi_6 (simulated): 0.3762 (var=0.0253)
- Observed on real data: -0.0056 (var=0.0030)

## EXPERIMENT 2 (proposed by pi_6)

### DESIGN
**Validities (n_features=32):** [0.82, 0.82, 0.5, 0.95, 0.5, 0.5, 0.82, 0.82, 0.5, 0.5, 0.5, 0.5, 0.82, 0.82, 0.5, 0.55, 0.5, 0.5, 0.82, 0.82, 0.5, 0.5, 0.5, 0.5, 0.82, 0.82, 0.5, 0.5, 0.5, 0.5, 0.82, 0.82]

**Trial pairs (n=12):**
  trial 1: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]

**Rationale:** This design translates a fixed two-cue competitive cluster across a 32-expert display while holding its size, within-cluster spacing, validity, recommendation, and opposing evidence constant. In each base pair, exactly two adjacent 82%-valid experts favor A, and the isolated 55%-valid Expert 16 favors B. The six A-cluster locations are positions 1–2, 7–8, 13–14, 19–20, 25–26, and 31–32. The 95%-valid Expert 4 always ties. Take The Best therefore skips the tied 95% expert and stops at the first active 82% expert, predicting the same A-choice probability q=(1-epsilon)*sigmoid(beta)+epsilon/2 in every condition. The Competitive Source-Binding Cascade predicts a different quantitative profile. Because the two A cues are adjacent, equally valid, and recommend the same option, their mutual competitive crowding is identical across conditions; because their moderate validity lies near the model's binding-uncertainty region, failed retrieval should frequently permit the fixed, isolated B cue to control choice. Translating the cluster later selectively increases its binding support through the 32-cue long-display positional contrast, with an additional sharply localized recency benefit for the terminal cluster. Thus the model predicts increasing A choices from early to late cluster locations, typically with the largest improvement at positions 31–32 and potentially a crossover from following B to following A. This isolates serial-position recovery from cluster density, unlike a spacing manipulation: all competitive relationships within the cluster remain unchanged. The final six pairs are exact A/B reversals, balancing response-side preferences while preserving the predicted location gradient. Twelve unique pairs repeated eight times yield 96 trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
This is the terminal-cluster recovery contrast: the rate of choosing the option supported by the adjacent 82%-valid cue pair at positions 31–32 minus the corresponding rate pooled across positions 1–2 through 25–26. Scoring choices relative to the cluster, rather than as raw A choices, combines the exact A/B reversals and removes response-side preference. Pooling all five nonterminal locations provides 80 comparison trials per subject while retaining 16 terminal trials, reducing sampling variance. Take The Best predicts the same cluster-following probability at every location, so its expected contrast is zero for every parameter setting. The Competitive Source-Binding Cascade predicts a sizable positive contrast because the terminal pair receives both the long-display positional advantage and the sharply localized recency boost. This is a substantive behavioral interaction with display position rather than a direct restatement of an internal model quantity.

Source:
def metric(data: pd.DataFrame) -> float:
    # Score whether each response follows the adjacent, same-direction cue pair.
    # The six possible zero-based starts are fixed by the experimental design.
    starts = (0, 6, 12, 18, 24, 30)
    follows_cluster = []
    is_terminal = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        d = a - b

        cluster_start = None
        cluster_direction = 0.0
        for s in starts:
            if d[s] != 0.0 and d[s + 1] == d[s]:
                cluster_start = s
                cluster_direction = d[s]
                break

        if cluster_start is None:
            continue

        # Positive direction means the cluster supports A (response 0), while
        # negative direction means it supports B (response 1).
        cluster_response = 0 if cluster_direction > 0.0 else 1
        follows_cluster.append(float(int(row["response"]) == cluster_response))
        is_terminal.append(cluster_start == 30)

    follows_cluster = np.asarray(follows_cluster, dtype=float)
    is_terminal = np.asarray(is_terminal, dtype=bool)

    if follows_cluster.size == 0 or not np.any(is_terminal) or not np.any(~is_terminal):
        return float("nan")

    terminal_rate = float(np.mean(follows_cluster[is_terminal]))
    earlier_rate = float(np.mean(follows_cluster[~is_terminal]))
    return terminal_rate - earlier_rate


### RESULTS
- Predicted under pi_1 (simulated): -0.0100 (var=0.0083)
- Predicted under pi_6 (simulated): 0.3420 (var=0.0223)
- Observed on real data: -0.0045 (var=0.0098)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1]
  A=[1, 0, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 1, 0, 1]  B=[1, 0, 0, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Classify whether choices are predominantly controlled by the highest-ranked
    # discriminating cue on trials where at least three net lower-cue votes point
    # in the opposite direction.
    agreements = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        d = a - b

        nz = np.flatnonzero(d != 0)
        if nz.size == 0:
            continue

        # Features are already ordered from highest to lowest validity.
        first = int(nz[0])
        ttb_sign = 1.0 if d[first] > 0 else -1.0  # +1 means A, -1 means B
        tally_margin = float(np.sum(np.sign(d)))  # positive means tally favors A

        # Retain only strong conflicts: tally favors the other option by at
        # least three net cue votes. This selects patterns 2, 3, 5, and 6 and
        # their A/B reversals in the specified design.
        if ttb_sign * tally_margin < 0 and abs(tally_margin) >= 3.0:
            ttb_response = 0 if ttb_sign > 0 else 1
            agreements.append(float(int(row["response"]) == ttb_response))

    if len(agreements) == 0:
        return 0.5

    # A categorical dominance index: 1 if the top discriminating cue controls
    # a strict majority of these high-conflict choices, otherwise 0.
    return float(np.mean(agreements) > 0.5)

```

**Observed (real) value:** 1.0000 (var=0.0000)
**Predicted under pi_1:** 1.0000 (var=0.0000)
**Predicted under pi_6:** 1.0000 (var=0.0384)

### Experiment 4
**Design**
  A=[1, 1, 0, 1, 0, 0, 1, 0]  B=[0, 0, 1, 0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 1, 0, 0]
  A=[1, 1, 0, 1, 0, 1, 0, 1]  B=[0, 0, 1, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 1, 0, 1, 1]  B=[0, 0, 0, 1, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 1, 0, 1]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 1, 1, 0, 0]  B=[1, 1, 0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 0, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 1, 0, 0]  B=[1, 1, 1, 0, 1, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 1, 1, 1, 0, 1, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """AIC model-selection indicator for graded tally-margin sensitivity."""
    if data is None or len(data) == 0:
        return 0.0

    margins = []
    successes = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        signed_margin = int(np.sum(a > b) - np.sum(b > a))
        if signed_margin == 0:
            continue

        winner = 0 if signed_margin > 0 else 1
        margins.append(abs(signed_margin))
        successes.append(float(int(row["response"]) == winner))

    if len(successes) == 0:
        return 0.0

    margins = np.asarray(margins, dtype=int)
    successes = np.asarray(successes, dtype=float)
    levels = np.arange(1, 7, dtype=float)
    n_by_margin = np.asarray([np.sum(margins == m) for m in range(1, 7)], dtype=float)
    k_by_margin = np.asarray(
        [np.sum(successes[margins == m]) for m in range(1, 7)], dtype=float
    )
    keep = n_by_margin > 0
    levels = levels[keep]
    n_by_margin = n_by_margin[keep]
    k_by_margin = k_by_margin[keep]

    if len(levels) < 2:
        return 0.0

    # Best margin-invariant winner probability: the observable implication
    # of TTB for this design. Its parameterization permits probabilities
    # arbitrarily close to the full interval [0.5, 1].
    n = float(np.sum(n_by_margin))
    k = float(np.sum(k_by_margin))
    q_const = float(np.clip(k / n, 0.5 + 1e-12, 1.0 - 1e-12))
    ll_ttb = k * np.log(q_const) + (n - k) * np.log1p(-q_const)

    # Maximize the exact Tallying likelihood over its stated beta and lapse
    # ranges. The grid is denser where sigmoid(beta*m) changes most rapidly.
    beta = np.concatenate(
        [np.linspace(0.1, 3.0, 241), np.linspace(3.1, 20.0, 170)]
    )
    epsilon = np.linspace(0.0, 0.5, 251)

    core = 1.0 / (1.0 + np.exp(-beta[:, None] * levels[None, :]))
    q = (
        (1.0 - epsilon[None, :, None]) * core[:, None, :]
        + epsilon[None, :, None] * 0.5
    )
    q = np.clip(q, 1e-12, 1.0 - 1e-12)

    ll = np.sum(
        k_by_margin[None, None, :] * np.log(q)
        + (n_by_margin - k_by_margin)[None, None, :] * np.log1p(-q),
        axis=2,
    )
    ll_tally = float(np.max(ll))

    # Tallying has one additional identifiable probability-shape parameter.
    # Thus AIC favors it exactly when the likelihood-ratio deviance exceeds 2.
    deviance = 2.0 * (ll_tally - ll_ttb)
    return float(deviance > 2.0)

```

**Observed (real) value:** 0.0000 (var=0.0000)
**Predicted under pi_1:** 0.0000 (var=0.0736)
**Predicted under pi_6:** 0.0000 (var=0.0564)

### Experiment 5
**Design**
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Return 1 when first-discriminating-cue consistency is at least 90%."""
    if len(data) == 0:
        return float("nan")

    follows_first_cue = []
    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"],
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        expected_response = None
        for j in range(min(a.size, b.size)):
            if a[j] > b[j]:
                expected_response = 0
                break
            if b[j] > a[j]:
                expected_response = 1
                break

        if expected_response is not None:
            follows_first_cue.append(int(int(response) == expected_response))

    if len(follows_first_cue) == 0:
        return float("nan")

    consistency = float(np.mean(follows_first_cue))
    return float(consistency >= 0.90)

```

**Observed (real) value:** 0.0000 (var=0.0000)
**Predicted under pi_1:** 0.0000 (var=0.2400)
**Predicted under pi_6:** 0.0000 (var=0.0000)

### Experiment 6
**Design**
  A=[1, 1, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 0, 1, 0, 1, 0]  B=[0, 0, 1, 1, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 1, 0, 1]  B=[1, 0, 1, 1, 1, 1, 0, 1]
  A=[0, 1, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 1, 0, 1]
  A=[0, 0, 1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 1, 1, 0, 1]  B=[1, 1, 0, 0, 0, 1, 0, 1]
  A=[0, 0, 1, 1, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0, 1, 0]
  A=[1, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Proportion of choices following the net direction of all weak cues on
    # trials where that direction opposes Expert 2. These are conflict levels
    # M=2,...,6; the M=1 cancellation trials are omitted.
    scores = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        directions = np.sign(a - b)

        # Feature 0 always ties. Features 1 onward are the equally valid weak
        # cues that the advocated theory integrates.
        weak_net = float(np.sum(directions[1:]))
        expert2_direction = float(directions[1])

        # Retain only genuine reversal trials: the weak-cue aggregate points
        # opposite to the first discriminating cue used by TTB.
        if weak_net == 0.0 or expert2_direction == 0.0:
            continue
        if np.sign(weak_net) == np.sign(expert2_direction):
            continue

        integrated_winner = 0 if weak_net > 0.0 else 1
        scores.append(float(int(row["response"]) == integrated_winner))

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** 0.1620 (var=0.0198)
**Predicted under pi_1:** 0.1670 (var=0.0108)
**Predicted under pi_6:** 0.1693 (var=0.0024)

### Experiment 7
**Design**
  A=[1, 0, 1, 1, 1, 1, 1, 0, 1]  B=[1, 0, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 0, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 0, 1, 1, 1, 1, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 0, 1, 1, 1, 1, 0, 0, 1]
  A=[1, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 0, 1, 1, 1, 0, 0, 0, 1]
  A=[1, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 0, 1, 1, 0, 0, 0, 0, 1]
  A=[1, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 0, 1, 0, 0, 0, 0, 0, 1]
  A=[1, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Rate at which the 70%-cue majority overrides the opposing 72% cue
    on the two strongest conflict levels (five or six active 70% cues)."""
    matches = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.size < 9 or b.size < 9:
            continue

        # Features 3--8 (zero-based indices 2:8) are the six 70% cues.
        d70 = a[2:8] - b[2:8]
        active = d70[d70 != 0]
        if active.size < 5:
            continue

        # Retain only designed conflicts in which the active 70% cues agree
        # and the 72% cue recommends the opposite option.
        direction = np.sign(active[0])
        if not np.all(np.sign(active) == direction):
            continue
        d72 = a[8] - b[8]
        if d72 == 0 or np.sign(d72) == direction:
            continue

        majority_choice = 0 if direction > 0 else 1
        response = int(row["response"])
        if response in (0, 1):
            matches.append(float(response == majority_choice))

    if len(matches) == 0:
        return 0.0
    return float(np.mean(matches))
```

**Observed (real) value:** 0.1613 (var=0.0081)
**Predicted under pi_1:** 0.1450 (var=0.0101)
**Predicted under pi_6:** 0.1850 (var=0.0060)

### Experiment 8
**Design**
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    validities = np.array([
        0.55, 0.95, 0.5525, 0.55, 0.5, 0.5868,
        0.55, 0.95, 0.6343, 0.55, 0.5, 0.6901,
        0.55, 0.95, 0.7504, 0.55, 0.5, 0.8177
    ], dtype=float)
    log_odds = np.log(validities / (1.0 - validities))

    gaps = []
    follows_higher = []

    for a_cell, b_cell, response in zip(
        data['option_a_ratings'],
        data['option_b_ratings'],
        data['response']
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        diff = a - b
        discriminating = np.flatnonzero(diff != 0)
        if discriminating.size != 2:
            continue

        j0, j1 = int(discriminating[0]), int(discriminating[1])
        if validities[j0] == validities[j1]:
            continue
        high = j0 if validities[j0] > validities[j1] else j1
        low = j1 if high == j0 else j0

        high_favors_a = diff[high] > 0
        chose_a = int(response) == 0
        follows_higher.append(float(chose_a == high_favors_a))
        gaps.append(float(log_odds[high] - log_odds[low]))

    gaps = np.asarray(gaps, dtype=float)
    follows_higher = np.asarray(follows_higher, dtype=float)
    if gaps.size == 0:
        return float('nan')

    unique_gaps = np.unique(np.round(gaps, 10))
    if unique_gaps.size < 2:
        return float('nan')

    smallest = unique_gaps[0]
    largest = unique_gaps[-1]
    low_mask = np.isclose(gaps, smallest, atol=1e-9, rtol=0.0)
    high_mask = np.isclose(gaps, largest, atol=1e-9, rtol=0.0)

    return float(
        np.mean(follows_higher[high_mask])
        - np.mean(follows_higher[low_mask])
    )
```

**Observed (real) value:** -0.1275 (var=0.0523)
**Predicted under pi_1:** 0.0113 (var=0.0129)
**Predicted under pi_6:** -0.0913 (var=0.0324)

### Experiment 9
**Design**
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Measure whether choices favoring the decisive high-validity cue decline
    # as that cue moves toward the end of the display.
    positions = np.array([2, 7, 13, 19, 25, 31], dtype=int)
    # Approximate centered accessibility-cascade profile. The weights sum to
    # zero, so any position-invariant choice reliability contributes zero.
    weights = np.array([0.27, 0.24, 0.16, -0.02, -0.25, -0.40], dtype=float)

    weighted_choices = []
    used_weights = []

    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        diff = a - b

        # In this design, exactly one of these six positions contains the
        # higher-validity target cue that opposes the two early cues.
        target_matches = [j for j in positions if j < diff.size and diff[j] != 0]
        if len(target_matches) != 1:
            continue

        target_pos = target_matches[0]
        k = int(np.where(positions == target_pos)[0][0])
        w = weights[k]

        response = int(row['response'])
        target_choice = 1.0 if ((diff[target_pos] > 0 and response == 0) or
                                (diff[target_pos] < 0 and response == 1)) else 0.0
        weighted_choices.append(w * target_choice)
        used_weights.append(abs(w))

    if len(weighted_choices) == 0:
        return float('nan')

    # Absolute-weight normalization gives an interpretable bounded contrast
    # while preserving the matched-filter ordering of the six conditions.
    denom = float(np.sum(used_weights))
    if denom <= 0.0:
        return float('nan')
    return float(np.sum(weighted_choices) / denom)

```

**Observed (real) value:** -0.0543 (var=0.0193)
**Predicted under pi_1:** -0.0015 (var=0.0020)
**Predicted under pi_6:** -0.0510 (var=0.0058)

### Experiment 10
**Design**
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Measure choice of the option favored by the highest-validity cue,
    # separately for sparse versus highly redundant terminal-cue conditions.
    high_redundancy = []
    low_redundancy = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.size < 24 or b.size < 24:
            continue

        diff = a - b
        # Features 19--24 (zero-based indices 18--23) are the nested set
        # of high-validity cues favoring the same option.
        redundancy = int(np.count_nonzero(diff[18:24]))

        # Feature 24 has validity .95 and is TTB's decisive cue on every
        # experimental trial. Code whether the response follows that cue.
        if diff[23] > 0:
            followed_top_cue = float(int(row["response"]) == 0)
        elif diff[23] < 0:
            followed_top_cue = float(int(row["response"]) == 1)
        else:
            continue

        if redundancy in (1, 2):
            low_redundancy.append(followed_top_cue)
        elif redundancy in (5, 6):
            high_redundancy.append(followed_top_cue)

    if len(low_redundancy) == 0 or len(high_redundancy) == 0:
        return float("nan")

    return float(np.mean(high_redundancy) - np.mean(low_redundancy))

```

**Observed (real) value:** -0.2675 (var=0.0382)
**Predicted under pi_1:** 0.0169 (var=0.0052)
**Predicted under pi_6:** -0.2762 (var=0.0511)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all ten experiments, neither theory is task-invariant. pi_1 decisively captures Experiments 1 and 2: both observed contrasts are essentially zero (-0.0056 and -0.0045), matching pi_1's invariant predictions (-0.0003 and -0.0100) and sharply contradicting pi_6's large positive predictions (0.3762 and 0.3420). The low observed between-subject variance in Experiment 1 further suggests that the null spacing effect is systematic, not merely an average produced by opposing individual effects. Experiments 3-7 are broadly compatible with lexicographic control and do not separate the theories strongly: both match the pooled categorical outcomes in Experiments 3-5 and are close in Experiments 6-7, with pi_1 slightly closer on the canonical point estimates. However, pi_1 fails structurally on Experiments 8-10. It cannot generate the negative validity-gap contrast in Experiment 8, the positional decline in following an isolated decisive cue in Experiment 9, or the large redundancy penalty in Experiment 10. pi_6 captures all three well, especially Experiments 9 and 10, but its proposed local-spacing competition and terminal recovery mechanisms are directly disconfirmed by Experiments 1 and 2. These are not small calibration errors: pi_6 predicts effects around +0.35 where the data are effectively zero. Thus the evidence supports a lexicographic backbone but rejects both strict, position-invariant TTB and pi_6's particular combination of spatially local competition and terminal recency. Because pi_1's prose claim makes all lower-cue, spacing, redundancy, and position effects impossible by construction, parameter or implementation changes cannot repair its failures without abandoning the theory. pi_6 should remain as a competitor because it captures the otherwise diagnostic Experiments 8-10, while pi_1 should be replaced by a broader theory.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace pi_1 with a Chunked Source-Retrieval Lexicographic theory. Communicated validity still determines an intended descending search order, explaining top-cue dominance in Experiments 3-7, but cue use requires retrieval of cue identity, validity, and recommendation. Retrieval should have three constrained properties. First, interference should depend mainly on the number and validity similarity of active same-direction sources, not their physical spacing; holding the source set fixed must therefore predict the null spacing contrast in Experiment 1. Second, redundant adjacent sources may be encoded as a single recommendation chunk, making translation of a fixed two-cue cluster largely position-invariant and predicting the null terminal contrast in Experiment 2. Third, unchunked isolated sources should show modest primacy or late-display accessibility loss, with no generic terminal-recency bonus, allowing the negative positional profile in Experiment 9. Increasing the number of similar same-direction sources should increase source-attribution ambiguity or chunk overload, producing the redundancy penalty in Experiment 10 even though moving a fixed set farther apart does not help. Include nonlinear validity encoding with maximal source uncertainty in a moderate-to-high diagnosticity region to accommodate Experiment 8's inverse validity-gap effect. Once a cue or recommendation chunk is retrieved, choice should remain lexicographic, followed by a small execution lapse. Parameterization should be parsimonious and shared across feature counts: validity-compression/uncertainty parameters, a global similar-source load coefficient, a chunking probability based on adjacency and common direction, a weak singleton position slope, retrieval baseline, and response lapse. It should explicitly exclude pi_6's distance-decaying local inhibition and positive long-display terminal contrast, since those mechanisms generated the two large failed predictions."
}
```

## Usage

```json
{
  "input_tokens": 23793,
  "output_tokens": 1690
}
```
