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
**Validities (n_features=24):** [0.7, 0.55, 0.57, 0.59, 0.61, 0.63, 0.95, 0.65, 0.54, 0.56, 0.58, 0.6, 0.62, 0.82, 0.64, 0.82, 0.53, 0.82, 0.57, 0.82, 0.61, 0.82, 0.55, 0.82]

**Trial pairs (n=12):**
  trial 1: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  trial 10: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  trial 11: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  trial 12: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

**Rationale:** This 24-expert design isolates the advocated theory's terminal-recency mechanism without activating its long-display positional contrast, which is exactly zero at n_features=24. In each base pair, one 82%-valid singleton favors A and the fixed 70%-valid Expert 1 favors B; the 95%-valid expert and every other expert tie. Across six conditions, the otherwise identical 82% singleton appears at positions 14, 16, 18, 20, 22, and 24. There is no same-direction crowding because only one cue supports each option, and neither adjacency chunking nor recommendation-level shared gates can occur. Competitive Source-Binding Cascade predicts a sharply increasing, convex A-choice profile as the target approaches the display endpoint: its recency contribution rises from approximately exp(-10/2.76) to 1 across these positions, producing a terminal binding-logit gain of at least about 1.3 relative to position 14. When the 82% cue is not bound, search reaches the fixed 70% B cue, making target-binding changes consequential and potentially producing a crossover toward A at the terminal position. Chunked Source-Retrieval Lexicographic Search has no recency benefit. Its target retrieval probability is position-invariant except for the weak monotonic singleton late-position loss; across positions 14 to 24, that penalty changes by at most about 0.026 logits. It therefore predicts an approximately flat or slightly decreasing A-choice profile, directly opposing the advocated theory's pronounced terminal increase. Exact A/B reversals balance response-side preferences. The 12 unique pairs are repeated eight times, yielding 96 trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Competitive Source-Binding Cascade. Communicated validities establish an ordinal, usually lexicographic cue hierarchy, but successful use of a cue requires binding its validity label, display identity, and current recommendation into one retrievable representation. Binding fidelity is determined jointly rather than by independent cue accessibility. In particular, nearby discriminating cues with similar compressed validities and the same recommendation form a competitive cluster. Members of such a cluster mutually inhibit one another and can become collectively difficult to identify, so adding redundant support can reduce rather than increase use of the highest-validity cue. When the leading cue is not retrieved, search falls through the validity hierarchy, often allowing isolated, salient opposing cues to control choice. Validities are represented nonlinearly, with moderately diagnostic labels especially vulnerable to source-binding uncertainty. Display position contributes modest primacy and a terminally localized recency boost but no obligatory late-position cost. Thus, pure terminal position can be beneficial while dense terminal redundancy remains harmful because competition overwhelms recency. In unusually long displays, memory load additionally produces a zero-mean early-to-late binding contrast: early sources receive modest interference while late sources retain greater temporal distinctiveness. Stable individual differences govern validity compression, the location and width of maximal binding uncertainty, competitive interference, serial-position biases, long-display positional contrast, and response reversal.

**Parameters:**
- n_features: n_features
- validities: validities
- validity_compression: [0.85, 1.15]
- baseline_binding: [2.4, 3.6]
- binding_uncertainty: [2.8, 4.5]
- uncertainty_center: [0.52, 0.64]
- uncertainty_width: [0.14, 0.22]
- interference_strength: [1.6, 2.6]
- intermediate_competition: [0.8, 1.5]
- crowding_exponent: [2.0, 2.5]
- similarity_bandwidth: [0.16, 0.4]
- neighborhood_width: [1.5, 3.5]
- primacy_strength: [0.0, 0.08]
- recency_strength: [1.4, 2.4]
- long_display_position_strength: [0.45, 1.05]
- execution_flip: [0.08, 0.18]

**`predict source code`:**
```python
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
```

**`policy source code`:**
```python
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / len(probs)
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** Chunked Source-Retrieval Lexicographic Search. People intend to inspect expert recommendations in descending order of communicated validity, but a cue can govern choice only if its source identity, validity label, and current recommendation are jointly retrieved. Retrieval interference is source-based rather than spatial: active sources recommending the same option interfere in proportion to their subjective-validity similarity, regardless of their physical distance. Consequently, rearranging a fixed source set does not release interference. Adjacent sources giving the same recommendation can instead be encoded as a single recommendation chunk. A chunk enters the validity-ordered search at the rank of its most valid member and, if retrieved, directly recommends an option; this makes a translated two-source cluster nearly position-invariant. Chunk retrieval nevertheless retains the attribution load generated by its constituent sources, so increasing a redundant set can overload retrieval even when the recommendation is coherent. In addition, sets of at least three similar same-direction sources share a correlated source-attribution gate. If that common binding episode fails, all representations carrying that recommendation are temporarily unavailable; thus redundant cues cannot always rescue one another through independent retrieval attempts. The shared gate depends on source count and validity similarity but not physical spacing. Isolated sources have a very weak monotonic late-position accessibility loss, while chunks receive no positional bonus or penalty; there is no terminal-recency mechanism. Communicated validity is nonlinearly compressed, and source binding has a localized uncertainty peak in the moderate-to-high diagnosticity region. This can make a moderately high-validity source less retrievable than nearly nondiagnostic sources and thereby produce inverse validity-gap effects. Search remains lexicographic after retrieval: the first retrieved discriminating cue or chunk determines the intended response. If nothing is retrieved, the person guesses, and a small independent execution lapse can reverse the intended response.

**Parameters:**
- n_features: n_features
- validities: validities
- validity_compression: [0.85, 1.2]
- retrieval_baseline: [2.4, 3.2]
- uncertainty_strength: [1.0, 2.0]
- uncertainty_center: [0.58, 0.68]
- uncertainty_width: [0.12, 0.2]
- similar_source_load: [0.45, 0.9]
- chunk_probability: [0.85, 0.98]
- singleton_position_slope: [0.0, 0.06]
- response_lapse: [0.08, 0.16]

**`predict source code`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Chunked Source-Retrieval Lexicographic Search expects "
            f"shape (2, n_features); got {stim.shape}."
        )

    n_features = int(parameters["n_features"])
    if stim.shape[1] != n_features:
        raise ValueError(
            f"Stimulus has {stim.shape[1]} features but "
            f"n_features={n_features}."
        )

    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities must have length {n_features}; got "
            f"shape {validities.shape}."
        )

    compression = float(parameters["validity_compression"])
    baseline = float(parameters["retrieval_baseline"])
    uncertainty_strength = float(parameters["uncertainty_strength"])
    uncertainty_center = float(parameters["uncertainty_center"])
    uncertainty_width = float(parameters["uncertainty_width"])
    load_strength = float(parameters["similar_source_load"])
    chunk_probability = float(parameters["chunk_probability"])
    position_slope = float(parameters["singleton_position_slope"])
    lapse = float(parameters["response_lapse"])

    differences = stim[0] - stim[1]
    directions = np.sign(differences)
    active = np.flatnonzero(directions != 0.0)
    if active.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    # Nonlinear subjective diagnosticity. Search order still uses the
    # communicated validities, whereas similarity and uncertainty operate on
    # this compressed internal scale.
    objective_strength = np.clip((validities - 0.5) / 0.5, 0.0, 1.0)
    subjective_strength = np.power(objective_strength, compression)

    width = max(uncertainty_width, 1e-9)
    uncertainty = np.exp(
        -0.5 * np.square(
            (subjective_strength - uncertainty_center) / width
        )
    )

    # Similarity is deliberately independent of display distance. Only active
    # sources giving the same recommendation interfere. The fixed bandwidth
    # avoids introducing a separate spatial or neighborhood parameter.
    similarity_bandwidth = 0.18
    source_load = np.zeros(n_features, dtype=np.float64)
    for j in active:
        same = active[(active != j) & (directions[active] == directions[j])]
        if same.size:
            source_load[j] = float(np.sum(np.exp(
                -np.abs(
                    subjective_strength[same] - subjective_strength[j]
                ) / similarity_bandwidth
            )))

    # Similar same-direction sets also share a correlated attribution gate.
    # The gate is exact for a recommendation direction, so multiple units do
    # not provide independent rescue opportunities when source attribution
    # fails. Pairs are exempt to preserve stable translation of a two-cue
    # chunk. The equation contains no display-distance term.
    shared_gates = {}
    for direction_key in (-1, 1):
        group = active[directions[active] == float(direction_key)]
        if group.size <= 2:
            shared_gates[direction_key] = 1.0
            continue

        group_ambiguity = float(np.mean(source_load[group]))
        gate_logit = (
            baseline
            - 0.45 * load_strength * max(group_ambiguity, 0.0)
        )
        if gate_logit >= 0.0:
            gate_prob = 1.0 / (1.0 + np.exp(-gate_logit))
        else:
            ez = np.exp(gate_logit)
            gate_prob = ez / (1.0 + ez)
        shared_gates[direction_key] = float(np.clip(
            gate_prob, 1e-9, 1.0 - 1e-9
        ))

    def make_units(use_chunks):
        units = []
        active_set = set(int(x) for x in active)
        consumed = set()

        for j_raw in active:
            j = int(j_raw)
            if j in consumed:
                continue

            members = [j]
            if use_chunks:
                k = j + 1
                while (
                    k in active_set
                    and directions[k] == directions[j]
                ):
                    members.append(k)
                    k += 1

            for m in members:
                consumed.add(m)

            # A recommendation chunk is ranked by its most valid source.
            member_array = np.asarray(members, dtype=int)
            member_validities = validities[member_array]
            best_local = int(np.argmax(member_validities))
            representative = int(member_array[best_local])

            # Constituent-source ambiguity remains present after chunking.
            # Averaging source loads prevents mere chunk length from becoming
            # evidence accumulation, while source_load itself increases with
            # the number and similarity of redundant sources.
            attribution_load = float(np.mean(source_load[member_array]))
            load_cost = load_strength * np.power(
                max(attribution_load, 0.0), 1.25
            )

            retrieval_logit = (
                baseline
                - uncertainty_strength * uncertainty[representative]
                - load_cost
            )

            # Only an unchunked singleton receives the weak monotonic
            # late-display accessibility loss. There is no recency bonus.
            if len(members) == 1 and n_features > 1:
                relative_position = representative / float(n_features - 1)
                retrieval_logit -= position_slope * relative_position

            if retrieval_logit >= 0.0:
                retrieval_prob = 1.0 / (1.0 + np.exp(-retrieval_logit))
            else:
                ez = np.exp(retrieval_logit)
                retrieval_prob = ez / (1.0 + ez)

            units.append({
                "priority": float(validities[representative]),
                "tie_position": min(members),
                "direction": float(directions[j]),
                "gate_key": int(np.sign(directions[j])),
                "retrieval": float(np.clip(
                    retrieval_prob, 1e-9, 1.0 - 1e-9
                )),
            })

        # Descending validity, with display position used only for exact ties.
        units.sort(key=lambda u: (-u["priority"], u["tie_position"]))
        return units

    def cascade_probability_a(units, available_directions=None):
        reach = 1.0
        intended_a = 0.0
        intended_b = 0.0

        for unit in units:
            if (
                available_directions is not None
                and unit["gate_key"] not in available_directions
            ):
                continue
            stop = reach * unit["retrieval"]
            if unit["direction"] > 0.0:
                intended_a += stop
            else:
                intended_b += stop
            reach *= 1.0 - unit["retrieval"]

        # Failure to retrieve any discriminating representation yields a guess.
        intended_a += 0.5 * reach
        intended_b += 0.5 * reach
        total = intended_a + intended_b
        if not np.isfinite(total) or total <= 0.0:
            return 0.5
        return float(intended_a / total)

    def gated_cascade_probability_a(units):
        # Exactly marginalize the two recommendation-level gate states rather
        # than drawing a latent gate inside predict.
        p_positive = shared_gates[1]
        p_negative = shared_gates[-1]
        result = 0.0
        for positive_open, p_pos_state in (
            (False, 1.0 - p_positive),
            (True, p_positive),
        ):
            for negative_open, p_neg_state in (
                (False, 1.0 - p_negative),
                (True, p_negative),
            ):
                available = set()
                if positive_open:
                    available.add(1)
                if negative_open:
                    available.add(-1)
                result += (
                    p_pos_state
                    * p_neg_state
                    * cascade_probability_a(units, available)
                )
        return float(result)

    # Marginalize the encoding state rather than randomly choosing it inside
    # predict. This keeps returned probabilities stable and interpretable.
    p_a_unchunked = gated_cascade_probability_a(make_units(False))
    p_a_chunked = gated_cascade_probability_a(make_units(True))
    p_a_intended = (
        chunk_probability * p_a_chunked
        + (1.0 - chunk_probability) * p_a_unchunked
    )

    # Independent execution lapse reverses the intended binary response.
    p_a = (
        (1.0 - lapse) * p_a_intended
        + lapse * (1.0 - p_a_intended)
    )
    p_a = float(np.clip(p_a, 0.0, 1.0))
    probs = np.array([p_a, 1.0 - p_a], dtype=np.float64)
    probs /= probs.sum()
    return probs
```

**`policy source code`:**
```python
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / len(probs)
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Score whether each response followed the 82%-valid singleton rather than\n    # the fixed 70%-valid cue at feature 1.\n    positions = []\n    followed_target = []\n\n    for _, row in data.iterrows():\n        a = np.asarray(row['option_a_ratings'], dtype=np.float64)\n        b = np.asarray(row['option_b_ratings'], dtype=np.float64)\n        diff = a - b\n\n        # Every designed trial has cue 1 and exactly one later cue active.\n        active_late = np.flatnonzero((diff != 0.0) & (np.arange(diff.size) != 0))\n        if active_late.size != 1:\n            continue\n\n        j = int(active_late[0])\n        response = int(row['response'])\n        target_choice = (response == 0) if diff[j] > 0.0 else (response == 1)\n        positions.append(j + 1)  # Convert to the design's one-based positions.\n        followed_target.append(float(target_choice))\n\n    if len(positions) == 0:\n        return float('nan')\n\n    positions = np.asarray(positions, dtype=int)\n    followed_target = np.asarray(followed_target, dtype=np.float64)\n\n    terminal = followed_target[positions == 24]\n    early = followed_target[np.isin(positions, np.array([14, 16, 18]))]\n    if terminal.size == 0 or early.size == 0:\n        return float('nan')\n\n    return float(np.mean(terminal) - np.mean(early))",
  "rationale": "The metric is the increase in choices following the 82%-valid target when it appears at the terminal position versus the three early target positions. It pools both A/B reversals by scoring agreement with the target cue, thereby canceling response-side effects. The advocated binding cascade predicts a large positive contrast because its localized recency boost is negligible at positions 14\u201318 but maximal at position 24. The competing retrieval theory predicts a contrast near zero or slightly negative because isolated cues have no recency benefit and only a weak late-position loss. Averaging 48 early-position observations against 16 terminal observations per subject keeps binomial variance substantially lower than a single endpoint-to-endpoint contrast while retaining most of the predicted recency separation."
}
```

## Usage

```json
{
  "input_tokens": 8891,
  "output_tokens": 2610
}
```
