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


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] The 12 unique pairs form six diagnostic comparison patterns plus exact A/B reversals, yielding 96 trials with eight repetitions per pair. In every pattern, the highest-validity cue that discriminates favors the option designated by Take The Best, while the unweighted tally either favors the other option or is tied. The first three pattern types hold TTB's prediction constant while making the opposing tally margin equal to 1, 3, or 5 cues. Thus TTB predicts the same choice probability q=(1-epsilon)*sigmoid(beta)+epsilon/2 across these margins, whereas Tallying predicts probability (1-epsilon)*sigmoid(-beta*m)+epsilon/2 for the TTB-designated option, with m in {1,3,5}. A fourth pattern gives an exact tally tie, so Tallying predicts 0.5 while TTB remains directional. The fifth and sixth patterns make the highest-validity cue tie and place the first discriminating cue at validity rank 2 or 3; TTB still follows that cue, whereas the majority of lower-ranked cues favors the opposite option by three. Exact reversals balance response side and should reverse TTB's preference without changing effect magnitudes. The broad, strictly ordered validities make the cue cascade salient and unambiguous.

[1] This design tests a quantitative confidence signature rather than a directional conflict. In every pair, the highest-validity discriminating cue and the unweighted majority favor the same option, but the tally margin varies systematically from 1 through 6. The six base patterns have A-minus-B win-count margins of 1, 2, 3, 4, 5, and 6; each also contains at least one cue favoring B, avoiding trivial unanimous dominance. Tallying predicts a graded increase in choice probability for A: (1-epsilon)*sigmoid(beta*m)+epsilon/2, where m is the tally margin. Take The Best always stops at Expert 1 and therefore predicts the same probability, (1-epsilon)*sigmoid(beta)+epsilon/2, at every margin. Margin 1 provides a useful probability anchor because the two rules have the same functional prediction there, whereas margins 2-6 reveal the nonlinear accumulation unique to Tallying. Exact A/B reversals control response-side preferences while preserving each margin. With 12 unique pairs repeated eight times, the experiment contains exactly 96 trials. Unlike previously explored cue-majority conflict designs, this congruent design asks whether additional agreeing comparative evidence changes choice reliability after the top cue has already selected the same winner.

[2] This design isolates a quantitative confidence dissociation between the two cascade theories. The six base pairs place the first discriminating expert at validity ranks 1 through 6; their exact A/B reversals balance response side. Take The Best assigns the same binary winner score on every pair, so it predicts one rank-invariant probability of following the first discriminating cue: q=(1-epsilon)*sigmoid(beta)+epsilon/2. In contrast, every advertised validity is guaranteed to exceed the competing theory's confidence threshold throughout its parameter range: even the weakest expert has minimum subjective confidence 0.75*log(0.57/0.43)=0.211, above the maximum threshold of 0.20. Consequently, Confidence-Capped Cascade Integration also stops at the first discriminating cue, but its evidence magnitude equals sensitivity times that cue's log odds. It therefore predicts a strictly decreasing choice probability as the first discrimination moves from the 95% expert to the 57% expert: p_r=(1-epsilon)*sigmoid(beta*sensitivity*log(v_r/(1-v_r)))+epsilon/2. Thus, direction of choice is matched across theories while the predicted rank-by-confidence gradient is present only under the competing theory. Lower-ranked ratings alternate in direction where possible, preventing most displays from being simple unanimous dominance while remaining irrelevant after stopping. The 12 unique pairs are each repeated eight times, yielding exactly 96 trials.

[3] The 12 pairs comprise six weak-cue conflict levels and their exact A/B reversals, giving 96 trials with eight repetitions per pair. Expert 1 is highly valid (95%) but always ties, so it cannot determine a choice. Experts 2–8 all have validity 50.8%. Even at the advocated theory's maximum validity sensitivity, their subjective confidence is only 1.5*log(0.508/0.492)≈0.048, below its minimum confidence threshold of 0.05. Consequently, no discriminating cue can trigger cascade stopping, and Confidence-Capped Cascade Integration must combine all discriminating weak cues. Within each base pair, Expert 2 favors A and is the first discriminating cue under stable validity-tie ordering, while M of the remaining experts favor B, for M=1,...,6. Take The Best therefore predicts the same above-chance preference for A at every conflict level. The advocated theory instead predicts exact indifference when M=1 and an increasingly strong preference for B as M rises: its integrated evidence is proportional to (1-M)*[log(0.508/0.492)]^integration_exponent. Thus the critical signature is a reversal plus a graded conflict-level effect, neither of which TTB can generate with its single condition-invariant winner probability. Exact option reversals balance response-side biases while preserving these predictions. The tied 95% expert also supplies the required broad validity spread without accidentally activating the advocated theory's stopping rule.

[4] The six base pairs create a cumulative priority-race manipulation and are followed by exact A/B reversals, yielding 12 unique pairs repeated eight times for 96 trials. Expert 1 (95%) and Expert 2 (55%) always tie. The last-displayed expert has 72% validity and favors A in every base pair, whereas one through six earlier-displayed 70% experts favor B. Take The Best sorts cues strictly by advertised validity, skips the tied 95% cue, and always stops at the 72% cue. It therefore predicts the same above-chance probability q=(1-epsilon)*sigmoid(beta)+epsilon/2 of choosing A at every base conflict level, irrespective of how many 70% cues oppose it. Noisy Priority Race Search instead gives the earlier 70% cues a serial-position advantage. Conflict level 1 activates only the 70% cue immediately before the 72% cue; each subsequent level adds the next-earlier 70% cue. Adding a B-favoring competitor strictly increases the race probability that B supplies the stopping cue, so the competing theory predicts a monotonic decline in A choices across levels. The highest conflict level also provides a parameter-robust directional reversal: for the earliest active 70% cue, its minimum priority advantage over the 72% cue is 6(0.15)-5[logit(0.72)-logit(0.70)]-0.16, approximately 0.254. Even at retrieval noise 0.08, that cue alone has over 23 times the race weight of the 72% cue, and all other discriminating race cues also favor B. Thus the competing theory predicts a strong B preference at level 6, while TTB must continue to prefer A. Exact option reversals balance response-side biases and reverse directions without changing the predicted conflict-level profile.

[5] The six base pairs isolate conflicts between two discriminating experts separated by exactly two display positions; all other experts tie and therefore cannot terminate either search rule. In every base pair, the earlier expert has validity 0.55 and favors B, whereas the later expert has slightly to substantially higher validity and favors A. The later-minus-earlier validity-log-odds gaps are approximately 0.01, 0.15, 0.35, 0.60, 0.90, and 1.30. Take The Best always ranks the later expert above the 0.55 expert, so it predicts the same A-choice probability q=(1-epsilon)*sigmoid(beta)+epsilon/2 at all six levels, regardless of the size of the validity gap. Noisy Priority Race Search instead predicts a validity-gap-dependent crossover because the earlier B-favoring expert receives a constant two-position serial advantage. Its later-minus-earlier priority difference is sensitivity*D - 2*serial_priority + (bias_later-bias_earlier). At D≈0.01 this difference is at most 5(0.01)-2(0.15)+0.16=-0.09 across the entire parameter range, so the advocated theory robustly predicts that the nominally less-valid, earlier expert wins the race and B is chosen more often. At D≈1.30 the difference is at least 1(1.30)-2(0.45)-0.16=0.24, so it robustly predicts that the more-valid expert wins and A is chosen more often. Intermediate gaps locate each subject's stochastic priority crossover and should produce a rising psychometric profile, potentially with stable cue-specific deviations from perfect monotonicity. Thus the critical signature is both a directional reversal at the smallest gap and a quantitative increase in following the higher-validity expert as the log-odds gap grows; TTB can only fit one condition-invariant response probability. The intervening tied experts preserve the same two-position separation, while tied 95% and 50% experts provide a broad advertised-validity spread without affecting choices. Each base pair has an exact A/B reversal, balancing response-side preferences. The 12 unique pairs are repeated eight times, yielding exactly 96 trials.

[6] The six base pairs hold Take The Best's prediction constant while moving the sole higher-validity A-favoring discriminating cue through display positions 3, 8, 14, 20, 26, and 32 (zero-based positions 2, 7, 13, 19, 25, and 31). Two lower-validity cues at the first two display positions always favor B; every other cue ties. TTB skips the higher-validity tying cues, stops at the designated target cue, and therefore predicts the same A-choice probability q=(1-epsilon)*sigmoid(beta)+epsilon/2 in all six conditions. The accessibility-gated cascade instead predicts a steep decline in A choices with target position. If a_t is target accessibility and R=(1-a_0)(1-a_1) is the probability that both early B cues are inaccessible, its intended A probability is a_t+0.5*(1-a_t)*R. The linear and quadratic position costs make a_t decrease strongly across conditions, while the two early fallback cues remain highly accessible. This produces a parameter-robust crossover: at position 2, target accessibility is at least about 0.78, whereas at position 31 it is at most about 0.38. Even using the least-accessible permissible fallback cues, intended A probability in the final condition is at most about 0.40, so execution flips and lapses attenuate but do not reverse the predicted B preference. Thus the competitor predicts a declining psychometric profile from a clear A preference to a B preference, while TTB predicts rank- and position-invariant choice reliability. Exact A/B reversals balance response-side preferences. The 12 unique pairs are repeated eight times, yielding 96 trials.

[7] The six base conditions manipulate same-direction redundancy within the validity cascade while holding Take The Best's decisive cue fixed. In every base pair, the 95%-valid Expert 24 favors A, so TTB always stops there and predicts the same A-choice probability q=(1-epsilon)*sigmoid(beta)+epsilon/2. Experts 1 and 2 favor B but have lower validities (72% and 70%). Condition 1 contains only the late-displayed 95% A cue; conditions 2 through 6 successively add A-favoring backup cues with validities 93%, 91%, 89%, 87%, and 85%. All remaining cues tie. Under the Accessibility-Gated Validity Cascade, each added backup supplies another opportunity for an A-favoring cue to be accessible before search reaches the B-favoring cues. If Q_m is the probability that all m active A cues are inaccessible and S is the probability that both B cues are inaccessible, intended P(A)=1-Q_m+0.5*Q_m*S. Thus A choice must increase monotonically as backups are added, with a characteristic diminishing-returns profile from the multiplicative accessibility failures. The manipulation is especially diagnostic because all A cues occupy terminal display positions 19–24, where the advocated theory predicts appreciable accessibility failures in a 24-cue list, whereas the two B fallbacks occupy the highly accessible first positions. Execution flips and lapses attenuate but preserve the positive redundancy gradient. TTB cannot produce this condition effect because its 95% cue already discriminates in every condition and lower-validity cues are never consulted. The final six pairs are exact A/B reversals, balancing response-side preferences while preserving the predicted redundancy profile. Twelve unique pairs repeated eight times yield 96 trials.

[8] The six base conditions hold both the amount and direction of evidence constant while manipulating only the spatial density of a same-direction cue cluster. Expert 17 (index 16) has the uniquely highest validity, 92%, and always favors A. Six 90%-valid experts also favor A, while an isolated 55%-valid expert favors B. Across conditions, the six redundant A cues move symmetrically away from Expert 17: their offsets are ±{1,2,3}, ±{1,3,5}, ±{2,4,6}, ±{2,5,8}, ±{3,7,11}, and ±{5,10,15}. This symmetry approximately balances early-versus-late positional effects in the 32-cue display, and the number, validities, and recommendations of all discriminating cues remain fixed. Take The Best always stops at the 92% cue and therefore predicts one condition-invariant probability q=(1-epsilon)*sigmoid(beta)+epsilon/2 of choosing A. The Competitive Source-Binding Cascade instead predicts a pronounced spacing gradient. In the densest condition, the seven nearby, similarly valid A sources form a mutually inhibiting cluster; even the leading 92% source receives crowding from both sides, permitting the isolated B source to control many choices. As spacing increases, positional similarity and hence crowding decline exponentially, so successful retrieval of at least one A source should increase and A choices should rise, potentially crossing from below to above chance. This is an adversarial redundancy effect: distributing exactly the same supporting evidence improves its use, whereas TTB is indifferent to both redundancy and spacing. Exact A/B reversals balance response-side preferences and should invert the competing theory's spacing profile without changing its magnitude. The 12 unique pairs are repeated eight times, yielding 96 trials.

[9] This design translates a fixed two-cue competitive cluster across a 32-expert display while holding its size, within-cluster spacing, validity, recommendation, and opposing evidence constant. In each base pair, exactly two adjacent 82%-valid experts favor A, and the isolated 55%-valid Expert 16 favors B. The six A-cluster locations are positions 1–2, 7–8, 13–14, 19–20, 25–26, and 31–32. The 95%-valid Expert 4 always ties. Take The Best therefore skips the tied 95% expert and stops at the first active 82% expert, predicting the same A-choice probability q=(1-epsilon)*sigmoid(beta)+epsilon/2 in every condition. The Competitive Source-Binding Cascade predicts a different quantitative profile. Because the two A cues are adjacent, equally valid, and recommend the same option, their mutual competitive crowding is identical across conditions; because their moderate validity lies near the model's binding-uncertainty region, failed retrieval should frequently permit the fixed, isolated B cue to control choice. Translating the cluster later selectively increases its binding support through the 32-cue long-display positional contrast, with an additional sharply localized recency benefit for the terminal cluster. Thus the model predicts increasing A choices from early to late cluster locations, typically with the largest improvement at positions 31–32 and potentially a crossover from following B to following A. This isolates serial-position recovery from cluster density, unlike a spacing manipulation: all competitive relationships within the cluster remain unchanged. The final six pairs are exact A/B reversals, balancing response-side preferences while preserving the predicted location gradient. Twelve unique pairs repeated eight times yield 96 trials.

[10] The six base conditions vary the number of widely separated, same-direction 82%-valid sources supporting A from 2 through 7. Their positions are selected to remain broadly symmetric around the display midpoint, with at least seven positions between neighboring A sources. A single 70%-valid terminal source supports B, while the 95%-valid source and all remaining sources tie. The advocated Chunked Source-Retrieval Lexicographic Search treats similarity-based interference as source-based and independent of spatial distance. Because the A sources are nonadjacent, they cannot form chunks. Increasing their number therefore raises each source's attribution load; from three sources onward it also invokes the correlated recommendation-level gate. It predicts a pronounced decline in A choices as the A set grows, particularly from approximately 3–4 to 7 sources, despite all added evidence supporting A. The Competitive Source-Binding Cascade predicts the opposite profile. Its inhibition is strongly attenuated by the large spatial separations, so each added 82% source provides another relatively independent opportunity to stop the validity-ordered cascade before the 70% B source. It therefore predicts increasing A choices with source count. Thus the critical quantitative test is the sign of the redundancy slope: negative under the advocated theory but positive under the competitor. The six exact A/B reversals balance response-side preferences and should invert choice direction while preserving the count effect. Twelve unique pairs repeated eight times yield 96 trials.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "This 24-expert design isolates the advocated theory's terminal-recency mechanism without activating its long-display positional contrast, which is exactly zero at n_features=24. In each base pair, one 82%-valid singleton favors A and the fixed 70%-valid Expert 1 favors B; the 95%-valid expert and every other expert tie. Across six conditions, the otherwise identical 82% singleton appears at positions 14, 16, 18, 20, 22, and 24. There is no same-direction crowding because only one cue supports each option, and neither adjacency chunking nor recommendation-level shared gates can occur. Competitive Source-Binding Cascade predicts a sharply increasing, convex A-choice profile as the target approaches the display endpoint: its recency contribution rises from approximately exp(-10/2.76) to 1 across these positions, producing a terminal binding-logit gain of at least about 1.3 relative to position 14. When the 82% cue is not bound, search reaches the fixed 70% B cue, making target-binding changes consequential and potentially producing a crossover toward A at the terminal position. Chunked Source-Retrieval Lexicographic Search has no recency benefit. Its target retrieval probability is position-invariant except for the weak monotonic singleton late-position loss; across positions 14 to 24, that penalty changes by at most about 0.026 logits. It therefore predicts an approximately flat or slightly decreasing A-choice profile, directly opposing the advocated theory's pronounced terminal increase. Exact A/B reversals balance response-side preferences. The 12 unique pairs are repeated eight times, yielding 96 trials.",
  "validities": [
    0.7,
    0.55,
    0.57,
    0.59,
    0.61,
    0.63,
    0.95,
    0.65,
    0.54,
    0.56,
    0.58,
    0.6,
    0.62,
    0.82,
    0.64,
    0.82,
    0.53,
    0.82,
    0.57,
    0.82,
    0.61,
    0.82,
    0.55,
    0.82
  ],
  "trial_a_ratings": [
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
      0,
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
      0,
      0,
      0,
      1,
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
      0,
      0,
      0,
      0,
      0,
      1,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1
    ],
    [
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ]
  ],
  "trial_b_ratings": [
    [
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
      0,
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
      0,
      0,
      0,
      1,
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
      0,
      0,
      0,
      0,
      0,
      1,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 10126,
  "output_tokens": 3667
}
```
