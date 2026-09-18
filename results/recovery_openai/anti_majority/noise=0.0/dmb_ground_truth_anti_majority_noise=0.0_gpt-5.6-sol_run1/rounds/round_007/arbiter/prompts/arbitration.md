# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_9" and "pi_8") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_9" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_8" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_9
Competition-Normalized Recommendation Race. People first compress all currently active experts favoring the same option into a single recommendation-level representation. Consequently, redundant experts are retrieved correlatively rather than as independent opportunities for rescue, and reliability gains from additional experts saturate almost immediately. Each recommendation is indexed by its strongest communicated validity and the two opposing recommendation representations compete for retrieval before a validity-ordered, predominantly lexicographic decision. Source-to-validity binding has a localized nonlinear uncertainty region, allowing moderate-to-high validity cues to be less accessible than cues outside that region. Spatial arrangement ordinarily has no effect. The exception is density overload: a recommendation loses binding fidelity when most of its support forms an extended contiguous run and the opposing recommendation itself has multiple active supporters. This competition gate prevents long runs from being penalized merely for containing many sources and leaves adjacent pairs, separated redundancy, and long runs facing a singleton largely unaffected. The strongest member's location within a run provides no protection. In long displays only, an isolated recommendation receives a weak monotonic accessibility loss with later position; there is no terminal-recency process. Failure to retrieve either recommendation produces guessing, followed by a small symmetric response lapse.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            "Competition-Normalized Recommendation Race expects "
            f"shape (2, n_features); got {stim.shape}."
        )

    n_features = int(parameters["n_features"])
    if stim.shape[1] != n_features:
        raise ValueError(
            f"Stimulus has {stim.shape[1]} features but n_features={n_features}."
        )

    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities must have length {n_features}; got {validities.shape}."
        )

    compression = float(parameters["validity_compression"])
    baseline = float(parameters["retrieval_baseline"])
    uncertainty_strength = float(parameters["binding_uncertainty_strength"])
    uncertainty_center = float(parameters["binding_uncertainty_center"])
    uncertainty_width = max(
        float(parameters["binding_uncertainty_width"]), 1e-9
    )
    overload_strength = float(parameters["contiguous_density_overload"])
    saturation_gain = float(parameters["recommendation_saturation_gain"])
    position_loss = float(parameters["contextual_position_loss"])
    lapse = float(parameters["response_lapse"])

    differences = stim[0] - stim[1]
    directions = np.sign(differences)
    active = np.flatnonzero(directions != 0.0)
    if active.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    objective_strength = np.clip(
        (validities - 0.5) / 0.5, 0.0, 1.0
    )
    subjective_strength = np.power(objective_strength, compression)
    binding_uncertainty = np.exp(
        -0.5 * np.square(
            (subjective_strength - uncertainty_center) / uncertainty_width
        )
    )

    groups = {}
    for direction in (-1, 1):
        members = np.flatnonzero(directions == float(direction)).astype(int)
        if members.size:
            groups[direction] = members

    def longest_contiguous_run(members):
        if members.size == 0:
            return 0
        longest = 1
        current = 1
        for k in range(1, members.size):
            if int(members[k]) == int(members[k - 1]) + 1:
                current += 1
                longest = max(longest, current)
            else:
                current = 1
        return int(longest)

    units = {}
    for direction, members in groups.items():
        member_validities = validities[members]
        rep = int(members[int(np.argmax(member_validities))])
        count = int(members.size)

        # All same-direction sources share one recommendation representation.
        # Extra sources give only a tiny and rapidly saturating reliability gain.
        count_gain = saturation_gain * (
            1.0 - np.exp(-2.0 * float(max(count - 1, 0)))
        )

        retrieval_logit = (
            baseline
            - uncertainty_strength * binding_uncertainty[rep]
            + count_gain
        )

        # Density overload requires an extended contiguous run, concentration
        # of the recommendation's support inside that run, and at least two
        # active sources supporting the competing recommendation.
        run_length = longest_contiguous_run(members)
        opponent = groups.get(-direction, np.asarray([], dtype=int))
        opponent_excess = max(int(opponent.size) - 1, 0)
        competition_gate = 1.0 - np.exp(-float(opponent_excess))
        run_excess = max(run_length - 3, 0)
        if run_excess > 0 and competition_gate > 0.0:
            density = float(run_length) / float(count)
            overload_cost = (
                overload_strength
                * float(run_excess)
                * density * density
                * competition_gate
            )
            retrieval_logit -= overload_cost

        # Only isolated recommendations in substantially loaded displays have
        # a monotonic late-position accessibility loss. There is no recency.
        if count == 1 and n_features > 1:
            display_load = float(np.clip(
                (n_features - 24.0) / 8.0, 0.0, 1.0
            ))
            relative_position = rep / float(n_features - 1)
            retrieval_logit -= (
                position_loss * display_load * relative_position
            )

        units[direction] = {
            "direction": int(direction),
            "representative": rep,
            "priority": float(validities[rep]),
            "raw_logit": float(retrieval_logit),
        }

    # Normalize each recommendation's accessibility against activation of its
    # competitor. The fixed soft competition operation introduces no spatial
    # or source-count interference parameter.
    for direction, unit in units.items():
        opponent = units.get(-direction)
        if opponent is None:
            competition = 0.0
        else:
            competition = float(np.logaddexp(
                0.0, opponent["raw_logit"] - baseline
            ))
        normalized_logit = unit["raw_logit"] - competition
        if normalized_logit >= 0.0:
            retrieval = 1.0 / (1.0 + np.exp(-normalized_logit))
        else:
            ex = np.exp(normalized_logit)
            retrieval = ex / (1.0 + ex)
        unit["retrieval"] = float(np.clip(
            retrieval, 1e-9, 1.0 - 1e-9
        ))

    ordered_units = sorted(
        units.values(),
        key=lambda unit: (-unit["priority"], unit["representative"]),
    )

    # A retrieved higher-validity recommendation terminates the search. Search
    # continues only when that recommendation is unavailable.
    reach = 1.0
    intended_a = 0.0
    intended_b = 0.0
    for unit in ordered_units:
        stop = reach * unit["retrieval"]
        if unit["direction"] > 0:
            intended_a += stop
        else:
            intended_b += stop
        reach *= 1.0 - unit["retrieval"]

    intended_a += 0.5 * reach
    intended_b += 0.5 * reach
    total = intended_a + intended_b
    if not np.isfinite(total) or total <= 0.0:
        p_a_intended = 0.5
    else:
        p_a_intended = intended_a / total

    p_a = (
        (1.0 - lapse) * p_a_intended
        + lapse * (1.0 - p_a_intended)
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
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))


## THEORY 2 — pi_8
Structure-Gated Ensemble Lexicographic Retrieval. Decision makers preserve the communicated validity hierarchy and search encoded recommendation units from highest to lowest validity, but the units available to search depend on the grouping structure of the display. An isolated expert is represented as a source-specific item. Two adjacent experts making the same recommendation are encoded as a compact pair chunk with high, position-invariant fidelity. Contiguous same-direction runs of three or more experts are instead compressed into a single ensemble whose source-to-validity binding becomes progressively less reliable with run length. A uniquely strongest expert embedded inside such a run remains structurally identifiable, whereas a strongest expert at a run boundary receives no such protection from overload. Same-direction experts separated by inactive or opposing experts are represented by a recommendation-level ensemble: additional separated sources provide only a negligible, rapidly saturating retrieval benefit and impose no distance-based crowding cost. Independently, validity labels have a localized nonlinear binding-uncertainty region, so a larger communicated validity gap need not imply more reliable retrieval. Serial position has no terminal-recency bonus. Only isolated sources in sufficiently long displays exhibit a weak subject-specific monotonic accessibility gradient centered near zero, allowing heterogeneous serial-position effects without a systematic pooled recency advantage. Once the encoded units are formed, choice is primarily lexicographic: the first successfully retrieved unit in descending communicated-validity order determines the intended response. Complete retrieval failure produces guessing, and a small execution lapse can reverse the intended response.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            "Structure-Gated Ensemble Lexicographic Retrieval expects "
            f"shape (2, n_features); got {stim.shape}."
        )

    n_features = int(parameters["n_features"])
    if stim.shape[1] != n_features:
        raise ValueError(
            f"Stimulus has {stim.shape[1]} features but n_features={n_features}."
        )

    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities must have length {n_features}; got {validities.shape}."
        )

    compression = float(parameters["validity_compression"])
    baseline = float(parameters["retrieval_baseline"])
    uncertainty_strength = float(parameters["binding_uncertainty_strength"])
    uncertainty_center = float(parameters["binding_uncertainty_center"])
    uncertainty_width = max(
        float(parameters["binding_uncertainty_width"]), 1e-9
    )
    pair_fidelity = float(parameters["pair_chunk_fidelity"])
    run_overload = float(parameters["contiguous_run_overload"])
    separated_gain = float(
        parameters["noncontiguous_redundancy_gain"]
    )
    position_slope = float(parameters["long_display_position_slope"])
    lapse = float(parameters["response_lapse"])

    differences = stim[0] - stim[1]
    directions = np.sign(differences)
    active = np.flatnonzero(directions != 0.0)
    if active.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    # Validity labels are transformed only for binding fidelity. Search rank
    # below continues to use communicated validity itself.
    objective_strength = np.clip(
        (validities - 0.5) / 0.5, 0.0, 1.0
    )
    subjective_strength = np.power(objective_strength, compression)
    binding_uncertainty = np.exp(
        -0.5 * np.square(
            (subjective_strength - uncertainty_center) / uncertainty_width
        )
    )

    def logistic(x):
        if x >= 0.0:
            return 1.0 / (1.0 + np.exp(-x))
        ex = np.exp(x)
        return ex / (1.0 + ex)

    def representative(members):
        member_array = np.asarray(members, dtype=int)
        vals = validities[member_array]
        # Earliest position resolves exact communicated-validity ties.
        return int(member_array[int(np.argmax(vals))])

    # Identify maximal contiguous runs of active sources giving the same
    # recommendation. Inactive cues and opposite recommendations both break
    # a run; no metric of generic spatial distance is used.
    runs = []
    i = 0
    while i < active.size:
        current = [int(active[i])]
        j = i + 1
        while (
            j < active.size
            and int(active[j]) == int(active[j - 1]) + 1
            and directions[int(active[j])] == directions[int(active[i])]
        ):
            current.append(int(active[j]))
            j += 1
        runs.append(current)
        i = j

    units = []
    isolated_by_direction = {-1: [], 1: []}

    for members in runs:
        direction = int(np.sign(directions[members[0]]))
        rep = representative(members)
        base_logit = (
            baseline
            - uncertainty_strength * binding_uncertainty[rep]
        )

        if len(members) == 1:
            isolated_by_direction[direction].append(rep)
            continue

        if len(members) == 2:
            # A same-direction adjacent pair is a compact chunk. Pair fidelity
            # supplies a high retrieval floor and there is no position term.
            ordinary = logistic(base_logit)
            retrieval = pair_fidelity + (1.0 - pair_fidelity) * ordinary
            kind = "pair"
        else:
            # Runs of three or more are over-compressed. Internal source
            # attribution deteriorates with run length, independently of the
            # run's absolute location. A uniquely strongest member strictly
            # inside the run remains structurally identifiable; an endpoint
            # anchor or a validity tie receives no protection.
            excess_length = float(len(members) - 2)
            overload_cost = run_overload * np.power(excess_length, 1.0)
            member_array = np.asarray(members, dtype=int)
            member_validities = validities[member_array]
            strongest = np.flatnonzero(
                np.isclose(
                    member_validities,
                    np.max(member_validities),
                    atol=1e-12,
                    rtol=0.0,
                )
            )
            uniquely_embedded = (
                strongest.size == 1
                and 0 < int(strongest[0]) < len(members) - 1
            )
            if uniquely_embedded:
                overload_cost *= 0.10
            retrieval = logistic(base_logit - overload_cost)
            kind = "run"

        units.append({
            "priority": float(validities[rep]),
            "tie_position": int(min(members)),
            "direction": direction,
            "retrieval": float(np.clip(retrieval, 1e-9, 1.0 - 1e-9)),
            "kind": kind,
        })

    # Separated isolated sources supporting the same option enter one
    # recommendation-level ensemble. Source count has only a small,
    # saturating benefit and never generates a spacing or crowding penalty.
    for direction in (-1, 1):
        members = isolated_by_direction[direction]
        if not members:
            continue

        rep = representative(members)
        retrieval_logit = (
            baseline
            - uncertainty_strength * binding_uncertainty[rep]
        )

        if len(members) == 1:
            # Under substantial display load, isolated sources receive a weak
            # subject-specific monotonic accessibility gradient. It contains
            # no terminal-specific recency bonus.
            load_factor = np.clip((n_features - 24.0) / 8.0, 0.0, 1.0)
            if n_features > 1:
                relative_position = rep / float(n_features - 1)
            else:
                relative_position = 0.0
            retrieval_logit -= (
                position_slope * load_factor * relative_position
            )
            kind = "singleton"
        else:
            extra_sources = float(len(members) - 1)
            saturation = 1.0 - np.exp(-extra_sources)
            retrieval_logit += separated_gain * saturation
            kind = "separated_ensemble"

        retrieval = logistic(retrieval_logit)
        units.append({
            "priority": float(validities[rep]),
            "tie_position": int(min(members)),
            "direction": direction,
            "retrieval": float(np.clip(retrieval, 1e-9, 1.0 - 1e-9)),
            "kind": kind,
        })

    # Communicated validity supplies the primary lexicographic hierarchy.
    units.sort(key=lambda u: (-u["priority"], u["tie_position"]))

    reach = 1.0
    intended_a = 0.0
    intended_b = 0.0
    for unit in units:
        stop = reach * unit["retrieval"]
        if unit["direction"] > 0:
            intended_a += stop
        else:
            intended_b += stop
        reach *= 1.0 - unit["retrieval"]

    intended_a += 0.5 * reach
    intended_b += 0.5 * reach
    total = intended_a + intended_b
    if not np.isfinite(total) or total <= 0.0:
        p_a_intended = 0.5
    else:
        p_a_intended = intended_a / total

    # A symmetric execution lapse reverses the intended binary response.
    p_a = (
        (1.0 - lapse) * p_a_intended
        + lapse * (1.0 - p_a_intended)
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


## EXPERIMENT 1 (proposed by pi_9)

### DESIGN
**Validities (n_features=20):** [0.97, 0.92, 0.86, 0.84, 0.82, 0.8, 0.78, 0.5, 0.7, 0.53, 0.59, 0.54, 0.58, 0.55, 0.57, 0.56, 0.56, 0.52, 0.55, 0.51]

**Trial pairs (n=12):**
  trial 1: A=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0]
  trial 5: A=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0]
  trial 6: A=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
  trial 7: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** This design directly targets the advocated theory's competition gate while keeping the contiguous recommendation itself fixed. In every base trial, Experts 2–7 form the same six-source contiguous run favoring A, with the uniquely strongest 92%-valid member at the run boundary. A lower-validity recommendation favoring B is supported by one through six mutually separated experts at positions 9, 11, 13, 15, 17, and 19. The first B expert is always the strongest B source (70%); added B experts are weaker and separated by inactive experts, so they never form pairs or runs. Competition-Normalized Recommendation Race predicts a large, sharply saturating decline in A choices. With one B supporter, its opponent-excess term is zero and the six-source A run receives no density-overload penalty. Adding the second B supporter opens the competition gate, imposing an A-retrieval-logit cost of 3*contiguous_density_overload*(1-exp(-1)), approximately 1.90–3.03 logits. Further B supporters raise the gate toward one, yielding a total cost near 2.98–4.77 logits at six supporters. Thus the largest choice change should occur from one to two B sources, followed by saturation. Structure-Gated Ensemble Lexicographic Retrieval predicts an approximately flat count profile: the fixed A run receives the same length-six overload in every condition, while all separated B sources are compressed into one recommendation-level ensemble whose count-dependent gain is at most 0.02*(1-exp(-(m-1))) logits. Hence it cannot produce the advocated theory's large one-to-two-source cliff. The tied 97% expert broadens the validity range without entering either search. The final six pairs are exact A/B reversals, balancing response side while preserving the predicted count profile. Twelve unique pairs repeated eight times yield 96 trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
The metric is the within-subject drop in choosing the fixed six-expert run-supported option when the opposing recommendation changes from one supporter to multiple supporters. It first recodes the exact A/B reversals into a common run-choice outcome, eliminating response-side preference, then contrasts the 16 observations with one opponent (two orientations × eight repetitions) against the 80 observations with two through six opponents. Competition-Normalized Recommendation Race predicts a large positive value because its overload gate switches on at the second opposing supporter and then saturates. Structure-Gated Ensemble Lexicographic Retrieval predicts a value near zero because the run representation is already overloaded in every condition and separated opposing sources receive only a negligible count gain. Pooling all post-gate conditions both captures the sustained cliff and substantially lowers per-subject sampling variance; the within-subject contrast also cancels stable differences in lapse rate and overall propensity to follow the run.

Source:
def metric(data: pd.DataFrame) -> float:
    def subject_cliff(df):
        counts = []
        run_choices = []

        for _, row in df.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float)
            b = np.asarray(row["option_b_ratings"], dtype=float)
            diff = a - b

            # Features 2-7 (zero-based positions 1:7) are the fixed run.
            run_direction = np.sign(np.sum(diff[1:7]))
            if run_direction == 0:
                continue

            # Count active sources recommending against the run-supported option.
            opponent_count = int(np.sum(diff == -run_direction))
            response = int(row["response"])
            chose_run_option = float(
                (run_direction > 0 and response == 0)
                or (run_direction < 0 and response == 1)
            )

            counts.append(opponent_count)
            run_choices.append(chose_run_option)

        counts = np.asarray(counts, dtype=int)
        run_choices = np.asarray(run_choices, dtype=float)
        first = run_choices[counts == 1]
        competitive = run_choices[counts >= 2]

        if first.size == 0 or competitive.size == 0:
            return np.nan
        return float(np.mean(first) - np.mean(competitive))

    if "subject_id" in data.columns:
        values = []
        for _, df in data.groupby("subject_id", sort=False):
            value = subject_cliff(df)
            if np.isfinite(value):
                values.append(value)
        if values:
            return float(np.mean(values))

    return subject_cliff(data)

### RESULTS
- Predicted under pi_9 (simulated): 0.5137 (var=0.0192)
- Predicted under pi_8 (simulated): -0.0035 (var=0.0146)
- Observed on real data: -0.0205 (var=0.0094)

## EXPERIMENT 2 (proposed by pi_8)

### DESIGN
**Validities (n_features=18):** [0.97, 0.55, 0.82, 0.82, 0.82, 0.82, 0.82, 0.82, 0.82, 0.82, 0.82, 0.82, 0.6, 0.5, 0.58, 0.7, 0.65, 0.52]

**Trial pairs (n=12):**
  trial 1: A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  trial 2: A=[0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  trial 3: A=[0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  trial 4: A=[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  trial 5: A=[0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  trial 6: A=[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  trial 7: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]

**Rationale:** The six base trials form two matched structure sets with three or four 82%-valid experts favoring A and one fixed 70%-valid singleton favoring B. Within each source-count set, the A experts are arranged as (i) one contiguous run, (ii) structures containing adjacent pairs (a pair plus a singleton for count three; two separated pairs for count four), or (iii) entirely separated sources. All potentially active A positions have identical 82% validity, so recommendation count, validity, and opposing evidence are held constant within each set; only grouping structure changes. The 18-feature display also eliminates both theories' long-display position terms. Structure-Gated Ensemble Lexicographic Retrieval predicts a pronounced structural ordering. Adjacent pairs become high-fidelity chunks and remain separate from other A units, providing multiple lexicographic retrieval opportunities; therefore pair-plus-singleton and especially two-pair arrangements should produce the most A choices. Entirely separated A sources collapse into one recommendation-level ensemble with negligible redundancy gain, producing an intermediate A-choice rate. A contiguous run is compressed into one overloaded ensemble; because the 82% members are tied, there is no uniquely embedded strongest-member protection, and overload doubles from the length-three to the length-four run. Thus the advocated theory predicts pair-containing structures > separated structures > contiguous runs, with a particularly large two-pairs-versus-length-four-run contrast. Competition-Normalized Recommendation Race instead compresses all A supporters into one recommendation representation. Within a fixed source count, its accessibility is exactly invariant to these spatial arrangements: the length-three run never reaches its run-excess threshold, and although the length-four run does, its singleton opponent makes the competition gate zero. It therefore predicts equal choice probabilities across the three structures at each count, apart from no structural effect at all. The six exact A/B reversals balance response side and invert the advocated structural profile. Twelve unique pairs repeated eight times yield 96 trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
This is the orientation-invariant two-pairs advantage among the four-source trials: the probability of choosing the option supported by two separated adjacent pairs minus the probability of choosing the option supported by one contiguous run of four. It pools each condition with its exact A/B reversal, cancelling response-side bias, and uses 16 observations per condition per subject. The advocated theory predicts a large positive contrast because two high-fidelity pair chunks provide separate retrieval opportunities, whereas a length-four run suffers substantial overload. The competing theory predicts essentially zero because it compresses all four supporters into one recommendation and its run-overload gate is inactive against a singleton opponent. Restricting the metric to the strongest predicted structural contrast avoids diluting it with the smaller three-source effect.

Source:
def metric(data: pd.DataFrame) -> float:
    pair_choices = []
    run_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=int)
        b = np.asarray(row["option_b_ratings"], dtype=int)
        diff = a - b

        pos = np.flatnonzero(diff > 0)
        neg = np.flatnonzero(diff < 0)

        # Retain the four-supporters-versus-singleton trials only.
        if len(pos) == 4 and len(neg) == 1:
            target_idx = pos
            chose_target = float(int(row["response"]) == 0)
        elif len(neg) == 4 and len(pos) == 1:
            target_idx = neg
            chose_target = float(int(row["response"]) == 1)
        else:
            continue

        target_idx = np.sort(target_idx)
        longest_run = 1
        current_run = 1
        for j in range(1, len(target_idx)):
            if target_idx[j] == target_idx[j - 1] + 1:
                current_run += 1
                longest_run = max(longest_run, current_run)
            else:
                current_run = 1

        if longest_run == 4:
            run_choices.append(chose_target)
        elif longest_run == 2:
            # In this design, this identifies the two-separated-pairs condition.
            pair_choices.append(chose_target)

    if len(pair_choices) == 0 or len(run_choices) == 0:
        return float("nan")

    return float(np.mean(pair_choices) - np.mean(run_choices))

### RESULTS
- Predicted under pi_9 (simulated): 0.0088 (var=0.0220)
- Predicted under pi_8 (simulated): 0.3287 (var=0.0175)
- Observed on real data: 0.0550 (var=0.0207)

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
**Predicted under pi_9:** 1.0000 (var=0.0000)
**Predicted under pi_8:** 1.0000 (var=0.0000)

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
**Predicted under pi_9:** 0.0000 (var=0.0384)
**Predicted under pi_8:** 0.0000 (var=0.0000)

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
**Predicted under pi_9:** 0.0000 (var=0.0196)
**Predicted under pi_8:** 0.0000 (var=0.0384)

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
**Predicted under pi_9:** 0.1457 (var=0.0026)
**Predicted under pi_8:** 0.1487 (var=0.0015)

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
**Predicted under pi_9:** 0.1688 (var=0.0073)
**Predicted under pi_8:** 0.1844 (var=0.0040)

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
**Predicted under pi_9:** -0.0962 (var=0.0183)
**Predicted under pi_8:** -0.1362 (var=0.0170)

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
**Predicted under pi_9:** -0.0071 (var=0.0015)
**Predicted under pi_8:** 0.0028 (var=0.0019)

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
**Predicted under pi_9:** -0.2544 (var=0.0160)
**Predicted under pi_8:** -0.2625 (var=0.0123)

### Experiment 11
**Design**
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

**Metric**
```python
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
```

**Observed (real) value:** -0.0056 (var=0.0030)
**Predicted under pi_9:** -0.0047 (var=0.0080)
**Predicted under pi_8:** 0.0038 (var=0.0085)

### Experiment 12
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]

**Metric**
```python
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

```

**Observed (real) value:** -0.0045 (var=0.0098)
**Predicted under pi_9:** -0.0120 (var=0.0102)
**Predicted under pi_8:** 0.0237 (var=0.0050)

### Experiment 13
**Design**
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Indices of the seven 82%-valid experts manipulated across conditions.
    source_idx = np.array([3, 10, 17, 24, 31, 38, 45], dtype=int)

    counts = []
    chose_redundant_side = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.size <= int(source_idx.max()) or b.size <= int(source_idx.max()):
            continue

        d = a[source_idx] - b[source_idx]
        active = d != 0
        n_sources = int(np.sum(active))
        direction = float(np.sum(d[active]))
        if n_sources == 0 or direction == 0.0:
            continue

        response = int(row["response"])
        redundant_side_is_a = direction > 0.0
        chose_support = (response == 0) if redundant_side_is_a else (response == 1)

        counts.append(n_sources)
        chose_redundant_side.append(float(chose_support))

    counts = np.asarray(counts, dtype=int)
    chose_redundant_side = np.asarray(chose_redundant_side, dtype=float)

    low = counts <= 3
    high = counts >= 6
    if not np.any(low) or not np.any(high):
        raise ValueError("Metric requires both low (2-3) and high (6-7) redundancy trials.")

    return float(np.mean(chose_redundant_side[high]) -
                 np.mean(chose_redundant_side[low]))

```

**Observed (real) value:** -0.0225 (var=0.0063)
**Predicted under pi_9:** -0.0088 (var=0.0093)
**Predicted under pi_8:** -0.0156 (var=0.0119)

### Experiment 14
**Design**
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Score whether each response followed the 82%-valid singleton rather than
    # the fixed 70%-valid cue at feature 1.
    positions = []
    followed_target = []

    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=np.float64)
        b = np.asarray(row['option_b_ratings'], dtype=np.float64)
        diff = a - b

        # Every designed trial has cue 1 and exactly one later cue active.
        active_late = np.flatnonzero((diff != 0.0) & (np.arange(diff.size) != 0))
        if active_late.size != 1:
            continue

        j = int(active_late[0])
        response = int(row['response'])
        target_choice = (response == 0) if diff[j] > 0.0 else (response == 1)
        positions.append(j + 1)  # Convert to the design's one-based positions.
        followed_target.append(float(target_choice))

    if len(positions) == 0:
        return float('nan')

    positions = np.asarray(positions, dtype=int)
    followed_target = np.asarray(followed_target, dtype=np.float64)

    terminal = followed_target[positions == 24]
    early = followed_target[np.isin(positions, np.array([14, 16, 18]))]
    if terminal.size == 0 or early.size == 0:
        return float('nan')

    return float(np.mean(terminal) - np.mean(early))
```

**Observed (real) value:** 0.0150 (var=0.0426)
**Predicted under pi_9:** 0.0171 (var=0.0149)
**Predicted under pi_8:** -0.0463 (var=0.0163)

### Experiment 15
**Design**
  A=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    validities = np.array([
        0.97, 0.70, 0.92, 0.90, 0.88, 0.90, 0.92, 0.88,
        0.92, 0.90, 0.88, 0.86, 0.90, 0.92, 0.88, 0.86,
        0.92, 0.90, 0.88, 0.86, 0.84, 0.90, 0.88, 0.92,
        0.86, 0.84
    ], dtype=float)

    cells = {(length, embedded): []
             for length in (3, 4, 5)
             for embedded in (False, True)}

    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        difference = a - b

        i = 0
        run = None
        while i < difference.size:
            if difference[i] == 0:
                i += 1
                continue
            direction = difference[i]
            j = i + 1
            while j < difference.size and difference[j] == direction:
                j += 1
            if j - i >= 3:
                run = (i, j, direction)
                break
            i = j

        if run is None:
            continue

        start, stop, direction = run
        length = stop - start
        if length not in (3, 4, 5):
            continue

        run_validities = validities[start:stop]
        strongest = np.flatnonzero(np.isclose(
            run_validities, np.max(run_validities), atol=1e-12, rtol=0.0
        ))
        embedded = bool(
            strongest.size == 1
            and 0 < int(strongest[0]) < length - 1
        )

        run_supported_response = 0 if direction > 0 else 1
        chose_run = float(int(row['response']) == run_supported_response)
        cells[(length, embedded)].append(chose_run)

    # Longer runs receive more weight because the competing account's
    # overload manipulation grows with excess run length.
    weights = {3: 1.0, 4: 2.5, 5: 4.0}
    weighted_contrasts = []
    used_weights = []
    for length in (3, 4, 5):
        boundary = cells[(length, False)]
        embedded = cells[(length, True)]
        if len(boundary) == 0 or len(embedded) == 0:
            continue
        contrast = float(np.mean(embedded) - np.mean(boundary))
        weighted_contrasts.append(weights[length] * contrast)
        used_weights.append(weights[length])

    if len(used_weights) == 0:
        return float('nan')
    return float(np.sum(weighted_contrasts) / np.sum(used_weights))
```

**Observed (real) value:** -0.0030 (var=0.0059)
**Predicted under pi_9:** 0.0281 (var=0.0061)
**Predicted under pi_8:** 0.1210 (var=0.0087)

### Experiment 16
**Design**
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    adjacent_choices = []
    separated_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b

        a_support = np.flatnonzero(diff > 0)
        b_support = np.flatnonzero(diff < 0)

        # Identify the option supported by the two-expert 82% pair.
        if a_support.size == 2 and b_support.size == 1:
            pair_positions = a_support
            pair_response = 0
        elif b_support.size == 2 and a_support.size == 1:
            pair_positions = b_support
            pair_response = 1
        else:
            continue

        chose_pair = float(int(row["response"]) == pair_response)
        is_adjacent = int(np.abs(pair_positions[1] - pair_positions[0])) == 1

        if is_adjacent:
            adjacent_choices.append(chose_pair)
        else:
            separated_choices.append(chose_pair)

    if len(adjacent_choices) == 0 or len(separated_choices) == 0:
        return 0.0

    return float(np.mean(adjacent_choices) - np.mean(separated_choices))
```

**Observed (real) value:** -0.0183 (var=0.0044)
**Predicted under pi_9:** 0.0267 (var=0.0071)
**Predicted under pi_8:** 0.1392 (var=0.0118)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all 16 experiments, pi_9 is the stronger account, but neither theory is task-invariant. The clearest head-to-head evidence is asymmetric. In Experiment 1, the observed opponent-count cliff was -0.0205, very close to pi_8's -0.0035 and radically inconsistent with pi_9's 0.5137 prediction. The observed between-subject variance was also low (0.0094), so this is not merely a pooled null hiding a large but heterogeneous competition-gate effect. Thus pi_9's distinctive claim that multiple opposing supporters trigger a large overload penalty is directly falsified. In Experiment 2, however, the observed pair-versus-run advantage was only 0.0550: much closer to pi_9's 0.0088 than to pi_8's 0.3287. This rejects the large pair-chunk benefit and run-overload contrast central to pi_8.\n\nThe wider record strengthens the case against pi_8's structural encoding rules. Experiment 15 found essentially no embedded-strongest protection (-0.0030), versus pi_8's 0.1210. Experiment 16 found no adjacent-pair advantage (-0.0183), versus pi_8's 0.1392. Experiments 11 and 12 likewise showed negligible effects of spacing and terminal pair placement. These results jointly indicate that adjacency, pair chunking, and the strongest member's interior-versus-boundary location do not strongly reorganize recommendation retrieval. Their low observed variances in Experiments 11, 15, and 16 make the repeated null pattern especially consequential. Pi_9 is substantially closer on these tests and also fits Experiments 3-7 and 10-14 reasonably well, including lexicographic dominance, weak-cue override rates, the large redundancy/run-length loss in Experiment 10, negligible spacing effects, and the null terminal-pair effect. Pi_8 has a local advantage in Experiment 8's nonlinear validity-gap result, but both accounts capture its negative direction.\n\nPi_9 nevertheless cannot simply be accepted: its defining opponent-triggered overload gate produces the largest miss in the entire set. Both theories also underpredict the monotonic late-position loss in the 32-feature Experiment 9, although pi_9 is slightly closer. The overall empirical pattern is therefore predominantly validity-ordered and recommendation-normalized, with rapidly saturating redundancy, little sensitivity to adjacency or strongest-member placement, a run-length/redundancy cost when the supported run itself grows, and a late-position accessibility cost only under substantial display load. Crucially, overload appears tied to changes in the recommendation's own run length or representational load, not to the number of separately located opposing supporters. Because pi_8's core structural taxonomy is contradicted by several independent experiments, it should be replaced rather than merely reparameterized.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace pi_8 with a new theory of Load-Limited Recommendation-Level Lexicographic Retrieval. The new theory should propose that same-direction active experts are ordinarily compressed into one recommendation representation indexed by the strongest communicated validity, regardless of whether supporters are adjacent, separated, paired, or arranged around the strongest source. Additional same-direction sources should provide at most a small, rapidly saturating reliability gain; adjacent pairs should have no privileged fidelity, and an internally located strongest member should receive no special protection.\n\nRetain validity-ordered, predominantly lexicographic competition and localized nonlinear source-validity binding uncertainty, since Experiments 3-8 support those components. Introduce an intrinsic run-load mechanism based on the recommendation's own contiguous run length or representational concentration. It should increase smoothly when the run itself grows, thereby allowing the strong negative redundancy contrast in Experiment 10, but it must not be activated merely because the opponent gains multiple separated supporters; this yields the null opponent-count cliff in Experiment 1. The mechanism should also avoid predicting embedded-versus-boundary differences, pair advantages, or generic spacing effects, matching Experiments 2, 11, 13, 15, and 16.\n\nAdd a display-load-gated monotonic accessibility gradient for isolated decisive recommendations that becomes appreciable only in very long displays, around 32 features, to address Experiment 9 without creating terminal recency or a strong effect in the 24-feature Experiment 14. Complete retrieval failure may still produce guessing, followed by a small symmetric lapse. The resulting theory should compete directly with pi_9 by preserving its successful recommendation normalization while making a sharply different claim: overload is intrinsic to growth of a recommendation's own representation, not induced by the count of opposing recommendation sources."
}
```

## Usage

```json
{
  "input_tokens": 39304,
  "output_tokens": 1811
}
```
