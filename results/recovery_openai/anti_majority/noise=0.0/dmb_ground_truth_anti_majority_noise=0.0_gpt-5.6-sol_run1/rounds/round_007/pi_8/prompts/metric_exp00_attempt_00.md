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



## ADVOCATED THEORY
**Description:** Structure-Gated Ensemble Lexicographic Retrieval. Decision makers preserve the communicated validity hierarchy and search encoded recommendation units from highest to lowest validity, but the units available to search depend on the grouping structure of the display. An isolated expert is represented as a source-specific item. Two adjacent experts making the same recommendation are encoded as a compact pair chunk with high, position-invariant fidelity. Contiguous same-direction runs of three or more experts are instead compressed into a single ensemble whose source-to-validity binding becomes progressively less reliable with run length. A uniquely strongest expert embedded inside such a run remains structurally identifiable, whereas a strongest expert at a run boundary receives no such protection from overload. Same-direction experts separated by inactive or opposing experts are represented by a recommendation-level ensemble: additional separated sources provide only a negligible, rapidly saturating retrieval benefit and impose no distance-based crowding cost. Independently, validity labels have a localized nonlinear binding-uncertainty region, so a larger communicated validity gap need not imply more reliable retrieval. Serial position has no terminal-recency bonus. Only isolated sources in sufficiently long displays exhibit a weak subject-specific monotonic accessibility gradient centered near zero, allowing heterogeneous serial-position effects without a systematic pooled recency advantage. Once the encoded units are formed, choice is primarily lexicographic: the first successfully retrieved unit in descending communicated-validity order determines the intended response. Complete retrieval failure produces guessing, and a small execution lapse can reverse the intended response.

**Parameters:**
- n_features: n_features
- validities: validities
- validity_compression: [0.85, 1.2]
- retrieval_baseline: [2.4, 3.2]
- binding_uncertainty_strength: [1.0, 2.0]
- binding_uncertainty_center: [0.58, 0.68]
- binding_uncertainty_width: [0.12, 0.2]
- pair_chunk_fidelity: [0.9, 0.99]
- contiguous_run_overload: [0.45, 0.65]
- noncontiguous_redundancy_gain: [0.0, 0.02]
- long_display_position_slope: [-0.35, 0.35]
- response_lapse: [0.08, 0.14]

**`predict source code`:**
```python
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
**Description:** Competition-Normalized Recommendation Race. People first compress all currently active experts favoring the same option into a single recommendation-level representation. Consequently, redundant experts are retrieved correlatively rather than as independent opportunities for rescue, and reliability gains from additional experts saturate almost immediately. Each recommendation is indexed by its strongest communicated validity and the two opposing recommendation representations compete for retrieval before a validity-ordered, predominantly lexicographic decision. Source-to-validity binding has a localized nonlinear uncertainty region, allowing moderate-to-high validity cues to be less accessible than cues outside that region. Spatial arrangement ordinarily has no effect. The exception is density overload: a recommendation loses binding fidelity when most of its support forms an extended contiguous run and the opposing recommendation itself has multiple active supporters. This competition gate prevents long runs from being penalized merely for containing many sources and leaves adjacent pairs, separated redundancy, and long runs facing a singleton largely unaffected. The strongest member's location within a run provides no protection. In long displays only, an isolated recommendation receives a weak monotonic accessibility loss with later position; there is no terminal-recency process. Failure to retrieve either recommendation produces guessing, followed by a small symmetric response lapse.

**Parameters:**
- n_features: n_features
- validities: validities
- validity_compression: [0.85, 1.2]
- retrieval_baseline: [2.8, 3.5]
- binding_uncertainty_strength: [0.8, 1.5]
- binding_uncertainty_center: [0.58, 0.68]
- binding_uncertainty_width: [0.12, 0.2]
- contiguous_density_overload: [1.0, 1.6]
- recommendation_saturation_gain: [0.0, 0.003]
- contextual_position_loss: [0.0, 0.15]
- response_lapse: [0.07, 0.11]

**`predict source code`:**
```python
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    pair_choices = []\n    run_choices = []\n\n    for _, row in data.iterrows():\n        a = np.asarray(row[\"option_a_ratings\"], dtype=int)\n        b = np.asarray(row[\"option_b_ratings\"], dtype=int)\n        diff = a - b\n\n        pos = np.flatnonzero(diff > 0)\n        neg = np.flatnonzero(diff < 0)\n\n        # Retain the four-supporters-versus-singleton trials only.\n        if len(pos) == 4 and len(neg) == 1:\n            target_idx = pos\n            chose_target = float(int(row[\"response\"]) == 0)\n        elif len(neg) == 4 and len(pos) == 1:\n            target_idx = neg\n            chose_target = float(int(row[\"response\"]) == 1)\n        else:\n            continue\n\n        target_idx = np.sort(target_idx)\n        longest_run = 1\n        current_run = 1\n        for j in range(1, len(target_idx)):\n            if target_idx[j] == target_idx[j - 1] + 1:\n                current_run += 1\n                longest_run = max(longest_run, current_run)\n            else:\n                current_run = 1\n\n        if longest_run == 4:\n            run_choices.append(chose_target)\n        elif longest_run == 2:\n            # In this design, this identifies the two-separated-pairs condition.\n            pair_choices.append(chose_target)\n\n    if len(pair_choices) == 0 or len(run_choices) == 0:\n        return float(\"nan\")\n\n    return float(np.mean(pair_choices) - np.mean(run_choices))",
  "rationale": "This is the orientation-invariant two-pairs advantage among the four-source trials: the probability of choosing the option supported by two separated adjacent pairs minus the probability of choosing the option supported by one contiguous run of four. It pools each condition with its exact A/B reversal, cancelling response-side bias, and uses 16 observations per condition per subject. The advocated theory predicts a large positive contrast because two high-fidelity pair chunks provide separate retrieval opportunities, whereas a length-four run suffers substantial overload. The competing theory predicts essentially zero because it compresses all four supporters into one recommendation and its run-overload gate is inactive against a singleton opponent. Restricting the metric to the strongest predicted structural contrast avoids diluting it with the smaller three-source effect."
}
```

## Usage

```json
{
  "input_tokens": 8067,
  "output_tokens": 2116
}
```
