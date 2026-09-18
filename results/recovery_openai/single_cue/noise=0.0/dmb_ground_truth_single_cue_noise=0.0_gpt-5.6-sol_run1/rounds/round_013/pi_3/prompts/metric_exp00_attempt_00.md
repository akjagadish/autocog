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
**Validities (n_features=26):** [0.8, 0.99, 0.52, 0.75, 0.52, 0.99, 0.6, 0.7, 0.68, 0.55, 0.68, 0.7, 0.85, 0.68, 0.7, 0.58, 0.7, 0.68, 0.65, 0.52, 0.99, 0.72, 0.99, 0.52, 0.9, 0.5]

**Trial pairs (n=16):**
  trial 1: A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  trial 9: A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 13: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 14: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 15: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  trial 16: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]

**Rationale:** This is a 2×2 reliability-drop-orientation experiment with mirrored spatial replication. Every base trial has exactly two discriminating cues: the higher-validity cue favors A and the lower-validity cue favors B. Consequently, descending-validity processing always yields the identical direction sequence A,B. Recency-Gated Reason Overwrite therefore predicts exactly the same strong preference for the lower-validity cue's option B in all eight base trials: cue count is two, so there is no fatigue, and validity magnitude and physical orientation are otherwise ignored. The critical factors are validity gap (large: 99% versus 52%; shallow: 70% versus 68%) and physical orientation. In high-before-low banks, the physical sequence is A(high),B(low), creating a directional boundary after a validity drop. In low-before-high banks, the same validity pair appears as B(low),A(high), so the competitor's left-neighbor drop is exactly zero. Priority-Latency Coalition Accumulation thus predicts a positive orientation effect on choice of the low-validity-supported option: a high-before-low drop delays the opposing low cue, amplifies its surprise-weighted update, and preferentially refreshes it. This effect should be much larger for the 47-point gap, where local_drop=2.35, than for the 2-point gap, where local_drop=0.10. The diagnostic statistic is therefore an interaction: [low-option choice in large-gap high-before-low minus large-gap low-before-high] should exceed the corresponding shallow-gap contrast. Overwrite predicts both contrasts and their interaction to be exactly zero. Each cell is instantiated in two banks mirrored around the display center, equating mean position; no discriminating cue occupies an endpoint, and cues within each bank are adjacent, minimizing endpoint and isolation confounds. Exact A/B reversals balance response side. Sixteen unique pairs repeated six times yield exactly 96 trials.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Recency-Gated Reason Overwrite with Processing Fatigue. People inspect cues sequentially in descending order of instructed validity and continue after finding a discriminating cue. An attended discriminating cue establishes a provisional choice, and each subsequently attended discriminating cue can overwrite that choice. Consequently, the final attended reason has disproportionate control. Occasional failures to attend a cue preserve an earlier reason, while response lapses dilute strong recency effects. In addition, processing more than two discriminating reasons produces task-invariant update fatigue: the retained decision is expressed with progressively lower reliability as the discriminating-cue count increases. This preserves strong second-reason overwrite on short conflicts while moderating near-deterministic last-reason control on longer sequences.

**Parameters:**
- validities: validities
- cue_attention: [0.95, 1.0]
- overwrite_rate: [0.95, 1.0]
- update_fatigue: [0.06, 0.17]
- lapse_rate: [0.15, 0.41]

**`predict source code`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Recency-Gated Reason Overwrite expects shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    attention = float(parameters["cue_attention"])
    overwrite = float(parameters["overwrite_rate"])
    fatigue = float(parameters["update_fatigue"])
    lapse = float(parameters["lapse_rate"])

    # Stable sorting represents the displayed high-to-low validity scan;
    # ties retain their display order.
    cue_order = np.argsort(-validities, kind="stable")
    diff = stim[0] - stim[1]

    # Exact marginal distribution over the provisional state:
    # A selected, B selected, or no discriminating cue encoded yet.
    p_a, p_b, p_uncommitted = 0.0, 0.0, 1.0
    discriminating_count = 0

    for j in cue_order:
        direction = diff[j]
        if direction == 0:
            continue

        discriminating_count += 1
        old_a, old_b, old_u = p_a, p_b, p_uncommitted

        if direction > 0:
            # An attended A cue initializes an uncommitted decision and
            # overwrites a conflicting B decision with the overwrite rate.
            p_a = old_a + attention * overwrite * old_b + attention * old_u
            p_b = old_b * (1.0 - attention * overwrite)
            p_uncommitted = old_u * (1.0 - attention)
        else:
            p_b = old_b + attention * overwrite * old_a + attention * old_u
            p_a = old_a * (1.0 - attention * overwrite)
            p_uncommitted = old_u * (1.0 - attention)

    # If no discriminating cue was encoded, either response is an equal guess.
    p_core = np.array([
        p_a + 0.5 * p_uncommitted,
        p_b + 0.5 * p_uncommitted
    ], dtype=np.float64)

    # Processing the first reason and its immediate successor is reliable.
    # Additional discriminating updates progressively weaken expression of
    # the retained choice without disrupting the crucial second-cue reversal.
    excess_updates = max(0, discriminating_count - 2)
    retention = (1.0 - fatigue) ** excess_updates
    p_core = retention * p_core + (1.0 - retention) * np.array([0.5, 0.5])

    # Independent execution/inattention lapse.
    probs = (1.0 - lapse) * p_core + lapse * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, None)
    probs /= probs.sum()
    return probs
```

**`policy source code`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** Priority-Latency Coalition Accumulation. Before deliberation, the display is encoded in parallel into a priority-latency map. Absolute instructed reliability gives diagnostically strong cues early access, a stable but heterogeneous directional scan tendency delays cues along the preferred scan path, scan-congruent endpoints attract selective reinspection, and an opposing cue following a sharp absolute validity drop receives surprise priority. Priority is sampled with modest trial-level noise rather than imposing a deterministic order. Evidence then enters a bounded leaky accumulator, so later or refreshed evidence is more available at choice. Updates remain graded: reliability affects update magnitude only weakly and compressively, local surprise increases revision, capacity load attenuates late encoding, and a coherent incumbent coalition continuously gates opposition. Coalition resistance is reduced for spatially isolated challengers, allowing isolated terminal reasons to reverse a coalition without making all terminal cues dominant. Subject-level heterogeneity in validity sensitivity, scan direction, endpoint salience, contrast gain, retention, anchoring, capacity, priority noise, and lapse produces nonuniform recency, crossover, and endpoint effects without changing mechanisms across experiments.

**Parameters:**
- validities: validities
- validity_sensitivity: [0.45, 0.85]
- scan_direction: [-0.25, 1.45]
- endpoint_salience: [0.45, 1.65]
- contrast_gain: [0.28, 0.72]
- accumulator_retention: [0.40, 0.62]
- anchoring_strength: [0.22, 0.58]
- capacity: [4.2, 7.4]
- priority_temperature: [0.02, 0.18]
- lapse_rate: [0.05, 0.16]

**`predict source code`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Priority-Latency Coalition Accumulation expects shape "
            f"(2, n_features); got {stim.shape}."
        )

    n_features = int(stim.shape[1])
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    validity_sensitivity = float(parameters["validity_sensitivity"])
    scan_direction = float(parameters["scan_direction"])
    endpoint_salience = float(parameters["endpoint_salience"])
    contrast_gain = float(parameters["contrast_gain"])
    accumulator_retention = float(parameters["accumulator_retention"])
    anchoring_strength = float(parameters["anchoring_strength"])
    capacity = float(parameters["capacity"])
    priority_temperature = float(parameters["priority_temperature"])
    lapse_rate = float(parameters["lapse_rate"])

    diff = stim[0] - stim[1]
    discriminating = np.flatnonzero(diff != 0.0)
    if discriminating.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    clipped_v = np.clip(validities, 0.5001, 0.999)
    reliability = np.log(clipped_v / (1.0 - clipped_v))

    # A fixed, task-invariant reliability scale preserves absolute gaps. It
    # therefore distinguishes nearly equal weak cues from a .99-versus-.52
    # conflict instead of mapping both two-cue displays to the same +/-1 pair.
    absolute_reliability = np.clip(
        (reliability - 0.20) / 1.55, -0.15, 3.0
    )

    # The complete map is computed before any sequential accumulation.
    # Local contrast is defined relative to the nearest discriminating item
    # on the left, reflecting perceptual grouping in the displayed layout.
    local_drop = np.zeros(n_features, dtype=float)
    directional_boundary = np.zeros(n_features, dtype=float)
    isolation = np.zeros(n_features, dtype=float)
    previous_j = None
    for j_raw in discriminating:
        j = int(j_raw)
        if previous_j is not None:
            gap_fraction = float(j - previous_j - 1) / float(
                max(1, n_features - 1)
            )
            isolation[j] = max(isolation[j], gap_fraction)
            isolation[previous_j] = max(isolation[previous_j], gap_fraction)

            previous_direction = float(np.sign(diff[previous_j]))
            current_direction = float(np.sign(diff[j]))
            if previous_direction != current_direction:
                directional_boundary[j] = 1.0
                drop = max(0.0, float(validities[previous_j] - validities[j]))
                # Fixed scaling retains the absolute distinction between a
                # shallow and a conspicuous instructed-validity drop.
                local_drop[j] = min(3.0, drop / 0.20)
        previous_j = j

    positions = np.arange(n_features, dtype=float) / float(max(1, n_features - 1))
    signed_position = 2.0 * positions - 1.0
    left_edge = (np.arange(n_features) == 0).astype(float)
    right_edge = (np.arange(n_features) == n_features - 1).astype(float)

    # Lower latency means earlier sampling. Reliability has the strongest
    # effect on access. Endpoint delay follows the subject's scan direction,
    # is amplified by isolation, and is attenuated for exceptionally reliable
    # cues, which can consequently enter before a spatially earlier weak cue.
    scan_sign = 1.0 if scan_direction >= 0.0 else -1.0
    endpoint_delay = (
        endpoint_salience
        * scan_sign
        * (right_edge - left_edge)
        * (0.35 + 0.90 * isolation)
        / (1.0 + np.maximum(0.0, absolute_reliability))
    )
    contrast_delay = contrast_gain * local_drop * directional_boundary
    latency = (
        -validity_sensitivity * absolute_reliability
        + scan_direction * signed_position
        + endpoint_delay
        + contrast_delay
    )

    # Modest Gumbel perturbations instantiate priority-dependent sampling
    # rather than a perfectly deterministic ranking on every presentation.
    sampled_latency = latency + np.random.gumbel(
        loc=0.0, scale=priority_temperature, size=n_features
    )

    # Stable index tie-breaking preserves the physical layout when sampled map
    # values are equal. Only discriminating cues consume integration capacity.
    cue_order = sorted(
        [int(j) for j in discriminating],
        key=lambda j: (float(sampled_latency[j]), j)
    )

    accumulator = 0.0
    incumbent_direction = 0.0
    coherent_count = 0.0
    coalition_strength = 0.0

    def apply_update(j, refresh_fraction, step_index,
                     accumulator, incumbent_direction,
                     coherent_count, coalition_strength):
        direction = float(np.sign(diff[j]))

        # Reliability primarily controls access priority. Its direct effect on
        # evidence magnitude is deliberately shallow, so a highly valid cue
        # sampled early does not remain disproportionately strong after later
        # weak evidence arrives.
        reliability_strength = float(np.exp(
            0.035 * validity_sensitivity * absolute_reliability[j]
        ))
        reliability_strength = float(np.clip(
            reliability_strength, 0.94, 1.14
        ))

        # A reliability drop is surprising only at a directional boundary.
        # Surprise boosts the challenger smoothly, with no reversal switch.
        surprise = 1.0 + (
            contrast_gain
            * local_drop[j]
            * directional_boundary[j]
            * (0.30 + 0.35 * min(1.0, abs(accumulator)))
        )
        surprise = float(np.clip(surprise, 1.0, 3.2))

        # Capacity loss is gradual and applies most strongly after the
        # subject-specific integration span has been exceeded.
        overload = max(0.0, float(step_index + 1) - capacity)
        capacity_factor = 1.0 / np.sqrt(1.0 + 0.28 * overload)

        opposing = bool(
            incumbent_direction != 0.0 and direction != incumbent_direction
        )
        if opposing:
            # Coherent agreement anchors the incumbent continuously. Spatial
            # isolation weakens the gate, allowing a conspicuous separated
            # challenger to revise more than a contiguous singleton.
            coherence = max(0.0, coherent_count - 1.0)
            anchor_mass = (
                coherence
                * min(2.0, coalition_strength)
                * (1.0 - 0.72 * min(1.0, isolation[j]))
            )
            anchor_gate = 1.0 + anchoring_strength * anchor_mass
        else:
            anchor_gate = 1.0

        update = (
            refresh_fraction
            * reliability_strength
            * surprise
            * capacity_factor
            / anchor_gate
        )

        accumulator = (
            accumulator_retention * accumulator + direction * update
        )
        accumulator = float(np.clip(accumulator, -2.75, 2.75))

        # Coalition state is graded. Opposition erodes the incumbent before
        # a new coherent coalition is installed; there is no categorical run
        # policy or thresholded strategy change.
        if incumbent_direction == 0.0:
            incumbent_direction = direction
            coherent_count = refresh_fraction
            coalition_strength = min(2.5, update)
        elif direction == incumbent_direction:
            coherent_count += refresh_fraction
            coalition_strength = min(
                2.5,
                0.78 * coalition_strength + 0.55 * update
            )
        else:
            erosion = refresh_fraction * update
            coalition_strength -= 0.72 * erosion
            coherent_count -= 0.62 * refresh_fraction
            if coalition_strength <= 0.0 or coherent_count <= 0.0:
                incumbent_direction = direction
                coherent_count = max(0.5, refresh_fraction)
                coalition_strength = min(2.5, 0.65 * update)
            else:
                coalition_strength = max(0.0, coalition_strength)
                coherent_count = max(0.0, coherent_count)

        return (
            accumulator,
            incumbent_direction,
            coherent_count,
            coalition_strength
        )

    for step_index, j in enumerate(cue_order):
        accumulator, incumbent_direction, coherent_count, coalition_strength = apply_update(
            j, 1.0, step_index, accumulator, incumbent_direction,
            coherent_count, coalition_strength
        )

    # Refresh is selective rather than a universal terminal bonus. It depends
    # on endpoint-by-scan congruence, endpoint isolation, and absolute local
    # contrast, preventing contiguous weak terminals from controlling every
    # display while preserving conspicuous endpoint effects.
    refresh_scores = {}
    for j in cue_order:
        scan_endpoint = right_edge[j] if scan_direction >= 0.0 else left_edge[j]
        opposite_endpoint = left_edge[j] if scan_direction >= 0.0 else right_edge[j]
        endpoint_refresh = endpoint_salience * (
            scan_endpoint * (0.20 + 1.35 * isolation[j])
            + 0.12 * opposite_endpoint * isolation[j]
        )
        refresh_scores[j] = (
            endpoint_refresh
            + contrast_gain * local_drop[j] * directional_boundary[j]
            + 0.04 * max(0.0, absolute_reliability[j])
        )

    refresh_j = max(
        cue_order,
        key=lambda j: (refresh_scores[j], sampled_latency[j])
    )
    raw_refresh = float(refresh_scores[refresh_j])
    refresh_fraction = 0.32 * raw_refresh / (1.0 + raw_refresh)
    if refresh_fraction > 1e-8:
        accumulator, incumbent_direction, coherent_count, coalition_strength = apply_update(
            refresh_j, refresh_fraction, len(cue_order), accumulator,
            incumbent_direction, coherent_count, coalition_strength
        )

    # Display load reduces readout precision even for unanimous evidence,
    # yielding a small rather than catastrophic decline with additional cues.
    nominal_overload = max(0.0, float(n_features) - capacity)
    evidence_overload = max(0.0, float(len(cue_order)) - capacity)
    precision = 2.65 / (
        1.0 + 0.045 * nominal_overload + 0.075 * evidence_overload
    )
    precision = float(np.clip(precision, 0.65, 2.65))

    logit_a = float(np.clip(precision * accumulator, -60.0, 60.0))
    p_a = 1.0 / (1.0 + np.exp(-logit_a))
    core = np.array([p_a, 1.0 - p_a], dtype=np.float64)
    probs = (
        (1.0 - lapse_rate) * core
        + lapse_rate * np.array([0.5, 0.5], dtype=np.float64)
    )
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    validities = np.asarray([\n        0.8, 0.99, 0.52, 0.75, 0.52, 0.99, 0.6, 0.7, 0.68,\n        0.55, 0.68, 0.7, 0.85, 0.68, 0.7, 0.58, 0.7, 0.68,\n        0.65, 0.52, 0.99, 0.72, 0.99, 0.52, 0.9, 0.5\n    ], dtype=float)\n\n    high_before_low = []\n    low_before_high = []\n\n    for _, row in data.iterrows():\n        a = np.asarray(row[\"option_a_ratings\"], dtype=float)\n        b = np.asarray(row[\"option_b_ratings\"], dtype=float)\n        diff = a - b\n        idx = np.flatnonzero(diff != 0)\n        if idx.size != 2:\n            continue\n\n        cue_validities = validities[idx]\n        high_idx = int(idx[np.argmax(cue_validities)])\n        low_idx = int(idx[np.argmin(cue_validities)])\n        gap = float(validities[high_idx] - validities[low_idx])\n\n        # Restrict the statistic to the diagnostic .99-versus-.52 conflicts.\n        if gap < 0.40:\n            continue\n\n        response = int(row[\"response\"])\n        low_option = 0 if diff[low_idx] > 0 else 1\n        chose_low = float(response == low_option)\n\n        if high_idx < low_idx:\n            high_before_low.append(chose_low)\n        else:\n            low_before_high.append(chose_low)\n\n    if len(high_before_low) == 0 or len(low_before_high) == 0:\n        return float(\"nan\")\n\n    return float(np.mean(high_before_low) - np.mean(low_before_high))\n",
  "rationale": "This is the physical-orientation effect on choosing the option supported by the lower-validity cue, restricted to the large .99-versus-.52 conflicts. Recency-Gated Reason Overwrite processes the high-validity cue and then the low-validity cue in that same validity order regardless of their display locations, so it predicts essentially zero difference between high-before-low and low-before-high layouts. Priority-Latency Coalition Accumulation predicts a sizable positive difference: only high-before-low layouts give the opposing low cue a large left-neighbor reliability drop, delaying it and boosting its surprise and refresh updates. Restricting to the large-gap trials avoids diluting this effect with the shallow-gap manipulation, while pooling all mirrored banks and six repetitions supplies 24 observations per orientation per subject and keeps sampling variance relatively low."
}
```

## Usage

```json
{
  "input_tokens": 8818,
  "output_tokens": 2085
}
```
