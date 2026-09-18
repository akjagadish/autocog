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
**Validities (n_features=6):** [0.74, 0.95, 0.58, 0.83, 0.52, 0.89]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 2: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 3: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  trial 4: A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 5: A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1]
  trial 6: A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 0, 1, 0, 0]
  trial 7: A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0, 1]
  trial 8: A=[0, 1, 1, 1, 0, 1]  B=[1, 0, 0, 0, 1, 0]

**Rationale:** This fixed-conflict-count design isolates relational organization while keeping all six experts discriminating on every trial (k=6). The compact pattern has one leading cue favoring the designated option followed by a single contiguous five-cue opposing coalition (+-----). The three fragmented patterns retain the same leading-cue winner but distribute a second supporting cue at positions 3, 4, or 5, producing four alternating coalitions and a 2-to-4 tally disadvantage. Conflict-Density Adaptive Serial–Parallel Control remains essentially entirely in sparse serial mode at k=6, well below its 7.25–7.35 transition. It should therefore continue to favor the first-cue option in both compact and fragmented displays, generally around .55–.62, with only a modest positional gradient as the residual supporting cue moves later and encounters the terminal-gap penalty. Relational Chunk–Anchor Arbitration instead predicts a large structural discontinuity. The compact display has a maximally exclusive singleton anchor and a perfectly concentrated opposing coalition, so it should choose the anchor option well above chance, approximately .67–.90 after recurrence stabilizes the common focal location. Fragmentation rises from .20 to .60 in the other displays, their anchor schemas are substantially weaker, and saturation is maximal; this strongly recruits comparison. Because comparison sees a 2-to-4 majority against the anchor, the predicted probability of choosing the first-cue option should collapse to roughly .05–.30. Thus the critical dissociation is near-invariance with continued first-cue preference under the advocated theory versus a large compact-to-fragmented reversal under the competing theory, despite identical k, saturation, earliest discriminating location, and absence of ties. All patterns and their A/B mirrors are included. The eight unique pairs repeat 12 times each, yielding 96 trials; because every display has the same singleton coalition at feature 1, random trial order does not undermine the competing theory's recurrence mechanism.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Conflict-Density Adaptive Serial–Parallel Control theory. People do not commit to one search architecture for an entire experiment. Instead, each display automatically recruits a control mode based on the absolute number of discriminating cues. Below a capacity-like transition point, attention follows a stable hierarchy dominated by display position and choice is normally based on the first discriminating cue. A constrained accessibility failure can occasionally make that cue unavailable, after which attention continues serially through the remaining discriminating cues using a shallow rank-normalized accessibility gradient. Residual cues supporting both options create modest interference with the initial stopping decision and therefore smoothly increase the probability of continuing beyond the leading cue; unanimous residual evidence does not. A terminal-gap penalty selectively reduces access to residual cues that are separated from the leading cue by a very large positional gap. Instructed validity only provides a weak tie-breaking adjustment and cannot independently promote a substantially later cue. When the number of discriminating cues exceeds the transition point, the display supports parallel comparison and evidence is integrated using a noisy tally with only weak positional and validity weighting. The transition between modes is smooth and stimulus-contingent, rather than determined by experiment identity, trial history, or feedback.

**Parameters:**
- validities: validities
- transition_count: [7.25, 7.35]
- transition_steepness: [7.0, 8.5]
- position_bias: [0.90, 1.05]
- validity_modifier: [0.01, 0.04]
- serial_second_look: [0.36, 0.40]
- mixed_residual_boost: [0.10, 0.14]
- residual_rank_decay: [0.08, 0.12]
- terminal_gap_penalty: [4.0, 5.0]
- serial_precision: [1.65, 1.90]
- positional_discount: [0.06, 0.14]
- tally_precision: [1.05, 1.35]
- epsilon: [0.01, 0.04]

**`predict source code`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    # Conflict-Density Adaptive Serial-Parallel Control.
    # History is intentionally unused: without outcome feedback, the stable
    # accessibility hierarchy and stimulus-triggered control rule do not learn
    # across trials.
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expected state with shape (2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.shape != (n_features,):
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    diff = a - b
    discriminating = np.flatnonzero(diff != 0.0)
    k = int(discriminating.size)
    if k == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    # Put validity on a bounded scale. Its influence remains weak in both
    # modes and therefore cannot by itself overcome a full position step.
    v_range = float(np.max(validities) - np.min(validities))
    if np.isfinite(v_range) and v_range > 0.0:
        v_scaled = (validities - np.mean(validities)) / v_range
    else:
        v_scaled = np.zeros(n_features, dtype=np.float64)

    validity_modifier = float(parameters["validity_modifier"])

    # Sparse mode: stable, position-dominated serial hierarchy. The validity
    # adjustment is much smaller than the priority difference between adjacent
    # display positions, so it serves only as a local tie-breaker.
    position_bias = float(parameters["position_bias"])
    positions = np.arange(n_features, dtype=np.float64)
    serial_priority = -position_bias * positions + validity_modifier * v_scaled
    serial_order = np.argsort(-serial_priority, kind="stable")

    discriminating_order = [int(j) for j in serial_order if diff[j] != 0.0]
    first_discriminating = discriminating_order[0]
    first_winner = 0 if diff[first_discriminating] > 0.0 else 1

    # The hierarchy itself remains stable, but a constrained accessibility
    # failure sometimes omits its leading discriminating member. Mixed support
    # among the residual cues creates interference with stopping and modestly
    # increases continuation; unanimous residual support leaves it unchanged.
    # Attention then continues through the residual hierarchy. Conditional on
    # a miss, residual accessibility decays shallowly by ordinal rank. A
    # separate terminal-gap penalty limits access to cues lying very far from
    # the leading cue without reinstating a steep gradient over early/middle
    # cues.
    serial_second_look = float(np.clip(
        parameters["serial_second_look"], 0.0, 1.0
    ))
    if len(discriminating_order) > 2:
        residual_signs = np.sign(diff[np.asarray(discriminating_order[1:], dtype=int)])
        residual_mixedness = 1.0 - abs(float(np.mean(residual_signs)))
        mixed_residual_boost = float(parameters["mixed_residual_boost"])
        serial_second_look += mixed_residual_boost * residual_mixedness
        serial_second_look = float(np.clip(serial_second_look, 0.0, 1.0))

    p_selected = np.zeros(2, dtype=np.float64)
    p_selected[first_winner] += 1.0 - serial_second_look

    if len(discriminating_order) == 1:
        p_selected[first_winner] += serial_second_look
    else:
        residual = np.asarray(discriminating_order[1:], dtype=int)
        residual_rank_decay = float(parameters["residual_rank_decay"])
        terminal_gap_penalty = float(parameters["terminal_gap_penalty"])

        if residual.size > 1:
            residual_rank = np.arange(
                residual.size, dtype=np.float64
            ) / float(residual.size - 1)
        else:
            residual_rank = np.zeros(1, dtype=np.float64)

        if n_features > 1:
            normalized_gap = (
                positions[residual] - positions[first_discriminating]
            ) / float(n_features - 1)
        else:
            normalized_gap = np.zeros(residual.size, dtype=np.float64)

        large_gap = np.maximum(normalized_gap - 0.55, 0.0)
        residual_logits = (
            -residual_rank_decay * residual_rank
            -terminal_gap_penalty * large_gap
        )
        residual_logits -= np.max(residual_logits)
        residual_probs = np.exp(residual_logits)
        residual_probs /= residual_probs.sum()

        for j, cue_prob in zip(residual, residual_probs):
            residual_winner = 0 if diff[j] > 0.0 else 1
            p_selected[residual_winner] += serial_second_look * cue_prob

    serial_precision = float(parameters["serial_precision"])
    if serial_precision >= 0.0:
        serial_reliability = 1.0 / (1.0 + np.exp(-serial_precision))
    else:
        ex = np.exp(serial_precision)
        serial_reliability = ex / (1.0 + ex)

    p_serial = (
        serial_reliability * p_selected
        + (1.0 - serial_reliability) * p_selected[::-1]
    )

    # Dense mode: all discriminating cues contribute in parallel. Position
    # produces only a shallow gradient here, and validity is a weak multiplier
    # rather than a lexicographic ordering mechanism.
    if n_features > 1:
        normalized_position = positions / float(n_features - 1)
    else:
        normalized_position = np.zeros(1, dtype=np.float64)

    positional_discount = float(parameters["positional_discount"])
    tally_weights = np.exp(-positional_discount * normalized_position)
    tally_weights *= 1.0 + validity_modifier * v_scaled
    tally_weights = np.clip(tally_weights, 1e-8, None)

    signed_tally = float(np.dot(tally_weights, diff))
    tally_precision = float(parameters["tally_precision"])
    tally_logits = np.array(
        [tally_precision * signed_tally, 0.0], dtype=np.float64
    )
    tally_logits -= np.max(tally_logits)
    p_tally = np.exp(tally_logits)
    p_tally /= p_tally.sum()

    # A smooth capacity transition makes the architecture contingent on the
    # current conflict count. The narrow parameter ranges encode constrained
    # subject heterogeneity while preserving a common control regime.
    transition_count = float(parameters["transition_count"])
    transition_steepness = float(parameters["transition_steepness"])
    transition_argument = float(np.clip(
        transition_steepness * (float(k) - transition_count), -60.0, 60.0
    ))
    dense_gate = 1.0 / (1.0 + np.exp(-transition_argument))

    p_core = (1.0 - dense_gate) * p_serial + dense_gate * p_tally

    epsilon = float(np.clip(parameters["epsilon"], 0.0, 1.0))
    probs = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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
**Description:** Relational Chunk–Anchor Arbitration theory. Before choosing, people perceptually organize discriminating cues into contiguous, same-direction coalitions rather than treating conflict count as the primary control variable. An early compact coalition becomes an anchor when it has a coherent opposing coalition, recurs across trials, or occupies an invariant focal location. The anchor normally determines choice, and communicated validity can only slightly reinforce cues already contained in that focal coalition. Comparative processing is recruited when the display is saturated and structurally fragmented—especially when evidence alternates among several coalitions—causing a weakly weighted majority comparison. Thus, equal conflict counts can produce different strategies, while different counts such as k=7 and k=9 can produce the same strategy when their relational organization is alike. Compact two-coalition displays preserve strong primacy, whereas fully saturated, fragmented displays elicit majority-sensitive responding. Subject-specific variation in anchor reliability and comparison recruitment produces stable individual differences without outcome learning.

**Parameters:**
- validities: validities
- anchor_accuracy: [0.84, 0.94]
- validity_reinforcement: [0.00, 0.03]
- stability_weight: [0.25, 0.35]
- comparison_bias: [-4.5, -3.5]
- saturation_weight: [6.5, 7.5]
- fragmentation_weight: [3.5, 4.5]
- anchor_schema_weight: [7.0, 8.0]
- comparison_position_discount: [0.00, 0.04]
- comparison_validity_weight: [0.00, 0.04]
- comparison_precision: [1.5, 2.2]
- epsilon: [0.01, 0.04]

**`predict source code`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    # Relational Chunk-Anchor Arbitration.
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expected state with shape (2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = int(stim.shape[1])
    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.shape != (n_features,):
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    diff = a - b
    signs = np.sign(diff)
    discriminating = np.flatnonzero(signs != 0.0)
    k = int(discriminating.size)
    if k == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    def first_coalition(sign_vector):
        idx = np.flatnonzero(sign_vector != 0.0)
        if idx.size == 0:
            return np.asarray([], dtype=int)
        first = int(idx[0])
        direction = float(sign_vector[first])
        members = [first]
        j = first + 1
        # Spatial contiguity matters: a tie or reversal terminates the chunk.
        while j < sign_vector.size and sign_vector[j] == direction:
            members.append(j)
            j += 1
        return np.asarray(members, dtype=int)

    anchor = first_coalition(signs)
    anchor_direction = float(signs[int(anchor[0])])
    anchor_len = int(anchor.size)

    # Count contiguous, same-direction discriminating runs. Ties separate runs.
    run_count = 0
    previous_sign = 0.0
    previous_index = -2
    for j in discriminating:
        j = int(j)
        s = float(signs[j])
        if j != previous_index + 1 or s != previous_sign:
            run_count += 1
        previous_index = j
        previous_sign = s

    if k > 1:
        fragmentation = float(run_count - 1) / float(k - 1)
    else:
        fragmentation = 0.0
    saturation = float(k) / float(max(n_features, 1))

    # A good anchor schema has a compact initial chunk, little same-direction
    # evidence outside that chunk, and a spatially coherent opposing coalition.
    same_total = int(np.sum(signs == anchor_direction))
    opposite_positions = np.flatnonzero(signs == -anchor_direction)
    if opposite_positions.size == 0:
        opposite_concentration = 0.0
    else:
        longest = 1
        current = 1
        for m in range(1, opposite_positions.size):
            if int(opposite_positions[m]) == int(opposite_positions[m - 1]) + 1:
                current += 1
                longest = max(longest, current)
            else:
                current = 1
        opposite_concentration = float(longest) / float(opposite_positions.size)

    focal_exclusivity = float(anchor_len) / float(max(same_total, 1))
    compactness = float(np.exp(-0.70 * float(anchor_len - 1)))
    current_schema = compactness * focal_exclusivity * opposite_concentration

    # Repeated displays stabilize a relational anchor without learning cue
    # correctness. We measure whether the same earliest location and coalition
    # have recurred in the subject's preceding displays.
    past_a = history.get("option_a_ratings", [])
    past_b = history.get("option_b_ratings", [])
    focal_matches = []
    coalition_matches = []
    for old_a, old_b in zip(past_a, past_b):
        old_a = np.asarray(old_a, dtype=np.float64)
        old_b = np.asarray(old_b, dtype=np.float64)
        if old_a.shape != (n_features,) or old_b.shape != (n_features,):
            continue
        old_signs = np.sign(old_a - old_b)
        old_anchor = first_coalition(old_signs)
        if old_anchor.size == 0:
            continue
        focal_matches.append(float(int(old_anchor[0]) == int(anchor[0])))
        coalition_matches.append(float(np.array_equal(old_anchor, anchor)))

    if focal_matches:
        recurrence = 0.5 * float(np.mean(focal_matches)) + 0.5 * float(np.mean(coalition_matches))
    else:
        recurrence = 0.0

    stability_weight = float(parameters["stability_weight"])
    schema_strength = current_schema * (0.65 + stability_weight * recurrence)
    schema_strength = float(np.clip(schema_strength, 0.0, 1.0))

    # Anchor route. Validity cannot select a nonfocal cue; it only produces a
    # small reliability increment if the already focal coalition is relatively
    # valid. This precludes global validity-rank search and late-cue promotion.
    if np.ptp(validities) > 0.0:
        v_scaled = (validities - np.mean(validities)) / np.ptp(validities)
    else:
        v_scaled = np.zeros(n_features, dtype=np.float64)

    focal_validity = max(0.0, float(np.mean(v_scaled[anchor])))
    anchor_accuracy = float(parameters["anchor_accuracy"])
    validity_reinforcement = float(parameters["validity_reinforcement"])
    anchor_accuracy += validity_reinforcement * focal_validity
    anchor_accuracy = float(np.clip(anchor_accuracy, 0.5, 0.97))

    anchor_winner = 0 if anchor_direction > 0.0 else 1
    p_anchor = np.empty(2, dtype=np.float64)
    p_anchor[anchor_winner] = anchor_accuracy
    p_anchor[1 - anchor_winner] = 1.0 - anchor_accuracy

    # Comparison route: a majority-sensitive tally with only shallow position
    # and validity modulation. Neither modulation can create lexicographic use
    # of a late high-validity cue.
    if n_features > 1:
        normalized_position = np.arange(n_features, dtype=np.float64) / float(n_features - 1)
    else:
        normalized_position = np.zeros(1, dtype=np.float64)

    comparison_position_discount = float(parameters["comparison_position_discount"])
    comparison_validity_weight = float(parameters["comparison_validity_weight"])
    comparison_weights = np.exp(-comparison_position_discount * normalized_position)
    comparison_weights *= 1.0 + comparison_validity_weight * v_scaled
    comparison_weights = np.clip(comparison_weights, 1e-8, None)

    signed_comparison = float(np.dot(comparison_weights, signs))
    comparison_precision = float(parameters["comparison_precision"])
    comparison_logits = np.array(
        [comparison_precision * signed_comparison, 0.0], dtype=np.float64
    )
    comparison_logits -= np.max(comparison_logits)
    p_compare = np.exp(comparison_logits)
    p_compare /= p_compare.sum()

    # Arbitration depends jointly on saturation, coalition fragmentation,
    # relational anchor quality, and its recurrence. There is no fixed k
    # threshold, so similarly organized k=7 and k=9 displays remain alike.
    comparison_bias = float(parameters["comparison_bias"])
    saturation_weight = float(parameters["saturation_weight"])
    fragmentation_weight = float(parameters["fragmentation_weight"])
    anchor_schema_weight = float(parameters["anchor_schema_weight"])

    gate_argument = (
        comparison_bias
        + saturation_weight * saturation
        + fragmentation_weight * fragmentation
        - anchor_schema_weight * schema_strength
    )
    gate_argument = float(np.clip(gate_argument, -60.0, 60.0))
    comparison_gate = 1.0 / (1.0 + np.exp(-gate_argument))

    p_core = (1.0 - comparison_gate) * p_anchor + comparison_gate * p_compare

    epsilon = float(np.clip(parameters["epsilon"], 0.0, 1.0))
    probs = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    \"\"\"Compact-minus-fragmented change in first-cue-consistent choice.\"\"\"\n    compact_scores = []\n    fragmented_scores = []\n\n    for _, row in data.iterrows():\n        a = np.asarray(row[\"option_a_ratings\"], dtype=float)\n        b = np.asarray(row[\"option_b_ratings\"], dtype=float)\n        if a.shape != b.shape or a.size == 0:\n            continue\n\n        signs = np.sign(a - b)\n        if signs[0] == 0 or np.any(signs == 0):\n            continue\n\n        # Code whether the response follows the option favored by feature 1.\n        first_cue_response = 0 if signs[0] > 0 else 1\n        follows_first = float(int(row[\"response\"]) == first_cue_response)\n\n        # The compact pattern has a singleton initial coalition followed by\n        # one unanimous opposing coalition: +----- or its mirror -+++++.\n        is_compact = bool(np.all(signs[1:] == -signs[0]))\n        if is_compact:\n            compact_scores.append(follows_first)\n        else:\n            fragmented_scores.append(follows_first)\n\n    if len(compact_scores) == 0 or len(fragmented_scores) == 0:\n        return float(\"nan\")\n\n    return float(np.mean(compact_scores) - np.mean(fragmented_scores))",
  "rationale": "The metric measures the structural compactness effect on adherence to the leading cue, while coding choices relative to that cue so that A/B mirrors cancel response-side bias. Conflict-Density Adaptive Serial\u2013Parallel Control places every display in the same sparse serial regime and therefore predicts a value near zero, potentially slightly negative because fragmented displays contain an additional cue supporting the leading-cue option. Relational Chunk\u2013Anchor Arbitration predicts a large positive value: compact displays preserve the singleton anchor, whereas fragmented displays recruit majority comparison and reverse choices away from that anchor. Averaging 24 compact and 72 fragmented observations per subject makes the contrast estimable for each subject with relatively low sampling variance."
}
```

## Usage

```json
{
  "input_tokens": 7011,
  "output_tokens": 1460
}
```
