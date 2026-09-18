# feedback_iter_08

## System Prompt

You are a renowned cognitive scientist critiquing a freshly proposed candidate theory and model in the Decision Making (Binary Features) domain.

The candidate has been simulated on every previously run experiment. For each experiment you are shown the design, the metric, the value the metric takes on real data, and the value it takes on the candidate's simulated data.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the feedback is to SURFACE theories that are EXPERIMENT-INVARIANT: that is,theories that explain data across multiple experiments. 
Your task is to determine whether the candidate captures the human/real behavior well enough across these experiments. Return a verdict:
  * "continue"   — the candidate is good enough; carry on.
  * "regenerate" — the candidate fails to capture the empirical pattern; the proposing agent must produce a new candidate, taking your rationale into account.

Justify the verdict with a concrete diagnosis (which experiments fail, in what direction, what mechanism is likely missing or miscalibrated).

## SCOPE OF YOUR CRITIQUE — STAY INSIDE THE ARBITER'S MECHANISM FAMILY
When an "## ARBITER RECOMMENDATION" block is present below, the proposer was explicitly instructed to implement the mechanism family the arbiter prescribed. Your job is to grade FIT QUALITY *within that prescribed family*, not to relitigate which family should be used — that is the arbiter's call, made one level above this loop.

Concretely:
  * If the candidate misses the data, you may push for MINOR ADJUSTMENTS that keep the prescribed mechanism intact: tightening / widening parameter ranges, adding a temperature, swapping a normalization scheme, fixing a softmax / distance metric, re-balancing attention weights, fixing a learning-rate sign, correcting a bug in the gating or recurrence, etc.
  * You MUST NOT recommend switching to a different mechanism family. Such a switch is the arbiter's prerogative; recommending it here will mislead the proposer into oscillating between families across iterations.
  * Also grade FAITHFULNESS to the recommendation explicitly: if the candidate has clearly drifted into a different family than the one prescribed, say so in the rationale and ask for a return to the prescribed family — again, with minor adjustments, not a re-design.

## ACCEPT GATE — HOW THE LOOP DECIDES WHAT TO BUILD ON NEXT
This propose-loop has a programmatic accept gate. After every iteration the candidate's `aggregate_loss` is compared against the running-best loss (`accepted_loss`):
  * `loss < accepted_loss` → ACCEPTED. The candidate becomes the new running-best base; the next iteration's proposer will build on THIS candidate.
  * `loss >= accepted_loss` → REJECTED. The base is unchanged; the next iteration's proposer will build on the SAME `accepted` candidate again, with your new feedback on top. Rejected candidates are discarded — the loop guarantees the base never regresses, so you do NOT need to ask the proposer to "revert" anything; that already happens for free.

Two consequences for your verdict:
  * If the candidate you are grading was REJECTED by the gate, returning `"continue"` is silently downgraded to `"regenerate"` (returning a worse candidate would defeat the gate). Spend your rationale on a NEW direction the proposer should try on top of the unchanged accepted base, not on defending the rejected attempt.
  * If the candidate was ACCEPTED, you can return `"continue"` to stop the loop and ship this candidate, or `"regenerate"` to keep tuning further.

## LEARN FROM YOUR OWN PAST ADVICE
When a "## YOUR PRIOR CRITIQUES" block is present below, each prior iteration ends with an "Outcome of your advice" line that says whether the next candidate the proposer produced was ACCEPTED (your advice helped — its loss strictly beat the running best) or REJECTED (your advice didn't help — the proposer discarded the result and reset to the previous accepted base). This is the loop's ground-truth signal on whether *your own previous critique was good*. Use it explicitly:
  * If a previous piece of advice was ACCEPTED, it is OK to repeat / extend it. Reinforce in the same direction.
  * If a previous piece of advice was REJECTED, do NOT repeat the same recommendation; in your new rationale, briefly acknowledge that the previous push in that direction was rejected by the gate and try a different in-family knob (or a smaller step in the same direction) instead.
  * If you find yourself oscillating (e.g. iter 1 said "increase α", iter 2 said "decrease α", iter 3 about to say "increase α" again), STOP and recommend a value between the two flanking iterations instead.
  * The "## LOSS TRAJECTORY" block at the top of the user prompt summarises the same information at the loop level — consult it before issuing a new regenerate-with-direction recommendation.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## ARBITER RECOMMENDATION (mechanism family the proposer was told to implement)
The arbiter labelled this round's two theories in its recommendation as follows:
- THEORY 1 = `pi_9`
- THEORY 2 = `pi_10`
- The recommendation below acts on THEORY 1 (= `pi_9`).

Replace pi_9 with a causal-dependence or redundancy-suspicion theory. Subjects should ordinarily compress same-direction reports into an option-level recommendation and use a predominantly validity-ordered decision, but they should also infer whether numerous similar reports are likely to share a common source or reflect duplicated information. The inferred common-cause probability should rise nonlinearly with supporter count and with the conspicuousness or homogeneity of very high communicated validities. Once a bundle is judged redundant, its strongest cue is not merely harder to retrieve; the recommendation's evidential credibility is actively discounted, potentially enough for a lower-validity singleton to dominate. This differs from pi_9 because the penalty is not triggered by opponent count, run density, or contiguity, and it differs from pi_10 because the penalty reflects epistemic causal inference about correlated evidence rather than an intrinsic memory-binding capacity. The new theory should predict: near-zero opponent-count effects in Experiments 1 and 17; a very strong count penalty for six 94%-valid supporters in Experiment 2, targeting a following rate near 0.14; a substantial high-validity redundancy loss in Experiment 10; only weak count effects for 80%-82% bundles as in Experiment 13; and negligible effects of adjacency, spacing, strongest-source placement, and terminal clustering in Experiments 11, 12, 15, and 16. It should retain a localized nonlinear mapping from communicated validity to effective accessibility for Experiment 8 and a weak monotonic late-position accessibility loss confined to genuinely long displays, consistent with Experiment 9 but negligible around 24 features. Parameterization should separate baseline lexicographic reliability, common-cause inference, validity-gated redundancy discounting, long-display position loss, and response lapse; it should not reintroduce an opponent-count gate.


## CANDIDATE THEORY
Causal-Copy Discounted Lexicographic Choice with Local Priority Uncertainty. Decision makers compress all discriminating experts supporting the same option into an option-level recommendation and infer whether agreement reflects copied reports, shared evidence, or a common source. Common-cause suspicion rises nonlinearly with supporter count and is amplified by conspicuously high, homogeneous validities, actively discounting the recommendation's epistemic accessibility. Recommendations are ordinarily considered lexicographically by their strongest communicated validity. However, within a very narrow priority-gap neighborhood, validity ranks are not perfectly resolved: a small proportion of comparisons probabilistically interchange the two recommendation priorities. This local uncertainty permits an equal-validity or nearly equal-validity opposing recommendation to govern some choices without weakening lexicographic dominance for clearly separated validities. The causal discount remains independent of opponent count, adjacency, spacing, run structure, strongest-source placement, and terminal clustering. Separately, source-validity binding has a localized nonlinear uncertainty region, and isolated recommendations suffer a weak late-position accessibility loss only in genuinely long displays. Failure to access either recommendation leads to guessing, followed by a small response lapse.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            "Causal-Copy Discounted Lexicographic Choice expects state "
            f"shape (2, n_features); got {stim.shape}."
        )

    n_features = int(parameters["n_features"])
    if stim.shape[1] != n_features:
        raise ValueError(
            f"State has {stim.shape[1]} features but n_features={n_features}."
        )

    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities must have length {n_features}; got {validities.shape}."
        )

    baseline = float(parameters["lexicographic_reliability"])
    compression = float(parameters["validity_compression"])
    uncertainty_strength = float(parameters["binding_uncertainty_strength"])
    uncertainty_center = float(parameters["binding_uncertainty_center"])
    uncertainty_width = max(
        float(parameters["binding_uncertainty_width"]), 1e-9
    )
    count_threshold = float(parameters["common_cause_count_threshold"])
    count_slope = float(parameters["common_cause_count_slope"])
    homogeneity_scale = max(
        float(parameters["common_cause_homogeneity_scale"]), 1e-9
    )
    validity_threshold = float(parameters["redundancy_validity_threshold"])
    validity_width = max(
        float(parameters["redundancy_validity_width"]), 1e-9
    )
    discount_strength = float(parameters["redundancy_discount_strength"])
    position_loss = float(parameters["long_display_position_loss"])
    priority_mix = float(parameters["near_tie_priority_mixing"])
    lapse = float(parameters["response_lapse"])

    difference = stim[0] - stim[1]
    directions = np.sign(difference)
    if not np.any(directions != 0.0):
        return np.array([0.5, 0.5], dtype=np.float64)

    def sigmoid_scalar(x):
        x = float(x)
        if x >= 0.0:
            return 1.0 / (1.0 + np.exp(-x))
        ex = np.exp(x)
        return ex / (1.0 + ex)

    # Communicated validity is retained for lexicographic ordering, while a
    # compressed diagnosticity representation controls source-binding error.
    objective_strength = np.clip((validities - 0.5) / 0.5, 0.0, 1.0)
    subjective_strength = np.power(objective_strength, compression)
    uncertainty_profile = np.exp(
        -0.5 * np.square(
            (subjective_strength - uncertainty_center) / uncertainty_width
        )
    )

    units = []
    for direction in (-1, 1):
        members = np.flatnonzero(directions == float(direction)).astype(int)
        if members.size == 0:
            continue

        member_validities = validities[members]
        best_local = int(np.argmax(member_validities))
        representative = int(members[best_local])
        priority = float(validities[representative])
        count = int(members.size)

        # Infer whether agreement reflects a common source. The normalized
        # count signal is exactly zero for a singleton and approaches one only
        # after the nonlinear count threshold. It uses this bundle alone.
        count_at_one = sigmoid_scalar(
            count_slope * (1.0 - count_threshold)
        )
        count_raw = sigmoid_scalar(
            count_slope * (float(count) - count_threshold)
        )
        count_signal = (count_raw - count_at_one) / max(
            1.0 - count_at_one, 1e-9
        )
        count_signal = float(np.clip(count_signal, 0.0, 1.0))

        # Equal or nearly equal communicated validities are treated as a clue
        # that reports may derive from the same underlying analysis. Physical
        # positions and contiguity do not enter this inference.
        if count <= 1:
            homogeneity = 1.0
        else:
            spread = float(np.std(member_validities))
            homogeneity = float(np.exp(
                -np.square(spread / homogeneity_scale)
            ))

        # A conspicuously strong member can make the provenance of the whole
        # compressed bundle salient. This avoids requiring every supporter to
        # clear the high-validity threshold while retaining validity gating.
        conspicuous_high_validity = sigmoid_scalar(
            (priority - validity_threshold) / validity_width
        )
        common_cause_probability = float(np.clip(
            count_signal * homogeneity * conspicuous_high_validity,
            0.0,
            1.0,
        ))

        # Epistemic discounting is additionally validity-gated. Thus large
        # homogeneous 94% bundles can be strongly discredited, whereas equal
        # 80%-82% bundles acquire little cost even at comparable counts.
        validity_gate = sigmoid_scalar(
            (priority - validity_threshold) / validity_width
        )
        redundancy_cost = (
            discount_strength
            * common_cause_probability
            * validity_gate
        )

        retrieval_logit = (
            baseline
            - uncertainty_strength * uncertainty_profile[representative]
            - redundancy_cost
        )

        # Only isolated recommendations receive a display-position loss. The
        # quadratic display gate is zero through 24 features and reaches full
        # strength at 32, preventing ordinary layouts from producing effects.
        if count == 1 and n_features > 1:
            display_gate = float(np.clip(
                (float(n_features) - 24.0) / 8.0, 0.0, 1.0
            )) ** 2
            relative_position = representative / float(n_features - 1)
            retrieval_logit -= (
                position_loss * display_gate * relative_position
            )

        retrieval = sigmoid_scalar(retrieval_logit)
        units.append({
            "direction": int(direction),
            "priority": priority,
            "representative": representative,
            "retrieval": float(np.clip(retrieval, 1e-9, 1.0 - 1e-9)),
        })

    # Search remains validity ordered after causal credibility determines
    # whether each compressed recommendation is accessible. Position supplies
    # the default exact-tie convention but confers no accessibility advantage.
    units.sort(key=lambda u: (-u["priority"], u["representative"]))

    def cascade_probability_a(ordered_units):
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
            return 0.5
        return float(intended_a / total)

    p_a_lexicographic = cascade_probability_a(units)

    # A small fraction of comparisons have uncertain priority only within a
    # narrow validity-gap neighborhood. The fixed 0.012 scale makes the soft
    # order effectively deterministic for clearly separated recommendations.
    # Both possible orders are marginalized, rather than sampled in predict.
    if len(units) == 2:
        priority_gap = max(
            float(units[0]["priority"] - units[1]["priority"]), 0.0
        )
        first_order_probability = sigmoid_scalar(priority_gap / 0.012)
        p_a_soft_order = (
            first_order_probability * cascade_probability_a(units)
            + (1.0 - first_order_probability)
            * cascade_probability_a([units[1], units[0]])
        )
        p_a_intended = (
            (1.0 - priority_mix) * p_a_lexicographic
            + priority_mix * p_a_soft_order
        )
    else:
        p_a_intended = p_a_lexicographic

    p_a = (1.0 - lapse) * p_a_intended + lapse * 0.5
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

`parameters`:
- n_features: n_features
- validities: validities
- lexicographic_reliability: [3.0, 3.5]
- validity_compression: [0.85, 1.15]
- binding_uncertainty_strength: [2.17, 2.30]
- binding_uncertainty_center: [0.59, 0.67]
- binding_uncertainty_width: [0.11, 0.16]
- common_cause_count_threshold: [4.0, 4.4]
- common_cause_count_slope: [2.0, 2.7]
- common_cause_homogeneity_scale: [0.028, 0.038]
- redundancy_validity_threshold: [0.895, 0.91]
- redundancy_validity_width: [0.016, 0.024]
- redundancy_discount_strength: [7.2, 8.0]
- long_display_position_loss: [0.0, 0.15]
- near_tie_priority_mixing: [0.12, 0.18]
- response_lapse: [0.1, 0.16]

`rationale`:
This is a one-line parameter-range edit to the accepted iteration-7 model. The productive near-tie priority mixer and every equation are retained unchanged. Only `binding_uncertainty_strength` moves from the accepted broad range [1.7, 2.2] to the tightly bounded intermediate range [2.17, 2.30]. Its midpoint lies between the accepted setting, which underproduced the Experiment 6 contrast, and the rejected [2.2, 2.7] setting, which overproduced it. The narrow range reduces between-subject parameter variation and aims to interpolate the Experiment 6 prediction toward -0.1275 without the excessive global depth responsible for the rejected iteration's spillover. The uncertainty remains localized by the unchanged center and width, while all common-cause, homogeneity, redundancy, position, lapse, and priority-order parameters stay fixed. No opponent-count, adjacency, run, placement, or clustering mechanism is introduced.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.0661 -> ACCEPTED
- iter 2: loss=0.0659 -> ACCEPTED
- iter 3: loss=0.0439 -> ACCEPTED
- iter 4: loss=0.0464 -> REJECTED
- iter 5: loss=0.0446 -> REJECTED
- iter 6: loss=0.0471 -> REJECTED
- iter 7: loss=0.0435 -> ACCEPTED
- iter 8: loss=0.0577 -> REJECTED
- iter 9 (current candidate you are grading): loss=0.0451 -> REJECTED
Running-best (last accepted) base: iter 7 at loss=0.0435.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
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
**Candidate trajectory (this loop):**
  - iter 1: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 2: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 3: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 4: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 5: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 6: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 7: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 8: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 9 (current): 1.0000 (var=0.0000) (Δ vs real +0.0000)
**Other theories' values on this metric (for reference):**
- pi_1: 1.0000 (var=0.0000)
- pi_2: 0.0000 (var=0.0196)
- pi_3: 1.0000 (var=0.0000)
- pi_4: 1.0000 (var=0.0000)
- pi_5: 1.0000 (var=0.0000)
- pi_6: 1.0000 (var=0.0384)
- pi_7: 1.0000 (var=0.0000)
- pi_8: 1.0000 (var=0.0000)
- pi_9: 1.0000 (var=0.0000)
- pi_10: 1.0000 (var=0.0000)

### Experiment 2
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
**Candidate trajectory (this loop):**
  - iter 1: 0.0000 (var=0.0900) (Δ vs real +0.0000)
  - iter 2: 0.0000 (var=0.1056) (Δ vs real +0.0000)
  - iter 3: 0.0000 (var=0.0564) (Δ vs real +0.0000)
  - iter 4: 0.0000 (var=0.0384) (Δ vs real +0.0000)
  - iter 5: 0.0000 (var=0.0384) (Δ vs real +0.0000)
  - iter 6: 0.0000 (var=0.0564) (Δ vs real +0.0000)
  - iter 7: 0.0000 (var=0.0196) (Δ vs real +0.0000)
  - iter 8: 0.0000 (var=0.0736) (Δ vs real +0.0000)
  - iter 9 (current): 0.0000 (var=0.0564) (Δ vs real +0.0000)
**Other theories' values on this metric (for reference):**
- pi_2: 1.0000 (var=0.1204)
- pi_1: 0.0000 (var=0.0736)
- pi_3: 0.0000 (var=0.0900)
- pi_4: 0.0000 (var=0.1056)
- pi_5: 0.0000 (var=0.0564)
- pi_6: 0.0000 (var=0.0564)
- pi_7: 0.0000 (var=0.0196)
- pi_8: 0.0000 (var=0.0000)
- pi_9: 0.0000 (var=0.0384)
- pi_10: 0.0000 (var=0.0000)

### Experiment 3
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
**Candidate trajectory (this loop):**
  - iter 1: 0.0000 (var=0.2100) (Δ vs real +0.0000)
  - iter 2: 0.0000 (var=0.1716) (Δ vs real +0.0000)
  - iter 3: 0.0000 (var=0.1476) (Δ vs real +0.0000)
  - iter 4: 0.0000 (var=0.1924) (Δ vs real +0.0000)
  - iter 5: 0.0000 (var=0.1476) (Δ vs real +0.0000)
  - iter 6: 0.0000 (var=0.1824) (Δ vs real +0.0000)
  - iter 7: 0.0000 (var=0.1204) (Δ vs real +0.0000)
  - iter 8: 0.0000 (var=0.0736) (Δ vs real +0.0000)
  - iter 9 (current): 0.0000 (var=0.1600) (Δ vs real +0.0000)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0000 (var=0.2400)
- pi_3: 1.0000 (var=0.1056)
- pi_2: 0.0000 (var=0.0000)
- pi_4: 0.0000 (var=0.0564)
- pi_5: 0.0000 (var=0.1204)
- pi_6: 0.0000 (var=0.0000)
- pi_7: 0.0000 (var=0.0384)
- pi_8: 0.0000 (var=0.0384)
- pi_9: 0.0000 (var=0.0196)
- pi_10: 0.0000 (var=0.0564)

### Experiment 4
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
**Candidate trajectory (this loop):**
  - iter 1: 0.0993 (var=0.0012) (Δ vs real -0.0628)
  - iter 2: 0.0895 (var=0.0014) (Δ vs real -0.0725)
  - iter 3: 0.0983 (var=0.0014) (Δ vs real -0.0638)
  - iter 4: 0.0985 (var=0.0010) (Δ vs real -0.0635)
  - iter 5: 0.0968 (var=0.0009) (Δ vs real -0.0653)
  - iter 6: 0.1000 (var=0.0009) (Δ vs real -0.0620)
  - iter 7: 0.1490 (var=0.0023) (Δ vs real -0.0130)
  - iter 8: 0.1568 (var=0.0019) (Δ vs real -0.0053)
  - iter 9 (current): 0.1600 (var=0.0011) (Δ vs real -0.0020)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5913 (var=0.0137)
- pi_1: 0.1670 (var=0.0108)
- pi_2: 0.8492 (var=0.0103)
- pi_4: 0.1683 (var=0.0038)
- pi_5: 0.1505 (var=0.0020)
- pi_6: 0.1693 (var=0.0024)
- pi_7: 0.1578 (var=0.0020)
- pi_8: 0.1487 (var=0.0015)
- pi_9: 0.1457 (var=0.0026)
- pi_10: 0.1348 (var=0.0021)

### Experiment 5
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
**Candidate trajectory (this loop):**
  - iter 1: 0.1094 (var=0.0041) (Δ vs real -0.0519)
  - iter 2: 0.1212 (var=0.0030) (Δ vs real -0.0400)
  - iter 3: 0.1325 (var=0.0046) (Δ vs real -0.0287)
  - iter 4: 0.1194 (var=0.0034) (Δ vs real -0.0419)
  - iter 5: 0.1350 (var=0.0038) (Δ vs real -0.0262)
  - iter 6: 0.1363 (var=0.0030) (Δ vs real -0.0250)
  - iter 7: 0.1456 (var=0.0037) (Δ vs real -0.0156)
  - iter 8: 0.1388 (var=0.0033) (Δ vs real -0.0225)
  - iter 9 (current): 0.1669 (var=0.0054) (Δ vs real +0.0056)
**Other theories' values on this metric (for reference):**
- pi_1: 0.1450 (var=0.0101)
- pi_4: 0.8588 (var=0.0039)
- pi_2: 0.8450 (var=0.0101)
- pi_3: 0.0369 (var=0.0012)
- pi_5: 0.1588 (var=0.0035)
- pi_6: 0.1850 (var=0.0060)
- pi_7: 0.1775 (var=0.0051)
- pi_8: 0.1844 (var=0.0040)
- pi_9: 0.1688 (var=0.0073)
- pi_10: 0.1531 (var=0.0047)

### Experiment 6
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
**Candidate trajectory (this loop):**
  - iter 1: -0.0625 (var=0.0123) (Δ vs real +0.0650)
  - iter 2: -0.1138 (var=0.0187) (Δ vs real +0.0137)
  - iter 3: -0.1363 (var=0.0150) (Δ vs real -0.0088)
  - iter 4: -0.1450 (var=0.0193) (Δ vs real -0.0175)
  - iter 5: -0.1212 (var=0.0151) (Δ vs real +0.0063)
  - iter 6: -0.1412 (var=0.0153) (Δ vs real -0.0137)
  - iter 7: -0.0537 (var=0.0188) (Δ vs real +0.0738)
  - iter 8: -0.1825 (var=0.0262) (Δ vs real -0.0550)
  - iter 9 (current): -0.1425 (var=0.0170) (Δ vs real -0.0150)
**Other theories' values on this metric (for reference):**
- pi_4: 0.7075 (var=0.0151)
- pi_1: 0.0113 (var=0.0129)
- pi_2: -0.0112 (var=0.0229)
- pi_3: 0.1400 (var=0.0120)
- pi_5: -0.1363 (var=0.0200)
- pi_6: -0.0913 (var=0.0324)
- pi_7: -0.1250 (var=0.0233)
- pi_8: -0.1362 (var=0.0170)
- pi_9: -0.0962 (var=0.0183)
- pi_10: -0.0900 (var=0.0174)

### Experiment 7
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
**Candidate trajectory (this loop):**
  - iter 1: 0.0286 (var=0.0019) (Δ vs real +0.0829)
  - iter 2: -0.0057 (var=0.0018) (Δ vs real +0.0487)
  - iter 3: -0.0065 (var=0.0015) (Δ vs real +0.0478)
  - iter 4: 0.0198 (var=0.0023) (Δ vs real +0.0741)
  - iter 5: -0.0096 (var=0.0013) (Δ vs real +0.0447)
  - iter 6: -0.0140 (var=0.0017) (Δ vs real +0.0404)
  - iter 7: -0.0085 (var=0.0020) (Δ vs real +0.0458)
  - iter 8: 0.0132 (var=0.0025) (Δ vs real +0.0675)
  - iter 9 (current): 0.0038 (var=0.0025) (Δ vs real +0.0582)
**Other theories' values on this metric (for reference):**
- pi_1: -0.0015 (var=0.0020)
- pi_5: 0.2664 (var=0.0022)
- pi_2: -0.0100 (var=0.0017)
- pi_3: 0.0028 (var=0.0006)
- pi_4: 0.2945 (var=0.0059)
- pi_6: -0.0510 (var=0.0058)
- pi_7: -0.0026 (var=0.0021)
- pi_8: 0.0028 (var=0.0019)
- pi_9: -0.0071 (var=0.0015)
- pi_10: -0.0234 (var=0.0013)

### Experiment 8
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
**Candidate trajectory (this loop):**
  - iter 1: -0.0156 (var=0.0069) (Δ vs real +0.2519)
  - iter 2: -0.6375 (var=0.0109) (Δ vs real -0.3700)
  - iter 3: -0.2562 (var=0.0146) (Δ vs real +0.0113)
  - iter 4: -0.2794 (var=0.0180) (Δ vs real -0.0119)
  - iter 5: -0.2650 (var=0.0155) (Δ vs real +0.0025)
  - iter 6: -0.2706 (var=0.0156) (Δ vs real -0.0031)
  - iter 7: -0.2600 (var=0.0195) (Δ vs real +0.0075)
  - iter 8: -0.2812 (var=0.0250) (Δ vs real -0.0137)
  - iter 9 (current): -0.2975 (var=0.0210) (Δ vs real -0.0300)
**Other theories' values on this metric (for reference):**
- pi_5: 0.3306 (var=0.0175)
- pi_1: 0.0169 (var=0.0052)
- pi_2: 0.5481 (var=0.0232)
- pi_3: -0.0113 (var=0.0033)
- pi_4: 0.0087 (var=0.0075)
- pi_6: -0.2762 (var=0.0511)
- pi_7: -0.2888 (var=0.0210)
- pi_8: -0.2625 (var=0.0123)
- pi_9: -0.2544 (var=0.0160)
- pi_10: -0.2950 (var=0.0113)

### Experiment 9
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
**Candidate trajectory (this loop):**
  - iter 1: -0.0037 (var=0.0085) (Δ vs real +0.0019)
  - iter 2: -0.0178 (var=0.0064) (Δ vs real -0.0122)
  - iter 3: -0.0006 (var=0.0121) (Δ vs real +0.0050)
  - iter 4: -0.0016 (var=0.0089) (Δ vs real +0.0041)
  - iter 5: 0.0141 (var=0.0102) (Δ vs real +0.0197)
  - iter 6: -0.0150 (var=0.0111) (Δ vs real -0.0094)
  - iter 7: 0.0219 (var=0.0100) (Δ vs real +0.0275)
  - iter 8: 0.0269 (var=0.0149) (Δ vs real +0.0325)
  - iter 9 (current): 0.0162 (var=0.0086) (Δ vs real +0.0219)
**Other theories' values on this metric (for reference):**
- pi_1: -0.0003 (var=0.0059)
- pi_6: 0.3762 (var=0.0253)
- pi_2: 0.0038 (var=0.0049)
- pi_3: -0.0047 (var=0.0013)
- pi_4: 0.2303 (var=0.0907)
- pi_5: 0.0066 (var=0.0045)
- pi_7: 0.0225 (var=0.0115)
- pi_8: 0.0038 (var=0.0085)
- pi_9: -0.0047 (var=0.0080)
- pi_10: -0.0047 (var=0.0075)

### Experiment 10
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
**Candidate trajectory (this loop):**
  - iter 1: 0.0015 (var=0.0106) (Δ vs real +0.0060)
  - iter 2: -0.0092 (var=0.0126) (Δ vs real -0.0047)
  - iter 3: 0.0167 (var=0.0173) (Δ vs real +0.0212)
  - iter 4: -0.0185 (var=0.0150) (Δ vs real -0.0140)
  - iter 5: -0.0045 (var=0.0149) (Δ vs real -0.0000)
  - iter 6: 0.0022 (var=0.0125) (Δ vs real +0.0067)
  - iter 7: -0.0205 (var=0.0142) (Δ vs real -0.0160)
  - iter 8: -0.0282 (var=0.0184) (Δ vs real -0.0237)
  - iter 9 (current): -0.0015 (var=0.0192) (Δ vs real +0.0030)
**Other theories' values on this metric (for reference):**
- pi_6: 0.3420 (var=0.0223)
- pi_1: -0.0100 (var=0.0083)
- pi_2: 0.0002 (var=0.0078)
- pi_3: 0.0035 (var=0.0028)
- pi_4: -0.4612 (var=0.0902)
- pi_5: -0.4055 (var=0.0218)
- pi_7: -0.0140 (var=0.0142)
- pi_8: 0.0237 (var=0.0050)
- pi_9: -0.0120 (var=0.0102)
- pi_10: 0.0035 (var=0.0146)

### Experiment 11
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
**Candidate trajectory (this loop):**
  - iter 1: 0.0138 (var=0.0061) (Δ vs real +0.0363)
  - iter 2: 0.0075 (var=0.0121) (Δ vs real +0.0300)
  - iter 3: 0.0063 (var=0.0089) (Δ vs real +0.0288)
  - iter 4: -0.0019 (var=0.0104) (Δ vs real +0.0206)
  - iter 5: -0.0056 (var=0.0167) (Δ vs real +0.0169)
  - iter 6: 0.0112 (var=0.0082) (Δ vs real +0.0337)
  - iter 7: -0.0013 (var=0.0071) (Δ vs real +0.0212)
  - iter 8: 0.0269 (var=0.0107) (Δ vs real +0.0494)
  - iter 9 (current): 0.0306 (var=0.0113) (Δ vs real +0.0531)
**Other theories' values on this metric (for reference):**
- pi_7: -0.4881 (var=0.0165)
- pi_6: 0.1419 (var=0.0138)
- pi_1: -0.0069 (var=0.0079)
- pi_2: 0.0138 (var=0.0111)
- pi_3: -0.0031 (var=0.0033)
- pi_4: 0.0075 (var=0.0088)
- pi_5: 0.0738 (var=0.0093)
- pi_8: -0.0156 (var=0.0119)
- pi_9: -0.0088 (var=0.0093)
- pi_10: -0.0225 (var=0.0113)

### Experiment 12
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
**Candidate trajectory (this loop):**
  - iter 1: -0.0079 (var=0.0103) (Δ vs real -0.0229)
  - iter 2: -0.0067 (var=0.0151) (Δ vs real -0.0217)
  - iter 3: 0.0133 (var=0.0127) (Δ vs real -0.0017)
  - iter 4: 0.0017 (var=0.0174) (Δ vs real -0.0133)
  - iter 5: 0.0404 (var=0.0122) (Δ vs real +0.0254)
  - iter 6: 0.0096 (var=0.0126) (Δ vs real -0.0054)
  - iter 7: 0.0033 (var=0.0213) (Δ vs real -0.0117)
  - iter 8: -0.0021 (var=0.0179) (Δ vs real -0.0171)
  - iter 9 (current): -0.0067 (var=0.0133) (Δ vs real -0.0217)
**Other theories' values on this metric (for reference):**
- pi_6: 0.1892 (var=0.0198)
- pi_7: -0.0079 (var=0.0203)
- pi_1: 0.0088 (var=0.0055)
- pi_2: -0.0187 (var=0.0139)
- pi_3: -0.0025 (var=0.0024)
- pi_4: -0.0171 (var=0.0249)
- pi_5: -0.2475 (var=0.0268)
- pi_8: -0.0463 (var=0.0163)
- pi_9: 0.0171 (var=0.0149)
- pi_10: 0.0217 (var=0.0137)

### Experiment 13
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
**Candidate trajectory (this loop):**
  - iter 1: -0.0034 (var=0.0056) (Δ vs real -0.0004)
  - iter 2: -0.0323 (var=0.0126) (Δ vs real -0.0293)
  - iter 3: 0.0089 (var=0.0105) (Δ vs real +0.0119)
  - iter 4: -0.0126 (var=0.0091) (Δ vs real -0.0096)
  - iter 5: -0.0039 (var=0.0096) (Δ vs real -0.0009)
  - iter 6: 0.0203 (var=0.0063) (Δ vs real +0.0233)
  - iter 7: -0.0200 (var=0.0080) (Δ vs real -0.0170)
  - iter 8: 0.0111 (var=0.0109) (Δ vs real +0.0141)
  - iter 9 (current): -0.0048 (var=0.0085) (Δ vs real -0.0018)
**Other theories' values on this metric (for reference):**
- pi_7: -0.0313 (var=0.0119)
- pi_8: 0.1210 (var=0.0087)
- pi_1: 0.0009 (var=0.0050)
- pi_2: 0.0097 (var=0.0054)
- pi_3: 0.0143 (var=0.0019)
- pi_4: -0.1334 (var=0.0298)
- pi_5: -0.0434 (var=0.0066)
- pi_6: 0.0202 (var=0.0136)
- pi_9: 0.0281 (var=0.0061)
- pi_10: 0.0273 (var=0.0083)

### Experiment 14
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
**Candidate trajectory (this loop):**
  - iter 1: -0.0033 (var=0.0049) (Δ vs real +0.0150)
  - iter 2: 0.0042 (var=0.0081) (Δ vs real +0.0225)
  - iter 3: -0.0150 (var=0.0069) (Δ vs real +0.0033)
  - iter 4: -0.0042 (var=0.0061) (Δ vs real +0.0142)
  - iter 5: 0.0192 (var=0.0075) (Δ vs real +0.0375)
  - iter 6: -0.0242 (var=0.0083) (Δ vs real -0.0058)
  - iter 7: -0.0012 (var=0.0074) (Δ vs real +0.0171)
  - iter 8: 0.0129 (var=0.0094) (Δ vs real +0.0313)
  - iter 9 (current): 0.0004 (var=0.0107) (Δ vs real +0.0187)
**Other theories' values on this metric (for reference):**
- pi_8: 0.1392 (var=0.0118)
- pi_7: -0.1175 (var=0.0113)
- pi_1: -0.0021 (var=0.0051)
- pi_2: -0.0067 (var=0.0050)
- pi_3: -0.0150 (var=0.0011)
- pi_4: -0.2296 (var=0.0095)
- pi_5: -0.0083 (var=0.0034)
- pi_6: -0.2642 (var=0.0145)
- pi_9: 0.0267 (var=0.0071)
- pi_10: 0.0025 (var=0.0048)

### Experiment 15
**Design**
  A=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
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
```

**Observed (real) value:** -0.0205 (var=0.0094)
**Candidate trajectory (this loop):**
  - iter 1: -0.0152 (var=0.0060) (Δ vs real +0.0052)
  - iter 2: 0.0078 (var=0.0168) (Δ vs real +0.0282)
  - iter 3: 0.0038 (var=0.0100) (Δ vs real +0.0243)
  - iter 4: 0.0008 (var=0.0092) (Δ vs real +0.0213)
  - iter 5: -0.0085 (var=0.0100) (Δ vs real +0.0120)
  - iter 6: 0.0158 (var=0.0113) (Δ vs real +0.0363)
  - iter 7: -0.0255 (var=0.0144) (Δ vs real -0.0050)
  - iter 8: 0.0172 (var=0.0094) (Δ vs real +0.0377)
  - iter 9 (current): -0.0082 (var=0.0080) (Δ vs real +0.0123)
**Other theories' values on this metric (for reference):**
- pi_9: 0.5137 (var=0.0192)
- pi_8: -0.0035 (var=0.0146)
- pi_1: 0.0105 (var=0.0095)
- pi_2: 0.0730 (var=0.0107)
- pi_3: -0.0035 (var=0.0022)
- pi_4: 0.0067 (var=0.0063)
- pi_5: 0.0035 (var=0.0044)
- pi_6: 0.0530 (var=0.0194)
- pi_7: -0.0058 (var=0.0258)
- pi_10: 0.0340 (var=0.0153)

### Experiment 16
**Design**
  A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
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
```

**Observed (real) value:** 0.0550 (var=0.0207)
**Candidate trajectory (this loop):**
  - iter 1: -0.0038 (var=0.0215) (Δ vs real -0.0588)
  - iter 2: 0.0025 (var=0.0155) (Δ vs real -0.0525)
  - iter 3: -0.0112 (var=0.0234) (Δ vs real -0.0662)
  - iter 4: -0.0138 (var=0.0241) (Δ vs real -0.0688)
  - iter 5: -0.0150 (var=0.0174) (Δ vs real -0.0700)
  - iter 6: -0.0400 (var=0.0200) (Δ vs real -0.0950)
  - iter 7: -0.0225 (var=0.0331) (Δ vs real -0.0775)
  - iter 8: -0.0100 (var=0.0241) (Δ vs real -0.0650)
  - iter 9 (current): 0.0175 (var=0.0230) (Δ vs real -0.0375)
**Other theories' values on this metric (for reference):**
- pi_8: 0.3287 (var=0.0175)
- pi_9: 0.0088 (var=0.0220)
- pi_1: -0.0187 (var=0.0233)
- pi_2: -0.0088 (var=0.0164)
- pi_3: 0.0112 (var=0.0062)
- pi_4: 0.0200 (var=0.0137)
- pi_5: -0.0125 (var=0.0094)
- pi_6: 0.1738 (var=0.0304)
- pi_7: 0.0663 (var=0.0387)
- pi_10: 0.0337 (var=0.0218)

### Experiment 17
**Design**
  A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Measure the opponent-count effect specifically for six-source runs.
    # Positive values mean greater choice of the run-supported option when it
    # faces one opposing supporter rather than two.
    one_opponent = []
    two_opponents = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        d = np.sign(a - b)

        n_pos = int(np.sum(d > 0))
        n_neg = int(np.sum(d < 0))

        if n_pos == 6 and n_neg in (1, 2):
            run_supports_a = True
            opponent_count = n_neg
        elif n_neg == 6 and n_pos in (1, 2):
            run_supports_a = False
            opponent_count = n_pos
        else:
            continue

        response = int(row["response"])
        chose_run = float((response == 0) if run_supports_a else (response == 1))

        if opponent_count == 1:
            one_opponent.append(chose_run)
        else:
            two_opponents.append(chose_run)

    if len(one_opponent) == 0 or len(two_opponents) == 0:
        return float("nan")

    return float(np.mean(one_opponent) - np.mean(two_opponents))
```

**Observed (real) value:** -0.0200 (var=0.0140)
**Candidate trajectory (this loop):**
  - iter 1: -0.0100 (var=0.0226) (Δ vs real +0.0100)
  - iter 2: -0.0038 (var=0.0183) (Δ vs real +0.0162)
  - iter 3: 0.0138 (var=0.0121) (Δ vs real +0.0338)
  - iter 4: -0.0050 (var=0.0134) (Δ vs real +0.0150)
  - iter 5: 0.0025 (var=0.0162) (Δ vs real +0.0225)
  - iter 6: 0.0000 (var=0.0144) (Δ vs real +0.0200)
  - iter 7: -0.0175 (var=0.0192) (Δ vs real +0.0025)
  - iter 8: 0.0150 (var=0.0117) (Δ vs real +0.0350)
  - iter 9 (current): 0.0500 (var=0.0145) (Δ vs real +0.0700)
**Other theories' values on this metric (for reference):**
- pi_9: 0.3450 (var=0.0322)
- pi_10: 0.0287 (var=0.0211)
- pi_1: 0.0200 (var=0.0093)
- pi_2: -0.0075 (var=0.0129)
- pi_3: -0.0012 (var=0.0046)
- pi_4: -0.0088 (var=0.0152)
- pi_5: 0.0400 (var=0.0075)
- pi_6: 0.1050 (var=0.0219)
- pi_7: 0.0550 (var=0.0204)
- pi_8: -0.0125 (var=0.0361)

### Experiment 18
**Design**
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Experts 2, 5, 8, 11, 14, and 17 (zero-based indices below) are
    # the six 94%-valid sources used in the critical count-six condition.
    high94 = np.asarray([1, 4, 7, 10, 13, 16], dtype=int)
    following = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        d = a - b

        active94 = high94[d[high94] != 0]
        if active94.size != 6:
            continue

        directions = np.sign(d[active94])
        if not np.all(directions == directions[0]):
            continue

        # Require the intended opposing singleton at Expert 22.
        if d.size <= 21 or d[21] == 0 or np.sign(d[21]) == directions[0]:
            continue

        focal_response = 0 if directions[0] > 0 else 1
        following.append(float(int(row["response"]) == focal_response))

    if len(following) == 0:
        return float("nan")
    return float(np.mean(following))
```

**Observed (real) value:** 0.1375 (var=0.0106)
**Candidate trajectory (this loop):**
  - iter 1: 0.1875 (var=0.0114) (Δ vs real +0.0500)
  - iter 2: 0.1550 (var=0.0183) (Δ vs real +0.0175)
  - iter 3: 0.1675 (var=0.0090) (Δ vs real +0.0300)
  - iter 4: 0.1300 (var=0.0089) (Δ vs real -0.0075)
  - iter 5: 0.1150 (var=0.0072) (Δ vs real -0.0225)
  - iter 6: 0.1487 (var=0.0087) (Δ vs real +0.0112)
  - iter 7: 0.1537 (var=0.0096) (Δ vs real +0.0162)
  - iter 8: 0.1487 (var=0.0072) (Δ vs real +0.0112)
  - iter 9 (current): 0.1462 (var=0.0048) (Δ vs real +0.0087)
**Other theories' values on this metric (for reference):**
- pi_10: 0.4850 (var=0.0238)
- pi_9: 0.8525 (var=0.0070)
- pi_1: 0.8850 (var=0.0102)
- pi_2: 0.8838 (var=0.0111)
- pi_3: 0.9487 (var=0.0051)
- pi_4: 0.8363 (var=0.0087)
- pi_5: 0.8900 (var=0.0056)
- pi_6: 0.8650 (var=0.0083)
- pi_7: 0.3725 (var=0.0355)
- pi_8: 0.8425 (var=0.0066)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed causal-dependence family and successfully removes the unwanted structural effects: Experiments 9, 10, 13, 14, 15, and 17 are close to zero in the required direction, including the crucial absence of opponent-count effects in Experiment 17. It also produces the distinctive reversal for six homogeneous 94%-valid sources in Experiment 18, although following remains somewhat too high (0.1875 versus 0.1375). The major failure is Experiment 8: the observed high-redundancy penalty is -0.2675, whereas the candidate produces only -0.0156. Thus the current causal-suspicion computation works for six identical 94% reports but does not generalize to the broader high-validity bundle that should also be treated as redundant. Secondary misses include an attenuated validity-gap effect in Experiment 6 (-0.0625 versus -0.1275), excessive lexicographic dominance in Experiments 4 and 5 (minority/majority override rates 0.099 and 0.109 versus 0.162 and 0.161), and the wrong-signed long-display contrast in Experiment 7 (0.0286 versus -0.0543). Most invariance controls nevertheless fit well.
Rationale: Keep the causal-copy discounted lexicographic architecture, which is promising and was accepted, but recalibrate how high-validity bundles trigger common-cause inference. In Experiment 8, the product of an extremely narrow homogeneity kernel and a mean-validity sigmoid evidently suppresses suspicion almost completely. Broaden the homogeneity tolerance and/or replace the mean-validity bottleneck with a top-tail measure such as strongest validity, proportion of members above a high-validity cutoff, or a blend of strongest and mean validity. Retain the explicit validity gate so homogeneous 80%-82% bundles remain nearly unaffected in Experiments 11 and 13, and do not introduce spatial or opponent-count terms. After making that gate operative for Experiment 8, modestly strengthen the saturated six-source discount to move Experiment 18 from 0.188 toward 0.14 without merely increasing lapse. Separately, strengthen or better center the localized binding-uncertainty profile to recover the larger negative contrast in Experiment 6. The current late-position loss moves Experiment 7 farther from the data; shrink it toward a genuinely weak effect rather than reversing its sign, preserving the prescribed mechanism while avoiding a sizable positive contrast. Any baseline/lapse adjustment should be small because raising lapse enough to repair Experiments 4 and 5 would raise the floor in Experiment 18 and undermine its successful reversal.

**Outcome of your advice:** iter 1 candidate loss=0.0661 -> iter 2 candidate loss=0.0659 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed causal-dependence family and the accepted recalibration improved several intended signatures. Experiment 6 is now close to the negative validity-gap contrast (-0.1138 versus -0.1275), Experiment 7 moved toward the weak late-position loss (-0.0057 versus -0.0543), and the critical six-source reversal in Experiment 18 is close to target (0.1550 versus 0.1375). It also preserves near-zero opponent-count and layout effects in Experiments 1, 9, 10, 12, 14, 15, and 17. However, Experiment 8 has swung from severe under-discounting to severe over-discounting: the predicted redundancy contrast is -0.6375, compared with -0.2675 observed. Thus using the strongest bundle validity together with the broadened homogeneity kernel makes heterogeneous high-validity bundles almost maximally suspicious, rather than moderately discounted. Secondary misses remain in Experiment 4 (0.0895 versus 0.1620), Experiment 5 (0.1212 versus 0.1613), and Experiment 7, while the nominally null effects in Experiments 11 and 13 have drifted modestly negative or positive. The model captures the qualitative cross-experiment pattern but not yet the required magnitude of the central high-validity redundancy effect.
Rationale: Keep the accepted causal-copy discounted lexicographic architecture, but interpolate rather than reversing the prior accepted change. The previous recommendation successfully activated Experiment 8 and improved the overall loss, yet activating both a strongest-validity trigger and a broader homogeneity tolerance was too large a step. Selectively weaken suspicion for heterogeneous high-validity bundles while retaining nearly saturated suspicion for the perfectly homogeneous six-94% bundle in Experiment 18. The cleanest in-family adjustment is to tighten `common_cause_homogeneity_scale` to an intermediate range, or replace the pure strongest-validity signal with a calibrated blend of strongest and mean/top-tail validity. Do not return fully to the old mean-only bottleneck, which failed to produce any Experiment 8 effect. Target an Experiment 8 contrast around -0.27 while holding Experiment 18 near 0.14–0.16 and preserving the high validity threshold that protects the 80%–82% bundles. Retain the strengthened binding-uncertainty settings, since they materially repaired Experiment 6, and retain the reduced positive position coefficient, which improved Experiment 7. If further tuning is available after fixing Experiment 8, use a small baseline reliability or response-temperature adjustment—not a larger lapse—to raise the override rates in Experiments 4 and 5; increasing lapse risks lifting the Experiment 18 floor. No opponent-count or spatial-structure gate should be added.

**Outcome of your advice:** iter 2 candidate loss=0.0659 -> iter 3 candidate loss=0.0439 -> the gate marked it ACCEPTED.

### Iteration 3
Verdict: regenerate
Interpretation: The accepted homogeneity-scale interpolation was highly successful: aggregate loss fell from 0.0659 to 0.0439, and the central Experiment 8 redundancy contrast is now essentially correct (-0.2562 versus -0.2675) without destroying the homogeneous six-source reversal in Experiment 18 (0.1675 versus 0.1375). The candidate remains faithful to the prescribed causal-dependence family. It also preserves the required near-null opponent-count and spatial-layout signatures: Experiments 9, 10, 12, 13, 14, 15, and 17 are all small, with no opponent-count gate or adjacency mechanism. The nonlinear validity effect is well captured in Experiment 6 (-0.1363 versus -0.1275). Remaining systematic misses are the weak long-display position effect in Experiment 7, which is substantially underpredicted (-0.0065 versus -0.0543), and somewhat excessive following of the homogeneous 94% bundle in Experiment 18. The model also remains too lexicographically dominant in Experiment 4 (0.0983 versus 0.1620), though Experiment 5 has moved closer (0.1325 versus 0.1613). Deviations in Experiments 15–17 are modest relative to their between-subject variances and should not motivate structural spatial or opponent-count terms.
Rationale: Keep the current accepted architecture and the newly tightened homogeneity scale; the previous advice on that parameter was accepted and repaired the principal Experiment 8 failure. Further tuning should use orthogonal, narrowly targeted in-family knobs. First, raise and narrow `long_display_position_loss` above its current [0.0, 0.15] range so that the 32-feature singleton manipulation in Experiment 7 produces a weak but detectable negative gradient near -0.05. Because the display gate is exactly zero at 24 features and the loss applies only to singleton recommendations, this should leave Experiments 8, 10, and 12 largely protected. Second, make only a small increase in saturated redundancy discounting—or a slight decrease in the count threshold—so the perfectly homogeneous six-94% bundle in Experiment 18 moves from 0.168 toward 0.14. Check that this does not push Experiment 8 past its already excellent fit; changing homogeneity scale again is not warranted. Do not use a broad lapse increase or a global baseline reduction to repair Experiment 4, since either could raise the floor for the discounted bundle in Experiment 18. If Experiment 4 is addressed, use a small validity-localized accessibility/response-temperature adjustment affecting its weak-cue conflict regime rather than altering causal suspicion. Preserve the absence of opponent-count, contiguity, and placement terms.

**Outcome of your advice:** iter 3 candidate loss=0.0439 -> iter 4 candidate loss=0.0464 -> the gate marked it REJECTED.

### Iteration 4
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed causal-copy/dependence family and retains its main qualitative successes. It exactly captures the categorical signatures in Experiments 1–3, preserves negligible layout and opponent-count effects in Experiments 9–17, and closely fits both central redundancy results: Experiment 8 is -0.2794 versus -0.2675 and Experiment 18 is 0.1300 versus 0.1375. The nonlinear validity contrast in Experiment 6 also remains close (-0.1450 versus -0.1275). However, this iteration was rejected by the accept gate: loss rose from the accepted 0.0439 to 0.0464. The main attempted repair failed. Raising long-display position loss moved Experiment 7 away from the observed negative effect, from -0.0065 in the accepted base to +0.0198 rather than toward -0.0543. Several unrelated fits also deteriorated, notably Experiment 5 (0.1325 to 0.1194), Experiment 12 (0.0133 to 0.0017), and Experiment 14 (-0.0150 to -0.0042). The stronger redundancy discount did improve Experiment 18 and only slightly overshot Experiment 8, but that gain did not compensate for the position-related and sampling-wide regressions. Experiment 4 remains systematically too lexicographic (0.0985 versus 0.1620), while the nonzero Experiment 16 contrast is missed in the opposite direction; the latter should not motivate a prohibited spatial mechanism because the prescribed account predicts negligible structural effects.
Rationale: Build again from the unchanged accepted iteration-3 base, not from this rejected instantiation. My previous recommendation to make a large increase in `long_display_position_loss` was rejected by the gate and should not be repeated. Instead, audit the implementation and expected-probability behavior of the position term: under the stated search rule, discounting the later high-validity singleton in Experiment 7 should monotonically reduce its choice rate, so the observed sign reversal after increasing the coefficient suggests either excessive Monte Carlo instability or a mismatch between the intended target and the unit receiving the loss. Use deterministic expected probabilities or substantially more simulation while tuning this contrast. If a parameter change is still needed, use an intermediate coefficient between the accepted [0, 0.15] and rejected [0.8, 1.2] ranges, or reshape the same permitted long-display loss into a weak convex/late-only monotonic profile rather than simply increasing its magnitude. Preserve the gate at zero through 24 features and keep the term confined to singleton recommendations. The modestly stronger redundancy discount moved Experiment 18 in the correct direction, but reintroduce it only as a smaller one-knob step from the accepted base—approximately an interpolation between the iteration-3 and current ranges—and verify that Experiment 8 remains near -0.27. Do not change the successful homogeneity scale, add opponent-count or spatial gates, or use a broad lapse/baseline adjustment. The next iteration should isolate position-profile and saturated-discount changes rather than combining large shifts, because the accepted base already fits the principal experiment-invariant redundancy pattern well.

**Outcome of your advice:** iter 4 candidate loss=0.0464 -> iter 5 candidate loss=0.0446 -> the gate marked it REJECTED.

### Iteration 5
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed causal-dependence family and captures its central qualitative pattern: Experiments 1–3 have the correct categorical results; Experiment 8 closely matches the high-validity redundancy loss (-0.2650 versus -0.2675); Experiment 6 retains the localized nonlinear validity effect (-0.1212 versus -0.1275); and opponent-count, adjacency, placement, and clustering contrasts are generally small. However, this iteration was rejected by the accept gate (0.0446 versus the accepted 0.0439). The sole intended change—raising redundancy_discount_strength—was counterproductive overall. It pushed the six-source 94% condition in Experiment 18 past the target, from 0.1675 in the accepted base to 0.1150 versus 0.1375 observed, while the excellent Experiment 8 fit gained only marginally. The candidate also still underpredicts the long-display position effect in Experiment 7 (-0.0096 versus -0.0543) and is too lexicographically dominant in Experiment 4 (0.0968 versus 0.1620). Experiment 16 has the wrong-signed contrast, but its observed effect is modest relative to between-subject variance and should not motivate a forbidden spatial mechanism. Several large changes in nominally unaffected metrics, such as Experiments 12 and 14, despite a one-parameter edit suggest appreciable simulation or parameter-sampling noise.
Rationale: Build from the unchanged accepted iteration-3 base. Do not repeat the global increase in redundancy_discount_strength: both the present smaller increase and the previous stronger-discount attempt were rejected, and the current result overshoots Experiment 18. If the residual Experiment 18 error is addressed later, use a more selective in-family homogeneity nonlinearity that distinguishes perfectly identical 94% bundles from merely similar high-validity bundles—for example, a steep near-one homogeneity transform coupled with normalization that preserves the Experiment 8 prediction—rather than increasing the discount applied to all high-validity bundles. For the next iteration, preferably isolate the remaining Experiment 7 miss. The earlier large position-loss increase was rejected, so use only a genuinely intermediate, narrowly bounded coefficient or a weak late-only monotonic profile, while retaining the zero gate through 24 features and singleton-only application. First verify with deterministic expected probabilities or substantially increased simulation that increasing this loss monotonically lowers choice of the late target; the prior sign reversal and the broad movement of unaffected metrics indicate that noisy evaluation may be obscuring the parameter effect. Leave the successful homogeneity scale, binding-uncertainty profile, lapse, and absence of opponent-count or spatial-structure gates unchanged. Do not pursue Experiment 4 with a broad baseline or lapse adjustment, because that would endanger the well-fitted redundancy reversals.

**Outcome of your advice:** iter 5 candidate loss=0.0446 -> iter 6 candidate loss=0.0471 -> the gate marked it REJECTED.

### Iteration 6
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed causal-dependence family and retains its central successes: Experiment 8 closely matches the high-validity redundancy loss (-0.2706 versus -0.2675), Experiment 18 is near the six-source target (0.1487 versus 0.1375), Experiment 6 captures the nonlinear validity effect, and opponent-count/layout effects are generally negligible. However, the current candidate was rejected: aggregate loss rose to 0.0471 from the accepted iteration-3 loss of 0.0439. The sign-corrected position term moved Experiment 7 modestly toward the data (-0.0140 versus -0.0543), but not enough to offset deterioration elsewhere, especially Experiment 16 (-0.0400 versus 0.0550), Experiment 15 (0.0158 versus -0.0205), and Experiment 13 (0.0203 versus -0.0030). The persistent mechanistic miss is excessive deterministic lexicographic dominance in near-tie conflicts: Experiment 4 remains at 0.1000 versus 0.1620, and Experiment 5 remains somewhat low at 0.1363 versus 0.1613. The model's hard priority ordering, including deterministic position-based resolution of equal validity, leaves too little opportunity for the opposing option-level recommendation to win when priorities are equal or nearly equal.
Rationale: Build on the unchanged accepted iteration-3 base. Repeated attempts to repair Experiment 7 by changing the long-display position coefficient—including the present sign correction—have all been rejected by the gate, so do not tune that coefficient again in the next iteration. Instead, address Experiments 4 and 5 with a new, narrowly targeted in-family knob: replace hard deterministic ordering for equal or nearly equal option-level priorities with a low-temperature probabilistic priority comparison or small priority-gap softmax. Keep ordering effectively lexicographic outside a very narrow validity-gap region. This should raise the integrated/majority-side choice rate for equal-validity bundles in Experiment 4 and the 70% versus 72% near-tie in Experiment 5, while leaving the large validity gaps in Experiments 1–3 and 8 largely unchanged. It should also leave the causal-copy calculation itself intact; retain the accepted homogeneity scale, validity gate, count nonlinearity, binding-uncertainty profile, and redundancy discount so Experiments 6, 8, and 18 remain protected. Do not add opponent-count, adjacency, run, or clustering terms. Given the substantial movement of nominally unaffected control metrics across one-parameter iterations, evaluate this change using deterministic expected probabilities or substantially more simulation before comparing losses.

**Outcome of your advice:** iter 6 candidate loss=0.0471 -> iter 7 candidate loss=0.0435 -> the gate marked it ACCEPTED.

### Iteration 7
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed causal-dependence family and the newly introduced near-tie priority uncertainty was productive: the gate accepted it, aggregate loss improved from 0.0439 to 0.0435, Experiment 4 rose substantially from about 0.10 to 0.149 versus 0.162 observed, and Experiment 5 improved to 0.146 versus 0.161. The central experiment-invariant signatures remain strong: the high-validity redundancy contrast is accurate in Experiment 8 (-0.260 versus -0.268), the six homogeneous 94% bundle is strongly discounted in Experiment 18 (0.154 versus 0.138), and opponent-count and layout effects remain negligible in Experiments 15 and 17 and generally small in Experiments 9–14. The main cost of the accepted near-tie mechanism is Experiment 6: its nonlinear validity-gap contrast deteriorated from -0.136 in the previous accepted base to -0.054, compared with -0.128 observed. This is mechanistically interpretable: the priority mixer strongly randomizes the extremely small validity gaps in Experiment 6, reducing higher-cue following in the condition that anchors the negative contrast. Experiment 7 also continues to understate the long-display effect (-0.009 versus -0.054), but repeated position-coefficient interventions were rejected and should not be revisited. Experiment 16 is wrong-signed (-0.023 versus 0.055), although its magnitude is modest relative to the observed between-subject variance and the prescribed theory explicitly predicts negligible structural effects.
Rationale: Retain the accepted near-tie priority mechanism: my previous recommendation produced an accepted improvement and successfully repaired Experiments 4 and 5 without damaging the core causal-copy results. The next change should be orthogonal and narrowly restore Experiment 6. Modestly strengthen or sharpen the existing localized binding-uncertainty trough around the subjective strength corresponding to roughly 80%–82% validity—for example, slightly increase `binding_uncertainty_strength` or narrow/recenter its profile—so accessibility of the relevant high-validity singleton falls enough to recover a contrast near -0.12. This should have little effect on the 94%–95% bundles in Experiments 8 and 18, whose subjective strengths lie outside that localized region. Check Experiments 11–13 because they also use approximately 82% cues, but their within-experiment contrasts should remain near zero if the adjustment depends on validity rather than count or position. Do not remove the accepted priority mixer, globally raise lapse, alter the successful homogeneity/redundancy settings, or again tune `long_display_position_loss`; those position-directed attempts were repeatedly rejected. Preserve the absence of opponent-count, adjacency, run, and clustering terms.

**Outcome of your advice:** iter 7 candidate loss=0.0435 -> iter 8 candidate loss=0.0577 -> the gate marked it REJECTED.

### Iteration 8 (most recent)
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed causal-copy/redundancy-suspicion family, and its central signatures are still credible: Experiment 8 closely captures the high-validity redundancy loss (-0.2812 versus -0.2675), Experiment 18 remains near the target for six homogeneous 94% supporters (0.1487 versus 0.1375), and Experiments 1–3 retain the correct categorical outcomes. However, this iteration was rejected because loss increased sharply from 0.0435 to 0.0577. The stronger binding-uncertainty trough overcorrected Experiment 6: the accepted base underpredicted the negative contrast (-0.0537 versus -0.1275), whereas the current candidate overshoots it (-0.1825). It also sacrifices the accepted balance elsewhere: Experiment 5 falls from 0.1456 to 0.1388, and nominally negligible effects become appreciably wrong in Experiments 9, 11, 14, 15, and 17. In particular, Experiment 11 changes to +0.0269 versus -0.0225, Experiment 15 to +0.0172 versus -0.0205, and Experiment 17 to +0.0150 versus -0.0200. These broad changes from a one-range edit also indicate substantial simulation or parameter-sampling noise. Experiment 7 remains underfit, but repeated position-loss interventions have already been rejected and should not be revisited.
Rationale: Build on the unchanged accepted iteration-7 base and retain its productive near-tie priority mixer. My previous recommendation to raise binding_uncertainty_strength to [2.2, 2.7] was rejected, so do not repeat that full increase. Instead use a tightly bounded intermediate value between the accepted [1.7, 2.2] and rejected [2.2, 2.7] ranges, centered around roughly 2.2–2.3, to interpolate Experiment 6 between -0.054 and -0.183 toward the observed -0.128. If additional selectivity is needed, modestly narrow and center the existing uncertainty profile around the subjective strength corresponding to the 81.77% singleton, rather than increasing its global depth; this should reduce spillover into Experiments 5 and 9 while remaining a validity-localized adjustment. Keep the accepted common-cause threshold, homogeneity scale, redundancy discount, priority mixer, lapse, and position term unchanged. Evaluate the interpolation with deterministic expected probabilities, common random numbers, or substantially more simulation, because the large movements in nominally unaffected controls obscure the true parameter effect. Do not add opponent-count or spatial-structure terms, retune long-display position loss, or globally alter lapse or redundancy strength.

**Outcome of your advice:** iter 8 candidate loss=0.0577 -> CURRENT candidate loss=0.0451 -> the gate marked it REJECTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate remains faithful to the prescribed causal-copy/redundancy-suspicion family. Its principal signatures are strong: Experiments 1\u20133 have the correct categorical outcomes, Experiment 18 closely matches the severe discount for six homogeneous 94% supporters (0.1462 versus 0.1375), Experiment 8 retains a substantial high-validity redundancy loss (-0.2975 versus -0.2675), and the near-tie mixer gives excellent fits in Experiments 4 and 5. The current strength interpolation also brings Experiment 6 close to its observed nonlinear contrast (-0.1425 versus -0.1275). Nevertheless, this iteration was rejected because its loss, 0.0451, exceeds the accepted iteration-7 loss of 0.0435. The gain in Experiment 6 is offset by deterioration in nominal invariance controls, especially Experiment 11 (+0.0306 versus -0.0225) and Experiment 17 (+0.0500 versus -0.0200), along with smaller misses in Experiments 9, 12, 14, and 15. Experiment 8 also now modestly overstates redundancy loss. Experiment 7 remains underfit and Experiment 16 misses its positive contrast, but repeated position-loss interventions were rejected, and the arbiter specifically requires negligible layout effects, so neither should prompt a spatial mechanism.",
  "verdict": "regenerate",
  "rationale": "Build again from the unchanged accepted iteration-7 base. My previous recommendation to interpolate upward through binding_uncertainty_strength produced a closer Experiment 6 point estimate but was still rejected, so do not repeat another global depth increase. Instead, leave strength near its accepted range and improve localization through the existing binding profile: narrowly recenter the uncertainty trough around the subjective strength associated with the approximately 81.8% cue and reduce binding_uncertainty_width, rather than deepening the trough. This is the distinct in-family knob most likely to recover part of the Experiment 6 contrast while limiting spillover into other validity regimes. Keep the accepted near-tie priority mixer, common-cause threshold, homogeneity scale, redundancy discount, lapse, and position term unchanged. Do not tune long-display position loss again, add opponent-count sensitivity, or introduce adjacency/run/placement terms. Because Experiment 17 should be nearly invariant under the equations yet moved by 0.0675 relative to the accepted iteration, use deterministic expected-probability evaluation, common random numbers, or substantially more simulation before interpreting these control contrasts; otherwise sampling noise will continue to dominate small parameter comparisons. Target a smaller, selective Experiment 6 improvement rather than exact point matching, while preserving Experiments 4, 5, 8, 11, 17, and 18."
}
```

## Usage

```json
{
  "input_tokens": 53932,
  "output_tokens": 1977
}
```
