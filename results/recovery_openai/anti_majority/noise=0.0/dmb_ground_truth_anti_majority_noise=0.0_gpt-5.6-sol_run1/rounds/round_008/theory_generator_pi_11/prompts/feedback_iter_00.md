# feedback_iter_00

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
Causal-Copy Discounted Lexicographic Choice. Decision makers first compress all currently discriminating experts that recommend the same option into a single option-level recommendation. Each recommendation is indexed by its strongest communicated validity and recommendations are normally considered lexicographically. However, agreement among many experts is not automatically interpreted as independent corroboration. The observer infers a latent probability that a recommendation bundle arose from copied reports, shared evidence, or a common information source. This common-cause probability grows nonlinearly with the bundle's own supporter count, is amplified when communicated validities are conspicuously high, and is greatest when those validities are unusually homogeneous. Inferred dependence actively reduces the epistemic credibility and hence accessibility of the entire recommendation. A sufficiently homogeneous bundle of very reliable reports can consequently become less influential than an opposing singleton, despite the bundle containing the nominally strongest expert. This discount is intrinsic to the evidential provenance of the bundle: it does not depend on opponent count, adjacency, spacing, run structure, strongest-source placement, or terminal clustering. Separately, source-validity binding has a localized nonlinear uncertainty region, and isolated recommendations suffer a weak late-position accessibility loss only in genuinely long displays. Failure to access either recommendation leads to guessing, followed by a small response lapse.

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

        mean_validity = float(np.mean(member_validities))
        conspicuous_high_validity = sigmoid_scalar(
            (mean_validity - validity_threshold) / validity_width
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
    # whether each compressed recommendation is accessible. Position resolves
    # exact validity ties deterministically but confers no accessibility bonus.
    units.sort(key=lambda u: (-u["priority"], u["representative"]))

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

    # If all compressed recommendations are inaccessible, choose at random.
    intended_a += 0.5 * reach
    intended_b += 0.5 * reach
    total = intended_a + intended_b
    if not np.isfinite(total) or total <= 0.0:
        p_a_intended = 0.5
    else:
        p_a_intended = intended_a / total

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
- binding_uncertainty_strength: [1.0, 1.55]
- binding_uncertainty_center: [0.59, 0.67]
- binding_uncertainty_width: [0.11, 0.16]
- common_cause_count_threshold: [4.0, 4.4]
- common_cause_count_slope: [2.0, 2.7]
- common_cause_homogeneity_scale: [0.018, 0.03]
- redundancy_validity_threshold: [0.895, 0.91]
- redundancy_validity_width: [0.016, 0.024]
- redundancy_discount_strength: [6.2, 7.2]
- long_display_position_loss: [0.8, 1.35]
- response_lapse: [0.1, 0.16]

`rationale`:
The model replaces retrieval overload with an epistemic causal inference. Same-direction reports are always represented as one recommendation, so rearranging, spacing, clustering, or moving its strongest member does not alter the redundancy computation. Unlike pi_9, no term asks how many experts support the opponent; therefore adding a second opponent cannot directly trigger failure of an unchanged six-source recommendation. Unlike pi_10, the cost is not a generic capacity penalty attached to any large bundle. It arises specifically when three observable provenance cues coincide: numerous supporters, conspicuously high communicated validity, and unusually homogeneous validity labels. The steep count transition and high-validity gate make six identical 94% reports strongly suspicious. Their recommendation can have a retrieval logit driven well below zero, allowing the opposing singleton to dominate and bringing Experiment 18's following rate toward the observed 0.14 after lapse. In contrast, 80%-82% bundles receive almost no validity-gated causal discount, supporting weak count effects in Experiments 11 and 13. Heterogeneous high-validity runs also receive much less discount than six identical 94% sources, preventing a universal count penalty. Because causal suspicion uses neither source indices nor distances, the model predicts negligible adjacency, spacing, strongest-source-placement, and terminal-clustering contrasts. The localized binding-uncertainty profile preserves the inverse validity-gap behavior needed in Experiments 6 and 8 independently of causal discounting. Finally, the separate long-display gate introduces a modest monotonic loss for late isolated cues around 32 features while being exactly inactive through 24 features. Baseline lexicographic reliability, causal inference, validity-gated discounting, long-display position loss, and response lapse are therefore mechanistically and parametrically distinct.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1 (current candidate you are grading): loss=0.0661 -> ACCEPTED
Running-best (last accepted) base: iter 1 at loss=0.0661.

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
**Candidate (simulated) value:** 1.0000 (var=0.0000)
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
**Candidate (simulated) value:** 0.0000 (var=0.0900)
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
**Candidate (simulated) value:** 0.0000 (var=0.2100)
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
**Candidate (simulated) value:** 0.0993 (var=0.0012)
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
**Candidate (simulated) value:** 0.1094 (var=0.0041)
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
**Candidate (simulated) value:** -0.0625 (var=0.0123)
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
**Candidate (simulated) value:** 0.0286 (var=0.0019)
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
**Candidate (simulated) value:** -0.0156 (var=0.0069)
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
**Candidate (simulated) value:** -0.0037 (var=0.0085)
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
**Candidate (simulated) value:** 0.0015 (var=0.0106)
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
**Candidate (simulated) value:** 0.0138 (var=0.0061)
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
**Candidate (simulated) value:** -0.0079 (var=0.0103)
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
**Candidate (simulated) value:** -0.0034 (var=0.0056)
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
**Candidate (simulated) value:** -0.0033 (var=0.0049)
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
**Candidate (simulated) value:** -0.0152 (var=0.0060)
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
**Candidate (simulated) value:** -0.0038 (var=0.0215)
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
**Candidate (simulated) value:** -0.0100 (var=0.0226)
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
**Candidate (simulated) value:** 0.1875 (var=0.0114)
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

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate is faithful to the prescribed causal-dependence family and successfully removes the unwanted structural effects: Experiments 9, 10, 13, 14, 15, and 17 are close to zero in the required direction, including the crucial absence of opponent-count effects in Experiment 17. It also produces the distinctive reversal for six homogeneous 94%-valid sources in Experiment 18, although following remains somewhat too high (0.1875 versus 0.1375). The major failure is Experiment 8: the observed high-redundancy penalty is -0.2675, whereas the candidate produces only -0.0156. Thus the current causal-suspicion computation works for six identical 94% reports but does not generalize to the broader high-validity bundle that should also be treated as redundant. Secondary misses include an attenuated validity-gap effect in Experiment 6 (-0.0625 versus -0.1275), excessive lexicographic dominance in Experiments 4 and 5 (minority/majority override rates 0.099 and 0.109 versus 0.162 and 0.161), and the wrong-signed long-display contrast in Experiment 7 (0.0286 versus -0.0543). Most invariance controls nevertheless fit well.",
  "verdict": "regenerate",
  "rationale": "Keep the causal-copy discounted lexicographic architecture, which is promising and was accepted, but recalibrate how high-validity bundles trigger common-cause inference. In Experiment 8, the product of an extremely narrow homogeneity kernel and a mean-validity sigmoid evidently suppresses suspicion almost completely. Broaden the homogeneity tolerance and/or replace the mean-validity bottleneck with a top-tail measure such as strongest validity, proportion of members above a high-validity cutoff, or a blend of strongest and mean validity. Retain the explicit validity gate so homogeneous 80%-82% bundles remain nearly unaffected in Experiments 11 and 13, and do not introduce spatial or opponent-count terms. After making that gate operative for Experiment 8, modestly strengthen the saturated six-source discount to move Experiment 18 from 0.188 toward 0.14 without merely increasing lapse. Separately, strengthen or better center the localized binding-uncertainty profile to recover the larger negative contrast in Experiment 6. The current late-position loss moves Experiment 7 farther from the data; shrink it toward a genuinely weak effect rather than reversing its sign, preserving the prescribed mechanism while avoiding a sizable positive contrast. Any baseline/lapse adjustment should be small because raising lapse enough to repair Experiments 4 and 5 would raise the floor in Experiment 18 and undermine its successful reversal."
}
```

## Usage

```json
{
  "input_tokens": 44089,
  "output_tokens": 1609
}
```
