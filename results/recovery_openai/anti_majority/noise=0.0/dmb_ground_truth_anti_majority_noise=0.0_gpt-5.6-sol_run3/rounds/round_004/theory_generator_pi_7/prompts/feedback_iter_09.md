# feedback_iter_09

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
- THEORY 1 = `pi_6`
- THEORY 2 = `pi_5`
- The recommendation below acts on THEORY 1 (= `pi_6`).

Replace pi_6 with a new task-invariant theory of configuration-dependent selective integration that competes with ADCG without treating feedback-free exposure as sufficient evidence for learned cue dependence. The theory should combine: (1) communicated-validity weighting; (2) redundancy-discounted coalition accumulation; (3) static minority or dissent distinctiveness that can operate even when the highest-validity cue has allies, allowing singleton dissent to matter in Experiment 2; (4) a sharply nonmonotonic isolated-anchor mechanism in which one or two credible opponents can challenge the anchor but a larger concordant coalition can trigger redundancy-based anchor restoration, as required by Experiment 8; (5) bounded presentation-order influence, including tied-top anchoring, without a general primacy crossover; and (6) stable subject-level mixtures or random effects over validity sensitivity, order use, minority amplification, restoration thresholds, temperature, and lapse to reproduce the observed heterogeneity. History-dependent anti-concurrence should be absent without correctness feedback, or confined to a small optional component that activates only with strong independent evidence and cannot generate the spurious Experiment 9 contrast. The new model should separately predict compact isolated-anchor trials and singleton-dissent trials rather than fitting their pooled Experiment 2 metric through one mechanism. Quantitatively, it should target intermediate anchor following in Experiment 1, low pooled anchor following in Experiment 2, a near-zero diagnostic-learning contrast in Experiment 9, and the large negative one-versus-multiple-opponent contrast in Experiment 8. This preserves SCCI's useful challenge/restoration insight while removing its degenerative restrictions and provides a stronger, more falsifiable competitor to ADCG.


## CANDIDATE THEORY
Partitioned Dissent and Redundancy Restoration (PDRR) proposes that people construct a stable, instruction-based evidence representation and classify each choice configuration before integrating its cues. Communicated validities determine both relative cue strength and absolute anchor credibility, while cues supporting the same option accumulate sublinearly because concordant reports are treated as partly redundant. In ordinary embedded coalitions, individual cue weights are strongly compressed before accumulation, limiting the influence of any single non-anchor cue while preserving communicated-validity ordering. Opposition to a validity anchor is statically distinctive even when the anchor has allies, but balanced opposition receives a counter-anchor bonus only when the opposing cues are individually close in communicated validity to the anchor and the coalition is broad enough to support a coalition-level interpretation. Isolated-anchor opposition is processed separately. A single opponent receives especially strong challenge status, with stable subject types differing substantially in their sensitivity to this challenge, while exactly two opponents retain a joint-credibility mechanism with greater type-specific heterogeneity. These challenge and singleton-dissent mechanisms are additionally amplified when the locally strongest active cue has low absolute communicated validity, because being best among weak active cues does not create a compelling anchor. Coalitions of three or more trigger a sharp, nearly type-invariant redundancy-restoration branch. Embedded singleton dissent is strongly amplified only against very broad coalitions or weak absolute anchors; its generic effect against smaller coalitions is attenuated to avoid excessive profile variance. Presentation order affects choice primarily as a bounded tie breaker among currently discriminating cues with equal maximal communicated validity, avoiding a general primacy crossover. All mechanisms are history-invariant in this feedback-free task. Stable subject types and random effects over validity sensitivity, balanced dissent, compact challenge, redundancy, temperature, lapse, and absolute-anchor sensitivity generate persistent individual differences.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"PDRR expects state with shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.shape[0] != n_features:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {n_features}."
        )

    # Positive directions favor B and negative directions favor A.
    directions = np.sign(stim[1] - stim[0])
    active = np.flatnonzero(directions != 0)
    if active.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    validity_sensitivity = float(parameters["validity_sensitivity"])
    redundancy_exponent = float(parameters["redundancy_exponent"])
    redundancy_saturation = float(parameters["redundancy_saturation"])
    ordinary_validity_compression = float(parameters["ordinary_validity_compression"])
    minority_amplification = float(parameters["minority_amplification"])
    balanced_opposition_bonus = float(parameters["balanced_opposition_bonus"])
    singleton_bonus = float(parameters["singleton_bonus"])
    compact_challenge = float(parameters["compact_challenge"])
    compact_width = float(parameters["compact_width"])
    credibility_gap_scale = float(parameters["credibility_gap_scale"])
    absolute_anchor_threshold = float(parameters["absolute_anchor_threshold"])
    absolute_anchor_slope = float(parameters["absolute_anchor_slope"])
    weak_anchor_gain = float(parameters["weak_anchor_gain"])
    restoration_threshold = float(parameters["restoration_threshold"])
    restoration_slope = float(parameters["restoration_slope"])
    restoration_gap_threshold = float(parameters["restoration_gap_threshold"])
    restoration_gap_slope = float(parameters["restoration_gap_slope"])
    restoration_credibility_threshold = float(parameters["restoration_credibility_threshold"])
    restoration_credibility_slope = float(parameters["restoration_credibility_slope"])
    restoration_ceiling = float(parameters["restoration_ceiling"])
    tie_order_gate = float(parameters["tie_order_gate"])
    beta = float(parameters["beta"])
    lapse = float(parameters["lapse"])
    configuration_type = int(parameters["configuration_type"])

    # Correlated, stable subject types alter configuration sensitivity without
    # introducing trial-wise strategy switching or exposure-based learning.
    if configuration_type == 0:       # compact-challenge-sensitive
        challenge_type_gain = 2.40
        one_challenge_type_gain = 4.20
        singleton_type_gain = 1.50
        restoration_type_gain = 0.95
    elif configuration_type == 1:     # count-restoring
        challenge_type_gain = 0.50
        one_challenge_type_gain = 0.75
        singleton_type_gain = 0.90
        restoration_type_gain = 1.25
    else:                              # weakly configuration-sensitive
        challenge_type_gain = 0.15
        one_challenge_type_gain = 0.15
        singleton_type_gain = 0.55
        restoration_type_gain = 1.00

    def sigmoid(x):
        x = float(np.clip(x, -60.0, 60.0))
        return 1.0 / (1.0 + np.exp(-x))

    # Translate communicated accuracy into reliability evidence, retaining
    # absolute ordering but using the strongest active cue as a scale origin.
    v = np.clip(validities, 0.500001, 0.999999)
    reliability = np.log(v / (1.0 - v))
    active_reference = float(np.max(reliability[active]))
    cue_weight = np.exp(
        validity_sensitivity * (reliability - active_reference)
    )

    a_cues = np.flatnonzero(directions < 0)
    b_cues = np.flatnonzero(directions > 0)
    n_a = int(a_cues.size)
    n_b = int(b_cues.size)

    def coalition_support(indices, compress=False):
        n = int(indices.size)
        if n == 0:
            return 0.0
        weights = cue_weight[indices]
        if compress:
            # Bound cue-composition effects in ordinary embedded coalitions.
            # The power transform preserves validity ordering but prevents a
            # single non-anchor cue from creating a broad primacy crossover.
            weights = np.power(
                np.clip(weights, 1e-12, None), ordinary_validity_compression
            )
        raw = float(np.sum(weights))
        # The power term gives ordinary sublinear accumulation. The bounded
        # saturation term makes additional concordant reports increasingly
        # interpretable as redundant without erasing coalition size entirely.
        discount = (float(n) ** redundancy_exponent) * (
            1.0 + redundancy_saturation * float(max(n - 1, 0))
            / float(n + 1)
        )
        return raw / max(discount, 1e-12)

    support_a = coalition_support(a_cues)
    support_b = coalition_support(b_cues)

    # Highest communicated validity among discriminating cues defines the
    # anchor. Stable display order is used only to resolve a top-validity tie.
    active_v = validities[active]
    best_v = float(np.max(active_v))
    best_candidates = active[np.isclose(active_v, best_v, atol=1e-10)]
    anchor_cue = int(np.min(best_candidates))
    anchor_direction = float(directions[anchor_cue])
    unique_anchor = float(best_candidates.size == 1)

    # Relative weights alone make every locally strongest cue equivalent.
    # This bounded gate preserves that representation while marking anchors
    # that are weak in absolute communicated-validity terms as challengeable.
    weak_absolute_anchor = sigmoid(
        absolute_anchor_slope * (absolute_anchor_threshold - best_v)
    )
    weak_anchor_multiplier = 1.0 + weak_anchor_gain * weak_absolute_anchor

    if active.size > 1:
        sorted_active_v = np.sort(active_v)[::-1]
        validity_gap = float(sorted_active_v[0] - sorted_active_v[1])
    else:
        validity_gap = 0.5

    if anchor_direction > 0:
        anchor_side_count = n_b
        opposition_count = n_a
        anchor_support = support_b
        opposition_support = support_a
        opposition_cues = a_cues
    else:
        anchor_side_count = n_a
        opposition_count = n_b
        anchor_support = support_a
        opposition_support = support_b
        opposition_cues = b_cues

    isolated_anchor = anchor_side_count == 1 and opposition_count > 0

    # Credibility is based on the strongest one or two individual opponents.
    # For two-opponent challenge, the weaker member must also be credible, so
    # one strong report cannot make a heterogeneous pair diagnostic.
    if opposition_count > 0:
        opponent_validities = np.sort(validities[opposition_cues])[::-1]
        credibility_set = opponent_validities[:min(2, opposition_count)]
        opponent_credibility = float(np.mean(credibility_set))
        credibility_gap = max(best_v - opponent_credibility, 0.0)
        joint_credibility_gap = max(
            best_v - float(np.min(credibility_set)), 0.0
        )
    else:
        credibility_gap = 0.5
        joint_credibility_gap = 0.5

    if isolated_anchor:
        # Challenge is sharply asymmetric across counts. The one-opponent
        # branch has a less suppressive credibility gate and substantially
        # separated stable subject types. Weak absolute anchors amplify the
        # existing challenge rather than creating a general coalition bonus.
        if opposition_count == 1:
            compact_profile = 3.00
            challenge_gap = 0.35 * credibility_gap
            active_challenge_type_gain = one_challenge_type_gain
        elif opposition_count == 2:
            compact_profile = 1.35
            challenge_gap = joint_credibility_gap
            active_challenge_type_gain = challenge_type_gain
        else:
            distance = (float(opposition_count) - 1.5) / max(compact_width, 1e-6)
            compact_profile = np.exp(-(distance ** 4))
            challenge_gap = joint_credibility_gap
            active_challenge_type_gain = challenge_type_gain
        credibility_gate = np.exp(
            -challenge_gap / max(credibility_gap_scale, 1e-6)
        )
        challenged_opposition = opposition_support * (
            1.0
            + active_challenge_type_gain
            * compact_challenge
            * compact_profile
            * credibility_gate
            * weak_anchor_multiplier
        )
        margin = (anchor_support - challenged_opposition) / max(
            anchor_support + challenged_opposition, 1e-12
        )
        core_evidence = anchor_direction * margin
    else:
        # Compress cue-composition effects only for unique-anchor ordinary
        # coalitions. Tied-top profiles retain the accepted integrator so the
        # calibrated bounded order effect is not disturbed.
        if unique_anchor > 0.0:
            support_a = coalition_support(a_cues, compress=True)
            support_b = coalition_support(b_cues, compress=True)
            if anchor_direction > 0:
                anchor_support = support_b
                opposition_support = support_a
            else:
                anchor_support = support_a
                opposition_support = support_b

        # Balanced opposition is distinctive only when the strongest opposing
        # cue is close to the anchor in communicated validity. A bounded breadth
        # factor further prevents a compact 2-vs-2 profile from receiving the
        # same categorical boost as a broader coalition-level conflict.
        if (
            unique_anchor > 0.0
            and anchor_side_count == opposition_count
            and anchor_side_count > 1
        ):
            strongest_opponent_v = float(np.max(validities[opposition_cues]))
            balanced_validity_gap = max(best_v - strongest_opponent_v, 0.0)
            geometry_gate = np.exp(
                -balanced_validity_gap / max(credibility_gap_scale, 1e-6)
            )
            breadth_gate = min(
                1.0, float(anchor_side_count - 1) / 2.0
            )
            balanced_gain = (
                balanced_opposition_bonus * geometry_gate * breadth_gate
            )
            opposition_support *= 1.0 + balanced_gain
            if anchor_direction > 0:
                support_a = opposition_support
            else:
                support_b = opposition_support

        # Embedded distinctiveness is sharply singleton-localized. Its broad-
        # coalition component remains strong, while the generic component is
        # attenuated for smaller coalitions to reduce excessive profile variance.
        # Absolute weakness amplifies this existing channel without altering
        # balanced or other ordinary embedded configurations.
        if opposition_count == 1 and anchor_side_count > 1:
            imbalance = (
                float(anchor_side_count - opposition_count)
                / float(anchor_side_count + opposition_count)
            )
            dissent_multiplier = 1.0 + (
                minority_amplification * imbalance * weak_anchor_multiplier
            )
            breadth_weight = 1.0 if anchor_side_count >= 5 else 0.20
            dissent_multiplier += (
                singleton_type_gain
                * singleton_bonus
                * breadth_weight
                * np.log1p(float(anchor_side_count - 1))
                * weak_anchor_multiplier
            )
            # A very broad concordant coalition makes its sole dissenter
            # especially distinctive, while leaving three-versus-one control
            # profiles close to the accepted static accumulator.
            if anchor_side_count >= 5:
                dissent_multiplier += (
                    singleton_type_gain
                    * singleton_bonus
                    * 0.45
                    * float(anchor_side_count - 4)
                    * weak_anchor_multiplier
                )
            opposition_support *= dissent_multiplier

            if anchor_direction > 0:
                support_a = opposition_support
            else:
                support_b = opposition_support

        total_support = support_a + support_b
        core_evidence = (support_b - support_a) / max(total_support, 1e-12)

    # Gap-sensitive restoration remains available, but exactly two opponents
    # receive little automatic restoration. A separate steep count branch is
    # almost off at two and effectively reaches the common gate ceiling at
    # three or more, without requiring the gap gates or subject type to agree.
    restoration_gate = 0.0
    if isolated_anchor and unique_anchor > 0.0:
        count_gate = sigmoid(
            restoration_slope
            * (float(opposition_count) - restoration_threshold)
        )
        gap_gate = sigmoid(
            restoration_gap_slope
            * (validity_gap - restoration_gap_threshold)
        )
        credibility_restoration = sigmoid(
            restoration_credibility_slope
            * (credibility_gap - restoration_credibility_threshold)
        )
        selective_restoration = (
            restoration_ceiling
            * count_gate
            * gap_gate
            * credibility_restoration
        )

        if opposition_count == 2:
            selective_restoration *= 0.30

        large_count_gate = sigmoid(
            restoration_slope
            * (float(opposition_count) - 2.50)
        )
        count_dominant_restoration = 0.99 * large_count_gate
        restoration_gate = max(
            selective_restoration,
            count_dominant_restoration,
        )

    # Equal top validities remain exchangeable in integration. Order only
    # supplies a bounded tie anchor rather than a general first-cue bonus.
    tied_top = float(best_candidates.size > 1)
    order_gate = tie_order_gate * tied_top
    gate = 1.0 - (1.0 - restoration_gate) * (1.0 - order_gate)
    gate = float(np.clip(gate, 0.0, 0.95))

    choice_evidence = (
        (1.0 - gate) * core_evidence
        + gate * anchor_direction
    )

    logits = beta * np.array(
        [-0.5 * choice_evidence, 0.5 * choice_evidence],
        dtype=np.float64,
    )
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs /= probs.sum()

    probs = (1.0 - lapse) * probs + lapse * np.array(
        [0.5, 0.5], dtype=np.float64
    )
    probs = np.clip(probs, 0.0, None)
    probs /= probs.sum()
    return probs.astype(np.float64)


`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- validities: validities
- validity_sensitivity: [0.15, 0.85]
- redundancy_exponent: [0.28, 0.68]
- redundancy_saturation: [0.15, 0.65]
- ordinary_validity_compression: [0.03, 0.18]
- minority_amplification: [0.20, 0.70]
- balanced_opposition_bonus: [0.55, 1.25]
- singleton_bonus: [1.8, 4.5]
- compact_challenge: [2.4, 6.0]
- compact_width: [0.75, 1.15]
- credibility_gap_scale: [0.018, 0.055]
- absolute_anchor_threshold: [0.68, 0.78]
- absolute_anchor_slope: [20.0, 45.0]
- weak_anchor_gain: [1.2, 3.0]
- restoration_threshold: [1.65, 2.15]
- restoration_slope: [9.0, 18.0]
- restoration_gap_threshold: [0.035, 0.085]
- restoration_gap_slope: [70.0, 150.0]
- restoration_credibility_threshold: [0.025, 0.065]
- restoration_credibility_slope: [70.0, 150.0]
- restoration_ceiling: [0.82, 0.98]
- tie_order_gate: [0.48, 0.78]
- beta: [1.2, 4.2]
- lapse: [0.0, 0.14]
- configuration_type: {0, 1, 2}

`rationale`:
This is a localized edit of the accepted iteration-7 model. It retains the accepted validity-geometry gate and does not repeat either rejected balanced-coalition modification. The principal addition is a bounded absolute-anchor-credibility gate. Because the existing relative normalization assigns weight 1 to whichever active cue is locally strongest, it previously treated a late, low-validity anchor like a genuinely high-validity anchor. The new sigmoid uses the anchor's communicated validity directly and amplifies only the already existing isolated-challenge and embedded-singleton channels. It therefore targets low-rank active anchors in Experiment 2 without creating another general balanced-opposition bonus that could reverse Experiment 3 or revive Experiment 4's crossover. High-validity anchors receive a multiplier close to one, protecting Experiments 6 and 9. The only other substantive change makes the accepted count-dominant restoration branch nearly type-invariant and effectively ceiling-level for three or more opponents while remaining almost inactive at two. This addresses the residual Experiment 6 contrast from its multi-opponent side and leaves Experiment 9's exactly-two calibration essentially unchanged. History invariance, ordinary-coalition compression, tied-top order use, exactly-two selective restoration, temperature, lapse, and all other equations and parameter ranges are preserved.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.2803 -> ACCEPTED
- iter 2: loss=0.2859 -> REJECTED
- iter 3: loss=0.1994 -> ACCEPTED
- iter 4: loss=0.1886 -> ACCEPTED
- iter 5: loss=0.1677 -> ACCEPTED
- iter 6: loss=0.1123 -> ACCEPTED
- iter 7: loss=0.1073 -> ACCEPTED
- iter 8: loss=0.1238 -> REJECTED
- iter 9: loss=0.1204 -> REJECTED
- iter 10 (current candidate you are grading): loss=0.0914 -> ACCEPTED
Running-best (last accepted) base: iter 10 at loss=0.0914.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Prevalence of subjects choosing the first-discriminating-cue winner above chance."""
    if data is None or len(data) == 0:
        return float("nan")

    def subject_score(subj):
        agreements = []
        for _, row in subj.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float)
            b = np.asarray(row["option_b_ratings"], dtype=float)
            diff = a - b
            discriminating = np.flatnonzero(diff != 0)
            if discriminating.size == 0:
                continue
            first = int(discriminating[0])
            ttb_response = 0 if diff[first] > 0 else 1
            agreements.append(float(int(row["response"]) == ttb_response))

        if len(agreements) == 0:
            return float("nan")
        rate = float(np.mean(agreements))
        if rate > 0.5:
            return 1.0
        if rate < 0.5:
            return 0.0
        return 0.5

    scores = []
    for _, subj in data.groupby("subject_id", sort=False):
        score = subject_score(subj)
        if np.isfinite(score):
            scores.append(score)

    return float(np.mean(scores)) if len(scores) else float("nan")
```

**Observed (real) value:** 1.0000 (var=0.0000)
**Candidate trajectory (this loop):**
  - iter 1: 0.9500 (var=0.0425) (Δ vs real -0.0500)
  - iter 2: 0.9800 (var=0.0196) (Δ vs real -0.0200)
  - iter 3: 0.9000 (var=0.0800) (Δ vs real -0.1000)
  - iter 4: 0.9800 (var=0.0196) (Δ vs real -0.0200)
  - iter 5: 0.9500 (var=0.0425) (Δ vs real -0.0500)
  - iter 6: 0.9200 (var=0.0736) (Δ vs real -0.0800)
  - iter 7: 0.9900 (var=0.0049) (Δ vs real -0.0100)
  - iter 8: 0.9300 (var=0.0601) (Δ vs real -0.0700)
  - iter 9: 0.9700 (var=0.0241) (Δ vs real -0.0300)
  - iter 10 (current): 0.9700 (var=0.0241) (Δ vs real -0.0300)
**Other theories' values on this metric (for reference):**
- pi_1: 0.9900 (var=0.0049)
- pi_2: 0.0000 (var=0.0000)
- pi_3: 1.0000 (var=0.0000)
- pi_4: 1.0000 (var=0.0000)
- pi_5: 1.0000 (var=0.0000)
- pi_6: 0.9600 (var=0.0284)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Proportion of choices consistent with the highest-validity discriminating cue."""
    if len(data) == 0:
        return float("nan")

    aligned = []
    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"],
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        favored = None

        # Features are already ordered from highest to lowest validity.
        for j in range(min(len(a), len(b))):
            if a[j] > b[j]:
                favored = 0
                break
            if b[j] > a[j]:
                favored = 1
                break

        # Omit completely nondiscriminating pairs, although none occur in
        # the specified design.
        if favored is not None:
            aligned.append(float(int(response) == favored))

    if not aligned:
        return float("nan")
    return float(np.mean(aligned))
```

**Observed (real) value:** 0.3242 (var=0.0174)
**Candidate trajectory (this loop):**
  - iter 1: 0.5373 (var=0.0075) (Δ vs real +0.2131)
  - iter 2: 0.5833 (var=0.0093) (Δ vs real +0.2592)
  - iter 3: 0.5813 (var=0.0030) (Δ vs real +0.2571)
  - iter 4: 0.5787 (var=0.0041) (Δ vs real +0.2546)
  - iter 5: 0.5867 (var=0.0036) (Δ vs real +0.2625)
  - iter 6: 0.4267 (var=0.0045) (Δ vs real +0.1025)
  - iter 7: 0.5281 (var=0.0037) (Δ vs real +0.2040)
  - iter 8: 0.5290 (var=0.0039) (Δ vs real +0.2048)
  - iter 9: 0.5429 (var=0.0032) (Δ vs real +0.2188)
  - iter 10 (current): 0.5231 (var=0.0044) (Δ vs real +0.1990)
**Other theories' values on this metric (for reference):**
- pi_2: 0.4848 (var=0.0022)
- pi_1: 0.8492 (var=0.0121)
- pi_3: 0.3298 (var=0.0054)
- pi_4: 0.9925 (var=0.0001)
- pi_5: 0.3463 (var=0.0061)
- pi_6: 0.4562 (var=0.0022)

### Experiment 3
**Design**
  A=[0, 1, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Balanced-minus-imbalanced change in choosing Expert 5's winner."""
    if data is None or len(data) == 0:
        return float("nan")

    interaction_scores = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.size < 5 or b.size != a.size:
            continue

        diff = b - a
        signs = np.sign(diff)
        n_b = int(np.sum(signs > 0))
        n_a = int(np.sum(signs < 0))
        imbalance = abs(n_b - n_a)

        # The design's critical conditions are exact 2-vs-2 balance and
        # 3-vs-1 imbalance. Ignore any unexpected rows.
        if imbalance == 0:
            condition_sign = 1.0
        elif imbalance == 2:
            condition_sign = -1.0
        else:
            continue

        # Expert 5 is feature index 4 and always identifies the TTB winner.
        if diff[4] > 0:
            expert5_winner = 1
        elif diff[4] < 0:
            expert5_winner = 0
        else:
            continue

        chose_expert5_winner = float(int(row["response"]) == expert5_winner)
        # Averaging this score gives P(E5 winner | balanced) minus
        # P(E5 winner | imbalanced), since the schedule is balanced.
        interaction_scores.append(condition_sign * chose_expert5_winner)

    if len(interaction_scores) == 0:
        return float("nan")
    return float(np.mean(interaction_scores))
```

**Observed (real) value:** 0.0721 (var=0.0052)
**Candidate trajectory (this loop):**
  - iter 1: 0.1102 (var=0.0068) (Δ vs real +0.0381)
  - iter 2: 0.0063 (var=0.0035) (Δ vs real -0.0658)
  - iter 3: 0.0779 (var=0.0034) (Δ vs real +0.0058)
  - iter 4: 0.1079 (var=0.0051) (Δ vs real +0.0358)
  - iter 5: 0.1367 (var=0.0033) (Δ vs real +0.0646)
  - iter 6: -0.0617 (var=0.0043) (Δ vs real -0.1338)
  - iter 7: 0.0525 (var=0.0037) (Δ vs real -0.0196)
  - iter 8: 0.0427 (var=0.0038) (Δ vs real -0.0294)
  - iter 9: 0.0356 (var=0.0043) (Δ vs real -0.0365)
  - iter 10 (current): 0.0398 (var=0.0035) (Δ vs real -0.0323)
**Other theories' values on this metric (for reference):**
- pi_1: -0.0008 (var=0.0015)
- pi_3: -0.3629 (var=0.0033)
- pi_2: -0.1762 (var=0.0033)
- pi_4: 0.0050 (var=0.0001)
- pi_5: 0.1125 (var=0.0053)
- pi_6: -0.0462 (var=0.0019)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def subject_score(df):
        if len(df) == 0:
            return np.nan

        a = np.stack(df["option_a_ratings"].apply(lambda x: np.asarray(x, dtype=float)).to_numpy())
        b = np.stack(df["option_b_ratings"].apply(lambda x: np.asarray(x, dtype=float)).to_numpy())
        response = df["response"].to_numpy(dtype=int)

        # Expert 5 is uniquely most valid, so its favored option is the
        # Take-The-Best winner in this design.
        ttb_winner = (b[:, 4] > a[:, 4]).astype(int)
        chose_ttb_winner = (response == ttb_winner).astype(float)

        # Whether Expert 1 favors the same option as Expert 5.
        expert1_winner = (b[:, 0] > a[:, 0]).astype(int)
        expert1_supports = expert1_winner == ttb_winner

        if not np.any(expert1_supports) or not np.any(~expert1_supports):
            return np.nan

        crossover = (chose_ttb_winner[expert1_supports].mean() -
                     chose_ttb_winner[~expert1_supports].mean())

        # Classify a subject as showing a substantively sized primacy
        # crossover. The 0.12 margin suppresses chance sampling contrasts
        # around TTB's population prediction of exactly zero.
        return float(crossover > 0.12)

    if "subject_id" in data.columns and data["subject_id"].nunique() > 1:
        scores = [subject_score(g) for _, g in data.groupby("subject_id", sort=False)]
        scores = np.asarray(scores, dtype=float)
        scores = scores[np.isfinite(scores)]
        return float(scores.mean()) if scores.size else np.nan

    return subject_score(data)

```

**Observed (real) value:** 0.0000 (var=0.0000)
**Candidate trajectory (this loop):**
  - iter 1: 0.6000 (var=0.2400) (Δ vs real +0.6000)
  - iter 2: 0.5000 (var=0.2500) (Δ vs real +0.5000)
  - iter 3: 0.1200 (var=0.1056) (Δ vs real +0.1200)
  - iter 4: 0.1800 (var=0.1476) (Δ vs real +0.1800)
  - iter 5: 0.0800 (var=0.0736) (Δ vs real +0.0800)
  - iter 6: 0.2000 (var=0.1600) (Δ vs real +0.2000)
  - iter 7: 0.1000 (var=0.0900) (Δ vs real +0.1000)
  - iter 8: 0.1000 (var=0.0900) (Δ vs real +0.1000)
  - iter 9: 0.1400 (var=0.1204) (Δ vs real +0.1400)
  - iter 10 (current): 0.1200 (var=0.1056) (Δ vs real +0.1200)
**Other theories' values on this metric (for reference):**
- pi_3: 1.0000 (var=0.0000)
- pi_1: 0.0400 (var=0.0384)
- pi_2: 0.0600 (var=0.0564)
- pi_4: 0.0000 (var=0.0000)
- pi_5: 0.0000 (var=0.0000)
- pi_6: 0.2400 (var=0.1824)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Proportion of choices that follow the stable-order winner among the tied top cues."""
    if len(data) == 0:
        return float("nan")

    follows_top = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)

        # Feature 0 is consulted first under the communicated validity order.
        if a[0] > b[0]:
            top_choice = 0
        elif b[0] > a[0]:
            top_choice = 1
        else:
            continue

        follows_top.append(float(int(row["response"]) == top_choice))

    if len(follows_top) == 0:
        return float("nan")
    return float(np.mean(follows_top))
```

**Observed (real) value:** 0.7900 (var=0.0184)
**Candidate trajectory (this loop):**
  - iter 1: 0.7925 (var=0.0066) (Δ vs real +0.0025)
  - iter 2: 0.7473 (var=0.0092) (Δ vs real -0.0427)
  - iter 3: 0.7977 (var=0.0071) (Δ vs real +0.0077)
  - iter 4: 0.7690 (var=0.0092) (Δ vs real -0.0210)
  - iter 5: 0.7910 (var=0.0051) (Δ vs real +0.0010)
  - iter 6: 0.8106 (var=0.0083) (Δ vs real +0.0206)
  - iter 7: 0.7898 (var=0.0055) (Δ vs real -0.0002)
  - iter 8: 0.8048 (var=0.0077) (Δ vs real +0.0148)
  - iter 9: 0.7971 (var=0.0069) (Δ vs real +0.0071)
  - iter 10 (current): 0.7908 (var=0.0081) (Δ vs real +0.0008)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8244 (var=0.0081)
- pi_4: 0.0179 (var=0.0002)
- pi_2: 0.1456 (var=0.0074)
- pi_3: 0.6381 (var=0.0031)
- pi_5: 0.8223 (var=0.0088)
- pi_6: 0.7727 (var=0.0071)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Difference in adherence to the highest-validity cue between
    # one-opponent and multi-opponent coalitions.
    lone_top_choices = []
    multi_top_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)

        # Feature 0 is the uniquely highest-validity expert in this design.
        top_direction = np.sign(b[0] - a[0])  # +1 means B is top-cue winner
        if top_direction == 0:
            continue

        lower_directions = np.sign(b[1:] - a[1:])
        n_opponents = int(np.sum(lower_directions * top_direction < 0))
        chose_top_winner = float(
            int(row["response"]) == (1 if top_direction > 0 else 0)
        )

        if n_opponents == 1:
            lone_top_choices.append(chose_top_winner)
        elif n_opponents >= 2:
            multi_top_choices.append(chose_top_winner)

    # The fixed schedule supplies both classes for every subject. These
    # fallbacks keep the function scalar-valued on unexpected partial data.
    if len(lone_top_choices) == 0 or len(multi_top_choices) == 0:
        return 0.0

    return float(np.mean(lone_top_choices) - np.mean(multi_top_choices))
```

**Observed (real) value:** -0.5450 (var=0.0462)
**Candidate trajectory (this loop):**
  - iter 1: -0.0944 (var=0.0101) (Δ vs real +0.4506)
  - iter 2: -0.1503 (var=0.0158) (Δ vs real +0.3947)
  - iter 3: 0.0741 (var=0.0161) (Δ vs real +0.6191)
  - iter 4: -0.2309 (var=0.0132) (Δ vs real +0.3141)
  - iter 5: -0.2650 (var=0.0127) (Δ vs real +0.2800)
  - iter 6: -0.3928 (var=0.0156) (Δ vs real +0.1522)
  - iter 7: -0.4072 (var=0.0215) (Δ vs real +0.1378)
  - iter 8: -0.3181 (var=0.0371) (Δ vs real +0.2269)
  - iter 9: -0.4503 (var=0.0180) (Δ vs real +0.0947)
  - iter 10 (current): -0.4372 (var=0.0185) (Δ vs real +0.1078)
**Other theories' values on this metric (for reference):**
- pi_4: 0.5559 (var=0.0104)
- pi_1: 0.0103 (var=0.0049)
- pi_2: 0.3781 (var=0.0114)
- pi_3: -0.3150 (var=0.0123)
- pi_5: -0.5416 (var=0.0261)
- pi_6: -0.1356 (var=0.0077)

### Experiment 7
**Design**
  A=[0, 0, 0, 0, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 1, 1]  B=[0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0, 0, 1]
  A=[1, 0, 1, 1, 0, 0, 1]  B=[0, 1, 0, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Post-exposure diagnostic minus control adherence to the top-validity cue."""
    if data is None or len(data) == 0:
        return float("nan")

    def subject_score(df):
        diagnostic = {False: [], True: []}
        control = {False: [], True: []}
        occurrence = {}

        for _, row in df.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float)
            b = np.asarray(row["option_b_ratings"], dtype=float)
            if a.size < 5 or b.size < 5:
                continue

            direction = np.sign(b - a)
            if direction[0] == 0:
                continue

            anchor_b = bool(direction[0] > 0)
            followed_anchor = float(int(row["response"]) == int(anchor_b))

            # Diagnostic trials have Expert 5 opposing the validity anchor.
            is_diagnostic = direction[4] == -direction[0]
            # Control trials instead have Expert 2 opposing the anchor.
            is_control = direction[4] == 0 and direction[1] == -direction[0]

            if is_diagnostic:
                # Use repetitions 3--8 of each exact ordered stimulus. This
                # removes acquisition trials while preserving equal numbers
                # of A- and B-anchor trials through the reversal pairs.
                key = (tuple(a.tolist()), tuple(b.tolist()))
                prior = occurrence.get(key, 0)
                occurrence[key] = prior + 1
                if prior >= 2:
                    diagnostic[anchor_b].append(followed_anchor)
            elif is_control:
                control[anchor_b].append(followed_anchor)

        if any(len(diagnostic[s]) == 0 or len(control[s]) == 0
               for s in (False, True)):
            return float("nan")

        diag_balanced = 0.5 * (
            float(np.mean(diagnostic[False])) +
            float(np.mean(diagnostic[True]))
        )
        control_balanced = 0.5 * (
            float(np.mean(control[False])) +
            float(np.mean(control[True]))
        )
        return diag_balanced - control_balanced

    if "subject_id" in data.columns:
        scores = []
        for _, subj_df in data.groupby("subject_id", sort=False):
            score = subject_score(subj_df)
            if np.isfinite(score):
                scores.append(score)
        return float(np.mean(scores)) if scores else float("nan")

    return subject_score(data)

```

**Observed (real) value:** 0.0052 (var=0.0075)
**Candidate trajectory (this loop):**
  - iter 1: -0.0899 (var=0.0173) (Δ vs real -0.0951)
  - iter 2: -0.0942 (var=0.0201) (Δ vs real -0.0993)
  - iter 3: -0.0333 (var=0.0166) (Δ vs real -0.0384)
  - iter 4: -0.0398 (var=0.0126) (Δ vs real -0.0450)
  - iter 5: -0.0208 (var=0.0249) (Δ vs real -0.0260)
  - iter 6: -0.0049 (var=0.0187) (Δ vs real -0.0101)
  - iter 7: -0.0294 (var=0.0188) (Δ vs real -0.0346)
  - iter 8: -0.0292 (var=0.0204) (Δ vs real -0.0343)
  - iter 9: -0.0272 (var=0.0156) (Δ vs real -0.0323)
  - iter 10 (current): -0.0211 (var=0.0177) (Δ vs real -0.0263)
**Other theories' values on this metric (for reference):**
- pi_1: -0.0058 (var=0.0071)
- pi_5: -0.1060 (var=0.0121)
- pi_2: -0.0056 (var=0.0083)
- pi_3: 0.0227 (var=0.0111)
- pi_4: 0.0025 (var=0.0020)
- pi_6: 0.0092 (var=0.0215)

### Experiment 8
**Design**
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def subject_score(df: pd.DataFrame) -> float:
        # Encode every response as adherence to Expert 1, which is the TTB
        # winner on all trials, and group A/B reversals into invariant cue
        # profiles by orienting each profile toward Expert 1's direction.
        groups = {}
        for _, row in df.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float)
            b = np.asarray(row["option_b_ratings"], dtype=float)
            d = np.sign(b - a).astype(int)
            if d.size == 0 or d[0] == 0:
                continue

            orientation = int(d[0])
            profile = tuple((d * orientation).tolist())
            ttb_response = 1 if d[0] > 0 else 0
            adherence = float(int(row["response"]) == ttb_response)
            groups.setdefault(profile, []).append(adherence)

        # The intended design supplies six profiles with many observations
        # each. Requiring at least two observations permits an unbiased
        # estimate of each profile mean's sampling variance.
        usable = [np.asarray(v, dtype=float) for v in groups.values() if len(v) >= 2]
        k = len(usable)
        if k < 2:
            return 0.0

        means = np.asarray([np.mean(v) for v in usable], dtype=float)

        # Observed population variance of profile-specific TTB-adherence
        # rates. Under TTB all profiles have the same latent rate, but its
        # finite-sample value is positive because of response noise.
        observed_profile_variance = float(np.var(means, ddof=0))

        # For Bernoulli observations, p_hat*(1-p_hat)/(n-1) is an unbiased
        # estimator of Var(p_hat). Subtract the exact contribution of these
        # estimation errors to the population variance across k profiles.
        mean_sampling_variances = []
        for v, p_hat in zip(usable, means):
            n = len(v)
            mean_sampling_variances.append(
                float(p_hat * (1.0 - p_hat) / float(n - 1))
            )
        noise_bias = (1.0 - 1.0 / float(k)) * float(
            np.mean(mean_sampling_variances)
        )

        return observed_profile_variance - noise_bias

    if len(data) == 0:
        return 0.0
    if "subject_id" in data.columns:
        scores = [
            subject_score(g)
            for _, g in data.groupby("subject_id", sort=False)
        ]
        return float(np.mean(scores)) if scores else 0.0
    return subject_score(data)

```

**Observed (real) value:** 0.0070 (var=0.0002)
**Candidate trajectory (this loop):**
  - iter 1: 0.0059 (var=0.0002) (Δ vs real -0.0011)
  - iter 2: 0.0017 (var=0.0001) (Δ vs real -0.0053)
  - iter 3: 0.0124 (var=0.0002) (Δ vs real +0.0054)
  - iter 4: 0.0164 (var=0.0003) (Δ vs real +0.0094)
  - iter 5: 0.0238 (var=0.0004) (Δ vs real +0.0168)
  - iter 6: 0.0090 (var=0.0002) (Δ vs real +0.0020)
  - iter 7: 0.0025 (var=0.0001) (Δ vs real -0.0045)
  - iter 8: 0.0041 (var=0.0001) (Δ vs real -0.0029)
  - iter 9: 0.0014 (var=0.0001) (Δ vs real -0.0056)
  - iter 10 (current): 0.0059 (var=0.0002) (Δ vs real -0.0011)
**Other theories' values on this metric (for reference):**
- pi_5: 0.0058 (var=0.0001)
- pi_1: 0.0002 (var=0.0000)
- pi_2: 0.0206 (var=0.0003)
- pi_3: 0.0317 (var=0.0005)
- pi_4: 0.0000 (var=0.0000)
- pi_6: 0.0018 (var=0.0001)

### Experiment 9
**Design**
  A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]
  A=[0, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if data is None or len(data) == 0:
        return 0.5

    def subject_score(subj):
        anchor_choices = []

        for _, row in subj.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float)
            b = np.asarray(row["option_b_ratings"], dtype=float)
            if a.size < 5 or b.size < 5:
                continue

            directions = np.sign(b - a)
            active = set(np.flatnonzero(directions != 0).tolist())

            # The two isolated-anchor target classes and their A/B reversals.
            is_outer_target = active == {0, 3, 4}
            is_middle_target = active == {1, 2, 4}
            if not (is_outer_target or is_middle_target):
                continue

            lower = [0, 3] if is_outer_target else [1, 2]
            coalition_is_concordant = directions[lower[0]] == directions[lower[1]]
            coalition_opposes_anchor = directions[lower[0]] == -directions[4]
            if not (coalition_is_concordant and coalition_opposes_anchor):
                continue

            anchor_response = 1 if directions[4] > 0 else 0
            anchor_choices.append(float(int(row["response"]) == anchor_response))

        if len(anchor_choices) == 0:
            return np.nan
        return float(np.mean(anchor_choices))

    if "subject_id" in data.columns:
        scores = [subject_score(subj) for _, subj in data.groupby("subject_id", sort=False)]
        scores = [x for x in scores if np.isfinite(x)]
        return float(np.mean(scores)) if len(scores) > 0 else 0.5

    score = subject_score(data)
    return float(score) if np.isfinite(score) else 0.5

```

**Observed (real) value:** 0.7250 (var=0.0723)
**Candidate trajectory (this loop):**
  - iter 1: 0.3887 (var=0.0206) (Δ vs real -0.3362)
  - iter 2: 0.7444 (var=0.0417) (Δ vs real +0.0194)
  - iter 3: 0.8075 (var=0.0116) (Δ vs real +0.0825)
  - iter 4: 0.8219 (var=0.0096) (Δ vs real +0.0969)
  - iter 5: 0.7325 (var=0.0148) (Δ vs real +0.0075)
  - iter 6: 0.7412 (var=0.0155) (Δ vs real +0.0162)
  - iter 7: 0.7425 (var=0.0146) (Δ vs real +0.0175)
  - iter 8: 0.7675 (var=0.0097) (Δ vs real +0.0425)
  - iter 9: 0.7575 (var=0.0171) (Δ vs real +0.0325)
  - iter 10 (current): 0.7350 (var=0.0124) (Δ vs real +0.0100)
**Other theories' values on this metric (for reference):**
- pi_6: 0.8094 (var=0.0121)
- pi_5: 0.5687 (var=0.0166)
- pi_1: 0.8600 (var=0.0122)
- pi_2: 0.1737 (var=0.0171)
- pi_3: 0.2675 (var=0.0088)
- pi_4: 0.9844 (var=0.0007)

### Experiment 10
**Design**
  A=[0, 0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def subject_score(subj):
        followed = []

        for _, row in subj.iterrows():
            a = np.asarray(row['option_a_ratings'], dtype=int)
            b = np.asarray(row['option_b_ratings'], dtype=int)
            if a.shape[0] != 6 or b.shape[0] != 6:
                continue

            directions = np.sign(b - a)

            # Diagnostic compact opposition: Experts 3 and 5 oppose the
            # unique validity anchor (trial-pair 1 and its reversal).
            compact_diagnostic = (
                np.array_equal(np.abs(directions),
                               np.array([1, 0, 1, 0, 1, 0]))
                and directions[2] == directions[4]
                and directions[0] == -directions[2]
            )

            # Singleton diagnostic dissent: Expert 3 alone opposes all five
            # other discriminating experts (trial-pair 3 and its reversal).
            singleton_diagnostic = (
                np.all(np.abs(directions) == 1)
                and directions[2] == -directions[0]
                and directions[1] == directions[0]
                and directions[3] == directions[0]
                and directions[4] == directions[0]
                and directions[5] == directions[0]
            )

            if not (compact_diagnostic or singleton_diagnostic):
                continue

            anchor_response = 1 if directions[0] > 0 else 0
            followed.append(float(int(row['response']) == anchor_response))

        if len(followed) == 0:
            return np.nan
        return float(np.mean(followed))

    scores = []
    for _, subj in data.groupby('subject_id', sort=False):
        value = subject_score(subj)
        if np.isfinite(value):
            scores.append(value)

    if len(scores) == 0:
        return np.nan
    return float(np.mean(scores))
```

**Observed (real) value:** 0.4462 (var=0.0224)
**Candidate trajectory (this loop):**
  - iter 1: 0.3488 (var=0.0112) (Δ vs real -0.0975)
  - iter 2: 0.6813 (var=0.0230) (Δ vs real +0.2350)
  - iter 3: 0.5725 (var=0.0070) (Δ vs real +0.1263)
  - iter 4: 0.5288 (var=0.0041) (Δ vs real +0.0825)
  - iter 5: 0.4612 (var=0.0062) (Δ vs real +0.0150)
  - iter 6: 0.4906 (var=0.0068) (Δ vs real +0.0444)
  - iter 7: 0.4775 (var=0.0084) (Δ vs real +0.0312)
  - iter 8: 0.4813 (var=0.0068) (Δ vs real +0.0350)
  - iter 9: 0.4825 (var=0.0055) (Δ vs real +0.0363)
  - iter 10 (current): 0.4731 (var=0.0057) (Δ vs real +0.0269)
**Other theories' values on this metric (for reference):**
- pi_5: 0.4656 (var=0.0070)
- pi_6: 0.7306 (var=0.0084)
- pi_1: 0.8250 (var=0.0150)
- pi_2: 0.5275 (var=0.0076)
- pi_3: 0.7612 (var=0.0061)
- pi_4: 0.9844 (var=0.0006)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1
Verdict: regenerate
Interpretation: PDRR is faithful to the prescribed configuration-dependent, history-invariant mechanism family and is an accepted improvement, but it is not yet experiment-invariant. It fits Experiment 5 almost exactly (0.7925 vs. 0.7900), Experiment 8 well (0.0059 vs. 0.0070), and is reasonably close on Experiments 1 and 3. The central challenge/restoration predictions, however, remain substantially miscalibrated. Anchor following is too high in Experiment 2 (0.5373 vs. 0.3242), while the one-versus-multiple contrast in Experiment 6 is far too weak (-0.0944 vs. -0.5450). Conversely, two-opponent isolated-anchor adherence is much too low in Experiment 9 (0.3887 vs. 0.7250). Thus, opponent count alone is not separating challenge from restoration correctly across configurations. Experiment 4 also shows a large spurious crossover prevalence (0.6000 vs. 0), despite the explicit order gate being inactive there; this crossover must therefore be induced indirectly by the broad count-based minority amplification and validity-weighted coalition composition. Experiment 7 is history-invariant as required, but still produces a non-null static diagnostic-control contrast (-0.0899 vs. 0.0052), so omitting learning is necessary but not sufficient. Experiment 10 is moderately too anti-anchor (0.3488 vs. 0.4462). Between-subject variance is also too low in several critical experiments, especially Experiments 2, 6, 9, and 10.
Rationale: Keep PDRR's mechanism family, but recalibrate how configuration gates challenge, dissent, and restoration. First, the compact challenge should not be a function of opponent count alone. Gate its amplitude by the opponents' combined communicated reliability relative to the anchor. This can distinguish the credible compact opposition requiring low adherence in Experiments 2 and 10 from the weaker two-cue coalition in Experiment 9, where adherence must rise toward 0.725. Second, make the challenge peak more concentrated at one opponent and lower the restoration threshold so that multi-opponent trials restore the anchor much more strongly. That joint change should both lower one-opponent adherence in Experiment 6 and raise multi-opponent adherence, producing the required large negative contrast rather than merely shifting all isolated-anchor trials together. Third, replace the broad embedded-dissent multiplier applied whenever the anchor side is numerically larger with a sharply singleton-localized, bounded distinctiveness gate based on support ratio as well as count. The current general imbalance amplification creates the severe Experiment 4 crossover even without an explicit primacy term. Preserve strong singleton effects where needed for Experiment 2, but suppress amplification for ordinary two-versus-four and similar coalitions. Fourth, tune this singleton gate so that the matched diagnostic and control configurations in Experiment 7 have nearly equal static anchor adherence; the current -0.09 contrast is a configuration artifact, not learning. Finally, widen stable subject-level random effects for challenge strength, restoration threshold, singleton amplification, temperature, and lapse, since the critical simulated variances are systematically below the observed heterogeneity. Do not add history dependence or broaden the order mechanism: the tied-top order fit in Experiment 5 is already excellent, and the Experiment 4 failure originates elsewhere.

**Outcome of your advice:** iter 1 candidate loss=0.2803 -> iter 2 candidate loss=0.2859 -> the gate marked it REJECTED.

### Iteration 2
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed history-invariant, configuration-dependent selective-integration family, but the current revision was rejected by the loss gate and should not replace the accepted iteration-1 base. It successfully repaired Experiment 9 (0.7444 vs. 0.7250) and modestly strengthened the Experiment 6 contrast, but these gains came from a global challenge/restoration recalibration that substantially worsened Experiment 2 (0.5833 vs. 0.3242), Experiment 10 (0.6813 vs. 0.4462), Experiment 3 (0.0063 vs. 0.0721), and Experiment 5 (0.7473 vs. 0.7900). The key dissociation is now especially clear: the two-opponent coalition in Experiment 9 should preserve anchor following, whereas compact opposition in Experiment 10 should challenge it. Redundancy-discounted total support does not distinguish these cases adequately. Experiment 6 still has far too little challenge/restoration separation (-0.1503 vs. -0.5450), and Experiment 4 retains a severe spurious crossover prevalence (0.5000 vs. 0), even though its configurations do not activate the new singleton gate. That implicates ordinary validity-weighted coalition composition, rather than minority amplification alone. Experiment 7 also remains spuriously negative (-0.0942 vs. 0.0052). Fits to Experiments 1, 8, and 9 are good, but the pattern is not experiment-invariant.
Rationale: Because the previous recommendation's global lowering/sharpening of restoration and support-ratio gating was rejected, do not repeat or intensify those shifts. Build from the unchanged accepted iteration-1 base and use a more selective in-family factorization. For isolated-anchor trials, distinguish coalitions using the reliability of the strongest and second-strongest individual opponents, or an opponent-reliability dispersion statistic, rather than only their redundancy-discounted summed support. Couple restoration to a two-dimensional count-by-validity-gap gate: multi-opponent coalitions with a clearly superior anchor should restore strongly, as in Experiments 6 and 9, while two individually credible opponents with a small anchor gap should retain challenge, as in Experiment 10. Keep compact challenge and embedded singleton dissent on separately parameterized channels so lowering pooled anchor following in Experiment 2 does not globally suppress anchor following in Experiment 9 or 10.

Separately address Experiment 4 at the ordinary-coalition accumulator. Since the singleton gate is inactive there, bound the marginal influence of any one non-anchor cue or normalize non-anchor weights within each coalition before count discounting. This should make swapping Expert 1's alignment less capable of producing a subject-level crossover while preserving communicated-validity weighting and the tied-top order mechanism. Apply this compression only to embedded ordinary coalitions, not isolated-anchor trials. Re-establish the accepted base's tied-top parameters, which already fit Experiment 5 almost exactly, rather than changing order-related ranges. Finally, tune stable mixture weights rather than simply widening all parameter ranges: include challenge-sensitive, gap-restoring, and weakly configuration-sensitive subject types so the variances in Experiments 6, 9, and 10 can increase without moving all means together. Keep history dependence absent.

**Outcome of your advice:** iter 2 candidate loss=0.2859 -> iter 3 candidate loss=0.1994 -> the gate marked it ACCEPTED.

### Iteration 3
Verdict: regenerate
Interpretation: The accepted revision is a meaningful improvement and remains faithful to the prescribed configuration-dependent, history-invariant PDRR family. The previously recommended selective factorization helped: Experiment 3 is now nearly exact (0.0779 vs. 0.0721), the spurious Experiment 4 crossover fell sharply from 0.6000 to 0.1200, Experiment 7 moved toward zero (-0.0333 vs. 0.0052), and the tied-top order effect remains excellent in Experiment 5 (0.7977 vs. 0.7900). Experiments 8 and 9 are also reasonably close in mean. Nevertheless, the central challenge/restoration signature still fails. Experiment 6 has the wrong sign and a very large error: the candidate predicts greater anchor adherence with one opponent than with multiple opponents (0.0741), whereas humans show a large negative contrast (-0.5450). Experiment 2 also remains much too anchor-following (0.5813 vs. 0.3242), and Experiment 10 is too anchor-following (0.5725 vs. 0.4462). Thus, credible compact opposition and singleton dissent remain too weak, while count-based restoration is not reliably dominating for larger coalitions. Experiment 9 is somewhat over-restored (0.8075 vs. 0.7250). The critical between-subject variances are also compressed, especially in Experiments 2, 6, 9, and 10. Experiment 1 has deteriorated to 0.9000 versus unanimous observed prevalence, likely because some large-coalition profiles are not entering restoration consistently.
Rationale: Continue from this accepted candidate and preserve the successful ordinary-coalition compression and tied-top order settings; the previous advice in that direction was accepted and substantially repaired Experiments 3, 4, and 7. The next revision should target the isolated-anchor gates rather than globally changing integration. First, inspect and recalibrate the conjunction of the three restoration gates. Multiplying count, validity-gap, and credibility-gap sigmoids appears to suppress restoration too often in Experiment 6, even for large coalitions. Add a count-dominant restoration branch for clearly large coalitions, such as count at least three, that is bounded but does not require all gap gates to be simultaneously near one. This should raise multi-opponent anchor adherence and recover the negative Experiment 6 contrast while also protecting Experiment 1. Do not globally lower the restoration threshold, since that earlier broad strategy was rejected and would further overfit Experiment 9. Second, make compact challenge asymmetric across counts rather than using the current profile centered symmetrically at 1.5: allow especially strong one-opponent challenge, with a separately calibrated two-opponent challenge controlled tightly by individual opponent credibility. Stronger challenge for genuinely near-anchor opponents should lower adherence in Experiment 10 and relevant compact trials in Experiment 2 without undoing weak-opponent restoration in Experiment 9. Third, strengthen or broaden the embedded singleton-dissent channel selectively for the Experiment 2 configurations; its current bounded multiplier is insufficient to produce the observed pooled rate. Use a support-ratio or dissent-validity gate so this adjustment does not revive the Experiment 4 crossover. Finally, implement genuinely stable mixture components or correlated random effects—challenge-sensitive, count-restoring, and weakly selective types—rather than only independent narrow parameter draws. This is needed to increase variance in Experiments 6 and 9 while allowing some subjects to challenge strongly enough in Experiments 2 and 10. Keep history dependence absent and retain the current bounded tied-top order mechanism.

**Outcome of your advice:** iter 3 candidate loss=0.1994 -> iter 4 candidate loss=0.1886 -> the gate marked it ACCEPTED.

### Iteration 4
Verdict: regenerate
Interpretation: The accepted revision is a further aggregate improvement and remains faithful to the prescribed history-invariant, configuration-dependent selective-integration family. The count-dominant restoration branch moved Experiment 6 in the correct direction, from +0.0741 to -0.2309, and restored Experiment 1 from 0.9000 to 0.9800. The asymmetric challenge also improved Experiment 10 from 0.5725 to 0.5288. These gains confirm that the most recent, accepted advice identified useful in-family controls. Fits are good for Experiments 1, 3, 5, 7, and 8. Nevertheless, the model is not yet experiment-invariant. Experiment 2 remains the largest qualitative failure: anchor following is 0.5787 versus 0.3242, barely changed despite stronger compact and singleton mechanisms. Experiment 6 still expresses less than half of the observed challenge/restoration separation (-0.2309 versus -0.5450). Experiment 9 is now over-restored (0.8219 versus 0.7250), while Experiment 10 remains too anchor-following (0.5288 versus 0.4462). Experiment 4 also retains a nonzero spurious crossover prevalence (0.1800 versus 0), although it is much better than early iterations. Finally, stable subject heterogeneity is still much too weak in the critical configurations: variances are 0.0041 versus 0.0174 in Experiment 2, 0.0132 versus 0.0462 in Experiment 6, 0.0096 versus 0.0723 in Experiment 9, and 0.0041 versus 0.0224 in Experiment 10.
Rationale: Continue from this accepted candidate and preserve the successful ordinary-coalition compression, history invariance, and tied-top order mechanism. Extend the accepted count-restoration adjustment rather than reversing it: make the count-dominant gate closer to zero for exactly two opponents but steeper and nearer its ceiling for three or more. Also attenuate the selective restoration branch more strongly at exactly two opponents. This should lower the over-restored Experiment 9 value while raising multi-opponent adherence in Experiment 6, increasing the required negative contrast without globally lowering the restoration threshold—a broad move that was previously rejected.

The persistent Experiment 2 failure now requires condition-level calibration rather than another pooled global shift. Report or internally target its compact isolated-anchor and embedded-dissent trial classes separately. Increase challenge selectively for a single credible opponent, and increase two-opponent challenge only when both opponents are individually near the anchor in communicated validity. A modestly sharper credibility-gap gate paired with greater challenge amplitude can lower the credible compact conditions in Experiments 2 and 10 while sparing the weaker two-opponent coalition in Experiment 9. Strengthen embedded singleton dissent only where the dissenter is sufficiently credible relative to the anchor coalition; do not broaden it to ordinary two-versus-four profiles, given the remaining Experiment 4 crossover.

For Experiment 4, tighten the existing ordinary-coalition compression slightly or cap the marginal contribution of the strongest non-anchor cue, rather than changing order use or the isolated-anchor branches. Finally, increase separation among the stable configuration types specifically in one- and two-opponent challenge strength and restoration thresholds, and adjust their mixture weights. The current type gains do not produce the observed heterogeneity. Avoid merely widening all independent ranges, which could damage the near-unanimous Experiment 1 pattern. These are localized calibrations within PDRR, not a change of mechanism family.

**Outcome of your advice:** iter 4 candidate loss=0.1886 -> iter 5 candidate loss=0.1677 -> the gate marked it ACCEPTED.

### Iteration 5
Verdict: regenerate
Interpretation: The current candidate is an accepted aggregate improvement and remains faithful to the prescribed configuration-dependent, history-invariant PDRR family. The latest localized changes were productive: Experiment 9 is now essentially correct in mean (0.7325 vs. 0.7250), Experiment 10 is close (0.4612 vs. 0.4462), Experiment 4 improved to 0.0800, and the tied-top order prediction in Experiment 5 remains nearly exact. Experiment 7 is also acceptably near zero, and no spurious exposure-learning mechanism has appeared. However, the core cross-experiment signature is still not captured. Experiment 2 remains dramatically too anchor-following (0.5867 vs. 0.3242), and the large negative one-versus-multiple contrast in Experiment 6 is still only about half its observed magnitude (-0.2650 vs. -0.5450). Experiment 3 has also moved away from its earlier near-exact fit (0.1367 vs. 0.0721), while Experiment 8 now has excessive configuration-dependent profile variance (0.0238 vs. 0.0070). The model also continues to underproduce subject heterogeneity in the central conditions: variances are 0.0036 vs. 0.0174 in Experiment 2, 0.0127 vs. 0.0462 in Experiment 6, 0.0148 vs. 0.0723 in Experiment 9, and 0.0062 vs. 0.0224 in Experiment 10. Thus the mean fit has improved substantially, but the candidate is not yet experiment-invariant.
Rationale: Continue from this accepted candidate, preserving history invariance, the tied-top order gate, the current exactly-two-opponent calibration, and the strong restoration branch for three or more opponents. The previous advice in this direction was accepted, and Experiment 9 and Experiment 10 should not be disturbed by another global challenge/restoration shift.

The next adjustment should isolate the one-opponent branch. Experiment 9 shows that exactly-two opposition is now correctly calibrated, whereas Experiment 6 still requires much lower anchor adherence with one opponent. Increase the one-opponent compact challenge amplitude—or make its credibility gate somewhat less suppressive—without changing the two-opponent branch. Keep restoration nearly deterministic for large coalitions across subject types. This should enlarge the negative Experiment 6 contrast and lower at least the compact one-opponent portion of Experiment 2 while protecting Experiments 1, 9, and the two-opponent compact component of Experiment 10.

Do not continue strengthening ordinary validity compression globally: although it helped Experiment 4, Experiment 3 has now overshot and Experiment 4 is already close enough. Likewise, reduce the generic embedded `minority_amplification` if condition-level checks show it drives Experiment 8's excessive profile variance, while retaining the separately localized broad-coalition `singleton_bonus` needed for the five-versus-one diagnostic in Experiment 10. This separates ordinary singleton effects from broad singleton dissent rather than moving both together.

Most importantly, calculate or target Experiment 2's constituent classes separately—isolated one-opponent, isolated two-opponent, restored three-opponent, broad singleton dissent, and ordinary balanced coalitions. Its pooled value has barely responded despite several accepted mechanism changes, so another pooled parameter shift is unlikely to work. Tune the existing class-specific gains based on which class remains excessively anchor-following; do not introduce a new mechanism family.

Finally, increase heterogeneity through mean-preserving stable mixtures rather than wider independent ranges. Use strongly separated one-opponent challenge types to raise variance in Experiments 2 and 6. For Experiment 9, introduce greater type-specific variation in the balance between exactly-two challenge and restoration while holding their mixture-weighted mean near 0.725. Keep large-count restoration relatively invariant across types to protect Experiment 1, and avoid type effects on the matched Experiment 7 configurations. This is a narrower extension of the accepted direction, not a reversal or a return to the previously rejected global threshold changes.

**Outcome of your advice:** iter 5 candidate loss=0.1677 -> iter 6 candidate loss=0.1123 -> the gate marked it ACCEPTED.

### Iteration 6
Verdict: regenerate
Interpretation: Iteration 6 is the strongest accepted candidate so far and is faithful to the prescribed history-invariant, configuration-dependent selective-integration family. The previous recommendation was productive: isolating the one-opponent challenge reduced Experiment 6's error substantially (-0.3928 versus -0.5450), and the new balanced-opposition channel finally moved Experiment 2 toward its low observed anchor adherence (0.4267 versus 0.3242). Experiment 8 has returned to a close fit, while Experiments 5, 7, and 9 remain good. However, the balanced-opposition adjustment appears too indiscriminate. Experiment 3 has flipped from the previously correct positive interaction to a negative one (-0.0617 versus 0.0721), indicating excessive challenge to the unique validity anchor specifically in balanced 2-versus-2 profiles. Experiment 4 also worsened from 0.08 to 0.20 spurious crossover prevalence, plausibly because the unconditional bonus makes balanced profiles overly sensitive to which coalition contains an early cue. Thus, the same change that repaired Experiment 2 damaged other balanced-coalition experiments. Experiment 2 also remains too anchor-following by 0.1025, and Experiment 6 still captures only about 72% of the observed negative contrast. Experiment 1 is somewhat weak at 0.92 versus unanimous observed prevalence, and Experiment 10 is modestly too anchor-following. Subject-level heterogeneity remains substantially compressed in Experiments 2, 6, 9, and 10 despite the stronger type separation.
Rationale: Build on this accepted candidate and retain the successful one-opponent branch, exactly-two calibration, large-count restoration, tied-top order gate, history invariance, and localized singleton treatment. The latest accepted direction should be refined rather than reversed.

The priority is to replace the unconditional `balanced_opposition_bonus` with a bounded validity-geometry gate. At present every non-isolated, unique-anchor balanced profile receives the same categorical opposition boost. Make that boost strong only when the opposing coalition contains individually credible cues and its reliability gap from the anchor is small, or when its redundancy-discounted support is already near the anchor coalition's support. Suppress it when the unique anchor is clearly stronger than every opponent. This should preserve the Experiment 2 gain while preventing the overchallenge that reversed Experiment 3. A bounded coalition-breadth interaction may also help distinguish Experiment 2's broader balanced profiles, but it should not be a raw count-only bonus. Do not tighten ordinary validity compression globally; prior results show that this risks disturbing Experiment 3 and is not necessary for the current diagnosis.

The same gated balanced channel should reduce Experiment 4's renewed crossover. In particular, cap how much alignment of one early non-anchor cue can change the gate, rather than changing the order mechanism. Experiment 5 demonstrates that tied-top order use is already calibrated correctly.

For Experiment 6, take a smaller further step in the accepted one-opponent direction: modestly increase the challenge-sensitive type's one-opponent gain or further relax its credibility suppression, while leaving exactly-two and three-or-more branches fixed. This should move -0.3928 toward -0.5450 and should also lower the isolated one-opponent component of Experiment 2 without disrupting the good Experiment 9 fit. Avoid another global restoration-threshold change, since that strategy was previously rejected.

Finally, increase heterogeneity through more separated but mean-preserving stable type mixtures. The simulated variances remain far below observed values in Experiment 6 (0.0156 versus 0.0462) and Experiment 9 (0.0155 versus 0.0723), with similar compression in Experiments 2 and 10. Give the stable types more divergent one- and two-opponent challenge/restoration balances and tune their mixture probabilities to preserve current means; keep large-count restoration comparatively invariant to protect Experiment 1. These are localized calibrations within PDRR, not a change of mechanism family.

**Outcome of your advice:** iter 6 candidate loss=0.1123 -> iter 7 candidate loss=0.1073 -> the gate marked it ACCEPTED.

### Iteration 7
Verdict: regenerate
Interpretation: Iteration 7 is accepted and is the best aggregate fit so far, and it remains faithful to the prescribed history-invariant, configuration-dependent selective-integration family. The validity-geometry refinement successfully repaired damage from the unconditional balanced-opposition bonus: Experiment 3 returned near the observed positive interaction (0.0525 vs. 0.0721), Experiment 4 improved from 0.20 to 0.10 crossover prevalence, and Experiment 1 rose to 0.99. Experiments 5, 8, and 9 are also close in mean. However, this repair largely erased the previous gain on the most important unresolved condition: Experiment 2 regressed from 0.4267 to 0.5281 against an observed 0.3242. The strongest-opponent validity gate plus the reduced 2-vs-2 breadth factor is evidently suppressing distinctive opposition across too many of Experiment 2's balanced configurations. Experiment 6 remains materially too weak as well (-0.4072 vs. -0.5450), although the accepted one-opponent adjustment continues to move it in the correct direction. Experiment 10 is modestly too anchor-following (0.4775 vs. 0.4462), and Experiment 7 retains a small unwanted static contrast (-0.0294 vs. 0.0052). Most importantly, heterogeneity remains compressed in the central tests: variances are 0.0037 vs. 0.0174 in Experiment 2, 0.0215 vs. 0.0462 in Experiment 6, 0.0146 vs. 0.0723 in Experiment 9, and 0.0084 vs. 0.0224 in Experiment 10. Thus the model has a strong overall fit but has not yet achieved the prescribed cross-experiment dissociation between balanced opposition, compact challenge, and restoration.
Rationale: Build on this accepted candidate. The latest advice was accepted, so retain the validity-geometry gate rather than returning to the unconditional balanced bonus that damaged Experiments 3 and 4. The next adjustment should refine what counts as credible balanced opposition. The current gate uses only the strongest opponent and therefore misses distributed coalition credibility. Add a bounded interaction based on the number or proportion of opposing cues that are near the anchor in validity, or on redundancy-discounted opposition-to-anchor support parity. Let this distributed-credibility term become appreciable mainly in broader balanced coalitions, while keeping the present suppression for compact 2-vs-2 profiles with only one credible opponent. This should recover Experiment 2's lost reduction without recreating Experiment 3's sign reversal or Experiment 4's crossover. Do not simply enlarge `balanced_opposition_bonus` globally or remove its geometry gate.

Take one further small step along the accepted one-opponent direction: modestly increase the challenge-sensitive type's one-opponent gain or relax only that branch's credibility attenuation. Leave exactly-two and large-count restoration fixed. This should move Experiment 6 toward -0.545 and may improve the isolated one-opponent subset of Experiment 2 while preserving the good Experiment 1 and 9 means. Avoid a global restoration-threshold change, which was previously rejected.

Finally, make the stable types more behaviorally separated in a mean-preserving way. In particular, increase divergence in one-opponent challenge for Experiment 6 and in the exactly-two challenge/restoration balance for Experiment 9, compensating with type weights or opposing gain shifts so the current means remain near target. Correlating type-specific temperature or lapse with these configuration sensitivities may produce persistent subject differences more effectively than widening independent ranges. Keep large-count restoration relatively invariant across types and leave the tied-top order mechanism unchanged, since Experiments 1 and 5 are already essentially exact.

**Outcome of your advice:** iter 7 candidate loss=0.1073 -> iter 8 candidate loss=0.1238 -> the gate marked it REJECTED.

### Iteration 8
Verdict: regenerate
Interpretation: The current candidate remains faithful to the prescribed history-invariant PDRR mechanism family, but it was rejected by the accept gate and is worse than the iteration-7 base on several central tests. The new distributed-credibility term did not improve Experiment 2 at all: anchor following remains 0.5290 versus 0.3242. Meanwhile, the stronger type separation degraded Experiment 6 from -0.4072 to -0.3181, moved Experiment 9 farther above target from 0.7425 to 0.7675, and reduced Experiment 1 from 0.9900 to 0.9300. It also failed to improve the relevant heterogeneity consistently: Experiment 9 variance fell from 0.0146 to 0.0097 despite an observed 0.0723. The candidate still fits tied-top order in Experiment 5, profile variance in Experiment 8, and Experiment 10 reasonably well, while Experiment 3 remains acceptably positive and Experiment 7 remains only mildly non-null. The unresolved failures are therefore concentrated in Experiment 2's balanced subclasses and Experiment 6's one-opponent challenge, not in the overall mechanism family.
Rationale: Build on the unchanged iteration-7 accepted base. The most recent recommendation to add broad distributed credibility and increase type separation was rejected, so do not strengthen either adjustment. In particular, restore the iteration-7 degree of type separation and leave the exactly-two isolated restoration and large-count restoration branches fixed; those settings gave better means in Experiments 1, 6, and 9.

Use a different, more diagnostic calibration of Experiment 2. Its schedule contains distinct 1-vs-1, compact 2-vs-2, and broader balanced profiles, and the failed broad-coalition term could affect only part of that pooled metric. First obtain or internally target predictions for those subclasses. For isolated 1-vs-1 trials, add a modest type-independent challenge floor or baseline one-opponent gain, rather than increasing only the challenge-sensitive type. The last type-divergence edit perversely weakened the aggregate Experiment 6 contrast; a small challenge shared across types should lower one-opponent anchor adherence more reliably while leaving Experiment 9's exactly-two branch untouched.

For compact balanced 2-vs-2 trials, replace the strongest-opponent-only bonus with a bounded joint-credibility gate that requires both opposing cues to be near the anchor, or uses the weaker opponent's validity gap. This is distinct from the rejected broad distributed-mean term and may separate Experiment 2's credible pairs from Experiment 3 profiles where only one opponent is competitive. Keep the bonus suppressed when the second opponent is weak, protecting Experiment 3's positive interaction and limiting Experiment 4 crossover. Do not enlarge the balanced bonus globally.

Defer further heterogeneity expansion until these subclass means improve. The latest hard-coded gain separation increased Experiment 6 variance but damaged its mean and reduced Experiment 9 variance. Once the means are calibrated, mixture probabilities or temperature correlations can be adjusted in a mean-preserving manner. Preserve history invariance and the tied-top order gate, since Experiments 5 and 7 provide no reason to change them.

**Outcome of your advice:** iter 8 candidate loss=0.1238 -> iter 9 candidate loss=0.1204 -> the gate marked it REJECTED.

### Iteration 9 (most recent)
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed PDRR family—communicated-validity weighting, redundancy discounting, configuration-specific challenge/restoration, bounded tied-top order, stable heterogeneity, and no exposure learning—but the current revision was rejected and does not improve on iteration 7. The added one-opponent floor moved Experiment 6 in the desired direction (-0.4503 versus -0.5450), but it did not reduce pooled anchor following in Experiment 2; that value instead rose to 0.5429 versus 0.3242. The joint-credibility edit likewise failed its intended dissociation: Experiment 3 moved farther below its positive target (0.0356 versus 0.0721), Experiment 4 crossover prevalence rose to 0.14 versus 0, and Experiment 2 did not improve. Other means remain broadly serviceable: Experiments 1, 5, 7, 8, 9, and 10 are reasonably close, although Experiment 8 slightly underproduces profile variance and Experiment 9 is somewhat over-restored. The dominant failure is still Experiment 2, followed by an insufficiently negative Experiment 6 contrast. Subject heterogeneity also remains markedly compressed in Experiments 2, 6, 9, and 10. Thus the theory family remains plausible, but this parameterization is not experiment-invariant.
Rationale: Build the next revision on the unchanged iteration-7 accepted base. The two most recent proposals—broad distributed credibility and then a shared one-opponent floor plus compact joint-credibility bonus—were both rejected, so do not repeat or intensify them. In particular, remove the new joint-credibility modification and do not globally increase one-opponent challenge.

Use a different in-family discriminator that the current model discards: absolute anchor credibility. Cue weights are normalized to the strongest active cue, making every locally strongest cue have weight 1 regardless of whether it is a genuinely high-validity expert or merely the best among low-validity active experts. Experiment 2 contains several profiles whose first discriminating cue occurs later in the validity ordering; these lower-absolute-validity anchors should be more challengeable than the strong anchors in Experiments 6 and 9. Add a bounded absolute-validity gate—based on the anchor's communicated log-odds or validity, not just its gap from opponents—to amplify existing isolated challenge and embedded singleton distinctiveness when the active anchor is weak. Keep this gate near neutral for highly credible anchors and do not apply it as a general balanced-coalition bonus, which would risk Experiments 3 and 4. Internally score Experiment 2 by active-anchor rank and by isolated versus embedded configuration so this adjustment is calibrated to the subclasses actually responsible for its high pooled adherence.

Address the remaining Experiment 6 error from the other side of the contrast: sharpen only the count-dominant restoration branch for three or more opponents, making it more nearly type-invariant and closer to its ceiling. Leave one- and exactly-two-opponent branches unchanged. This should increase multi-opponent anchor adherence and make the contrast more negative without disturbing Experiment 9's two-opponent mean or reviving the failed global challenge shifts; it should also preserve Experiment 1's near-unanimous anchor-following pattern.

Preserve iteration 7's ordinary-coalition compression, validity-geometry gate, tied-top order settings, and history invariance. Defer further expansion of hard-coded type separation: the previous attempt was rejected and did not reliably increase variance. Once the Experiment 2 subclass means and Experiment 6 contrast improve, heterogeneity can be tuned through mean-preserving mixture probabilities or correlated temperature/lapse effects rather than larger configuration gains.

**Outcome of your advice:** iter 9 candidate loss=0.1204 -> CURRENT candidate loss=0.0914 -> the gate marked it ACCEPTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "Iteration 10 is an accepted and meaningful improvement, and the absolute-anchor-credibility direction recommended previously was productive at the aggregate level. The candidate remains faithful to the prescribed family: communicated-validity weighting, redundancy discounting, static configuration-specific challenge and restoration, bounded tied-top order use, stable subject effects, and no feedback-free learning. It fits Experiments 5, 8, and 9 very closely in mean, is good on Experiments 1, 3, 7, and 10, and improves the central Experiment 6 contrast relative to the iteration-7 accepted base. Nevertheless, it still misses a principal target of the arbiter recommendation. Experiment 2 remains far too anchor-following at 0.5231 versus 0.3242. The new absolute-credibility multiplier changed this pooled result only slightly because it currently affects isolated challenge and embedded singleton dissent but not the balanced 2-vs-2, 3-vs-3, and 4-vs-4 profiles that constitute much of Experiment 2. Experiment 6 also remains insufficiently negative (-0.4372 versus -0.5450), although the direction is qualitatively correct. Experiment 4 retains a spurious crossover prevalence of 0.12 versus zero. The model also underestimates critical between-subject heterogeneity, especially in Experiments 2, 6, 9, and 10; for example, Experiment 9 has variance 0.0124 versus 0.0723 despite an accurate mean. Thus the current model is the best PDRR instantiation so far, but its low-absolute-validity mechanism is connected to too few configuration branches to deliver the required experiment-invariant account.",
  "verdict": "regenerate",
  "rationale": "Build on iteration 10, since the latest advice was accepted. Preserve history invariance, tied-top order, ordinary-coalition compression, the exactly-two calibration that fits Experiment 9, and the count-dominant restoration branch. Do not return to the rejected unconditional balanced bonus, broad distributed-credibility term, or global one-opponent challenge floor.\n\nExtend the successful absolute-anchor-credibility variable in a narrowly targeted way. At present, weak absolute credibility amplifies only isolated challenge and embedded singleton dissent. Allow it also to modulate the already existing validity-geometry balanced-opposition channel when higher-validity cues are nondiscriminating and the locally strongest active cue is low in absolute communicated validity. This should be a bounded interaction\u2014weak absolute anchor multiplied by existing coalition geometry\u2014not a general balanced-coalition bonus. It directly targets Experiment 2's later-active-anchor balanced profiles while protecting Experiments 3 and 4 when their unique anchor is absolutely credible. Internally score Experiment 2 separately for its 1-vs-1, 2-vs-2, 3-vs-3, and 4-vs-4 profiles and by active-anchor rank; the tiny pooled improvement indicates that the current gate is reaching only a minority of the problematic trials.\n\nDo not further push large-count restoration blindly. The new branch improved Experiment 6 relative to the accepted iteration-7 base but remains near its ceiling and did not outperform the rejected iteration-9 value on that metric. First determine whether the residual Experiment 6 error comes from one-opponent adherence or multi-opponent adherence. If restoration is already saturated for three or more opponents, use only a small adjustment to the existing one-opponent credibility attenuation, not a shared challenge floor or a change to exactly-two opposition. Keep Experiment 9's mean near 0.725.\n\nAfter repairing these subclass means, increase heterogeneity through random effects on thresholds rather than larger hard-coded type gains, since the prior type-separation attempt was rejected. In particular, vary the absolute-anchor threshold/gain and the exactly-two restoration threshold across stable subjects, with distributions centered to preserve the fitted means. This can increase variance in Experiments 2 and 9 without moving every subject in the same direction. Avoid adding history dependence, and leave the tied-top order parameters unchanged because Experiment 5 is essentially exact."
}
```

## Usage

```json
{
  "input_tokens": 27625,
  "output_tokens": 2051
}
```
