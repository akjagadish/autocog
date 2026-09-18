# feedback_iter_07

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
- THEORY 1 = `pi_1`
- THEORY 2 = `pi_3`
- The recommendation below acts on THEORY 2 (= `pi_3`).

Replace pi_3 with a dominance-gated evidence-integration theory. The new theory should propose that people first assess whether the communicated validity profile contains a sufficiently dominant cue. When such a cue is recognized, choices are anchored on it and lower-validity cues exert only weak, approximately order-invariant influence, producing the near invariance in Experiment 1 and no reliable subject-level crossover in Experiment 2. When no cue clears the subject's dominance threshold, signed reliability evidence from multiple discriminating cues is accumulated, with bounded or saturating contributions and possible redundancy discounting; this compensatory mode should allow collections of lower-validity cues to overturn the highest-validity discriminating cue, as required by Experiment 4. Serial position and the final cue should not have intrinsically directional effects: conflict may increase noise, deliberation, or switching between dominance and integration modes, but it should not automatically open a recency gate. Stable subject traits could include the dominance threshold, sensitivity to validity gaps, degree of evidence saturation or redundancy discounting, and response noise. The theory should predict continuous, non-extreme effects and permit heterogeneity rather than hard-coding crossovers throughout the parameter range. It differs from pi_1 by allowing genuine multi-cue compensation and differs from pi_3 by making strategy selection depend on validity dominance and evidence structure rather than obligatory primacy and conflict-triggered recency. Its model should be calibrated jointly to yield an approximately zero Experiment-1 interaction, zero prevalence above the 0.12 Experiment-2 crossover threshold, near-unanimous first-cue agreement in Experiment 3 when dominance conditions support it, and approximately 0.32 highest-cue alignment in Experiment 4.


## CANDIDATE THEORY
Dominance-Calibrated Saturating Coalitions with Elbow Recognition proposes that people first inspect the shape of the communicated validity ranking. A dominance regime is activated only when the drop from the strongest cue to the second-strongest cue is unusually large relative to the ordinary validity gaps among the remaining cues. This feature-count-invariant validity elbow signals that one expert deserves privileged status. In that regime, the highest-validity discriminating cue anchors choice and lower-validity cues exert only weak, order-invariant influence. When the validity profile lacks a clear elbow, people instead integrate signed reliability evidence across all discriminating cues. Reliability differences are compressed and repeated concordant evidence has diminishing marginal impact, while a compensatory-specific evidence gain allows lower-validity coalitions to decisively overturn the best cue. Serial position has no directional role. Stable individual differences concern elbow threshold, sensitivity to the validity-profile transition, reliability compression, redundancy saturation, coalition gain, anchor leakage, conflict-related deliberation, and response sensitivity.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"DCSC expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.shape[0] != n_features:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {n_features}."
        )

    # Positive signs favor B and negative signs favor A. Presentation
    # position never enters the evidence weights.
    signs = np.sign(stim[1] - stim[0])
    discriminating = np.flatnonzero(signs != 0)
    if discriminating.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    dominance_threshold = float(parameters["dominance_threshold"])
    gap_sensitivity = float(parameters["gap_sensitivity"])
    validity_compression = float(parameters["validity_compression"])
    redundancy_saturation = float(parameters["redundancy_saturation"])
    anchor_strength = float(parameters["anchor_strength"])
    dominance_leak = float(parameters["dominance_leak"])
    compensatory_gain = float(parameters["compensatory_gain"])
    conflict_deliberation = float(parameters["conflict_deliberation"])
    beta = float(parameters["beta"])

    # Communicated validities are converted to nonnegative diagnostic
    # strengths. The small exponent compresses differences in the
    # compensatory regime without reversing the communicated ordering.
    diagnosticity = np.clip(2.0 * validities - 1.0, 1e-6, 1.0)
    cue_weights = diagnosticity ** validity_compression

    # Dominance recognition depends on the shape of the ranked validity
    # decline. A cue is dominant when the top-to-second diagnosticity gap
    # forms an elbow relative to the RMS size of gaps in the remainder of
    # the profile. Unlike top share or HHI, this statistic is insensitive
    # to feature count and common scaling of diagnosticity.
    ranked = np.sort(diagnosticity)[::-1]
    if n_features == 1:
        elbow_score = 1.0
    else:
        top_gap = float(max(ranked[0] - ranked[1], 0.0))
        if n_features > 2:
            tail_gaps = np.maximum(ranked[1:-1] - ranked[2:], 0.0)
            typical_tail_gap = float(np.sqrt(np.mean(tail_gaps ** 2)))
        else:
            typical_tail_gap = 0.0
        gap_scale = top_gap + typical_tail_gap
        elbow_score = top_gap / gap_scale if gap_scale > 1e-12 else 0.0

    gate_logit = gap_sensitivity * (
        elbow_score - dominance_threshold
    )
    gate_logit = float(np.clip(gate_logit, -60.0, 60.0))
    dominance_gate = 1.0 / (1.0 + np.exp(-gate_logit))

    # Aggregate reliability separately for the two coalitions. The
    # exponential transform gives diminishing marginal impact to repeated
    # concordant evidence and is invariant to the order in which cues are
    # displayed.
    support_b = float(np.sum(cue_weights[(signs > 0)]))
    support_a = float(np.sum(cue_weights[(signs < 0)]))
    rho = max(redundancy_saturation, 1e-8)
    bounded_b = -np.expm1(-rho * support_b) / rho
    bounded_a = -np.expm1(-rho * support_a) / rho

    # Express coalition evidence in approximately cue-vote units so the
    # response scale remains comparable across validity profiles.
    mean_discriminating_weight = float(np.mean(cue_weights[discriminating]))
    coalition_evidence = (bounded_b - bounded_a) / max(
        mean_discriminating_weight, 1e-12
    )

    # The anchor is the most valid cue that actually discriminates. Ties
    # in communicated validity are resolved without using serial position:
    # their signed evidence is averaged, so tied opposing anchors cancel.
    disc_weights = diagnosticity[discriminating]
    best_weight = float(np.max(disc_weights))
    best_mask = np.isclose(disc_weights, best_weight, rtol=1e-10, atol=1e-12)
    anchor_direction = float(np.mean(signs[discriminating][best_mask]))
    anchor_evidence = anchor_strength * anchor_direction

    # In dominance mode coalition evidence only leaks weakly into the
    # anchored judgment. In integration mode it is fully compensatory and
    # receives a branch-specific gain so coherent coalitions can be
    # consequential without sharpening dominance-mode choices.
    dominance_evidence = anchor_evidence + dominance_leak * coalition_evidence
    integration_evidence = compensatory_gain * coalition_evidence
    choice_evidence = (
        dominance_gate * dominance_evidence
        + (1.0 - dominance_gate) * integration_evidence
    )

    # Balanced conflict can prompt more careful implementation of the
    # selected rule, but it supplies no directional or last-cue evidence.
    count_b = int(np.sum(signs > 0))
    count_a = int(np.sum(signs < 0))
    balance = np.exp(-abs(count_b - count_a))
    beta_eff = beta * (1.0 + conflict_deliberation * balance)

    utilities = np.array([-0.5 * choice_evidence, 0.5 * choice_evidence])
    logits = beta_eff * utilities
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs /= probs.sum()
    return probs.astype(np.float64)

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

`parameters`:
- validities: validities
- dominance_threshold: [0.58, 0.62]
- gap_sensitivity: [70.0, 110.0]
- validity_compression: [0.02, 0.18]
- redundancy_saturation: [0.08, 0.25]
- anchor_strength: [0.8, 1.2]
- dominance_leak: [0.02, 0.10]
- compensatory_gain: [2.0, 4.0]
- conflict_deliberation: [0.30, 0.60]
- beta: [2.8, 4.8]

`rationale`:
This is a minimal edit of the accepted iteration-3 model. The additive top-share-plus-HHI gate is replaced by the critic's proposed rank-gap elbow statistic: the top diagnosticity gap is normalized by the RMS gap among the remaining ranked cues. This tests whether the strongest cue is structurally exceptional rather than merely owning a large share of total reliability. The narrow threshold and retained steep slope are intended to classify a clear-elbow Experiment-1 profile consistently as dominant while sending a gradual Experiment-2 profile into compensation. The rejected coverage, signed-coalition veto, runner-up reserve, and conjunctive share gates are not reused. A single compensatory_gain parameter is added because gate correction alone cannot help if integration has the right trialwise direction but insufficient decisiveness; critically, this gain operates only outside dominance mode, leaving Experiment-1 anchoring and the weak symmetric leakage that protects Experiment 4 unchanged. All evidence remains order-invariant, and no named cue, serial position, or coalition membership enters the gate.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.4999 -> ACCEPTED
- iter 2: loss=0.2901 -> ACCEPTED
- iter 3: loss=0.2889 -> ACCEPTED
- iter 4: loss=0.3472 -> REJECTED
- iter 5: loss=0.2952 -> REJECTED
- iter 6: loss=0.7795 -> REJECTED
- iter 7: loss=0.4038 -> REJECTED
- iter 8 (current candidate you are grading): loss=0.5130 -> REJECTED
Running-best (last accepted) base: iter 3 at loss=0.2889.

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
  - iter 1: 0.1200 (var=0.1056) (Δ vs real -0.8800)
  - iter 2: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 3: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 4: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 5: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 6: 0.0000 (var=0.0000) (Δ vs real -1.0000)
  - iter 7: 0.5400 (var=0.2384) (Δ vs real -0.4600)
  - iter 8 (current): 0.1000 (var=0.0900) (Δ vs real -0.9000)
**Other theories' values on this metric (for reference):**
- pi_1: 0.9900 (var=0.0049)
- pi_2: 0.0000 (var=0.0000)
- pi_3: 1.0000 (var=0.0000)

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
  - iter 1: 0.8244 (var=0.0086) (Δ vs real +0.5002)
  - iter 2: 0.9933 (var=0.0001) (Δ vs real +0.6692)
  - iter 3: 0.9954 (var=0.0001) (Δ vs real +0.6712)
  - iter 4: 0.8054 (var=0.0107) (Δ vs real +0.4813)
  - iter 5: 0.9058 (var=0.0024) (Δ vs real +0.5817)
  - iter 6: 0.8406 (var=0.0006) (Δ vs real +0.5165)
  - iter 7: 0.9654 (var=0.0012) (Δ vs real +0.6413)
  - iter 8 (current): 0.8946 (var=0.0104) (Δ vs real +0.5704)
**Other theories' values on this metric (for reference):**
- pi_2: 0.4848 (var=0.0022)
- pi_1: 0.8492 (var=0.0121)
- pi_3: 0.3298 (var=0.0054)

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
  - iter 1: -0.0004 (var=0.0001) (Δ vs real -0.0725)
  - iter 2: 0.0023 (var=0.0001) (Δ vs real -0.0698)
  - iter 3: 0.0044 (var=0.0001) (Δ vs real -0.0677)
  - iter 4: -0.1333 (var=0.0041) (Δ vs real -0.2054)
  - iter 5: 0.0196 (var=0.0010) (Δ vs real -0.0525)
  - iter 6: 0.0050 (var=0.0001) (Δ vs real -0.0671)
  - iter 7: 0.0046 (var=0.0001) (Δ vs real -0.0675)
  - iter 8 (current): 0.0060 (var=0.0001) (Δ vs real -0.0660)
**Other theories' values on this metric (for reference):**
- pi_1: -0.0008 (var=0.0015)
- pi_3: -0.3629 (var=0.0033)
- pi_2: -0.1762 (var=0.0033)

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
  - iter 1: 0.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 2: 0.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 3: 0.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 4: 0.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 5: 0.0600 (var=0.0564) (Δ vs real +0.0600)
  - iter 6: 0.9800 (var=0.0196) (Δ vs real +0.9800)
  - iter 7: 0.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 8 (current): 0.0000 (var=0.0000) (Δ vs real +0.0000)
**Other theories' values on this metric (for reference):**
- pi_3: 1.0000 (var=0.0000)
- pi_1: 0.0400 (var=0.0384)
- pi_2: 0.0600 (var=0.0564)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed dominance-gated, order-invariant integration family, and it succeeds on Experiment 4: crossover prevalence is exactly 0.0000 with zero variance, correctly eliminating pi_3's obligatory positional crossover. However, the central regime selection is badly calibrated. In Experiment 1, humans are unanimously above chance in following the first/highest-validity discriminating cue (1.0000, variance 0), whereas DCSC produces only 0.1200 with substantial heterogeneity (variance 0.1056). Conversely, in Experiment 2, where compensation should reduce highest-cue alignment to about 0.3242, the candidate remains strongly anchor-aligned at 0.8244. Thus, the model is effectively assigning too much integration to the Experiment-1 profile and too much dominance to the Experiment-2 profile—the reverse of the required regime pattern. Experiment 3 is a smaller miss: the predicted interaction is essentially zero (-0.0004) rather than the observed modest positive 0.0721, although it is much closer than the directional effects of pi_2 and pi_3.
Rationale: Retain the prescribed mechanism family, but revise the dominance-gate calibration and its profile normalization. The current normalized top-versus-second gap does not discriminate the experimental validity profiles in the required direction. Do not simply increase dominance globally: that might repair Experiment 1 while worsening the already excessive 0.8244 alignment in Experiment 2. Instead, use an in-family profile-sensitive dominance statistic—such as a calibrated combination of absolute top-second separation, the top cue's share relative to the remaining reliability mass, or a different normalization of the validity gap—and set its threshold/slope so Experiment 1 is consistently in the dominance regime while Experiment 2 is predominantly compensatory. The Experiment-1 variance also indicates that the gate is too soft or straddles subjects' thresholds there; once profiles are correctly ordered, tighten the threshold range or steepen the gate to obtain near-unanimous classification. Preserve weak, order-invariant leakage so Experiment 4 remains at zero crossover. After fixing regime selection, modestly tune conflict deliberation or response sensitivity to raise Experiment 3's balanced-minus-imbalanced effect toward +0.07 without adding directional recency.

**Outcome of your advice:** iter 1 candidate loss=0.4999 -> iter 2 candidate loss=0.2901 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The accepted revision substantially improves aggregate fit and remains faithful to the prescribed dominance-gated, order-invariant integration family. It completely repairs Experiment 1, moving from 0.1200 with high heterogeneity to the observed 1.0000 with zero variance, while preserving the exact zero crossover prevalence in Experiment 4. However, it does so by assigning dominance far too broadly. Experiment 2 is now the principal failure: highest-validity-cue alignment rises from 0.8244 to 0.9933, whereas humans show only 0.3242. Thus, the new concentration gate successfully identifies Experiment 1 as dominant but incorrectly places Experiment 2 in the same near-deterministic regime. Experiment 3 remains a secondary miss: 0.0023 versus the observed modest positive 0.0721, showing that the stronger conflict-deliberation setting has almost no behavioral leverage under the current evidence calculation.
Rationale: Keep the accepted concentration-based improvement, but do not further increase global dominance, gate steepness, or response sensitivity. The prior advice was helpful overall, yet using strongest-cue share alone is insufficient because it collapses the Experiment-1 and Experiment-2 profiles into the same regime. Add a second in-family profile-shape term to the gate—such as effective cue count/entropy, runner-up mass, or the distribution of diagnosticity within the remainder—and calibrate the resulting gate explicitly so its score satisfies dominance(Experiment 1) > threshold > dominance(Experiment 2). Inspect these profile-level gate values directly before simulation. Also verify with a dominance-gate ablation that the existing compensatory branch can produce approximately 0.32 alignment in Experiment 2; if it cannot, strengthen lower-cue coalition impact through slightly greater validity compression or weaker redundancy discounting rather than changing mechanism family. Preserve weak order-invariant leakage to protect Experiment 4's exact zero crossover. Only after Experiment 2 is moved into compensation should Experiment 3 be tuned: the failed increase in conflict deliberation should not simply be repeated. Instead, use a smaller adjustment to how balance modulates evidence saturation or response temperature, contingent on the compensatory branch, to seek the modest +0.07 interaction without directional recency.

**Outcome of your advice:** iter 2 candidate loss=0.2901 -> iter 3 candidate loss=0.2889 -> the gate marked it ACCEPTED.

### Iteration 3
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed dominance-gated, order-invariant evidence-integration family and preserves two important successes: Experiment 1 is matched exactly at 1.0000 with zero variance, and Experiment 4 retains the required zero crossover prevalence. The current revision was accepted, but its aggregate improvement is extremely small and the profile-shape addition did not accomplish its main purpose. Experiment 2 actually worsened from 0.9933 to 0.9954 highest-cue alignment, versus the observed 0.3242. The model is therefore still placing this experiment in an almost deterministic dominance regime; its near-zero between-subject variance also contrasts with the observed variance of 0.0174. Experiment 3 improves only marginally to 0.0044 and remains well below the observed positive interaction of 0.0721. Thus, the candidate captures order invariance but not the required experiment-dependent transition from dominance to compensation.
Rationale: Do not further strengthen the HHI term, gate slope, anchor, or global response sensitivity. The previously recommended profile-shape modification was technically accepted by the loss gate, but it failed its intended diagnostic test: Experiment 2 remained fully dominant. Before another simulation, calculate and report the actual dominance_score and dominance_gate for every experiment across the parameter bounds. Replace or rescale the HHI contribution only if those values demonstrate the required ordering, and calibrate the threshold between the Experiment-1 and Experiment-2 scores rather than assuming that positive HHI weighting separates them. If the complete-profile scores cannot be separated, retain the same mechanism family but let the gate incorporate the trial's discriminating evidence structure—for example, the strongest discriminating cue's diagnosticity relative to the opposing discriminating reliability mass—while preserving profile-level dominance as the prior gate input. Then ablate the gate by setting dominance_gate=0 for Experiment 2. If the compensatory branch itself does not approach 0.32 alignment, reduce redundancy saturation and/or further equalize bounded cue contributions so coalitions of lower-validity cues can genuinely overturn the highest cue; merely softening the gate will not be sufficient. Use a less extreme gate slope or a somewhat wider threshold distribution around the calibrated boundary to recover the observed Experiment-2 heterogeneity without jeopardizing Experiment 1. Preserve the order-invariant leakage that gives Experiment 4 its exact fit. Tune Experiment 3 only after this regime-selection failure is repaired; the earlier increases in generic conflict deliberation had little leverage and should not simply be repeated.

**Outcome of your advice:** iter 3 candidate loss=0.2889 -> iter 4 candidate loss=0.3472 -> the gate marked it REJECTED.

### Iteration 4
Verdict: regenerate
Interpretation: The diagnostic-coverage gate is faithful to the prescribed dominance-gated integration family, but this instantiation is not experiment-invariant and was rejected by the loss gate. It preserves the exact Experiment-1 result (1.0000, variance 0) and the zero Experiment-4 crossover prevalence. It also lowers Experiment-2 highest-cue alignment from 0.9954 to 0.8054 and increases its heterogeneity toward the data, showing that exposing the compensatory branch has some leverage. Nevertheless, 0.8054 remains far above the observed 0.3242, so the integration branch still does not permit lower-validity coalitions to overturn the highest cue often enough. More importantly, the coverage gate damages Experiment 3: the interaction moves from 0.0044 to -0.1333, opposite the observed +0.0721. Coverage count therefore groups trials by sparsity rather than by whether their evidence actually warrants dominance, and moving Experiment 3 wholesale into compensation exposes a systematic anti-Expert-5 effect. The Experiment-4 metric establishes absence of a positional crossover, not by itself the requested approximately 0.32 highest-cue alignment.
Rationale: Build from the unchanged accepted iteration-3 base. The previous push toward a trial-coverage gate was rejected, so do not retry it by merely shifting the coverage threshold or slope. Use a different in-family calibration: define dominance accessibility from a validity-based dominance reserve, such as the strongest cue's bounded diagnostic strength relative to the bounded reliability mass available to the remainder or to the opposing coalition. Combine that reserve with the existing profile prior, and inspect its values in all four experiments before simulation. The gate must keep Experiment 1 decisively dominant while selectively allowing compensation in Experiment 2 without forcing Experiment 3 wholesale into the same regime. Separately ablate the compensatory branch on Experiment 2. Its current 0.8054 alignment indicates that gating alone cannot reach 0.3242. Reduce redundancy saturation substantially toward an approximately additive limit and, if needed, compress validity differences further or rescale coalition evidence so multiple concordant lower cues can reliably overturn the best cue. Calibrate this branch before blending it with dominance. Avoid increasing anchor strength, gate steepness, beta, or generic conflict deliberation; those knobs favor excessive highest-cue alignment, and conflict-dependent temperature only sharpens the wrong sign in Experiment 3. Preserve order-invariant leakage and verify Experiment 4's zero-crossover result after tuning.

**Outcome of your advice:** iter 4 candidate loss=0.3472 -> iter 5 candidate loss=0.2952 -> the gate marked it REJECTED.

### Iteration 5
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed dominance-gated, order-invariant integration family, but it was rejected by the accept gate and is worse overall than the iteration-3 base. Experiment 1 remains an exact success at 1.0000 with zero variance. The main failure is still Experiment 2: highest-validity-cue alignment is 0.9058 rather than 0.3242, with variance 0.0024 rather than 0.0174. Thus, even after stronger validity compression and nearly additive coalition accumulation, the model behaves predominantly as an anchor model and produces too little subject-level heterogeneity. Experiment 3 improves to 0.0196, but remains below the observed positive 0.0721. Experiment 4 also regresses from the accepted base's exact zero crossover prevalence to 0.0600. The runner-up-reserve profile statistic therefore does not provide the required experiment-specific regime separation, while globally reducing response sensitivity introduces noise without solving the compensatory failure.
Rationale: Build from the unchanged accepted iteration-3 base. The latest runner-up-reserve and near-additive-compression package was rejected, so do not repeat it or push compression, saturation, or globally lower beta still further. The new direction should be a selective, order-invariant trial-level veto on an otherwise profile-level dominance gate. Compute the strongest discriminating cue's bounded strength and compare it directly with the bounded opposing coalition strength; attenuate the profile dominance gate smoothly when the opposing coalition is strong enough to challenge the anchor. This is not a cue-count or coverage gate: it is an evidence-structure check using signed reliability mass. It should permit compensation specifically on Experiment-2 trials where lower-validity cues form a sufficient opposing coalition, while leaving Experiment 1 dominant when the opposing reserve is weak. Use a soft veto temperature and heterogeneous veto threshold rather than changing global beta, so Experiment 2 can acquire the observed variance without degrading all experiments. Before simulation, report profile-gate values and the trial-level anchor-versus-opposition reserve distributions for all experiments, and verify in an integration-only ablation that the affected Experiment-2 trials actually favor the lower-cue coalition. Preserve the accepted base's weak, order-invariant dominance leakage and response scale to recover Experiment 4's zero crossover. Do not separately tune Experiment 3 yet; its modest positive movement is encouraging, but the dominant Experiment-2 error must be repaired first, after which a small compensatory-branch-specific balance modulation may be considered.

**Outcome of your advice:** iter 5 candidate loss=0.2952 -> iter 6 candidate loss=0.7795 -> the gate marked it REJECTED.

### Iteration 6
Verdict: regenerate
Interpretation: The current candidate is broadly faithful to the prescribed dominance-gated, order-invariant integration family, but the newly added trial-level counterevidence veto is decisively miscalibrated. It produces the worst loss in the trajectory and was rejected by the accept gate. Experiment 1 collapses from the accepted base's exact fit to 0.0000 versus the observed 1.0000: every subject is now below chance in following the first/highest-validity discriminating cue. Experiment 4 similarly collapses from zero crossover prevalence to 0.9800 versus the observed 0.0000. This indicates that the veto is activated systematically and condition-dependently: whether Expert 1 joins or opposes Expert 5 changes opposing mass and therefore changes mode accessibility, creating almost exactly the crossover the order-invariant theory was intended to eliminate. Yet this damage does not solve Experiment 2, where alignment remains 0.8406 rather than 0.3242 and variance remains much too low, 0.0006 versus 0.0174. Experiment 3 is essentially unchanged at 0.0050 versus 0.0721. Thus the veto overturns dominance in the wrong trials while leaving the principal high-alignment failure largely intact.
Rationale: Build from the unchanged accepted iteration-3 base. My previous recommendation to add a signed opposing-coalition veto was strongly rejected by the gate, so do not repeat it with a different veto threshold, softness, counterevidence-focus value, or saturation setting. Remove the trial-level multiplicative veto and the asymmetric reserve calculation; their dependence on coalition membership is what generates the catastrophic Experiment-4 crossover and reverses Experiment 1. Try a different in-family knob: make dominance classification depend only on the communicated validity profile, using a genuinely two-dimensional or conjunctive gate rather than another additive concentration score. For example, require both sufficient absolute top-cue diagnosticity and sufficient top-versus-remainder separation, with separate calibrated thresholds; do not collapse these dimensions into top share plus HHI. Compute these profile statistics for all experiments first and only simulate a specification for which Experiment 1 lies robustly inside the dominance region and Experiment 2 lies outside it across parameter bounds. Because a profile-only gate is constant across trials within an experiment, it can expose Experiment 2 to compensation without inducing the Expert-1-dependent crossover measured in Experiment 4. Retain the accepted base's symmetric, signed coalition integrator, weak dominance leakage, and response scale rather than again pushing compression, saturation, or global beta—the prior attempts in those directions were rejected. Perform an integration-only ablation for Experiment 2 before blending modes: if its alignment is not already near 0.32, adjust the normalization or add a compensatory-branch-specific evidence scale, not another asymmetric counterevidence rule. Allow heterogeneity through the two profile thresholds or gate temperature to approach Experiment 2's observed variance. Leave Experiment 3 tuning secondary until Experiments 1, 2, and 4 are simultaneously restored.

**Outcome of your advice:** iter 6 candidate loss=0.7795 -> iter 7 candidate loss=0.4038 -> the gate marked it REJECTED.

### Iteration 7 (most recent)
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed dominance-gated, order-invariant evidence-integration family, and it preserves the important Experiment-4 success: crossover prevalence is exactly 0.0000 with zero variance. However, the conjunctive profile gate fails its intended discrimination and the candidate was rejected by the accept gate. Experiment 1 regresses sharply from the accepted base’s exact 1.0000 fit to 0.5400, with substantial between-subject variance (0.2384 versus 0). Meanwhile, Experiment 2 remains almost completely anchor-dominated at 0.9654 highest-cue alignment rather than the observed 0.3242, with far too little heterogeneity. Thus, multiplying absolute-strength and top-share gates places the two profiles in essentially the wrong relative regions: Experiment 1 straddles the gate while Experiment 2 comfortably passes it. The compensatory branch consequently remains inaccessible where it is most needed. Experiment 3 also remains near zero (0.0046 versus 0.0721), indicating that conflict-dependent precision still has little leverage. The result is not experiment-invariant despite its correct lack of positional crossover.
Rationale: Build the next attempt on the unchanged accepted iteration-3 base. My previous recommendation to use this absolute-strength-by-share conjunctive gate was rejected, so do not retry it by merely shifting either threshold, changing the sigmoid slope, or widening their ranges. Likewise, do not return to the rejected signed trial-level veto, cue-coverage gate, runner-up reserve, or global compression/beta changes.

Try a new profile-only normalization that captures the shape of communicated validity decline rather than total reliability concentration—for example, a feature-count-invariant rank-gap or “elbow” statistic comparing the top-minus-second diagnosticity gap with the typical or summed gaps among the remaining ranked cues. This remains a dominance assessment within the prescribed family but is distinct from top share, HHI, raw runner-up reserve, and trial-level coalition membership. Before simulation, compute this statistic for every experiment over all proposed parameter bounds and require a robust ordering that puts Experiment 1 on the dominant side and Experiment 2 on the compensatory side; if it does not achieve that ordering, discard the statistic before running the model rather than compensating with extreme slopes.

In parallel, perform the repeatedly requested integration-only ablation for Experiment 2. If the symmetric coalition branch has the correct trialwise signs but choices are too weak, add a compensatory-branch-specific evidence gain or temperature while leaving anchor strength and dominance-mode noise unchanged. This can make lower-cue coalitions decisive without sacrificing Experiment 1 or creating the Experiment-4 crossover. If integration-only alignment cannot approach 0.32 at any gain, revise only its order-invariant normalization—such as scaling bounded coalition difference by total rather than mean discriminating support—because further gate tuning cannot solve a branch that lacks the required sign pattern. Preserve weak symmetric leakage and avoid any gate depending on which named expert joins a coalition. Treat Experiment 3 as secondary; after Experiments 1, 2, and 4 are jointly repaired, a small compensatory-mode modulation of saturation by conflict balance may be tested instead of another increase in generic conflict-dependent precision.

**Outcome of your advice:** iter 7 candidate loss=0.4038 -> CURRENT candidate loss=0.5130 -> the gate marked it REJECTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate remains broadly faithful to the prescribed dominance-gated, order-invariant integration family, and Experiment 4 remains an exact success at 0.0000 crossover prevalence with zero variance. Nevertheless, this iteration was rejected and is substantially worse than the accepted iteration-3 base. The elbow gate produces almost the reverse of the required regime allocation: Experiment 1 collapses from the accepted base\u2019s 1.0000 to 0.1000, with unwanted heterogeneity, indicating excessive compensation; Experiment 2 remains strongly anchor-aligned at 0.8946 rather than 0.3242, indicating excessive dominance. Thus, the elbow statistic does not separate the validity profiles in the required direction. The compensatory-specific gain cannot repair this because the integration branch remains largely inaccessible in Experiment 2. Experiment 3 is again effectively null, 0.0060 versus the observed modest positive 0.0721; conflict-dependent precision is not generating the relevant interaction.",
  "verdict": "regenerate",
  "rationale": "Build from the unchanged accepted iteration-3 base. My previous recommendation to try a rank-gap elbow was rejected decisively, so do not retry this statistic by adjusting its threshold, slope, RMS normalization, or compensatory gain. Also avoid the previously rejected coverage, runner-up-reserve, conjunctive-share, and signed coalition-veto gates.\n\nTry a genuinely different, profile-level dominance diagnostic based on coalition overturnability. From the communicated validity weights, calculate the distribution of possible signed evidence supplied by all non-top cues\u2014for example, the normalized probability that a randomly composed lower-cue coalition can exceed the top cue, or an upper quantile of signed lower-cue evidence relative to top-cue strength. Dominance should decrease as this overturnability rises. Unlike top share, HHI, runner-up mass, or an elbow, this statistic uses the full combinatorial distribution of lower-cue coalitions while remaining order-invariant and independent of the coalition realized on a particular trial. It therefore avoids the Expert-1-dependent mode switching that destroyed Experiment 4 under the signed veto.\n\nBefore simulation, compute this score for every experimental validity profile and require robust separation with Experiment 1 on the dominant side and Experiment 2 on the compensatory side across parameter bounds. Do not rescue a reversed ordering with an extreme sigmoid. If full-profile overturnability does not separate them, test a signed-direction-free version conditional only on which cues discriminate, while verifying that it neither creates an Experiment-4 crossover nor disrupts Experiment 1.\n\nRun an integration-only diagnostic for Experiment 2 across compensatory evidence scales. The current gain of 2\u20134 did not solve the problem, so do not merely increase it: retain a branch-specific gain only if the branch\u2019s trialwise directions can actually approach 0.32 alignment. First restore the Experiment-1 mean and zero variance and substantially reduce Experiment-2 alignment; then introduce threshold heterogeneity to approach Experiment 2\u2019s observed variance. Keep Experiment 3 secondary. Generic conflict-dependent precision has repeatedly produced an approximately zero interaction; after the main regime separation is fixed, a small balance-dependent shift toward dominance, rather than another temperature increase, may be tested to obtain the modest positive interaction without adding directional serial effects."
}
```

## Usage

```json
{
  "input_tokens": 13405,
  "output_tokens": 3471
}
```
