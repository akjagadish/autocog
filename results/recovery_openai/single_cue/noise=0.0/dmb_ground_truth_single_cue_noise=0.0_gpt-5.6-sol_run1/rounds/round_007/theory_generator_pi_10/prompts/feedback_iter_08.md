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
- THEORY 1 = `pi_3`
- THEORY 2 = `pi_9`
- The recommendation below acts on THEORY 2 (= `pi_9`).

Replace pi_9 with a stimulus-local bounded arbitration theory. The new theory should propose that people encode a small set of decision objects—an early high-validity commitment, reliability-weighted support from the currently visible coalition, and a privileged terminal reason or terminal run—and arbitrate among them on each trial. Terminal evidence should receive a genuine recency advantage, but its control should be moderated by current-stimulus conflict structure, relative coalition strength, and cue reliability rather than by a learned experiment-level overwrite state. Coherent added evidence should saturate and may produce only a mild vigilance decline, matching the small positive load contrast in Experiment 1 rather than pi_3's strong exponential count fatigue. History effects should be absent or weak carryover effects, consistent with Experiment 2, and should never be the principal source of terminal control. Allow stable subject-level variation in terminal susceptibility, coalition reliance, and lapse or attentional reliability to capture the elevated between-subject variance. This theory differs from pi_3 by rejecting obligatory cue-by-cue overwrite plus task-invariant count fatigue, and differs from pi_9 by rejecting susceptibility-dominant contextual learning and a continuously evolving accumulator as the primary explanation. Its critical predictions should include strong but nonuniform terminal control, near-zero inducer-history covariance, only modest decline under unanimous cue load, and limited effects of physical gaps or switch density unless those features change the perceived terminal run or coalition balance.


## CANDIDATE THEORY
Reliability-Gated Decision-Object Arbitration with Prefix-Coherence-Conditioned Terminal Access. People construct at most three bounded objects from each display: an initial high-validity commitment, a reliability-weighted summary of the visible coalition, and a privileged terminal reason or terminal directional run. Arbitration is stimulus-local and history-free. Terminal evidence has a genuine recency advantage, but weak terminal evidence receives reliable access only when it constitutes the sole coherent reversal against a unified prefix at manageable load. A terminal reason following an already fragmented prefix receives no such accessibility floor and may be mildly discounted. Very short conflicts provide too little separation for a strongly privileged counterargument, whereas seven-or-more-reason conflicts reduce access through object competition rather than general count fatigue. The initial commitment remains a weak but non-negligible fallback, coalition evidence saturates, and unanimous evidence experiences only a modest vigilance decline. Because endpoint cues already instantiate dedicated early and terminal objects, their contributions to the coalition summary are softly residualized to avoid counting the same evidence twice. Stable variation in coherent-terminal accessibility, terminal susceptibility, coalition reliance, attention, endpoint redundancy, and lapses produces heterogeneous conflict choices without learned context.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Decision-object arbitration expects shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    validity_compression = float(parameters["validity_compression"])
    early_commitment = float(parameters["early_commitment"])
    coalition_reliance = float(parameters["coalition_reliance"])
    endpoint_redundancy = float(parameters["endpoint_redundancy"])
    terminal_susceptibility = float(parameters["terminal_susceptibility"])
    reliability_threshold = float(parameters["terminal_reliability_threshold"])
    reliability_sharpness = float(parameters["terminal_reliability_sharpness"])
    coherent_access_floor = float(parameters["coherent_access_floor"])
    terminal_run_bonus = float(parameters["terminal_run_bonus"])
    terminal_singleton_bonus = float(parameters["terminal_singleton_bonus"])
    conflict_moderation = float(parameters["conflict_moderation"])
    attentional_reliability = float(parameters["attentional_reliability"])
    choice_gain = float(parameters["choice_gain"])
    vigilance_decline = float(parameters["vigilance_decline"])
    lapse_rate = float(parameters["lapse_rate"])

    # Communicated validity determines psychological inspection order. Exact
    # ties retain display order, but physical gaps otherwise have no role.
    cue_order = np.argsort(-validities, kind="stable")
    diff = stim[0] - stim[1]
    discriminating = [int(j) for j in cue_order if diff[j] != 0.0]
    m = len(discriminating)
    if m == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    directions = np.asarray(
        [float(np.sign(diff[j])) for j in discriminating], dtype=float
    )

    # Convert instructed validities to reliability strengths. Compression
    # preserves whether a validity profile is flat or steep without allowing
    # one extreme cue to create an unbounded accumulator.
    clipped_v = np.clip(validities, 0.5001, 0.999)
    log_odds = np.log(clipped_v / (1.0 - clipped_v))
    positive = log_odds[log_odds > 0.0]
    scale = float(np.mean(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    normalized = np.clip(log_odds / scale, 1e-4, 30.0)
    strengths_all = np.power(normalized, validity_compression)
    strengths = np.asarray(
        [float(strengths_all[j]) for j in discriminating], dtype=float
    )

    def sigmoid(x):
        x = float(np.clip(x, -60.0, 60.0))
        return 1.0 / (1.0 + np.exp(-x))

    # OBJECT 1: an early high-validity commitment. Its availability receives
    # a small boost only when the first reason is unusually reliable.
    first_direction = float(directions[0])
    first_strength = float(strengths[0])
    maximum_strength = max(float(np.max(strengths)), 1e-8)
    first_relative = first_strength / maximum_strength
    early_availability = 1.0 - np.exp(-first_strength)
    w_early = early_commitment * early_availability * (
        1.0 + 0.5 * first_relative
    )

    # OBJECT 2: a bounded reliability-weighted summary of the visible
    # coalition. Directional mass saturates; no sequential accumulator or
    # learned contextual state is constructed. The endpoint reasons already
    # instantiate dedicated early and terminal objects, so their coalition
    # contributions are softly residualized to prevent redundant voting.
    coalition_strengths = strengths.copy()
    coalition_strengths[0] *= 1.0 - endpoint_redundancy
    if m > 1:
        coalition_strengths[-1] *= 1.0 - endpoint_redundancy
    signed_support = float(np.sum(directions * coalition_strengths))
    total_support = max(float(np.sum(coalition_strengths)), 1e-12)
    coalition_balance = abs(signed_support) / total_support
    coalition_balance = float(np.clip(coalition_balance, 0.0, 1.0))
    coalition_direction = float(np.sign(signed_support))
    coalition_saturation = 1.0 - np.exp(-total_support / 2.0)
    if coalition_direction == 0.0:
        w_coalition = 0.0
    else:
        w_coalition = (
            coalition_reliance
            * coalition_saturation
            * (0.22 + 0.58 * coalition_balance)
        )

    # OBJECT 3: the final directional run in validity order. Run support is
    # represented as one chunk, and its length benefit rapidly saturates.
    terminal_direction = float(directions[-1])
    run_start = m - 1
    while run_start > 0 and directions[run_start - 1] == terminal_direction:
        run_start -= 1
    terminal_run_length = m - run_start
    terminal_run_strength = float(np.sum(strengths[run_start:]))
    run_saturation = 1.0 - np.exp(-terminal_run_strength)
    n_runs = 1 + int(np.sum(directions[1:] != directions[:-1]))

    # Terminal access depends on reliability relative to the strongest
    # currently visible reason. This creates nonuniform recency across
    # validity environments without learning an experiment-level overwrite
    # state. Even weak terminal reasons retain a small intrinsic privilege.
    terminal_relative = float(strengths[-1] / maximum_strength)
    terminal_access = sigmoid(
        (terminal_relative - reliability_threshold)
        / max(reliability_sharpness, 1e-6)
    )

    # A weak terminal counterargument is distinctly retrievable when it is a
    # short, sole reversal against a coherent prefix at manageable load. No
    # such floor is granted after a fragmented prefix; short terminal runs in
    # those displays receive only a mild local discount. Four-cue conflicts
    # and seven-plus coherent reversals are left unchanged.
    prefix = directions[:run_start]
    prefix_coherent = bool(
        prefix.size > 0
        and np.all(prefix == prefix[0])
        and prefix[0] != terminal_direction
    )
    short_terminal_run = terminal_run_length <= max(2, m // 2)
    if m in (5, 6) and prefix_coherent and short_terminal_run:
        singleton_boost = 0.08 * float(terminal_run_length == 1)
        terminal_access = max(
            terminal_access,
            min(0.92, coherent_access_floor + singleton_boost)
        )
    elif m >= 5 and terminal_run_length <= 2 and prefix.size > 1:
        prefix_fragmented = bool(np.any(prefix[1:] != prefix[:-1]))
        if prefix_fragmented:
            terminal_access *= 0.88

    reliability_gate = 0.08 + 0.92 * terminal_access

    # A coherent reversal is maximally accessible at manageable loads. The
    # smooth window weakens this privilege for very short conflicts and for
    # seven-or-more-reason overload without imposing general count fatigue.
    manageable_access = (
        sigmoid((float(m) - 3.5) / 0.35)
        * sigmoid((6.5 - float(m)) / 0.35)
    )
    geometry_multiplier = 1.0 + 0.25 * terminal_run_bonus * run_saturation
    if n_runs == 2:
        singleton = float(terminal_run_length == 1)
        coherent_access = 2.0 + terminal_singleton_bonus * singleton
        geometry_multiplier *= (
            1.0
            + terminal_run_bonus * manageable_access * coherent_access
        )
    elif n_runs >= 3:
        # Fragmented conflicts provide several competing retrieval boundaries;
        # their terminal reason remains privileged but is less diagnostic.
        fragmentation = min(2.0, float(n_runs - 2))
        geometry_multiplier /= (
            1.0 + 0.50 * terminal_run_bonus * fragmentation
        )
    w_terminal = (
        terminal_susceptibility
        * reliability_gate
        * (0.35 + 0.65 * attentional_reliability)
        * geometry_multiplier
    )

    # A conflicting coalition moderates, but cannot categorically erase, the
    # terminal object. Switch count and physical gaps do not enter this gate.
    if coalition_direction != 0.0 and coalition_direction != terminal_direction:
        opposing_advantage = coalition_balance * min(
            2.0, np.sqrt(float(m) / 2.0)
        )
        w_terminal /= 1.0 + conflict_moderation * opposing_advantage

    # Each object has a bounded probability of expressing its favored option.
    # Attentional reliability affects readout globally while the object
    # strengths determine confidence locally.
    early_confidence = sigmoid(
        choice_gain * attentional_reliability * (0.65 + 0.35 * early_availability)
    )
    coalition_confidence = sigmoid(
        choice_gain
        * attentional_reliability
        * (0.40 + 0.50 * coalition_balance)
    )
    terminal_confidence = sigmoid(
        choice_gain
        * attentional_reliability
        * (0.62 + 0.38 * run_saturation)
    )

    def p_a_from_object(direction, confidence):
        if direction > 0.0:
            return float(confidence)
        if direction < 0.0:
            return float(1.0 - confidence)
        return 0.5

    object_weights = np.asarray(
        [w_early, w_coalition, w_terminal], dtype=np.float64
    )
    object_predictions = np.asarray([
        p_a_from_object(first_direction, early_confidence),
        p_a_from_object(coalition_direction, coalition_confidence),
        p_a_from_object(terminal_direction, terminal_confidence)
    ], dtype=np.float64)

    weight_sum = float(np.sum(object_weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        p_a_core = 0.5
    else:
        p_a_core = float(np.dot(object_weights, object_predictions) / weight_sum)

    # Only coherent added evidence incurs a mild vigilance decrement. Because
    # all objects agree on such trials, evidence remains strong and the load
    # effect is much smaller than obligatory count fatigue.
    unanimous = bool(np.all(directions == directions[0]))
    extra_lapse = 0.0
    if unanimous:
        extra_lapse = vigilance_decline * min(
            1.0, max(0.0, float(m) - 2.0) / 8.0
        )
    effective_lapse = float(np.clip(lapse_rate + extra_lapse, 0.0, 0.45))

    p_a = (1.0 - effective_lapse) * p_a_core + effective_lapse * 0.5
    probs = np.array([p_a, 1.0 - p_a], dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
    return probs / total


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
- validities: validities
- validity_compression: [0.18, 0.42]
- early_commitment: [0.04, 0.22]
- coalition_reliance: [0.20, 0.85]
- endpoint_redundancy: [0.10, 0.35]
- terminal_susceptibility: [1.2, 8.3]
- terminal_reliability_threshold: [0.28, 0.62]
- terminal_reliability_sharpness: [0.06, 0.14]
- coherent_access_floor: [0.58, 0.78]
- terminal_run_bonus: [0.25, 0.95]
- terminal_singleton_bonus: [0.40, 1.00]
- conflict_moderation: [0.10, 0.65]
- attentional_reliability: [0.82, 0.99]
- choice_gain: [2.4, 4.8]
- vigilance_decline: [0.12, 0.25]
- lapse_rate: [0.03, 0.15]

`rationale`:
This is an isolated edit to the accepted iteration-7 model. All accepted terminal-access, prefix-coherence, geometry, load-window, confidence, vigilance, and history-free mechanisms are unchanged. The sole mechanistic addition is endpoint residualization within the coalition object: the first and final discriminating cues are softly downweighted when coalition support is computed because those cues already instantiate dedicated early and terminal objects. The new bounded endpoint_redundancy parameter applies independently of cue direction and does not alter either dedicated endpoint object. It should reduce redundant endpoint voting that contributes to excessive tally sensitivity in Experiment 1, first-cue influence in Experiments 2 and 9, terminal overexpression in fragmented Experiment 3 displays, and the mixed-versus-block contrast in Experiment 5. Dedicated terminal control is preserved for Experiments 4, 10, 11, and 13, while unanimous-load behavior, physical-gap invariance, switch-density invariance, and absent history learning remain unchanged.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.1988 -> ACCEPTED
- iter 2: loss=0.1247 -> ACCEPTED
- iter 3: loss=0.1168 -> ACCEPTED
- iter 4: loss=0.1215 -> REJECTED
- iter 5: loss=0.2062 -> REJECTED
- iter 6: loss=0.1315 -> REJECTED
- iter 7: loss=0.1013 -> ACCEPTED
- iter 8: loss=0.1182 -> REJECTED
- iter 9 (current candidate you are grading): loss=0.1056 -> REJECTED
Running-best (last accepted) base: iter 7 at loss=0.1013.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0, 1]
  A=[1, 1, 1, 1, 0, 1, 0]  B=[0, 0, 0, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 1, 0]  B=[1, 1, 1, 1, 0, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Contrast TTB-consistent choice rates between trials where the
    # equal-weight tally supports versus opposes the TTB winner.
    support_choices = []
    oppose_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        d = a - b

        differing = np.flatnonzero(d != 0)
        if differing.size == 0:
            continue

        # Features are already ordered from highest to lowest validity.
        ttb_direction = float(np.sign(d[differing[0]]))  # +1: A, -1: B
        ttb_response = 0 if ttb_direction > 0 else 1
        chose_ttb = float(int(row["response"]) == ttb_response)

        # Positive iff the tally margin points toward the TTB winner.
        tally_a_margin = float(np.sign(d).sum())
        oriented_tally_margin = ttb_direction * tally_a_margin

        if oriented_tally_margin > 0:
            support_choices.append(chose_ttb)
        elif oriented_tally_margin < 0:
            oppose_choices.append(chose_ttb)

    if len(support_choices) == 0 or len(oppose_choices) == 0:
        return float("nan")

    return float(np.mean(support_choices) - np.mean(oppose_choices))
```

**Observed (real) value:** 0.4944 (var=0.0204)
**Candidate trajectory (this loop):**
  - iter 1: 0.6935 (var=0.0126) (Δ vs real +0.1990)
  - iter 2: 0.7712 (var=0.0091) (Δ vs real +0.2768)
  - iter 3: 0.7196 (var=0.0100) (Δ vs real +0.2251)
  - iter 4: 0.7228 (var=0.0098) (Δ vs real +0.2283)
  - iter 5: 0.7317 (var=0.0115) (Δ vs real +0.2372)
  - iter 6: 0.6887 (var=0.0102) (Δ vs real +0.1943)
  - iter 7: 0.6940 (var=0.0120) (Δ vs real +0.1996)
  - iter 8: 0.7207 (var=0.0117) (Δ vs real +0.2263)
  - iter 9 (current): 0.7183 (var=0.0160) (Δ vs real +0.2239)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0000 (var=0.0049)
- pi_2: 0.7368 (var=0.0343)
- pi_3: 0.4743 (var=0.0168)
- pi_4: 0.5279 (var=0.0111)
- pi_5: 0.5949 (var=0.0094)
- pi_6: 0.6093 (var=0.0085)
- pi_7: 0.5681 (var=0.0095)
- pi_8: 0.5276 (var=0.0147)
- pi_9: 0.8585 (var=0.0048)

### Experiment 2
**Design**
  A=[1, 1, 0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1, 0, 1, 1]
  A=[1, 0, 1, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 1, 0, 0]  B=[1, 0, 1, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 1, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 0, 1, 1, 0]  B=[1, 0, 1, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 0, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Rate at which choices follow the highest-validity discriminating cue."""
    if len(data) == 0:
        return float("nan")

    matches = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)

        # Features are already ordered from highest to lowest validity.
        diff = a - b
        discriminating = np.flatnonzero(diff != 0)
        if discriminating.size == 0:
            # An all-tie pair provides no directional diagnosticity.
            continue

        first = int(discriminating[0])
        ttb_response = 0 if diff[first] > 0 else 1
        matches.append(float(int(row["response"]) == ttb_response))

    if not matches:
        return float("nan")
    return float(np.mean(matches))
```

**Observed (real) value:** 0.1938 (var=0.0049)
**Candidate trajectory (this loop):**
  - iter 1: 0.2825 (var=0.0081) (Δ vs real +0.0887)
  - iter 2: 0.1725 (var=0.0054) (Δ vs real -0.0213)
  - iter 3: 0.2583 (var=0.0109) (Δ vs real +0.0646)
  - iter 4: 0.2387 (var=0.0054) (Δ vs real +0.0450)
  - iter 5: 0.2454 (var=0.0045) (Δ vs real +0.0517)
  - iter 6: 0.2371 (var=0.0080) (Δ vs real +0.0433)
  - iter 7: 0.2294 (var=0.0067) (Δ vs real +0.0356)
  - iter 8: 0.2706 (var=0.0101) (Δ vs real +0.0769)
  - iter 9 (current): 0.2385 (var=0.0081) (Δ vs real +0.0448)
**Other theories' values on this metric (for reference):**
- pi_2: 0.4942 (var=0.0026)
- pi_1: 0.8694 (var=0.0103)
- pi_3: 0.1931 (var=0.0026)
- pi_4: 0.2060 (var=0.0019)
- pi_5: 0.1969 (var=0.0043)
- pi_6: 0.2027 (var=0.0056)
- pi_7: 0.1940 (var=0.0026)
- pi_8: 0.1765 (var=0.0015)
- pi_9: 0.3002 (var=0.0056)

### Experiment 3
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Proportion of choices agreeing with the least-valid discriminating cue."""
    if len(data) == 0:
        return float("nan")

    agreements = []
    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"]
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        differing = np.flatnonzero(a != b)
        if differing.size == 0:
            continue

        # Features are displayed in descending validity order, so the
        # largest differing index is the final discriminating reason.
        final_idx = int(differing[-1])
        final_cue_choice = 0 if a[final_idx] > b[final_idx] else 1
        agreements.append(float(int(response) == final_cue_choice))

    if not agreements:
        return float("nan")
    return float(np.mean(agreements))
```

**Observed (real) value:** 0.5750 (var=0.0026)
**Candidate trajectory (this loop):**
  - iter 1: 0.7667 (var=0.0053) (Δ vs real +0.1917)
  - iter 2: 0.8140 (var=0.0036) (Δ vs real +0.2390)
  - iter 3: 0.7677 (var=0.0085) (Δ vs real +0.1927)
  - iter 4: 0.7458 (var=0.0052) (Δ vs real +0.1708)
  - iter 5: 0.5954 (var=0.0065) (Δ vs real +0.0204)
  - iter 6: 0.7429 (var=0.0077) (Δ vs real +0.1679)
  - iter 7: 0.7435 (var=0.0062) (Δ vs real +0.1685)
  - iter 8: 0.7271 (var=0.0094) (Δ vs real +0.1521)
  - iter 9 (current): 0.7362 (var=0.0099) (Δ vs real +0.1613)
**Other theories' values on this metric (for reference):**
- pi_3: 0.7129 (var=0.0033)
- pi_2: 0.1462 (var=0.0111)
- pi_1: 0.1419 (var=0.0079)
- pi_4: 0.5346 (var=0.0080)
- pi_5: 0.5996 (var=0.0042)
- pi_6: 0.6577 (var=0.0054)
- pi_7: 0.7098 (var=0.0021)
- pi_8: 0.6569 (var=0.0032)
- pi_9: 0.4806 (var=0.0078)

### Experiment 4
**Design**
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Signed association between choice and the lowest-validity (fourth) cue.
    # +1 means the response follows that cue, -1 means it opposes that cue.
    products = []
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        terminal_direction = np.sign(a[-1] - b[-1])  # +1 favors A, -1 favors B
        if terminal_direction == 0:
            continue
        response = int(row['response'])
        choice_direction = 1.0 if response == 0 else -1.0
        products.append(choice_direction * terminal_direction)
    if len(products) == 0:
        return float('nan')
    return float(np.mean(products))
```

**Observed (real) value:** 0.6650 (var=0.0621)
**Candidate trajectory (this loop):**
  - iter 1: 0.5546 (var=0.0259) (Δ vs real -0.1104)
  - iter 2: 0.6479 (var=0.0091) (Δ vs real -0.0171)
  - iter 3: 0.6025 (var=0.0231) (Δ vs real -0.0625)
  - iter 4: 0.6421 (var=0.0231) (Δ vs real -0.0229)
  - iter 5: 0.3117 (var=0.0076) (Δ vs real -0.3533)
  - iter 6: 0.6004 (var=0.0152) (Δ vs real -0.0646)
  - iter 7: 0.6000 (var=0.0199) (Δ vs real -0.0650)
  - iter 8: 0.5617 (var=0.0290) (Δ vs real -0.1033)
  - iter 9 (current): 0.6038 (var=0.0190) (Δ vs real -0.0613)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0075 (var=0.0116)
- pi_3: 0.5262 (var=0.0122)
- pi_1: -0.2525 (var=0.0072)
- pi_4: 0.6104 (var=0.0504)
- pi_5: 0.6737 (var=0.0181)
- pi_6: 0.6704 (var=0.0140)
- pi_7: 0.6104 (var=0.0134)
- pi_8: 0.6408 (var=0.0101)
- pi_9: 0.2967 (var=0.0253)

### Experiment 5
**Design**
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 1, 1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    block_terminal = []
    mixed_terminal = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        d = a - b

        # The block permutation has one option favored by cues 1--5 and
        # the other option favored by cues 6--10.
        is_block = (
            d.size == 10
            and np.all(d[:5] == d[0])
            and np.all(d[5:] == d[5])
            and d[0] == -d[5]
        )

        # In this design the final two cues agree. Identify whether the
        # response selected the option favored by those terminal cues.
        terminal_response = 0 if d[-1] > 0 else 1
        chose_terminal = float(int(row["response"]) == terminal_response)

        if is_block:
            block_terminal.append(chose_terminal)
        else:
            mixed_terminal.append(chose_terminal)

    if len(block_terminal) == 0 or len(mixed_terminal) == 0:
        return float("nan")

    return float(np.mean(mixed_terminal) - np.mean(block_terminal))
```

**Observed (real) value:** 0.0033 (var=0.0049)
**Candidate trajectory (this loop):**
  - iter 1: 0.2300 (var=0.0191) (Δ vs real +0.2267)
  - iter 2: 0.0529 (var=0.0070) (Δ vs real +0.0496)
  - iter 3: 0.1692 (var=0.0250) (Δ vs real +0.1658)
  - iter 4: 0.1458 (var=0.0151) (Δ vs real +0.1425)
  - iter 5: -0.1350 (var=0.0358) (Δ vs real -0.1383)
  - iter 6: 0.2100 (var=0.0191) (Δ vs real +0.2067)
  - iter 7: 0.1654 (var=0.0176) (Δ vs real +0.1621)
  - iter 8: 0.1471 (var=0.0104) (Δ vs real +0.1438)
  - iter 9 (current): 0.1917 (var=0.0254) (Δ vs real +0.1883)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0150 (var=0.0104)
- pi_4: 0.4617 (var=0.0125)
- pi_1: 0.6621 (var=0.0603)
- pi_2: -0.0008 (var=0.0123)
- pi_5: 0.0071 (var=0.0075)
- pi_6: -0.0142 (var=0.0089)
- pi_7: 0.0079 (var=0.0079)
- pi_8: 0.0112 (var=0.0099)
- pi_9: -0.0658 (var=0.0097)

### Experiment 6
**Design**
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    early_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        discriminating = np.flatnonzero(diff != 0)

        # Restrict to the 7- and 8-reason overload conditions.
        if discriminating.size < 7:
            continue

        # The first discriminating cue identifies the option favored by the
        # high-validity, early run, regardless of physical A/B orientation.
        first_direction = diff[discriminating[0]]
        response = int(row["response"])
        chose_early_option = (
            (first_direction > 0 and response == 0) or
            (first_direction < 0 and response == 1)
        )
        early_choices.append(float(chose_early_option))

    if not early_choices:
        return float("nan")
    return float(np.mean(early_choices))
```

**Observed (real) value:** 0.3067 (var=0.0051)
**Candidate trajectory (this loop):**
  - iter 1: 0.3392 (var=0.0214) (Δ vs real +0.0325)
  - iter 2: 0.1721 (var=0.0092) (Δ vs real -0.1346)
  - iter 3: 0.2863 (var=0.0225) (Δ vs real -0.0204)
  - iter 4: 0.2371 (var=0.0093) (Δ vs real -0.0696)
  - iter 5: 0.2625 (var=0.0209) (Δ vs real -0.0442)
  - iter 6: 0.3075 (var=0.0134) (Δ vs real +0.0008)
  - iter 7: 0.2796 (var=0.0158) (Δ vs real -0.0271)
  - iter 8: 0.2846 (var=0.0175) (Δ vs real -0.0221)
  - iter 9 (current): 0.2863 (var=0.0197) (Δ vs real -0.0204)
**Other theories' values on this metric (for reference):**
- pi_4: 0.7804 (var=0.0054)
- pi_3: 0.3379 (var=0.0051)
- pi_1: 0.8433 (var=0.0097)
- pi_2: 0.6654 (var=0.0060)
- pi_5: 0.3113 (var=0.0065)
- pi_6: 0.3067 (var=0.0061)
- pi_7: 0.3404 (var=0.0071)
- pi_8: 0.3050 (var=0.0041)
- pi_9: 0.1708 (var=0.0061)

### Experiment 7
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0, 1, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 1, 0, 1, 0, 0, 1, 1]  B=[1, 0, 1, 0, 1, 0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    low_follow = []
    high_follow = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        discriminating = diff[diff != 0]

        # The strongest predicted chunking contrast occurs for the 8- and
        # 10-reason displays.
        if discriminating.size < 8:
            continue

        directions = np.sign(discriminating)
        n_switches = int(np.sum(directions[1:] != directions[:-1]))

        # Response coding is 0=A and 1=B. Determine whether the response
        # follows the terminal discriminating cue, independent of side.
        terminal_response = 0 if directions[-1] > 0 else 1
        follows_terminal = float(int(row["response"]) == terminal_response)

        if n_switches <= 1:
            low_follow.append(follows_terminal)
        elif n_switches >= 4:
            high_follow.append(follows_terminal)

    if len(low_follow) == 0 or len(high_follow) == 0:
        return float("nan")

    contrast = float(np.mean(low_follow) - np.mean(high_follow))

    # A preregisterable practical-significance diagnostic: whether coherent
    # low-switch sequences have more than an 8-percentage-point terminal-cue
    # advantage over switch-dense sequences.
    return float(contrast > 0.08)

```

**Observed (real) value:** 0.0000 (var=0.2016)
**Candidate trajectory (this loop):**
  - iter 1: 0.0000 (var=0.1344) (Δ vs real +0.0000)
  - iter 2: 0.0000 (var=0.2176) (Δ vs real +0.0000)
  - iter 3: 0.0000 (var=0.2016) (Δ vs real +0.0000)
  - iter 4: 0.0000 (var=0.2100) (Δ vs real +0.0000)
  - iter 5: 1.0000 (var=0.0384) (Δ vs real +1.0000)
  - iter 6: 0.0000 (var=0.2100) (Δ vs real +0.0000)
  - iter 7: 0.0000 (var=0.2304) (Δ vs real +0.0000)
  - iter 8: 0.0000 (var=0.2016) (Δ vs real +0.0000)
  - iter 9 (current): 0.0000 (var=0.2464) (Δ vs real +0.0000)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0000 (var=0.1824)
- pi_5: 1.0000 (var=0.1924)
- pi_1: 0.0000 (var=0.1476)
- pi_2: 0.0000 (var=0.2016)
- pi_4: 0.0000 (var=0.0384)
- pi_6: 0.0000 (var=0.2100)
- pi_7: 0.0000 (var=0.2356)
- pi_8: 0.0000 (var=0.1924)
- pi_9: 1.0000 (var=0.2436)

### Experiment 8
**Design**
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if len(data) == 0:
        return float('nan')

    agreements = []
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        diff = a - b
        discriminating = np.flatnonzero(diff != 0)
        if discriminating.size == 0:
            continue

        # Direction of the terminal discriminating cue in displayed order.
        terminal_direction = diff[discriminating[-1]]
        terminal_response = 0 if terminal_direction > 0 else 1
        agreements.append(float(int(row['response']) == terminal_response))

    if len(agreements) == 0:
        return float('nan')
    return float(np.mean(agreements))
```

**Observed (real) value:** 0.1721 (var=0.0112)
**Candidate trajectory (this loop):**
  - iter 1: 0.1996 (var=0.0057) (Δ vs real +0.0275)
  - iter 2: 0.1381 (var=0.0029) (Δ vs real -0.0340)
  - iter 3: 0.1440 (var=0.0029) (Δ vs real -0.0281)
  - iter 4: 0.1385 (var=0.0022) (Δ vs real -0.0335)
  - iter 5: 0.3129 (var=0.0021) (Δ vs real +0.1408)
  - iter 6: 0.1550 (var=0.0029) (Δ vs real -0.0171)
  - iter 7: 0.1325 (var=0.0023) (Δ vs real -0.0396)
  - iter 8: 0.1446 (var=0.0020) (Δ vs real -0.0275)
  - iter 9 (current): 0.1458 (var=0.0027) (Δ vs real -0.0262)
**Other theories' values on this metric (for reference):**
- pi_5: 0.7169 (var=0.0030)
- pi_3: 0.2754 (var=0.0030)
- pi_1: 0.3269 (var=0.0033)
- pi_2: 0.4110 (var=0.0016)
- pi_4: 0.4802 (var=0.0042)
- pi_6: 0.2250 (var=0.0059)
- pi_7: 0.3131 (var=0.0029)
- pi_8: 0.3069 (var=0.0027)
- pi_9: 0.3046 (var=0.0042)

### Experiment 9
**Design**
  A=[1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[1, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    scores = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        discriminating = np.flatnonzero(diff != 0)
        if discriminating.size == 0:
            continue

        # Features are already arranged in descending instructed validity.
        first_direction = float(np.sign(diff[int(discriminating[0])]))
        choice_direction = 1.0 if int(row["response"]) == 0 else -1.0
        scores.append(first_direction * choice_direction)

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(np.asarray(scores, dtype=float)))
```

**Observed (real) value:** -0.0325 (var=0.0099)
**Candidate trajectory (this loop):**
  - iter 1: 0.0788 (var=0.0077) (Δ vs real +0.1113)
  - iter 2: 0.0258 (var=0.0052) (Δ vs real +0.0583)
  - iter 3: 0.0517 (var=0.0043) (Δ vs real +0.0842)
  - iter 4: 0.0483 (var=0.0056) (Δ vs real +0.0808)
  - iter 5: -0.1858 (var=0.0079) (Δ vs real -0.1533)
  - iter 6: 0.0521 (var=0.0079) (Δ vs real +0.0846)
  - iter 7: 0.0492 (var=0.0059) (Δ vs real +0.0817)
  - iter 8: 0.0629 (var=0.0054) (Δ vs real +0.0954)
  - iter 9 (current): 0.0488 (var=0.0064) (Δ vs real +0.0813)
**Other theories' values on this metric (for reference):**
- pi_3: -0.0083 (var=0.0103)
- pi_6: 0.2096 (var=0.0101)
- pi_1: 0.7021 (var=0.0307)
- pi_2: 0.7054 (var=0.0384)
- pi_4: 0.0375 (var=0.0075)
- pi_5: 0.0108 (var=0.0053)
- pi_7: 0.0638 (var=0.0071)
- pi_8: 0.0083 (var=0.0044)
- pi_9: 0.0933 (var=0.0102)

### Experiment 10
**Design**
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Coherence-weighted agreement with the least-valid terminal cue."""
    if len(data) == 0:
        return float("nan")

    pattern_weights = {
        (1, 1, 1, 1): 1.00,   # fully coherent
        (-1, 1, 1, 1): 0.80,  # earliest cue conflicts
        (1, -1, 1, 1): 0.65,  # second cue conflicts
        (-1, -1, 1, 1): 0.35  # both prefix cues conflict
    }

    weighted_agreement = 0.0
    total_weight = 0.0

    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"]
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        diff = a - b
        if diff.size != 4 or diff[3] == 0:
            continue

        # Orient every trial so that the terminal (least-valid) cue is +1.
        terminal_sign = float(np.sign(diff[3]))
        relative_pattern = tuple(
            int(np.sign(x) * terminal_sign) for x in diff
        )
        weight = pattern_weights.get(relative_pattern, 0.0)
        if weight <= 0.0:
            continue

        terminal_favoring_response = 0 if terminal_sign > 0 else 1
        agreement = float(int(response) == terminal_favoring_response)
        weighted_agreement += weight * agreement
        total_weight += weight

    if total_weight <= 0.0:
        return float("nan")
    return float(weighted_agreement / total_weight)
```

**Observed (real) value:** 0.8748 (var=0.0069)
**Candidate trajectory (this loop):**
  - iter 1: 0.8459 (var=0.0034) (Δ vs real -0.0288)
  - iter 2: 0.8764 (var=0.0031) (Δ vs real +0.0016)
  - iter 3: 0.8737 (var=0.0025) (Δ vs real -0.0010)
  - iter 4: 0.8876 (var=0.0027) (Δ vs real +0.0128)
  - iter 5: 0.8319 (var=0.0033) (Δ vs real -0.0429)
  - iter 6: 0.8650 (var=0.0032) (Δ vs real -0.0098)
  - iter 7: 0.8685 (var=0.0030) (Δ vs real -0.0063)
  - iter 8: 0.8720 (var=0.0025) (Δ vs real -0.0028)
  - iter 9 (current): 0.8718 (var=0.0029) (Δ vs real -0.0030)
**Other theories' values on this metric (for reference):**
- pi_6: 0.9281 (var=0.0013)
- pi_3: 0.7843 (var=0.0034)
- pi_1: 0.5635 (var=0.0023)
- pi_2: 0.8101 (var=0.0072)
- pi_4: 0.8645 (var=0.0034)
- pi_5: 0.9256 (var=0.0034)
- pi_7: 0.8717 (var=0.0018)
- pi_8: 0.8753 (var=0.0015)
- pi_9: 0.8852 (var=0.0027)

### Experiment 11
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    scores = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        m = int(np.count_nonzero(diff))

        # The diagnostic manageable-load conflicts contain 5 or 6
        # discriminating cues. The final (lowest-validity) cue is index 9.
        if m not in (5, 6) or diff[-1] == 0:
            continue

        terminal_response = 0 if diff[-1] > 0 else 1
        scores.append(float(int(row["response"]) == terminal_response))

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** 0.8425 (var=0.0123)
**Candidate trajectory (this loop):**
  - iter 1: 0.4667 (var=0.0234) (Δ vs real -0.3758)
  - iter 2: 0.6754 (var=0.0134) (Δ vs real -0.1671)
  - iter 3: 0.7208 (var=0.0160) (Δ vs real -0.1217)
  - iter 4: 0.7929 (var=0.0084) (Δ vs real -0.0496)
  - iter 5: 0.7550 (var=0.0176) (Δ vs real -0.0875)
  - iter 6: 0.6750 (var=0.0219) (Δ vs real -0.1675)
  - iter 7: 0.8046 (var=0.0070) (Δ vs real -0.0379)
  - iter 8: 0.7975 (var=0.0080) (Δ vs real -0.0450)
  - iter 9 (current): 0.7729 (var=0.0107) (Δ vs real -0.0696)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6863 (var=0.0047)
- pi_7: 0.0642 (var=0.0023)
- pi_1: 0.1583 (var=0.0130)
- pi_2: 0.1342 (var=0.0084)
- pi_4: 0.3117 (var=0.0174)
- pi_5: 0.6737 (var=0.0053)
- pi_6: 0.6196 (var=0.0088)
- pi_8: 0.8558 (var=0.0026)
- pi_9: 0.6025 (var=0.0106)

### Experiment 12
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Side-balanced rate of choosing the 3-to-2 majority in strong-coalition trials."""
    majority_a = []
    majority_b = []
    strong_set = {0, 1, 2, 11, 12}

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        discriminating = set(np.flatnonzero(diff != 0).tolist())
        if discriminating != strong_set:
            continue

        response = int(row["response"])
        majority_direction = float(np.sum(np.sign(diff)))
        if majority_direction > 0:
            majority_a.append(float(response == 0))
        elif majority_direction < 0:
            majority_b.append(float(response == 1))

    if majority_a and majority_b:
        return float(0.5 * (np.mean(majority_a) + np.mean(majority_b)))

    pooled = majority_a + majority_b
    if pooled:
        return float(np.mean(pooled))
    return 0.5

```

**Observed (real) value:** 0.1713 (var=0.0095)
**Candidate trajectory (this loop):**
  - iter 1: 0.4931 (var=0.0223) (Δ vs real +0.3219)
  - iter 2: 0.2450 (var=0.0120) (Δ vs real +0.0737)
  - iter 3: 0.3063 (var=0.0213) (Δ vs real +0.1350)
  - iter 4: 0.2706 (var=0.0134) (Δ vs real +0.0994)
  - iter 5: 0.2562 (var=0.0165) (Δ vs real +0.0850)
  - iter 6: 0.3281 (var=0.0258) (Δ vs real +0.1569)
  - iter 7: 0.1938 (var=0.0092) (Δ vs real +0.0225)
  - iter 8: 0.1581 (var=0.0067) (Δ vs real -0.0131)
  - iter 9 (current): 0.1663 (var=0.0075) (Δ vs real -0.0050)
**Other theories' values on this metric (for reference):**
- pi_7: 0.7800 (var=0.0127)
- pi_3: 0.2650 (var=0.0077)
- pi_1: 0.8450 (var=0.0106)
- pi_2: 0.8638 (var=0.0097)
- pi_4: 0.3144 (var=0.0590)
- pi_5: 0.2125 (var=0.0050)
- pi_6: 0.1788 (var=0.0107)
- pi_8: 0.1313 (var=0.0048)
- pi_9: 0.2231 (var=0.0108)

### Experiment 13
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    d = data.copy()
    if len(d) == 0:
        return float('nan')

    # The competing theory's contextual calibration has reached full weight
    # well before this point; the advocated theory has no history dependence.
    d['_trial_index'] = d.groupby('subject_id', sort=False).cumcount()
    d = d.loc[d['_trial_index'] >= 24].copy()
    if len(d) == 0:
        return float('nan')

    def terminal_match(row):
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        terminal_diff = a[-1] - b[-1]
        if terminal_diff > 0:
            terminal_response = 0
        elif terminal_diff < 0:
            terminal_response = 1
        else:
            return np.nan
        return float(int(row['response']) == terminal_response)

    d['_terminal_match'] = d.apply(terminal_match, axis=1)
    d['_stimulus_key'] = d.apply(
        lambda r: (tuple(r['option_a_ratings']), tuple(r['option_b_ratings'])),
        axis=1
    )

    # Equal-weight subject-by-stimulus cells, eliminating variance caused by
    # random trial order and unequal late-block counts of the six pairs.
    cells = d.groupby(
        ['subject_id', '_stimulus_key'], sort=False
    )['_terminal_match'].mean()
    return float(cells.mean())
```

**Observed (real) value:** 0.8810 (var=0.0070)
**Candidate trajectory (this loop):**
  - iter 1: 0.6726 (var=0.0118) (Δ vs real -0.2084)
  - iter 2: 0.8268 (var=0.0060) (Δ vs real -0.0542)
  - iter 3: 0.7846 (var=0.0075) (Δ vs real -0.0964)
  - iter 4: 0.8182 (var=0.0035) (Δ vs real -0.0628)
  - iter 5: 0.8000 (var=0.0079) (Δ vs real -0.0810)
  - iter 6: 0.7732 (var=0.0093) (Δ vs real -0.1078)
  - iter 7: 0.8424 (var=0.0038) (Δ vs real -0.0385)
  - iter 8: 0.8430 (var=0.0036) (Δ vs real -0.0380)
  - iter 9 (current): 0.8307 (var=0.0057) (Δ vs real -0.0503)
**Other theories' values on this metric (for reference):**
- pi_3: 0.7201 (var=0.0040)
- pi_8: 0.3399 (var=0.0051)
- pi_1: 0.1647 (var=0.0106)
- pi_2: 0.4955 (var=0.0024)
- pi_4: 0.4758 (var=0.0129)
- pi_5: 0.6957 (var=0.0044)
- pi_6: 0.7470 (var=0.0062)
- pi_7: 0.5510 (var=0.0027)
- pi_9: 0.7048 (var=0.0044)

### Experiment 14
**Design**
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Classify each trial structurally and code whether the response followed
    # the physically terminal discriminating reason. In this design that
    # reason is also the final reason in instructed-validity scan order.
    records = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        idx = np.flatnonzero(diff != 0)
        if idx.size == 0:
            continue

        terminal_idx = int(idx[-1])
        terminal_is_b = bool(diff[terminal_idx] < 0)
        chose_terminal = float(int(row["response"]) == int(terminal_is_b))

        if idx.size > 1:
            largest_gap = int(np.max(np.diff(idx) - 1))
        else:
            largest_gap = 0
        foregrounded = bool(terminal_idx == (len(diff) - 1) and largest_gap > 0)
        records.append((foregrounded, terminal_is_b, chose_terminal))

    if not records:
        return float("nan")

    frame = pd.DataFrame(
        records, columns=["foregrounded", "terminal_is_b", "chose_terminal"]
    )

    # Compute the foregrounding contrast separately for terminal-A and
    # terminal-B trials, then average. This exactly removes any stable A/B
    # response-side preference rather than allowing it to inflate variance.
    side_contrasts = []
    for terminal_is_b in (False, True):
        side = frame[frame["terminal_is_b"] == terminal_is_b]
        fg = side.loc[side["foregrounded"], "chose_terminal"]
        clustered = side.loc[~side["foregrounded"], "chose_terminal"]
        if len(fg) > 0 and len(clustered) > 0:
            side_contrasts.append(float(fg.mean() - clustered.mean()))

    if not side_contrasts:
        return float("nan")
    return float(np.mean(side_contrasts))
```

**Observed (real) value:** 0.0192 (var=0.0062)
**Candidate trajectory (this loop):**
  - iter 1: -0.0025 (var=0.0097) (Δ vs real -0.0217)
  - iter 2: 0.0083 (var=0.0094) (Δ vs real -0.0108)
  - iter 3: 0.0179 (var=0.0075) (Δ vs real -0.0013)
  - iter 4: -0.0104 (var=0.0029) (Δ vs real -0.0296)
  - iter 5: 0.0338 (var=0.0086) (Δ vs real +0.0146)
  - iter 6: -0.0042 (var=0.0090) (Δ vs real -0.0233)
  - iter 7: -0.0004 (var=0.0054) (Δ vs real -0.0196)
  - iter 8: -0.0025 (var=0.0047) (Δ vs real -0.0217)
  - iter 9 (current): -0.0058 (var=0.0046) (Δ vs real -0.0250)
**Other theories' values on this metric (for reference):**
- pi_8: 0.0863 (var=0.0106)
- pi_3: -0.0217 (var=0.0090)
- pi_1: -0.0004 (var=0.0063)
- pi_2: -0.0008 (var=0.0037)
- pi_4: -0.0146 (var=0.0066)
- pi_5: 0.0054 (var=0.0094)
- pi_6: -0.0137 (var=0.0073)
- pi_7: -0.0042 (var=0.0041)
- pi_9: 0.0017 (var=0.0085)

### Experiment 15
**Design**
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Winner-choice rate at each unanimous evidence load.
    correct_by_load = {2: [], 4: [], 7: [], 10: []}

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        load = int(np.count_nonzero(diff))
        net_direction = float(np.sum(diff))

        if load not in correct_by_load or net_direction == 0.0:
            continue

        response = int(row["response"])
        favored_response = 0 if net_direction > 0.0 else 1
        correct_by_load[load].append(float(response == favored_response))

    rates = {}
    for load in (2, 4, 7, 10):
        values = correct_by_load[load]
        if len(values) == 0:
            return float("nan")
        rates[load] = float(np.mean(values))

    # Endpoint-emphasized, zero-sum decline contrast. Positive values indicate
    # that unanimous evidence becomes less effective as more reasons are added.
    return float(
        0.75 * rates[2]
        + 0.25 * rates[4]
        - 0.25 * rates[7]
        - 0.75 * rates[10]
    )
```

**Observed (real) value:** 0.0471 (var=0.0135)
**Candidate trajectory (this loop):**
  - iter 1: 0.0121 (var=0.0062) (Δ vs real -0.0350)
  - iter 2: 0.0635 (var=0.0054) (Δ vs real +0.0165)
  - iter 3: 0.0750 (var=0.0050) (Δ vs real +0.0279)
  - iter 4: 0.0425 (var=0.0061) (Δ vs real -0.0046)
  - iter 5: 0.0544 (var=0.0056) (Δ vs real +0.0073)
  - iter 6: 0.0715 (var=0.0061) (Δ vs real +0.0244)
  - iter 7: 0.0648 (var=0.0083) (Δ vs real +0.0177)
  - iter 8: 0.0779 (var=0.0066) (Δ vs real +0.0308)
  - iter 9 (current): 0.0658 (var=0.0087) (Δ vs real +0.0188)
**Other theories' values on this metric (for reference):**
- pi_3: 0.1525 (var=0.0093)
- pi_9: 0.0019 (var=0.0017)
- pi_1: -0.0021 (var=0.0054)
- pi_2: -0.0167 (var=0.0071)
- pi_4: -0.0006 (var=0.0027)
- pi_5: 0.1579 (var=0.0074)
- pi_6: 0.3079 (var=0.0057)
- pi_7: 0.2092 (var=0.0108)
- pi_8: 0.0406 (var=0.0094)

### Experiment 16
**Design**
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    validities = np.asarray([0.99, 0.75, 0.73, 0.71, 0.69, 0.67, 0.65, 0.51], dtype=float)
    reliability = np.log(np.clip(validities, 0.5001, 0.999) / (1.0 - np.clip(validities, 0.5001, 0.999)))
    scale = float(np.mean(reliability[reliability > 0.0]))
    strengths = np.power(np.clip(reliability / scale, 0.15, 6.0), 0.30)

    def summarize(a, b):
        d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
        idx = np.flatnonzero(d != 0.0)
        if idx.size == 0:
            return None
        dirs = np.sign(d[idx])
        m = int(dirs.size)
        switches = np.flatnonzero(dirs[1:] != dirs[:-1]) + 1

        run_lengths = []
        run = 1
        for k in range(1, m):
            if dirs[k] == dirs[k - 1]:
                run += 1
            else:
                run_lengths.append(run)
                run = 1
        run_lengths.append(run)

        accumulator = 0.0
        susceptibility = []
        k = 0
        while k < m:
            direction = float(dirs[k])
            end = k + 1
            while end < m and dirs[end] == direction:
                end += 1
            incoming = strengths[idx[k:end]]
            run_length = end - k
            if accumulator != 0.0 and np.sign(accumulator) != direction:
                incumbent = abs(accumulator) * (0.55 ** run_length)
                attack = 0.0
                for s in incoming:
                    attack = 0.55 * attack + float(s)
                advantage = (attack - incumbent) / max(attack + incumbent, 1e-12)
                susceptibility.append(1.0 / (1.0 + np.exp(-4.0 * np.clip(advantage, -15.0, 15.0))))
            for s in incoming:
                accumulator = 0.55 * accumulator + direction * float(s)
            k = end

        reversal_susceptibility = float(np.mean(susceptibility)) if susceptibility else 0.35
        fractions = switches.astype(float) / float(m) if switches.size else np.asarray([], dtype=float)
        canonical = tuple(int(x) for x in (dirs * dirs[0]))
        return {
            'terminal_singleton': float(len(run_lengths) > 1 and run_lengths[-1] == 1),
            'two_run': float(len(run_lengths) == 2),
            'switch_density': float(switches.size) / float(max(1, m - 1)),
            'susceptibility': reversal_susceptibility,
            'fractions': fractions,
            'canonical': canonical,
            'terminal_direction': float(dirs[-1])
        }

    def context_value(summaries, learning_rate):
        ages = np.arange(len(summaries) - 1, -1, -1, dtype=float)
        w = np.power(1.0 - learning_rate, ages)
        if len(summaries) > 1:
            w[-1] *= 0.35
        w = w / max(float(np.sum(w)), 1e-12)

        terminal = float(np.sum([wi * s['terminal_singleton'] for wi, s in zip(w, summaries)]))
        two_run = float(np.sum([wi * s['two_run'] for wi, s in zip(w, summaries)]))
        switch_density = float(np.sum([wi * s['switch_density'] for wi, s in zip(w, summaries)]))
        susceptibility = float(np.sum([wi * s['susceptibility'] for wi, s in zip(w, summaries)]))

        hist = np.zeros(5, dtype=float)
        mass = 0.0
        for wi, s in zip(w, summaries):
            fr = s['fractions']
            if fr.size == 0:
                continue
            contribution = float(wi) / float(fr.size)
            for f in fr:
                j = min(4, max(0, int(np.floor(5.0 * float(f)))))
                hist[j] += contribution
                mass += contribution
        if mass > 0.0:
            p = hist / mass
            concentration = float(np.sum(p ** 2))
            positive = p[p > 0.0]
            entropy = float(-np.sum(positive * np.log(positive)) / np.log(5.0))
        else:
            concentration = 0.2
            entropy = 1.0

        validity_gradient = float(np.clip(np.std(reliability) / max(np.mean(reliability), 1e-8), 0.0, 2.0))
        score = (1.30
                 + 1.20 * (terminal - 0.30)
                 + 1.20 * (concentration - 0.30)
                 + 0.80 * (two_run - 0.50)
                 - 1.50 * (entropy - 0.70)
                 - 0.70 * (switch_density - 0.25)
                 - 0.20 * validity_gradient
                 + 0.85 * (susceptibility - 0.35))
        return 1.0 / (1.0 + np.exp(-1.90 * np.clip(score - 1.30, -30.0, 30.0)))

    def one_subject(frame):
        summaries = []
        records = []
        for _, row in frame.iterrows():
            s = summarize(row['option_a_ratings'], row['option_b_ratings'])
            if s is None:
                continue
            summaries.append(s)

            # Average across the advocated learning-rate range to reduce
            # sensitivity to unknown individual parameter values.
            x = float(np.mean([context_value(summaries, lr) for lr in (0.05, 0.085, 0.12)]))

            # Retain conflict sequences only. The three canonical geometries
            # are +-----, +++-, and +++++- (including their A/B reversals).
            if len(s['canonical']) > 1 and len(set(s['canonical'])) > 1:
                chosen_direction = 1.0 if int(row['response']) == 0 else -1.0
                y = 1.0 if chosen_direction == s['terminal_direction'] else -1.0
                records.append((s['canonical'], x, y))

        if len(records) < 4:
            return 0.0

        residual_x = np.zeros(len(records), dtype=float)
        keys = [r[0] for r in records]
        xvals = np.asarray([r[1] for r in records], dtype=float)
        yvals = np.asarray([r[2] for r in records], dtype=float)

        for key in set(keys):
            loc = np.asarray([k == key for k in keys], dtype=bool)
            residual_x[loc] = xvals[loc] - float(np.mean(xvals[loc]))

        rms = float(np.sqrt(np.mean(residual_x ** 2)))
        if not np.isfinite(rms) or rms < 1e-10:
            return 0.0
        z = residual_x / rms
        return float(np.mean(z * yvals))

    if 'subject_id' in data.columns:
        values = [one_subject(g) for _, g in data.groupby('subject_id', sort=False)]
        return float(np.mean(values)) if values else 0.0
    return one_subject(data)

```

**Observed (real) value:** 0.0311 (var=0.0170)
**Candidate trajectory (this loop):**
  - iter 1: -0.0066 (var=0.0138) (Δ vs real -0.0377)
  - iter 2: -0.0204 (var=0.0067) (Δ vs real -0.0515)
  - iter 3: 0.0088 (var=0.0104) (Δ vs real -0.0223)
  - iter 4: -0.0047 (var=0.0057) (Δ vs real -0.0358)
  - iter 5: -0.0075 (var=0.0044) (Δ vs real -0.0386)
  - iter 6: 0.0090 (var=0.0077) (Δ vs real -0.0221)
  - iter 7: 0.0062 (var=0.0055) (Δ vs real -0.0249)
  - iter 8: -0.0172 (var=0.0093) (Δ vs real -0.0483)
  - iter 9 (current): 0.0124 (var=0.0092) (Δ vs real -0.0187)
**Other theories' values on this metric (for reference):**
- pi_9: 0.1406 (var=0.0138)
- pi_3: -0.0111 (var=0.0098)
- pi_1: -0.0000 (var=0.0077)
- pi_2: -0.0071 (var=0.0059)
- pi_4: 0.0090 (var=0.0080)
- pi_5: 0.0102 (var=0.0090)
- pi_6: 0.0001 (var=0.0112)
- pi_7: -0.0004 (var=0.0066)
- pi_8: -0.0155 (var=0.0095)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed stimulus-local decision-object family and correctly captures several qualitative invariances: no history covariance in Experiment 16, negligible physical-gap effect in Experiment 14, no practically large switch-density effect in Experiment 7, and reasonably close fits in Experiments 6, 8, and 10. However, the aggregate fit is not yet adequate. Major errors occur in Experiment 5 (mixed-minus-block terminal following 0.230 versus 0.003), Experiment 11 (terminal following 0.467 versus 0.843), Experiment 12 (majority choice 0.493 versus 0.171), and Experiment 13 (terminal following 0.673 versus 0.881). Terminal control is also too strong in Experiment 3 (0.767 versus 0.575) while being too weak in Experiment 4 (0.555 versus 0.665). This opposing pattern shows that a global increase or decrease in terminal susceptibility would not work. The current stimulus-local gate is miscalibrated: terminal reliability relative to the single strongest cue plus net coalition balance does not adequately distinguish coherent two-run conflicts from fragmented conflicts. The early and coalition objects are also too influential: TTB agreement is too high in Experiments 2 and 9, and the tally-support contrast is too large in Experiment 1. Finally, unanimous-load decline is too small in Experiment 15 (0.012 versus 0.047), and several conflict-condition variances are misfit, especially the underpredicted heterogeneity in Experiment 4.
Rationale: Keep the three-object, history-free arbitration architecture, but retune its stimulus-local arbitration. First, do not simply raise terminal_susceptibility: that would worsen Experiment 3. Instead, make coalition moderation depend more explicitly on conflict geometry within the existing object representation. A coherent two-run display or a clearly chunked terminal coalition should moderate the terminal object less than a fragmented, switch-rich conflict. This should raise terminal control in Experiments 11-13 and in the block condition of Experiment 5, while allowing lower terminal control on the more fragmented Experiment 3 displays. The current tiny terminal_run_bonus range is probably insufficient for that distinction; widen it or add a bounded two-run/coherence multiplier, while retaining saturation and keeping the resulting Experiment 7 contrast below its practical threshold. Second, reduce the early object's weight and modestly flatten coalition-balance sensitivity. The early object is currently duplicated inside the all-cue coalition and appears to create the large Experiment 5 arrangement contrast and excessive first-cue influence in Experiments 1, 2, and 9. Consider lowering early_commitment and using a less steep coalition-balance readout rather than broadly suppressing terminal evidence. Third, increase or rescale vigilance_decline enough to reproduce the modest positive Experiment 15 contrast without introducing count fatigue on conflict trials. Broaden subject-level terminal and lapse variation selectively to approach the large Experiment 4 variance, but avoid the already excessive variance in Experiments 5, 6, 11, and 12. These are in-family calibration changes; the core decision-object theory should be retained.

**Outcome of your advice:** iter 1 candidate loss=0.1988 -> iter 2 candidate loss=0.1247 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The accepted revision substantially improves the aggregate loss and remains faithful to the prescribed stimulus-local, history-free decision-object family. The prior geometry-based advice was productive: Experiments 4, 5, 10, 12, 13, and 15 all improved, while the desired invariances remain intact in Experiments 7, 14, and 16. Nevertheless, the model is not yet experiment-invariant. Terminal control is still much too strong in Experiment 3 (0.814 versus 0.575) and in the overload conflicts of Experiment 6, where early-option choice is only 0.172 versus 0.307. At the same time it remains too weak in the manageable-load terminal conflicts of Experiment 11 (0.675 versus 0.843), with smaller shortfalls in Experiments 12 and 13. Experiment 1 also deteriorated sharply: its tally-support contrast is 0.771 versus 0.494. Thus, the successful geometry adjustment has become too coarse: the unconditional extra multiplier for every two-run conflict produces excessive terminal dominance at short and overloaded displays but still cannot provide enough terminal access in the diagnostic 5–6-reason conflicts. The reduction of the early object also appears to have overshot, given the now-low TTB rate in Experiment 2, the low early choice in Experiment 6, and the enlarged Experiment 1 contrast. Finally, the mean in Experiment 4 is excellent, but its between-subject variance remains severely compressed (0.009 versus 0.062), contrary to the recommendation's emphasis on stable heterogeneity.
Rationale: Keep the three-object architecture and build on the accepted geometry improvement, but refine rather than globally strengthen terminal recency. Replace the unconditional double two-run amplification with bounded conflict-load-sensitive access: terminal distinctiveness should be strongest for manageable 4–6-reason conflicts, while being weaker for very short conflicts and for 7+ reason overload displays. This is an object-access calibration, not general count fatigue, and should raise Experiments 11–13 while lowering terminal dominance in Experiments 3 and 6. Retain or slightly strengthen fragmentation attenuation for three-or-more-run patterns, because Experiment 3 remains far too recency-driven, but preserve the near-zero practical switch-density result in Experiment 7. Separating a terminal-singleton privilege from terminal-run-length saturation may also help: the present run bonus rewards already-large terminal coalitions too strongly, whereas Experiment 11 requires privileged access to a singleton terminal counterargument. Partially restore early commitment to an intermediate range rather than undoing the prior accepted change wholesale. A modest restoration should improve Experiments 1, 2, 6, and 9, while the refined manageable-load terminal gate can protect Experiments 11–13. Do not further increase vigilance decline; Experiment 15 is already close. Preserve the history-free and gap-insensitive implementation. Finally, widen effective subject heterogeneity without increasing mean terminal dominance—e.g., use less saturated arbitration weights or a subject-specific terminal-versus-coalition mixing temperature—so stable susceptibility differences can express themselves in Experiment 4 rather than being washed out by the currently dominant terminal weight.

**Outcome of your advice:** iter 2 candidate loss=0.1247 -> iter 3 candidate loss=0.1168 -> the gate marked it ACCEPTED.

### Iteration 3
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed stimulus-local, history-free three-object mechanism, and the latest revision was directionally useful: aggregate loss improved from 0.1247 to 0.1168. The load-sensitive gate improved Experiment 3 relative to iteration 2, brought the overload result in Experiment 6 close to the data, and further improved manageable-load terminal control in Experiment 11. The intended invariances are also excellent: Experiment 7 shows no practically large switch-density effect, Experiment 14 has essentially the correct near-zero gap effect, Experiment 16 has only weak history covariance, and Experiment 10 is almost exact. Nevertheless, the model is not yet experiment-invariant. Terminal following remains much too high in Experiment 3 (0.768 versus 0.575), while it remains too low in Experiments 11 (0.721 versus 0.843), 12 (equivalently, majority following is 0.306 versus 0.171), and 13 (0.785 versus 0.881). Experiment 5 also regressed sharply, with a spurious mixed-minus-block arrangement effect of 0.169 versus 0.003. More generally, Experiment 1 still has an excessive tally-support contrast (0.720 versus 0.494), and first-cue influence is too high in Experiments 2 and 9. The partially restored early commitment moved those latter metrics past their targets: iteration 2 was slightly below the Experiment 2 target, whereas iteration 3 is substantially above it. Finally, broad terminal-susceptibility variation increased variance in Experiment 4 but did so indiscriminately: its variance remains too low there (0.023 versus 0.062) while becoming much too high in Experiments 5, 6, and 12.
Rationale: Retain the three bounded decision objects and the successful stimulus-local load gate, but sharpen rather than globally strengthen recency. The current implementation leaves an unconditional run-length multiplier, `1 + 0.25 * terminal_run_bonus * run_saturation`, even when manageable access is near zero. Gate or greatly shrink that residual run bonus outside the manageable conflict window. This should reduce excessive short-conflict recency in Experiment 3 and suppress the unwanted arrangement effect among ten-reason displays in Experiment 5 without imposing general count fatigue. At the same time, strengthen the coherent 5–6-reason peak or singleton privilege within the window to close the remaining deficits in Experiments 11–13. Strengthen fragmentation attenuation for three-or-more-run conflicts as well, since Experiment 3 remains too terminal-driven despite the accepted improvement. Do not globally raise terminal susceptibility. Reduce coalition influence modestly—preferably its balance-dependent slope rather than deleting the coalition object—because the coalition currently produces both the excessive tally modulation in Experiment 1 and excessive majority following in Experiment 12. For early commitment, avoid another oscillation: iteration 2 was too low and iteration 3 too high on Experiment 2, so use a range centered between those two settings rather than repeating either a broad reduction or restoration. Replace the broad susceptibility-range solution to heterogeneity with an in-family ambiguity-sensitive mixing temperature or nonlinear object contest: stable subject differences should be expressed strongly when terminal and coalition objects are closely matched, as in Experiment 4, but compressed when one object clearly dominates, as in Experiments 5, 6, and 12. Vigilance decline should be nudged slightly back toward the iteration-2 calibration, since Experiment 15 has moved from near-target to a modest overprediction. Preserve the history-free implementation and the successful gap and switch-density invariances.

**Outcome of your advice:** iter 3 candidate loss=0.1168 -> iter 4 candidate loss=0.1215 -> the gate marked it REJECTED.

### Iteration 4
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed stimulus-local, history-free decision-object family, and several qualitative invariances are preserved: Experiment 7 shows no practically significant switch-density effect, Experiment 14 remains near zero for physical foregrounding, Experiment 16 shows only weak history covariance, and Experiment 15 now closely matches the modest unanimous-load decline. The revision also improved the means in Experiments 3, 4, 5, 11, 12, and 13 relative to iteration 3. Nevertheless, it was rejected by the loss gate (0.1215 versus the accepted 0.1168), so it cannot be shipped. Important failures remain: terminal following is too high in Experiment 3 (0.746 versus 0.575), the arrangement contrast is too large in Experiment 5 (0.146 versus 0.003), and tally/majority effects remain excessive in Experiments 1 and 12. Terminal control is still too weak in Experiments 11 and 13. Moreover, the new ambiguity-sensitive temperature did not solve the targeted heterogeneity problem: Experiment 4 variance remains 0.023 versus 0.062, while variance is compressed in several other conditions. The revision also moved Experiment 6 away from the accepted candidate's near-exact fit and degraded the near-zero effects in Experiments 14 and 16. Thus the combined geometry, coalition-slope, and temperature intervention traded improvements in selected means for broader losses.
Rationale: Build on the unchanged accepted iteration-3 base, not on this rejected candidate. The previous recommendation to further sharpen the load window, reduce the coalition-balance slope, and add an ambiguity-sensitive arbitration temperature was rejected as a package, so do not repeat it. Instead, target a different in-family deficiency: terminal readout confidence currently depends only on run saturation and remains nearly deterministic even when the terminal object was retrieved from a fragmented display. Add a bounded retrieval-fidelity term to terminal_confidence, based on terminal-run integrity relative to the competing runs, while leaving terminal object access and the accepted load gate largely unchanged. Terminal confidence should remain high for coherent two-run or singleton-terminal displays, preserving Experiments 4 and 11–13, but move modestly toward 0.5 for fragmented multi-run displays. This should reduce excessive terminal agreement in Experiment 3 and the mixed-condition excess in Experiment 5 without recreating the Experiment 6 regression or globally weakening recency. Separately, replace the coalition's linear signed-support balance with separately saturated support for the two directional camps—for example, saturate positive and negative reliability mass before taking their bounded difference. This is preferable to another global reduction in coalition weight: it can reduce raw tally sensitivity in Experiment 1 and majority dominance in Experiment 12 while retaining a genuine coalition object. Keep the accepted iteration-3 early-commitment and vigilance settings rather than repeating the rejected intermediate-range and vigilance changes. Remove the ineffective ambiguity-temperature addition for now; once the mean structure improves, heterogeneity can be revisited with a more localized subject parameter such as variation in fragmented-terminal retrieval fidelity rather than broad contest sharpening.

**Outcome of your advice:** iter 4 candidate loss=0.1215 -> iter 5 candidate loss=0.2062 -> the gate marked it REJECTED.

### Iteration 5
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed stimulus-local, history-free decision-object family, but it is decisively worse than the accepted iteration-3 base: aggregate loss rose from 0.1168 to 0.2062, so the gate correctly rejected it. The newly introduced run-fragmentation fidelity rule fixes Experiment 3 (0.595 versus 0.575), but it does not generalize. It creates a false switch-density effect in Experiment 7 (1.0 versus 0.0), collapses terminal control in Experiment 4 (0.312 versus 0.665), reverses and exaggerates the arrangement contrast in Experiment 5 (-0.135 versus 0.003), and worsens Experiment 8 (0.313 versus 0.172) and Experiment 9 (-0.186 versus -0.033). Thus, counting three or more directional runs is not a valid general marker of unreliable terminal retrieval: some short fragmented displays still show strong human terminal control. The separately saturated coalition calculation also failed to solve its targets. Experiment 1 remains far too tally-sensitive (0.732 versus 0.494), while Experiment 12 improves only modestly (0.256 versus 0.171), with collateral shifts elsewhere. The candidate retains good fits for the physical-gap invariance in Experiment 14, modest unanimous-load decline in Experiment 15, and weak history covariance in Experiment 16, but those successes were already present in the accepted base. Between-subject variance also remains poorly captured, especially the severe underprediction in Experiment 4.
Rationale: Build on the unchanged accepted iteration-3 model. My previous recommendation to use generic multi-run retrieval attenuation and separately saturated coalition camps was rejected by the gate, so do not repeat either modification. In particular, remove `n_runs >= 3` as a direct determinant of terminal confidence: it incorrectly treats the alternating four-cue displays in Experiment 4 as unreliable and directly violates the required limited switch-density effect in Experiment 7. Try a narrower in-family calibration aimed at the actual residual pattern in the accepted base: add only a mild conflict-load-dependent reduction in terminal readout fidelity for long conflicts (approximately seven or more discriminating reasons), applied similarly across sequence geometries at a given load and never on unanimous trials. This can reduce the high-load contribution to excessive terminal following in Experiment 3, slightly improve the overload balance in Experiment 6, and avoid manufacturing a low-switch/high-switch contrast in Experiment 7. Keep fidelity essentially unchanged for four-cue conflicts and manageable five-to-six-cue conflicts, protecting Experiments 4, 10, 11, and 13. Any reduction at ten reasons should be weak and geometry-neutral so that Experiment 5's mixed and block conditions move together rather than producing another arrangement contrast. Retain the accepted iteration-3 coalition computation, early commitment, vigilance settings, and load-sensitive terminal-access gate for this next test; the current broad coalition rewrite produced no aggregate benefit. Make this a small, isolated change rather than another multi-parameter package, since both of the last two packages were rejected.

**Outcome of your advice:** iter 5 candidate loss=0.2062 -> iter 6 candidate loss=0.1315 -> the gate marked it REJECTED.

### Iteration 6
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed stimulus-local, history-free three-object arbitration family and preserves the key qualitative invariances: no practical switch-density effect in Experiment 7, little physical-gap effect in Experiment 14, weak history covariance in Experiment 16, and modest unanimous-load decline in Experiment 15. It also makes Experiment 6 essentially exact. However, the candidate was rejected by the accept gate (loss 0.1315 versus the accepted iteration-3 loss 0.1168), so it is not adequate. The geometry-neutral long-conflict fidelity reduction modestly lowers excessive terminal following in Experiment 3, from 0.768 in the accepted base to 0.743, but remains far above 0.575. It also worsens the already spurious Experiment 5 arrangement contrast to 0.210 versus 0.003. More importantly, the broader simulation moves away from the data in the manageable coherent reversals that the theory should explain: Experiment 11 falls to 0.675 versus 0.843, Experiment 12 predicts too much majority following (0.328 versus 0.171), and Experiment 13 falls to 0.773 versus 0.881. Experiment 1 remains too tally-sensitive (0.689 versus 0.494), and terminal control remains slightly weak in Experiment 4. Thus terminal control is not simply too strong at high load; it is too strong for fragmented terminal-singleton conflicts in Experiment 3 but too weak when a low-reliability terminal reason forms a coherent reversal against an otherwise unified prefix in Experiments 11–13. The current relative-reliability gate does not represent that distinction adequately. Between-subject variance also remains too compressed in Experiment 4, although mean calibration should be addressed first.
Rationale: Build on the unchanged accepted iteration-3 base. My previous recommendation for a geometry-neutral long-load readout reduction was rejected, so do not repeat or strengthen that intervention. Instead, modify the existing terminal-access reliability gate in one narrowly targeted, stimulus-local way: condition the minimum accessibility of a weak terminal reason on coherence of the preceding coalition. For manageable five-to-six-reason displays in which the prefix is directionally coherent and the terminal singleton or short terminal run creates the only reversal, impose a bounded floor on terminal access despite low terminal-relative reliability. This should raise terminal following in Experiments 11 and 13 and reduce majority following in Experiment 12 without globally increasing terminal susceptibility. Conversely, do not grant that floor when a terminal singleton follows an already fragmented prefix; for such displays with at least five discriminating reasons, retain the ordinary reliability gate or apply only a mild attenuation. That interaction targets the recurring high terminal agreement in Experiment 3 while leaving the alternating four-cue displays in Experiment 4 untouched and avoiding the generic switch-count rule that previously failed. Implement this as a small prefix-coherence-by-terminal-run adjustment to `reliability_gate`, not as another broad change to terminal confidence, coalition computation, or overall load fatigue. Keep the accepted early, coalition, vigilance, and history-free settings. Exclude seven-plus coherent reversals from any new penalty so Experiment 6 is not pushed away from its already good accepted-base fit. Once this mean structure improves, localized heterogeneity in the coherence-conditioned access floor can be considered for Experiment 4; broad susceptibility variation has already produced poor variance tradeoffs.

**Outcome of your advice:** iter 6 candidate loss=0.1315 -> iter 7 candidate loss=0.1013 -> the gate marked it ACCEPTED.

### Iteration 7
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed stimulus-local, history-free three-object arbitration family, and the latest narrowly targeted change was clearly successful: aggregate loss improved from 0.1168 to 0.1013 and was accepted. In particular, coherence-conditioned terminal access moved Experiment 11 to 0.805 versus 0.843, Experiment 12 to 0.194 versus 0.171, and Experiment 13 to 0.842 versus 0.881, while preserving the desired invariances in Experiments 7, 14, and 16 and the modest unanimous-load decline in Experiment 15. Experiments 6 and 10 also remain close. Nevertheless, three large, systematic discrepancies prevent an experiment-invariant fit: Experiment 1 remains too sensitive to whether the tally supports the first cue (0.694 versus 0.494), Experiment 3 still shows excessive terminal agreement (0.744 versus 0.575), and Experiment 5 retains a large spurious mixed-minus-block contrast (0.165 versus 0.003). First-cue influence is also somewhat excessive in Experiments 2 and 9, whereas terminal control is slightly weak in Experiments 4, 8, 11, and 13. The pattern suggests duplication among the bounded objects rather than a global terminal-strength error: in fragmented patterns where the initial and terminal directions coincide, the early and terminal objects effectively cast two votes for the same option, creating excessive terminal agreement and arrangement sensitivity. The remaining variance mismatch is secondary but notable, especially the compressed variance in Experiment 4 (0.020 versus 0.062).
Rationale: Build on this accepted candidate and retain the successful prefix-coherence accessibility floor. The previous coherence-conditioned advice was accepted, so it is appropriate to refine it: modestly raise the manageable coherent-reversal floor to close the small remaining deficits in Experiments 11 and 13 and the excess majority following in Experiment 12, while making the existing fragmented-prefix discount somewhat stronger specifically for five-to-six-reason conflicts. Do not extend that discount generically to all multi-run or long displays, since earlier generic fragmentation and long-load interventions were rejected and damaged Experiments 4, 5, and 7.

The main new adjustment should address object redundancy. Add a bounded overlap correction when the early and terminal objects favor the same option—especially when the coalition favors the other option—so that early commitment and terminal retrieval are not counted as two independent reasons. For example, attenuate the early object's arbitration weight as a saturating function of terminal access when their directions coincide, rather than globally reducing early commitment. This should directly reduce the mixed-condition excess in Experiment 5, where early and terminal support align, and reduce excessive terminal agreement in odd-run fragmented cases in Experiment 3. It may also reduce excessive first-cue influence in Experiments 2 and 9 without weakening terminal access in the coherent reversals of Experiments 11–13. Because unanimous trials have identical object predictions, this correction should preserve Experiment 15 and Experiment 10.

Avoid another broad coalition rewrite, ambiguity temperature, generic switch-count penalty, or global change in terminal susceptibility; those directions were previously rejected or would worsen the opposing terminal-control errors. After improving the mean structure, heterogeneity can be increased through subject variation in the localized overlap correction or coherence-access floor, rather than by widening global terminal susceptibility, which previously produced poor variance tradeoffs.

**Outcome of your advice:** iter 7 candidate loss=0.1013 -> iter 8 candidate loss=0.1182 -> the gate marked it REJECTED.

### Iteration 8 (most recent)
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed stimulus-local, history-free decision-object family, but the current revision was rejected and is worse than the accepted iteration-7 base (loss 0.1182 versus 0.1013). The new overlap correction and stronger accessibility adjustments produced only a modest improvement in Experiment 5 (0.165 to 0.147) and a useful improvement in Experiment 12 (0.194 to 0.158), while causing broader regressions. The tally-support contrast in Experiment 1 increased from 0.694 to 0.721 versus 0.494 observed; first-cue following in Experiment 2 increased from 0.229 to 0.271 versus 0.194; four-cue terminal control in Experiment 4 fell from 0.600 to 0.562 versus 0.665; and the unanimous-load decline in Experiment 15 rose from 0.065 to 0.078 versus 0.047. Experiment 16 also moved farther from the desired near-zero history covariance. The principal accepted-base failures therefore remain: excessive tally sensitivity in Experiment 1, excessive terminal agreement in Experiment 3, and a spurious arrangement effect in Experiment 5. Meanwhile, strong terminal control is still slightly underpredicted in Experiments 4, 11, and 13. The rejected overlap rule does not isolate these cases reliably and appears to distort arbitration more generally. Between-subject heterogeneity also remains too compressed, particularly in Experiment 4, but mean structure should be fixed first.
Rationale: Build on the unchanged accepted iteration-7 base. My previous recommendation to strengthen the coherent-access floor and add direction-contingent early-terminal overlap correction was rejected by the gate, so do not repeat or enlarge either intervention. Retain iteration 7's successful coherence-conditioned terminal-access rule as it was.

Try a different, narrow in-family normalization: prevent the coalition object from fully reusing the endpoint cues that already instantiate the early and terminal objects. Compute the coalition from all visible evidence as before, but softly downweight the first discriminating cue and final discriminating cue when forming signed and total coalition support, using one small bounded redundancy parameter. This differs from the rejected pairwise overlap correction: it applies regardless of whether early and terminal directions agree and changes only the coalition summary, rather than attenuating an arbitration object after directions are known. It should reduce triple counting of endpoint information, potentially lowering first-cue influence in Experiments 1, 2, and 9 and terminal overexpression in fragmented Experiment 3 displays, while leaving the dedicated terminal object intact for Experiments 4, 10, 11, and 13. Applying the same endpoint residualization across arrangements may also reduce the Experiment 5 mixed-versus-block contrast without introducing a switch-count rule.

Make this an isolated, modest change and preserve the accepted geometry multiplier, load window, early-commitment range, terminal-susceptibility range, and vigilance calibration. Do not introduce another generic fragmentation penalty, long-load fidelity decline, ambiguity temperature, or separately saturated coalition rewrite; those directions were previously rejected. After testing the mean effects, localized subject variation in the new coalition-redundancy coefficient could be considered, but it should not be broadened yet because Experiment 4 variance remains the only major heterogeneity deficit while several other variances are already adequate or excessive.

**Outcome of your advice:** iter 8 candidate loss=0.1182 -> CURRENT candidate loss=0.1056 -> the gate marked it REJECTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate remains faithful to the prescribed stimulus-local, history-free three-object family, but it was rejected by the accept gate: loss 0.1056 is worse than the accepted iteration-7 loss of 0.1013. Endpoint residualization therefore did not provide a general improvement. Relative to iteration 7, it slightly reduced excessive terminal agreement in Experiment 3, but worsened the principal Experiment 5 arrangement effect from 0.165 to 0.192, increased the Experiment 1 tally contrast from 0.694 to 0.718, and weakened the previously good coherent-terminal fits in Experiments 11 and 13. The main failures remain substantial: Experiment 1 is too tally-sensitive, Experiment 3 is too terminal-driven, and Experiment 5 has a large mixed-minus-block effect that is nearly absent in humans. Terminal control is simultaneously somewhat too weak in Experiments 4, 11, and 13, ruling out a global reduction in terminal susceptibility. The model otherwise captures important invariances and secondary results: Experiments 7, 10, 12, 14, 15, and 16 are reasonably close, and Experiment 6 is also satisfactory. Between-subject variance remains too compressed in Experiment 4, but mean calibration is still the priority.",
  "verdict": "regenerate",
  "rationale": "Build on the unchanged accepted iteration-7 base. My previous recommendation to residualize both endpoints from the coalition was rejected, so remove that addition and do not enlarge or otherwise repeat it. Try a different isolated calibration of the existing terminal reliability gate. The fixed transformation `reliability_gate = 0.08 + 0.92 * terminal_access` gives even poorly accessible terminal cues a consequential weight once multiplied by high terminal susceptibility and geometry bonuses. Make this transformation more selective only for conflicts with at least five discriminating cues that do not receive the accepted coherent-prefix accessibility floor\u2014for example, use a small parameterized intrinsic baseline and a mildly convex mapping such as `b + (1-b) * terminal_access**gamma`. Retain the current linear gate for four-cue displays and retain the iteration-7 coherent 5\u20136 cue floor unchanged. This should suppress weak, noncoherent terminal access in much of Experiment 3 and reduce excessive terminal/tally opposition in the seven-cue trials of Experiment 1, while protecting Experiments 4, 8, 10, 11, 12, and 13. It should also move Experiment 6 slightly toward the observed higher early-choice rate. Do not globally lower terminal susceptibility, strengthen generic fragmentation penalties, add high-load fidelity decay, or alter the coalition normalization again; those interventions have either been rejected or would damage the strong coherent-terminal cases and switch-density invariance. Treat Experiment 5 as a hard monitoring constraint: reject settings that enlarge its arrangement contrast, rather than adding another direction-contingent overlap rule, which was already unsuccessful. Keep vigilance, history independence, and physical-gap insensitivity fixed. Revisit localized heterogeneity only after this mean-level gate is improved."
}
```

## Usage

```json
{
  "input_tokens": 32059,
  "output_tokens": 3578
}
```
