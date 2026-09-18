# feedback_iter_01

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
- THEORY 1 = `pi_4`
- THEORY 2 = `pi_3`
- The recommendation below acts on THEORY 2 (= `pi_3`).

Replace pi_3 with a Structure-Learning Coalition Arbitration theory. The new theory should propose that subjects learn how discriminating cues are organized into recurring coalitions across a block, rather than treating representational load as a function only of cue count and current discrimination density. A stable, repeatedly encountered coalition of lower-ranked cues can be encoded as a single relational chunk and compete with the highest-validity cue; coalition recurrence, coherence, and accumulated conflict with the leading cue determine its accessibility. Conversely, when the opposing lower-cue coalition changes composition or fails to form a stable chunk, the advertised validity hierarchy remains accessible even with ten cues. Arbitration should therefore depend jointly on history-sensitive coalition learning, current coalition evidence, advertised validity dominance, and individual thresholds, with recency and hysteresis preventing complete trial-by-trial switching. Choices should arise from competition between a rank-based signal and a learned coalition signal, preferably preserving graded validity and consensus strength rather than reducing both signals immediately to signs. This mechanism can produce incomplete majority capture in the uniformly dense block of Experiment 1, very strong majority responding when a coherent anti-leading-cue coalition is learned in Experiment 2, lexical dominance in Experiment 5 despite nominally high load, and intermediate behavior in Experiments 4 and 6. It differs from pi_3 by rejecting raw load as the causal bottleneck and differs from pi_4 by learning cue-relationship structure rather than using only average block density. The new theory should make identifiable predictions by crossing cue count and density with coalition recurrence, coalition membership stability, block order, and abrupt coalition reversals while holding individual displays fixed.


## CANDIDATE THEORY
Relational Coalition Consolidation theory proposes that people retain the advertised validity hierarchy while also learning recurring relationships among lower-ranked cues. On each encounter, the learner represents lower cues relative to the highest-validity discriminating cue and records which cues jointly oppose that leader. Repeated, internally coherent opposition patterns become relational chunks: their accessibility increases with recurrence, pairwise co-opposition, consensus, and repeated conflict with the leading cue. Accessibility changes gradually and exhibits hysteresis, so an established coalition remains available across occasional mismatching trials while an isolated coalition does not immediately displace rank-based processing. Decision evidence is a continuous competition between a graded rank signal, strengthened by advertised validity dominance, and a graded coalition signal reflecting the validity-weighted consensus of lower cues. Coalition access is strongly nonlinear: baseline access is low, only closely matching recurring memberships consolidate, and a large advertised-validity advantage for the leading cue suppresses coalition retrieval. Thus, cue count alone does not cause pooling. Many cues can preserve lexical dominance when their coalitional membership rotates, whereas a recurrent and coherent anti-leader coalition can capture choice when validity dominance is modest. The account predicts order-dependent acquisition, persistence after coalition reversals, and different choices for identical displays following stable versus unstable coalition histories.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Relational Coalition Consolidation expects shape (2, n_features); got {stim.shape}."
        )

    n_features = int(parameters["n_features"])
    if stim.shape[1] != n_features:
        raise ValueError(
            f"Stimulus has {stim.shape[1]} features but n_features={n_features}."
        )

    validities = np.asarray(parameters["validities"], dtype=float).reshape(-1)
    if validities.size != n_features:
        raise ValueError("validities must contain one value per feature.")

    learning_rate = float(parameters["learning_rate"])
    relation_prior = float(parameters["relation_prior"])
    recurrence_rate = float(parameters["recurrence_rate"])
    coalition_threshold = float(parameters["coalition_threshold"])
    gate_steepness = float(parameters["gate_steepness"])
    activation_retention = float(parameters["activation_retention"])
    current_retrieval_rate = float(parameters["current_retrieval_rate"])
    hysteresis_strength = float(parameters["hysteresis_strength"])
    dominance_suppression = float(parameters["dominance_suppression"])
    initial_access = float(parameters["initial_access"])
    access_floor = float(parameters["access_floor"])
    access_ceiling = float(parameters["access_ceiling"])
    dominance_gain = float(parameters["dominance_gain"])
    coalition_gain = float(parameters["coalition_gain"])
    sensitivity = float(parameters["evidence_sensitivity"])
    lapse = float(parameters["lapse"])

    cue_order = np.argsort(-validities, kind="stable")
    rank_position = np.empty(n_features, dtype=int)
    rank_position[cue_order] = np.arange(n_features)

    # The advertised validity reliability above chance supplies a graded,
    # bounded evidence scale. A small floor handles validity exactly 0.5.
    reliability = np.maximum(2.0 * validities - 1.0, 0.02)

    pair_opposition = np.zeros((n_features, n_features), dtype=float)
    pair_opportunity = np.zeros((n_features, n_features), dtype=float)
    prototypes = []
    activation = float(np.clip(initial_access, 0.0, 1.0))

    def describe(display):
        arr = np.asarray(display, dtype=float)
        if arr.ndim != 2 or arr.shape != (2, n_features):
            return None
        directions = np.sign(arr[0] - arr[1])
        leader = None
        for cue in cue_order:
            if directions[cue] != 0.0:
                leader = int(cue)
                break
        if leader is None:
            return {
                "directions": directions,
                "leader": None,
                "leader_sign": 0.0,
                "lower": np.array([], dtype=int),
                "opponents": np.array([], dtype=int),
                "mask": np.zeros(n_features, dtype=bool)
            }

        leader_sign = float(directions[leader])
        lower = np.asarray([
            j for j in range(n_features)
            if rank_position[j] > rank_position[leader] and directions[j] != 0.0
        ], dtype=int)
        opponents = np.asarray([
            j for j in lower if directions[j] == -leader_sign
        ], dtype=int)
        mask = np.zeros(n_features, dtype=bool)
        mask[opponents] = True
        return {
            "directions": directions,
            "leader": leader,
            "leader_sign": leader_sign,
            "lower": lower,
            "opponents": opponents,
            "mask": mask
        }

    def advertised_dominance(desc):
        if desc is None or desc["leader"] is None:
            return 0.0
        leader = int(desc["leader"])
        lower = desc["lower"]
        if lower.size > 0:
            strongest_lower = float(np.max(reliability[lower]))
        else:
            strongest_lower = 0.0
        return float(max(
            0.0,
            (float(reliability[leader]) - strongest_lower)
            / max(float(reliability[leader]), 0.02)
        ))

    def relational_strength(desc):
        lower = desc["lower"]
        opponents = desc["opponents"]
        mask = desc["mask"]
        if lower.size == 0 or opponents.size < 2:
            return 0.0

        n_opp = float(opponents.size)
        n_support = float(lower.size - opponents.size)
        # Net coherence is zero when lower cues split evenly and one when a
        # unanimous lower coalition opposes the leading cue.
        coherence = max(0.0, (n_opp - n_support) / float(lower.size))

        pair_values = []
        for u in range(opponents.size):
            for v in range(u + 1, opponents.size):
                j = int(opponents[u])
                k = int(opponents[v])
                numerator = pair_opposition[j, k] + 0.5 * relation_prior
                denominator = pair_opportunity[j, k] + relation_prior
                pair_values.append(numerator / max(denominator, 1e-12))
        pair_consistency = float(np.mean(pair_values)) if pair_values else 0.5

        # Retrieval now requires close membership matches. The high similarity
        # exponent prevents nested or broadly overlapping coalitions from being
        # treated as repetitions of the same relational chunk.
        recurrence_mass = 0.0
        for old_mask, old_weight in prototypes:
            intersection = int(np.sum(mask & old_mask))
            union = int(np.sum(mask | old_mask))
            similarity = (float(intersection) / float(union)) if union > 0 else 0.0
            recurrence_mass += old_weight * similarity ** 8
        recurrence = 1.0 - np.exp(-recurrence_rate * recurrence_mass)

        return float(coherence * pair_consistency * recurrence)

    def update_memory(desc):
        nonlocal prototypes
        if desc is None or desc["leader"] is None:
            return
        lower = desc["lower"]
        opponents = desc["opponents"]
        mask = desc["mask"]

        # Recency-weight all previously learned relations and prototypes.
        pair_opposition[:] *= (1.0 - learning_rate)
        pair_opportunity[:] *= (1.0 - learning_rate)
        prototypes = [
            (m, w * (1.0 - learning_rate))
            for m, w in prototypes
            if w * (1.0 - learning_rate) > 1e-6
        ]

        for u in range(lower.size):
            for v in range(u + 1, lower.size):
                j = int(lower[u])
                k = int(lower[v])
                pair_opportunity[j, k] += learning_rate
                pair_opportunity[k, j] += learning_rate

        for u in range(opponents.size):
            for v in range(u + 1, opponents.size):
                j = int(opponents[u])
                k = int(opponents[v])
                pair_opposition[j, k] += learning_rate
                pair_opposition[k, j] += learning_rate

        if opponents.size >= 2:
            prototypes.append((mask.copy(), learning_rate))

    past_a = history.get("option_a_ratings", [])
    past_b = history.get("option_b_ratings", [])
    n_past = min(len(past_a), len(past_b))

    # Reconstruct the subject's latent coalition memory in actual trial order.
    for i in range(n_past):
        a_i = np.asarray(past_a[i], dtype=float).reshape(-1)
        b_i = np.asarray(past_b[i], dtype=float).reshape(-1)
        if a_i.size != n_features or b_i.size != n_features:
            continue
        desc_i = describe(np.vstack([a_i, b_i]))
        structure_i = relational_strength(desc_i)
        dominance_i = advertised_dominance(desc_i)
        drive = (
            gate_steepness * (structure_i - coalition_threshold)
            - dominance_suppression * dominance_i ** 2
            + hysteresis_strength * (activation - 0.5) * 4.0
        )
        drive = float(np.clip(drive, -60.0, 60.0))
        target = 1.0 / (1.0 + np.exp(-drive))
        activation = (
            activation_retention * activation
            + (1.0 - activation_retention) * target
        )
        activation = float(np.clip(activation, 0.0, 1.0))
        update_memory(desc_i)

    current = describe(stim)
    if current is None or current["leader"] is None:
        return np.array([0.5, 0.5], dtype=np.float64)

    # The current display retrieves but does not rapidly create accessibility;
    # most activation must have been consolidated over preceding trials.
    current_structure = relational_strength(current)
    current_dominance = advertised_dominance(current)
    current_drive = (
        gate_steepness * (current_structure - coalition_threshold)
        - dominance_suppression * current_dominance ** 2
        + hysteresis_strength * (activation - 0.5) * 4.0
    )
    current_drive = float(np.clip(current_drive, -60.0, 60.0))
    current_target = 1.0 / (1.0 + np.exp(-current_drive))
    activation = (
        (1.0 - current_retrieval_rate) * activation
        + current_retrieval_rate * current_target
    )
    activation = float(np.clip(activation, 0.0, 1.0))

    coalition_access = access_floor + (access_ceiling - access_floor) * activation
    coalition_access = float(np.clip(coalition_access, 0.0, 1.0))

    leader = int(current["leader"])
    leader_sign = float(current["leader_sign"])
    lower = current["lower"]

    # Rank evidence remains graded: a leader separated in advertised validity
    # from all currently discriminating lower cues receives a dominance bonus.
    validity_dominance = current_dominance
    rank_evidence = leader_sign * (1.0 + dominance_gain * validity_dominance)

    # A retrieved coalition pools lower directions without discarding their
    # advertised validities or consensus magnitude.
    if lower.size == 0:
        coalition_evidence = 0.0
    else:
        lower_weights = reliability[lower]
        coalition_evidence = float(
            np.dot(lower_weights, current["directions"][lower])
            / max(float(np.sum(lower_weights)), 1e-12)
        )

    decision_delta = (
        (1.0 - coalition_access) * rank_evidence
        + coalition_access * coalition_gain * coalition_evidence
    )

    logits = np.array([
        0.5 * sensitivity * decision_delta,
        -0.5 * sensitivity * decision_delta
    ], dtype=float)
    logits = logits - np.max(logits)
    core = np.exp(logits)
    core /= core.sum()

    probs = (1.0 - lapse) * core + lapse * np.array([0.5, 0.5])
    probs = np.asarray(probs, dtype=np.float64)
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
- n_features: n_features
- validities: validities
- learning_rate: [0.08, 0.14]
- relation_prior: [0.8, 1.6]
- recurrence_rate: [2.5, 4.5]
- coalition_threshold: [0.10, 0.16]
- gate_steepness: [14.0, 22.0]
- activation_retention: [0.78, 0.88]
- current_retrieval_rate: [0.03, 0.08]
- hysteresis_strength: [0.10, 0.30]
- dominance_suppression: [8.0, 14.0]
- initial_access: [0.05, 0.12]
- access_floor: [0.02, 0.08]
- access_ceiling: [0.94, 0.99]
- dominance_gain: [1.0, 2.2]
- coalition_gain: [3.2, 4.8]
- evidence_sensitivity: [1.0, 1.8]
- lapse: [0.0, 0.06]

`rationale`:
This is a minimal modification of the accepted coalition model. It preserves its relational memory, graded rank and coalition evidence, and history-sensitive hysteresis. Four localized changes address the critic's diagnosis. First, the coalition-access floor and initial activation are reduced sharply, preventing unlearned lower-cue consensus from producing the spurious Experiment 1 tally slope. Second, prototype similarity is raised from the third to the eighth power, so nested or partially overlapping opponent sets no longer masquerade as recurrence. Third, advertised validity dominance now enters the accessibility gate nonlinearly, rather than acting only as a modest rank-evidence bonus; this is intended to restore categorical lexical alignment in Experiment 3 and moderate excessive capture in Experiment 6. Fourth, current-display retrieval receives only a small update rate, while consolidated coalitions receive a steeper transition and a higher ceiling. This separates weak immediate consensus from strong learned influence, allowing stable coalitions to increase majority responding in Experiments 2 and 5 without globally increasing coalition gain. Hysteresis is also reduced so accidental early activation is less self-sustaining.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.4095 -> ACCEPTED
- iter 2 (current candidate you are grading): loss=0.5483 -> REJECTED
Running-best (last accepted) base: iter 1 at loss=0.4095.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 1, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0, 1]
  A=[0, 1, 1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0, 1]
  A=[1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Recode each response as agreement with Take-The-Best (TTB), then
    # estimate the within-data slope of that agreement on the direction
    # of the equal-weight tally relative to the TTB winner.
    x = []
    y = []

    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"]
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        diff = a - b

        # Features are ordered from highest to lowest advertised validity.
        discriminating = np.flatnonzero(diff != 0)
        if discriminating.size == 0:
            continue
        first = int(discriminating[0])
        ttb_winner = 0 if diff[first] > 0 else 1

        a_wins = int(np.sum(diff > 0))
        b_wins = int(np.sum(diff < 0))
        tally_margin_for_ttb = (
            a_wins - b_wins if ttb_winner == 0 else b_wins - a_wins
        )

        # -1: tally opposes TTB; 0: tally ties; +1: tally supports TTB.
        x.append(float(np.sign(tally_margin_for_ttb)))
        y.append(float(int(response) == ttb_winner))

    if len(x) < 2:
        return float("nan")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xc = x - np.mean(x)
    denom = float(np.dot(xc, xc))
    if denom <= 0:
        return float("nan")

    return float(np.dot(xc, y - np.mean(y)) / denom)
```

**Observed (real) value:** 0.0011 (var=0.0014)
**Candidate trajectory (this loop):**
  - iter 1: 0.3331 (var=0.0030) (Δ vs real +0.3321)
  - iter 2 (current): 0.2404 (var=0.0038) (Δ vs real +0.2393)
**Other theories' values on this metric (for reference):**
- pi_1: -0.0095 (var=0.0016)
- pi_2: 0.3604 (var=0.0118)
- pi_3: 0.0019 (var=0.0011)
- pi_4: 0.0091 (var=0.0018)

### Experiment 2
**Design**
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 0, 1, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 1, 0, 1, 0, 1, 1]  B=[0, 0, 0, 1, 0, 1, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 1, 1, 0, 1, 0, 1]  B=[1, 0, 0, 1, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 0, 1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 1, 1, 0, 1]  B=[0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 1, 0, 0, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 1, 0, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 1, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 0, 1, 0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0, 1, 0, 1, 1, 0]
  A=[0, 0, 1, 0, 0, 1, 0, 0, 1, 0]  B=[0, 0, 0, 1, 1, 0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    contrasts = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        cue_directions = np.sign(a - b)

        # +1 denotes an A winner and -1 a B winner.
        tally_winner = float(np.sign(cue_directions.sum()))

        discriminating = np.flatnonzero(cue_directions != 0)
        if tally_winner == 0 or discriminating.size == 0:
            continue
        ttb_winner = float(cue_directions[discriminating[0]])

        # Retain only trials on which the theories make opposite predictions.
        if tally_winner != ttb_winner:
            chosen_side = 1.0 if int(row["response"]) == 0 else -1.0
            contrasts.append(chosen_side * tally_winner)

    if len(contrasts) == 0:
        return float("nan")
    return float(np.mean(contrasts))
```

**Observed (real) value:** 0.3433 (var=0.0123)
**Candidate trajectory (this loop):**
  - iter 1: 0.1500 (var=0.0240) (Δ vs real -0.1933)
  - iter 2 (current): -0.4925 (var=0.0187) (Δ vs real -0.8358)
**Other theories' values on this metric (for reference):**
- pi_2: 0.7633 (var=0.0349)
- pi_1: -0.6767 (var=0.0457)
- pi_3: 0.3675 (var=0.0173)
- pi_4: 0.2442 (var=0.0221)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Polarity of preference for the highest-validity discriminating cue."""
    aligned = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        discriminating = np.flatnonzero(diff != 0)
        if discriminating.size == 0:
            continue

        # Features are supplied in descending validity order.
        first = int(discriminating[0])
        lexical_winner = 0 if diff[first] > 0 else 1
        aligned.append(float(int(row["response"]) == lexical_winner))

    if len(aligned) == 0:
        return 0.0

    rate = float(np.mean(aligned))
    if rate > 0.5:
        return 1.0
    if rate < 0.5:
        return -1.0
    return 0.0
```

**Observed (real) value:** 1.0000 (var=0.0000)
**Candidate trajectory (this loop):**
  - iter 1: -1.0000 (var=0.0000) (Δ vs real -2.0000)
  - iter 2 (current): -1.0000 (var=0.0000) (Δ vs real -2.0000)
**Other theories' values on this metric (for reference):**
- pi_1: 1.0000 (var=0.0000)
- pi_3: -1.0000 (var=0.0196)
- pi_2: -1.0000 (var=0.0784)
- pi_4: 1.0000 (var=0.0000)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    scores = []
    for a, b, response in zip(
        data['option_a_ratings'],
        data['option_b_ratings'],
        data['response']
    ):
        a_arr = np.asarray(a, dtype=float)
        b_arr = np.asarray(b, dtype=float)
        cue_directions = np.sign(a_arr - b_arr)
        pooled_direction = float(np.sign(cue_directions.sum()))

        # Tie-tally trials carry no pooled directional prediction.
        if pooled_direction == 0.0:
            continue

        # Positive choice direction denotes choosing A; negative denotes B.
        choice_direction = 1.0 if int(response) == 0 else -1.0
        scores.append(choice_direction * pooled_direction)

    if len(scores) == 0:
        return 0.0
    return float(np.mean(np.asarray(scores, dtype=float)))
```

**Observed (real) value:** 0.5200 (var=0.0305)
**Candidate trajectory (this loop):**
  - iter 1: 0.4550 (var=0.0150) (Δ vs real -0.0650)
  - iter 2 (current): 0.2422 (var=0.0109) (Δ vs real -0.2778)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5706 (var=0.0125)
- pi_1: 0.0139 (var=0.0047)
- pi_2: 0.7294 (var=0.0327)
- pi_4: 0.5411 (var=0.0106)

### Experiment 5
**Design**
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 1, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 1, 1, 1]  B=[1, 0, 0, 1, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if data is None or len(data) == 0:
        return float("nan")

    d = data.copy()
    # Trial position is reconstructed separately for each subject because rows
    # are supplied in within-subject trial order.
    d["trial_position"] = d.groupby("subject_id", sort=False).cumcount()

    a_sum = d["option_a_ratings"].apply(
        lambda x: float(np.sum(np.asarray(x, dtype=float)))
    )
    b_sum = d["option_b_ratings"].apply(
        lambda x: float(np.sum(np.asarray(x, dtype=float)))
    )

    # All designed trials have an unambiguous cue-majority winner. Determine
    # it from the displayed ratings so the score remains invariant to A/B
    # reversal and does not depend on memorized trial labels.
    majority_response = np.where(b_sum > a_sum, 1, 0)
    d["majority_choice"] = (
        d["response"].to_numpy(dtype=int) == majority_response
    ).astype(float)

    # Omit the first eight observations, during which the advocated model is
    # acquiring the dense-block context. The competitor is stationary, while
    # the advocated model should predominantly use compressed pooling after
    # this burn-in.
    late = d[d["trial_position"] >= 8]
    if len(late) == 0:
        late = d

    # Averaging subject-level rates makes the pooled-data definition agree
    # with the quantity recomputed on each individual subject.
    subject_rates = late.groupby("subject_id", sort=False)["majority_choice"].mean()
    if len(subject_rates) == 0:
        return float("nan")
    return float(subject_rates.mean())
```

**Observed (real) value:** 0.5773 (var=0.0012)
**Candidate trajectory (this loop):**
  - iter 1: 0.4177 (var=0.0033) (Δ vs real -0.1595)
  - iter 2 (current): 0.2166 (var=0.0031) (Δ vs real -0.3607)
**Other theories' values on this metric (for reference):**
- pi_4: 0.6725 (var=0.0026)
- pi_3: 0.2905 (var=0.0033)
- pi_1: 0.1370 (var=0.0071)
- pi_2: 0.8314 (var=0.0161)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Proportion of choices favoring the equal-weight cue-majority winner."""
    majority_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float).reshape(-1)
        b = np.asarray(row["option_b_ratings"], dtype=float).reshape(-1)
        if a.size != b.size or a.size == 0:
            continue

        majority_direction = float(np.sum(np.sign(a - b)))
        if majority_direction == 0.0:
            continue

        majority_response = 0 if majority_direction > 0.0 else 1
        response = int(row["response"])
        if response in (0, 1):
            majority_choices.append(float(response == majority_response))

    if len(majority_choices) == 0:
        return float("nan")
    return float(np.mean(majority_choices))
```

**Observed (real) value:** 0.8300 (var=0.0061)
**Candidate trajectory (this loop):**
  - iter 1: 0.9131 (var=0.0028) (Δ vs real +0.0831)
  - iter 2 (current): 0.7154 (var=0.0060) (Δ vs real -0.1146)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6185 (var=0.0035)
- pi_4: 0.1183 (var=0.0028)
- pi_1: 0.1504 (var=0.0063)
- pi_2: 0.8581 (var=0.0081)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1 (most recent)
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed structure-learning coalition family, but its arbitration is not sufficiently experiment-invariant. The largest failure is Experiment 3: humans are categorically aligned with the highest-validity discriminating cue (+1), whereas the model is categorically opposed to it (-1). Experiment 1 also fails strongly: the simulated TTB-agreement slope with tally direction is 0.333 versus an observed value near zero, showing excessive immediate sensitivity to lower-cue consensus. At the same time, coalition influence is too weak in Experiments 2 and 5: majority-oriented behavior is underestimated (0.150 versus 0.343 in Experiment 2; 0.418 versus 0.577 in Experiment 5). Experiment 6 goes in the opposite direction, with excessive majority capture (0.913 versus 0.830). Only Experiment 4 is reasonably close (0.455 versus 0.520). Thus, a global increase or decrease in coalition gain will not solve the pattern.
Rationale: Keep the relational-coalition mechanism, but recalibrate how learned structure and advertised validity dominance control access. The current high access floor (0.34–0.44), large coalition gain, and similarity-tolerant prototype retrieval allow current consensus or broadly overlapping/nested coalitions to influence choice before sufficiently diagnostic coalition learning has occurred. This likely produces the spurious tally slope in Experiment 1 and the complete majority reversal in Experiment 3. Make coalition influence more contingent on genuinely consolidated recurrence: lower the access floor substantially, reduce direct current-display activation, and require stronger membership stability or stricter prototype similarity rather than allowing pairwise overlap and nested subsets to generalize so freely. Crucially, incorporate advertised validity dominance into the accessibility gate, not merely as a modest additive bonus to rank evidence: a large leader–lower validity gap should raise the coalition threshold or suppress effective coalition access. This should restore lexical choice in Experiment 3 and moderate Experiment 6. To avoid worsening Experiments 2 and 5, compensate with a steeper recurrence-dependent transition or higher learned ceiling for stable coalitions under weak-to-moderate validity dominance. In short, separate low baseline/current-consensus influence from strong learned-coalition influence, while making dominance a nonlinear moderator. Also consider reducing hysteresis until activation is established, since the present self-reinforcement can preserve an accidentally activated coalition. The next fit should prioritize correcting Experiment 3 and flattening Experiment 1 without globally weakening the coalition signal needed in Experiments 2 and 5.

**Outcome of your advice:** iter 1 candidate loss=0.4095 -> CURRENT candidate loss=0.5483 -> the gate marked it REJECTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The current candidate is not experiment-invariant and was correctly rejected by the accept gate. Its modifications improve Experiment 1 only partially, reducing the excessive tally slope from 0.333 to 0.240, but the result remains far above the observed 0.001. More importantly, they severely damage coalition-sensitive behavior: Experiment 2 changes from modest majority preference (0.150) to strong lexical preference (-0.493 versus observed +0.343), Experiment 5 falls from 0.418 to 0.217 versus 0.577, and Experiment 4 falls from 0.455 to 0.242 versus 0.520. Experiment 6 also crosses from a small overestimate to an underestimate. Despite stronger dominance suppression and stricter matching, Experiment 3 remains categorically wrong (-1 versus +1). Thus, the previous package of lower baseline access, similarity exponent 8, weak current retrieval, and strong dominance suppression overconstrained coalition learning without fixing its most important false generalization.",
  "verdict": "regenerate",
  "rationale": "Because the previous recommendation produced a rejected candidate, do not repeat the broad push toward still stricter prototype matching, lower access, or stronger dominance suppression. Build on the unchanged accepted iteration-1 base and target a different in-family defect: latent coalition activation is currently a global license to pool all lower cues. Once activation is high, coalition_evidence averages every currently discriminating lower cue even when those cues do not match the coalition that produced activation. Hysteresis therefore transfers learned access to mismatching or nested displays, which can sustain the incorrect reversal in Experiment 3 and the tally slope in Experiment 1. Make expression of the coalition signal retrieval-specific while retaining latent hysteresis: multiply effective coalition influence by current prototype compatibility/coherence, or compute coalition evidence only over cues belonging to the retrieved prototype. Condition prototypes on the identity or rank of the leading discriminating cue so a coalition learned against one leader is not freely reused against another. Use a moderate similarity kernel near the accepted model rather than the rejected exponent-8 bottleneck, allowing stable coalitions in Experiments 2, 4, and 5 to remain learnable. In short, preserve the accepted model's learning strength and Experiment-6 fit, but separate persistent latent availability from display-specific coalition expression. This localized gating change is more diagnostic than globally weakening coalition access and should suppress inappropriate transfer in Experiments 1 and 3 without collapsing the valid coalition effects elsewhere."
}
```

## Usage

```json
{
  "input_tokens": 14777,
  "output_tokens": 1399
}
```
