# feedback_iter_02

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
- THEORY 2 = `pi_6`
- The recommendation below acts on THEORY 2 (= `pi_6`).

Replace pi_6 with a new theory of reliability-segregated consensus and conflict-triggered hierarchy protection. The theory should posit a stable highest-validity lexical anchor plus a separate, graded representation of numerical consensus. Consensus strength should depend continuously on normalized tally margin, allowing the strong margin contrast in Experiment 2, rather than being sign-normalized as in pi_4. Crucially, advertised validity should not monotonically facilitate the majority route as in pi_6. When an anti-leader coalition contains several individually diagnostic experts, the recognized high-reliability conflict should trigger scrutiny and renewed deference to the top-ranked expert; this produces the negative coalition-validity slope in Experiment 1. By contrast, approximately equivalent or low-diagnosticity lower cues should be encoded as fungible votes, so a conspicuous numerical majority can override the leader, especially in dense or structurally simple blocks, accounting for Experiments 2 and 8. Block density and recent display statistics may alter the accessibility of consensus encoding, but they should modulate rather than determine arbitration. The new account should therefore predict lexical behavior in sparse or reliability-heterogeneous settings, graded majority effects when many cues are readily compressed into votes, and hierarchy protection when strong lower experts create explicit reliability conflict. A quantitative implementation could use a persistent latent task set with lexical and graded-consensus evidence, plus an inhibitory diagnostic-conflict term based on the number and strength of anti-leader cues. It should avoid cue-identity learning without feedback, constrain stable subject heterogeneity so it does not create pi_6's excessive variance in Experiment 5, and be jointly calibrated against all ten metrics rather than tuned primarily to the dense-margin experiment.


## CANDIDATE THEORY
Reliability-Segregated Conflict Arbitration theory proposes that people preserve the advertised validity hierarchy as a stable lexical representation while separately constructing a graded representation of numerical consensus. The lexical representation retrieves the highest-validity discriminating expert. The consensus representation encodes the signed tally and its continuous normalized margin rather than merely its sign. A slowly updated task-set state, inferred from the density and margins of recently viewed displays, regulates how accessible consensus encoding is. Reliability structure determines whether lower experts are treated as fungible votes or as individually diagnostic sources. Similar-reliability experts and uniformly weak lower experts are readily compressed into votes, whereas a heterogeneous hierarchy preserves their reliability identities. When a numerical majority opposes the lexical leader, hierarchy protection depends specifically on whether diagnostic support is concentrated in a coherent high-validity prefix of the lower-cue ranking. Such a coalition is recognized as an explicit reliability conflict and renews lexical scrutiny; dispersed or low-ranked support is more readily treated as fungible consensus. Ordinary consensus accessibility is additionally nonlinear in normalized tally margin, allowing conspicuous majorities to become salient without globally increasing pooling. Consequently, increasing the reliability of a coherent anti-leader coalition can reduce majority following even though increasing its numerical margin ordinarily increases majority following. Because no outcomes are provided, the model learns only block-level display statistics and never learns cue identities or cue-response associations.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Reliability-Segregated Conflict Arbitration expects shape (2, n_features); got {stim.shape}."
        )

    n_features = int(parameters["n_features"])
    if stim.shape[1] != n_features:
        raise ValueError(
            f"Stimulus has {stim.shape[1]} features but n_features={n_features}."
        )

    validities = np.asarray(parameters["validities"], dtype=float).reshape(-1)
    if validities.size != n_features:
        raise ValueError("validities must contain one value per feature.")

    # Advertised reliabilities are represented as diagnostic log odds. No
    # feedback-dependent estimates or learned cue-identity terms are used.
    v = np.clip(validities, 0.500001, 0.999999)
    diagnosticity = np.log(v / (1.0 - v))

    # Reliability segregation describes whether the cue set naturally divides
    # into distinguishable reliability levels. Nearly equivalent cues have low
    # segregation even if all of them have fairly high advertised validity.
    mean_diagnosticity = float(np.mean(diagnosticity))
    spread = float(np.std(diagnosticity))
    segregation_scale = float(parameters["segregation_scale"])
    segregation = float(np.tanh(spread / max(segregation_scale, 1e-9)))

    # Individually diagnostic cues retain their reliability identity. The
    # smooth criterion avoids imposing an arbitrary categorical validity tier.
    cutoff = float(parameters["diagnostic_cutoff"])
    sharpness = float(parameters["diagnostic_sharpness"])
    z = np.clip(sharpness * (diagnosticity - cutoff), -60.0, 60.0)
    individual_diagnosticity = 1.0 / (1.0 + np.exp(-z))

    # Lower cues are fungible if their reliability levels are approximately
    # equivalent or if they are mostly below the diagnosticity criterion.
    top_index = int(np.argmax(validities))
    if n_features > 1:
        lower_mask = np.ones(n_features, dtype=bool)
        lower_mask[top_index] = False
        lower_strength = float(np.mean(individual_diagnosticity[lower_mask]))
    else:
        lower_strength = 0.0
    fungibility = float(np.clip(1.0 - segregation * lower_strength, 0.0, 1.0))

    # Reconstruct a persistent task-set state from displays alone. Density and
    # normalized tally margin are exponentially integrated across the block.
    span = max(float(parameters["context_memory_span"]), 1.0)
    alpha = 1.0 / span
    density_trace = float(parameters["density_prior"])
    margin_trace = float(parameters["margin_prior"])

    past_a = history.get("option_a_ratings", [])
    past_b = history.get("option_b_ratings", [])
    n_past = min(len(past_a), len(past_b))

    for i in range(n_past):
        a_i = np.asarray(past_a[i], dtype=float).reshape(-1)
        b_i = np.asarray(past_b[i], dtype=float).reshape(-1)
        if a_i.size != n_features or b_i.size != n_features:
            continue
        d_i = np.sign(a_i - b_i)
        active_i = int(np.count_nonzero(d_i))
        density_i = float(active_i) / float(max(n_features, 1))
        margin_i = (
            abs(float(np.sum(d_i))) / float(active_i)
            if active_i > 0 else 0.0
        )
        density_trace += alpha * (density_i - density_trace)
        margin_trace += alpha * (margin_i - margin_trace)

    directions = np.sign(stim[0] - stim[1])
    discriminating = np.flatnonzero(directions != 0.0)
    n_active = int(discriminating.size)
    density = float(n_active) / float(max(n_features, 1))
    raw_tally = float(np.sum(directions))
    tally_sign = float(np.sign(raw_tally))
    margin = abs(raw_tally) / float(n_active) if n_active > 0 else 0.0

    # The current display transiently moves, but does not replace, the latent
    # task set. This lets ecology modulate arbitration without determining it.
    context_density = density_trace + 0.35 * (density - density_trace)
    recent_margin = margin_trace + 0.35 * (margin - margin_trace)

    # Stable lexical representation: retrieve the most valid currently
    # discriminating expert. Stable sorting respects instructed rank on ties.
    lexical_direction = 0.0
    if n_active > 0:
        order = np.argsort(-validities, kind="stable")
        for cue in order:
            if directions[cue] != 0.0:
                lexical_direction = float(directions[cue])
                break

    # Diagnostic conflict is protected when anti-leader support forms a
    # coherent high-validity prefix of the lower-cue hierarchy. Equally large
    # coalitions dispersed among low-ranked cues remain compressible votes.
    conflict = 0.0
    if tally_sign != 0.0 and lexical_direction != 0.0 and tally_sign != lexical_direction:
        coalition = np.flatnonzero(directions == tally_sign)
        coalition_count = int(coalition.size)
        if coalition_count > 0:
            lower_order = [
                int(cue) for cue in np.argsort(-validities, kind="stable")
                if int(cue) != top_index
            ]
            lower_rank = np.full(n_features, float(n_features), dtype=float)
            for rank, cue in enumerate(lower_order):
                lower_rank[cue] = float(rank)

            prefix_span = max(float(parameters["ranked_prefix_span"]), 1e-9)
            coalition_weights = np.exp(-lower_rank[coalition] / prefix_span)
            prefix_strength = float(np.dot(
                individual_diagnosticity[coalition], coalition_weights
            ))
            ideal_weights = np.exp(
                -np.arange(coalition_count, dtype=float) / prefix_span
            )
            ideal_mass = float(np.sum(ideal_weights))
            prefix_coherence = (
                prefix_strength / ideal_mass if ideal_mass > 1e-12 else 0.0
            )
            prefix_coherence = float(np.clip(prefix_coherence, 0.0, 1.0))

            count_scale = max(float(parameters["conflict_count_scale"]), 1e-9)
            count_salience = 1.0 - np.exp(-float(coalition_count) / count_scale)
            conflict = float(segregation * prefix_coherence * count_salience)

    density_excess = max(
        context_density - float(parameters["density_threshold"]), 0.0
    )

    # Consensus salience changes nonlinearly around a psychologically visible
    # normalized margin. Fungibility modulates, but does not veto, this signal.
    margin_transition = float(parameters["margin_transition"])
    margin_steepness = float(parameters["margin_steepness"])
    margin_logit = np.clip(
        margin_steepness * (margin - margin_transition), -60.0, 60.0
    )
    visible_margin = 1.0 / (1.0 + np.exp(-margin_logit))
    effective_margin_access = visible_margin * (0.65 + 0.35 * fungibility)

    # Consensus access receives independent contributions from the persistent
    # dense-task set, visible graded margin, recent margin statistics, and cue
    # fungibility. Coherent ranked conflict selectively inhibits access.
    gate_logit = (
        float(parameters["base_consensus_access"])
        + float(parameters["density_access_gain"]) * density_excess
        + float(parameters["margin_access_gain"]) * effective_margin_access
        + float(parameters["recent_margin_gain"]) * recent_margin
        + float(parameters["fungibility_gain"]) * fungibility
        - float(parameters["conflict_inhibition"]) * conflict
    )
    gate_logit = float(np.clip(gate_logit, -60.0, 60.0))
    consensus_probability = 1.0 / (1.0 + np.exp(-gate_logit))

    # Conflict also renews attention to the instructed hierarchy. This is a
    # protection response, not negative weighting of reliable coalition cues.
    lexical_evidence = lexical_direction * (
        1.0 + float(parameters["hierarchy_protection_gain"]) * conflict
    )

    # Numerical consensus retains continuous normalized margin. Thus a 9-1
    # split is represented more strongly than a 6-4 split even though both
    # have the same tally sign.
    if tally_sign == 0.0:
        consensus_evidence = 0.0
    else:
        transformed_margin = margin ** float(parameters["consensus_margin_exponent"])
        consensus_evidence = (
            tally_sign
            * float(parameters["consensus_evidence_scale"])
            * transformed_margin
        )

    lexical_sensitivity = float(parameters["lexical_sensitivity"])
    consensus_sensitivity = float(parameters["consensus_sensitivity"])

    def binary_probs(delta, sensitivity):
        logits = np.array([
            0.5 * sensitivity * delta,
            -0.5 * sensitivity * delta
        ], dtype=float)
        logits -= np.max(logits)
        p = np.exp(logits)
        p /= p.sum()
        return p

    p_lexical = binary_probs(lexical_evidence, lexical_sensitivity)
    p_consensus = binary_probs(consensus_evidence, consensus_sensitivity)
    p_core = (
        (1.0 - consensus_probability) * p_lexical
        + consensus_probability * p_consensus
    )

    lapse = float(parameters["lapse"])
    probs = (1.0 - lapse) * p_core + lapse * np.array([0.5, 0.5])
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
- segregation_scale: [0.45, 0.65]
- diagnostic_cutoff: [0.75, 0.95]
- diagnostic_sharpness: [6.0, 9.0]
- context_memory_span: [16.0, 22.0]
- density_prior: [0.48, 0.54]
- margin_prior: [0.15, 0.21]
- density_threshold: [0.56, 0.62]
- base_consensus_access: [-2.0, -1.7]
- density_access_gain: [5.5, 6.5]
- margin_access_gain: [3.5, 4.5]
- recent_margin_gain: [0.5, 1.0]
- fungibility_gain: [1.3, 1.8]
- ranked_prefix_span: [0.9, 1.3]
- conflict_count_scale: [1.5, 2.2]
- conflict_inhibition: [5.0, 7.0]
- hierarchy_protection_gain: [0.5, 1.0]
- margin_transition: [0.23, 0.26]
- margin_steepness: [15.0, 19.0]
- consensus_margin_exponent: [0.90, 1.10]
- consensus_evidence_scale: [4.5, 5.5]
- lexical_sensitivity: [2.9, 3.3]
- consensus_sensitivity: [1.4, 1.8]
- lapse: [0.02, 0.06]

`rationale`:
This is a minimal edit of the accepted candidate. The global reliability representation, task-set traces, lexical and consensus routes, conflict inhibition, and graded consensus evidence are retained. Only two selectivity mechanisms change. First, conflict strength now depends on diagnostic anti-leader support occupying a short high-validity lower-cue prefix. This preserves hierarchy protection for recurring structured coalitions such as Experiment 8 and for reliability conflicts in Experiments 1 and 9, while releasing consensus when support is dispersed across lower ranks, as required in Experiments 2, 5, and 6. Second, the linear margin contribution to consensus access is replaced by a steep but continuous sigmoid around a normalized margin of approximately 0.25, weakly modulated by fungibility. This selectively boosts conspicuous 5-3 and larger majorities, while changing small dense-block margins and Experiment 7 much less. Consensus evidence itself remains continuously graded, preserving the Experiment 10 contrast. A small narrowing and increase of lexical sensitivity makes ordinary lexical responses more stable, helping flatten Experiment 1 and protect Experiment 8 without introducing new subject-level variance. No cue-identity or response learning was added.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.2756 -> ACCEPTED
- iter 2: loss=0.3578 -> REJECTED
- iter 3 (current candidate you are grading): loss=0.2587 -> ACCEPTED
Running-best (last accepted) base: iter 3 at loss=0.2587.

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
  - iter 1: 0.1571 (var=0.0018) (Δ vs real +0.1560)
  - iter 2: 0.2130 (var=0.0017) (Δ vs real +0.2120)
  - iter 3 (current): 0.1974 (var=0.0018) (Δ vs real +0.1963)
**Other theories' values on this metric (for reference):**
- pi_1: -0.0095 (var=0.0016)
- pi_2: 0.3604 (var=0.0118)
- pi_3: 0.0019 (var=0.0011)
- pi_4: 0.0091 (var=0.0018)
- pi_5: 0.3640 (var=0.0028)
- pi_6: 0.3328 (var=0.0017)

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
  - iter 1: -0.4758 (var=0.0233) (Δ vs real -0.8192)
  - iter 2: -0.4517 (var=0.0240) (Δ vs real -0.7950)
  - iter 3 (current): -0.0575 (var=0.0212) (Δ vs real -0.4008)
**Other theories' values on this metric (for reference):**
- pi_2: 0.7633 (var=0.0349)
- pi_1: -0.6767 (var=0.0457)
- pi_3: 0.3675 (var=0.0173)
- pi_4: 0.2442 (var=0.0221)
- pi_5: 0.3233 (var=0.0234)
- pi_6: -0.1700 (var=0.0303)

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
  - iter 1: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 2: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 3 (current): 1.0000 (var=0.0000) (Δ vs real +0.0000)
**Other theories' values on this metric (for reference):**
- pi_1: 1.0000 (var=0.0000)
- pi_3: -1.0000 (var=0.0196)
- pi_2: -1.0000 (var=0.0784)
- pi_4: 1.0000 (var=0.0000)
- pi_5: -1.0000 (var=0.0000)
- pi_6: 1.0000 (var=0.1536)

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
  - iter 1: 0.5522 (var=0.0113) (Δ vs real +0.0322)
  - iter 2: 0.1250 (var=0.0086) (Δ vs real -0.3950)
  - iter 3 (current): 0.3022 (var=0.0237) (Δ vs real -0.2178)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5706 (var=0.0125)
- pi_1: 0.0139 (var=0.0047)
- pi_2: 0.7294 (var=0.0327)
- pi_4: 0.5411 (var=0.0106)
- pi_5: 0.4717 (var=0.0088)
- pi_6: 0.6739 (var=0.0057)

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
  - iter 1: 0.2995 (var=0.0041) (Δ vs real -0.2777)
  - iter 2: 0.3302 (var=0.0026) (Δ vs real -0.2470)
  - iter 3 (current): 0.2216 (var=0.0020) (Δ vs real -0.3557)
**Other theories' values on this metric (for reference):**
- pi_4: 0.6725 (var=0.0026)
- pi_3: 0.2905 (var=0.0033)
- pi_1: 0.1370 (var=0.0071)
- pi_2: 0.8314 (var=0.0161)
- pi_5: 0.4523 (var=0.0026)
- pi_6: 0.5434 (var=0.0026)

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
  - iter 1: 0.2677 (var=0.0033) (Δ vs real -0.5623)
  - iter 2: 0.3835 (var=0.0028) (Δ vs real -0.4465)
  - iter 3 (current): 0.4542 (var=0.0039) (Δ vs real -0.3758)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6185 (var=0.0035)
- pi_4: 0.1183 (var=0.0028)
- pi_1: 0.1504 (var=0.0063)
- pi_2: 0.8581 (var=0.0081)
- pi_5: 0.8013 (var=0.0034)
- pi_6: 0.5779 (var=0.0042)

### Experiment 7
**Design**
  A=[1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0]  B=[0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0]  B=[0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1]  B=[0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1]  B=[0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0]
  A=[1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1]  B=[0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1]  B=[1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1]  B=[1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1]  B=[1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0]  B=[1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1]
  A=[0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1]  B=[1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1]  B=[1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if data is None or len(data) == 0:
        return float('nan')

    # Omit only the brief initialization period in which Ecological Encoding
    # Hysteresis is still updating its estimate of the block ecology.
    if 'subject_id' in data.columns:
        serial_position = data.groupby('subject_id', sort=False).cumcount().to_numpy()
    else:
        serial_position = np.arange(len(data))
    late = serial_position >= 4

    scores = []
    for keep, a_cell, b_cell, response in zip(
        late,
        data['option_a_ratings'],
        data['option_b_ratings'],
        data['response']
    ):
        if not keep or pd.isna(response):
            continue
        a = np.asarray(a_cell, dtype=float).reshape(-1)
        b = np.asarray(b_cell, dtype=float).reshape(-1)
        if a.size == 0 or a.size != b.size:
            continue
        difference = float(np.sum(a) - np.sum(b))
        if difference == 0.0 or not np.isfinite(difference):
            continue
        majority_response = 0 if difference > 0.0 else 1
        scores.append(float(int(response) == majority_response))

    if len(scores) == 0:
        return float('nan')
    return float(np.mean(np.asarray(scores, dtype=float)))
```

**Observed (real) value:** 0.6404 (var=0.0014)
**Candidate trajectory (this loop):**
  - iter 1: 0.7220 (var=0.0039) (Δ vs real +0.0815)
  - iter 2: 0.7393 (var=0.0021) (Δ vs real +0.0989)
  - iter 3 (current): 0.7230 (var=0.0034) (Δ vs real +0.0826)
**Other theories' values on this metric (for reference):**
- pi_4: 0.6617 (var=0.0029)
- pi_5: 0.0711 (var=0.0013)
- pi_1: 0.1522 (var=0.0140)
- pi_2: 0.8537 (var=0.0083)
- pi_3: 0.8793 (var=0.0044)
- pi_6: 0.5474 (var=0.0043)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 0, 1]  B=[0, 1, 1, 1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1, 1, 0, 1]  B=[0, 1, 1, 1, 0, 1, 0, 1]
  A=[1, 0, 0, 0, 1, 1, 1, 0]  B=[0, 1, 1, 1, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 1]  B=[1, 0, 0, 0, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0, 0, 1, 0]  B=[1, 0, 0, 0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 0, 0, 1, 1]  B=[1, 0, 0, 0, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0, 1, 0, 1]  B=[1, 0, 0, 0, 1, 1, 0, 1]
  A=[0, 1, 1, 1, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0, 1, 1, 1]  B=[1, 0, 0, 0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Acquisition-weighted capture by the recurring anti-leader coalition."""
    if data is None or len(data) == 0:
        return float("nan")

    scores = []
    for _, subj in data.groupby("subject_id", sort=False):
        coalition_choices = []
        for _, row in subj.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float).reshape(-1)
            b = np.asarray(row["option_b_ratings"], dtype=float).reshape(-1)
            if a.size == 0 or b.size == 0 or a[0] == b[0]:
                continue

            # Expert 1 is the highest-validity cue. The coalition winner is
            # the option opposed to the option favored by that expert.
            leader_response = 0 if a[0] > b[0] else 1
            response = int(row["response"])
            coalition_choices.append(float(response != leader_response))

        n = len(coalition_choices)
        if n == 0:
            continue

        choices = np.asarray(coalition_choices, dtype=float)
        early_n = min(8, n)
        late_n = min(32, n)
        early_rate = float(np.mean(choices[:early_n]))
        late_rate = float(np.mean(choices[-late_n:]))

        # Late coalition capture plus a modest acquisition bonus.
        scores.append(late_rate + 0.25 * (late_rate - early_rate))

    if not scores:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** 0.1631 (var=0.0139)
**Candidate trajectory (this loop):**
  - iter 1: 0.1336 (var=0.0104) (Δ vs real -0.0295)
  - iter 2: 0.3578 (var=0.0142) (Δ vs real +0.1947)
  - iter 3 (current): 0.1406 (var=0.0070) (Δ vs real -0.0225)
**Other theories' values on this metric (for reference):**
- pi_5: 0.9923 (var=0.0035)
- pi_4: 0.1520 (var=0.0100)
- pi_1: 0.1422 (var=0.0186)
- pi_2: 0.8414 (var=0.0147)
- pi_3: 0.1195 (var=0.0100)
- pi_6: 0.1853 (var=0.0081)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 1, 0, 1]  B=[0, 1, 1, 0, 0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 1, 0, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 1, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 1, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 1, 1, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.90, 0.84, 0.79, 0.74, 0.69, 0.64, 0.60, 0.57, 0.54, 0.52], dtype=float)
    diagnosticity = np.log(validities / (1.0 - validities))

    def subject_slope(df):
        df = df.reset_index(drop=True)
        n = len(df)
        if n < 4:
            return np.nan

        coalition_scores = np.empty(n, dtype=float)
        majority_b = np.empty(n, dtype=float)
        majority_choice = np.empty(n, dtype=float)

        for i in range(n):
            a = np.asarray(df.loc[i, "option_a_ratings"], dtype=float)
            b = np.asarray(df.loc[i, "option_b_ratings"], dtype=float)
            directions = np.sign(a - b)
            active = directions != 0.0
            tally = float(np.sum(directions))
            majority_sign = float(np.sign(tally))

            if majority_sign == 0.0 or not np.any(active):
                coalition_scores[i] = 0.0
                majority_b[i] = 0.5
                majority_choice[i] = 0.0
                continue

            denom = float(np.sum(diagnosticity[active]))
            coalition_scores[i] = majority_sign * float(
                np.dot(diagnosticity[active], directions[active])
            ) / denom

            maj_b = 1.0 if majority_sign < 0.0 else 0.0
            majority_b[i] = maj_b
            majority_choice[i] = 1.0 if float(df.loc[i, "response"]) == maj_b else 0.0

        rounded = np.round(coalition_scores, 10)
        levels = np.sort(np.unique(rounded))
        if levels.size < 2:
            return np.nan

        rank_map = {float(v): float(j) for j, v in enumerate(levels)}
        x = np.array([rank_map[float(v)] for v in rounded], dtype=float)
        x -= (levels.size - 1.0) / 2.0

        # Remove assignment to short serial-position strata, as well as any
        # incidental association with which screen side contains the majority.
        bins = np.arange(n, dtype=int) // 6
        unique_bins = np.unique(bins)
        design = np.zeros((n, unique_bins.size + 1), dtype=float)
        for j, value in enumerate(unique_bins):
            design[:, j] = (bins == value).astype(float)
        design[:, -1] = majority_b

        beta = np.linalg.lstsq(design, x, rcond=None)[0]
        x_resid = x - design.dot(beta)
        denom = float(np.dot(x_resid, x_resid))
        if denom <= 1e-12:
            return np.nan

        return float(np.dot(x_resid, majority_choice) / denom)

    if "subject_id" in data.columns:
        values = []
        for _, subject_df in data.groupby("subject_id", sort=False):
            value = subject_slope(subject_df)
            if np.isfinite(value):
                values.append(value)
        return float(np.mean(values)) if values else np.nan

    return float(subject_slope(data))

```

**Observed (real) value:** -0.1996 (var=0.0064)
**Candidate trajectory (this loop):**
  - iter 1: -0.1244 (var=0.0022) (Δ vs real +0.0752)
  - iter 2: -0.1343 (var=0.0015) (Δ vs real +0.0653)
  - iter 3 (current): -0.1543 (var=0.0018) (Δ vs real +0.0453)
**Other theories' values on this metric (for reference):**
- pi_4: -0.0002 (var=0.0023)
- pi_6: 0.0516 (var=0.0017)
- pi_1: -0.0045 (var=0.0013)
- pi_2: -0.0036 (var=0.0011)
- pi_3: 0.0020 (var=0.0015)
- pi_5: 0.2398 (var=0.0019)

### Experiment 10
**Design**
  A=[1, 0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 1, 0, 1, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 0, 1, 0, 0, 1]  B=[0, 1, 1, 0, 1, 1, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 0, 1, 1, 0, 1]  B=[1, 0, 1, 0, 0, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 1, 0, 1, 1, 0]  B=[1, 0, 0, 1, 0, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 0, 1, 0, 1]  B=[1, 0, 0, 0, 0, 0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if len(data) == 0:
        return float('nan')

    # Code whether the response followed the highest-validity cue (cue 1).
    lexical = []
    small_margin = []
    for a_cell, b_cell, response in zip(
        data['option_a_ratings'], data['option_b_ratings'], data['response']
    ):
        a = np.asarray(a_cell, dtype=float).reshape(-1)
        b = np.asarray(b_cell, dtype=float).reshape(-1)
        directions = np.sign(a - b)

        cue1_direction = directions[0]
        r = int(response)
        followed_cue1 = ((cue1_direction > 0 and r == 0) or
                         (cue1_direction < 0 and r == 1))
        lexical.append(float(followed_cue1))

        n_active = int(np.count_nonzero(directions))
        margin = (abs(float(np.sum(directions))) / float(n_active)
                  if n_active > 0 else 0.0)
        # Separates the subthreshold 6-4 majority from all larger majorities.
        small_margin.append(float(margin < 0.3))

    work = pd.DataFrame({
        'subject_id': data['subject_id'].to_numpy(),
        'lexical': np.asarray(lexical, dtype=float),
        'small': np.asarray(small_margin, dtype=float)
    })

    # Rows are in trial order within subject. Twelve-trial strata remove the
    # large common acquisition trajectory while retaining six repetitions per
    # stimulus and adequate observations from both margin classes per stratum.
    work['position'] = work.groupby('subject_id', sort=False).cumcount()
    work['time_bin'] = (work['position'] // 12).astype(int)
    bin_mean = work.groupby(
        ['subject_id', 'time_bin'], sort=False
    )['small'].transform('mean')
    residual_small = work['small'].to_numpy(dtype=float) - bin_mean.to_numpy(dtype=float)
    y = work['lexical'].to_numpy(dtype=float)

    denominator = float(np.dot(residual_small, residual_small))
    if denominator <= 1e-12:
        x = work['small'].to_numpy(dtype=float)
        x = x - float(np.mean(x))
        denominator = float(np.dot(x, x))
        if denominator <= 1e-12:
            return 0.0
        return float(np.dot(x, y) / denominator)

    return float(np.dot(residual_small, y) / denominator)
```

**Observed (real) value:** 0.3168 (var=0.0102)
**Candidate trajectory (this loop):**
  - iter 1: 0.2342 (var=0.0093) (Δ vs real -0.0826)
  - iter 2: 0.1723 (var=0.0065) (Δ vs real -0.1445)
  - iter 3 (current): 0.4105 (var=0.0123) (Δ vs real +0.0937)
**Other theories' values on this metric (for reference):**
- pi_6: 0.3067 (var=0.0079)
- pi_4: 0.0268 (var=0.0100)
- pi_1: 0.0091 (var=0.0063)
- pi_2: 0.0179 (var=0.0059)
- pi_3: -0.0022 (var=0.0102)
- pi_5: 0.0194 (var=0.0036)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate captures several important signatures of the prescribed family: lexical dominance is correct in Experiment 3; anti-leader capture is close in Experiment 8; the coalition-validity effect has the correct negative sign in Experiment 9; and the graded-margin effect is reasonably close in Experiment 10. Experiment 4 is also well fit. However, it fails the central experiment-invariance requirement. It remains much too lexical in Experiments 2, 5, and 6: Experiment 2 reverses the observed majority preference (candidate -0.4758 versus observed +0.3433), while majority choice is severely underestimated in Experiment 5 (0.2995 versus 0.5773) and Experiment 6 (0.2677 versus 0.8300). At the same time, tally influence is too strong in Experiment 1 (slope 0.1571 versus approximately zero), and majority use is somewhat excessive in Experiment 7 (0.7220 versus 0.6404). Thus, a uniform increase in consensus access would not solve the pattern; consensus must become more selectively available for fungible lower cues while hierarchy protection becomes more selective and effective for genuinely diagnostic coalitions.
Rationale: The accepted candidate is a useful in-family base, but its conflict detector appears miscalibrated. In particular, global `segregation` is computed across all cues, including the top lexical anchor. A large leader-to-lower-cue gap can therefore make the entire set look segregated, lower `fungibility`, and activate conflict inhibition even when the lower cues themselves are approximately equivalent or weak—the exact situations that should support pooling in Experiments 2, 5, and especially 6. Revise segregation/fungibility to characterize the lower-cue set separately from the stable leader. Conflict should depend on the count or summed strength of anti-leader lower cues that individually exceed the diagnostic threshold, together with coalition-specific reliability heterogeneity, rather than global spread times the coalition's mean diagnosticity. This should release consensus for weak or mutually similar coalitions without abandoning hierarchy protection. Once that selectivity is fixed, modestly increase fungibility-driven consensus access or lower the density threshold so dense and structurally simple blocks reach the observed majority rates. Do not simply raise baseline or margin access globally, because that would worsen Experiments 1 and 7. For Experiment 1, make the qualified diagnostic-conflict response more sharply count-sensitive or let qualified conflict inhibit the margin contribution, so increasing an opposing diagnostic coalition does not leave the positive tally slope of 0.157. Preserve the current negative effect in Experiment 9 and the good lexical fits in Experiments 3 and 8. Finally, after correcting mean fit, narrow the context/access ranges if needed: between-subject variance remains too high in Experiments 2 and 5 relative to the data.

**Outcome of your advice:** iter 1 candidate loss=0.2756 -> iter 2 candidate loss=0.3578 -> the gate marked it REJECTED.

### Iteration 2 (most recent)
Verdict: regenerate
Interpretation: The current candidate is not an improvement over the accepted base: aggregate loss rose from 0.2756 to 0.3578, so the gate correctly rejected it. The revised lower-cue segregation rule produced only modest gains in the intended experiments—Experiment 2 remains directionally wrong (-0.4517 versus +0.3433), and majority choice remains far too low in Experiments 5 (0.3302 versus 0.5773) and 6 (0.3835 versus 0.8300). Those small gains came at substantial cost elsewhere. Experiment 4 deteriorated from a close 0.5522 to 0.1250 against 0.5200, Experiment 8's coalition capture rose from a close 0.1336 to 0.3578 against 0.1631, Experiment 1's excessive tally slope worsened from 0.1571 to 0.2130 against approximately zero, and Experiment 10's graded-margin signature weakened from 0.2342 to 0.1723 against 0.3168. Experiment 7 also remains somewhat too majority-oriented. The candidate still preserves lexical polarity in Experiment 3 and the negative coalition-validity slope in Experiment 9, but it does not achieve experiment-invariant arbitration.
Rationale: Because this iteration was rejected, build on the unchanged iteration-1 base rather than extending the rejected lower-cue-spread revision. My previous recommendation to define fungibility through lower-set or coalition-specific reliability heterogeneity did not help: it released consensus on the wrong displays, especially Experiment 8, while barely addressing Experiments 2, 5, and 6. Try a different in-family selectivity variable. Preserve the accepted base's conflict machinery and characterize protected conflict by reliability-ranked coalition coherence—for example, the concentration of anti-leader support within a short high-validity lower-cue prefix—rather than by the standard deviation of all lower cues or the within-coalition spread. This can protect the recurring high-ranked anti-leader configuration in Experiment 8 and structured diagnostic conflicts in Experiments 1 and 9 while treating dispersed or predominantly low-ranked support as fungible votes. Separately, make ordinary consensus access more nonlinear in normalized margin: use a steeper thresholded/saturating margin transformation or a margin-by-fungibility interaction so conspicuous majorities can dominate in Experiments 2, 5, and 6 without uniformly increasing baseline consensus. Keep diagnostic conflict as a direct lexical-protection term, so stronger reliable conflict can still suppress the tally effect in Experiment 1. Avoid the current combination in which conflict simultaneously suppresses the gate, suppresses margin, and amplifies lexical evidence; that triple action makes calibration brittle. A cleaner division is graded margin/fungibility controlling consensus access and one conflict-dependent hierarchy-protection term controlling lexical resistance. Preserve the accepted base's Experiment 4 and 8 settings, retain the successful negative sign in Experiment 9, and restore rather than weaken Experiment 10's margin contrast. Parameter ranges should remain narrow because the present variances are mostly acceptable; the principal problem is mean-pattern selectivity, not heterogeneity.

**Outcome of your advice:** iter 2 candidate loss=0.3578 -> CURRENT candidate loss=0.2587 -> the gate marked it ACCEPTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The accepted revision is a genuine improvement and validates the prior recommendation: ranked-prefix conflict preserved the close Experiment 8 fit and strengthened the negative coalition-validity effect in Experiment 9, while nonlinear margin access moved Experiment 2 substantially toward the observed majority preference and improved Experiment 6. Experiment 3 remains exact. Nevertheless, the model is not yet experiment-invariant. It remains too lexical in Experiment 2 (-0.0575 versus 0.3433), Experiment 5 (0.2216 versus 0.5773), Experiment 6 (0.4542 versus 0.8300), and Experiment 4 (0.3022 versus 0.5200). Conversely, tally influence in Experiment 1 remains much too positive (0.1974 versus 0.0011), Experiment 7 is somewhat too majority-oriented (0.7230 versus 0.6404), and the graded-margin contrast now overshoots in Experiment 10 (0.4105 versus 0.3168). The largest improvement therefore came from the correct in-family selectivity idea, but its threshold and conflict implementation are still miscalibrated.",
  "verdict": "regenerate",
  "rationale": "Continue from this accepted ranked-coherence/nonlinear-margin solution rather than revisiting the rejected lower-set heterogeneity proposal. First fix a likely structural defect in the conflict calculation: `prefix_coherence` divides by an `ideal_mass` that grows with total coalition size. Adding low-ranked anti-leader cues can therefore dilute an already coherent diagnostic prefix, causing hierarchy protection to weaken precisely as the opposing tally grows. That is a plausible source of Experiment 1's large positive tally slope. Normalize prefix strength against a fixed prefix capacity, or compute saturating occupancy of the first few diagnostic lower ranks independently of additional low-ranked supporters. Qualified conflict should then remain stable or increase when extra votes are added. This should flatten Experiment 1 while preserving Experiment 8 and the successful negative sign in Experiment 9; avoid a global increase in conflict inhibition, which would suppress the needed pooling elsewhere. Second, broaden and modestly lower the currently narrow margin-access transition rather than increasing baseline consensus. The steep threshold at roughly 0.25 strongly discounts Experiment 5's recurring 5-4 margins and many moderate-majority trials in Experiments 4 and 6. A somewhat lower transition with reduced steepness should raise consensus for these moderate margins, further improve Experiment 2, and reduce the excessive categorical separation now seen in Experiment 10. Keep the change modest because Experiment 7 is already too majority-oriented. If additional selectivity is needed, let fungibility scale the broadened subthreshold part of margin access, while conspicuous margins retain access across cue structures. Finally, after improving the means, narrow the margin/access parameter ranges: Experiment 2's between-subject variance remains appreciably above the observed value. The prescribed reliability-segregated lexical-plus-graded-consensus family should be retained; the remaining failures concern conflict normalization and access calibration, not the mechanism family."
}
```

## Usage

```json
{
  "input_tokens": 23466,
  "output_tokens": 1939
}
```
