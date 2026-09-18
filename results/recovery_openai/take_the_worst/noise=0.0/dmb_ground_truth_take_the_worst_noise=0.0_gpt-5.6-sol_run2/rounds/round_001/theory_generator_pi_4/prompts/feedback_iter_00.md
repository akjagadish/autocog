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
- THEORY 1 = `pi_1`
- THEORY 2 = `pi_3`
- The recommendation below acts on THEORY 1 (= `pi_1`).

Replace pi_1 with a Context-Calibrated Evidence-Encoding theory. The central claim should be that people calibrate an encoding strategy from the recent block-level ecology, rather than always using one cue or arbitrating solely from current-trial load. Tie-rich, sparsely discriminating blocks make the validity hierarchy easy to maintain and promote a persistent ordered-search task set; this yields robust lexical choices in Experiment 1, including occasional dense trials embedded in that context. Blocks in which nearly every cue repeatedly discriminates encourage a compressed relational or gist representation, so cue directions are pooled and majority effects emerge, as in Experiment 2. Dense conflict structures should likewise produce the majority preference in Experiment 4, while mixed or tie-rich environments can yield the negligible tally modulation seen in Experiment 3. The model should include a slowly updated contextual state based on recent discrimination density, tie prevalence, and possibly stability of the cue-ranking shortcut; subject-specific thresholds or learning rates can capture the elevated observed variance in Experiment 2. Conditional on the contextual state, choices can interpolate between lexicographic and approximately equal-direction pooling, with sensitivity and lapse noise. This differs from pi_1 by allowing genuine multi-cue influence and differs from pi_3 by making arbitration history- and context-dependent rather than a memory-failure gate determined anew by cue count and current comparison load. A critical future test should present identical diagnostic trials inside sparse versus saturated filler blocks: the new theory predicts context-dependent reversals, whereas pi_1 predicts invariant lexical choice and pi_3's current implementation predicts identical choices for identical trials.


## CANDIDATE THEORY
Ecological Encoding Hysteresis theory claims that decision makers learn how to encode an entire block of comparisons. They begin with a sparse-ecology prior favoring maintenance of the advertised validity hierarchy. After each comparison, they update a slowly changing estimate of the proportion of cues that discriminate. When discrimination remains sparse and ties are common, cue ranks remain cognitively useful and an ordered-search task set persists: the highest-validity discriminating cue controls choice, even on an occasional dense trial. When comparisons repeatedly contain many simultaneous discriminations, maintaining separate cue ranks becomes inefficient. People then compress cue directions into a relational gist and approximately pool them with equal weights. A steep contextual transition and a ceiling on compression implement hysteresis-like persistence rather than trial-by-trial load arbitration. Individual differences in the contextual threshold, prior strength, compression ceiling, sensitivity, and lapses produce heterogeneous behavior near the transition. Thus, identical diagnostic comparisons can elicit lexical choices in sparse filler blocks but majority choices in saturated filler blocks.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Ecological Encoding Hysteresis expects shape (2, n_features); got {stim.shape}."
        )

    n_features = int(parameters["n_features"])
    if stim.shape[1] != n_features:
        raise ValueError(
            f"Stimulus has {stim.shape[1]} features but n_features={n_features}."
        )

    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError("validities must contain one value per feature.")

    # Reconstruct the slowly accumulated block ecology. Each observation is
    # the fraction of cues that discriminate in a comparison; its complement
    # is tie prevalence. A prior prevents one unusual early trial from
    # immediately replacing the initial ordered-search task set.
    density_sum = 0.0
    n_observed = 0
    past_a = history.get("option_a_ratings", [])
    past_b = history.get("option_b_ratings", [])
    n_past = min(len(past_a), len(past_b))

    for i in range(n_past):
        a_i = np.asarray(past_a[i], dtype=float).reshape(-1)
        b_i = np.asarray(past_b[i], dtype=float).reshape(-1)
        if a_i.size != n_features or b_i.size != n_features:
            continue
        density_sum += float(np.mean(a_i != b_i))
        n_observed += 1

    # Viewing the current display supplies another ecological observation,
    # but its effect shrinks as evidence about the block accumulates.
    current_density = float(np.mean(stim[0] != stim[1]))
    density_sum += current_density
    n_observed += 1

    prior_density = float(parameters["context_prior"])
    prior_strength = float(parameters["prior_strength"])
    context_density = (
        prior_strength * prior_density + density_sum
    ) / (prior_strength + float(n_observed))

    # Saturated ecologies trigger a compressed relational encoding. The
    # ceiling allows some ordered processing to survive even in dense blocks.
    threshold = float(parameters["context_threshold"])
    steepness = float(parameters["context_steepness"])
    gate_logit = steepness * (context_density - threshold)
    gate_logit = float(np.clip(gate_logit, -60.0, 60.0))
    compression_activation = 1.0 / (1.0 + np.exp(-gate_logit))
    pooling_probability = (
        float(parameters["pooling_ceiling"]) * compression_activation
    )

    directions = np.sign(stim[0] - stim[1])

    # Ordered encoding: consult cues by advertised validity and stop at the
    # first discrimination.
    lexical_delta = 0.0
    cue_order = np.argsort(-validities, kind="stable")
    for cue in cue_order:
        if directions[cue] != 0.0:
            lexical_delta = float(directions[cue])
            break

    # Compressed encoding: retain all cue directions but discard fine-grained
    # rank and magnitude information. Sign normalization puts its response
    # scale on the same footing as the lexical representation.
    pooled_delta = float(np.sign(np.sum(directions)))

    sensitivity = float(parameters["evidence_sensitivity"])

    def choice_probs(delta):
        logits = np.array([
            0.5 * sensitivity * delta,
            -0.5 * sensitivity * delta
        ], dtype=float)
        logits = logits - np.max(logits)
        p = np.exp(logits)
        p /= p.sum()
        return p

    p_ordered = choice_probs(lexical_delta)
    p_compressed = choice_probs(pooled_delta)
    p_core = (
        (1.0 - pooling_probability) * p_ordered
        + pooling_probability * p_compressed
    )

    lapse = float(parameters["lapse"])
    probs = (1.0 - lapse) * p_core + lapse * np.array([0.5, 0.5])
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
    return probs / total


`policy(probs) -> int`:
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- n_features: n_features
- validities: validities
- context_prior: [0.50, 0.60]
- prior_strength: [1.0, 3.0]
- context_threshold: [0.68, 0.71]
- context_steepness: [35.0, 55.0]
- pooling_ceiling: [0.68, 0.76]
- evidence_sensitivity: [1.6, 3.0]
- lapse: [0.0, 0.08]

`rationale`:
The model addresses the central failure of invariant Take-The-Best by permitting genuine lower-cue influence, but only after the block ecology induces a compressed encoding. Experiment 1 has a relatively tie-rich average ecology, so its contextual estimate remains below threshold and even dense embedded trials are usually processed lexicographically, yielding negligible tally modulation. Experiment 2 repeatedly presents high discrimination density, rapidly moving the contextual state above threshold; the resulting mixture is majority-favoring but retains enough ordered processing to avoid the excessive tally contrast of a pure equal-weight rule. Experiment 3 averages below the transition despite containing comparisons with several discriminating cues, preserving a reliably lexical preference. Experiment 4 is maximally saturated, so compression approaches its subject-specific ceiling and produces the observed majority preference. Unlike a current-load gate, the decision rule for a stimulus depends on preceding filler trials. The model therefore predicts that placing identical conflict trials in sparse versus saturated blocks will reverse their dominant influence from the validity-leading cue toward the cue majority.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1 (current candidate you are grading): loss=0.0191 -> ACCEPTED
Running-best (last accepted) base: iter 1 at loss=0.0191.

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
**Candidate (simulated) value:** -0.0048 (var=0.0013)
**Other theories' values on this metric (for reference):**
- pi_1: -0.0095 (var=0.0016)
- pi_2: 0.3604 (var=0.0118)
- pi_3: 0.0019 (var=0.0011)

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
**Candidate (simulated) value:** 0.2750 (var=0.0178)
**Other theories' values on this metric (for reference):**
- pi_2: 0.7633 (var=0.0349)
- pi_1: -0.6767 (var=0.0457)
- pi_3: 0.3675 (var=0.0173)

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
**Candidate (simulated) value:** 1.0000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_1: 1.0000 (var=0.0000)
- pi_3: -1.0000 (var=0.0196)
- pi_2: -1.0000 (var=0.0784)

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
**Candidate (simulated) value:** 0.5294 (var=0.0154)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5706 (var=0.0125)
- pi_1: 0.0139 (var=0.0047)
- pi_2: 0.7294 (var=0.0327)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate captures the central cross-experiment dissociation within the prescribed context-calibrated encoding family. Experiment 1 remains essentially tally-invariant: the simulated slope is -0.0048 versus 0.0011 observed, with nearly exact between-subject variance (0.0013 versus 0.0014). Experiment 3 correctly preserves a lexical preference (1.0 observed and simulated). In the dense ecology of Experiment 4, it produces the required majority preference and closely matches its magnitude (0.5294 versus 0.5200). Experiment 2 is the main residual miss: majority preference is somewhat too weak (0.2750 versus 0.3433), while heterogeneity is somewhat too high (variance 0.0178 versus 0.0123). Experiment 4 shows the converse variance mismatch, with too little simulated heterogeneity (0.0154 versus 0.0305). These variance discrepancies suggest that individual differences may not be allocated optimally across contextual thresholds and pooling strength, but they do not undermine the principal experiment-level pattern.",
  "verdict": "continue",
  "rationale": "The accepted candidate is sufficiently experiment-invariant: it closely fits Experiments 1, 3, and 4 and gets both the direction and substantial magnitude of the Experiment 2 majority effect. It is also faithful to the arbiter recommendation, using slowly accumulated discrimination density to interpolate between persistent ordered search and genuine multi-cue pooling rather than reverting to invariant TTB or a current-trial load gate. The modest underprediction in Experiment 2 and variance mismatches could justify later parameter refinement, but they are not severe enough to require regeneration given the low accepted aggregate loss of 0.0191 and the absence of any qualitative experimental failure."
}
```

## Usage

```json
{
  "input_tokens": 9179,
  "output_tokens": 506
}
```
