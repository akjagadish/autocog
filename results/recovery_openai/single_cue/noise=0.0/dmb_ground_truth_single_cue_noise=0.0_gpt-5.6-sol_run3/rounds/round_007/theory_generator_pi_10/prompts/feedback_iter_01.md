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
- THEORY 1 = `pi_8`
- THEORY 2 = `pi_9`
- The recommendation below acts on THEORY 2 (= `pi_9`).

Replace pi_9 with a monotone contextual reactance-and-consensus theory. Communicated validities should generate smoothly increasing skeptical evidence against endorsed options, with no intrinsic discontinuity or notch at 95% or 100% and no predicted endpoint crossover. The strength of this evidence should depend on a coalition-level representation rather than pi_9's claim-by-claim symmetric competition: observers first summarize each side's mean assertiveness, breadth, and coherence, then compare the two coalition summaries. Repetition of similarly valid experts should reinforce a coalition summary sublinearly rather than merely causing mutual attentional suppression; this is needed to produce the very strong preference for the lower-validity coalition in Experiment 2. Coalition breadth and mixed composition should then normalize or compress the skeptical contrast, accounting for attenuation across broad conflicts without condition-specific exemptions. Add a separate raw-tally channel whose influence increases nonlinearly with absolute vote margin, allowing the tally to dominate in Experiments 4, 6, and 8 while leaving tied and one-vote-margin conflicts highly susceptible to anti-validity reversals. Subject-level parameters should continuously vary in reactance strength, tally sensitivity, and decision noise, with correlated heterogeneity permitted so that strong skeptics can also differ in margin protection; avoid categorical skeptic/conventional classes. The theory should predict persistent rejection of the 100% option against 50%, 70%, and 90% opponents, with the effect shrinking toward chance but not crossing it, thereby matching Experiment 1 while retaining the concentrated endpoint reversal in Experiment 12. It should also predict approximately smooth 90%–95%–100% behavior, as indicated by Experiment 15. Preserve option-swap symmetry and history independence. This creates a principled competitor to pi_8: both posit tally and pragmatic channels, but the new theory derives context effects from coalition-level sublinear consensus and normalization rather than endpoint distrust bonuses, validity compression notches, sparse exemptions, or pi_9's finite pairwise attentional pool.


## CANDIDATE THEORY
Coalition-Level Skeptical Consensus theory proposes that people do not compare communicated validities expert by expert. They first construct a representation of each option's endorsing coalition from three attributes: its mean assertiveness, its breadth, and its internal coherence. Mean assertiveness increases smoothly with communicated validity from 50% through 100%, without endpoint bonuses, notches, or categorical boundaries. Repeated similarly assertive endorsements strengthen the coalition representation according to a sublinear consensus function, whereas heterogeneous coalitions are represented less coherently. The contrast between the two coalition representations is additionally normalized by total conflict breadth and compositional mixture. People pragmatically react against the coalition making the stronger aggregate claim, so the coalition contrast enters choice with a skeptical sign. This reactance is continuous across individuals and can be weak rather than defining discrete skeptic classes. A smooth breadth-dependent gain makes isolated conflicts less reactance-provoking than repeated consensus, while preserving compression in very broad conflicts. A separate categorical tally channel ignores validity and increases nonlinearly with the absolute vote margin. It is weak enough at ties and one-vote margins to permit anti-validity reversals, but gradually protects decisive majorities without forcing ceiling-level tally choices. A shared continuous reactance trait also modulates the curvature of margin protection, allowing skepticism and resistance to decisive margins to covary without latent classes. Independent continuous variation in tally sensitivity and decision noise supplies further subject heterogeneity. Because choices receive no correctness feedback, coalition construction is history independent; because all evidence is expressed as signed option contrasts, the model is exactly invariant to swapping A and B.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Coalition-Level Skeptical Consensus model. Choice history is deliberately
    # ignored because no outcome feedback is available for learning validity.
    stimulus = np.asarray(state, dtype=np.float64)
    if stimulus.ndim != 2 or stimulus.shape[0] != 2:
        raise ValueError(
            f"Expected state with shape (2, n_features); got {stimulus.shape}."
        )

    validities = np.asarray(parameters["validities"], dtype=np.float64)
    n_features = stimulus.shape[1]
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} does not match "
            f"n_features {n_features}."
        )

    # +1 denotes a discriminating endorsement of A and -1 an endorsement of B.
    signed_votes = np.sign(stimulus[0] - stimulus[1])
    active = signed_votes != 0.0
    tally_margin = float(np.sum(signed_votes))

    # Assertiveness is completely smooth and monotone on the communicated
    # validity scale. In particular, 90%, 95%, and 100% receive neighboring
    # values on the same linear continuum.
    validity = np.clip(validities, 0.5, 1.0)
    assertiveness = 2.0 * (validity - 0.5)

    def coalition_summary(side):
        members = assertiveness[active & (signed_votes == side)]
        breadth = int(members.size)
        if breadth == 0:
            return 0.0, 0.0, 0

        mean_assertiveness = float(np.mean(members))
        dispersion = float(np.mean((members - mean_assertiveness) ** 2))

        # Similar members form a coherent aggregate claim. Mixed validity
        # compositions reduce coherence continuously rather than triggering an
        # exception for any particular validity or coalition size.
        coherence_scale = float(parameters["coherence_scale"])
        coherence_floor = float(parameters["coherence_floor"])
        coherence = coherence_floor + (1.0 - coherence_floor) * np.exp(
            -dispersion / coherence_scale
        )

        # Repetition reinforces consensus, but with diminishing returns. This
        # differs from allocating a fixed attentional pool among individual
        # claims: adding a similar endorser always increases the summary.
        consensus_exponent = float(parameters["consensus_exponent"])
        consensus = float(breadth) ** consensus_exponent
        summary = mean_assertiveness * consensus * coherence
        return float(summary), dispersion, breadth

    summary_a, dispersion_a, breadth_a = coalition_summary(1.0)
    summary_b, dispersion_b, breadth_b = coalition_summary(-1.0)
    raw_coalition_contrast = summary_a - summary_b
    total_breadth = breadth_a + breadth_b

    if total_breadth == 0:
        contextual_contrast = 0.0
    else:
        # Broad conflicts compress pragmatic contrast. Internal mixture adds a
        # second coalition-level normalizer, weighted by coalition breadth.
        excess_breadth = max(total_breadth - 2, 0)
        breadth_normalizer = 1.0 + float(
            parameters["breadth_normalization"]
        ) * np.log1p(float(excess_breadth))

        pooled_dispersion = (
            breadth_a * dispersion_a + breadth_b * dispersion_b
        ) / float(total_breadth)
        composition_normalizer = 1.0 + float(
            parameters["composition_normalization"]
        ) * float(total_breadth) * pooled_dispersion

        contextual_contrast = raw_coalition_contrast / (
            breadth_normalizer * composition_normalizer + 1e-12
        )

        # A singleton conflict evokes less coalition-level reactance than a
        # repeated consensus. The gain rises smoothly with breadth and remains
        # distinct from the broad-conflict normalization above.
        reinforcement_floor = float(parameters["breadth_reinforcement_floor"])
        reinforcement_gain = reinforcement_floor + (
            1.0 - reinforcement_floor
        ) * (1.0 - np.exp(-float(excess_breadth) / 2.0))
        contextual_contrast *= float(reinforcement_gain)

    # A continuous trait controls skeptical construal. It also contributes to
    # margin curvature below, permitting correlated heterogeneity without a
    # categorical skeptic/conventional mixture.
    reactance_trait = float(parameters["reactance_trait"])
    reactance_strength = float(parameters["reactance_strength"]) * (
        0.65 + 0.70 * reactance_trait
    )
    accumulated_contrast = np.tanh(
        float(parameters["coalition_accumulation"]) * contextual_contrast
    )
    skeptical_evidence = -reactance_strength * float(accumulated_contrast)

    # The raw tally is represented independently of communicated validity. Its
    # mildly superlinear margin function leaves a one-vote lead vulnerable but
    # lets decisive margins gradually dominate bounded reactance without
    # producing near-deterministic tally choices.
    tally_trait = float(parameters["tally_trait"])
    tally_sensitivity = float(parameters["tally_sensitivity"]) * (
        0.70 + 0.60 * tally_trait
    )
    margin_exponent = float(parameters["margin_exponent"]) + float(
        parameters["reactance_margin_coupling"]
    ) * (reactance_trait - 0.5)
    margin_exponent = float(np.clip(margin_exponent, 1.02, 1.65))

    if abs(tally_margin) < 1e-12:
        tally_evidence = 0.0
    else:
        tally_evidence = (
            tally_sensitivity
            * np.sign(tally_margin)
            * abs(tally_margin) ** margin_exponent
        )

    total_evidence = tally_evidence + skeptical_evidence

    # Decision noise varies continuously and independently across subjects.
    noise_trait = float(parameters["noise_trait"])
    choice_sensitivity = float(parameters["choice_sensitivity"]) * (
        0.65 + 0.70 * noise_trait
    )
    decision_variable = choice_sensitivity * total_evidence

    # Positive evidence favors A. Symmetric logits guarantee exact option-swap
    # symmetry, including on tied and mirrored trials.
    logits = np.array(
        [0.5 * decision_variable, -0.5 * decision_variable],
        dtype=np.float64,
    )
    logits -= np.max(logits)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()

    lapse = float(parameters["lapse_rate"])
    probabilities = (1.0 - lapse) * probabilities + 0.5 * lapse
    probabilities = np.clip(probabilities, 0.0, 1.0)
    probabilities /= probabilities.sum()
    return probabilities

`policy(probs) -> int`:
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = np.clip(probabilities, 0.0, None)
    total = float(np.sum(probabilities))
    if not np.isfinite(total) or total <= 0.0:
        probabilities = np.full(
            probabilities.size,
            1.0 / probabilities.size,
            dtype=np.float64,
        )
    else:
        probabilities /= total
    return int(np.random.choice(probabilities.size, p=probabilities))

`parameters`:
- consensus_exponent: [0.68, 0.84]
- coherence_scale: [0.025, 0.090]
- coherence_floor: [0.42, 0.68]
- breadth_normalization: [0.05, 0.14]
- composition_normalization: [0.8, 2.2]
- breadth_reinforcement_floor: [0.62, 0.76]
- coalition_accumulation: [1.15, 1.75]
- reactance_strength: [2.4, 3.7]
- reactance_trait: [0, 1]
- tally_sensitivity: [0.20, 0.38]
- tally_trait: [0, 1]
- margin_exponent: [1.15, 1.40]
- reactance_margin_coupling: [-0.18, 0.28]
- choice_sensitivity: [0.78, 1.24]
- noise_trait: [0, 1]
- lapse_rate: [0.0, 0.08]
- validities: validities

`rationale`:
This is a minimal recalibration of the accepted model. The tally-sensitivity and margin-exponent ranges are substantially reduced, while remaining superlinear, to prevent margins of three through six from producing ceiling-level choices. This should reduce the excessive Bayes factor in Experiment 2, increase TTB-conflict and minority-tally choices in Experiments 1 and 3, move Experiments 4 and 6 away from near-determinism, and expose more subject-level tally heterogeneity. The only added mechanism is a smooth breadth-reinforcement gain on the existing coalition contrast. It attenuates two-endorser conflicts to soften the excessive reactance in Experiments 5 and 8, but approaches one as repeated endorsers form a genuine coalition. The consensus exponent is modestly increased so broad homogeneous coalitions retain their strong sublinear reinforcement in Experiments 7, 11, and 12. Because a concentrated 100%-versus-50% contrast remains maximal before this moderate attenuation, the endpoint reversal in Experiment 10 should persist. The gain is continuous, symmetric, validity-neutral, and applied before the existing breadth and composition normalization, so it introduces no endpoint exception, validity notch, or condition-specific rule.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.1769 -> ACCEPTED
- iter 2 (current candidate you are grading): loss=0.1311 -> ACCEPTED
Running-best (last accepted) base: iter 2 at loss=0.1311.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1]
  A=[0, 1, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 1, 0, 1]
  A=[1, 0, 1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 1, 0]  B=[1, 0, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Choice rate for the Take-The-Best winner on TTB-vs-tally conflict trials."""
    aligned = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b

        # Validities are strictly descending in feature-index order, so the
        # first nonzero difference identifies the Take-The-Best winner.
        discriminating = np.flatnonzero(diff != 0)
        if discriminating.size == 0:
            continue
        ttb_winner = 0 if diff[discriminating[0]] > 0 else 1

        a_wins = int(np.sum(diff > 0))
        b_wins = int(np.sum(diff < 0))
        if a_wins == b_wins:
            continue
        tally_winner = 0 if a_wins > b_wins else 1

        # Retain only directional conflicts between the two heuristics.
        if ttb_winner != tally_winner:
            aligned.append(float(int(row["response"]) == ttb_winner))

    # The selected design always contains conflict trials; the fallback keeps
    # the function defined for any unexpected slice.
    return float(np.mean(aligned)) if aligned else 0.5
```

**Observed (real) value:** 0.4033 (var=0.0042)
**Candidate trajectory (this loop):**
  - iter 1: 0.1950 (var=0.0044) (Δ vs real -0.2083)
  - iter 2 (current): 0.3575 (var=0.0085) (Δ vs real -0.0458)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8450 (var=0.0143)
- pi_2: 0.1350 (var=0.0104)
- pi_3: 0.3775 (var=0.0056)
- pi_4: 0.3392 (var=0.0040)
- pi_5: 0.3108 (var=0.0052)
- pi_6: 0.4721 (var=0.0071)
- pi_7: 0.0896 (var=0.0023)
- pi_8: 0.3383 (var=0.0039)
- pi_9: 0.3208 (var=0.0107)

### Experiment 2
**Design**
  A=[1, 1, 1, 0, 1, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 1, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 1, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Total log marginal-likelihood ratio favoring Tallying over TTB.
    # The parameter grids approximate uniform priors over the stated ranges.
    beta_grid = np.linspace(0.1, 20.0, 120)
    epsilon_grid = np.linspace(0.0, 0.5, 61)
    beta = beta_grid[:, None]
    epsilon = epsilon_grid[None, :]

    def logmeanexp(x):
        x = np.asarray(x, dtype=float)
        m = float(np.max(x))
        return float(m + np.log(np.mean(np.exp(x - m))))

    def subject_log_bf(df):
        margins = []
        correct = []

        for _, row in df.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float)
            b = np.asarray(row["option_b_ratings"], dtype=float)
            a_wins = int(np.sum(a > b))
            b_wins = int(np.sum(b > a))
            margin = abs(a_wins - b_wins)

            if margin == 0:
                continue

            winner = 0 if a_wins > b_wins else 1
            margins.append(float(margin))
            correct.append(float(int(row["response"]) == winner))

        if len(margins) == 0:
            return np.nan

        margins = np.asarray(margins, dtype=float)
        correct = np.asarray(correct, dtype=float)
        tiny = 1e-12

        # TTB assigns the same winner probability to every non-tied pair.
        core_ttb = 1.0 / (1.0 + np.exp(-beta))
        p_ttb = (1.0 - epsilon) * core_ttb + epsilon * 0.5
        p_ttb = np.clip(p_ttb, tiny, 1.0 - tiny)
        n_correct = float(np.sum(correct))
        n_error = float(correct.size - n_correct)
        ll_ttb = n_correct * np.log(p_ttb) + n_error * np.log1p(-p_ttb)

        # Tallying's winner probability depends on the observed tally margin.
        ll_tally = np.zeros_like(p_ttb, dtype=float)
        for d, y in zip(margins, correct):
            x = beta * d
            core = np.where(x >= 0.0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))
            p = (1.0 - epsilon) * core + epsilon * 0.5
            p = np.clip(p, tiny, 1.0 - tiny)
            ll_tally += y * np.log(p) + (1.0 - y) * np.log1p(-p)

        log_ml_tally = logmeanexp(ll_tally)
        log_ml_ttb = logmeanexp(ll_ttb)
        return float(log_ml_tally - log_ml_ttb)

    if "subject_id" in data.columns:
        values = [subject_log_bf(g) for _, g in data.groupby("subject_id", sort=False)]
    else:
        values = [subject_log_bf(data)]

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")

    # Independent participants contribute additive evidence. On a one-subject
    # slice this is simply that participant's log Bayes factor.
    return float(np.sum(values))

```

**Observed (real) value:** 89.1422 (var=7.5896)
**Candidate trajectory (this loop):**
  - iter 1: 328.5148 (var=8.4447) (Δ vs real +239.3726)
  - iter 2 (current): 49.8260 (var=7.2164) (Δ vs real -39.3163)
**Other theories' values on this metric (for reference):**
- pi_2: 1.9009 (var=0.0916)
- pi_1: 0.5503 (var=0.0488)
- pi_3: 49.6494 (var=1.7369)
- pi_4: 100.3141 (var=2.8389)
- pi_5: 59.0660 (var=1.4267)
- pi_6: 84.3674 (var=2.5041)
- pi_7: 83.3362 (var=1.6438)
- pi_8: 56.1425 (var=1.2063)
- pi_9: 59.9467 (var=3.7118)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Proportion of choices favoring the option with fewer positive ratings."""
    if data is None or len(data) == 0:
        return float("nan")

    minority_choices = []
    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"],
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        tally_margin_a = float(np.sum(a > b) - np.sum(b > a))
        r = int(response)

        if tally_margin_a < 0:       # A has fewer cue wins
            minority_choices.append(float(r == 0))
        elif tally_margin_a > 0:     # B has fewer cue wins
            minority_choices.append(float(r == 1))
        # Tied tallies have no minority option and are omitted.

    if len(minority_choices) == 0:
        return float("nan")
    return float(np.mean(minority_choices))
```

**Observed (real) value:** 0.4508 (var=0.0031)
**Candidate trajectory (this loop):**
  - iter 1: 0.2906 (var=0.0036) (Δ vs real -0.1602)
  - iter 2 (current): 0.3473 (var=0.0032) (Δ vs real -0.1035)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5394 (var=0.0025)
- pi_2: 0.1190 (var=0.0098)
- pi_1: 0.7531 (var=0.0085)
- pi_4: 0.4698 (var=0.0021)
- pi_5: 0.4573 (var=0.0018)
- pi_6: 0.3958 (var=0.0029)
- pi_7: 0.1542 (var=0.0036)
- pi_8: 0.4106 (var=0.0035)
- pi_9: 0.3440 (var=0.0028)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if len(data) == 0:
        return float("nan")

    a = np.vstack([np.asarray(x, dtype=float) for x in data["option_a_ratings"]])
    b = np.vstack([np.asarray(x, dtype=float) for x in data["option_b_ratings"]])
    responses = np.asarray(data["response"], dtype=int)

    # Positive margin means that A wins more binary feature comparisons;
    # negative margin means that B does.
    delta = a - b
    tally_margin = np.sum(delta > 0, axis=1) - np.sum(delta < 0, axis=1)

    # Select the two orientations of the diagnostic 6-versus-1 conflict:
    # the tally winner has a five-cue margin but is opposed by expert 1.
    strongest_cue = delta[:, 0]
    critical = (np.abs(tally_margin) == 5) & (strongest_cue * tally_margin < 0)
    if not np.any(critical):
        return float("nan")

    chose_a = responses == 0
    tally_winner_is_a = tally_margin > 0

    # Rate at which the numerous weak experts override the strongest expert.
    return float(np.mean(chose_a[critical] == tally_winner_is_a[critical]))
```

**Observed (real) value:** 0.8213 (var=0.0169)
**Candidate trajectory (this loop):**
  - iter 1: 0.9819 (var=0.0005) (Δ vs real +0.1606)
  - iter 2 (current): 0.9387 (var=0.0027) (Δ vs real +0.1175)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8606 (var=0.0060)
- pi_3: 0.3869 (var=0.0089)
- pi_1: 0.1325 (var=0.0094)
- pi_4: 0.7775 (var=0.0083)
- pi_5: 0.8337 (var=0.0061)
- pi_6: 0.7837 (var=0.0060)
- pi_7: 0.9413 (var=0.0016)
- pi_8: 0.8544 (var=0.0058)
- pi_9: 0.9294 (var=0.0023)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Recode each response as choosing the option endorsed by the 95%-valid
    # expert (feature 0), irrespective of whether that option is A or B.
    by_size = {1: [], 2: [], 4: []}

    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        difference = a - b

        # In this design, coalition size is the number of experts endorsing A
        # over B (equal to the number endorsing B over A).
        coalition_size = int(np.sum(difference > 0.0))
        if coalition_size not in by_size:
            continue

        high_expert_option = 0 if difference[0] > 0.0 else 1
        chose_high_expert = float(int(row['response']) == high_expert_option)
        by_size[coalition_size].append(chose_high_expert)

    if any(len(by_size[k]) == 0 for k in (1, 2, 4)):
        return float('nan')

    p1 = float(np.mean(by_size[1]))
    p2 = float(np.mean(by_size[2]))
    p4 = float(np.mean(by_size[4]))

    # Approximate inverse-noise weighting of the conditions with the largest
    # predicted between-theory probability differences. The size-3 condition
    # is omitted because both theories predict it close to chance.
    return float(1.8 * (p1 - 0.5) + 0.9 * (p2 - 0.5) - (p4 - 0.5))
```

**Observed (real) value:** -0.0382 (var=0.0352)
**Candidate trajectory (this loop):**
  - iter 1: -0.6893 (var=0.0624) (Δ vs real -0.6512)
  - iter 2 (current): -0.5991 (var=0.0524) (Δ vs real -0.5609)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5887 (var=0.0712)
- pi_4: 0.0071 (var=0.0478)
- pi_1: 0.5484 (var=0.0538)
- pi_2: -0.0256 (var=0.0579)
- pi_5: -0.0126 (var=0.0452)
- pi_6: -0.0866 (var=0.0563)
- pi_7: -0.5556 (var=0.0918)
- pi_8: 0.0104 (var=0.0390)
- pi_9: -0.4666 (var=0.0908)

### Experiment 6
**Design**
  A=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Proportion of choices favoring the option with the larger raw vote tally."""
    scores = []
    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"],
    ):
        a = np.asarray(a_cell, dtype=np.float64)
        b = np.asarray(b_cell, dtype=np.float64)
        margin = float(np.sum(a - b))
        r = int(response)

        if margin > 0.0:
            scores.append(1.0 if r == 0 else 0.0)
        elif margin < 0.0:
            scores.append(1.0 if r == 1 else 0.0)
        else:
            scores.append(0.5)

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** 0.8462 (var=0.0122)
**Candidate trajectory (this loop):**
  - iter 1: 0.9794 (var=0.0005) (Δ vs real +0.1331)
  - iter 2 (current): 0.9319 (var=0.0022) (Δ vs real +0.0856)
**Other theories' values on this metric (for reference):**
- pi_4: 0.8037 (var=0.0033)
- pi_3: 0.5665 (var=0.0017)
- pi_1: 0.8565 (var=0.0102)
- pi_2: 0.8612 (var=0.0065)
- pi_5: 0.8506 (var=0.0014)
- pi_6: 0.8273 (var=0.0015)
- pi_7: 0.8325 (var=0.0029)
- pi_8: 0.8592 (var=0.0031)
- pi_9: 0.9035 (var=0.0023)

### Experiment 7
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Score whether each response selected the option endorsed by the
    # high-validity experts (features 0 through 6), irrespective of screen side.
    weighted_success = 0.0
    total_weight = 0.0

    # Increasing weights emphasize the larger-coalition conditions, where the
    # theories' predicted choice probabilities diverge most, while retaining
    # lower-m conditions to keep the subject-level estimate stable.
    weights = np.asarray([0.30, 0.55, 0.72, 0.85, 0.94, 1.00], dtype=float)

    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"],
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        if a.size < 7 or b.size < 7:
            continue

        high_margin = float(np.sum(a[:7] - b[:7]))
        m = int(round(abs(high_margin)))
        if high_margin == 0.0 or m < 2 or m > 7:
            continue

        chose_a = int(response) == 0
        high_option_is_a = high_margin > 0.0
        chose_high_coalition = float(chose_a == high_option_is_a)

        weight = float(weights[m - 2])
        weighted_success += weight * chose_high_coalition
        total_weight += weight

    if total_weight == 0.0:
        return float("nan")
    return float(weighted_success / total_weight)
```

**Observed (real) value:** 0.1161 (var=0.0079)
**Candidate trajectory (this loop):**
  - iter 1: 0.1109 (var=0.0046) (Δ vs real -0.0052)
  - iter 2 (current): 0.1055 (var=0.0038) (Δ vs real -0.0107)
**Other theories' values on this metric (for reference):**
- pi_5: 0.7494 (var=0.0030)
- pi_4: 0.5919 (var=0.0025)
- pi_1: 0.8622 (var=0.0079)
- pi_2: 0.8381 (var=0.0106)
- pi_3: 0.9864 (var=0.0001)
- pi_6: 0.1972 (var=0.0033)
- pi_7: 0.2755 (var=0.0544)
- pi_8: 0.2347 (var=0.0051)
- pi_9: 0.3269 (var=0.0321)

### Experiment 8
**Design**
  A=[0, 1, 0, 0]  B=[1, 0, 0, 0]
  A=[0, 1, 1, 1]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[0, 1, 1, 1]  B=[1, 1, 1, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 0, 1, 0]
  A=[1, 1, 0, 1]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 0, 0, 1]
  A=[1, 1, 1, 0]  B=[0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if data is None or len(data) == 0:
        return 0.0

    # Bounded reliability code specified by Compressed-Reliability Consensus.
    validities = np.array([0.5, 0.6, 0.8, 1.0], dtype=np.float64)
    reliability_code = np.tanh(3.0 * (validities - 0.5)) / np.tanh(1.5)

    numerators = []
    weights = []
    subjects = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=np.float64)
        b = np.asarray(row["option_b_ratings"], dtype=np.float64)
        contrast = float(np.dot(a - b, reliability_code))
        weight = abs(contrast)
        if not np.isfinite(weight) or weight <= 0.0:
            continue

        # +1 means the response followed the more reliable endorser, and -1
        # means it followed the less reliable endorser. This is invariant to
        # whether that endorser appeared on option A or option B.
        choice_sign = 1.0 - 2.0 * float(row["response"])
        aligned_choice = np.sign(contrast) * choice_sign
        numerators.append(weight * aligned_choice)
        weights.append(weight)
        subjects.append(row["subject_id"])

    if len(weights) == 0:
        return 0.0

    work = pd.DataFrame({
        "subject_id": subjects,
        "numerator": numerators,
        "weight": weights,
    })
    grouped = work.groupby("subject_id", sort=False)[["numerator", "weight"]].sum()
    valid = grouped["weight"] > 0.0
    if not bool(valid.any()):
        return 0.0

    subject_scores = (
        grouped.loc[valid, "numerator"] / grouped.loc[valid, "weight"]
    ).to_numpy(dtype=np.float64)

    # Normalized cumulative directional evidence: for one subject this is the
    # subject's weighted alignment score; pooled evidence grows with sqrt(N).
    return float(np.sqrt(subject_scores.size) * np.mean(subject_scores))

```

**Observed (real) value:** -3.5372 (var=0.0438)
**Candidate trajectory (this loop):**
  - iter 1: -5.0413 (var=0.0212) (Δ vs real -1.5041)
  - iter 2 (current): -4.5626 (var=0.0253) (Δ vs real -1.0254)
**Other theories' values on this metric (for reference):**
- pi_4: 0.4221 (var=0.0108)
- pi_5: -0.0028 (var=0.0102)
- pi_1: 4.5627 (var=0.0649)
- pi_2: 0.0176 (var=0.0099)
- pi_3: 3.9747 (var=0.0054)
- pi_6: -3.4648 (var=0.0108)
- pi_7: -1.4013 (var=0.0925)
- pi_8: -4.4717 (var=0.0102)
- pi_9: -2.4502 (var=0.0542)

### Experiment 9
**Design**
  A=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if data is None or len(data) == 0:
        return float('nan')

    weighted_success = 0.0
    total_weight = 0.0

    # Conditions are represented in a canonical orientation using the sign of
    # the overall tally. This makes exact A/B-reversed pairs contribute to the
    # same planned condition.
    condition_weights = {
        7: 2.00,
        3: 1.45,
        -3: 0.80,
        -6: 1.10,
    }

    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=np.float64)
        b = np.asarray(row['option_b_ratings'], dtype=np.float64)
        if a.size < 7 or b.size != a.size:
            continue

        signed_votes = np.sign(a - b)
        high_validity_margin = float(np.sum(signed_votes[:7]))
        tally_margin = float(np.sum(signed_votes))
        if high_validity_margin == 0.0 or tally_margin == 0.0:
            continue

        canonical_margin = int(round(high_validity_margin * np.sign(tally_margin)))
        weight = condition_weights.get(canonical_margin, 1.0)

        response = int(row['response'])
        chose_high_validity_side = (
            (high_validity_margin > 0.0 and response == 0) or
            (high_validity_margin < 0.0 and response == 1)
        )
        weighted_success += weight * float(chose_high_validity_side)
        total_weight += weight

    if total_weight == 0.0:
        return float('nan')
    return float(weighted_success / total_weight)
```

**Observed (real) value:** 0.3248 (var=0.0018)
**Candidate trajectory (this loop):**
  - iter 1: 0.2191 (var=0.0038) (Δ vs real -0.1058)
  - iter 2 (current): 0.1912 (var=0.0042) (Δ vs real -0.1337)
**Other theories' values on this metric (for reference):**
- pi_5: 0.6632 (var=0.0032)
- pi_6: 0.1697 (var=0.0029)
- pi_1: 0.7324 (var=0.0054)
- pi_2: 0.5977 (var=0.0022)
- pi_3: 0.9801 (var=0.0004)
- pi_4: 0.5348 (var=0.0021)
- pi_7: 0.3215 (var=0.0354)
- pi_8: 0.2814 (var=0.0029)
- pi_9: 0.3376 (var=0.0157)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 0, 0, 1]
  A=[1, 1, 0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 0, 0, 1]  B=[1, 1, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 0, 1, 1]  B=[1, 1, 0, 1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    scores = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b

        # The concentrated endpoint condition has exactly two discriminating
        # experts: one 100%-valid expert and the final 50%-valid expert.
        if np.count_nonzero(diff) != 2 or diff.size < 8 or diff[7] == 0:
            continue

        response = int(row["response"])
        low_validity_side = 0 if diff[7] > 0 else 1
        scores.append(float(response == low_validity_side))

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** 0.8617 (var=0.0104)
**Candidate trajectory (this loop):**
  - iter 1: 0.8883 (var=0.0103) (Δ vs real +0.0267)
  - iter 2 (current): 0.8617 (var=0.0110) (Δ vs real +0.0000)
**Other theories' values on this metric (for reference):**
- pi_6: 0.7625 (var=0.0137)
- pi_5: 0.5375 (var=0.0072)
- pi_1: 0.1683 (var=0.0142)
- pi_2: 0.4767 (var=0.0123)
- pi_3: 0.0008 (var=0.0000)
- pi_4: 0.4383 (var=0.0129)
- pi_7: 0.8467 (var=0.0208)
- pi_8: 0.8408 (var=0.0059)
- pi_9: 0.7475 (var=0.0177)

### Experiment 11
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if len(data) == 0:
        return float('nan')

    scores = []
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        diff = a - b

        # The first five experts have 70% validity. Restrict attention to the
        # diffuse conflicts (3-vs-2, 4-vs-3, and 5-vs-4), where at least five
        # experts discriminate between the products.
        discriminating_count = int(np.count_nonzero(diff))
        if discriminating_count < 5:
            continue

        high_validity_margin = float(np.sum(diff[:5]))
        if high_validity_margin > 0:
            chose_high_validity_coalition = int(row['response']) == 0
        elif high_validity_margin < 0:
            chose_high_validity_coalition = int(row['response']) == 1
        else:
            continue

        scores.append(float(chose_high_validity_coalition))

    if not scores:
        return float('nan')
    return float(np.mean(scores))
```

**Observed (real) value:** 0.1572 (var=0.0077)
**Candidate trajectory (this loop):**
  - iter 1: 0.1492 (var=0.0088) (Δ vs real -0.0081)
  - iter 2 (current): 0.1139 (var=0.0067) (Δ vs real -0.0433)
**Other theories' values on this metric (for reference):**
- pi_7: 0.9125 (var=0.0016)
- pi_6: 0.2006 (var=0.0082)
- pi_1: 0.8514 (var=0.0098)
- pi_2: 0.8394 (var=0.0083)
- pi_3: 0.7772 (var=0.0054)
- pi_4: 0.5706 (var=0.0031)
- pi_5: 0.6192 (var=0.0027)
- pi_8: 0.2039 (var=0.0054)
- pi_9: 0.4628 (var=0.0171)

### Experiment 12
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    conventional_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        d = a - b

        # Select the 12 mirrored conditions comparing equal-sized coalitions
        # of 70%-valid experts (indices 0:6) and 50%-valid experts (6:12),
        # excluding all conditions involving 95%-valid experts.
        d70 = d[:6]
        d50 = d[6:12]
        d95 = d[12:15]
        n70 = int(np.count_nonzero(d70))
        n50 = int(np.count_nonzero(d50))
        s70 = float(np.sum(d70))
        s50 = float(np.sum(d50))

        is_target = (
            n70 > 0
            and n70 == n50
            and abs(s70) == n70
            and abs(s50) == n50
            and np.sign(s70) == -np.sign(s50)
            and np.count_nonzero(d95) == 0
        )
        if not is_target:
            continue

        response = int(row["response"])
        # Score 1 when the chosen product is endorsed by the 70%-valid
        # coalition, regardless of whether that product appears as A or B.
        chose_70_option = ((s70 > 0 and response == 0) or
                           (s70 < 0 and response == 1))
        conventional_choices.append(float(chose_70_option))

    if len(conventional_choices) == 0:
        return float("nan")
    return float(np.mean(conventional_choices))
```

**Observed (real) value:** 0.1261 (var=0.0089)
**Candidate trajectory (this loop):**
  - iter 1: 0.1142 (var=0.0050) (Δ vs real -0.0119)
  - iter 2 (current): 0.1403 (var=0.0055) (Δ vs real +0.0142)
**Other theories' values on this metric (for reference):**
- pi_6: 0.2356 (var=0.0041)
- pi_7: 0.8364 (var=0.0032)
- pi_1: 0.8367 (var=0.0093)
- pi_2: 0.4964 (var=0.0036)
- pi_3: 0.7497 (var=0.0047)
- pi_4: 0.5364 (var=0.0028)
- pi_5: 0.4892 (var=0.0026)
- pi_8: 0.1583 (var=0.0037)
- pi_9: 0.3469 (var=0.0105)

### Experiment 13
**Design**
  A=[1, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def subject_contrast(df):
        successes = np.zeros(3, dtype=np.float64)
        totals = np.zeros(3, dtype=np.float64)

        for _, row in df.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=np.float64)
            b = np.asarray(row["option_b_ratings"], dtype=np.float64)
            signed = np.sign(a - b)

            # Exactly one of the first three (90%, 95%, 100%) experts is
            # discriminating in each designed trial.
            focal = np.flatnonzero(signed[:3] != 0.0)
            if focal.size != 1:
                continue
            condition = int(focal[0])
            focal_sign = float(signed[condition])
            response = int(row["response"])

            # Score choosing the option endorsed by the focal high-validity
            # expert, making exact A/B reversals contribute on the same scale.
            chose_focal_option = (
                (focal_sign > 0.0 and response == 0)
                or (focal_sign < 0.0 and response == 1)
            )
            successes[condition] += float(chose_focal_option)
            totals[condition] += 1.0

        if np.any(totals == 0.0):
            return np.nan
        rates = successes / totals
        return float(rates[1] - 0.5 * (rates[0] + rates[2]))

    if "subject_id" in data.columns and data["subject_id"].nunique() > 1:
        values = []
        for _, subj_df in data.groupby("subject_id", sort=False):
            value = subject_contrast(subj_df)
            if np.isfinite(value):
                values.append(value)
        return float(np.mean(values)) if values else np.nan

    return subject_contrast(data)

```

**Observed (real) value:** 0.0094 (var=0.0050)
**Candidate trajectory (this loop):**
  - iter 1: 0.0016 (var=0.0100) (Δ vs real -0.0078)
  - iter 2 (current): -0.0016 (var=0.0093) (Δ vs real -0.0109)
**Other theories' values on this metric (for reference):**
- pi_8: 0.0569 (var=0.0085)
- pi_6: -0.0147 (var=0.0080)
- pi_1: -0.0147 (var=0.0061)
- pi_2: 0.0028 (var=0.0045)
- pi_3: -0.0878 (var=0.0068)
- pi_4: -0.0187 (var=0.0085)
- pi_5: 0.0069 (var=0.0102)
- pi_7: -0.0338 (var=0.0098)
- pi_9: -0.0147 (var=0.0077)

### Experiment 14
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Preference for the option not endorsed by the perfectly valid expert,
    # scored separately at each coalition breadth and display orientation.
    cell_values = {(m, side): [] for m in range(1, 7) for side in (-1, 1)}

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.size == 0 or b.size != a.size:
            continue

        endpoint_side = int(np.sign(a[0] - b[0]))
        if endpoint_side == 0:
            continue

        active_count = int(np.count_nonzero(a - b))
        m = active_count // 2
        if m < 1 or m > 6 or active_count != 2 * m:
            continue

        response = int(row["response"])
        # endpoint_side=+1 means the certainty expert endorses A, so choosing
        # B is the anti-endpoint choice. For endpoint_side=-1, choosing A is.
        anti_endpoint_choice = float(
            (endpoint_side == 1 and response == 1)
            or (endpoint_side == -1 and response == 0)
        )
        cell_values[(m, endpoint_side)].append(anti_endpoint_choice)

    breadth_means = []
    for m in range(1, 7):
        orientation_means = []
        for side in (-1, 1):
            values = cell_values[(m, side)]
            if len(values) == 0:
                return float("nan")
            orientation_means.append(float(np.mean(values)))
        breadth_means.append(float(np.mean(orientation_means)))

    # A normalized linear low-to-high coalition-breadth contrast. Positive
    # values indicate that anti-endpoint preference attenuates as m increases.
    weights = np.array([2.5, 1.5, 0.5, -0.5, -1.5, -2.5], dtype=float)
    return float(np.dot(weights, np.asarray(breadth_means)) / 4.5)

```

**Observed (real) value:** 0.1781 (var=0.0129)
**Candidate trajectory (this loop):**
  - iter 1: 0.1878 (var=0.0094) (Δ vs real +0.0097)
  - iter 2 (current): 0.1576 (var=0.0098) (Δ vs real -0.0204)
**Other theories' values on this metric (for reference):**
- pi_6: -0.0096 (var=0.0069)
- pi_8: 0.1467 (var=0.0083)
- pi_1: 0.0031 (var=0.0052)
- pi_2: 0.0064 (var=0.0127)
- pi_3: 0.0007 (var=0.0000)
- pi_4: -0.0000 (var=0.0117)
- pi_5: -0.0346 (var=0.0125)
- pi_7: -0.0029 (var=0.0065)
- pi_9: 0.1044 (var=0.0094)

### Experiment 15
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    low_endpoint_choices = []
    high_endpoint_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.size < 5 or b.size != a.size:
            continue

        diff = a - b
        active = np.flatnonzero(diff != 0)
        if active.size != 2 or 0 not in active:
            continue

        opponents = active[active != 0]
        if opponents.size != 1:
            continue
        opponent = int(opponents[0])
        if diff[0] * diff[opponent] >= 0:
            continue

        response = int(row["response"])
        endpoint_chosen = (
            (diff[0] > 0 and response == 0)
            or (diff[0] < 0 and response == 1)
        )

        if opponent in (1, 2):
            low_endpoint_choices.append(float(endpoint_chosen))
        elif opponent == 4:
            high_endpoint_choices.append(float(endpoint_chosen))

    if len(low_endpoint_choices) == 0 or len(high_endpoint_choices) == 0:
        return float("nan")

    low_rate = float(np.mean(low_endpoint_choices))
    high_rate = float(np.mean(high_endpoint_choices))

    return float((low_rate < 0.5) and (high_rate > 0.5))

```

**Observed (real) value:** 0.0000 (var=0.2400)
**Candidate trajectory (this loop):**
  - iter 1: 0.0000 (var=0.0384) (Δ vs real +0.0000)
  - iter 2 (current): 0.0000 (var=0.1204) (Δ vs real +0.0000)
**Other theories' values on this metric (for reference):**
- pi_8: 1.0000 (var=0.1600)
- pi_9: 0.0000 (var=0.1204)
- pi_1: 0.0000 (var=0.0000)
- pi_2: 0.0000 (var=0.1600)
- pi_3: 0.0000 (var=0.0000)
- pi_4: 0.0000 (var=0.1204)
- pi_5: 0.0000 (var=0.0900)
- pi_6: 1.0000 (var=0.2484)
- pi_7: 0.0000 (var=0.0000)

### Experiment 16
**Design**
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    selected_choices = []

    # Coalitions opposed to the four 75%-valid experts. The maximally
    # dispersed coalition is intentionally excluded because the competing
    # theory predicts that its anti-reliability signal is strongly diluted.
    homogeneous = (4, 5, 6, 7)
    moderate_dispersion = (8, 9, 10, 11)
    selected = {homogeneous, moderate_dispersion}

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=int)
        b = np.asarray(row["option_b_ratings"], dtype=int)
        response = int(row["response"])

        # Locate the option endorsed by the four 75%-valid experts.
        a_has_low = int(np.sum(a[:4])) > int(np.sum(b[:4]))
        low_option = a if a_has_low else b
        opposing_option = b if a_has_low else a

        # Require the expected four-expert low-validity coalition and select
        # the two conditions where the theories' absolute predictions diverge.
        low_set = tuple(np.flatnonzero(low_option).tolist())
        opposing_set = tuple(np.flatnonzero(opposing_option).tolist())
        if low_set != (0, 1, 2, 3) or opposing_set not in selected:
            continue

        chose_low = (response == 0) if a_has_low else (response == 1)
        selected_choices.append(float(chose_low))

    if len(selected_choices) == 0:
        return 0.0
    return float(np.mean(selected_choices))

```

**Observed (real) value:** 0.8819 (var=0.0059)
**Candidate trajectory (this loop):**
  - iter 1: 0.8053 (var=0.0069) (Δ vs real -0.0766)
  - iter 2 (current): 0.8131 (var=0.0054) (Δ vs real -0.0687)
**Other theories' values on this metric (for reference):**
- pi_9: 0.5850 (var=0.0057)
- pi_8: 0.7128 (var=0.0076)
- pi_1: 0.1353 (var=0.0104)
- pi_2: 0.5084 (var=0.0042)
- pi_3: 0.2400 (var=0.0050)
- pi_4: 0.5028 (var=0.0039)
- pi_5: 0.5072 (var=0.0053)
- pi_6: 0.4859 (var=0.0035)
- pi_7: 0.7762 (var=0.0258)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1 (most recent)
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed mechanism family and captures several important qualitative signatures: strong anti-validity choices in Experiments 7, 10, 11, and 12; breadth attenuation in Experiment 14; smooth 90%-95%-100% behavior in Experiment 13; no endpoint crossover in Experiment 15; and the composition effect in Experiment 16. However, its tally channel is substantially overpowered. It drives TTB-conflict choice too far toward tallying in Experiment 1 (0.195 versus 0.403), suppresses minority-tally choices in Experiment 3 (0.291 versus 0.451), produces nearly deterministic tally choice in Experiments 4 and 6 (0.982 and 0.979 versus 0.821 and 0.846), and yields vastly excessive margin-dependent evidence in Experiment 2 (log BF 328.5 versus 89.1). The very low simulated variances in Experiments 4 and 6 further show that parameter heterogeneity is hidden by ceiling saturation. Relatedly, Experiment 9 underpredicts high-validity-coalition choice (0.219 versus 0.325). Independently of tally, skeptical contrast is too extreme in some tied or narrow conflicts: Experiment 5 is far too negative (-0.689 versus -0.038), and Experiment 8 overstates anti-reliability alignment (-5.041 versus -3.537). Thus the model gets the locations of several reversals right but exaggerates both decisive-margin protection and some narrow-context reactance.
Rationale: Retain the coalition-level skeptical-consensus architecture, smooth monotone validity code, symmetry, and history independence, but recalibrate its two channels. First, substantially reduce the effective tally margin curvature and/or tally sensitivity—especially at margins around 3-6—while preserving a nonlinear increase with absolute margin. The current exponent-sensitivity combination causes ceiling behavior. A lower exponent range and lower central tally gain should jointly move Experiments 1, 2, 3, 4, 6, and 9 in the required directions; it should also expose the existing continuous tally heterogeneity and raise the unrealistically small variances in Experiments 4 and 6. Avoid merely adding more lapse, because that would indiscriminately damage the well-fit coalition reversals. Second, soften singleton/narrow-coalition skeptical gain without removing strong repeated-coalition consensus. This can be done within the prescribed family by reducing baseline reactance or coalition accumulation while modestly strengthening sublinear consensus reinforcement, or by using a smooth breadth-dependent coalition gain before the existing broad-conflict normalization. The aim is to move Experiments 5 and 8 toward chance while preserving the excellent broad-coalition fits in Experiments 7, 11, and 12 and the concentrated endpoint reversal in Experiment 10. Any adjustment should be checked against Experiment 14 so that breadth attenuation is not lost. The accepted loss is promising, but the large, systematic tally overshoot and the severe Experiment 5 miss preclude stopping.

**Outcome of your advice:** iter 1 candidate loss=0.1769 -> CURRENT candidate loss=0.1311 -> the gate marked it ACCEPTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The accepted recalibration is a substantial improvement: aggregate loss fell from 0.1769 to 0.1311, and the prior advice to weaken the tally channel and attenuate narrow-coalition reactance moved Experiments 1, 2, 3, 4, 6, 8, and 10 in useful directions. The candidate remains faithful to the prescribed coalition-level skeptical-consensus family. It also preserves excellent signatures: Experiment 10 is exact, Experiments 7 and 12 retain strong repeated-coalition reversals, Experiment 14 captures breadth attenuation, Experiment 13 remains smooth around 90%-95%-100%, and Experiment 15 shows no endpoint crossover. Nevertheless, the fit is not yet experiment-invariant. Experiment 5 remains a severe failure (-0.599 versus -0.038), and Experiment 8 still overstates anti-reliability choice (-4.563 versus -3.537). Thus moderate tied or narrow validity contrasts still trigger much too much skeptical evidence, despite the previous attenuation. In contrast, the maximal singleton endpoint contrast in Experiment 10 is now exactly right, so another uniform reduction of singleton reactance would sacrifice a key success. The tally channel also has the wrong margin profile: it remains too decisive at large margins in Experiments 4 and 6 (0.939 and 0.932 versus 0.821 and 0.846), with residual ceiling compression of between-subject variance, yet it now produces too little margin-dependent evidence in Experiment 2 (49.8 versus 89.1). Experiments 3 and 9 are also too skeptical or insufficiently protected by tallying (0.347 versus 0.451 and 0.191 versus 0.325). This combination indicates that simply lowering tally sensitivity and its power exponent further is unlikely to work: the model needs relatively more separation at intermediate margins but less escalation at the largest margins.",
  "verdict": "regenerate",
  "rationale": "Build on the accepted candidate, since the previous recalibration helped, but use different in-family shape adjustments rather than another global reduction of reactance or tally gain. First, replace the nearly linear-at-zero coalition accumulation tanh(a*x) with a smooth convex-input accumulator such as tanh(a*sign(x)*abs(x)^q), with q modestly above 1 and jointly recalibrated a. This remains a monotone coalition-level reactance mechanism with no endpoint notch or bonus. It should selectively weaken moderate narrow contrasts driving the large Experiment 5 and Experiment 8 errors while retaining maximal 100%-versus-50% reactance in Experiment 10 and strong repeated-consensus contrasts in Experiments 7 and 12. Second, reshape rather than uniformly weaken the tally margin function. Introduce a smooth upper-margin normalization or saturation parameter, while retaining monotonic nonlinear growth: keep one-vote influence low, increase relative sensitivity around intermediate margins, and compress increments by margins five and above. This should raise Experiment 2's Bayes factor toward 89 without restoring the near-deterministic choices in Experiments 4 and 6, and should expose more subject heterogeneity at those large margins. Check the joint change against Experiments 1 and 3 so one-vote conflicts remain susceptible, and against Experiment 9 and Experiment 11, where the current model has become too anti-high-validity. Avoid increasing lapse globally or simply lowering the breadth-reinforcement floor again, because either would damage the exact concentrated endpoint fit and the successful broad-coalition reversals."
}
```

## Usage

```json
{
  "input_tokens": 25929,
  "output_tokens": 2227
}
```
