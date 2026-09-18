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
- THEORY 1 = `pi_8`
- THEORY 2 = `pi_6`
- The recommendation below acts on THEORY 2 (= `pi_6`).

Replace pi_6 with a Capacity-Limited Pragmatic Cue Allocation theory. The new theory should posit two separable signals: a categorical vote signal and a claim-interpretation signal. Communicated validity should induce a smooth, predominantly skeptical or reactance-like contribution whose polarity and strength vary continuously across people, rather than assigning almost everyone to a categorical reversed-validity class. Crucially, the claim signal should compete for finite attention across all discriminating experts, including nominally 50%-valid experts. As coalition breadth and compositional complexity increase, attention to any single high-validity claim should decline, generating Experiment 2's positive breadth gradient even though signed validity imbalance is constant. Raw tally evidence should increasingly dominate at large margins, preserving the results of Experiments 6 and 8.

Unlike pi_8, the new theory should not posit a special 95% compression notch, a literal-certainty bonus, or hand-built endpoint-versus-neutral exemptions. Claim strength should be a smooth function over the communicated scale, approximately locally linear from 90% to 100%, so Experiment 1 has near-zero curvature while still allowing strong overall skepticism toward highly assertive claims. Context dependence should arise from a single general allocation principle—such as divisive normalization by effective active-set size and similarity-based competition—not separate breadth, within-side dispersion, across-side dispersion, sparse-conflict, and endpoint exception terms. The theory should also allow continuous subject variation in skepticism, tally sensitivity, and attentional capacity, which may accommodate the elevated observed heterogeneity in Experiment 2. Its key joint predictions should be: negligible 90/95/100 curvature; strong attenuation of isolated validity evidence with coalition breadth; frequent anti-validity choices in weak or tied tallies; and reliable raw-tally dominance as the vote margin becomes decisive.


## CANDIDATE THEORY
Finite-Pool Pragmatic Claim Competition theory proposes that binary expert endorsements generate two separable signals. A categorical vote channel counts every discriminating expert equally and therefore represents coalition consensus independently of communicated validity. A pragmatic claim channel interprets an expert's stated validity as the assertiveness of the expert's claim. Claim assertiveness is encoded by a smooth linear function of the communicated scale, including from 90% through 100%. Most people respond skeptically to assertive claims, although the polarity and magnitude of this interpretation vary continuously across individuals and can include weakly conventional interpretations. Claim processing draws on a finite attentional pool. Every discriminating expert, including a nominally 50%-valid expert whose own claim carries no directional strength, competes for that pool. Competition increases both with active-set size and with similarity among simultaneously active claims. Consequently, an isolated high-validity claim can receive substantial attention, whereas its influence declines when embedded in a broad or compositionally redundant coalition. This single allocation rule produces breadth and composition effects without endpoint bonuses, validity notches, or condition-specific exemptions. Pragmatic claim evidence is bounded, while categorical tally evidence continues to increase with vote margin. Skepticism can thus reverse tied or weak tallies but normally yields to a decisive raw-vote coalition. In the absence of outcome feedback, neither signal is recalibrated from choice history.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Finite-Pool Pragmatic Claim Competition model. History is deliberately
    # ignored because previous choices provide no feedback about cue accuracy.
    stimulus = np.asarray(state, dtype=np.float64)
    if stimulus.ndim != 2 or stimulus.shape[0] != 2:
        raise ValueError(
            f"Expected state with shape (2, n_features); got {stimulus.shape}."
        )

    n_features = stimulus.shape[1]
    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} does not match "
            f"n_features {n_features}."
        )

    # +1 denotes a unique endorsement of A, -1 a unique endorsement of B.
    signed_votes = np.sign(stimulus[0] - stimulus[1])
    active = signed_votes != 0.0
    tally_margin = float(np.sum(signed_votes))

    # Categorical votes and interpreted numerical claims are represented in
    # separate channels. This exactly linear assertiveness code has no special
    # status for 95% or 100% and therefore has zero intrinsic 90/95/100
    # curvature.
    v = np.clip(validities, 0.5, 1.0)
    claim_assertiveness = 2.0 * (v - 0.5)

    if not np.any(active):
        allocated_claim_contrast = 0.0
    else:
        active_signs = signed_votes[active]
        active_claims = claim_assertiveness[active]
        active_count = active_claims.size

        # All discriminating experts enter the competition matrix. Thus a 50%
        # expert has zero directional claim strength but still consumes claim-
        # interpretation capacity. Competition has one general form: a global
        # component plus a similarity-dependent component.
        width = float(parameters["similarity_width"])
        competition_floor = float(parameters["competition_floor"])
        distances = np.abs(active_claims[:, None] - active_claims[None, :])
        similarity = np.exp(-distances / width)
        pairwise_competition = (
            competition_floor + (1.0 - competition_floor) * similarity
        )
        np.fill_diagonal(pairwise_competition, 0.0)
        competition_load = np.sum(pairwise_competition, axis=1)

        # Finite capacity determines the attention retained by each claim.
        # Similar claims in broad coalitions strongly compete, while isolated
        # claims receive most of the available processing weight.
        capacity = float(parameters["attentional_capacity"])
        attention = capacity / (capacity + competition_load)
        allocated_claim_contrast = float(
            np.dot(active_signs, attention * active_claims)
        )

    # Pragmatic interpretation is continuous across subjects. Positive values
    # encode skepticism/reactance and reverse the communicated claim contrast;
    # zero ignores it; negative values produce a conventional interpretation.
    # The declared range makes skepticism predominant without introducing a
    # categorical latent class.
    accumulation = float(parameters["claim_accumulation"])
    bounded_claim = float(np.tanh(accumulation * allocated_claim_contrast))
    claim_evidence = -float(parameters["claim_construal"]) * bounded_claim

    # The categorical vote channel grows smoothly with margin. Its saturation
    # level exceeds the maximum typical claim contribution, so large coalitions
    # reliably dominate even though claim evidence can reverse weak margins.
    tally_scale = float(parameters["tally_saturation"])
    tally_evidence = (
        float(parameters["tally_sensitivity"])
        * tally_scale
        * np.tanh(tally_margin / tally_scale)
    )

    total_evidence = tally_evidence + claim_evidence
    decision_variable = float(parameters["choice_sensitivity"]) * total_evidence

    # Positive evidence favors A. Symmetric logits ensure exact option-swap
    # invariance.
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
- claim_construal: [-0.2, 4.5]
- claim_accumulation: [0.8, 1.5]
- attentional_capacity: [0.55, 2.4]
- competition_floor: [0.22, 0.58]
- similarity_width: [0.16, 0.42]
- tally_sensitivity: [0.85, 1.25]
- tally_saturation: [4.0, 5.8]
- choice_sensitivity: [0.72, 1.30]
- lapse_rate: [0.0, 0.10]
- validities: validities

`rationale`:
The model directly implements the arbiter's requested two-signal architecture while replacing pi_6's almost categorical reversed-validity population with a continuous claim-construal parameter. Nearly all of its range is skeptical, but it passes smoothly through zero into weak conventional interpretation, allowing graded between-subject variation in both polarity and strength. Communicated validity is encoded linearly over the entire 50%-to-100% range, so the model introduces no 95% compression notch, endpoint bonus, or literal-certainty exception. In particular, the intrinsic second difference at 90%, 95%, and 100% is zero. The key new mechanism is cue-level finite-pool competition. Every active expert enters the same competition matrix, even when its validity is 50% and its directional claim value is zero. Attention falls with the number of competing claims and falls especially when claims are similar, providing one general explanation for breadth and compositional effects. This should attenuate isolated high-validity evidence as coalition breadth grows and produce the positive breadth contrast in Experiment 14 without hand-coded breadth conditions. The same mechanism preserves strong anti-validity behavior in sparse ties such as Experiments 8 and 10, while supporting broad anti-validity effects in Experiments 7, 11, and 12 when multiple assertive claims remain directionally coherent. Claim evidence is bounded, whereas raw tally evidence grows with margin, so decisive margins continue to control choices in Experiments 4, 6, and 8. Continuous variation in construal, capacity, tally sensitivity, and choice sensitivity also supplies the elevated subject heterogeneity that the categorical pi_6 model could not express.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1 (current candidate you are grading): loss=0.2275 -> ACCEPTED
Running-best (last accepted) base: iter 1 at loss=0.2275.

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
**Candidate (simulated) value:** 0.1913 (var=0.0055)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8450 (var=0.0143)
- pi_2: 0.1350 (var=0.0104)
- pi_3: 0.3775 (var=0.0056)
- pi_4: 0.3392 (var=0.0040)
- pi_5: 0.3108 (var=0.0052)
- pi_6: 0.4721 (var=0.0071)
- pi_7: 0.0896 (var=0.0023)
- pi_8: 0.3383 (var=0.0039)

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
**Candidate (simulated) value:** 95.5147 (var=4.5100)
**Other theories' values on this metric (for reference):**
- pi_2: 1.9009 (var=0.0916)
- pi_1: 0.5503 (var=0.0488)
- pi_3: 49.6494 (var=1.7369)
- pi_4: 100.3141 (var=2.8389)
- pi_5: 59.0660 (var=1.4267)
- pi_6: 84.3674 (var=2.5041)
- pi_7: 83.3362 (var=1.6438)
- pi_8: 56.1425 (var=1.2063)

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
**Candidate (simulated) value:** 0.2781 (var=0.0036)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5394 (var=0.0025)
- pi_2: 0.1190 (var=0.0098)
- pi_1: 0.7531 (var=0.0085)
- pi_4: 0.4698 (var=0.0021)
- pi_5: 0.4573 (var=0.0018)
- pi_6: 0.3958 (var=0.0029)
- pi_7: 0.1542 (var=0.0036)
- pi_8: 0.4106 (var=0.0035)

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
**Candidate (simulated) value:** 0.9669 (var=0.0013)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8606 (var=0.0060)
- pi_3: 0.3869 (var=0.0089)
- pi_1: 0.1325 (var=0.0094)
- pi_4: 0.7775 (var=0.0083)
- pi_5: 0.8337 (var=0.0061)
- pi_6: 0.7837 (var=0.0060)
- pi_7: 0.9413 (var=0.0016)
- pi_8: 0.8544 (var=0.0058)

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
**Candidate (simulated) value:** -0.3703 (var=0.1040)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5887 (var=0.0712)
- pi_4: 0.0071 (var=0.0478)
- pi_1: 0.5484 (var=0.0538)
- pi_2: -0.0256 (var=0.0579)
- pi_5: -0.0126 (var=0.0452)
- pi_6: -0.0866 (var=0.0563)
- pi_7: -0.5556 (var=0.0918)
- pi_8: 0.0104 (var=0.0390)

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
**Candidate (simulated) value:** 0.9515 (var=0.0010)
**Other theories' values on this metric (for reference):**
- pi_4: 0.8037 (var=0.0033)
- pi_3: 0.5665 (var=0.0017)
- pi_1: 0.8565 (var=0.0102)
- pi_2: 0.8612 (var=0.0065)
- pi_5: 0.8506 (var=0.0014)
- pi_6: 0.8273 (var=0.0015)
- pi_7: 0.8325 (var=0.0029)
- pi_8: 0.8592 (var=0.0031)

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
**Candidate (simulated) value:** 0.4194 (var=0.0547)
**Other theories' values on this metric (for reference):**
- pi_5: 0.7494 (var=0.0030)
- pi_4: 0.5919 (var=0.0025)
- pi_1: 0.8622 (var=0.0079)
- pi_2: 0.8381 (var=0.0106)
- pi_3: 0.9864 (var=0.0001)
- pi_6: 0.1972 (var=0.0033)
- pi_7: 0.2755 (var=0.0544)
- pi_8: 0.2347 (var=0.0051)

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
**Candidate (simulated) value:** -2.7763 (var=0.0719)
**Other theories' values on this metric (for reference):**
- pi_4: 0.4221 (var=0.0108)
- pi_5: -0.0028 (var=0.0102)
- pi_1: 4.5627 (var=0.0649)
- pi_2: 0.0176 (var=0.0099)
- pi_3: 3.9747 (var=0.0054)
- pi_6: -3.4648 (var=0.0108)
- pi_7: -1.4013 (var=0.0925)
- pi_8: -4.4717 (var=0.0102)

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
**Candidate (simulated) value:** 0.3378 (var=0.0216)
**Other theories' values on this metric (for reference):**
- pi_5: 0.6632 (var=0.0032)
- pi_6: 0.1697 (var=0.0029)
- pi_1: 0.7324 (var=0.0054)
- pi_2: 0.5977 (var=0.0022)
- pi_3: 0.9801 (var=0.0004)
- pi_4: 0.5348 (var=0.0021)
- pi_7: 0.3215 (var=0.0354)
- pi_8: 0.2814 (var=0.0029)

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
**Candidate (simulated) value:** 0.7642 (var=0.0215)
**Other theories' values on this metric (for reference):**
- pi_6: 0.7625 (var=0.0137)
- pi_5: 0.5375 (var=0.0072)
- pi_1: 0.1683 (var=0.0142)
- pi_2: 0.4767 (var=0.0123)
- pi_3: 0.0008 (var=0.0000)
- pi_4: 0.4383 (var=0.0129)
- pi_7: 0.8467 (var=0.0208)
- pi_8: 0.8408 (var=0.0059)

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
**Candidate (simulated) value:** 0.5831 (var=0.0252)
**Other theories' values on this metric (for reference):**
- pi_7: 0.9125 (var=0.0016)
- pi_6: 0.2006 (var=0.0082)
- pi_1: 0.8514 (var=0.0098)
- pi_2: 0.8394 (var=0.0083)
- pi_3: 0.7772 (var=0.0054)
- pi_4: 0.5706 (var=0.0031)
- pi_5: 0.6192 (var=0.0027)
- pi_8: 0.2039 (var=0.0054)

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
**Candidate (simulated) value:** 0.3431 (var=0.0184)
**Other theories' values on this metric (for reference):**
- pi_6: 0.2356 (var=0.0041)
- pi_7: 0.8364 (var=0.0032)
- pi_1: 0.8367 (var=0.0093)
- pi_2: 0.4964 (var=0.0036)
- pi_3: 0.7497 (var=0.0047)
- pi_4: 0.5364 (var=0.0028)
- pi_5: 0.4892 (var=0.0026)
- pi_8: 0.1583 (var=0.0037)

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
**Candidate (simulated) value:** 0.0191 (var=0.0090)
**Other theories' values on this metric (for reference):**
- pi_8: 0.0569 (var=0.0085)
- pi_6: -0.0147 (var=0.0080)
- pi_1: -0.0147 (var=0.0061)
- pi_2: 0.0028 (var=0.0045)
- pi_3: -0.0878 (var=0.0068)
- pi_4: -0.0187 (var=0.0085)
- pi_5: 0.0069 (var=0.0102)
- pi_7: -0.0338 (var=0.0098)

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
**Candidate (simulated) value:** 0.1075 (var=0.0105)
**Other theories' values on this metric (for reference):**
- pi_6: -0.0096 (var=0.0069)
- pi_8: 0.1467 (var=0.0083)
- pi_1: 0.0031 (var=0.0052)
- pi_2: 0.0064 (var=0.0127)
- pi_3: 0.0007 (var=0.0000)
- pi_4: -0.0000 (var=0.0117)
- pi_5: -0.0346 (var=0.0125)
- pi_7: -0.0029 (var=0.0065)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate is faithful to the prescribed mechanism family and establishes two important successes: Experiment 13 has negligible 90/95/100 curvature (0.019 versus 0.009), and Experiment 14 produces the required positive breadth gradient (0.108 versus 0.178). It also fits Experiment 2's tally-margin signature reasonably well (95.5 versus 89.1) and Experiment 9's mean almost exactly. However, the joint behavioral pattern is not yet captured. Tally evidence is too dominant: TTB-conflict choices are too rare in Experiment 1 (0.191 versus 0.403), minority-tally choices are too rare in Experiment 3 (0.278 versus 0.451), and decisive-tally choice is too deterministic in Experiments 4 and 6 (0.967 versus 0.821 and 0.952 versus 0.846). At the same time, skeptical claim evidence is too weak relative to a one-vote tally advantage in Experiments 7 and 11 (high-validity-side choice 0.419 versus 0.116 and 0.583 versus 0.157), and too weak even in the tied-tally contrasts of Experiments 8, 10, and 12. Experiment 5 is an important counterconstraint: its breadth/composition contrast is already much too negative (-0.370 versus -0.038), so a blanket increase in claim strength would worsen that result. Between-subject variability is also badly distributed: it is excessive in Experiments 5, 7, 9, and 11 while remaining below the elevated variance observed in Experiment 2.",
  "verdict": "regenerate",
  "rationale": "Keep the finite-pool two-signal architecture, but recalibrate the balance and normalization. First, reduce the tally-sensitivity range modestly rather than weakening large-margin dominance through a new mechanism. This single change should improve Experiments 1, 2, 3, 4, 6, 7, and 11 in the observed directions: weak margins would interfere less with skeptical claims, while the bounded claim channel still allows large margins to dominate. Second, strengthen skeptical claim influence moderately, especially for tied or sparse conflicts, to move Experiments 8, 10, and 12 toward the data. Do not implement this as an unrestricted increase in claim_construal, because Experiment 5 is already overexpressing the breadth contrast. Instead, retune the existing divisive allocation\u2014such as slightly stronger active-set competition together with a compensating increase in claim accumulation\u2014so isolated claims remain potent while broad effects are controlled. This should also enlarge Experiment 14's currently underestimated attenuation. Finally, narrow the extreme claim-construal/capacity ranges or reduce their interaction to curb the excessive heterogeneity in Experiments 5, 7, 9, and 11, while preserving continuous rather than categorical subject variation. The current model is a promising and faithful first instantiation, but the large systematic misses across Experiments 1, 3, 5, 7, 11, and 12 are too substantial to ship at aggregate loss 0.2275."
}
```

## Usage

```json
{
  "input_tokens": 21579,
  "output_tokens": 2100
}
```
