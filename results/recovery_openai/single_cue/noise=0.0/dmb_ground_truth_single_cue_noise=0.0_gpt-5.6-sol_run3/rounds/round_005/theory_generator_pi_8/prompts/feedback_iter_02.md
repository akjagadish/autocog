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
- THEORY 1 = `pi_7`
- THEORY 2 = `pi_6`
- The recommendation below acts on THEORY 1 (= `pi_7`).

Replace pi_7 with a new Contextual Anti-Reliability Arbitration theory. It should propose that communicated numerical validities are commonly treated as socially or pragmatically suspect rather than as direct diagnostic weights: above-chance claims generally contribute evidence opposite to their stated direction, including moderate 60%-70% claims, while near-perfect claims can evoke an additional endpoint distrust response. Unlike pi_6, this should not posit an almost categorical 99% reversed-construal subgroup or exact-tie and one-margin gates. Instead, use continuous subject heterogeneity in anti-reliability strength and a continuous composition-sensitive arbitration rule. Negative claim evidence should be strongest when the validity contrast is concentrated and the raw tally is tied or weak, should be divisively normalized or diluted in dense heterogeneous coalitions, and should lose control smoothly as the absolute tally margin becomes decisive. This architecture should preserve option-swap symmetry and history independence. It should reproduce the very low conventional-validity choice rates in Experiments 1, 2, 9, and 10; permit near-indifference and minority choices in Experiments 3, 5, and 7; preserve large-coalition tally dominance in Experiments 4, 6, and 8; and include an endpoint-specific distrust component strong enough for Experiment 12. To address Experiment 11, anti-reliability influence should attenuate when a high-validity side is embedded in a broad coalition or when opposing evidence is compositionally diffuse, without relying on pi_6's discontinuous special cases. Parameter ranges should allow more anti-validity strength and heterogeneity than pi_6 currently does, because Experiments 1 and 2 are more extreme—and Experiment 2 more variable—than its simulations predict. This new theory would be a genuine competitor to pi_6: both predict broad anti-validity behavior, but they differ on whether it reflects categorical scale reversal with hand-built gates or continuous pragmatic distrust regulated by evidence concentration and coalition composition.


## CANDIDATE THEORY
Contextual Anti-Reliability Arbitration theory proposes that communicated validity numbers are interpreted partly as pragmatic claims whose assertiveness can undermine their credibility. Every above-chance claim therefore generates continuous evidence against the option endorsed by that expert, rather than being categorically reversed by a special subgroup. The magnitude of this anti-reliability response varies continuously across people. Near-perfect claims additionally evoke endpoint distrust because unusually strong assurances are perceived as especially suspicious. Anti-reliability evidence is accumulated separately from the raw vote tally and then divisively normalized by coalition breadth and within-side compositional heterogeneity. Its influence is strongest when the validity contrast is concentrated on one side and the tally is tied or weak. It is diluted when both options are supported by dense, internally heterogeneous coalitions and decays smoothly as the absolute tally margin grows. Thus concentrated validity contrasts can reverse weak tallies, whereas decisive vote coalitions remain dominant. The mechanism is continuous throughout, option-swap symmetric, and history independent because choices are followed by no accuracy feedback.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Contextual Anti-Reliability Arbitration. History is intentionally ignored:
    # without outcome feedback, prior choices cannot reveal expert accuracy.
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

    # Positive endorsements favor A and negative endorsements favor B.
    signed_votes = np.sign(stimulus[0] - stimulus[1])
    active = signed_votes != 0.0
    tally_margin = float(np.sum(signed_votes))
    active_count = int(np.count_nonzero(active))

    # Encode the assertiveness of every communicated above-chance claim.
    # Moderate claims already carry pragmatic suspicion. The ordinary channel
    # is smoothly compressed for very high claims, preventing a 95% claim from
    # receiving both maximal ordinary distrust and endpoint amplification.
    v = np.clip(validities, 0.5, 1.0)
    above_chance = np.clip(2.0 * (v - 0.5), 0.0, 1.0)
    ordinary_base = np.power(above_chance, 0.82)

    def logistic(x):
        if x >= 0.0:
            return 1.0 / (1.0 + np.exp(-x))
        ex = np.exp(x)
        return ex / (1.0 + ex)

    upper_compression = np.array(
        [logistic(50.0 * (float(x) - 0.86)) for x in above_chance],
        dtype=np.float64,
    )
    ordinary_claim = ordinary_base * (1.0 - 0.42 * upper_compression)

    # Endpoint distrust is reserved for claims extremely close to certainty.
    endpoint_low = logistic(100.0 * (0.0 - 0.98))
    endpoint_high = logistic(100.0 * (1.0 - 0.98))
    endpoint_curve = np.array(
        [
            (logistic(100.0 * (float(x) - 0.98)) - endpoint_low)
            / (endpoint_high - endpoint_low)
            for x in above_chance
        ],
        dtype=np.float64,
    )
    endpoint_curve = np.clip(endpoint_curve, 0.0, 1.0)
    claim_strength = ordinary_claim + float(parameters["endpoint_distrust"]) * endpoint_curve

    if active_count == 0:
        anti_reliability = 0.0
    else:
        active_signs = signed_votes[active]
        active_claims = claim_strength[active]
        signed_claim_contrast = float(np.dot(active_signs, active_claims))
        total_claim_mass = float(np.sum(active_claims))

        # Relative contrast measures whether distrust is directionally
        # concentrated rather than spread across two similarly assertive sides.
        concentration = abs(signed_claim_contrast) / (total_claim_mass + 1e-12)
        concentration = float(np.clip(concentration, 0.0, 1.0))
        focus_floor = float(parameters["concentration_floor"])
        focus = focus_floor + (1.0 - focus_floor) * concentration ** 1.35

        # Estimate heterogeneity within each endorsing coalition. Mixing claims
        # of substantially different strength on the same side makes their
        # pragmatic interpretation less coherent and hence more strongly diluted.
        positive_claims = active_claims[active_signs > 0.0]
        negative_claims = active_claims[active_signs < 0.0]

        def side_dispersion(values):
            if values.size <= 1:
                return 0.0
            mean_value = float(np.mean(values))
            return float(np.mean((values - mean_value) ** 2))

        positive_dispersion = side_dispersion(positive_claims)
        negative_dispersion = side_dispersion(negative_claims)
        within_dispersion = (
            positive_claims.size * positive_dispersion
            + negative_claims.size * negative_dispersion
        ) / float(active_count)

        breadth_exponent = float(parameters["coalition_dilution_exponent"])
        breadth_normalizer = float(active_count) ** breadth_exponent
        heterogeneity_normalizer = 1.0 + (
            float(parameters["composition_dilution"])
            * float(active_count)
            * within_dispersion
        )
        contextual_contrast = (
            signed_claim_contrast * focus
            / (breadth_normalizer * heterogeneity_normalizer + 1e-12)
        )

        # The anti-reliability signal is bounded and continuously heterogeneous
        # across subjects. Its sign opposes the communicated validity contrast.
        accumulated = np.tanh(
            float(parameters["anti_accumulation"]) * contextual_contrast
        )
        anti_reliability = -float(parameters["anti_reliability_strength"]) * float(accumulated)

        # A smooth arbitration function cedes control to the tally as its margin
        # becomes decisive; there are no exact-tie or one-margin special cases.
        margin_scale = float(parameters["margin_protection_scale"])
        margin_power = float(parameters["margin_protection_power"])
        margin_gate = 1.0 / (
            1.0 + (abs(tally_margin) / margin_scale) ** margin_power
        )
        anti_reliability *= float(margin_gate)

    # Raw votes retain their categorical status but saturate gradually, limiting
    # overconfidence while preserving dominance by large vote coalitions.
    tally_evidence = 3.8 * np.tanh(tally_margin / 3.8)
    total_evidence = tally_evidence + anti_reliability

    evidence_scale = float(parameters["evidence_scale"])
    normalized_evidence = float(np.tanh(total_evidence / evidence_scale))
    decision_variable = float(parameters["choice_sensitivity"]) * normalized_evidence

    # Positive evidence favors A. Symmetric logits guarantee that swapping the
    # displayed options swaps their probabilities exactly.
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
            probabilities.size, 1.0 / probabilities.size, dtype=np.float64
        )
    else:
        probabilities /= total
    return int(np.random.choice(probabilities.size, p=probabilities))

`parameters`:
- anti_reliability_strength: [3.0, 5.6]
- anti_accumulation: [1.8, 3.2]
- endpoint_distrust: [0.95, 1.85]
- coalition_dilution_exponent: [0.52, 0.82]
- composition_dilution: [0.6, 1.8]
- concentration_floor: [0.30, 0.55]
- margin_protection_scale: [1.55, 2.25]
- margin_protection_power: [2.2, 3.4]
- evidence_scale: [2.2, 3.0]
- choice_sensitivity: [1.65, 2.35]
- lapse_rate: [0.0, 0.10]
- validities: validities

`rationale`:
This is a selective edit to the accepted iteration-1 model. All arbitration, breadth normalization, tally saturation, heterogeneity, symmetry, and history-independence mechanisms are unchanged. Only the separation between ordinary and endpoint distrust is sharpened. The ordinary channel is smoothly compressed above an above-chance code of roughly 0.86, so a 95%-valid expert no longer receives nearly maximal ordinary distrust. The endpoint sigmoid is moved from 0.88 to 0.98 and made much steeper, making its contribution negligible at 95% validity but fully active at 100%. The endpoint-distrust range is shifted upward to approximately preserve the accepted model's total anti-evidence for genuinely perfect claims. This should substantially reduce the excessive rejection of the 95%-endorsed option in Experiment 5 while preserving the concentrated 100%-versus-50% reversal in Experiment 10 and the endpoint effects in Experiment 8. Moderate 60%-70% claims and broad-coalition normalization are essentially unchanged, avoiding the rejected iteration's deterioration in Experiments 7, 11, and 12. The accepted tally transformation is also retained to avoid repeating the Experiment 2 overshoot.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.1379 -> ACCEPTED
- iter 2: loss=0.1534 -> REJECTED
- iter 3 (current candidate you are grading): loss=0.0897 -> ACCEPTED
Running-best (last accepted) base: iter 3 at loss=0.0897.

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
  - iter 1: 0.2450 (var=0.0029) (Δ vs real -0.1583)
  - iter 2: 0.2958 (var=0.0043) (Δ vs real -0.1075)
  - iter 3 (current): 0.3371 (var=0.0045) (Δ vs real -0.0662)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8450 (var=0.0143)
- pi_2: 0.1350 (var=0.0104)
- pi_3: 0.3775 (var=0.0056)
- pi_4: 0.3392 (var=0.0040)
- pi_5: 0.3108 (var=0.0052)
- pi_6: 0.4721 (var=0.0071)
- pi_7: 0.0896 (var=0.0023)

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
  - iter 1: 62.5625 (var=2.3906) (Δ vs real -26.5797)
  - iter 2: 134.7762 (var=4.5803) (Δ vs real +45.6340)
  - iter 3 (current): 56.9131 (var=1.3751) (Δ vs real -32.2292)
**Other theories' values on this metric (for reference):**
- pi_2: 1.9009 (var=0.0916)
- pi_1: 0.5503 (var=0.0488)
- pi_3: 49.6494 (var=1.7369)
- pi_4: 100.3141 (var=2.8389)
- pi_5: 59.0660 (var=1.4267)
- pi_6: 84.3674 (var=2.5041)
- pi_7: 83.3362 (var=1.6438)

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
  - iter 1: 0.3467 (var=0.0021) (Δ vs real -0.1042)
  - iter 2: 0.3623 (var=0.0025) (Δ vs real -0.0885)
  - iter 3 (current): 0.3927 (var=0.0037) (Δ vs real -0.0581)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5394 (var=0.0025)
- pi_2: 0.1190 (var=0.0098)
- pi_1: 0.7531 (var=0.0085)
- pi_4: 0.4698 (var=0.0021)
- pi_5: 0.4573 (var=0.0018)
- pi_6: 0.3958 (var=0.0029)
- pi_7: 0.1542 (var=0.0036)

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
  - iter 1: 0.8475 (var=0.0054) (Δ vs real +0.0262)
  - iter 2: 0.8213 (var=0.0063) (Δ vs real +0.0000)
  - iter 3 (current): 0.8431 (var=0.0027) (Δ vs real +0.0219)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8606 (var=0.0060)
- pi_3: 0.3869 (var=0.0089)
- pi_1: 0.1325 (var=0.0094)
- pi_4: 0.7775 (var=0.0083)
- pi_5: 0.8337 (var=0.0061)
- pi_6: 0.7837 (var=0.0060)
- pi_7: 0.9413 (var=0.0016)

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
  - iter 1: -0.6893 (var=0.0459) (Δ vs real -0.6512)
  - iter 2: -0.6038 (var=0.0941) (Δ vs real -0.5657)
  - iter 3 (current): -0.0890 (var=0.0375) (Δ vs real -0.0508)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5887 (var=0.0712)
- pi_4: 0.0071 (var=0.0478)
- pi_1: 0.5484 (var=0.0538)
- pi_2: -0.0256 (var=0.0579)
- pi_5: -0.0126 (var=0.0452)
- pi_6: -0.0866 (var=0.0563)
- pi_7: -0.5556 (var=0.0918)

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
  - iter 1: 0.8279 (var=0.0019) (Δ vs real -0.0183)
  - iter 2: 0.8140 (var=0.0021) (Δ vs real -0.0323)
  - iter 3 (current): 0.8323 (var=0.0020) (Δ vs real -0.0140)
**Other theories' values on this metric (for reference):**
- pi_4: 0.8037 (var=0.0033)
- pi_3: 0.5665 (var=0.0017)
- pi_1: 0.8565 (var=0.0102)
- pi_2: 0.8612 (var=0.0065)
- pi_5: 0.8506 (var=0.0014)
- pi_6: 0.8273 (var=0.0015)
- pi_7: 0.8325 (var=0.0029)

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
  - iter 1: 0.1955 (var=0.0042) (Δ vs real +0.0794)
  - iter 2: 0.2509 (var=0.0098) (Δ vs real +0.1348)
  - iter 3 (current): 0.2263 (var=0.0032) (Δ vs real +0.1102)
**Other theories' values on this metric (for reference):**
- pi_5: 0.7494 (var=0.0030)
- pi_4: 0.5919 (var=0.0025)
- pi_1: 0.8622 (var=0.0079)
- pi_2: 0.8381 (var=0.0106)
- pi_3: 0.9864 (var=0.0001)
- pi_6: 0.1972 (var=0.0033)
- pi_7: 0.2755 (var=0.0544)

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
  - iter 1: -4.5609 (var=0.0087) (Δ vs real -1.0237)
  - iter 2: -3.5294 (var=0.0204) (Δ vs real +0.0078)
  - iter 3 (current): -4.4411 (var=0.0084) (Δ vs real -0.9039)
**Other theories' values on this metric (for reference):**
- pi_4: 0.4221 (var=0.0108)
- pi_5: -0.0028 (var=0.0102)
- pi_1: 4.5627 (var=0.0649)
- pi_2: 0.0176 (var=0.0099)
- pi_3: 3.9747 (var=0.0054)
- pi_6: -3.4648 (var=0.0108)
- pi_7: -1.4013 (var=0.0925)

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
  - iter 1: 0.2965 (var=0.0029) (Δ vs real -0.0283)
  - iter 2: 0.2591 (var=0.0061) (Δ vs real -0.0657)
  - iter 3 (current): 0.3005 (var=0.0038) (Δ vs real -0.0243)
**Other theories' values on this metric (for reference):**
- pi_5: 0.6632 (var=0.0032)
- pi_6: 0.1697 (var=0.0029)
- pi_1: 0.7324 (var=0.0054)
- pi_2: 0.5977 (var=0.0022)
- pi_3: 0.9801 (var=0.0004)
- pi_4: 0.5348 (var=0.0021)
- pi_7: 0.3215 (var=0.0354)

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
  - iter 1: 0.8467 (var=0.0067) (Δ vs real -0.0150)
  - iter 2: 0.8158 (var=0.0074) (Δ vs real -0.0458)
  - iter 3 (current): 0.8283 (var=0.0071) (Δ vs real -0.0333)
**Other theories' values on this metric (for reference):**
- pi_6: 0.7625 (var=0.0137)
- pi_5: 0.5375 (var=0.0072)
- pi_1: 0.1683 (var=0.0142)
- pi_2: 0.4767 (var=0.0123)
- pi_3: 0.0008 (var=0.0000)
- pi_4: 0.4383 (var=0.0129)
- pi_7: 0.8467 (var=0.0208)

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
  - iter 1: 0.2303 (var=0.0044) (Δ vs real +0.0731)
  - iter 2: 0.2994 (var=0.0131) (Δ vs real +0.1422)
  - iter 3 (current): 0.2242 (var=0.0050) (Δ vs real +0.0669)
**Other theories' values on this metric (for reference):**
- pi_7: 0.9125 (var=0.0016)
- pi_6: 0.2006 (var=0.0082)
- pi_1: 0.8514 (var=0.0098)
- pi_2: 0.8394 (var=0.0083)
- pi_3: 0.7772 (var=0.0054)
- pi_4: 0.5706 (var=0.0031)
- pi_5: 0.6192 (var=0.0027)

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
  - iter 1: 0.1675 (var=0.0030) (Δ vs real +0.0414)
  - iter 2: 0.2369 (var=0.0047) (Δ vs real +0.1108)
  - iter 3 (current): 0.1761 (var=0.0018) (Δ vs real +0.0500)
**Other theories' values on this metric (for reference):**
- pi_6: 0.2356 (var=0.0041)
- pi_7: 0.8364 (var=0.0032)
- pi_1: 0.8367 (var=0.0093)
- pi_2: 0.4964 (var=0.0036)
- pi_3: 0.7497 (var=0.0047)
- pi_4: 0.5364 (var=0.0028)
- pi_5: 0.4892 (var=0.0026)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed Contextual Anti-Reliability Arbitration family and captures several important qualitative patterns: endpoint reversal in Experiment 10, large-margin tally dominance in Experiments 4 and 6, anti-validity behavior in Experiments 7, 9, 11, and 12, and option-swap symmetry. However, the fit is not yet experiment-invariant. The decisive failure is Experiment 5, where the predicted contrast is far too negative (-0.689 versus -0.038), indicating excessive anti-reliability control in sparse, equal-tally coalitions. The same sparse-conflict overreaction appears in Experiment 1 (TTB choice 0.245 versus 0.403), Experiment 3 (minority choice 0.347 versus 0.451), and Experiment 8 (alignment -4.561 versus -3.537). Conversely, anti-validity is somewhat too weak in broader coalitions: conventional/high-validity choices remain too high in Experiments 7 (0.196 versus 0.116), 11 (0.230 versus 0.157), and 12 (0.168 versus 0.126). Thus the current breadth normalization has the wrong quantitative gradient: it suppresses anti-reliability too strongly as coalitions become broad while leaving it too strong in sparse non-endpoint conflicts. Experiment 2 also shows insufficient margin-dependent tally evidence (log Bayes factor 62.6 versus 89.1) and substantially too little between-subject variance. Variance is generally compressed across experiments, consistent with the nested tanh transformations saturating individual differences.
Rationale: Retain the prescribed continuous pragmatic-distrust architecture, but rebalance sparse versus dense arbitration. Lower ordinary non-endpoint anti-reliability strength or accumulation while substantially weakening the active-count breadth penalty (for example, lower the coalition-dilution exponent). This paired change can reduce excessive reversal in sparse conditions—especially Experiments 1, 3, 5, and 8—without weakening, and potentially strengthening, the broad-coalition reversals needed in Experiments 7, 11, and 12. Keep endpoint distrust separately strong, or raise its coefficient to compensate for the lower ordinary strength, so that the already accurate Experiment 10 result is preserved. Experiment 5 should be the primary calibration target: the present model effectively treats its sparse equal-tally contrasts as near-categorical reversals, contrary to the required continuous arbitration. In addition, make raw tally evidence somewhat less saturating across margins so Experiment 2 exhibits stronger margin-dependent tally support, while retuning overall sensitivity to avoid overshooting Experiments 4 and 6. Finally, reduce saturation of the anti-evidence transform or widen effective subject-level strength and sensitivity ranges; nominally broad parameter ranges currently yield much less variance than observed, particularly in Experiment 2. These are in-family normalization, scaling, and heterogeneity adjustments rather than a change of mechanism.

**Outcome of your advice:** iter 1 candidate loss=0.1379 -> iter 2 candidate loss=0.1534 -> the gate marked it REJECTED.

### Iteration 2 (most recent)
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed continuous Contextual Anti-Reliability Arbitration family, but it is not experiment-invariant and was rejected by the accept gate (loss 0.1534 versus the accepted base's 0.1379). The recalibration improved Experiments 1, 3, 4, 5, and especially 8, but most gains were too small or came at larger costs elsewhere. Experiment 5 remains the dominant failure: -0.604 versus -0.038 indicates near-categorical rejection of the 95%-endorsed option where humans are approximately balanced. The current endpoint implementation likely contributes to this because a 95% validity maps to above_chance=0.90 and still receives both nearly maximal ordinary distrust and a nontrivial endpoint bonus. Meanwhile, the broad-coalition fits deteriorated substantially in Experiments 7 (0.251 versus 0.116), 11 (0.299 versus 0.157), and 12 (0.237 versus 0.126), showing that the reduction in ordinary anti-reliability outweighed the weaker breadth dilution. Experiment 2 also overshot sharply from insufficient to excessive margin-dependent evidence (134.8 versus 89.1), while Experiments 6, 9, and 10 moved away from the data. Thus the previous recommendation to jointly lower ordinary distrust, weaken breadth normalization substantially, and increase tally-margin separation was not successful as implemented.
Rationale: Build from the unchanged accepted iteration-1 base and try a more selective in-family recalibration rather than repeating the rejected global strength/breadth changes. First, separate ordinary distrust more cleanly from endpoint distrust: prevent 95% claims from receiving almost maximal ordinary evidence plus appreciable endpoint amplification, while retaining a steep continuous endpoint response very close to 100%. For example, cap or compress the ordinary-claim component near its upper range and move the endpoint sigmoid threshold closer to above_chance 0.97-0.99 with a steeper slope. This directly targets Experiment 5 while preserving the genuinely endpoint-specific reversal in Experiment 10. Second, retain the accepted base's broad-coalition normalization or alter it only modestly; the large reduction in breadth dilution was rejected and worsened Experiments 7, 11, and 12 once paired with weaker ordinary strength. If sparse-versus-broad balance still needs adjustment, use a continuous side-composition normalization that discounts contrasts dominated by one near-endpoint expert, rather than lowering ordinary anti-reliability globally. Third, do not retain the rejected increase in tally dynamic range: use the accepted tally saturation or an intermediate value, because the current change drove Experiment 2 far past its target and slightly harmed Experiments 6 and 9. Preserve the continuous heterogeneity, symmetry, and history independence of the prescribed mechanism.

**Outcome of your advice:** iter 2 candidate loss=0.1534 -> CURRENT candidate loss=0.0897 -> the gate marked it ACCEPTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The accepted candidate is a substantial improvement and remains faithful to the prescribed continuous Contextual Anti-Reliability Arbitration family. The selective separation of ordinary from endpoint distrust worked: Experiment 5 improved dramatically from -0.689 to -0.089 against -0.038, while Experiments 1 and 3 also moved closer, and tally dominance remains accurate in Experiments 4 and 6. Experiments 9 and 10 are reasonably captured, and the model preserves symmetry and history independence. Nevertheless, important experiment-invariance failures remain. Anti-validity is still too weak in broad moderate-validity conflicts: high-validity choices are too frequent in Experiments 7 (0.226 versus 0.116), 11 (0.224 versus 0.157), and 12 (0.176 versus 0.126). Conversely, Experiment 8 is too strongly anti-validity (-4.441 versus -3.537), even though the concentrated endpoint reversal in Experiment 10 is slightly too weak (0.828 versus 0.862). This indicates that endpoint distrust is not sufficiently regulated by contrast concentration: it overacts in the more compositionally distributed Experiment 8 while underacting in the sparse 100%-versus-50% condition. Experiment 2 also remains materially low (56.9 versus 89.1), with much too little between-subject variance (1.38 versus 7.59), indicating insufficient margin-dependent separation and compressed heterogeneity. The current nested tanh transformations likely contribute to that compression.",
  "verdict": "regenerate",
  "rationale": "Retain the successful ordinary/endpoint separation introduced in this accepted iteration; the prior selective endpoint advice clearly helped and should not be undone. Make the next changes small and targeted. First, apply a stronger concentration-sensitive multiplier specifically to the endpoint component, leaving concentrated 100%-versus-50% contrasts nearly intact or slightly stronger while diluting endpoint evidence when it is embedded among several heterogeneous claims. This should move Experiment 8 upward without sacrificing Experiment 10 or reintroducing the Experiment 5 failure. Second, modestly reduce breadth or composition dilution for moderate 60%-70% claim contrasts so broad coalitions exert somewhat more anti-reliability influence in Experiments 7, 11, and 12. The previous large global reduction in breadth dilution was rejected, so this should be a smaller adjustment, preferably validity- and composition-sensitive rather than another global strength change. Third, use an intermediate tally dynamic range between the accepted saturation and the rejected iteration-2 setting: the former produces 56.9 in Experiment 2 and the latter produced 134.8, bracketing the observed 89.1. Do not repeat the full rejected increase. Finally, mildly reduce saturation in the final evidence mapping or widen effective choice-sensitivity heterogeneity to raise Experiment 2's subject variance, while monitoring the already reasonable means in Experiments 4 and 6. These are calibration changes within the prescribed arbitration mechanism, not a change of theory family."
}
```

## Usage

```json
{
  "input_tokens": 21652,
  "output_tokens": 1481
}
```
