# metric_exp00_attempt_00

## System Prompt

You are a psychology researcher proposing a metric in the Decision Making (Binary Features) domain.

Your goal is adversarial: propose a metric that DISCRIMINATES the two theories — i.e., its value, computed on data simulated under your advocated theory, should be as far as possible from its value computed on data simulated under the competing theory. The direction of the gap does not matter; what matters is that the two theories produce visibly different numbers on this metric. The metric is computed on the data collected from the experimental design provided in the prompt. Produce a metric where you're prediction will be much more accurate than the competing theory's prediction on human data.

Your metric is a Python function

    metric(data: pd.DataFrame) -> float

Available imports inside `metric`:
- numpy as np
- pandas as pd

The system evaluates your metric in two ways and reports the pair as `point_estimate (var=between_subject_variance)` everywhere downstream:
- `point_estimate` is `metric(data)` applied to the FULL pooled DataFrame (all subjects together) — the canonical scalar;
- `between_subject_variance` is the population variance (`ddof=0`) of `metric(subj_df)` re-applied per `subject_id`, summarising how stable the metric is across subjects. If your metric only makes sense on multi-subject data this will fall back to `n/a` and the metric is rejected (the acceptance test below cannot run without it). Prefer metrics that work both on the pooled DataFrame and on a single subject's slice.

Acceptance rule: the system simulates each theory and runs Welch's two-sample t-test on `(point_estimate_self, between_subject_variance_self, N)` vs. `(point_estimate_adv, between_subject_variance_adv, N)`, where N is the number of HUMAN subjects the experiment will actually be run with (a fixed small number, currently 25). Your metric is admitted iff the two-sided p-value is below the significance level (currently alpha=0.01). Implication: a large between-theory gap is NOT enough — if either theory's metric is also highly variable across subjects, N humans won't reliably distinguish them and the metric will be rejected. Aim for contrasts that are both large in mean AND tight per subject.

Do NOT propose metrics that are trivially true for your theory.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

Each subject completes ~96 trials in a single block, with order randomized independently per subject. On every trial the subject sees two options A and B, each described by `n_features` binary expert ratings (each 0 or 1). The per-feature validities and n_features are fixed per experiment (design-time choices). Validities are communicated to the subject in the instructions. Both `n_features` and `validities` are exposed to your `predict` via the `parameters` dict. The subject chooses A or B; no correctness feedback is provided after the choice.

## CHOSEN EXPERIMENTAL DESIGN
**Validities (n_features=5):** [1.0, 0.5, 0.7, 0.8, 0.9]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 5: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 6: A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]

**Rationale:** This eight-pair design tests the advocated theory's condition-specific attenuation of endpoint distrust. In each base pair, the 100%-valid expert endorses A and exactly one opposing expert endorses B; the opponent's validity increases across 50%, 70%, 80%, and 90%. The categorical tally is always tied 1-1. Contextual Anti-Reliability Arbitration predicts a qualitative crossover. Against the 50% cue, the endpoint-versus-neutral exemption preserves the 100% expert's full ordinary and endpoint-specific distrust, producing a strong preference for B. A 70% opponent should still leave the endpoint claim more distrusted, whereas the 80% condition should approach the transition. Against the substantive 90% opponent, attentional focusing attenuates the endpoint cue's total distrust enough that the 90% claim becomes more distrusted than the 100% claim; the model therefore predicts a preference for A. Thus, paradoxically, strengthening the expert opposed to the endpoint cue should move choices from rejecting A to favoring A, crossing chance between the low- and high-validity conditions. Finite-Pool Pragmatic Claim Competition cannot produce this within-person crossover. With one endorser per side, both claims receive the same attention because their pairwise competition loads are identical, so allocated claim contrast is attention times [1-(2v_B-1)], which remains positive whenever v_B<1. A skeptical construal consequently favors B in every condition, merely declining toward chance as the opponent approaches 100%; a weak conventional construal favors A throughout and also cannot switch signs. The critical planned contrast is therefore endpoint-endorsed choice below chance for the 50%/70% opponents but above chance for the 90% opponent, rather than only a monotonic confidence reduction. The last four pairs are exact A/B reversals, balancing response side while preserving this interaction. Eight unique pairs are each repeated 12 times for approximately 96 trials.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Contextual Anti-Reliability Arbitration theory proposes that communicated validity numbers are interpreted partly as pragmatic claims whose assertiveness can undermine their credibility. Every above-chance claim therefore generates continuous evidence against the option endorsed by that expert, rather than being categorically reversed by a special subgroup. The magnitude of this anti-reliability response varies continuously across people. Near-perfect claims additionally evoke endpoint distrust because unusually strong assurances are perceived as especially suspicious. When an endpoint cue faces a substantive opposing claim, both its ordinary and endpoint-specific distrust contributions are attenuated; against a genuinely neutral cue, both remain intact. Claims around 95% are more strongly compressed in the ordinary channel and do not receive the endpoint bonus, separating near-endpoint skepticism from distrust of literal certainty. Anti-reliability evidence is accumulated separately from the raw vote tally and divisively normalized by coalition breadth, within-side compositional heterogeneity, and sparse across-side claim mismatch. The latter detects heterogeneous singleton conflicts that within-side dispersion cannot detect, while exempting an isolated certainty claim opposed by a neutral cue. A small smooth boost preserves moderate-claim distrust in broad but not maximally dense coalitions. Its influence is strongest when the validity contrast is concentrated and the tally is tied or weak, while it decays smoothly as the absolute tally margin grows. Raw tally evidence additionally receives a continuously heterogeneous, intermediate-margin decision-temperature adjustment. The mechanism remains continuous, option-swap symmetric, and history independent because choices are followed by no accuracy feedback.

**Parameters:**
- anti_reliability_strength: [3.0, 5.6]
- anti_accumulation: [2.0, 3.5]
- endpoint_distrust: [1.10, 2.05]
- coalition_dilution_exponent: [0.46, 0.72]
- composition_dilution: [0.8, 2.4]
- across_composition_dilution: [1.0, 3.5]
- concentration_floor: [0.38, 0.62]
- margin_protection_scale: [1.55, 2.25]
- margin_protection_power: [2.2, 3.4]
- evidence_scale: [2.2, 3.0]
- choice_sensitivity: [1.45, 2.65]
- margin_temperature_gain: [0.0, 0.32]
- lapse_rate: [0.0, 0.10]
- validities: validities

**`predict source code`:**
```python
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
    # is strongly compressed near 95%, preventing near-endpoint claims from
    # behaving like literal-certainty claims.
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
    ordinary_claim = ordinary_base * (1.0 - 0.55 * upper_compression)

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

    if active_count == 0:
        anti_reliability = 0.0
    else:
        active_signs = signed_votes[active]
        active_ordinary = ordinary_claim[active]
        active_endpoint = endpoint_curve[active]
        active_above_chance = above_chance[active]

        # Endpoint distrust is preserved against a neutral 50% opponent, but
        # attenuated by the smooth maximum ordinary strength on the opposing
        # side. The attenuation applies to the endpoint cue's total distrust,
        # not merely to its endpoint bonus.
        nonendpoint_ordinary = active_ordinary * (1.0 - active_endpoint)
        endpoint_focus = np.ones(active_count, dtype=np.float64)
        for i in range(active_count):
            opposing = nonendpoint_ordinary[active_signs != active_signs[i]]
            if opposing.size > 0:
                attention_logits = 6.0 * opposing
                attention_logits -= np.max(attention_logits)
                attention = np.exp(attention_logits)
                attention /= attention.sum()
                opposing_peak = float(np.dot(attention, opposing))
                endpoint_focus[i] = 1.0 / (
                    1.0 + 3.0 * opposing_peak ** 1.4
                )

        focused_endpoint = active_endpoint * endpoint_focus
        active_claims = (
            active_ordinary * (1.0 - active_endpoint)
            + (
                active_ordinary
                + float(parameters["endpoint_distrust"])
            ) * focused_endpoint
        )
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

        # Within-side dispersion is exactly zero when each option has only one
        # active endorser. Across-active dispersion therefore supplies a second,
        # smoothly sparse-weighted composition diagnostic. It is switched off
        # for a literal endpoint claim against a neutral claim, preserving the
        # isolated 100%-versus-50% distrust response.
        across_dispersion = float(np.mean((active_claims - np.mean(active_claims)) ** 2))
        sparse_weight = (2.0 / float(active_count)) ** 2
        endpoint_neutral_exemption = float(np.max(active_endpoint)) * (
            1.0 - float(np.max(nonendpoint_ordinary))
        )
        across_gate = 1.0 - np.clip(endpoint_neutral_exemption, 0.0, 1.0)
        across_normalizer = 1.0 + (
            float(parameters["across_composition_dilution"])
            * sparse_weight
            * across_gate
            * across_dispersion
        )

        contextual_contrast = (
            signed_claim_contrast * focus
            / (
                breadth_normalizer
                * heterogeneity_normalizer
                * across_normalizer
                + 1e-12
            )
        )

        # Moderate pragmatic claims receive a small accumulation boost in broad
        # but not maximally dense coalitions. The smooth validity and breadth
        # gates avoid strengthening sparse 95%-claim conflicts or dense diffuse
        # conflicts indiscriminately.
        strongest_assertiveness = float(np.max(active_above_chance))
        moderate_gate = logistic(12.0 * (0.72 - strongest_assertiveness))
        breadth_peak = (
            (1.0 - np.exp(-max(active_count - 2, 0) / 2.0))
            * np.exp(-((active_count - 7.0) / 4.0) ** 2)
        )
        contextual_contrast *= float(
            1.0 + 0.15 * moderate_gate * breadth_peak
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
    tally_evidence = 4.5 * np.tanh(tally_margin / 4.5)
    total_evidence = tally_evidence + anti_reliability

    evidence_scale = float(parameters["evidence_scale"])
    normalized_evidence = float(np.tanh(total_evidence / evidence_scale))
    decision_variable = float(parameters["choice_sensitivity"]) * normalized_evidence

    # Confidence separates most strongly at intermediate margins. The bounded
    # gain is zero through margin one and decreases again for very large margins,
    # increasing heterogeneous margin sensitivity without indiscriminately
    # sharpening every decisive coalition.
    margin_excess = max(abs(tally_margin) - 1.0, 0.0)
    intermediate_margin_shape = (
        0.5 * margin_excess * np.exp(1.0 - margin_excess / 2.0)
        if margin_excess > 0.0 else 0.0
    )
    margin_temperature = 1.0 + float(parameters["margin_temperature_gain"]) * intermediate_margin_shape
    decision_variable *= float(margin_temperature)

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
```

**`policy source code`:**
```python
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
```


## COMPETING THEORY
**Description:** Finite-Pool Pragmatic Claim Competition theory proposes that binary expert endorsements generate two separable signals. A categorical vote channel counts every discriminating expert equally and therefore represents coalition consensus independently of communicated validity. A pragmatic claim channel interprets an expert's stated validity as the assertiveness of the expert's claim. Claim assertiveness is encoded by a smooth linear function of the communicated scale, including from 90% through 100%. Most people respond skeptically to assertive claims, although the polarity and magnitude of this interpretation vary continuously across individuals and can include weakly conventional interpretations. Claim processing draws on a finite attentional pool. Every discriminating expert, including a nominally 50%-valid expert whose own claim carries no directional strength, competes for that pool. Competition increases both with active-set size and with similarity among simultaneously active claims. Consequently, an isolated high-validity claim can receive substantial attention, whereas its influence declines when embedded in a broad or compositionally redundant coalition. This single allocation rule produces breadth and composition effects without endpoint bonuses, validity notches, or condition-specific exemptions. Pragmatic claim evidence is bounded, while categorical tally evidence continues to increase with vote margin. Skepticism can thus reverse tied or weak tallies but normally yields to a decisive raw-vote coalition. In the absence of outcome feedback, neither signal is recalibrated from choice history.

**Parameters:**
- claim_construal: [-0.1, 3.6]
- claim_accumulation: [1.3, 2.0]
- attentional_capacity: [0.9, 1.6]
- competition_floor: [0.30, 0.52]
- similarity_width: [0.22, 0.38]
- tally_sensitivity: [0.55, 0.95]
- tally_saturation: [4.0, 5.8]
- choice_sensitivity: [0.68, 1.18]
- lapse_rate: [0.0, 0.10]
- validities: validities

**`predict source code`:**
```python
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
```

**`policy source code`:**
```python
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
```


## DATA SCHEMA
Your metric receives a tidy per-trial pandas DataFrame stacking all subjects (rows grouped by `subject_id`, in trial order). Columns:
- subject_id: Subject identifier (one row per trial per subject).
- option_a_ratings: List of n_features binary expert ratings (each 0 or 1) for option A on this trial.
- option_b_ratings: List of n_features binary expert ratings (each 0 or 1) for option B on this trial.
- response: 0 if subject chose A, 1 if subject chose B.

## IMPLEMENTATION GUARDRAILS
Any column in the schema above whose description names a list / tuple / np.ndarray (i.e. a per-trial sequence of values) holds non-scalar cells. Those cells are NOT hashable, so operations that hash row values fail with `TypeError: unhashable type: 'list'`. Treating `<seq_col>` as a placeholder for any such sequence-valued column:
- Avoid: `data.groupby('<seq_col>')`, `data['<seq_col>'].value_counts()`,     `data['<seq_col>'].nunique()`, `data['<seq_col>'].unique()` (returns     an object array but downstream `set()` / `in dict` will crash),     `set(data['<seq_col>'])`, `data['<seq_col>'].isin([...])` against list     values, or using a list cell as a dict key.
- If you need a hashable surrogate, project to one first, e.g.:
    - `data['<seq_col>_key'] = data['<seq_col>'].apply(tuple)` then group by `<seq_col>_key`
    - `data['<seq_col>_str'] = data['<seq_col>'].apply(lambda x: ''.join(map(str, x)))`
    Scalar columns (ints, floats, strings like `subject_id`, integer     responses, etc.) hash fine and can be used directly.
- Generator expressions inside function calls like `map()` or `join()` MUST be     parenthesized. For example:
    - WRONG: `map(str, int(v) for v in x)` → SyntaxError
    - RIGHT: `map(str, (int(v) for v in x))` or use a list comp: `[str(int(v)) for v in x]`
- Always verify your code is syntactically valid Python before returning it.

## METRICS YOU ALREADY TRIED AND FAILED ON
Each entry below is a metric you previously proposed in this round that did NOT discriminate the two theories at the human sample size — either it errored, its between-subject variance was unavailable, or Welch's t-test on `(self mean, self var, N)` vs. `(adv mean, adv var, N)` returned p ≥ alpha. The `outcome` line is the simulation result (means, between-subject variances, t-statistic and p-value at the human N) on the same `data_self` / `data_adv` your next metric will be evaluated on. Use the numbers to see where your hypothesised contrast collapsed — small mean gap, large per-subject variance, or both — and propose something qualitatively different. Don't repeat the same idea with cosmetic tweaks.
(none yet)

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    low_endpoint_choices = []\n    high_endpoint_choices = []\n\n    for _, row in data.iterrows():\n        a = np.asarray(row[\"option_a_ratings\"], dtype=float)\n        b = np.asarray(row[\"option_b_ratings\"], dtype=float)\n        if a.size < 5 or b.size != a.size:\n            continue\n\n        diff = a - b\n        active = np.flatnonzero(diff != 0)\n        if active.size != 2 or 0 not in active:\n            continue\n\n        opponents = active[active != 0]\n        if opponents.size != 1:\n            continue\n        opponent = int(opponents[0])\n        if diff[0] * diff[opponent] >= 0:\n            continue\n\n        response = int(row[\"response\"])\n        endpoint_chosen = (\n            (diff[0] > 0 and response == 0)\n            or (diff[0] < 0 and response == 1)\n        )\n\n        if opponent in (1, 2):\n            low_endpoint_choices.append(float(endpoint_chosen))\n        elif opponent == 4:\n            high_endpoint_choices.append(float(endpoint_chosen))\n\n    if len(low_endpoint_choices) == 0 or len(high_endpoint_choices) == 0:\n        return float(\"nan\")\n\n    low_rate = float(np.mean(low_endpoint_choices))\n    high_rate = float(np.mean(high_endpoint_choices))\n\n    return float((low_rate < 0.5) and (high_rate > 0.5))\n",
  "rationale": "This is a qualitative crossover-phenotype indicator. It equals one only when the certainty-endorsed option is rejected below chance against the pooled 50%/70% opponents but selected above chance against the 90% opponent. Choices are recoded relative to the endpoint endorser, so the exact A/B reversals remove response-side effects. Contextual Anti-Reliability Arbitration predicts both parts of this conjunction: intact endpoint distrust in the low-validity contexts and attenuated endpoint distrust against the substantive 90% claim. Finite-Pool Pragmatic Claim Competition predicts increasing endpoint-choice rates as the opponent approaches certainty, but its skeptical claim contrast remains directed against the endpoint option at 90%, so it should fail the above-chance component. The binary sign criterion also deemphasizes heterogeneous effect magnitudes and is bounded in per-subject variance."
}
```

## Usage

```json
{
  "input_tokens": 6842,
  "output_tokens": 3893
}
```
