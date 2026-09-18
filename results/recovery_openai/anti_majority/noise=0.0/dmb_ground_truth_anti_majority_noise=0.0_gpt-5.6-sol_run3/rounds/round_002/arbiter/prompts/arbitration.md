# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_1" and "pi_4") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_1" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_4" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_1
People compare two options by consulting cues one at a time in order of validity, stopping at the first cue that discriminates between the two options. That cue alone determines the choice: the option with the higher value on the discriminating cue wins, and no other cue is consulted. Cues with lower validity are never reached when a higher-validity cue already discriminates, so Take The Best is a "one-reason" decision rule — only a single feature is ever used on any given choice. Because only the sign of the comparison on the top discriminating cue matters, TTB ignores both the magnitude of that difference and all information on lower-validity cues, making it maximally frugal in the use of evidence. Cue validities are subjective and can be learned, inferred from instructions, or fixed by task structure. When no cue discriminates (all feature-wise comparisons tie) the learner has no basis for preference and must guess. Response noise enters through a softmax over the binary TTB score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    # Paper-faithful Take The Best (Gigerenzer & Goldstein 1996).
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B. Cue cascade: features are consulted in order
    # of descending validity; the first discriminating cue (strict
    # inequality) determines the winner; if no cue discriminates,
    # the model guesses uniformly. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )
    # Descending validity; argsort is stable so validity ties break
    # toward the earlier feature index.
    cue_order = np.argsort(-val, kind="stable").tolist()

    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    if winner is None:
        # No discriminating cue — pure guess.
        return np.ones(2) / 2.0

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. For the
    # binary TTB score this collapses to sigmoid(beta) for the winner,
    # giving a direct mapping from beta onto the paper's flip-noise
    # levels (beta=0 ↔ 50/50; beta ≫ 1 ↔ deterministic).
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


## THEORY 2 — pi_4
Dominance-Calibrated Saturating Coalitions (DCSC) proposes that people classify the communicated validity profile before evaluating individual options. A cue activates a dominance regime when it accounts for a sufficiently large share of total diagnosticity and the profile as a whole is concentrated rather than broadly distributed across similarly informative cues. In that regime, the highest-validity discriminating cue anchors the decision, while the remaining cues provide only a small, order-invariant adjustment. Without sufficient profile-level dominance, people enter a compensatory regime and accumulate signed reliability evidence from every discriminating cue. Reliability differences are compressed, and repeated support for the same option has diminishing marginal impact, representing perceived redundancy among concordant experts. Thus, several lower-validity cues can collectively defeat the best discriminating cue, but neither early nor late serial position is intrinsically favored. Evidential balance can induce additional deliberation, modeled as a modest increase in choice consistency rather than a directional recency rule. Stable individual differences concern the dominance threshold, sensitivity to profile concentration, compression and saturation of reliability evidence, leakage of coalition evidence into dominance decisions, deliberation under conflict, and response sensitivity.

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
    profile_shape_weight = float(parameters["profile_shape_weight"])
    validity_compression = float(parameters["validity_compression"])
    redundancy_saturation = float(parameters["redundancy_saturation"])
    anchor_strength = float(parameters["anchor_strength"])
    dominance_leak = float(parameters["dominance_leak"])
    conflict_deliberation = float(parameters["conflict_deliberation"])
    beta = float(parameters["beta"])

    # Communicated validities are converted to nonnegative diagnostic
    # strengths. The small exponent compresses differences in the
    # compensatory regime without reversing the communicated ordering.
    diagnosticity = np.clip(2.0 * validities - 1.0, 1e-6, 1.0)
    cue_weights = diagnosticity ** validity_compression

    # Dominance depends both on the strongest cue's diagnosticity share
    # and on concentration across the complete profile. The normalized
    # Herfindahl term distinguishes a genuinely concentrated profile from
    # one in which the strongest cue has a similar share but substantial
    # diagnosticity remains broadly distributed among lower cues.
    profile_shares = diagnosticity / max(float(np.sum(diagnosticity)), 1e-12)
    dominance_concentration = float(np.max(profile_shares))
    profile_hhi = float(np.sum(profile_shares ** 2))
    uniform_hhi = 1.0 / float(n_features)
    shape_concentration = (profile_hhi - uniform_hhi) / max(
        1.0 - uniform_hhi, 1e-12
    )
    shape_concentration = float(np.clip(shape_concentration, 0.0, 1.0))
    dominance_score = (
        dominance_concentration
        + profile_shape_weight * shape_concentration
    )
    gate_logit = gap_sensitivity * (
        dominance_score - dominance_threshold
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
    # anchored judgment. In integration mode it is fully compensatory.
    dominance_evidence = anchor_evidence + dominance_leak * coalition_evidence
    choice_evidence = (
        dominance_gate * dominance_evidence
        + (1.0 - dominance_gate) * coalition_evidence
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

## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
**Validities (n_features=8):** [0.9, 0.9, 0.82, 0.82, 0.75, 0.75, 0.65, 0.65]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 1, 0]
  trial 3: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  trial 4: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0]
  trial 6: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  trial 7: A=[0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[0, 1, 0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  trial 10: A=[0, 1, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  trial 12: A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** The validity profile contains four validity-matched cue pairs but remains broad enough to force DCSC into its compensatory regime. Using the most dominance-favoring allowed profile-shape weight, the profile's dominance score is approximately 0.212, still below the minimum dominance threshold of 0.285; hence the dominance gate is at most about 0.006. On every trial, the two highest-validity experts discriminate in opposite directions. Because their validities are tied, Take The Best uses its stable ordering rule and follows Expert 1, predicting the same probability q = epsilon/2 + (1-epsilon)*sigmoid(beta) for Expert 1's option on every trial. DCSC instead averages the opposing tied anchors, making anchor evidence exactly zero. One or more lower-validity experts then unanimously support Expert 2's option, so DCSC's coalition evidence always points against the TTB winner and predicts selection of the TTB winner below chance. Nested conditions increase the size of this opposing coalition from one to six lower experts, providing a graded DCSC signature while TTB remains exactly invariant. The two single-supporter conditions also contrast a relatively reliable lower expert (82%) with a weaker one (65%), testing DCSC's compressed reliability weighting without affecting TTB. Each pattern has an exact A/B reversal, controlling response-side preference. The 12 unique pairs are repeated eight times, yielding 96 trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
This is response-side-balanced adherence to Expert 1, the stable-order winner among the two equally valid highest-validity experts. It is computed identically for pooled data and individual subjects. TTB predicts consistently above-chance adherence because Expert 1 is the first discriminating cue on every trial, irrespective of the number or reliability of lower cues. DCSC instead cancels the opposing tied top anchors and lets the unanimous lower-cue coalition determine the choice, predicting below-chance Expert-1 adherence. Pooling all 96 trials yields a low-noise subject-level estimate, while the exact A/B reversals prevent a response-side bias from masquerading as cue adherence.

Source:
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

### RESULTS
- Predicted under pi_1 (simulated): 0.8244 (var=0.0081)
- Predicted under pi_4 (simulated): 0.0179 (var=0.0002)
- Observed on real data: 0.7900 (var=0.0184)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=8):** [0.85, 0.82, 0.79, 0.76, 0.73, 0.7, 0.67, 0.6]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  trial 4: A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0]
  trial 6: A=[0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  trial 8: A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0]
  trial 10: A=[0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  trial 12: A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** The broad validity profile forces DCSC into its compensatory regime while preserving a unique highest-validity cue. With diagnosticities [0.70, 0.64, 0.58, 0.52, 0.46, 0.40, 0.34, 0.20], the maximum DCSC dominance score is approximately 0.215, below the minimum dominance threshold of 0.285; even at the least decisive allowed gap sensitivity, the dominance gate is below 0.008. In every underlying pattern, Expert 1 discriminates for a designated top-cue winner, so Take The Best always selects that option with the same probability q = epsilon/2 + (1-epsilon)*sigmoid(beta), regardless of all lower cues. DCSC predicts a distinctive crossover. When only one lower expert opposes Expert 1, the top-cue winner remains weakly favored because its reliability weight is larger. This preference should be stronger when the lone opponent is the weakest expert (Expert 8) rather than Expert 2, directly testing compressed reliability weighting. Once two or more lower experts form the opposing coalition, DCSC predicts the coalition's option above chance, with increasingly strong but diminishing support as the coalition grows from two to three, five, and seven experts. Thus TTB predicts complete invariance, whereas DCSC predicts both a directional crossover between one- and multi-opponent trials and a nonlinear coalition-size gradient. Unlike the previously explored tied-anchor design, the highest-validity anchor is unique here and DCSC initially agrees with TTB before lower-cue accumulation overturns it. Exact A/B reversals control response-side preferences. The 12 unique pairs are repeated eight times, yielding 96 trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
This is a within-subject crossover contrast. It measures how much more often the highest-validity cue's winner is selected when it faces exactly one opposing lower cue than when it faces a coalition of two or more lower cues. Take The Best predicts the same top-cue adherence probability in both conditions for every beta and epsilon, so its expected contrast is zero; differencing also cancels stable response sensitivity and lapse variation. DCSC predicts a large positive contrast: the top-cue option remains near or above chance against one opponent, whereas compensatory accumulation makes the opposing option strongly preferred for multi-cue coalitions. Pooling all 32 one-opponent and 64 multi-opponent observations yields low sampling variance, while coding choices relative to the top-cue winner makes the exact A/B reversals cancel response-side preferences.

Source:
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

### RESULTS
- Predicted under pi_1 (simulated): 0.0103 (var=0.0049)
- Predicted under pi_4 (simulated): 0.5559 (var=0.0104)
- Observed on real data: -0.5450 (var=0.0462)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
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
**Predicted under pi_1:** 0.9900 (var=0.0049)
**Predicted under pi_4:** 1.0000 (var=0.0000)

### Experiment 4
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
**Predicted under pi_1:** 0.8492 (var=0.0121)
**Predicted under pi_4:** 0.9925 (var=0.0001)

### Experiment 5
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
**Predicted under pi_1:** -0.0008 (var=0.0015)
**Predicted under pi_4:** 0.0050 (var=0.0001)

### Experiment 6
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
**Predicted under pi_1:** 0.0400 (var=0.0384)
**Predicted under pi_4:** 0.0000 (var=0.0000)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all six experiments, pi_1 is the better surviving benchmark, but neither theory is task-invariant. Experiment 1 strongly favors pi_1: the observed Expert-1 adherence of 0.790 is close to pi_1's 0.824 and diametrically opposed to pi_4's 0.018. The higher observed subject variance (0.0184 versus 0.0081) suggests heterogeneous implementation, but it does not weaken the directional result. Experiment 2 falsifies pi_4 even more sharply: pi_4 predicts a large positive crossover (+0.556), whereas the observed crossover is comparably large and negative (-0.545). Pi_1's near-zero prediction is also wrong, but is substantially closer and does not predict the reversed direction. Experiment 4 is a second major failure for both theories: observed adherence to the highest-validity discriminating cue is below chance (0.324), compared with 0.849 for pi_1 and 0.993 for pi_4. Thus the data cannot be explained merely by changing response noise around either theory's preferred option. Experiments 3 and 6 are fitted well by both theories and therefore offer little discrimination; Experiment 5 shows a small positive effect missed by both. Overall, pi_1 wins comparatively because it captures Experiments 1 and 3 and is less wrong on Experiments 2 and 4. Nevertheless, its defining invariance is contradicted by the negative coalition-size crossover in Experiment 2 and systematic below-chance alignment in Experiment 4. Pi_4 is more decisively degenerate: its central claims of order-invariant coalition accumulation and coalition-driven reversal predict the wrong winner in Experiment 1 and the exact wrong crossover direction in Experiment 2, while its near-deterministic prediction in Experiment 4 is also opposite the data. These failures concern qualitative signs, not parameter calibration, so regenerating pi_4's current model would not suffice without violating its prose commitments.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace pi_4 with a new task-adaptive distinctiveness-and-redundancy theory, retaining pi_1 as the one-reason benchmark. The new theory should posit that subjects construct effective cue weights jointly from communicated validity, ordinal/presentation prominence, and the cue's distribution across the task. Isolated dissenting cues can receive a distinctiveness bonus, whereas large concordant coalitions are treated as redundant or suspiciously correlated and can be strongly discounted rather than monotonically accumulated. A conflict-dependent strategy gate should allow lexicographic anchoring when a cue is prominent or when top validities are tied, but distinctiveness-based selection when sparse opposing evidence stands out; importantly, this gate must be driven by observable profile and stimulus statistics rather than experiment-specific labels. This can accommodate strong first-cue adherence in Experiment 1, the negative crossover in Experiment 2 if lone dissent is more salient than a redundant multi-cue coalition, and below-chance highest-validity adherence in Experiment 4 when another cue or cue configuration is more distinctive. Include stable subject-level variation in attention to communicated validity, order prominence, distinctiveness, and redundancy to address the elevated observed variances. The new theory must make signed, trial-level predictions for Experiments 1, 2, and 4 simultaneously and preserve the near-null interaction in Experiment 5 and null prevalence result in Experiment 6; it should not merely add unrestricted experiment-specific mixtures or lapse rates."
}
```

## Usage

```json
{
  "input_tokens": 10750,
  "output_tokens": 1906
}
```
