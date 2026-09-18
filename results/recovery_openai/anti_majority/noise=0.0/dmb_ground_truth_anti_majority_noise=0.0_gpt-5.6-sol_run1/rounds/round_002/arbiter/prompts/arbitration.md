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
Noisy Priority Race Search proposes that communicated cue validities establish persistent but imperfect subjective cue priorities rather than a universally fixed ordering. Each subject transforms validity log-odds into priority, adds a stable cue-specific representation bias, and has a serial-position prior favoring earlier displayed cues. On each trial, independent retrieval fluctuations perturb these priorities. Cues are inspected in the resulting stochastic order, tied cues are skipped, and search stops at the first discriminating cue. Only that cue determines the intended response; evidence from uninspected or lower-priority cues is never integrated. Small validity gaps can therefore produce stable individual departures and trial-level reversals of the stated ordering, whereas large validity gaps produce nearly deterministic lexicographic search. Response execution is separately subject to temperature-controlled errors and occasional lapses.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Noisy Priority Race Search expects shape (2, n_features); got {stim.shape}."
        )

    n_features = int(parameters["n_features"])
    if stim.shape[1] != n_features:
        raise ValueError(
            f"Stimulus has {stim.shape[1]} features but n_features={n_features}."
        )

    validities = np.asarray(parameters["validities"], dtype=np.float64)
    persistent_bias = np.asarray(parameters["priority_bias"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities must have length {n_features}; got {validities.shape}."
        )
    if persistent_bias.ndim != 1 or persistent_bias.size != n_features:
        raise ValueError(
            f"priority_bias must have length {n_features}; got {persistent_bias.shape}."
        )

    validity_sensitivity = float(parameters["validity_sensitivity"])
    serial_priority = float(parameters["serial_priority"])
    retrieval_noise = float(parameters["retrieval_noise"])
    beta = float(parameters["response_temperature"])
    lapse_rate = float(parameters["lapse_rate"])

    # Persistent subjective priority representation. Validity log odds encode
    # communicated diagnosticity. The serial term supplies a weak display-order
    # prior and resolves equal-validity cues without assuming exact equality in
    # their subjective representations.
    v = np.clip(validities, 0.5 + 1e-9, 1.0 - 1e-9)
    validity_signal = np.log(v / (1.0 - v))
    serial_signal = (n_features - 1) - np.arange(n_features, dtype=np.float64)
    priority = (
        validity_sensitivity * validity_signal
        + serial_priority * serial_signal
        + persistent_bias
    )

    differences = stim[0] - stim[1]
    discriminating = np.flatnonzero(differences != 0.0)
    if discriminating.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    # If retrieval shocks are iid Gumbel, the probability that a particular
    # discriminating cue is reached first is a softmax over its priority. Cues
    # that tie on the current trial may be inspected but cannot terminate
    # search, so they do not enter the race among stopping cues.
    race_logits = priority[discriminating] / max(retrieval_noise, 1e-12)
    race_logits = race_logits - np.max(race_logits)
    first_prob = np.exp(race_logits)
    first_prob /= first_prob.sum()

    # Marginalize over which single cue wins the priority race. This is not
    # evidence integration: in every latent search realization exactly one
    # first discriminating cue determines the intended choice.
    favors_a = differences[discriminating] > 0.0
    intended_a = float(np.sum(first_prob[favors_a]))

    # Conditional on a stopping cue, response temperature permits execution
    # errors of equal size in either direction.
    if beta >= 0.0:
        follow_probability = 1.0 / (1.0 + np.exp(-beta))
    else:
        exp_beta = np.exp(beta)
        follow_probability = exp_beta / (1.0 + exp_beta)

    core_a = (
        intended_a * follow_probability
        + (1.0 - intended_a) * (1.0 - follow_probability)
    )
    p_a = (1.0 - lapse_rate) * core_a + lapse_rate * 0.5
    p_a = float(np.clip(p_a, 0.0, 1.0))

    probs = np.array([p_a, 1.0 - p_a], dtype=np.float64)
    probs /= probs.sum()
    return probs

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
**Validities (n_features=9):** [0.95, 0.55, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.72]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 1, 1, 1, 1, 1, 0, 1]  B=[1, 0, 1, 1, 1, 1, 1, 1, 0]
  trial 2: A=[1, 0, 1, 1, 1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 1, 1, 1, 1, 0]
  trial 3: A=[1, 0, 1, 1, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 1, 1, 1, 0]
  trial 4: A=[1, 0, 1, 1, 0, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 1, 1, 1, 0]
  trial 5: A=[1, 0, 1, 0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 1, 1, 1, 0]
  trial 6: A=[1, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 1, 1, 1, 0]
  trial 7: A=[1, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 0, 1, 1, 1, 1, 1, 0, 1]
  trial 8: A=[1, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 0, 1, 1, 1, 1, 0, 0, 1]
  trial 9: A=[1, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 0, 1, 1, 1, 0, 0, 0, 1]
  trial 10: A=[1, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 0, 1, 1, 0, 0, 0, 0, 1]
  trial 11: A=[1, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 0, 1, 0, 0, 0, 0, 0, 1]
  trial 12: A=[1, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 1]

**Rationale:** The six base pairs create a cumulative priority-race manipulation and are followed by exact A/B reversals, yielding 12 unique pairs repeated eight times for 96 trials. Expert 1 (95%) and Expert 2 (55%) always tie. The last-displayed expert has 72% validity and favors A in every base pair, whereas one through six earlier-displayed 70% experts favor B. Take The Best sorts cues strictly by advertised validity, skips the tied 95% cue, and always stops at the 72% cue. It therefore predicts the same above-chance probability q=(1-epsilon)*sigmoid(beta)+epsilon/2 of choosing A at every base conflict level, irrespective of how many 70% cues oppose it. Noisy Priority Race Search instead gives the earlier 70% cues a serial-position advantage. Conflict level 1 activates only the 70% cue immediately before the 72% cue; each subsequent level adds the next-earlier 70% cue. Adding a B-favoring competitor strictly increases the race probability that B supplies the stopping cue, so the competing theory predicts a monotonic decline in A choices across levels. The highest conflict level also provides a parameter-robust directional reversal: for the earliest active 70% cue, its minimum priority advantage over the 72% cue is 6(0.15)-5[logit(0.72)-logit(0.70)]-0.16, approximately 0.254. Even at retrieval noise 0.08, that cue alone has over 23 times the race weight of the 72% cue, and all other discriminating race cues also favor B. Thus the competing theory predicts a strong B preference at level 6, while TTB must continue to prefer A. Exact option reversals balance response-side biases and reverse directions without changing the predicted conflict-level profile.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
This is the strong-conflict majority-override rate, pooled across conflict levels 5 and 6 and their exact A/B reversals. It measures a psychologically interpretable event: choosing the option supported by a large coalition of slightly less valid cues even though the single more-valid 72% cue recommends the other option. TTB should produce a low rate because it stops at the 72% cue and never consults the 70% cues. Priority Race Search should produce a high rate because, at these conflict levels, at least one earlier-displayed 70% cue has a parameter-robust priority advantage and normally wins the stopping-cue race. Restricting the contrast to the two strongest levels preserves that robust separation, while combining 32 trials per subject and balancing choice side through exact reversals reduces sampling variance and response-side bias.

Source:
def metric(data: pd.DataFrame) -> float:
    """Rate at which the 70%-cue majority overrides the opposing 72% cue
    on the two strongest conflict levels (five or six active 70% cues)."""
    matches = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.size < 9 or b.size < 9:
            continue

        # Features 3--8 (zero-based indices 2:8) are the six 70% cues.
        d70 = a[2:8] - b[2:8]
        active = d70[d70 != 0]
        if active.size < 5:
            continue

        # Retain only designed conflicts in which the active 70% cues agree
        # and the 72% cue recommends the opposite option.
        direction = np.sign(active[0])
        if not np.all(np.sign(active) == direction):
            continue
        d72 = a[8] - b[8]
        if d72 == 0 or np.sign(d72) == direction:
            continue

        majority_choice = 0 if direction > 0 else 1
        response = int(row["response"])
        if response in (0, 1):
            matches.append(float(response == majority_choice))

    if len(matches) == 0:
        return 0.0
    return float(np.mean(matches))

### RESULTS
- Predicted under pi_1 (simulated): 0.1450 (var=0.0101)
- Predicted under pi_4 (simulated): 0.8588 (var=0.0039)
- Observed on real data: 0.1613 (var=0.0081)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=18):** [0.55, 0.95, 0.5525, 0.55, 0.5, 0.5868, 0.55, 0.95, 0.6343, 0.55, 0.5, 0.6901, 0.55, 0.95, 0.7504, 0.55, 0.5, 0.8177]

**Trial pairs (n=12):**
  trial 1: A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  trial 9: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  trial 11: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

**Rationale:** The six base pairs isolate conflicts between two discriminating experts separated by exactly two display positions; all other experts tie and therefore cannot terminate either search rule. In every base pair, the earlier expert has validity 0.55 and favors B, whereas the later expert has slightly to substantially higher validity and favors A. The later-minus-earlier validity-log-odds gaps are approximately 0.01, 0.15, 0.35, 0.60, 0.90, and 1.30. Take The Best always ranks the later expert above the 0.55 expert, so it predicts the same A-choice probability q=(1-epsilon)*sigmoid(beta)+epsilon/2 at all six levels, regardless of the size of the validity gap. Noisy Priority Race Search instead predicts a validity-gap-dependent crossover because the earlier B-favoring expert receives a constant two-position serial advantage. Its later-minus-earlier priority difference is sensitivity*D - 2*serial_priority + (bias_later-bias_earlier). At D≈0.01 this difference is at most 5(0.01)-2(0.15)+0.16=-0.09 across the entire parameter range, so the advocated theory robustly predicts that the nominally less-valid, earlier expert wins the race and B is chosen more often. At D≈1.30 the difference is at least 1(1.30)-2(0.45)-0.16=0.24, so it robustly predicts that the more-valid expert wins and A is chosen more often. Intermediate gaps locate each subject's stochastic priority crossover and should produce a rising psychometric profile, potentially with stable cue-specific deviations from perfect monotonicity. Thus the critical signature is both a directional reversal at the smallest gap and a quantitative increase in following the higher-validity expert as the log-odds gap grows; TTB can only fit one condition-invariant response probability. The intervening tied experts preserve the same two-position separation, while tied 95% and 50% experts provide a broad advertised-validity spread without affecting choices. Each base pair has an exact A/B reversal, balancing response-side preferences. The 12 unique pairs are repeated eight times, yielding exactly 96 trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
This endpoint crossover contrast subtracts the probability of following the higher-validity cue at the smallest validity-log-odds gap from that probability at the largest gap. It pools each trial with its exact A/B reversal, so response-side preferences are balanced, and it uses 16 observations at each endpoint per subject. Take The Best predicts the same probability of following the higher-validity cue at both endpoints, making the expected contrast zero even when its response-noise parameters vary across subjects. Noisy Priority Race Search predicts a large positive contrast: the two-position serial advantage makes the earlier, slightly less valid cue likely to win at the smallest gap, whereas the large validity advantage dominates serial priority and representation bias at the largest gap. Restricting the statistic to the two parameter-robust endpoints avoids the additional between-subject crossover-location variance present in intermediate conditions.

Source:
def metric(data: pd.DataFrame) -> float:
    validities = np.array([
        0.55, 0.95, 0.5525, 0.55, 0.5, 0.5868,
        0.55, 0.95, 0.6343, 0.55, 0.5, 0.6901,
        0.55, 0.95, 0.7504, 0.55, 0.5, 0.8177
    ], dtype=float)
    log_odds = np.log(validities / (1.0 - validities))

    gaps = []
    follows_higher = []

    for a_cell, b_cell, response in zip(
        data['option_a_ratings'],
        data['option_b_ratings'],
        data['response']
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        diff = a - b
        discriminating = np.flatnonzero(diff != 0)
        if discriminating.size != 2:
            continue

        j0, j1 = int(discriminating[0]), int(discriminating[1])
        if validities[j0] == validities[j1]:
            continue
        high = j0 if validities[j0] > validities[j1] else j1
        low = j1 if high == j0 else j0

        high_favors_a = diff[high] > 0
        chose_a = int(response) == 0
        follows_higher.append(float(chose_a == high_favors_a))
        gaps.append(float(log_odds[high] - log_odds[low]))

    gaps = np.asarray(gaps, dtype=float)
    follows_higher = np.asarray(follows_higher, dtype=float)
    if gaps.size == 0:
        return float('nan')

    unique_gaps = np.unique(np.round(gaps, 10))
    if unique_gaps.size < 2:
        return float('nan')

    smallest = unique_gaps[0]
    largest = unique_gaps[-1]
    low_mask = np.isclose(gaps, smallest, atol=1e-9, rtol=0.0)
    high_mask = np.isclose(gaps, largest, atol=1e-9, rtol=0.0)

    return float(
        np.mean(follows_higher[high_mask])
        - np.mean(follows_higher[low_mask])
    )

### RESULTS
- Predicted under pi_1 (simulated): 0.0113 (var=0.0129)
- Predicted under pi_4 (simulated): 0.7075 (var=0.0151)
- Observed on real data: -0.1275 (var=0.0523)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1]
  A=[1, 0, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 1, 0, 1]  B=[1, 0, 0, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Classify whether choices are predominantly controlled by the highest-ranked
    # discriminating cue on trials where at least three net lower-cue votes point
    # in the opposite direction.
    agreements = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        d = a - b

        nz = np.flatnonzero(d != 0)
        if nz.size == 0:
            continue

        # Features are already ordered from highest to lowest validity.
        first = int(nz[0])
        ttb_sign = 1.0 if d[first] > 0 else -1.0  # +1 means A, -1 means B
        tally_margin = float(np.sum(np.sign(d)))  # positive means tally favors A

        # Retain only strong conflicts: tally favors the other option by at
        # least three net cue votes. This selects patterns 2, 3, 5, and 6 and
        # their A/B reversals in the specified design.
        if ttb_sign * tally_margin < 0 and abs(tally_margin) >= 3.0:
            ttb_response = 0 if ttb_sign > 0 else 1
            agreements.append(float(int(row["response"]) == ttb_response))

    if len(agreements) == 0:
        return 0.5

    # A categorical dominance index: 1 if the top discriminating cue controls
    # a strict majority of these high-conflict choices, otherwise 0.
    return float(np.mean(agreements) > 0.5)

```

**Observed (real) value:** 1.0000 (var=0.0000)
**Predicted under pi_1:** 1.0000 (var=0.0000)
**Predicted under pi_4:** 1.0000 (var=0.0000)

### Experiment 4
**Design**
  A=[1, 1, 0, 1, 0, 0, 1, 0]  B=[0, 0, 1, 0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 1, 0, 0]
  A=[1, 1, 0, 1, 0, 1, 0, 1]  B=[0, 0, 1, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 1, 0, 1, 1]  B=[0, 0, 0, 1, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 1, 0, 1]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 1, 1, 0, 0]  B=[1, 1, 0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 0, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 1, 0, 0]  B=[1, 1, 1, 0, 1, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 1, 1, 1, 0, 1, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """AIC model-selection indicator for graded tally-margin sensitivity."""
    if data is None or len(data) == 0:
        return 0.0

    margins = []
    successes = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        signed_margin = int(np.sum(a > b) - np.sum(b > a))
        if signed_margin == 0:
            continue

        winner = 0 if signed_margin > 0 else 1
        margins.append(abs(signed_margin))
        successes.append(float(int(row["response"]) == winner))

    if len(successes) == 0:
        return 0.0

    margins = np.asarray(margins, dtype=int)
    successes = np.asarray(successes, dtype=float)
    levels = np.arange(1, 7, dtype=float)
    n_by_margin = np.asarray([np.sum(margins == m) for m in range(1, 7)], dtype=float)
    k_by_margin = np.asarray(
        [np.sum(successes[margins == m]) for m in range(1, 7)], dtype=float
    )
    keep = n_by_margin > 0
    levels = levels[keep]
    n_by_margin = n_by_margin[keep]
    k_by_margin = k_by_margin[keep]

    if len(levels) < 2:
        return 0.0

    # Best margin-invariant winner probability: the observable implication
    # of TTB for this design. Its parameterization permits probabilities
    # arbitrarily close to the full interval [0.5, 1].
    n = float(np.sum(n_by_margin))
    k = float(np.sum(k_by_margin))
    q_const = float(np.clip(k / n, 0.5 + 1e-12, 1.0 - 1e-12))
    ll_ttb = k * np.log(q_const) + (n - k) * np.log1p(-q_const)

    # Maximize the exact Tallying likelihood over its stated beta and lapse
    # ranges. The grid is denser where sigmoid(beta*m) changes most rapidly.
    beta = np.concatenate(
        [np.linspace(0.1, 3.0, 241), np.linspace(3.1, 20.0, 170)]
    )
    epsilon = np.linspace(0.0, 0.5, 251)

    core = 1.0 / (1.0 + np.exp(-beta[:, None] * levels[None, :]))
    q = (
        (1.0 - epsilon[None, :, None]) * core[:, None, :]
        + epsilon[None, :, None] * 0.5
    )
    q = np.clip(q, 1e-12, 1.0 - 1e-12)

    ll = np.sum(
        k_by_margin[None, None, :] * np.log(q)
        + (n_by_margin - k_by_margin)[None, None, :] * np.log1p(-q),
        axis=2,
    )
    ll_tally = float(np.max(ll))

    # Tallying has one additional identifiable probability-shape parameter.
    # Thus AIC favors it exactly when the likelihood-ratio deviance exceeds 2.
    deviance = 2.0 * (ll_tally - ll_ttb)
    return float(deviance > 2.0)

```

**Observed (real) value:** 0.0000 (var=0.0000)
**Predicted under pi_1:** 0.0000 (var=0.0736)
**Predicted under pi_4:** 0.0000 (var=0.1056)

### Experiment 5
**Design**
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Return 1 when first-discriminating-cue consistency is at least 90%."""
    if len(data) == 0:
        return float("nan")

    follows_first_cue = []
    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"],
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        expected_response = None
        for j in range(min(a.size, b.size)):
            if a[j] > b[j]:
                expected_response = 0
                break
            if b[j] > a[j]:
                expected_response = 1
                break

        if expected_response is not None:
            follows_first_cue.append(int(int(response) == expected_response))

    if len(follows_first_cue) == 0:
        return float("nan")

    consistency = float(np.mean(follows_first_cue))
    return float(consistency >= 0.90)

```

**Observed (real) value:** 0.0000 (var=0.0000)
**Predicted under pi_1:** 0.0000 (var=0.2400)
**Predicted under pi_4:** 0.0000 (var=0.0564)

### Experiment 6
**Design**
  A=[1, 1, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 0, 1, 0, 1, 0]  B=[0, 0, 1, 1, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 1, 0, 1]  B=[1, 0, 1, 1, 1, 1, 0, 1]
  A=[0, 1, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 1, 0, 1]
  A=[0, 0, 1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 1, 1, 0, 1]  B=[1, 1, 0, 0, 0, 1, 0, 1]
  A=[0, 0, 1, 1, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0, 1, 0]
  A=[1, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Proportion of choices following the net direction of all weak cues on
    # trials where that direction opposes Expert 2. These are conflict levels
    # M=2,...,6; the M=1 cancellation trials are omitted.
    scores = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        directions = np.sign(a - b)

        # Feature 0 always ties. Features 1 onward are the equally valid weak
        # cues that the advocated theory integrates.
        weak_net = float(np.sum(directions[1:]))
        expert2_direction = float(directions[1])

        # Retain only genuine reversal trials: the weak-cue aggregate points
        # opposite to the first discriminating cue used by TTB.
        if weak_net == 0.0 or expert2_direction == 0.0:
            continue
        if np.sign(weak_net) == np.sign(expert2_direction):
            continue

        integrated_winner = 0 if weak_net > 0.0 else 1
        scores.append(float(int(row["response"]) == integrated_winner))

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** 0.1620 (var=0.0198)
**Predicted under pi_1:** 0.1670 (var=0.0108)
**Predicted under pi_4:** 0.1683 (var=0.0038)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all six experiments, pi_1 is the substantially better task-invariant account. In Experiment 1, the observed majority-override rate of 0.1613 is extremely close to pi_1's 0.1450 and radically inconsistent with pi_4's 0.8588. The observed between-subject variance (0.0081) is also close to pi_1's 0.0101, whereas pi_4 predicts not merely the wrong magnitude but the opposite qualitative regime. In Experiment 2, the observed endpoint contrast is -0.1275. Pi_1 predicts an approximately null contrast (0.0113), while pi_4 predicts a very large positive contrast (0.7075). Thus pi_1 misses the modest negative trend, but pi_4 misses by much more and gets its direction wrong. The relatively large observed variance in this experiment (0.0523) further suggests heterogeneous or noisy position/cue effects rather than pi_4's supposedly robust positive crossover. Experiments 3 and 4 match both theories at the pooled point-estimate level, although their binary metrics are coarse and provide little affirmative discrimination. Experiment 5 likewise gives both theories the correct pooled classification, with pi_4 closer in variance, but the thresholded metric cannot establish its stochastic-priority mechanism. Experiment 6 again closely matches pi_1: observed 0.1620 versus 0.1670, compared with 0.1683 for pi_4; both are close, but this result reinforces one-reason control and strongly rejects broad weak-cue integration. Overall, the evidence consistently supports instruction-anchored lexicographic choice and provides no support for pi_4's central claim that serial priority and validity-gap-dependent races routinely reorder discriminating cues. Pi_4 fails decisively on both experiments specifically constructed to expose that mechanism. Pi_1 should therefore be retained as the current benchmark, but its constant execution-noise formulation does not explain the negative endpoint contrast in Experiment 2. That residual should motivate a new theoretical competitor rather than merely retuning pi_4 or altering pi_1's parameter ranges.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace pi_4 with an Accessibility-Gated Lexicographic Search theory. Communicated validities establish a predominantly fixed descending cue order, as in instruction-anchored lexicographic choice, but successful use of a cue requires that its identity, validity, and current value be retrieved. Retrieval or encoding failure should increase with absolute display position, list length, and cue-specific complexity, with stable subject-level accessibility differences. Search proceeds down the advertised-validity ranking among cues that are successfully accessible; it stops at the first accessible discriminating cue, without integrating lower-cue votes. This differs from pi_1 by making cue availability\u2014and therefore errors\u2014systematically dependent on absolute position and cognitive accessibility rather than using only cue-independent response noise. It differs from pi_4 because validities do not enter a noisy priority race and small validity gaps do not intrinsically cause order reversals. The theory predicts low majority override in Experiment 1 because the 72% cue is normally retrieved, while occasional failure can produce the observed roughly 16% override. In Experiment 2, the higher-validity cue moves from an early absolute position at the smallest gap to the end of an 18-cue display at the largest gap; declining accessibility can therefore produce the observed slightly negative endpoint contrast rather than pi_4's large positive crossover. It should retain lexicographic dominance in Experiments 3 and 6, avoid graded tally-margin sensitivity in Experiment 4, and allow imperfect first-cue consistency in Experiment 5. The new model should include subject-level baseline retrieval ability, an absolute-position/list-length accessibility gradient, modest cue-specific accessibility variation, and separate execution lapses, with ranges constrained so that advertised validity normally dominates rather than permitting an unconstrained race that can mimic arbitrary cue orders."
}
```

## Usage

```json
{
  "input_tokens": 12196,
  "output_tokens": 1862
}
```
