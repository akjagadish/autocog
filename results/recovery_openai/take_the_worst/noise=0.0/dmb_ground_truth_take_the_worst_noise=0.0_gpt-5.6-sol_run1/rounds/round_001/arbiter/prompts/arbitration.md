# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_2") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_2" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3
Prospective Capacity-Gated Comparison theory: before choosing, people perceptually register how many cues discriminate between the options and compare that comparison load with a stable cognitive capacity. If the discriminating set is manageable, they integrate all discriminating cues with an equal-weight majority rule. If it exceeds capacity, they compress the comparison into a validity-ranked lexicographic search and rely on the first discriminating cue. Strategy recruitment changes smoothly near the capacity boundary, reflecting variability in momentary load, but is determined by an observable property of the option pair rather than by experiment identity. Ordinary response noise is applied only after a strategy has produced its intended choice.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Capacity-Gated Comparison expects shape (2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.shape[0] != n_features:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {n_features}."
        )

    differences = a - b
    discriminating = np.flatnonzero(differences != 0)
    load = float(discriminating.size)
    if load == 0:
        return np.array([0.5, 0.5], dtype=float)

    capacity = float(parameters["capacity"])
    gate_slope = float(parameters["gate_slope"])

    # Probability that exhaustive equal-weight integration is recruited.
    # This stable logistic gate approaches one below capacity and zero above it.
    gate_logit = gate_slope * (capacity - load)
    if gate_logit >= 0.0:
        exp_neg = np.exp(-gate_logit)
        p_integrate = 1.0 / (1.0 + exp_neg)
    else:
        exp_pos = np.exp(gate_logit)
        p_integrate = exp_pos / (1.0 + exp_pos)

    # Manageable-load strategy: all discriminating cues receive equal weight.
    tally_margin = float(np.sum(np.sign(differences)))
    if tally_margin > 0.0:
        tally_scores = np.array([1.0, 0.0])
    elif tally_margin < 0.0:
        tally_scores = np.array([0.0, 1.0])
    else:
        tally_scores = np.array([0.0, 0.0])

    # Overload strategy: validity-ranked compression to the first diagnostic cue.
    cue_order = np.argsort(-validities, kind="stable")
    lex_scores = np.array([0.0, 0.0])
    for cue in cue_order:
        if differences[cue] > 0.0:
            lex_scores = np.array([1.0, 0.0])
            break
        if differences[cue] < 0.0:
            lex_scores = np.array([0.0, 1.0])
            break

    beta = float(parameters["beta"])

    def noisy_strategy_choice(scores):
        logits = beta * np.asarray(scores, dtype=float)
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs /= probs.sum()
        return probs

    # Response noise follows, rather than determines, strategy recruitment.
    p_tally = noisy_strategy_choice(tally_scores)
    p_lex = noisy_strategy_choice(lex_scores)
    p_core = p_integrate * p_tally + (1.0 - p_integrate) * p_lex

    epsilon = float(parameters["epsilon"])
    probs = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, 1.0)
    probs /= probs.sum()
    return probs

`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## THEORY 2 — pi_2
People compare two options by counting, across all features, how often one option has a higher value than the other. The option that wins on more features is chosen. Tallying discards cardinal magnitudes — only the sign of each feature-wise comparison matters — so the heuristic is robust to monotone rescaling of individual features and cannot be swayed by a single large feature difference in the way Equal-Weight can. Ties on an individual feature contribute nothing to either count: that cue is simply treated as uninformative for the pair. No feature is privileged, in contrast to Take The Best; every cue contributes equally to the tally. When the two counts are equal the heuristic has no basis for preference and the learner must guess. Response noise enters through a softmax over the two tallies with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    # Paper-faithful Tallying heuristic (Dawes 1979; Gigerenzer &
    # Goldstein 1999). Stimulus is the pair of option feature vectors
    # for the current trial: array-like of shape (2, n_features),
    # with row 0 = option A, row 1 = option B. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    a, b = stim[0], stim[1]
    # Count strict feature-wise wins; ties contribute to neither option.
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. When
    # a_wins == b_wins the softmax is exactly uniform regardless of
    # beta, which is the correct behavior for an undiscriminating
    # tally.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=8):** [0.95, 0.87, 0.79, 0.72, 0.66, 0.61, 0.56, 0.51]

**Trial pairs (n=12):**
  trial 1: A=[0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 0]
  trial 4: A=[0, 1, 1, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 0]
  trial 5: A=[0, 1, 1, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0, 1, 0]
  trial 6: A=[0, 1, 1, 1, 0, 1, 0, 1]  B=[1, 0, 0, 0, 1, 0, 1, 0]
  trial 7: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0]
  trial 8: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  trial 9: A=[1, 0, 0, 0, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  trial 10: A=[1, 0, 0, 0, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0, 1, 0, 0]
  trial 11: A=[1, 0, 0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0, 1, 0, 0]
  trial 12: A=[1, 0, 0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0, 1, 0, 1]

**Rationale:** This design targets the advocated theory's capacity gate by varying the number of discriminating cues from 3 through 8, spanning the capacity range of 4.5-5.5. In the first six pairs, equal-weight tallying favors A while the highest-validity discriminating cue always favors B; the remaining six pairs are exact A/B reversals. Tally margins are held constant within parity: A wins by one cue at loads 3, 5, and 7, and by two cues at loads 4, 6, and 8. Consequently, the competing Tallying theory predicts load-invariant A-choice probabilities within each parity series—sigmoid(beta) for odd loads and sigmoid(2*beta) for even loads—and always favors the cue-majority option. In contrast, Capacity-Gated Comparison predicts a systematic crossover: strong majority-consistent choices at loads 3-4, an intermediate transition near load 5, and strong highest-validity-cue-consistent choices at loads 6-8. Its post-strategy noise does not depend on tally margin, so the predicted load gradient appears in both parity series. The mirrored pairs balance response side and should exhibit the same crossover toward B rather than A. With 12 unique pairs repeated eight times, the experiment contains 96 trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
The metric measures the decline in choosing the cue-majority option from manageable to overloaded comparisons. It matches contrasts by tally-margin parity: load 3 is compared with load 7, while load 4 is compared with loads 6 and 8. Therefore, Tallying predicts an expected value of exactly zero even when beta and epsilon vary across subjects, because its choice probability is constant within each parity series. Capacity-Gated Comparison instead predicts a large positive value: majority-consistent responding is high at loads 3–4 but reverses toward the highest-validity cue at loads 6–8. Excluding boundary load 5 avoids variance caused by individual capacity differences, and averaging all repetitions, mirror trials, and two parity-specific contrasts reduces response-sampling variance.

Source:
def metric(data: pd.DataFrame) -> float:
    """Return a parity-matched low-to-high load crossover index."""
    if data is None or len(data) == 0:
        return float("nan")

    loads = []
    majority_consistent = []

    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"],
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        diff = a - b
        loads.append(int(np.count_nonzero(diff)))

        tally_margin = float(np.sum(np.sign(diff)))
        if tally_margin > 0:
            majority_response = 0
        elif tally_margin < 0:
            majority_response = 1
        else:
            majority_response = -1

        majority_consistent.append(
            float(int(response) == majority_response)
            if majority_response >= 0 else np.nan
        )

    work = pd.DataFrame({
        "load": np.asarray(loads, dtype=int),
        "majority_consistent": np.asarray(majority_consistent, dtype=float),
    })

    p = {}
    for load in (3, 4, 6, 7, 8):
        values = work.loc[
            work["load"] == load, "majority_consistent"
        ].dropna()
        if len(values) == 0:
            return float("nan")
        p[load] = float(values.mean())

    odd_crossover = p[3] - p[7]
    even_crossover = p[4] - 0.5 * (p[6] + p[8])
    return float(0.5 * (odd_crossover + even_crossover))


### RESULTS
- Predicted under pi_3 (simulated): 0.6512 (var=0.0114)
- Predicted under pi_2 (simulated): 0.0066 (var=0.0037)
- Observed on real data: 0.3612 (var=0.0106)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=8):** [0.95, 0.88, 0.81, 0.74, 0.67, 0.61, 0.56, 0.52]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 1]
  trial 2: A=[1, 1, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0, 1, 0]
  trial 3: A=[1, 0, 0, 0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  trial 4: A=[1, 1, 0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0, 0, 1]
  trial 5: A=[1, 0, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 0, 0, 0]
  trial 6: A=[1, 1, 0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 1, 1, 0, 0]
  trial 7: A=[0, 0, 1, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 1, 0, 0]
  trial 8: A=[1, 0, 0, 1, 0, 0, 1, 0]  B=[1, 1, 0, 0, 1, 0, 0, 0]
  trial 9: A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 0, 0]
  trial 10: A=[1, 0, 1, 1, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 1, 1, 1]
  trial 11: A=[0, 1, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 1, 1]
  trial 12: A=[0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 1, 1]

**Rationale:** This design tests a sharp prediction produced by exact tally ties at and above the proposed capacity boundary. Every pair has an equal number of cues favoring A and B, so Tallying predicts P(A)=0.50 exactly on every trial, independently of beta and epsilon. Discriminating-cue load is varied across 4, 6, and 8 cues. Within each load there are two structurally distinct exemplars, followed by exact A/B reversals. In each unreversed pair, the highest-validity discriminating cue favors A; in each reversal it favors B. Capacity-Gated Comparison therefore predicts near-chance choice at load 4 when integration is usually recruited, but a marked lexicographic preference at loads 6 and 8 as overload recruits the highest-validity cue. Given its parameter ranges, the model predicts roughly P(lex-consistent choice)=0.72-0.89 at load 6 and 0.80-0.89 at load 8, whereas Tallying remains exactly at 0.50. Thus the critical signature is a load-dependent, response-side-symmetric departure from chance toward the highest-validity discriminating cue. Twelve unique pairs repeated eight times yield 96 trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
This metric measures response-side-symmetric agreement with the highest-validity discriminating cue, restricted to the 6- and 8-cue trials where the competing theory predicts overload-driven lexicographic choice. Because every trial is an exact tally tie, Tallying predicts a tightly estimated value of 0.50. Capacity-Gated Comparison predicts substantially higher agreement on these overload trials. Coding agreement rather than raw A-choice also combines each stimulus with its reversal, eliminating fixed A/B response bias. Pooling 64 observations per subject keeps between-subject sampling variance low while excluding the load-4 trials, where both theories are relatively close to chance and would dilute the contrast.

Source:
def metric(data: pd.DataFrame) -> float:
    """Rate of validity-leading-cue-consistent choices on overload trials."""
    scores = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        differences = a - b
        discriminating = np.flatnonzero(differences != 0)

        # Focus on the 6- and 8-cue overload conditions.
        if discriminating.size < 6:
            continue

        # Validities are strictly descending by feature index, so the first
        # discriminating index is the highest-validity diagnostic cue.
        leading_cue = int(discriminating[0])
        lex_response = 0 if differences[leading_cue] > 0 else 1
        scores.append(float(int(row["response"]) == lex_response))

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))

### RESULTS
- Predicted under pi_3 (simulated): 0.8291 (var=0.0019)
- Predicted under pi_2 (simulated): 0.5031 (var=0.0037)
- Observed on real data: 0.8756 (var=0.0078)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Contrast top-cue adherence between tally-congruent and
    # tally-conflicting trials, matching on absolute tally margin.
    cells = {(d, c): [] for d in (2, 4, 6) for c in (-1, 1)}

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        response = int(row["response"])

        # Feature 0 is the highest-validity cue in this experiment.
        if a[0] > b[0]:
            top_choice = 0
            top_relative_margin = int(np.sum(a > b) - np.sum(b > a))
        elif b[0] > a[0]:
            top_choice = 1
            top_relative_margin = int(np.sum(b > a) - np.sum(a > b))
        else:
            continue

        d = abs(top_relative_margin)
        if d not in (2, 4, 6) or top_relative_margin == 0:
            continue

        congruence = 1 if top_relative_margin > 0 else -1
        top_adherence = 1.0 if response == top_choice else -1.0
        cells[(d, congruence)].append(top_adherence)

    matched_contrasts = []
    for d in (2, 4, 6):
        congruent = cells[(d, 1)]
        conflicting = cells[(d, -1)]
        if len(congruent) == 0 or len(conflicting) == 0:
            continue
        # Division by two keeps the metric in [-1, 1].
        matched_contrasts.append(
            0.5 * (float(np.mean(congruent)) - float(np.mean(conflicting)))
        )

    if len(matched_contrasts) == 0:
        return float("nan")
    return float(np.mean(matched_contrasts))
```

**Observed (real) value:** -0.0300 (var=0.0056)
**Predicted under pi_3:** -0.0028 (var=0.0068)
**Predicted under pi_2:** 0.8033 (var=0.0243)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    scores = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        d = a - b

        # +1 denotes an A prediction and -1 a B prediction.
        tally_margin = np.sum(np.sign(d))
        if tally_margin == 0:
            continue
        tally_direction = 1.0 if tally_margin > 0 else -1.0

        # Validities decrease with feature index in this experiment.
        discriminating = np.flatnonzero(d != 0)
        if discriminating.size == 0:
            continue
        first = int(discriminating[0])
        ttb_direction = 1.0 if d[first] > 0 else -1.0

        # Agreement trials cannot distinguish the theories.
        if tally_direction == ttb_direction:
            continue

        observed_direction = 1.0 if int(row["response"]) == 0 else -1.0
        scores.append(observed_direction * tally_direction)

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** 0.6683 (var=0.0367)
**Predicted under pi_3:** 0.6600 (var=0.0188)
**Predicted under pi_2:** 0.7008 (var=0.0514)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all four experiments, pi_3 is the clearly superior task-invariant account. Experiment 1 provides evidence against pure Tallying: the observed load-dependent crossover is 0.3612, far above pi_2's near-zero prediction (0.0066). Pi_3 predicts the correct direction and is somewhat closer at 0.6512, although it substantially overstates the crossover, suggesting that the transition from majority use to leading-cue use is weaker or more gradual than its current instantiation assumes. Experiment 2 strongly favors pi_3: on exact tally ties at high load, participants follow the highest-validity cue at 0.8756, close to pi_3's 0.8291 and incompatible with pi_2's chance prediction of 0.5031. Experiment 3 is an especially decisive failure of pi_2. The observed matched contrast is essentially zero (-0.0300), as predicted by pi_3 (-0.0028), whereas pi_2 predicts 0.8033. This indicates that under uniformly high comparison load, behavior is largely insensitive to whether the tally agrees or conflicts with the top cue, consistent with strong validity-based responding rather than tallying. Experiment 4 does not discriminate strongly because both theories are close, though pi_3 is again numerically closer: observed 0.6683 versus 0.6600 for pi_3 and 0.7008 for pi_2. Thus pi_2 succeeds mainly in the low-load setting where pi_3 itself recruits tallying, but it cannot explain the high-load and exact-tie results. Pi_3 captures the qualitative regime change across tasks and three of the four point estimates well, but its excessive Experiment 1 crossover shows that its sharp mixture of equal-weight and lexicographic strategies is not yet a complete account. Because pi_2 is structurally unable to generate the central high-load effects, changing only its parameters or implementation would not suffice without abandoning its defining claim.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace pi_2 with a continuous load-adaptive validity-concentration theory. The new theory should posit that people always integrate the signs of discriminating cues, but the distribution of attention across those cues changes with comparison load. With few discriminating cues, attention is approximately uniform, producing tally-like choices and explaining Experiment 4. As load increases, limited representational precision causes attention to concentrate progressively on cues with higher communicated validity; the decision remains an integration process rather than switching categorically between two strategies. At loads 6-8, this concentration should produce strong leading-cue adherence on exact tally ties, explaining Experiment 2, and similar top-cue adherence whether tally and top cue agree or conflict, explaining Experiment 3. A gradual, heterogeneous concentration function should produce the positive but moderate crossover in Experiment 1 rather than pi_3's overly large predicted crossover. Any implementation should use only pair-level observables such as the number of discriminating cues and the validity ordering, never experiment identity. A useful operationalization is a signed weighted sum whose validity-weight exponent or attention concentration increases smoothly with discriminating load, with stable subject-level capacity or concentration parameters and ordinary response noise applied afterward. This creates a genuinely stronger competitor to pi_3: it shares the empirically necessary load sensitivity but rejects pi_3's discrete prospective recruitment of equal-weight versus first-cue strategies."
}
```

## Usage

```json
{
  "input_tokens": 8286,
  "output_tokens": 1678
}
```
