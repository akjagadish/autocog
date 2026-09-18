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
Accessibility-Balanced Evidence Integration proposes that people encode each communicated cue validity as a subjective diagnosticity signal and combine it with a separate positional-accessibility signal. Diagnosticity favors objectively valid cues, with nonlinear compression or sharpening of validity differences, whereas accessibility can favor cues appearing later or lower in the display. A subject-specific balance determines how strongly each code influences cue weight. On each choice, all discriminating cues contribute their signed evidence to a common accumulator; therefore, several opposing cues can gradually overcome an initially favored cue. Because the accumulator retains evidence magnitude and configuration, the theory predicts graded coalition effects rather than selecting either the first or last discriminating cue. Evidence supporting each option exhibits mild, imbalance-dependent within-coalition diminishing returns before the two directional totals are compared, followed by ordinary response noise and occasional lapses.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Accessibility-Balanced Integration expects shape (2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = stim.shape[1]

    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    balance = float(parameters["validity_position_balance"])
    recency = float(parameters["recency_gradient"])
    curvature = float(parameters["validity_curvature"])
    coalition_saturation = float(parameters["coalition_saturation"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Communicated validities are represented as log-odds diagnosticities.
    # Curvature captures sharpening or compression of perceived differences.
    v = np.clip(validities, 0.500001, 0.999999)
    diagnosticity = np.log(v / (1.0 - v))
    max_diagnosticity = float(np.max(diagnosticity))
    if max_diagnosticity > 0.0:
        diagnosticity = diagnosticity / max_diagnosticity
    else:
        diagnosticity = np.ones(n_features, dtype=np.float64)
    diagnosticity = np.power(np.maximum(diagnosticity, 0.0), curvature)
    diagnosticity /= max(float(np.mean(diagnosticity)), 1e-12)

    # Later/lower display positions have greater accessibility when recency > 0.
    if n_features == 1:
        position = np.zeros(1, dtype=np.float64)
    else:
        position = np.arange(n_features, dtype=np.float64) / float(n_features - 1)
    accessibility = np.exp(recency * position)
    accessibility /= max(float(np.mean(accessibility)), 1e-12)

    # A convex mixture makes the validity-versus-position tradeoff explicit.
    weights = balance * diagnosticity + (1.0 - balance) * accessibility
    weights /= max(float(np.mean(weights)), 1e-12)

    differences = a - b
    discriminating = differences != 0.0
    n_discriminating = int(np.sum(discriminating))
    if n_discriminating == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    # Evidence is accumulated separately for the two options. Diminishing
    # returns depend on directional coalition imbalance rather than absolute
    # coalition size, preserving full configuration evidence for equal-sized
    # coalitions while tempering a numerical majority against a minority.
    positive = differences > 0.0
    negative = differences < 0.0
    n_positive = max(int(np.sum(positive)), 1)
    n_negative = max(int(np.sum(negative)), 1)
    min_coalition = min(n_positive, n_negative)
    positive_evidence = float(np.dot(weights, np.maximum(differences, 0.0)))
    negative_evidence = float(np.dot(weights, np.maximum(-differences, 0.0)))
    evidence_a = (
        positive_evidence
        / (max(1.0, float(n_positive) / float(min_coalition)) ** coalition_saturation)
        - negative_evidence
        / (max(1.0, float(n_negative) / float(min_coalition)) ** coalition_saturation)
    )

    logits = np.array(
        [0.5 * beta * evidence_a, -0.5 * beta * evidence_a],
        dtype=np.float64,
    )
    logits -= np.max(logits)
    probs = np.exp(logits)
    probs /= probs.sum()

    probs = (1.0 - epsilon) * probs + epsilon * 0.5
    probs = np.maximum(probs, 0.0)
    probs /= probs.sum()
    return probs

`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.maximum(probs, 0.0)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
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
**Validities (n_features=7):** [0.52, 0.58, 0.65, 0.72, 0.8, 0.88, 0.95]

**Trial pairs (n=12):**
  trial 1: A=[1, 1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 0, 0, 1]
  trial 3: A=[1, 0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0, 1]
  trial 4: A=[0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0, 1]
  trial 5: A=[1, 0, 0, 0, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0, 0]
  trial 6: A=[0, 0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 1, 0, 0, 0]
  trial 8: A=[0, 0, 0, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 1, 0]
  trial 9: A=[0, 1, 0, 0, 0, 0, 1]  B=[1, 0, 0, 1, 0, 1, 0]
  trial 10: A=[1, 0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0, 1, 0]
  trial 11: A=[0, 1, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 1]
  trial 12: A=[1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1, 1]

**Rationale:** This 7-cue design holds the tally margin constant while systematically relocating the same 3-versus-2 cue coalition from early, low-validity positions to late, high-validity positions. In each of the first six pairs, A wins exactly three cues, B wins exactly two, and two cues tie. Tallying therefore predicts exactly the same A-choice probability on all six: (1-epsilon)*sigmoid(beta)+epsilon/2. The final six pairs are A/B-swapped mirrors, for which Tallying predicts the complementary constant probability. Accessibility-Balanced Evidence Integration instead assigns progressively greater weight to later cues; the ascending validities reinforce this positional gradient throughout the advocated parameter ranges. Consequently, its predicted A-choice probability should increase monotonically across the first six configurations. At the low end, A's three early wins should be overcome by B's two late wins, reversing the tally preference; at the high end, A's late coalition should produce a strong A preference. Coalition saturation further penalizes A's three-cue majority relative to B's two-cue coalition but is constant in form across the progression. The mirrored pairs should exhibit the complementary graded pattern, controlling response-side bias. Thus the decisive signature is a large configuration-dependent slope, including a preference reversal despite an invariant tally margin; Tallying permits neither. There are 12 unique pairs, each repeated eight times, yielding exactly 96 trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
This is a preregistered configuration-sensitivity covariance after orienting every stimulus toward its 3-cue tally winner. The centered weight is the majority-choice probability predicted by a fixed midpoint instantiation of Accessibility-Balanced Evidence Integration. Thus configurations where the majority coalition occupies strong late cues receive positive weights, while configurations where its three cues are outweighed by two late cues receive negative weights. Under Tallying, every oriented trial has the identical 3-versus-2 margin and hence the same expected majority-choice rate; centering the weights makes its expected metric exactly zero for every beta and epsilon. The advocated theory predicts that observed majority choice covaries strongly with these weights. All 96 responses contribute to the scalar, mirror trials are automatically placed in the same canonical orientation, and the balanced repeated schedule keeps the per-subject estimate comparatively stable. The weights are fixed independently of responses rather than fitted to the observed data.

Source:
def metric(data: pd.DataFrame) -> float:
    if len(data) == 0:
        return 0.0

    # Fixed, preregistered midpoint instantiation of the advocated model.
    validities = np.array([0.52, 0.58, 0.65, 0.72, 0.80, 0.88, 0.95], dtype=float)
    balance = 0.30
    recency = 2.10
    curvature = 0.825
    saturation = 0.725
    beta = 0.90
    epsilon = 0.05

    v = np.clip(validities, 0.500001, 0.999999)
    diagnosticity = np.log(v / (1.0 - v))
    diagnosticity /= np.max(diagnosticity)
    diagnosticity = np.power(np.maximum(diagnosticity, 0.0), curvature)
    diagnosticity /= np.mean(diagnosticity)

    position = np.arange(len(validities), dtype=float) / float(len(validities) - 1)
    accessibility = np.exp(recency * position)
    accessibility /= np.mean(accessibility)

    cue_weights = balance * diagnosticity + (1.0 - balance) * accessibility
    cue_weights /= np.mean(cue_weights)

    predicted_majority_probs = []
    chose_majority = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        d = a - b

        # Orient every pair toward the option winning the 3-versus-2 tally.
        tally_margin = float(np.sum(d > 0) - np.sum(d < 0))
        if tally_margin == 0.0 or len(d) != len(cue_weights):
            continue
        majority_is_a = tally_margin > 0.0
        oriented = d if majority_is_a else -d

        positive = oriented > 0.0
        negative = oriented < 0.0
        n_pos = max(int(np.sum(positive)), 1)
        n_neg = max(int(np.sum(negative)), 1)
        min_n = min(n_pos, n_neg)

        pos_evidence = float(np.dot(cue_weights, np.maximum(oriented, 0.0)))
        neg_evidence = float(np.dot(cue_weights, np.maximum(-oriented, 0.0)))
        evidence = (
            pos_evidence / (max(1.0, float(n_pos) / float(min_n)) ** saturation)
            - neg_evidence / (max(1.0, float(n_neg) / float(min_n)) ** saturation)
        )

        core_p = 1.0 / (1.0 + np.exp(-np.clip(beta * evidence, -50.0, 50.0)))
        predicted_p = (1.0 - epsilon) * core_p + epsilon * 0.5
        predicted_majority_probs.append(predicted_p)

        response = int(row["response"])
        chose_majority.append(float((response == 0) if majority_is_a else (response == 1)))

    if len(predicted_majority_probs) == 0:
        return 0.0

    p = np.asarray(predicted_majority_probs, dtype=float)
    y = np.asarray(chose_majority, dtype=float)
    contrast = p - np.mean(p)
    return float(np.mean(contrast * (y - 0.5)))


### RESULTS
- Predicted under pi_3 (simulated): 0.0860 (var=0.0002)
- Predicted under pi_2 (simulated): -0.0017 (var=0.0001)
- Observed on real data: -0.0410 (var=0.0003)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=9):** [0.95, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.52]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0, 1, 0, 1]
  trial 2: A=[1, 0, 1, 1, 1, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0, 1, 0, 1]
  trial 3: A=[1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[1, 1, 1, 0, 0, 0, 1, 0, 1]
  trial 4: A=[1, 0, 1, 0, 1, 1, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0, 1, 0, 1]
  trial 5: A=[1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0, 0, 0, 1]
  trial 6: A=[1, 0, 1, 0, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 1, 0, 1, 0, 1]
  trial 7: A=[1, 1, 0, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1, 0, 1]
  trial 8: A=[1, 1, 1, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 1, 0, 1, 0, 1]
  trial 9: A=[1, 1, 1, 0, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1, 0, 1]
  trial 10: A=[1, 1, 1, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 1, 1, 0, 1]
  trial 11: A=[1, 1, 1, 0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1, 0, 1]
  trial 12: A=[1, 1, 1, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1, 1, 1]

**Rationale:** This design isolates a positional-accessibility gradient while holding both cue tally and diagnosticity exactly constant. In each of the first six pairs, A wins one cue and B wins one cue, so Tallying predicts P(A)=0.5 exactly, independently of beta and epsilon. B's win is always at Expert 2 (index 1), whereas A's win moves successively from Expert 3 through Expert 8 (indices 2-7). All of these discriminating experts have identical advertised validity (0.70), so their diagnosticity components are equal. Accessibility-Balanced Evidence Integration therefore predicts a strictly increasing A-choice probability as A's supporting cue moves later: its evidence difference is proportional to accessibility(target)-accessibility(index 1). Coalition saturation is inactive because the coalitions are always one cue versus one cue. Across the competing theory's parameter ranges, the nearest-position contrast should be only slightly above chance (approximately 0.51-0.52), while the most separated contrast should produce approximately 0.62-0.79 A choices. The final six pairs swap A and B, yielding complementary predictions and controlling response-side bias. The high- and low-validity experts provide the required communicated validity spread but are tied in these trials, ensuring that the graded competing-theory prediction is specifically attributable to positional accessibility rather than validity. There are 12 unique pairs, each repeated eight times for 96 trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
This is an accessibility-weighted rate of choosing the option supported by the later of the two discriminating experts. It recodes the mirrored A/B trials into a common direction, thereby cancelling response-side preference, and pools all repetitions so it remains stable at the individual-subject level. Tallying gives every trial an exact 0.5 probability because each option wins one cue. Accessibility-Balanced Integration instead predicts increasingly frequent selection of the later-supported option as its supporting cue moves farther down the display. The exponential contrast weight is a matched-filter approximation based on the midpoint of the competing theory's recency-gradient range: it downweights the weak, noisy nearest-position contrast and emphasizes trials on which the competing theory predicts its largest departure from chance.

Source:
def metric(data: pd.DataFrame) -> float:
    weighted_choices = []
    weights = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.shape != b.shape or a.ndim != 1 or a.size < 2:
            continue

        differing = np.flatnonzero(a != b)
        if differing.size != 2:
            continue

        early = int(np.min(differing))
        late = int(np.max(differing))
        late_supports_a = bool(a[late] > b[late])
        later_option_response = 0 if late_supports_a else 1
        chose_later_supported_option = float(int(row["response"]) == later_option_response)

        # Approximate matched-filter weight for the competing theory's
        # exponential positional-accessibility contrast.
        scale = float(a.size - 1)
        weight = np.exp(2.1 * late / scale) - np.exp(2.1 * early / scale)
        if np.isfinite(weight) and weight > 0.0:
            weighted_choices.append(chose_later_supported_option)
            weights.append(weight)

    if len(weights) == 0:
        return float("nan")

    return float(np.average(np.asarray(weighted_choices), weights=np.asarray(weights)))

### RESULTS
- Predicted under pi_3 (simulated): 0.6281 (var=0.0049)
- Predicted under pi_2 (simulated): 0.5017 (var=0.0041)
- Observed on real data: 0.5087 (var=0.0038)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Mean signed endorsement of the highest-validity cue's winner."""
    scores = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        response = int(row["response"])

        # Feature 0 is the uniquely highest-validity cue in this design.
        if a[0] > b[0]:
            top_cue_winner = 0
        elif b[0] > a[0]:
            top_cue_winner = 1
        else:
            continue

        scores.append(1.0 if response == top_cue_winner else -1.0)

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** -0.1217 (var=0.0117)
**Predicted under pi_3:** -0.1033 (var=0.0092)
**Predicted under pi_2:** -0.6354 (var=0.0266)

### Experiment 4
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 1, 1, 0, 1]
  A=[0, 0, 0, 1, 1, 1, 0, 1]  B=[1, 1, 1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0, 1, 0, 1]
  A=[0, 1, 1, 0, 0, 1, 0, 1]  B=[1, 0, 0, 1, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 1, 1, 0, 0]  B=[1, 0, 1, 1, 0, 0, 1, 0]
  A=[1, 0, 1, 1, 0, 0, 1, 0]  B=[1, 1, 0, 0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 1, 1, 0]  B=[1, 0, 0, 1, 1, 0, 0, 1]
  A=[1, 0, 0, 1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Mean signed agreement with the highest-validity discriminating cue."""
    if data is None or len(data) == 0:
        return 0.0

    signed_agreements = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)

        # Features are already ordered from highest to lowest validity.
        differing = np.flatnonzero(a != b)
        if differing.size == 0:
            continue

        j = int(differing[0])
        ttb_response = 0 if a[j] > b[j] else 1
        observed_response = int(row["response"])
        signed_agreements.append(1.0 if observed_response == ttb_response else -1.0)

    if not signed_agreements:
        return 0.0
    return float(np.mean(signed_agreements))
```

**Observed (real) value:** -0.2642 (var=0.0087)
**Predicted under pi_3:** -0.1892 (var=0.0123)
**Predicted under pi_2:** -0.0008 (var=0.0102)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Neither theory is task-invariant. In Experiment 1, the observed configuration covariance is negative (-0.0410), much closer to Tallying (-0.0017) than to Accessibility-Balanced Integration (+0.0860), and it reverses the latter's predicted direction. In Experiment 2, the observed later-cue choice rate (0.5087) is essentially chance and closely matches Tallying (0.5017), while strongly rejecting pi_3's fixed positive accessibility gradient (0.6281). These failures are not plausibly dismissed as instability: Experiment 1 has low between-subject variance, and Experiment 2's observed variance is no larger than the simulated variances. Conversely, Tallying fails badly on the earlier experiments. In Experiment 3, pi_3 nearly matches the observed endorsement of the highest-validity cue (-0.1033 versus -0.1217), whereas pi_2 predicts a much stronger majority-driven reversal (-0.6354). In Experiment 4, pi_3 also captures the negative agreement with the highest-validity discriminating cue better (-0.1892 versus observed -0.2642) than pi_2, which predicts approximately zero. Thus pi_2 succeeds only where the designs produce near-chance or weak configuration effects, but its rigid equal counting cannot represent the graded coalition behavior evident in Experiments 3 and 4. Pi_3 captures those coalition effects, yet its context-independent, monotonically increasing positional-accessibility component is contradicted by Experiments 1 and 2. Merely retuning either current model is unlikely to reconcile all four signatures: a positive recency gradient cannot generate Experiment 1's negative covariance, while pure tallying cannot approach Experiments 3 and 4. Pi_2 is the more structurally degenerate theory because it ignores communicated validity and all cue configuration by definition, so it should be replaced while retaining pi_3 as the current benchmark.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace Tallying with a Coalition-Structured Bounded Integration theory. The new theory should propose that people integrate multiple discriminating cues, but that cue influence is determined jointly by compressed communicated validity, coalition structure, and bounded sequential attention\u2014not by equal counts or a universal recency gradient. Positional influence should be conditional: isolated one-versus-one conflicts among equally valid cues should receive little or no positional weighting, explaining Experiment 2, whereas multi-cue coalitions may be chunked or attentionally gated so that an early coherent majority can receive a primacy advantage, allowing the negative covariance in Experiment 1. At the same time, additional opposing cues must accumulate gradually rather than trigger a deterministic tally rule, allowing growing coalitions to overcome a highly valid or first-discriminating cue with moderate probability as in Experiments 3 and 4. A useful implementation would include compressed validity weights, a coalition-size accumulation function with diminishing returns, a configuration-dependent attention term whose sign and magnitude depend on whether evidence is singleton or coalition-based, response noise, and subject-level heterogeneity. It should explicitly avoid both pi_2's invariant equal weighting and pi_3's fixed monotonic late-position accessibility. Predictions should be jointly calibrated across all four experiments, with preregistered constraints requiring near-chance behavior in Experiment 2, negative configuration covariance in Experiment 1, and moderate\u2014not deterministic\u2014coalition reversals in Experiments 3 and 4."
}
```

## Usage

```json
{
  "input_tokens": 8499,
  "output_tokens": 2237
}
```
