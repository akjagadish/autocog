# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_1" and "pi_2") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_1" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_2" (THEORY 2 below). Justify your choice.


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


## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
**Validities (n_features=7):** [0.95, 0.85, 0.75, 0.68, 0.61, 0.56, 0.51]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1, 1]
  trial 2: A=[1, 0, 0, 1, 0, 0, 0]  B=[0, 1, 1, 0, 1, 1, 1]
  trial 3: A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  trial 4: A=[0, 1, 0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1, 0, 1]
  trial 5: A=[1, 1, 0, 0, 1, 0, 0]  B=[1, 0, 1, 1, 0, 1, 1]
  trial 6: A=[0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1]
  trial 7: A=[0, 1, 0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 1, 0, 0]
  trial 8: A=[0, 1, 1, 0, 1, 1, 1]  B=[1, 0, 0, 1, 0, 0, 0]
  trial 9: A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1, 0]
  trial 11: A=[1, 0, 1, 1, 0, 1, 1]  B=[1, 1, 0, 0, 1, 0, 0]
  trial 12: A=[0, 0, 1, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0]

**Rationale:** This 7-cue design creates a direct quantitative dissociation. In each base pair, the highest-validity cue that discriminates favors A, while the equal-weight tally either favors B or is tied; the final six pairs swap A and B to eliminate response-side confounds. For the first three base pairs, cue 1 favors the TTB option, but that option loses the overall tally by margins of 1, 3, and 5. For the next three, cue 1 ties and cue 2 favors the TTB option, while the tally is respectively tied or favors the other option by margins of 2 and 4. TTB predicts the same probability of choosing its designated winner in every condition because its binary winner score is always 1 versus 0, regardless of search depth or tally margin. In contrast, Tallying predicts exactly 0.5 choice probability in the tally-tie condition and increasingly strong preference against the TTB winner as the opposing tally margin grows. Thus the critical signature is an approximately flat, validity-ordered choice profile under TTB versus a graded tally-margin profile under Tallying. The 12 unique pairs yield exactly 96 trials when repeated eight times.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
The metric contrasts agreement with the validity-ordered (TTB) winner between tally-tie trials and all trials on which the equal-weight tally favors the opposite option. TTB predicts the same TTB-winner probability in both sets, so its expected value is zero for every beta and epsilon; this within-subject contrast also cancels stable response accuracy and lapse tendencies. Tallying predicts 0.5 TTB agreement on tally ties but below 0.5 on every conflict trial, yielding a distinctly positive value. Pooling the five conflict-margin conditions supplies 80 conflict observations per subject while retaining 16 tie observations, reducing binomial between-subject noise. Side-swapped pairs ensure that the contrast cannot be generated by a preference for A or B.

Source:
def metric(data: pd.DataFrame) -> float:
    aligned = []
    tally_margins = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)

        # Validities are strictly descending in feature order, so the first
        # feature that differs identifies the Take-The-Best winner.
        differing = np.flatnonzero(a != b)
        if differing.size == 0:
            continue
        j = int(differing[0])
        ttb_winner = 0 if a[j] > b[j] else 1

        a_wins = int(np.sum(a > b))
        b_wins = int(np.sum(b > a))
        tally_margins.append(abs(a_wins - b_wins))
        aligned.append(float(int(row["response"]) == ttb_winner))

    if len(aligned) == 0:
        return float("nan")

    aligned = np.asarray(aligned, dtype=float)
    tally_margins = np.asarray(tally_margins, dtype=int)
    tie = tally_margins == 0
    conflict = tally_margins > 0

    if not np.any(tie) or not np.any(conflict):
        return float("nan")

    # Tally-margin sensitivity: excess TTB agreement on tally ties relative
    # to trials where the tally favors the opposite option.
    return float(np.mean(aligned[tie]) - np.mean(aligned[conflict]))

### RESULTS
- Predicted under pi_1 (simulated): -0.0052 (var=0.0117)
- Predicted under pi_2 (simulated): 0.3523 (var=0.0241)
- Observed on real data: -0.4355 (var=0.0764)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=6):** [0.95, 0.86, 0.77, 0.68, 0.59, 0.51]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1]
  trial 2: A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1]
  trial 3: A=[0, 1, 0, 1, 1, 0]  B=[0, 0, 1, 1, 1, 0]
  trial 4: A=[1, 1, 0, 0, 1, 0]  B=[1, 0, 0, 0, 1, 1]
  trial 5: A=[0, 1, 1, 0, 0, 1]  B=[0, 1, 0, 1, 0, 1]
  trial 6: A=[1, 0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0, 1]
  trial 7: A=[0, 1, 1, 1, 0, 0]  B=[0, 1, 1, 0, 1, 0]
  trial 8: A=[1, 0, 1, 1, 1, 0]  B=[1, 0, 1, 1, 0, 1]
  trial 9: A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 0, 1, 0, 1]
  trial 10: A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  trial 11: A=[0, 0, 1, 1, 1, 0]  B=[0, 1, 0, 1, 1, 0]
  trial 12: A=[1, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 1, 0]
  trial 13: A=[0, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 0, 1]
  trial 14: A=[1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0]
  trial 15: A=[0, 1, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0, 0]
  trial 16: A=[1, 0, 1, 1, 0, 1]  B=[1, 0, 1, 1, 1, 0]

**Rationale:** This 6-cue design makes Tallying parameter-free on every trial: each option wins on exactly one feature, while all remaining features tie, so the two tallies are always equal. Tallying therefore predicts P(A)=P(B)=0.5 regardless of beta or epsilon. In contrast, Take The Best selects the option favored by the highest-validity discriminating cue. The first discriminating cue occurs at ranks 1 through 5 across the eight base pairs, and the lower-ranked discriminating cue always favors the other option. TTB predicts the same above-chance probability for its designated winner at every search depth, namely (1-epsilon)*sigmoid(beta)+epsilon/2. The final eight pairs reverse A and B, balancing response side while preserving the cue-rank structure. Thus, systematic preference for the validity-ordered winner supports TTB, whereas chance-level choice on every pair—including no validity-ordered preference at any search depth—is uniquely predicted by Tallying. Sixteen unique pairs repeated six times yield exactly 96 trials.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
This is a thresholded validity-order alignment metric. Each response receives +1 when it agrees with the option favored by the highest-validity discriminating cue and -1 otherwise. Because the design balances whether A or B is that winner, the contrast is not response-side bias. Tallying has no preference on any trial and therefore predicts alignment near zero; with 96 trials, exceeding 0.25 (more than 62.5% agreement) is rare. Take The Best predicts positive alignment on every pair and should usually exceed this threshold despite lapse and softmax noise. Thresholding suppresses ordinary binomial fluctuations under Tallying and saturation among clearly TTB-like subjects, yielding low between-subject variance and a sharp theory contrast.

Source:
def metric(data: pd.DataFrame) -> float:
    # Score each choice against the winner designated by the highest-validity
    # discriminating cue. Validities are strictly decreasing with feature index.
    signed_scores = []

    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"]
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        differing = np.flatnonzero(a != b)
        if differing.size == 0:
            continue

        first = int(differing[0])
        ttb_winner = 0 if a[first] > b[first] else 1
        signed_scores.append(1.0 if int(response) == ttb_winner else -1.0)

    if not signed_scores:
        return float("nan")

    alignment = float(np.mean(signed_scores))

    # This corresponds to choosing the validity-ordered winner on more than
    # 62.5% of trials. It is well beyond ordinary finite-sample fluctuation
    # around Tallying's predicted 50%, while remaining below typical TTB rates.
    return float(alignment > 0.25)


### RESULTS
- Predicted under pi_1 (simulated): 1.0000 (var=0.0900)
- Predicted under pi_2 (simulated): 0.0000 (var=0.0000)
- Observed on real data: 0.0000 (var=0.0000)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
(no other experiments)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the two experiments, pi_2 is the better surviving theory, but neither theory is task-invariant. In Experiment 1, pi_1 is numerically closer to the observation (absolute error 0.4303 versus 0.7878 for pi_2), yet it still fails qualitatively: its architecture makes the tie-minus-conflict contrast approximately zero, whereas the observed contrast is strongly negative (-0.4355). Pi_2 predicts the opposite sign (+0.3523). The elevated observed between-subject variance (0.0764) also indicates substantially more heterogeneity than either implementation anticipates. In Experiment 2, pi_2 matches the observed metric exactly and pi_1 fails maximally. However, that metric is thresholded: an observed value of zero is compatible both with chance-level tallying and with systematic anti-TTB choice, so it does not uniquely establish pi_2's proposed guessing mechanism. Taken jointly, the results decisively reject positive-polarity TTB, leave pi_2 as the stronger incumbent, but also reveal a reversal in Experiment 1 that ordinary tallying cannot produce. Merely changing beta, epsilon, or their ranges cannot fix either theory's structural sign constraints.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace pi_1 with a task-invariant marked-feature minimization theory rather than retuning TTB. The new theory should propose that subjects may encode the marked state (1) as a burden, defect, or claim to be minimized: they primarily choose the option with fewer marked features, and when marked-feature counts tie, they minimize validity-weighted marked evidence or consult the highest-validity discriminating cue with reversed polarity. This differs from both competitors: TTB treats a positive value on one highest-validity cue as decisive, while Tallying counts positive feature wins without privileging validity. In Experiment 1, the TTB-designated option has fewer 1s on the conflict trials, so marked-feature minimization predicts relatively high TTB agreement there; on the tally-tie pair, reversed validity weighting favors the opposite option, producing the observed negative contrast. In Experiment 2, the options have equal counts, so the reversed-validity tie-break predicts anti-TTB behavior, which\u2014like chance choice\u2014maps to the observed thresholded metric of zero. Include stable subject-level variation in polarity interpretation and tie-break strength to accommodate Experiment 1's high variance, but make polarity a substantive cognitive construct rather than an unconstrained sign parameter. Future tests should complement-code the features and explicitly manipulate whether 1 denotes a benefit or a defect; they should also use an unthresholded alignment metric to distinguish anti-TTB choice from pi_2's chance prediction."
}
```

## Usage

```json
{
  "input_tokens": 5801,
  "output_tokens": 3683
}
```
