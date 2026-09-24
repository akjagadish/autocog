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
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 5: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 6: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 7: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 8: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 9: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 10: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 11: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 12: A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 13: A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  trial 14: A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  trial 15: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 16: A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Rationale:** This design creates a quantitative double dissociation between Take-The-Best (TTB) and Tallying by exploiting their core structural difference: TTB's choice depends ONLY on the single highest-validity discriminating cue, while Tallying's choice depends ONLY on the total count of feature-wise wins. Validities are strictly spread and strictly ordered (0.90, 0.80, 0.70, 0.60, 0.55) so TTB's cue cascade is unambiguous. The design contains three families of trials, each mirrored across A/B to control for side bias. (1) Tally-tie trials (e.g., A=[1,0,0,0,0] vs B=[0,1,0,0,0]): the top cue favors one option but the tally is exactly 1-1. TTB predicts P(top-cue winner) = (1-eps)*sigmoid(beta) + eps/2, which rises above 50% as beta grows; Tallying predicts EXACTLY 50% regardless of beta (softmax over equal scores is uniform), so any systematic preference here falsifies Tallying. (2) The critical conflict-gradient family: the top cue favors A, but B wins on an increasing number of lower cues (tally margins 1-2, 1-3, 1-4 against A). TTB predicts P(A) is CONSTANT across these margins (always sigmoid(beta) for A, since only cue 1 is consulted), whereas Tallying predicts a monotone, graded shift toward B as the tally margin grows (sigmoid(beta), sigmoid(2*beta), sigmoid(3*beta)). This flat-vs-graded profile is the central quantitative signature: a subject whose P(A) declines with tally margin is following Tallying; a subject whose P(A) stays flat is following TTB. (3) A second-order family where cues 1-2 tie and discrimination first occurs at cue 3 (with tally conflicts), testing whether the cascade truly skips tied cues. All 16 unique pairs are non-degenerate (no all-tie pairs) and every pair places the two theories in disagreement or in a guess-vs-decide asymmetry. With K = 96 // 16 = 6 repetitions per pair, each subject contributes 6 samples per cell of the design, giving good power to estimate the flat-vs-graded choice gradient that separates the theories.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
The design was built so that on 12 of the 16 unique pairs (72 of 96 trials per subject) the tallies for A and B differ while the top-validity discriminating cue favors the option with FEWER feature wins — i.e., TTB and Tallying make strictly opposite predictions on every one of these trials (e.g., trial 2: A=[1,0,0,0,0] vs B=[0,1,1,0,0]; TTB picks A via cue 1, Tallying picks B via a 2-vs-1 tally; trial 15: cue 3 favors A but the tally is 1-vs-2 for B). The metric computes, purely from the observed choices, the proportion of these conflict trials on which the subject picked the TTB-favored option. Under TTB-simulated data this equals (1-eps)*sigmoid(beta) + eps/2 on every such trial — well above 0.5 for any non-trivial beta, and approaching ~1 for moderate/large beta. Under Tallying-simulated data it equals (1-eps)*(1 - sigmoid(margin*beta)) + eps/2 with margin >= 1 — well below 0.5, approaching ~eps/2 for moderate/large beta. So the two theories produce a large mean gap (typically 0.5-0.9 apart) in opposite directions. Crucially, the metric is a bounded proportion estimated from ~72 Bernoulli trials per subject, so per-subject sampling variance is tiny (<= p(1-p)/72 ~ 0.0035), making the between-subject variance small for both theories; with N=25 human subjects Welch's t-test on the (mean, variance) pairs will separate the theories decisively. The metric is computed generically from the ratings columns (cue cascade + tally counts recomputed row-wise), so it also runs unchanged on a single subject's slice, satisfying the per-subject variance requirement. Unlike a raw 'fit to TTB' measure, it is restricted to the double-dissociation trials where the two theories make opposite deterministic predictions, which is exactly where the design has maximal diagnostic leverage and where noise from the tie-trials (where Tallying guesses) cannot dilute the contrast.

Source:
def metric(data: pd.DataFrame) -> float:
    """P(choose the Take-The-Best winner) on trials where TTB and Tallying
    make OPPOSITE deterministic predictions (i.e., trials where the two
    options win on different numbers of features, so the tallies disagree).

    High value -> TTB behavior; low value -> Tallying behavior.
    Works on the pooled DataFrame and on a single subject's slice.
    """
    validities = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    cue_order = np.argsort(-validities, kind="stable")

    def _to_matrix(col):
        rows = [list(r) for r in col]
        return np.array(rows, dtype=float)

    a = _to_matrix(data["option_a_ratings"])
    b = _to_matrix(data["option_b_ratings"])
    resp = np.asarray(data["response"].to_numpy(), dtype=float)

    n_trials = a.shape[0]

    # --- TTB winner per trial (cue cascade, first discriminating cue) ---
    winners = np.full(n_trials, -1, dtype=int)
    for j in cue_order:
        undecided = winners < 0
        a_better = undecided & (a[:, j] > b[:, j])
        b_better = undecided & (b[:, j] > a[:, j])
        winners[a_better] = 0
        winners[b_better] = 1

    # --- Tallying score per trial (number of strict feature-wise wins) ---
    a_wins = (a > b).sum(axis=1)
    b_wins = (b > a).sum(axis=1)

    # Conflict trials: tallies differ (so Tallying has a strict winner that,
    # in this design, is always the option OPPOSITE to the TTB winner),
    # and TTB also has a strict winner.
    conflict = (a_wins != b_wins) & (winners >= 0)

    if not np.any(conflict):
        # No diagnostic trials -> uninformative, return chance.
        return 0.5

    agree = float(np.mean(resp[conflict] == winners[conflict]))
    return agree

### RESULTS
- Predicted under pi_1 (simulated): 0.8444 (var=0.0125)
- Predicted under pi_2 (simulated): 0.1203 (var=0.0072)
- Observed on real data: 0.7117 (var=0.0205)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=6):** [0.95, 0.9, 0.85, 0.65, 0.6, 0.5]

**Trial pairs (n=24):**
  trial 1: A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  trial 2: A=[0, 0, 1, 1, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  trial 3: A=[0, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0]
  trial 5: A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0]
  trial 6: A=[0, 0, 1, 0, 1, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 7: A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  trial 8: A=[0, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 9: A=[0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0]
  trial 10: A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  trial 11: A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 12: A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 13: A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0, 0]
  trial 14: A=[0, 1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1]
  trial 15: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  trial 16: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 17: A=[0, 1, 0, 0, 1, 0]  B=[1, 0, 1, 1, 0, 0]
  trial 18: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 0]
  trial 19: A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  trial 20: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  trial 21: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 22: A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  trial 23: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0]
  trial 24: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 0]

**Rationale:** This design is the complementary mirror of the already-explored experiment: instead of holding TTB's deciding cue fixed and varying the tally margin, it holds the TALLY MARGIN fixed and varies WHICH CUE TTB's cascade lands on. Six experts with validities [0.95, 0.90, 0.85, 0.65, 0.60, 0.55] create a sharp gap between high-validity cues (1-3) and low-validity cues (4-6), so the identity of TTB's first discriminating cue is unambiguous. Every trial is built around a designated tally-winner W with a fixed tally margin m in {1, 2, 3}, crossed with the DIRECTION of the top discriminating cue: CONGRUENT cells (top cue favors W, so TTB and Tallying agree in direction) and CONFLICT cells (top cue favors the loser, so TTB and Tallying point in opposite directions). Within conflict cells at m=1, the deciding cue's position d is also varied (d=1..4), testing whether choices track the validity of the deciding cue. The quantitative signature is a 2x3 factorial of predicted P(W): Tallying predicts P(W) = sigmoid(m*beta) + lapse, i.e., IDENTICAL across congruent/conflict cells at each margin and a monotone gradient across margins [sigma(beta) < sigma(2*beta) < sigma(3*beta)]. TTB predicts P(W) = sigmoid(beta) in every congruent cell and 1 - sigmoid(beta) in every conflict cell, i.e., IDENTICAL across margins (its noise model does not scale with the number of supporting cues) and independent of which cue d decides. Thus Tallying predicts a main effect of margin only; TTB predicts a main effect of top-cue direction only. A subject whose P(W) rises with m at fixed direction is Tallying; a subject whose P(W) is flat across m but flips with cue direction is TTB. Even the congruent cells dissociate at m>=2 (sigma(beta) vs sigma(2*beta)), and the m=1 congruent cells serve as agreement baselines for estimating beta. All 12 base pairs are mirrored (A/B swapped) to control side bias, giving 24 unique pairs with K = 96 // 24 = 4 repetitions each (96 trials total, 8 samples per base cell per subject). No pair is degenerate (every pair has discriminating cues, none is an all-tie), and no pair leaves both theories in pure-guess agreement. Secondary exploratory axis: both theories predict FLAT P(W) across the deciding-cue position d (TTB because its softmax does not scale with validity, Tallying because it ignores validity entirely), so any graded sensitivity to d would falsify both — but the primary dissociation rests on the margin x direction interaction.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



### METRIC
Rationale:
The design's core dissociation is the CONFLICT cell: trials where the tally majority (the designated winner W) is opposed by the highest-validity discriminating cue. On those trials the two theories make opposite-signed predictions about P(choose W) for ANY noise parameters: Tallying gives P(W) = sigmoid(m*beta) with m >= 1, hence strictly > 0.5 (it obeys the cue majority even against the best cue); TTB gives P(W) = 1 - sigmoid(beta), hence strictly < 0.5 (it obeys the single top cue against the majority). So the metric P(W | conflict) - 0.5 is positive under Tallying and negative under TTB regardless of beta, epsilon, or which cue position d decides — the sign itself is diagnostic, and the magnitude grows with beta on both sides (the gap is (1-epsilon)*[weighted sigma(m*beta) + sigma(beta) - 1] over the 64 conflict trials per subject). This uses 64 of 96 trials per subject (conflict base trials 1,2,3,4,7,8,9,11,13,14,15,16,19,20,21,23 x 4 reps), giving tight per-subject estimates: e.g., at beta=1 the Tallying value is ~ +0.31 and the TTB value ~ -0.23 (gap ~0.55) with per-subject SD ~0.05-0.06, which at N=25 yields a Welch t in the dozens; even at the minimum beta=0.1 the gap (~0.07) still exceeds the per-subject noise (~0.018 SE). Congruent trials are deliberately excluded: at m=1 congruent both theories predict exactly sigmoid(beta), so they add variance without adding mean separation. The metric is computed identically on the pooled DataFrame and on single-subject slices (each subject has all 24 unique pairs), so between-subject variance is well defined. The classification (tally winner, top discriminating cue via the fixed descending-validity order) is recomputed from the raw ratings on every row, so no list-column hashing is needed.

Source:
def metric(data: pd.DataFrame) -> float:
    # Signature: allegiance to the tally winner on CONFLICT trials
    # (trials where the single top-validity discriminating cue favors
    # the tally LOSER). Tallying predicts P(choose tally winner) > 0.5
    # there (it follows the cue majority against the best cue);
    # Take-The-Best predicts P < 0.5 (it follows the top cue against
    # the majority). We return P(W | conflict) - 0.5, so the sign
    # itself identifies the theory for any noise level beta > 0.
    validities = np.array([0.95, 0.90, 0.85, 0.65, 0.60, 0.50])
    cue_order = np.argsort(-validities, kind="stable")

    a_list = data["option_a_ratings"].tolist()
    b_list = data["option_b_ratings"].tolist()
    resp = data["response"].to_numpy()

    conflict_w = []
    for a_raw, b_raw, r in zip(a_list, b_list, resp):
        a = np.asarray(a_raw, dtype=float)
        b = np.asarray(b_raw, dtype=float)

        a_w = float(np.sum(a > b))
        b_w = float(np.sum(b > a))
        if a_w == b_w:
            # Tally is tied -> heuristic guesses; uninformative cell.
            continue
        winner = 0 if a_w > b_w else 1

        # First discriminating cue in descending-validity order.
        top_favors = None
        for j in cue_order:
            if a[j] > b[j]:
                top_favors = 0
                break
            if b[j] > a[j]:
                top_favors = 1
                break
        if top_favors is None:
            continue  # no discriminating cue at all

        if top_favors == winner:
            continue  # congruent trial: both heuristics point the same way

        # Conflict trial: tally winner opposes the top cue's favorite.
        conflict_w.append(1.0 if int(r) == winner else 0.0)

    if len(conflict_w) == 0:
        return 0.0
    return float(np.mean(conflict_w)) - 0.5

### RESULTS
- Predicted under pi_1 (simulated): -0.3231 (var=0.0164)
- Predicted under pi_2 (simulated): 0.3394 (var=0.0118)
- Observed on real data: -0.2562 (var=0.0128)

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
  "interpretation": "Across both experiments, Take-The-Best (pi_1) is the clear winner. Experiment 1 was designed as a double dissociation: on conflict trials the metric (proportion of choices following the TTB winner) is predicted to be ~0.84 under TTB and ~0.12 under Tallying; the observed value is 0.7117, squarely on the TTB side of chance and far from the Tallying prediction. Experiment 2 was designed by the Tallying camp as the complementary dissociation (allegiance to the tally winner on conflict trials, centered at 0): TTB predicts a negative value (-0.323 simulated), Tallying a positive one (+0.339 simulated); the observed value is -0.2562, again on the TTB side. Critically, pi_1 wins not only on its own home turf but also on the experiment proposed by its rival \u2014 this is the strongest possible evidence of task-invariance within this domain. Tallying (pi_2) fails on both experiments, and fails in the predicted direction (subjects systematically follow the top-validity cue against the cue majority), so its failure is structural, not a parameter-tuning issue. That said, pi_1's fit is not perfect: the observed allegiance is consistently attenuated toward chance relative to TTB's simulated predictions (0.71 vs 0.84; -0.256 vs -0.323), suggesting real subjects weigh the top cue strongly but not exclusively \u2014 lower-validity cues and/or the tally margin exert a small pull. This attenuation is a systematic, replicable pattern across both experiments, and it is exactly the kind of residual that a pure one-reason rule cannot express (TTB's prediction is flat across tally margins by construction) but that a validity-weighted evidence-integration rule can. Since pi_2 is degenerate (wrong sign on both experiments) while pi_1 is qualitatively correct on both, the right move is to replace pi_2 with a genuinely stronger competitor rather than to modify the winning theory: a validity-weighted additive/integrative model that reduces to TTB-like behavior when validities are steeply ordered (as in both experiments) but predicts graded, margin-sensitive choice when they are not. This creates a real scientific contest for the next round instead of re-tuning a theory that already explains the data.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace pi_2 with a 'validity-weighted evidence integration' theory. Sketch: on each trial, compute for each feature j the discriminant d_j = sign(a_j - b_j) in {-1, 0, +1}, and accumulate evidence E = sum_j w_j * d_j, where w_j = log(v_j / (1 - v_j)) (the cue's log-odds validity, communicated in the instructions). Choice probability is softmax(beta * E) with an independent lapse epsilon, plus uniform guessing when E = 0. This theory differs from both incumbents: unlike Tallying, it does not weight all cues equally \u2014 high-validity cues dominate \u2014 so it correctly predicts top-cue allegiance on both existing experiments (E is dominated by the top cue under steep validity gradients like [0.9, 0.8, ...] and [0.95, 0.9, 0.85, ...]). Unlike TTB, it does not discard lower cues entirely \u2014 the accumulated evidence retains a graded dependence on the number and validity of supporting cues \u2014 so it naturally predicts the attenuated conflict-allegiance observed in the data (0.71 and -0.26 rather than near-deterministic TTB values), and it makes testable predictions where TTB is flat (e.g., P(choice) should shift with the validity-weighted margin of lower cues, and with the absolute validity of the deciding cue). Parameter ranges: beta in [0.5, 5], epsilon in [0, 0.2], with an optional steepness exponent gamma >= 1 on the weights (w_j proportional to [log(v_j/(1-v_j))]^gamma) so that gamma -> infinity recovers pure TTB and gamma = 1 gives smooth integration, making the TTB-vs-integration question empirically identifiable."
}
```

## Usage

```json
{
  "input_tokens": 7365,
  "output_tokens": 1932,
  "reasoning_tokens": 1017,
  "provider": "Sail Research"
}
```
