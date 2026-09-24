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
**Validities (n_features=5):** [0.95, 0.85, 0.65, 0.6, 0.5]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 4: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 5: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 6: A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  trial 7: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 8: A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  trial 9: A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 10: A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]
  trial 11: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 12: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  trial 13: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 14: A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  trial 15: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 16: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Rationale:** The design exploits two orthogonal dissociations between Take-The-Best (TTB) and Tallying. (1) SIGN dissociation with graded tally margins: On conflict trials, the top-validity cue favors one option while the majority of cues favors the other. Critically, TTB's predicted choice probability is a constant sigmoid(beta) on every such trial (it only ever uses the single top discriminating cue), whereas Tallying's confidence scales with the tally margin: sigmoid(beta) for 2-vs-1, sigmoid(2*beta) for 3-vs-1, sigmoid(3*beta) for 4-vs-1. So TTB predicts FLAT confidence across trials 1/3/5/13 (margins 3, 1, 2, 1 respectively) while Tallying predicts systematically increasing confidence in the majority option — a quantitative, trial-level signature that model recovery cannot confuse. Mirrored versions (trials 2/4/6/14) control for side bias. (2) TALLY-TIE dissociation: Trials 7/8 have a 2-2 tally with one tied cue, so Tallying must guess (exactly 50%) while TTB is confident (sigmoid(beta)) because the top cue discriminates. Agreement/control trials (9-12, 15-16) anchor both models and check that subjects are not simply counting or simply using cue 1. Validities are spread (0.95 down to 0.55) so the validity ordering is unambiguous and communicated up front; the second and third cues are placed far enough below the top cue that TTB's cascade is well defined, and low-validity cues (0.65, 0.60, 0.55) are close together so Tallying treats them as a coherent majority block. With 16 unique pairs and K=6 repetitions (96 trials), each dissociating cell gets enough repetitions for reliable per-cell choice proportions. Predicted pattern: if TTB is correct, choice proportions on trials 1, 3, 5, 13 should be statistically indistinguishable from each other (all ~sigmoid(beta) favoring the top-cue option) and well above 50% on trials 7/8; if Tallying is correct, choices on conflict trials should track the tally margin (increasingly favoring the majority option from trial 3 -> 5 -> 1) and trials 7/8 should sit at exactly 50%.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
The design contains 10 of 16 trial types where TTB and Tallying make different modal predictions: conflict trials (1-6, 13-14; 48 of 96 trials) where the top cue favors one option but the cue majority favors the other, and tally-tie trials (7-8; 12 of 96 trials) where the tally is 2-2 so Tallying must guess at exactly 50% while TTB is confident via the top cue. The metric recomputes both heuristics row-by-row from the ratings and the known validities, keeps only these dissociation rows, and returns P(choose the TTB-predicted option) - 0.5. Under TTB, every dissociation trial is chosen with probability (1-eps)*sigmoid(beta) + eps/2 > 0.5, so the metric is positive (approaching +0.5 for large beta). Under Tallying, the 48 conflict trials are chosen at (1-eps)*sigmoid(-beta*margin) + eps/2 < 0.5 and the 12 tie trials at exactly 0.5, so the metric is negative (approaching -0.4 for large beta). The sign itself flips between theories, giving a large mean gap (e.g., ~0.66 at beta=2, eps=0.1; still ~0.065 even at the near-degenerate beta=0.1, eps=0). Per-subject variability is pure binomial noise over ~60 trials (sd <= ~0.065 at worst), so Welch's t-test at N=25 yields enormous t-statistics across essentially the whole non-degenerate parameter space. The metric is computed identically on pooled and single-subject data (each subject has all 16 trial types x 6 reps), so between-subject variance is well defined. It avoids hashing any sequence-valued column (no groupby/unique on ratings), and it is not trivially true for TTB: it requires subjects to actually follow the one-reason cascade against the cue majority on the majority of dissociation trials.

Source:
def metric(data: pd.DataFrame) -> float:
    # Dissociation score: on trials where Take-The-Best (top-validity
    # cascade) and Tallying (cue-count majority) make DIFFERENT modal
    # predictions, how often does the subject side with the TTB option?
    # TTB predicts > 0.5 on every such trial (constant sigmoid(beta)
    # confidence, including on tally-tie trials 7/8 where Tallying must
    # guess at exactly 50%). Tallying predicts < 0.5 on conflict trials
    # (majority option wins) and exactly 0.5 on tally ties. So the metric
    # is positive under TTB and negative under Tallying.
    V = np.array([0.95, 0.85, 0.65, 0.60, 0.50])
    order = np.argsort(-V, kind="stable")

    a_col = data["option_a_ratings"].tolist()
    b_col = data["option_b_ratings"].tolist()
    resp = data["response"].to_numpy()

    matches = []
    for i in range(len(a_col)):
        a = np.asarray(a_col[i], dtype=float)
        b = np.asarray(b_col[i], dtype=float)

        # TTB winner: first discriminating cue in validity order.
        ttb = -1
        for j in order:
            if a[j] > b[j]:
                ttb = 0
                break
            if b[j] > a[j]:
                ttb = 1
                break
        if ttb == -1:
            continue  # TTB guesses; not diagnostic

        # Tallying winner: strict cue-count majority.
        a_wins = int(np.sum(a > b))
        b_wins = int(np.sum(b > a))
        if a_wins > b_wins:
            tal = 0
        elif b_wins > a_wins:
            tal = 1
        else:
            tal = -1  # tally tie -> Tallying guesses

        # Keep only dissociation trials: the two theories' modal
        # predictions differ (tally tie, or tally majority against TTB).
        if tal == ttb:
            continue

        matches.append(1.0 if resp[i] == ttb else 0.0)

    if len(matches) == 0:
        return 0.0
    # Center at 0.5: positive => TTB-like, negative => Tallying-like.
    return float(np.mean(matches)) - 0.5

### RESULTS
- Predicted under pi_1 (simulated): 0.3480 (var=0.0087)
- Predicted under pi_2 (simulated): -0.2963 (var=0.0068)
- Observed on real data: 0.1580 (var=0.0120)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=8):** [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]

**Trial pairs (n=24):**
  trial 1: A=[0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1]
  trial 2: A=[0, 1, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 1, 1]
  trial 3: A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1, 1]
  trial 4: A=[1, 1, 0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 1]
  trial 5: A=[0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 1, 1]
  trial 6: A=[1, 1, 0, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 1, 1, 1]
  trial 7: A=[1, 0, 0, 1, 0, 1, 1, 1]  B=[1, 0, 0, 1, 1, 0, 0, 0]
  trial 8: A=[0, 1, 1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 1, 0, 0, 1]
  trial 9: A=[1, 1, 0, 0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 1, 0, 1, 1]
  trial 10: A=[1, 0, 0, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1, 1, 0, 0]
  trial 11: A=[1, 0, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 0, 0, 0]
  trial 12: A=[1, 0, 0, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  trial 13: A=[1, 1, 1, 0, 0, 0, 0, 1]  B=[1, 1, 0, 1, 1, 1, 1, 0]
  trial 14: A=[0, 0, 1, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 1, 1, 0, 0]
  trial 15: A=[1, 1, 1, 0, 0, 1, 1, 1]  B=[1, 1, 0, 1, 1, 0, 0, 0]
  trial 16: A=[0, 1, 1, 0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 1, 1, 1]
  trial 17: A=[1, 1, 0, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 0, 1, 1, 0]
  trial 18: A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 1, 0, 0]
  trial 19: A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]
  trial 20: A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  trial 21: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  trial 22: A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  trial 23: A=[1, 0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0, 1, 0]
  trial 24: A=[1, 0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1]

**Rationale:** This design dissociates Tallying from Take-The-Best (TTB) with a 3x3 factorial grid plus controls, using 8 cues with strictly descending, well-separated validities (.95 down to .60) so TTB's cascade order is unambiguous. GRID: In the 9 core cells (G1-G9), the first cue that discriminates is cue d in {1, 3, 5} (cues above d are tied), and that cue always favors B, so TTB always picks B with p = sigmoid(beta) -- a completely FLAT prediction across the entire grid, independent of both the tally and the discriminating cue's validity. Orthogonally, the total tally margin M in {+2, 0, -2} (from A's perspective) is varied by arranging the cues below d. Tallying predicts p(B) = sigmoid(-M*beta): at M = -2 both theories pick B but with different confidence (Tallying sigmoid(2*beta) vs TTB sigmoid(beta) -- an agreement-direction CONFIDENCE dissociation); at M = 0 Tallying must guess at exactly 50% while TTB is confident at sigmoid(beta); at M = +2 the theories conflict in direction (Tallying strongly favors A at sigmoid(2*beta), TTB favors B at sigmoid(beta)). Thus the signature is a 2D response surface: TTB predicts a flat plane (no effect of margin, no effect of cue depth), Tallying predicts a monotone gradient in margin only, crossing exactly 50% at M = 0. The depth dimension additionally tests whether confidence depends on the discriminating cue's validity (.95 vs .85 vs .75) -- both theories predict no effect, which rules out validity-weighted alternatives and verifies TTB's order-based (not magnitude-based) cascade; if choices track cue validity both target theories fail. Mirrored versions (swapping A and B) control for side bias and must be analyzed by 'cue-winner side' rather than option label. EXTREME CONFLICT (T2/T2m): cue 1 alone favors one option against all 7 other cues -- TTB stays at sigmoid(beta) while Tallying is near-certain (sigmoid(6*beta)), the largest possible confidence gap. ANCHORS: T1/T1m (all cues agree) check that subjects follow the evidence at all; T3/T4 (identical vectors) verify that both models' 50% guessing prediction holds and catch key-bias artifacts. 24 unique pairs x K = 4 repetitions = 96 trials exactly, giving 8 observations per grid cell (4 per unmirrored cell + 4 mirrored) for stable per-cell choice proportions. Predicted outcomes: if Tallying is correct, p(cue-winner) declines monotonically with margin (high at M=-2, exactly 50% at M=0, low at M=+2) with no effect of depth; if TTB is correct, p(cue-winner) is statistically indistinguishable across all 18 grid trials and both extreme-conflict trials, at a common sigmoid(beta) < 1.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



### METRIC
Rationale:
The design's core dissociation is the set of CONFLICT trials where the tally majority and the top discriminating cue point to opposite options: the five margin+2 grid cells (trials 1, 4, 7, 10, 13) and the two extreme-conflict trials (20, 21) where one option wins 7 of 8 cues against cue 1. On exactly these trials the two theories make opposite directional predictions: Tallying chooses the majority-winner with probability sigmoid(2*beta) (grid) to sigmoid(6*beta) (extreme), while TTB chooses the cue-winner with probability sigmoid(beta), i.e. the tally-winner only 1-sigmoid(beta) of the time. The metric is simply the empirical proportion of tally-winner choices on those 7 unique trials x 4 repetitions = 28 conflict observations per subject. This maximizes the mean gap (near 1 vs near 0 for any moderate beta, and still a sizeable gap even at small beta or with lapses, since the extreme-conflict trials alone span sigmoid(6*beta) vs 1-sigmoid(beta)) while keeping between-subject variance at binomial level over 28 trials (~p(1-p)/28), so 25 human subjects easily separate the theories via Welch's t-test. Trials where the heuristics agree (M=-2 cells, anchors) and tie trials are deliberately excluded: they carry no discriminative signal and would only dilute the contrast and add variance. The metric is a simple proportion, so it is well-defined and stable both on the pooled data and on each single subject's slice.

Source:
def metric(data: pd.DataFrame) -> float:
    """P(choose the TALLY winner | Tallying and Take-The-Best disagree).

    On every trial we compute (a) the tally winner (option winning more
    strict feature-wise comparisons) and (b) the TTB winner (winner on
    the first discriminating cue in descending-validity order, cues here
    are already ordered by validity). We keep only CONFLICT trials where
    both heuristics have a definite winner and they point to opposite
    options, and return the proportion of choices falling on the tally
    winner's side.

    Under Tallying this proportion is high (sigmoid(2*beta) on the
    margin-2 grid conflicts, sigmoid(6*beta) on the extreme-conflict
    trials where one option wins 7 of 8 cues); under TTB it is low
    (1 - sigmoid(beta)). Works on the pooled DataFrame and on any
    single subject's slice (each subject has 7 unique conflict trials
    x 4 reps = 28 conflict observations).
    """
    validities = np.array([0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60])
    cue_order = np.argsort(-validities, kind="stable")

    hits = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.shape != b.shape:
            continue

        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))

        # Tallying needs a strict majority; skip tied tallies.
        if a_wins == b_wins:
            continue
        tally_winner = 0 if a_wins > b_wins else 1

        # Take-The-Best winner: first discriminating cue in validity order.
        ttb_winner = None
        for j in cue_order:
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break
        if ttb_winner is None:
            continue

        # Keep only genuine conflicts between the two heuristics.
        if ttb_winner == tally_winner:
            continue

        resp = int(row["response"])
        hits.append(1.0 if resp == tally_winner else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))


### RESULTS
- Predicted under pi_1 (simulated): 0.1536 (var=0.0134)
- Predicted under pi_2 (simulated): 0.8664 (var=0.0110)
- Observed on real data: 0.1814 (var=0.0118)

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
  "interpretation": "Across both experiments, the observed data consistently fall on the Take-The-Best (pi_1) side of the dissociation, while Tallying (pi_2) is decisively refuted. In Experiment 1, the dissociation score (P(choose TTB option) - 0.5) was predicted to be +0.348 under TTB and -0.296 under Tallying; the observed value is +0.158 \u2014 positive, i.e. in TTB's direction, with a magnitude consistent with TTB at a moderate noise level. In Experiment 2, the proportion of tally-winner choices on conflict trials was predicted to be 0.866 under Tallying and 0.154 under TTB; the observed value is 0.181, almost exactly matching TTB's prediction and far from Tallying's. So pi_1 explains the sign and rough magnitude of both datasets with a single qualitative mechanism (one-reason, top-cue-driven choice), whereas pi_2 gets the sign wrong in both experiments \u2014 it is degenerate, not merely miscalibrated. That said, pi_1 is not a perfect fit: in Experiment 1 the observed effect (0.158) is less than half of TTB's simulated value (0.348), suggesting subjects follow the top cue less deterministically than a high-beta TTB \u2014 either more noise or graded, validity-weighted integration. The right move is therefore not to touch pi_1 (it is the task-invariant winner so far), but to replace pi_2 with a theory that is a genuine, stronger competitor to TTB: one that reproduces TTB-like directional behavior on both designs while predicting the graded, sub-maximal confidence actually observed. A power-weighted evidence integration model does this: each cue contributes a weight proportional to validity raised to a power gamma, summed into an evidence difference, passed through a softmax with lapse. At gamma = 1 it reduces to validity-weighted tallying; as gamma grows the top cues dominate and it asymptotically approaches TTB (e.g., with steeply separated validities like 0.95 vs 0.90..., a large gamma makes the top cue's weight exceed the sum of all others, reproducing the 0.18 tally-winner rate in Experiment 2), while intermediate gamma naturally yields the attenuated +0.158 dissociation score of Experiment 1. Unlike TTB it predicts graded confidence that varies with the full weighted-evidence margin, and unlike Tallying it privileges high-validity cues \u2014 exactly the dimension along which the current data discriminate.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace pi_2 (Tallying) with a 'power-weighted evidence integration' theory. Description: when comparing two options, people integrate all cues additively, but each cue's vote is weighted by w_j = v_j^gamma, where v_j is the cue's (stated) validity and gamma >= 0 is a steepness parameter capturing how strongly validity is over-weighted. The choice probability is a softmax over the weighted evidence difference sum_j w_j * sign(a_j - b_j), with inverse temperature beta and an independent lapse epsilon to uniform choice. gamma = 0 recovers equal-weight Tallying; gamma = 1 gives validity-proportional weighting; gamma -> infinity converges to Take-The-Best behavior (the top cue's weight overwhelms all others). This single additional parameter lets the theory span the space between the two existing heuristics and, critically, fit the observed pattern: strongly top-cue-driven choices (Experiment 2: ~0.18 tally-winner rate) with attenuated, graded confidence (Experiment 1: +0.158 dissociation score, well below pure TTB's +0.348). Suggested parameter ranges: gamma in [0, 30] (log-scale search), beta in [0.1, 5], epsilon in [0, 0.3]. The predict function should compute the weighted evidence difference from the ratings and validities, apply the softmax/lapse as in the current implementations, and the policy remains a categorical sample over the two options. Future experiments should then dissociate this theory from TTB by pitting a steeply dominant top cue against a large weighted-vs-unweighted cue block, and by probing confidence gradients across tally margins at fixed top-cue structure."
}
```

## Usage

```json
{
  "input_tokens": 7709,
  "output_tokens": 2769,
  "reasoning_tokens": 1828,
  "provider": "Sail Research"
}
```
