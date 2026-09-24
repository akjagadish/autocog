# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3_2" and "pi_2") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3_2" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_2" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3_2
Reversed-Hierarchy Frugality (misbinding TTB). People are one-reason decision makers: they consult cues one at a time and let the first strictly discriminating cue decide, ignoring all lower cues and all magnitude information. However, the subjective cue hierarchy is ANTI-correlated with the instructed validities: subjects misbind the communicated validity list to feature positions (binding the highest validity to the last-presented feature), so the cascade effectively runs in ascending true validity (kappa = 1, full misbinding). When no cue discriminates, the model guesses uniformly. Response noise enters through a softmax with inverse temperature beta over the binary winner score, plus an independent lapse epsilon mixing in a uniform choice. Because no integration across cues occurs, the model predicts a flat choice profile across tally margins (zero psychometric slope), in direct opposition to Tallying's graded curve. The noise regime is moderate: the cascade-follow probability p = (1-eps)*sigmoid(beta) + eps/2 sits near 0.60, matching the human follow rate implied by both experiments.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Reversed-Hierarchy Frugality (misbinding TTB). One-reason
    # decision making: cues are consulted one at a time and the first
    # strictly discriminating cue decides. The subjective hierarchy is
    # ANTI-correlated with the instructed validities (kappa = 1: the
    # communicated validity list is misbound to feature positions,
    # highest validity bound to the last-presented feature), so the
    # cascade runs in ascending true validity. No integration across
    # cues occurs -> flat psychometric profile across tally margins.
    # Noise: softmax (inverse temperature beta) over the binary winner
    # score, plus an independent lapse epsilon mixing in a uniform
    # choice. History is ignored (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    # Subjective validities: mixture of the instructed vector and its
    # end-to-end reversal. kappa = 1 -> fully reversed hierarchy
    # (full positional misbinding), per the theory's fixed claim.
    kappa = float(parameters["kappa"])
    w = (1.0 - kappa) * val + kappa * val[::-1]

    # Descending SUBJECTIVE validity; stable argsort so ties break
    # toward the earlier feature index.
    cue_order = np.argsort(-w, kind="stable")

    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    n_opts = 2
    if winner is None:
        # No discriminating cue — pure guess (lapse mixing leaves a
        # uniform unchanged).
        return np.ones(n_opts) / float(n_opts)

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Numerically stable softmax over the binary winner score: the
    # winner's probability is sigmoid(beta). The implied cascade-follow
    # probability is p = (1-eps)*sigmoid(beta) + eps/2, which over the
    # sampled (beta, epsilon) ranges averages ~0.68 with per-subject
    # values spanning ~[0.62, 0.75].
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / float(n_opts))

`policy(probs) -> int`:
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
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


## EXPERIMENT 1 (proposed by pi_3_2)

### DESIGN
**Validities (n_features=6):** [0.6, 0.95, 0.6, 0.85, 0.65, 0.9]

**Trial pairs (n=16):**
  trial 1: A=[0, 0, 0, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1]
  trial 2: A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 1, 0, 0]
  trial 3: A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 5: A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 6: A=[1, 1, 0, 1, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  trial 7: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 8: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  trial 9: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 10: A=[0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 11: A=[0, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 13: A=[1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 14: A=[0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 15: A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 16: A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]

**Rationale:** GOAL: Quantitatively dissociate Reversed-Hierarchy Frugality (misbinding TTB, kappa=1) from Tallying via a validity arrangement whose MISBOUND hierarchy consults the three WEAKEST experts before any of the three strongest — a 'weak-expert dictatorship' schedule no prior design used. SETUP: 6 features, validities [0.60, 0.95, 0.55, 0.85, 0.65, 0.90] (f0..f5). The misbound weights w = val reversed = [0.90, 0.65, 0.85, 0.55, 0.95, 0.60], so the misbinding cascade consults f4 (advertised 65%) -> f0 (60%) -> f2 (55%) -> f1 (95%) -> f5 (90%) -> f3 (85%): the three lowest-validity experts occupy the top three rungs of the subjective hierarchy, and the 95% expert is not consulted until depth 4. Tallying ignores validities entirely: p(A) = (1-eps)*sigmoid(beta*m) + eps/2, depending only on the tally margin m. The misbinding model emits a constant p(decider winner) = (1-eps)*sigmoid(beta)+eps/2, which over its parameter ranges (beta in [0.6,1.2], eps in [0.08,0.18]) lies in [0.62, 0.80] — a hard ceiling. SIGNATURE 1 — WEAK-EXPERT DICTATORSHIP SWEEP (T1-T5): the misbound top cue f4 (a 65% expert) always decides for A while the margin sweeps -3, -1, 0, +1, +3. Misbinding TTB predicts a FLAT p(A) in [0.62, 0.80] on all five; Tallying predicts a strictly graded monotone curve passing through EXACTLY 0.5 at T3 (m=0, parameter-free knife-edge). Single-cell non-overlap at T1 (m=-3): Tallying is mathematically <= 0.463 for ANY (beta, eps) in its ranges (max at beta=0.1, eps=0.5), while misbinding is >= 0.62 — disjoint prediction intervals. T6-T8 are the A/B mirrors (decider f4 -> B at m=+3, 0, -3): at T6 the four STRONGEST experts (95%, 90%, 85%, 60%) all favor A with margin +3, yet misbinding follows the lone 65% expert to B (p(A) in [0.20, 0.38]) while Tallying is >= 0.537 — again disjoint. T5 provides the ceiling test: unanimous-ish margin +3 where Tallying can reach ~0.97 but misbinding is capped at 0.80; human consistency above ~0.82 falsifies the advocated theory at its parameter boundary. SIGNATURE 2 — ISO-MARGIN OPPOSITE-DECIDER PAIRS (parameter-free for Tallying): (T4, T12) at m=+1, (T3, T7) at m=0, (T5, T6) at m=+3 — identical margins, opposite f4 deciders. Tallying predicts IDENTICAL p(A) within each pair for ANY parameters; misbinding predicts a ~0.70 vs ~0.30 split within each pair, straddling 0.5. SIGNATURE 3 — KNIFE-EDGE LADDER AT m=0 WITH VARYING CASCADE DEPTH (T3, T7, T9, T10, T11): five margin-0 cells pinning Tallying to EXACTLY 0.50 regardless of (beta, epsilon) — any consistent preference on any rung falsifies Tallying — while the misbound decider sits at depth 1 (f4, T3/T7), depth 2 (f0, T9), depth 3 (f2, T10), and depth 4 (f1, T11), in both directions. Misbinding predicts the same ~0.70/0.30 split at every depth: depth invariance. SIGNATURE 4 — DECIDER-VALIDITY INVARIANCE AT m=+1 (T4, T13, T14, T15): four margin-+1 cells whose misbound deciders are advertised at 65% (f4), 60% (f0), 55% (f2), and 95% (f1). Misbinding predicts the SAME follow rate on all four — the follow probability cannot depend on the decider's advertised accuracy; any monotone effect of decider validity (e.g., subjects following the 95% expert in T15 more than the 55% expert in T14) falsifies the kappa=1 misbinding claim. These are agreement cells for Tallying (identical margin), so they isolate the misbinding model's internal structure. GLOBAL PROFILES: misbinding TTB emits a two-level step function over the 16 cells (~[0.62,0.80] toward the misbound decider on all 15 discriminating trials, exactly 0.5 on the full tie), invariant to margin, cascade depth, and decider validity; Tallying emits a monotone psychometric function of margin with exact 0.5 on all five margin-0 cells and exact equality within margin levels (T4=T12=T13=T14=T15; T5=T6; T1=T8). The profiles are structurally disjoint — e.g., misbinding says p(T1)=p(T5) (both decider f4->A) while Tallying says p(T1) <= 0.463 < 0.537 <= p(T5); misbinding says p(T3)=p(T4) while Tallying says p(T3)=0.5 < p(T4) — so model comparison on the full 16-cell response profile separates the theories even after fitting beta and epsilon to each. T16 (full tie) anchors lapse rate and side bias. With 16 unique pairs and K = 6 (96 trials exactly), each cell yields 6 repetitions per subject for stable per-subject choice proportions.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
The metric exploits a parameter-free knife-edge of Tallying: its choice probability depends only on the tally margin m via a symmetric sigmoid, p(A|-m) = 1 - p(A|m). I select the 10 trial cells whose tally margins are balanced across the misbinding cascade's decider direction: cascade-A cells {T1 (m=-3), T3, T9, T10 (m=0), T5 (m=+3)} and cascade-B cells {T8 (m=-3), T7, T11, T12 (m=0), T6 (m=+3)}. Within each group the margins are exactly mirrored, so under Tallying both P(choose A) group means equal 0.5 for ANY (beta, epsilon) in its ranges — the metric is identically 0 in expectation, with only binomial sampling noise (sd ~0.13 per subject over 30 trials per side). Under the advocated misbinding TTB, every one of these cells is decided by the first cue of the reversed hierarchy, so P(choose A | cascade-A) = p_follow and P(choose A | cascade-B) = 1 - p_follow, giving metric = 2*p_follow - 1, which lies in ~[0.27, 0.44] across the advocated beta/epsilon ranges (mean ~0.36) and is flat in margin. The between-theory mean gap (~0.36) is large relative to per-subject noise (~0.12-0.13 sd), so with N=25 subjects Welch's t is expected around 8-11, far below alpha=0.01. The metric works identically on a single subject's slice (each subject completes all 16 cells x 6 reps, so both groups are always populated), giving a well-defined between-subject variance. Unlike an overall agreement rate (which extreme-beta Tallying could push toward 0.6, eroding the gap) or a single-cell contrast (noisy), this contrast cannot be attenuated by any Tallying parameterization because Tallying's prediction is exactly 0 by symmetry.

Source:
def metric(data: pd.DataFrame) -> float:
    # Knife-edge decider-direction contrast at balanced tally margins.
    #
    # For every trial we compute (a) the tally margin m = (#features A wins)
    # - (#features B wins), and (b) the winner implied by the advocated
    # misbinding-TTB cascade (subjective hierarchy = instructed validities
    # reversed end-to-end: cue order f4 -> f0 -> f2 -> f1 -> f3 -> f5).
    #
    # We keep only trials with |m| in {0, 3} (excluding the |m|=1 cells,
    # whose decider directions are unbalanced across margins) and where the
    # cascade has a discriminating cue. In this subset the misbinding-A
    # cells are {T1 (m=-3), T3 (m=0), T5 (m=+3), T9 (m=0), T10 (m=0)} and
    # the misbinding-B cells are {T6 (m=+3), T7 (m=0), T8 (m=-3), T11 (m=0),
    # T12 (m=0)} -- perfectly mirror-balanced in margin.
    #
    # Metric = P(choose A | cascade says A) - P(choose A | cascade says B).
    #
    # Under Tallying, p(A) = (1-eps)*sigmoid(beta*m) + eps/2, which is
    # exactly symmetric in m. Within each group the margins are balanced
    # (-3/+3 pairs and 0s), so BOTH group means are exactly 0.5 for ANY
    # (beta, epsilon) -- the metric is pinned to 0 parameter-free.
    #
    # Under misbinding TTB, p(A) = p_follow on cascade-A cells and
    # 1 - p_follow on cascade-B cells, so the metric equals 2*p_follow - 1,
    # which over the advocated parameter ranges lies in ~[0.27, 0.44]
    # (mean ~0.36) regardless of margin.

    df = data.reset_index(drop=True)
    a_lists = df['option_a_ratings'].tolist()
    b_lists = df['option_b_ratings'].tolist()
    resp = df['response'].to_numpy()

    # Design validities and the kappa=1 reversed subjective hierarchy.
    val = np.array([0.6, 0.95, 0.6, 0.85, 0.65, 0.9])
    w = val[::-1]
    cue_order = np.argsort(-w, kind='stable').tolist()  # -> [4, 0, 2, 1, 3, 5]

    a_side = []  # trials where the misbound cascade favors A
    b_side = []  # trials where the misbound cascade favors B

    for i in range(len(df)):
        av = [int(v) for v in a_lists[i]]
        bv = [int(v) for v in b_lists[i]]
        n = len(av)

        margin = (sum(1 for j in range(n) if av[j] > bv[j])
                  - sum(1 for j in range(n) if bv[j] > av[j]))
        if abs(margin) == 1:
            # Unbalanced decider directions at |m|=1; excluded so the
            # Tallying expectation stays exactly 0.
            continue

        winner = -1
        for j in cue_order:
            if j >= n:
                continue
            if av[j] > bv[j]:
                winner = 0
                break
            if bv[j] > av[j]:
                winner = 1
                break
        if winner == -1:
            continue  # full-tie trial (T16): no discriminating cue

        chose_a = 1.0 if resp[i] == 0 else 0.0
        if winner == 0:
            a_side.append(chose_a)
        else:
            b_side.append(chose_a)

    pa = float(np.mean(a_side)) if len(a_side) > 0 else 0.5
    pb = float(np.mean(b_side)) if len(b_side) > 0 else 0.5
    return float(pa - pb)

### RESULTS
- Predicted under pi_3_2 (simulated): 0.3613 (var=0.0180)
- Predicted under pi_2 (simulated): -0.0100 (var=0.0118)
- Observed on real data: -0.3107 (var=0.0225)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=7):** [0.6, 0.9, 0.8, 0.95, 0.65, 0.75, 0.6]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 1, 1, 1, 1, 0]  B=[0, 0, 1, 0, 1, 0, 0]
  trial 2: A=[1, 0, 1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1, 0]
  trial 3: A=[1, 1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 1, 1]
  trial 4: A=[1, 0, 1, 0, 1, 0, 1]  B=[0, 0, 1, 1, 1, 1, 1]
  trial 5: A=[1, 0, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1, 0]
  trial 6: A=[0, 0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 1, 1, 1, 0]
  trial 7: A=[0, 0, 1, 0, 1, 1, 0]  B=[1, 0, 1, 1, 1, 0, 0]
  trial 8: A=[0, 0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0, 1]
  trial 9: A=[0, 0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0, 1]
  trial 10: A=[0, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1, 0]
  trial 11: A=[1, 0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1, 0]
  trial 12: A=[0, 1, 1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0, 1]
  trial 13: A=[1, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 0]
  trial 14: A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0]
  trial 15: A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  trial 16: A=[0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0]

**Rationale:** GOAL: Quantitatively dissociate Tallying (advocated) from Reversed-Hierarchy Frugality (misbinding TTB, kappa=1) via a design organized around a NOVEL structural centerpiece: a balanced, orthogonal 2x5 FACTORIAL GRID crossing tally margin with misbound-decider direction, plus a 'sandwich-scrambled' validity arrangement whose misbound hierarchy is non-monotone in advertised accuracy. SETUP: 7 features (no prior design used more than 6, giving a wider margin range up to +/-5) with validities [0.60, 0.90, 0.55, 0.80, 0.65, 0.75, 0.95] (f0..f6). The misbound weights w = val reversed = [0.95, 0.75, 0.65, 0.80, 0.55, 0.90, 0.60], so the misbinding cascade consults f0 -> f5 -> f3 -> f1 -> f2 -> f6 -> f4, i.e., advertised accuracies along the subjective hierarchy run 60% -> 75% -> 80% -> 90% -> 55% -> 95% -> 65%: a NON-MONOTONE sandwich in which the first-listed 60% expert is trusted most, the 95% expert is consulted SIXTH (next to last), and the 55% expert sits above it at depth 5. Tallying ignores validities entirely: p(A) = (1-eps)*sigmoid(beta*m) + eps/2, depending only on margin m. Misbinding TTB emits a constant follow probability p = (1-eps)*sigmoid(beta)+eps/2, which over its parameter ranges (beta in [0.6,1.2], eps in [0.08,0.18]) lies in [0.62, 0.75] toward the first discriminating cue in the order f0,f5,f3,f1,f2,f6,f4. SIGNATURE 1 - THE ORTHOGONAL FACTORIAL GRID (T1-T10): the misbound TOP cue f0 (advertised only 60%) always decides, toward A in T1-T5 and toward B in T6-T10 (exact A/B mirrors), while the margin sweeps -3, -1, 0, +1, +3. This is a fully balanced 2 (decider direction) x 5 (margin) design, and the two theories predict TRANSPOSED main-effect structures in the marginal means: Tallying predicts a NULL decider-row effect (both row means exactly 0.5, since margins are symmetric within each row) with a strictly graded margin-column effect passing through EXACTLY 0.5 at m=0; misbinding TTB predicts a strong decider-row effect (~0.70 vs ~0.30) with a NULL margin-column effect (every column mean exactly 0.5). A 2x5 ANOVA or model comparison on the grid cleanly awards the data to one theory: any significant margin effect with null decider effect falsifies misbinding; any significant decider effect with null margin effect falsifies Tallying. SIGNATURE 2 - SINGLE-CELL DISJOINT PREDICTION INTERVALS AT EXTREME MARGINS (T14, T15): on T14, ALL six other experts - including the 95% expert - favor A (margin +5) and only the first-listed 60% expert favors B. Misbinding follows the 60% expert to B, predicting p(A) <= 0.38; Tallying predicts p(A) = (1-eps)*sigmoid(5*beta)+eps/2 >= 0.56 even at its WORST parameters (beta=0.1, eps=0.5) - the intervals are disjoint, so this one cell alone falsifies one theory. T15 is the exact mirror (margin -5, lone 60% expert favors A): Tallying <= 0.44, misbinding >= 0.62. These cells also separate misbinding from STANDARD TTB (which follows the 95% expert to A on T14), making the design adversarial to the whole TTB family. SIGNATURE 3 - PARAMETER-FREE KNIFE EDGES WITH COMPOSITION CONTROL (T3, T8, T11): three margin-0 cells - a 2-2 tally with decider f0->A (T3), its mirror with decider f0->B (T8), and a sparse 1-1 tally with decider f0->A (T11). Tallying is pinned to EXACTLY 0.50 on all three regardless of (beta, eps) and regardless of tally composition; misbinding predicts ~0.70/~0.30 splits on all three, invariant to whether 2 or 6 cues sit on each side. SIGNATURE 4 - DECIDER-VALIDITY AND DEPTH INVARIANCE AT FIXED MARGIN (T2, T12, T13): three margin-+1 cells whose misbound deciders are the 60% first-listed expert at depth 1 (T2), the 90% expert at depth 4 (T12), and the 95% expert at depth 6 (T13, where it is the ONLY discriminating cue). Misbinding predicts the SAME follow rate on all three - the follow probability cannot depend on the decider's advertised accuracy or cascade depth; any monotone validity gradient (e.g., following the lone 95% expert more than the 60% expert) falsifies the kappa=1 misbinding claim. Tallying also predicts identical values here (same margin), so these cells isolate the competitor's internal structure. GLOBAL PROFILES: misbinding TTB emits a two-level step function over the 16 cells (~[0.62,0.75] toward the misbound decider on all 15 discriminating trials, exactly 0.5 on the full tie), invariant to margin, tally composition, cascade depth, and decider validity; Tallying emits a monotone psychometric function of margin with exact 0.5 on the three margin-0 cells, exact equality within margin levels (T1=T10-margin... specifically T1=T10? No: T1 (m=+3, dec->A) = T10 (m=+3, dec->B) in p(A); T2=T9=T12=T13; T5=T6; T14 highest, T15 lowest), and values far outside the misbinding cap on T14/T15. The profiles are structurally disjoint - e.g., misbinding says p(T15)=p(T1) (both decider->A) while Tallying says p(T15) <= 0.44 < 0.56 <= p(T1); misbinding says p(T3)=p(T2) while Tallying says p(T3)=0.5 < p(T2) - so model comparison on the full 16-cell response profile separates the theories even after fitting beta and epsilon to each. T16 (full tie) anchors lapse rate and side bias (both theories: exactly 0.5). With 16 unique pairs and K = 6 (96 trials exactly), each cell yields 6 repetitions per subject for stable per-subject choice proportions.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
The design contains exactly four tally-tied (margin-0) cells: T3, T8, T11, and the full tie T16. Tallying's softmax over equal counts is EXACTLY uniform for any (beta, epsilon), so its probability of choosing in the direction of any particular cue on a tied trial is exactly 0.5 — a parameter-free knife-edge. The reversed-hierarchy one-reason model instead always follows the first discriminating cue in its misbound order (f3, f5, f4, f1, f2, f0, f6): on T3 and T11 the decider points to B, on T8 to A, so its tied-trial follow rate is the cascade-follow probability p = (1-eps)*sigmoid(beta) + eps/2, which lies in [0.62, 0.75] over its parameter ranges. The metric pools all tied trials that possess a discriminating cue (3 cells x 6 reps = 18 trials per subject), codes each choice as 'followed the misbound decider' or not, and returns the follow rate minus 0.5. Expected values: Tallying ~ 0.00 (per-subject variance only binomial, ~0.25/18 = 0.014); misbinding ~ +0.18 (variance ~0.013). With N=25 the Welch t-statistic is ~5.5 (p ~ 1e-6) at typical misbinding parameters, and still ~3.6 (p < 0.001) at its worst-case corner (beta=0.6, eps=0.18), safely below alpha=0.01. The metric avoids all parameter-dependent contrasts (e.g., psychometric slope over margins), because Tallying's beta range [0.1, 20] is wide and a slope-based metric could flip sign or collapse; the tied-trial contrast is the unique signature that is exactly zero under Tallying for every parameter setting, while strictly positive under one-reason deciding. It works identically on the pooled DataFrame and on each single subject's 96-trial slice, so the between-subject variance is well defined.

Source:
import numpy as np
import pandas as pd

# Design-time constants for this experiment.
VALIDITIES = np.array([0.6, 0.9, 0.8, 0.95, 0.65, 0.75, 0.6])


def metric(data: pd.DataFrame) -> float:
    """Cue-following deviation from chance on tally-tied (margin-0) trials.

    For every trial whose tally margin is zero (A and B win on equal numbers
    of features), find the first strictly discriminating cue under the
    reversed-hierarchy cue order (the competing theory's subjective
    hierarchy). Record whether the subject's choice followed that cue.
    Tallying predicts exactly chance (0 deviation) on every tied trial,
    for ANY (beta, epsilon). One-reason misbinding predicts a systematic
    follow rate of p = (1-eps)*sigmoid(beta) + eps/2, i.e. a deviation of
    roughly +0.12 to +0.25 above chance.
    """
    # Reversed-hierarchy subjective weights and cue order (stable ties).
    w = VALIDITIES[::-1].astype(float)
    cue_order = np.argsort(-w, kind="stable")

    a_col = data["option_a_ratings"].tolist()
    b_col = data["option_b_ratings"].tolist()
    resp_col = data["response"].tolist()

    follows = []
    for a_raw, b_raw, r_raw in zip(a_col, b_col, resp_col):
        a = np.asarray(a_raw, dtype=float)
        b = np.asarray(b_raw, dtype=float)
        if a.shape != b.shape or a.ndim != 1:
            continue

        # Tally margin: number of features A wins minus number B wins.
        margin = int(np.sum(a > b) - np.sum(b > a))
        if margin != 0:
            # Only tied tallies carry the parameter-free signature.
            continue

        # First discriminating cue under the reversed hierarchy.
        decider = None
        for j in cue_order:
            if a[j] > b[j]:
                decider = 0  # cue favors A
                break
            if b[j] > a[j]:
                decider = 1  # cue favors B
                break
        if decider is None:
            # Full tie (T16): both theories say 0.5; no information.
            continue

        resp = int(r_raw)
        follows.append(1.0 if resp == decider else 0.0)

    if len(follows) == 0:
        return 0.0

    # Deviation of the tied-trial cue-follow rate from chance.
    return float(np.mean(follows)) - 0.5

### RESULTS
- Predicted under pi_3_2 (simulated): 0.1878 (var=0.0164)
- Predicted under pi_2 (simulated): 0.0011 (var=0.0175)
- Observed on real data: -0.2889 (var=0.0205)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=float)
    cue_order = np.argsort(-validities, kind="stable")

    margins = []
    matches = []
    for a_cell, b_cell, resp in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"],
    ):
        a = np.asarray(a_cell, dtype=float).ravel()
        b = np.asarray(b_cell, dtype=float).ravel()
        if a.shape[0] != b.shape[0] or a.shape[0] != validities.shape[0]:
            continue
        diff = a - b

        # Take-The-Best cascade: first discriminating cue in validity order.
        ttb_winner = -1
        for j in cue_order:
            if diff[j] > 0:
                ttb_winner = 0
                break
            if diff[j] < 0:
                ttb_winner = 1
                break
        if ttb_winner < 0:
            # Full tie (trial 13): no TTB winner, uninformative for both theories.
            continue

        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        # Tally margin of the TTB-favored option (positive = tally agrees with TTB).
        if ttb_winner == 0:
            margin = a_wins - b_wins
        else:
            margin = b_wins - a_wins

        r = int(resp)
        match = 1.0 if r == ttb_winner else 0.0
        margins.append(margin)
        matches.append(match)

    if not margins:
        return 0.0

    x = np.asarray(margins, dtype=float)
    y = np.asarray(matches, dtype=float)

    # (1) LEVEL: P(choosing the TTB-favored option) on trials where the
    # tally does NOT favor it (tally majority against, or exact tally tie).
    low = x <= 0
    level = float(np.mean(y[low])) if np.any(low) else 0.0

    # (2) SLOPE: OLS slope of the match indicator on the tally margin,
    # across all trials with a TTB winner (margins -3, -2, -1, 0, +1).
    vx = float(np.var(x))
    if len(x) > 1 and vx > 0:
        slope = float(np.mean((x - x.mean()) * (y - y.mean())) / vx)
    else:
        slope = 0.0

    return level - slope
```

**Observed (real) value:** 0.3633 (var=0.0060)
**Predicted under pi_3_2:** 0.3314 (var=0.0076)
**Predicted under pi_2:** 0.0323 (var=0.0195)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]

**Metric**
```python
import numpy as np
import pandas as pd


def _to_matrix(col):
    # col: pandas Series whose cells are lists/tuples/arrays of binary ratings
    return np.array([list(r) for r in col], dtype=float)


def metric(data: pd.DataFrame) -> float:
    if data is None or len(data) == 0:
        return 0.0

    A = _to_matrix(data['option_a_ratings'])
    B = _to_matrix(data['option_b_ratings'])
    if A.ndim != 2 or A.shape != B.shape or A.shape[0] == 0:
        return 0.0

    # Tally margin: number of features where A beats B minus number where B beats A.
    margin = (A > B).sum(axis=1) - (B > A).sum(axis=1)

    # Signed choice: +1 if the subject chose A (response == 0), -1 if B (response == 1).
    resp = pd.to_numeric(data['response'], errors='coerce').to_numpy(dtype=float)
    s = 1.0 - 2.0 * resp

    # Alignment between the signed choice and the tally margin.
    # Tallying: E[s | margin] = (1-eps) * tanh(beta*margin/2)  =>  E[s*margin] > 0,
    # scaling like beta * E[margin^2] / 2 for small beta and saturating at mean(|margin|)
    # for large beta (max value 28/16 = 1.75 on this design).
    # TTB: E[s | trial] = (1-eps)*tanh(beta/2)*ttb_sign, and on this stimulus set
    # sum over the 16 trial types of ttb_sign * margin = -2 (mean -0.125), so the
    # TTB value is slightly NEGATIVE for any beta > 0.
    return float(np.mean(s * margin))
```

**Observed (real) value:** 0.0358 (var=0.0181)
**Predicted under pi_3_2:** 0.0708 (var=0.0422)
**Predicted under pi_2:** 1.3208 (var=0.1532)

### Experiment 5
**Design**
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    # Reversed-cascade follow rate (centered).
    #
    # For every trial we determine the winner implied by a one-reason
    # cascade that consults cues in ASCENDING true validity (i.e., the
    # 0.5-validity expert first, ..., the 0.9-validity expert last) --
    # exactly the subjective hierarchy implied by full misbinding
    # (kappa = 1). Full-tie trials carry no directional prediction and
    # are dropped. The metric is the signed follow rate:
    #   +1  -> subject always picks the reversed-cascade winner
    #    0  -> chance
    #   -1  -> subject always picks the opposite option
    #
    # Predicted values:
    #   Misbinding TTB: p(follow) = (1-eps)*sigmoid(beta) + eps/2 ~ 0.60
    #                   on EVERY discriminating trial  ->  metric ~ +0.21
    #   Tallying:       on the 7 trials where the tally winner OPPOSES
    #                   the lowest-validity cue (trials 1,2,3,5,6,7,15)
    #                   p(follow) < 0.5 for ANY beta > 0; on the 5
    #                   zero-margin trials (4,8,11,12,13) p(follow)
    #                   = 0.5 EXACTLY for any (beta, eps); only the 3
    #                   agreeing trials (9,10,14) push above 0.5.
    #                   Hence the Tallying metric is strictly NEGATIVE
    #                   for every parameterization in its ranges
    #                   (approaching 0 only as beta->0 or eps->0.5).
    validities = np.array([0.90, 0.80, 0.70, 0.60, 0.50])
    # Ascending true validity == descending subjective validity under
    # full misbinding: consult index 4 first, then 3, 2, 1, 0.
    cue_order = np.argsort(validities, kind="stable")

    n_used = 0
    follow = 0
    for a_raw, b_raw, r in zip(data["option_a_ratings"],
                               data["option_b_ratings"],
                               data["response"]):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        if a.shape != b.shape or a.size != validities.size:
            continue
        winner = None
        for j in cue_order:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
        if winner is None:
            # Full tie (trial 16): no directional prediction.
            continue
        n_used += 1
        if int(r) == winner:
            follow += 1

    if n_used == 0:
        return 0.0
    return 2.0 * (follow / n_used) - 1.0
```

**Observed (real) value:** 0.4533 (var=0.0207)
**Predicted under pi_3_2:** 0.3960 (var=0.0138)
**Predicted under pi_2:** -0.1742 (var=0.0091)

### Experiment 6
**Design**
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 1, 1]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Contrast metric built from two parameter-free anchors:
    #
    # (1) Margin-0 cells with a reversed-cascade decider (T5, T6, T11, T16).
    #     Tallying predicts p(A) = 0.5 EXACTLY on any tie trial, for any
    #     (beta, epsilon), so the signed follow-the-decider score has
    #     expectation exactly 0. The reversed-hierarchy model follows its
    #     decider with q = (1-eps)*sigmoid(beta) + eps/2 in [0.57, 0.64],
    #     so its expected score is 2q-1 in [0.14, 0.28].
    #
    # (2) Disagreement cells (T1, T2, T10, T14): nonzero tally margin whose
    #     sign OPPOSES the reversed decider. Tallying follows the margin
    #     with p >= 0.5 for any beta > 0 (expected signed score 2p-1 >= 0);
    #     the reversed model follows the decider, i.e. follows the tally
    #     with 1-q <= 0.43 (expected signed score 1-2q < 0).
    #
    # metric = mean(follow-decider score on margin-0 cells)
    #        - mean(follow-tally score on disagreement cells)
    #
    # Tallying: 0 - (2p-1) <= 0 (typically around -0.4 to -0.9).
    # Reversed: (2q-1) - (1-2q) = 4q-2 in [0.28, 0.56] > 0.
    # The two theories land on opposite sides of zero with a large gap.
    s1 = []
    s2 = []
    n_rows = len(data)
    for i in range(n_rows):
        row = data.iloc[i]
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        n = len(a)

        # Tally margin: strict feature-wise wins for A minus for B.
        margin = 0
        for j in range(n):
            if a[j] > b[j]:
                margin += 1
            elif b[j] > a[j]:
                margin -= 1

        # Reversed-hierarchy decider: first strictly discriminating
        # feature scanning from the LAST index down (ascending true
        # validity, since validities descend by index).
        dec = 0
        for j in range(n - 1, -1, -1):
            if a[j] > b[j]:
                dec = 1
                break
            if b[j] > a[j]:
                dec = -1
                break
        if dec == 0:
            # Identical options: no decider, no information.
            continue

        resp = int(row["response"])
        chose_a = (resp == 0)

        if margin == 0:
            # Knife-edge cell: tally is exactly 0.5 under Tallying.
            follow_dec = (dec == 1 and chose_a) or (dec == -1 and not chose_a)
            s1.append(1.0 if follow_dec else -1.0)
        elif (margin > 0 and dec == -1) or (margin < 0 and dec == 1):
            # Disagreement cell: tally direction opposes the decider.
            follow_tally = (margin > 0 and chose_a) or (margin < 0 and not chose_a)
            s2.append(1.0 if follow_tally else -1.0)

    m1 = float(np.mean(s1)) if len(s1) > 0 else 0.0
    m2 = float(np.mean(s2)) if len(s2) > 0 else 0.0
    return m1 - m2
```

**Observed (real) value:** 0.7787 (var=0.0600)
**Predicted under pi_3_2:** 0.6893 (var=0.0735)
**Predicted under pi_2:** -0.6893 (var=0.0556)

### Experiment 7
**Design**
  A=[0, 0, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Decider-follow rate on the margin-0 DISCRIMINATING trials (the
    # "subjective-hierarchy ladder": T4, T7-T11 in the design).
    # On these trials the tally is exactly balanced, so Tallying is
    # pinned to p = 0.50 toward either side for ANY (beta, epsilon)
    # (softmax over equal tallies is uniform, lapse is uniform).
    # Misbinding TTB instead follows its (reversed-hierarchy) first
    # discriminating cue with p = (1-eps)*sigmoid(beta) + eps/2,
    # which is >= 0.63 over the stated parameter ranges. The metric
    # is the proportion of choices that agree with the misbound
    # decider on exactly those trials: ~0.71 under misbinding TTB,
    # ~0.50 under Tallying.
    val = np.array([0.7, 0.9, 0.6, 0.95, 0.75, 0.6], dtype=float)
    # kappa = 1 misbinding: subjective weights are the validity list
    # bound in reverse to feature positions.
    w = val[::-1]
    cue_order = np.argsort(-w, kind="stable")  # f2, f4, f1, f5, f3, f0

    a = np.array([list(x) for x in data['option_a_ratings']], dtype=float)
    b = np.array([list(x) for x in data['option_b_ratings']], dtype=float)
    resp = np.asarray(data['response'], dtype=int).ravel()

    if a.shape[0] == 0:
        return 0.5

    a_wins = (a > b).sum(axis=1)
    b_wins = (b > a).sum(axis=1)

    # Margin-0 trials that still have at least one discriminating cue
    # (excludes the full tie T16, where both theories say 0.5).
    mask = (a_wins == b_wins) & ((a_wins + b_wins) > 0)
    idx = np.where(mask)[0]

    if len(idx) == 0:
        return 0.5

    follow = []
    for i in idx:
        decider = None
        for j in cue_order:
            if a[i, j] > b[i, j]:
                decider = 0
                break
            if b[i, j] > a[i, j]:
                decider = 1
                break
        if decider is None:
            continue  # shouldn't happen on discriminating trials
        follow.append(1 if resp[i] == decider else 0)

    if len(follow) == 0:
        return 0.5
    return float(np.mean(follow))
```

**Observed (real) value:** 0.6133 (var=0.0052)
**Predicted under pi_3_2:** 0.6633 (var=0.0068)
**Predicted under pi_2:** 0.5072 (var=0.0084)

### Experiment 8
**Design**
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Knife-edge signed contrast on the four margin-0 discriminating cells.

    Tallying predicts p(A) = 0.50 EXACTLY on every margin-0 cell, for any
    (beta, epsilon): equal tallies give an exactly uniform softmax. The
    reversed-hierarchy one-reason model, in contrast, lets a single cue
    decide these same cells, producing a systematic ~0.70/0.30 split whose
    sign is set by which option wins on the cascade's first cue.

    Metric = mean p(choose A | T11, T13) - mean p(choose A | T8, T15),
    where T11/T13 are the margin-0 cells whose cascade-first cue favors A
    and T8/T15 are the margin-0 cells whose cascade-first cue favors B
    (under the design's validities [0.95, 0.85, 0.75, 0.65, 0.6, 0.6]
    reversed -> f5 consulted first).

    Tallying: expectation exactly 0 for ANY parameter values.
    Misbinding TTB: expectation ~ +/-(2p-1) in [0.33, 0.49].
    """
    def to_key(v):
        return tuple([int(x) for x in v])

    a_keys = data["option_a_ratings"].apply(to_key)
    b_keys = data["option_b_ratings"].apply(to_key)
    chose_a = 1.0 - data["response"].astype(float)  # response 0 = A

    # Margin-0 (knife-edge) cells, identified by their rating vectors.
    # Cascade-first cue (f5) favors A here:
    hi_cells = [
        ((0, 0, 0, 0, 0, 1), (1, 0, 0, 0, 0, 0)),  # T11: 1-1 split, f5->A
        ((0, 0, 0, 0, 1, 1), (1, 1, 0, 0, 0, 0)),  # T13: 2-2 split, f5->A
    ]
    # Cascade-first cue (f5) favors B here:
    lo_cells = [
        ((1, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1)),  # T8:  1-1 split, f5->B
        ((1, 1, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1)),  # T15: 3-3 split, f5->B
    ]

    def cell_pa(ak, bk):
        mask = (a_keys == ak) & (b_keys == bk)
        if int(mask.sum()) == 0:
            return np.nan
        return float(chose_a[mask].mean())

    hi = [cell_pa(ak, bk) for ak, bk in hi_cells]
    lo = [cell_pa(ak, bk) for ak, bk in lo_cells]
    hi = [v for v in hi if not np.isnan(v)]
    lo = [v for v in lo if not np.isnan(v)]
    if not hi or not lo:
        return 0.0

    return float(np.mean(hi) - np.mean(lo))
```

**Observed (real) value:** 0.3400 (var=0.0288)
**Predicted under pi_3_2:** 0.3950 (var=0.0319)
**Predicted under pi_2:** 0.0367 (var=0.0470)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all eight experiments, Tallying (pi_2) is decisively refuted, while the misbinding-TTB theory (pi_3_2) is broadly \u2014 but not universally \u2014 supported. Experiment-by-experiment: (1) observed -0.31 vs pi_3_2's +0.36 (sign flip) and pi_2's -0.01 (miss); (2) observed -0.29 vs pi_3_2's +0.19 (sign flip) and pi_2's +0.00 (miss); (3) observed 0.36 vs 0.33 (pi_3_2 hit, pi_2 miss); (4) observed 0.036 vs 0.07 (pi_3_2 hit) and 1.32 (pi_2 catastrophic miss \u2014 subjects' choices are essentially uncorrelated with tally margin, which alone falsifies any integration account); (5) observed 0.45 vs 0.40 (pi_3_2 hit, pi_2 sign-flipped miss); (6) observed 0.78 vs 0.69 (pi_3_2 hit, pi_2 opposite-sign miss); (7) observed 0.61 vs 0.66 (pi_3_2 hit, pi_2 at chance); (8) observed 0.34 vs 0.40 (pi_3_2 hit, pi_2 miss). So pi_2 wins nowhere: even in Experiments 1\u20132, where it is 'less wrong', it predicts ~0 where the data show a large systematic negative effect. The data uniformly indicate one-reason deciding, not cue integration. However, pi_3_2's two sign-flipped failures (Experiments 1 and 2) are diagnostic, not noise. In those two experiments the validity lists are non-monotone, so pi_3_2's implementation of 'anti-correlation' as a positional end-to-end reversal (w = val[::-1]) diverges from a true value-based anti-correlation (ascending validity order). Working through the cells: in Experiment 1 the positional-reversal hierarchy leads with f4 (advertised 0.65) and f0, whereas ascending-validity leads with the weak cues f0/f2 (0.60) \u2014 an ascending-validity cascade with follow rate q predicts metric (2-4q)/5, i.e., NEGATIVE, matching the observed -0.31 (whereas positional reversal predicts +0.36). In Experiment 2 the positional reversal happens to lead with f3 (the 95% expert), and the observed -0.29 on tied trials means subjects systematically chose AGAINST the top-validity cue in favor of the weakest cues (f0/f6, 0.60) \u2014 exactly what ascending-validity one-reason deciding predicts (metric \u2248 -q/2 \u2248 -0.3). In Experiments 3\u20138 the validity lists are monotone (or the first deciders coincide), so positional reversal and ascending validity make identical predictions \u2014 which is why pi_3_2 succeeds there. In other words, the latent construct pi_3_2's description gestures at ('the cascade effectively runs in ascending true validity') is correct in all eight experiments; only the code's positional-reversal implementation of it is wrong, and it breaks precisely where the validity vector is non-monotone. Since pi_2 is degenerate (refuted by the margin-flatness of Experiment 4 and the tied-trial following of Experiments 5\u20138) and pi_3_2 is good but fails two experiments, the right move is to replace the degenerate theory with a stronger competitor to the winner: a value-based anti-validity one-reason model that reproduces pi_3_2's successes on Experiments 3\u20138 while also producing the observed negative contrasts in Experiments 1\u20132.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace Tallying (pi_2) with 'Weakest-Expert-First Frugality' (value-based anti-validity TTB). Core claim: subjects are one-reason decision makers whose subjective hierarchy is anti-correlated in VALUE with the communicated validities \u2014 they consult the cue with the LOWEST advertised validity first, then the next-lowest, and so on (equivalently, subjective weights w_j = -validity_j, or 1 - validity_j), letting the first strictly discriminating cue decide and ignoring everything else; no integration across cues occurs, so the choice profile is flat across tally margins (consistent with Experiment 4's near-zero margin alignment). When no cue discriminates, guess uniformly. Validity ties should be broken stochastically or by a free tie-breaking parameter (e.g., a Bernoulli over early-vs-late position among equally weak cues), since Experiments 2 and 8 suggest tie-breaking can go either way across subjects. Noise: softmax with inverse temperature beta over the binary winner score plus an independent lapse epsilon, with parameter ranges widened relative to pi_3_2 (e.g., beta in [0.5, 3.0], epsilon in [0.02, 0.2]) so the implied follow probability p = (1-eps)*sigmoid(beta) + eps/2 spans roughly 0.55 to 0.90 \u2014 the higher end is needed for Experiment 1's steep negative contrast (-0.31) while moderate values cover Experiments 2, 5, and 7 (implied follow rates 0.58-0.73). This theory differs from pi_3_2 in exactly one respect \u2014 the anti-correlation is computed on validity VALUES (sort ascending) rather than by reversing the list's POSITIONS \u2014 which is invisible in the six monotone-validity experiments where pi_3_2 already succeeds, but flips the predictions to the correct negative signs in Experiments 1 and 2, where positional reversal wrongly promotes mid- and high-validity cues (f4 at 0.65 in Experiment 1; f3 at 0.95 in Experiment 2) to the top of the hierarchy. It differs from pi_2 in retaining the empirically validated one-reason, margin-blind structure. Expected metric signatures: Experiment 1 \u2248 (2-4q)/5 in [-0.32, -0.16] (observed -0.31); Experiment 2 \u2248 -q/2 in [-0.45, -0.28] (observed -0.29); Experiments 3-8 essentially unchanged from pi_3_2's (already accurate) predictions."
}
```

## Usage

```json
{
  "input_tokens": 17728,
  "output_tokens": 26998,
  "reasoning_tokens": 17722,
  "provider": "Io Net"
}
```
