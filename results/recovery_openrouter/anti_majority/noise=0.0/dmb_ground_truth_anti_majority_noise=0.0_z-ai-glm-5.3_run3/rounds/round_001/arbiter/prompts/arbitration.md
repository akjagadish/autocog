# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_1" and "pi_3") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_1" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_3" (THEORY 2 below). Justify your choice.


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


## THEORY 2 — pi_3
Power-Weighted Evidence Integration (PWEI), calibrated operating regime. When choosing between two options described by binary cues, people integrate ALL cues additively, but each cue's vote is weighted by a power transform of its stated validity, w_j proportional to v_j^gamma. The net evidence for option A is D = sum_j w_j * sign(a_j - b_j), and choice is a softmax over [D, -D] with inverse temperature beta, plus an independent lapse epsilon to uniform choice. The psychologically critical regime is a STEEP one: gamma around 40 makes the top cue's weight dominate a tightly-packed validity ladder (e.g. a 0.95 cue overriding a 0.90-led five-cue block, as in Experiment 2), producing near-TTB choices there, while the ~0.2 lapse keeps overall confidence graded and attenuated (as in Experiment 1). The steepness parameter gamma spans Tallying (gamma=0) to Take-The-Best (gamma -> infinity); the fitted human regime sits far toward the TTB end but retains graded, evidence-margin-dependent confidence on near-tie weighted conflicts, which is exactly what constant-confidence pure heuristics cannot produce.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Power-Weighted Evidence Integration.
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are 0/1 cue values. History is ignored
    # (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"PWEI expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Power-transformed validity weights, normalized by their sum.
    # Normalization is a per-experiment constant rescale (validities are
    # fixed), so it is absorbed by beta; it keeps the evidence scale
    # O(1) for any gamma and any n_features.
    w = np.power(val, gamma)
    w_sum = w.sum()
    if w_sum > 0:
        w = w / w_sum
    else:
        w = np.full(n_features, 1.0 / n_features)

    a, b = stim[0], stim[1]
    # Weighted evidence difference: each cue votes +/-w_j, ties vote 0.
    D = float(np.sum(w * np.sign(a - b)))
    scores = np.array([D, -D])

    # Numerically stable softmax over the weighted evidence.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    # Independent lapse to uniform choice.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # Guard against float drift.
    return np.random.choice(len(probs), p=probs)

## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
**Validities (n_features=6):** [0.9, 0.89, 0.88, 0.87, 0.7, 0.6]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  trial 3: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  trial 4: A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 5: A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 6: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0]
  trial 7: A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  trial 8: A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 1, 1, 0, 0]
  trial 9: A=[1, 1, 1, 1, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  trial 10: A=[0, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0]
  trial 11: A=[1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0]
  trial 12: A=[1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 1, 0]
  trial 13: A=[1, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0]
  trial 14: A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 0, 0]
  trial 15: A=[1, 0, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0]
  trial 16: A=[1, 0, 1, 1, 0, 0]  B=[1, 0, 1, 1, 0, 0]

**Rationale:** This design targets the specific operating regime PWEI claims for humans (gamma in [35,50], i.e., a steep but finite power transform) and opposes it to TTB's two structural invariants: (i) constant confidence sigmoid(beta) on EVERY trial where any cue discriminates, regardless of which cue, its validity, or what lower cues say, and (ii) 50% only when all cues tie. The validity ladder is deliberately TIGHT at the top (0.90, 0.89, 0.88, 0.87) with a long tail (0.70, 0.60). At any gamma in [35,50], the three sub-top cues' combined power-weight EXCEEDS the top cue's weight (e.g., at gamma=40: 0.89^40+0.88^40+0.87^40 = 0.0193 > 0.90^40 = 0.0148), so PWEI must REVERSE against the top cue on the flagship 1-vs-3 trials (T2/T2m), predicting p(top-cue option) between 0.25 (gamma=35) and 0.45 (gamma=50) — never above 0.5 — while TTB predicts the top-cue option at its full constant confidence p*. This direction conflict is sign-robust across PWEI's entire parameter box. Second, trials T2, T3, T4, T5 all have cue 1 (0.90) as the first discriminating cue favoring the same option, but with graded weighted evidence D (gamma=40: -0.13, -0.02, +0.16, +0.43). TTB predicts a FLAT p* across all four; PWEI predicts a monotone psychometric sweep from ~0.31 to ~0.88 — a ~0.6-wide confidence gradient on trials TTB treats as identical. Third, depth invariance: on T6/T6m cues 1-4 tie and cue 5 (validity only 0.70) decides; TTB predicts the SAME constant p* as on cue-1 trials, while PWEI's weight for 0.70^gamma is ~1e-5 of the total, forcing ~50% choices. T8/T9 probe depth 2: TTB stays at p*, PWEI is at ~50% (T8, near-canceling block) or 0.82 (T9). T1/T1m anchor both models near ceiling; T10 (identical vectors) verifies both models' 50% guessing prediction and catches key bias. Mirrored versions of every dissociating trial control side bias; analysis should pool by cue-winner. With 16 unique pairs and K=6 (96 trials exactly), each cell yields 12 observations per subject (6 + 6 mirrored). Predicted outcomes: if TTB is correct, choice proportions favoring the first-discriminating-cue winner are statistically indistinguishable across ALL of T1-T9 and their mirrors (one common p*), including the 1-vs-3 reversal trials and the 0.70-validity depth trials, with T10 at 50%; if PWEI is correct, confidence tracks the weighted evidence margin D: below 50% for the top cue on T2, near 50% on T3/T6/T8, and graded from ~0.7 to ~0.9 on T4/T5/T9 — a pattern TTB's two-level step function cannot produce, and conversely a flat p* pattern that PWEI's steep-gamma regime cannot produce because the tight ladder forces it to vote against the top cue.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
The metric is a zero-sum linear contrast (weights 2, -1, -1) over choice proportions favoring the TTB winner in three structurally identified trial classes, so it is exactly 0 in expectation for TTB at ANY parameter setting (beta in [0.1,20], epsilon in [0,0.5]): TTB's two structural invariants — one-reason stopping and constant confidence sigmoid(beta) on every discriminating trial — force p_HIGH = p_DEPTH5 = p_REV = p*. For PWEI the three classes sit at very different points of its evidence-margin psychometric: HIGH trials (all cues unanimous, or top cue alone deciding) have |D| ~ 0.4-1 normalized, giving p ~ 0.89-0.92 after lapse; DEPTH5 trials have D ~ 6e-5 (0.70^gamma is ~1e-5 of total weight at gamma in [35,50]), giving p ~ 0.50; REV trials have D negative (0.89^g+0.88^g+0.87^g > 0.90^g throughout the box), giving p ~ 0.12-0.37 — PWEI votes AGAINST the top cue. Hence PWEI's metric value is ~0.9-1.2 across its entire parameter box, vs ~0 for TTB: a sign- and magnitude-robust gap. Per-subject noise is binomial only (24, 12, 12 observations per class): worst-case per-subject SD is ~0.29 (TTB with p*=0.5) and ~0.21 (PWEI), so at N=25 the Welch t-statistic is ~13-16, far below alpha=0.01. The metric works identically on a single subject's 96-trial slice, so between-subject variance is well defined. It is falsifiable (human data showing graded confidence or top-cue reversals would push it away from 0), not trivially true: it directly tests TTB's constant-confidence/depth-invariance prediction against actual choices.

Source:
def metric(data: pd.DataFrame) -> float:
    """Depth/conflict invariance contrast for TTB vs PWEI.

    Three structurally identified trial classes (validity order is fixed:
    [0.90, 0.89, 0.88, 0.87, 0.70, 0.60]):

    HIGH  - the top cue (0.90) discriminates and every other discriminating
            cue agrees with it (design trials 1, 2, 9, 10; 24 obs/subject).
            PWEI evidence margin |D| is maximal (~1 or w_top).
    DEPTH5- cues 1-4 all tie; only the 0.70-validity cue discriminates
            (design trials 11, 12; 12 obs/subject). PWEI's weight for
            0.70^gamma is ~1e-5 of total, so PWEI predicts ~50% choices.
    REV   - the top cue discriminates one way, the three sub-top cues
            (0.89, 0.88, 0.87) discriminate the other way, tail cues tie
            (design trials 3, 4; 12 obs/subject). In PWEI's steep-gamma
            regime the sub-top block outweighs the top cue, so PWEI
            chooses AGAINST the TTB winner.

    For each class compute p = P(choose the TTB winner). TTB's structural
    invariant is a single constant confidence p* on every discriminating
    trial, so 2*p_HIGH - p_DEPTH5 - p_REV = 0 exactly, at ANY (beta, eps).
    PWEI predicts p_HIGH ~ 0.9, p_DEPTH5 ~ 0.5, p_REV ~ 0.1-0.37, giving a
    value ~0.9-1.2 across its entire parameter box.
    """
    VALS = np.array([0.90, 0.89, 0.88, 0.87, 0.70, 0.60])
    order = np.argsort(-VALS, kind="stable")  # validity-descending cue indices

    n_high, k_high = 0, 0
    n_d5, k_d5 = 0, 0
    n_rev, k_rev = 0, 0

    a_col = data["option_a_ratings"]
    b_col = data["option_b_ratings"]
    r_col = data["response"]

    for a_cell, b_cell, resp in zip(a_col, b_col, r_col):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        s = np.sign(a - b)  # +1 favors A, -1 favors B, 0 tie, per cue

        # First discriminating cue in validity order (TTB's search).
        first = None
        for j in order:
            if s[j] != 0.0:
                first = j
                break
        if first is None:
            continue  # identical options: both theories say 50%

        ttb_winner_is_a = s[first] > 0
        chose_ttb = (int(resp) == 0) == ttb_winner_is_a

        if first == 4:
            # DEPTH5: only the 0.70-validity cue discriminates.
            n_d5 += 1
            k_d5 += int(chose_ttb)
        elif first == 0 and s[1] == -s[0] and s[2] == -s[0] and s[3] == -s[0] \
                and s[4] == 0.0 and s[5] == 0.0:
            # REV: top cue vs the three sub-top cues, tail ties.
            n_rev += 1
            k_rev += int(chose_ttb)
        elif first == 0:
            # HIGH candidate: all other discriminating cues agree with top.
            others = [s[j] for j in range(1, 6) if s[j] != 0.0]
            if all(o == s[0] for o in others):
                n_high += 1
                k_high += int(chose_ttb)
        # All other trial types (mixed conflicts, cue-2-led trials, ties)
        # are deliberately excluded to keep the classes pure.

    p_high = k_high / n_high if n_high > 0 else 0.5
    p_d5 = k_d5 / n_d5 if n_d5 > 0 else 0.5
    p_rev = k_rev / n_rev if n_rev > 0 else 0.5

    return 2.0 * p_high - p_d5 - p_rev

### RESULTS
- Predicted under pi_1 (simulated): 0.0033 (var=0.0461)
- Predicted under pi_3 (simulated): 1.0483 (var=0.0759)
- Observed on real data: -0.7167 (var=0.0939)

## EXPERIMENT 2 (proposed by pi_3)

### DESIGN
**Validities (n_features=6):** [0.95, 0.92, 0.91, 0.9, 0.89, 0.88]

**Trial pairs (n=24):**
  trial 1: A=[1, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1]
  trial 2: A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1]
  trial 3: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 4: A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 5: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  trial 7: A=[1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  trial 8: A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1]
  trial 9: A=[1, 1, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  trial 10: A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  trial 11: A=[1, 1, 0, 0, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  trial 12: A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 1, 1]
  trial 13: A=[1, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 1]
  trial 14: A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 1]
  trial 15: A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  trial 16: A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 17: A=[1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 18: A=[1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  trial 19: A=[1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  trial 20: A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 21: A=[1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  trial 22: A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1]
  trial 23: A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  trial 24: A=[0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1]

**Rationale:** This design attacks TTB's central structural invariant — a SINGLE constant choice probability p* = (1-eps)*sigmoid(beta) + eps/2 on EVERY trial where any cue discriminates — using a validity architecture opposite to previously explored tight-top-ladder designs: one DOMINANT top cue (0.95, whose power-weight exceeds the entire rest of the ladder combined at every gamma in [35,50]: 0.95^40=0.1286 vs 0.0886 for cues 2-6) followed by a CONTESTED second tier (0.92 vs a 0.91/0.90/0.89/0.88 block whose combined power-weight exceeds cue 2's weight at every gamma in [35,50], e.g., at gamma=40: 0.0303 vs 0.0228). Consequence 1 (direction dissociation at depth 2): on L4-type trials cue 1 is tied, cue 2 (first discriminator) favors one option, and all four lower cues favor the other. TTB picks the cue-2 winner at p*; PWEI REVERSES, predicting p(cue-2 winner) between 0.33 (gamma=35) and 0.45 (gamma=50) — never above 0.5, sign-robust across PWEI's entire parameter box. Consequence 2 (graded confidence ladder): trials L0/L1/L2/L3/L4 hold cue 1 tied and cue 2 as first discriminator while the opposing block grows from 0 to 4 cues. PWEI predicts a monotone sweep of p(cue-2 winner) of roughly 0.73 / 0.59 / 0.48 / 0.42 / 0.37 (gamma=40, beta=8, eps=0.2), crossing 50% between L1 and L2 — a psychometric function of the weighted evidence margin D that TTB's two-level step function (p* or 50%) cannot produce at any (beta, eps). Consequence 3 (depth gradient on 'clean' trials): T1 (cue 1 decides, rest tied), L0 (cue 2 decides, rest tied), and U2 (cue 3 decides, rest tied) are identical to TTB — first discriminator, no other signal — so TTB predicts one common p*; PWEI predicts declining confidence 0.89 > 0.73 > 0.66 because the deciding cue's share of total weight shrinks with depth. Consequence 4 (agreement anchors): T3 (all cues agree, PWEI ~0.90) and T2 (cue 1 against all five others, PWEI 0.61-0.78, direction agreeing with TTB) show that even TTB's best single p* must simultaneously miss ~0.90 (T3), ~0.73 (L0), ~0.37 (L4) — no constant fits. Controls: C1/C2 (identical vectors) verify both models' 50% guessing and catch key bias; every dissociating trial has a mirrored twin (A/B swapped) to cancel side bias, with analysis pooled by cue-winner side. 24 unique pairs x K=4 = exactly 96 trials, giving 8 observations per cell per subject (4 + 4 mirrored). Decisive quantitative signatures: (i) Spearman correlation between cell choice proportion and PWEI's weighted margin D (strongly positive if PWEI true, ~0 if TTB true); (ii) L4 significantly below 50% for the first-discriminator winner; (iii) monotone L-ladder crossing 50%; (iv) T1 > L0 > U2 depth ordering. TTB predicts all discriminating cells statistically indistinguishable at one common p* and cannot approximate any of these patterns even with its full beta in [0.1,20], epsilon in [0,0.5] range, because its per-trial prediction is parameterized by a single constant independent of cue depth, cue validity, and lower-cue configuration.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



### METRIC
Rationale:
Core idea: TTB's structural invariant is that its choice probability is a single constant p* = (1-eps)*sigmoid(beta) + eps/2 on EVERY trial where any cue discriminates, regardless of cue depth, validity, or lower-cue configuration. PWEI instead produces a graded psychometric function of the power-weighted evidence margin. The metric therefore regresses, trial by trial, the binary indicator 'did the subject choose the TTB winner (the option favored by the first discriminating cue)?' on the PWEI weighted margin D re-signed toward that winner (x = +D if TTB winner is A, -D if B; weights v_j^40 normalized, gamma=40 the midpoint of PWEI's box). Under TTB, E[y|x] = p* for all x, so the OLS slope is exactly 0 for every parameter setting (beta in [0.1,20], eps in [0,0.5]) — this cancels TTB's parameter heterogeneity entirely, leaving only binomial noise. Under PWEI, E[y|x] = (1-eps)*sigmoid(beta*x)+eps/2, a strictly increasing psychometric curve; across this design's 11 discriminating cell types the margins span x in [-0.08, 1.0] with predicted conformities from 0.374 (L4: cue-2 winner opposed by all four lower cues) up to 0.90 (all-agree), giving a slope of roughly 0.50-0.59 robustly across PWEI's whole parameter box (checked at gamma=35/beta=7/eps=0.23 -> ~0.52; gamma=50/beta=10/eps=0.17-0.23 -> ~0.50). Per-subject noise: 88 discriminating trials per subject, residual variance ~0.2, Var(x) ~0.10, so per-subject slope sd ~0.15 for both theories (TTB's is at most ~0.17 even at p*=0.5). At N=25 the Welch t on the pooled point estimates is approx 0.55 / sqrt((0.022+0.028)/25) ~ 12, far below alpha=0.01. The metric works identically on the pooled DataFrame and on each single-subject slice (each subject sees all 24 pairs x 4 reps), so between-subject variance is well defined. Identical-option catch trials (trials 23/24) are excluded because both models guess there and including them would hand TTB a spurious non-zero slope. Mirror trials are handled automatically since x and y are computed from the actual A/B vectors, so side bias cancels in the regression.

Source:
def metric(data: pd.DataFrame) -> float:
    # OLS slope of "chose the TTB (first-discriminating-cue) winner" on the
    # PWEI power-weighted evidence margin measured TOWARD that winner.
    # TTB: choice prob is one constant p* on every discriminating trial
    #      (any beta, eps)  =>  slope exactly 0.
    # PWEI: prob is a psychometric function of the weighted margin D
    #      =>  strongly positive slope (~0.5 over the design's margin range).
    val = np.array([0.95, 0.92, 0.91, 0.90, 0.89, 0.88], dtype=float)
    gamma = 40.0  # mid of PWEI's calibrated regime [35, 50]
    w = np.power(val, gamma)
    w = w / w.sum()
    cue_order = np.argsort(-val, kind="stable")

    xs = []
    ys = []
    for a_cell, b_cell, r in zip(
        data["option_a_ratings"], data["option_b_ratings"], data["response"]
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        s = np.sign(a - b)
        if not np.any(s != 0):
            # Identical options: pure guessing for both theories; excluded.
            continue
        # TTB winner: first discriminating cue in descending-validity order.
        winner = None
        for j in cue_order:
            if s[j] != 0:
                winner = 0 if s[j] > 0 else 1
                break
        if winner is None:
            continue
        # Power-weighted evidence for option A.
        D = float(np.sum(w * s))
        # Margin re-signed so that positive = evidence favors the TTB winner.
        x = D if winner == 0 else -D
        y = 1.0 if int(r) == winner else 0.0
        xs.append(x)
        ys.append(y)

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size < 3:
        return 0.0
    sxx = float(np.sum((xs - xs.mean()) ** 2))
    if sxx <= 1e-12:
        return 0.0
    sxy = float(np.sum((xs - xs.mean()) * (ys - ys.mean())))
    return float(sxy / sxx)

### RESULTS
- Predicted under pi_1 (simulated): 0.0081 (var=0.0061)
- Predicted under pi_3 (simulated): 0.5548 (var=0.0148)
- Observed on real data: -0.5230 (var=0.0540)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
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
```

**Observed (real) value:** 0.1580 (var=0.0120)
**Predicted under pi_1:** 0.3480 (var=0.0087)
**Predicted under pi_3:** 0.2473 (var=0.0026)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0, 1, 1, 1]  B=[1, 0, 0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 1]  B=[1, 1, 0, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 1, 1, 1]  B=[1, 1, 0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 0, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1]

**Metric**
```python
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

```

**Observed (real) value:** 0.1814 (var=0.0118)
**Predicted under pi_1:** 0.1536 (var=0.0134)
**Predicted under pi_3:** 0.2557 (var=0.0053)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all four experiments, neither theory provides a task-invariant account, and the observed data reveal a regularity that BOTH theories miss in the same direction. Experiment 1: TTB (pi_1) structurally predicts a metric of exactly 0 (constant confidence p* on every discriminating trial at any beta/epsilon), yet the observed value is -0.717 \u2014 a huge deviation. PWEI (pi_3) predicts +1.05, i.e., the opposite sign of what was observed; its core 'steep-gamma reversal' prediction (sub-top block overriding the top cue) is not just quantitatively but directionally wrong. Experiment 2: TTB predicts slope ~0, PWEI predicts +0.55, observed is -0.523 \u2014 again opposite in sign to PWEI and far from TTB. Experiments 3 and 4 are the only ones where either theory is in the right ballpark: TTB fits Exp 4 well (0.154 vs 0.181 observed) and both overpredict Exp 3 (0.348 and 0.247 vs 0.158). So pi_1 is the relative incumbent (smaller total error, correct on Exp 4), while pi_3 is degenerate: it fails its own designed dissociations with sign reversals on both critical experiments.\n\nCrucially, the SIGN and magnitude of the failures are diagnostic. In Experiment 1, the observed -0.717 implies p_REV is HIGH (subjects follow the single top cue against the three-cue block) while p_HIGH is LOW (subjects abandon the TTB winner on unanimous trials) and p_DEPTH5 is low. In Experiment 2, the negative slope means subjects choose AGAINST the TTB winner precisely when the top-cue-dominated weighted margin favors it. The pattern that reconciles ALL FOUR experiments is a preference for the option with FEWER 1s (more 0s): (i) Exp 1 \u2014 on HIGH trials the TTB winner has more 1s (p_HIGH\u22480), on DEPTH5 trials likewise (p_D5\u22480), on REV trials the TTB winner is the sparse option (p_REV\u22481), giving a metric near -1, attenuated to \u2248-0.7 by noise \u2014 matching the observed -0.717 almost exactly; (ii) Exp 2 \u2014 high-x trials are ones where the top cue favors an option loaded with 1s (y=0) while moderate/negative-x trials often have the sparse option as TTB winner (y=1), producing a slope of roughly -0.6 to -0.7, attenuating to the observed -0.52; (iii) Exp 4 \u2014 on TTB/Tallying conflict trials the tally winner virtually always has more 1s, so a sparse-option preference masquerades as TTB behavior, predicting ~0.15-0.2 tally-winner choices, matching the observed 0.181; (iv) Exp 3 \u2014 on dissociation trials the sparse option usually coincides with the TTB winner, predicting a positive but noise- and heterogeneity-attenuated value, consistent with the modest observed 0.158. No reparameterization can rescue either existing theory: TTB's metric is exactly 0 for ALL (beta, epsilon), and PWEI's sign flip on its own experiments holds across its entire gamma box; even tallying (gamma\u21920) predicts a POSITIVE slope on Exp 2. A genuinely new mechanism \u2014 inverted cue polarity / sparse-option preference \u2014 is required.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace pi_3 with a new theory: **Sparse-Option Preference / Polarity-Inverted Evidence Integration (SPI)**. Core claim: subjects in this task systematically treat a cue value of 1 as evidence AGAINST an option (equivalently, they prefer the option with more 0s / fewer expert endorsements \u2014 e.g., mis-mapped cue polarity, a 'fewer red flags' reading, or a rarity-implies-quality inference), integrating ALL cues with near-uniform weights. Formal sketch: for options with cue vectors a, b, define evidence for A over B as E = sum_j w_j * (b_j - a_j), where w_j = v_j^gamma with gamma in [0, 2] (near-tallying; validity plays at most a mild role \u2014 this is what makes the behavior look TTB-like on Exp 3/4 conflict trials where the top-cue option happens to be sparse, but anti-TTB on loaded unanimous trials). Choice is a softmax over [E, -E] with inverse temperature beta in [0.5, 20], plus an independent lapse epsilon in [0.05, 0.35] to uniform. Optionally allow per-subject heterogeneity: a polarity mixing weight or a small subpopulation of validity-ordered one-reason subjects, which explains why Exp 3's observed value (0.158) is attenuated relative to the pure sparse-preference prediction (~0.35). Predicted signatures: Exp 1 metric \u2248 -0.7 to -1.0 (vs observed -0.717); Exp 2 slope \u2248 -0.5 to -0.7 (vs observed -0.523); Exp 4 tally-winner proportion \u2248 0.15-0.25 (vs observed 0.181); Exp 3 \u2248 0.1-0.35 (vs observed 0.158). This theory is sharply distinguished from both incumbents: unlike TTB it uses all cues and inverts the evidence sign (TTB's Exp 1 metric is pinned at 0 for all parameters), and unlike PWEI it places humans at the FLAT end of the weighting ladder with inverted polarity, predicting the negative slopes that PWEI's steep-gamma box cannot produce at any parameter setting. It should be proposed as a direct competitor to the surviving incumbent pi_1 (TTB), which it beats on Experiments 1 and 2 while matching it on Experiments 3 and 4."
}
```

## Usage

```json
{
  "input_tokens": 11980,
  "output_tokens": 12375,
  "reasoning_tokens": 11062,
  "provider": "Sail Research"
}
```
