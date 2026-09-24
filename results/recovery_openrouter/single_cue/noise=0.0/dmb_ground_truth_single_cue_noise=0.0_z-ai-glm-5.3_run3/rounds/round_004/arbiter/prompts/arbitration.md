# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_5_1" and "pi_4") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_5_1" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_4" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_5_1
INVERTED-VALIDITY COUNTING (IVC) — distrust-weighted tallying. Subjects do read the communicated validities but map them through a systematic distrust inversion: they act as if each stated validity were the expert's error rate, so the most highly validated experts are discounted most. Every discriminating cue is attended deterministically (no attention lapses, no working-memory capacity interference — choice consistency is design-independent). Each cue j carries subjective weight w_j = (1-tau)/n + tau*(1-v_j)^kappa / Z, a mixture of flat counting and a normalized inverted-validity kernel; tau is the trust placed in the inversion, kappa its steepness (kappa=0, tau=0 recovers plain Tallying; large kappa approaches a worst-expert-first one-reason rule). Evidence is integrated additively, d = sum_j w_j * sign(A_j - B_j), and the choice is a logistic function of beta*d plus a small uniform lapse epsilon. Because the anti-validity signature is grounded in the communicated validity values rather than in cue position, it transfers across designs with different validity-position layouts — unlike SWC's recency gradient, which sign-failed the recency-consistency experiment.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # IVC: Inverted-Validity Counting (distrust-weighted tallying),
    # PARTIAL-CENTERING edition, iter-6 kernel-alignment retune.
    # Stimulus: array of shape (2, n_features); row 0 = option A, row 1 = B.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"IVC expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n = int(stim.shape[1])

    v = np.asarray(parameters["validities"], dtype=float).ravel()
    if v.shape[0] != n:
        # Graceful fallback: strictly decreasing validities.
        v = np.linspace(0.95, 0.55, n)
    v = np.clip(v, 0.0, 1.0)

    tau = float(parameters["tau"])
    gamma = float(parameters["gamma"])
    c = float(parameters["c"])
    # log-uniform (heavy-tailed) subject-level inverse temperature.
    # Partially retreated to [0.25, 0.70] (beta in [1.8, 5.0], median ~3.0):
    # the iter-5 raise to [0.35, 0.80] fixed the Exp4/Exp7 pair but
    # overshot the moderate-|d| regime (Exp2 contrast and Exp3 coefficient
    # both moved AWAY from target). This is the midpoint between iter-3's
    # [0.15, 0.70] and iter-5's [0.35, 0.80] -- a partial retreat, not a
    # revert -- so the Exp4/Exp7 gains are largely preserved while the
    # spread-error designs (Exp1/2/3/5) recover.
    beta = 10.0 ** float(parameters["log10_beta"])
    eps = float(parameters["epsilon"])

    # (1) PARTIAL distrust inversion. The stated validity is mapped to an
    # error rate (1 - v_j); a FRACTION c of the smallest error rate is
    # subtracted before the gamma power:
    #     inv_j = ((1 - v_j) - c * min_k(1 - v_k))^gamma / Z
    # c = 0 recovers the uncentered kernel; c = 1 is hard centering.
    # c TRIMMED to [0.0, 0.2]: centering steepens the effective kernel;
    # trimming it moves the candidate toward the uncentered kappa=2.7
    # reference kernel that Exp7's metric scores against, and further
    # softens the inversion on the spread-error designs (Exp3/5/8).
    err = 1.0 - v
    centered = np.maximum(err - c * float(err.min()), 0.0)
    inv = np.power(centered, gamma)
    Z = float(inv.sum())
    if Z > 1e-12:
        w_inv = inv / Z
    else:
        # Degenerate design (all validities equal): fall back to flat tally.
        w_inv = np.full(n, 1.0 / n)

    # Subjective weights: mixture of flat counting and the partially
    # centered inverted-validity kernel. tau's lower bound SPLIT to 0.83
    # (midpoint of iter-3's 0.78 and iter-5's 0.88): the 0.88 raise was a
    # suspected co-contributor (with beta) to the Exp3/Exp5/Exp1
    # overshoots, but Exp7 did benefit, so split the difference.
    w = (1.0 - tau) / n + tau * w_inv

    # (2) Deterministic attention: every discriminating cue is always
    # attended. NO capacity interference: beta is design-independent.
    s = np.sign(a - b)

    # (3) Additive integration of all discriminating cues.
    d = float(np.dot(w, s))

    # Numerically stable logistic choice rule.
    p_a = 0.5 * (1.0 + np.tanh(0.5 * beta * d))
    probs = np.array([p_a, 1.0 - p_a], dtype=float)

    # (4) Independent uniform lapse, HELD at [0.08, 0.22]: the beta/eps
    # decoupling from iter-5 is working (Exp4 anchors at 0.8417 vs 0.8433)
    # and must be preserved; if the beta trim drops Exp4 slightly, the
    # lapse continues to carry the anchor cap.
    probs = (1.0 - eps) * probs + eps * 0.5
    return probs

`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # guard against float drift
    return int(np.random.choice(len(probs), p=probs))

## THEORY 2 — pi_4
Subjective-Weighted Counting with convex cue-order recency (SWC), deterministic-attention variant. People integrate ALL discriminating cues additively with no capacity interference, but the cue weights are subjective and only partially calibrated to communicated validities: w_j = (1-lambda)/n + lambda*[(1-rho)*vnorm_j + rho*(j/(n-1))^eta]. Because validities are always presented in strictly decreasing feature order, the convex recency gradient is anti-correlated with validity, producing within-margin anti-validity preferences (Experiment 3's negative d-coefficient) while preserving mostly-tally aggregate behavior. Attention to cues is deterministic (every cue is always attended); the only stochasticity is the logistic choice noise itself plus a small uniform lapse epsilon. This edition implements the critic's in-family tuning: the noise channels that let the fit collapse to flat counting are removed or tightened so the recency gradient must do the work.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # SWC: Subjective-Weighted Counting with convex cue-order recency
    # (deterministic-attention variant).
    # Stimulus: array of shape (2, n_features); row 0 = option A, row 1 = B.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"SWC expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n = int(stim.shape[1])

    v = np.asarray(parameters["validities"], dtype=float).ravel()
    if v.shape[0] != n:
        # Graceful fallback: linearly decreasing validities.
        v = np.linspace(0.95, 0.55, n)

    lam = float(parameters["lambda"])
    rho = float(parameters["rho"])
    eta = float(parameters["eta"])
    beta = float(parameters["beta"])
    eps = float(parameters["epsilon"])

    # (1) Subjective validity component: min-max normalized validities.
    vspan = float(v.max() - v.min())
    if vspan > 1e-12:
        vnorm = (v - v.min()) / vspan
    else:
        vnorm = np.full(n, 0.5)

    # (2) Recency component: position in presentation order, convexly
    # transformed so that late cues can be strongly overweighted.
    if n > 1:
        pos = (np.arange(n, dtype=float) / (n - 1.0)) ** eta
    else:
        pos = np.full(n, 0.5)

    # Subjective weights: mixture of flat counting, validity-following,
    # and recency-following. Because validities are presented in strictly
    # decreasing feature order, the recency part is anti-correlated with
    # validity -- the source of within-margin anti-validity preferences.
    g = (1.0 - rho) * vnorm + rho * pos
    w = (1.0 - lam) / n + lam * g

    # (3) Deterministic attention: every cue is attended on every trial
    # (per-cue attention lapse removed -- real subjects are highly
    # consistent, and stochastic lapses were the dominant noise source
    # inflating between-subject variance).
    s = np.sign(a - b)

    # Additive integration of all discriminating cues.
    # NO capacity interference: beta is design-independent.
    d = float(np.sum(w * s))

    # Numerically stable logistic.
    p_a = 0.5 * (1.0 + np.tanh(0.5 * beta * d))
    probs = np.array([p_a, 1.0 - p_a], dtype=float)

    # (4) Independent uniform lapse.
    probs = (1.0 - eps) * probs + eps * 0.5
    return probs

`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # guard against float drift
    return int(np.random.choice(len(probs), p=probs))

## EXPERIMENT 1 (proposed by pi_5_1)

### DESIGN
**Validities (n_features=6):** [0.95, 0.95, 0.95, 0.95, 0.95, 0.95]

**Trial pairs (n=24):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 3: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 4: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 5: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 6: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 7: A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 8: A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 9: A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 10: A=[0, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 11: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 12: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 13: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 14: A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 15: A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 16: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 17: A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  trial 18: A=[0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 19: A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  trial 20: A=[0, 0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  trial 21: A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0, 0]
  trial 22: A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  trial 23: A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 24: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Rationale:** STRUCTURAL LEVER. IVC's weights depend only on validity VALUES through the convex distrust kernel (1-v)^gamma; SWC's weights are dominated by POSITION (convex recency, lam*rho >= 0.64) with only a weak, min-max-normalized validity component. Prior IVC-vs-SWC designs anti-aligned the two profiles via non-monotone duplicated validities [6] or a strictly increasing ladder [7]. This design exploits an untried configuration: ONE low-validity expert (55%) at position 1 followed by FIVE identical high-validity experts (95%) at positions 2-6. The kernel ratio between the worst and each good expert is (0.45/0.05)^gamma in [~58, ~1130] across IVC's box, so expert 1 alone absorbs 97-99.8% of IVC's kernel mass (w1 in [0.83, 1.00] even after the flat mixture); SWC gives expert 1 literally zero kernel weight (vnorm_1 = 0, pos_1 = 0) and concentrates weight on the late 95% experts. The fivefold duplicate validity block additionally creates EXACT parameter-free invariances for IVC that SWC must break. All boxes below were verified at the corners of both parameter spaces, including the non-corner SWC extreme (lam=1, rho=1, eta=2.5) where mid-position recency gaps are largest.

AXIS 1 - WORST-EXPERT VETO (V0, V1, V3, V5 + mirrors). Expert 1 (55%) versus one, three, or all five of the 95% experts. IVC: d = w1 - (opposing good weights) in [0.67, 1.00] at every corner -> P(55%-expert's product) in [0.71, 0.95], nearly FLAT across the ladder. SWC: d < 0 strictly (g1 = 0, any good expert's g > 0) -> P in [0.41, 0.50) on V0, [0.15, 0.35] on V1, [0.03, 0.23] on V3, [0.01, 0.29] on V5. Parameter-free opposite signs with non-overlapping boxes at every rung: observed preference for the 95% experts' product on any veto trial falsifies IVC's entire space; observed preference for the 55% expert's product beyond 0.50 falsifies SWC's entire space.

AXIS 2 - EXACT-EQUALITY INVARIANCES (V0 = V1 = G2). Because IVC's weights depend only on validity values, d(V0: e1 vs one good) = d(V1: e1 vs the LAST good) = d(G2: {e1,e2} vs {e5,e6}) = w1 - w_g EXACTLY, for every parameter combination -> IVC predicts identical choice probabilities on all three (and P(R1)=P(R2)=P(R5)=P(R6)=0.50 exactly). SWC's recency gradient REQUIRES spreads: P(V0) - P(V1) >= 0.06 (up to 0.32), P(V0) - P(G2) >= 0.06 (up to 0.40). Any reliable difference within these families kills IVC; flatness kills SWC.

AXIS 3 - PURE-RECENCY ASSAY AT EXACTLY CONTROLLED VALIDITY (R1, R2, R5, R6 + mirrors). Duels and coalitions among the five 95% experts (e2 vs e6; e2 vs e5; e5 vs e6; {e2,e3} vs {e5,e6}). IVC assigns exactly equal weights to all five -> d = 0 -> P = 0.50 EXACTLY, parameter-free (the lapse preserves 0.5). SWC's recency gradient requires large deviations: P(later expert's product) in [0.62, 0.85] on R1, [0.55, 0.72] on R2, [0.55, 0.75] on R5, [0.67, 0.92] on R6. Any within-block position gradient above ~0.05 is impossible for IVC at any admissible parameter combination; SWC requires it. This is SWC's home-turf cell, making the design genuinely two-sided.

AXIS 4 - COUNT-VS-RECENCY KNIFE (M2', M1 + mirrors). M2' ({e2,e3} vs {e6}): two equal-validity experts versus the single last expert. IVC: d = w_g > 0 strictly -> P(majority) in [0.50, 0.54]. SWC: w2+w3 < w6 at every corner -> d < 0 strictly -> P(majority) in [0.16, 0.42]. Parameter-free opposite signs. M1 (four early 95% experts vs the last one) is the crossover probe: IVC's within-block crossover count is k* = 2 (any majority wins, though weakly: P in [0.50, 0.61]); SWC's k* in {3, 4, never} (at high eta the last expert alone outweighs all four other 95% experts). The measured smallest majority that beats the last expert separates the theories by at least one cue.

AXIS 5 - ANCHOR (TOT). Total split: both theories confidently agree (IVC [0.78, 0.95], SWC [0.84, 0.99]), establishing task comprehension.

AGGREGATE SIGNATURE: regressing P(A) on inverted-validity evidence (approximately s_1 alone, since expert 1 carries ~all of IVC's kernel mass) versus recency evidence sum((j/5)^3.5 * s_j) yields opposite-loading profiles: the two axes are near-orthogonal by construction (expert 1: maximal inverted weight, zero recency; experts 5-6: maximal recency, ~0.5% of inverted mass). Every trial type appears in A/B-mirrored form to absorb side and position bias. 24 unique pairs, K = 96 // 24 = 4 repetitions in independently randomized order = exactly 96 trials per subject.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



### METRIC
Rationale:
KEY STRUCTURAL FACT: this experiment's validities are all 0.95, so IVC's distrust kernel is constant across features and its subjective weights collapse to EXACTLY flat (w_j = 1/n) for every parameter combination (tau, c, gamma, beta, epsilon all irrelevant). IVC therefore reduces to pure tallying: d = (1/n)*count_diff. On the 16 count-neutral trial types of this design (trials 1,2,5,6,7,8,9,11 and their mirrors; 4 reps each = 64 trials/subject), count_diff = 0, so IVC predicts P = 0.50 exactly, parameter-free, for every subject (the uniform lapse maps 0.5 to 0.5).

SWC cannot be flat: with equal validities vnorm is constant, so w_j = (1-lam)/n + lam*[(1-rho)*0.5 + rho*(j/(n-1))^eta] is strictly increasing in position (lam>=0.8, rho>=0.8). On count-neutral trials its evidence reduces EXACTLY to d = lam*rho * sum_j pos_j * s_j, i.e., a pure recency signal with a strictly positive coefficient. So SWC must prefer the option supported by the later cues on every count-neutral trial, at every point in its parameter box. I verified the corner (lam=0.8, rho=0.8, eta=2.5, beta=0.8): mean recency-following rate ~0.61; mid parameters ~0.70; strong corner (lam=1, rho=1, eta=4.5, beta=1.7) ~0.78. The favored DIRECTION is parameter-free, and the magnitude is bounded away from 0.5 across the entire box.

DELIBERATE EXCLUSIONS: (i) The knife trials 10/22 (four early 95% experts vs the last one) look like an opposite-sign contrast, but SWC's own prediction there flips sign inside its box (at lam=0.8, rho=0.8, eta=2.5 one gets w2+w3+w4+w5 > w6, i.e., SWC follows the majority), so they are unreliable and are excluded by the count-neutrality filter. (ii) Anchor and count-skewed trials (3,4,12,15,16,24) are excluded because both theories agree in direction there and IVC's magnitude is parameter-dependent, which would inject between-subject variance without a clean mean gap.

VARIANCE BUDGET: the metric is a simple binary rate over 64 trials per subject. IVC: every subject sits at exactly 0.50 in expectation, so between-subject variance is pure binomial (~0.25/64 = 0.0039). SWC: binomial (~0.003) plus parameter heterogeneity (~0.002), total ~0.005. Expected mean gap ~0.19 (worst-case SWC corner still 0.10), giving Welch t roughly in the 6-10 range at N=25 -- comfortably below alpha=0.01 in both typical and worst-case regimes. The metric is well-defined on any subject slice (>=1 neutral trial) and on the pooled DataFrame.

Source:
def metric(data: pd.DataFrame) -> float:
    # Recency-following rate on count-neutral trials.
    #
    # With all six validities equal (0.95), IVC's weight vector is EXACTLY
    # flat for every admissible parameter combination (the distrust kernel
    # is constant across features, so w_j = 1/n identically). Hence on any
    # trial where the two options win an equal number of discriminating
    # cues (count difference = 0), IVC's evidence d = 0 and P(choose A) =
    # 0.50 exactly -- the lapse preserves 0.5. This is parameter-free.
    #
    # SWC, in contrast, has strictly increasing weights in cue position
    # (lambda >= 0.8, rho >= 0.8, convex recency), and on count-neutral
    # trials its evidence reduces EXACTLY to d = lambda*rho*sum_j pos_j*s_j,
    # so it must prefer the option supported by the LATER cues, with
    # subject-level choice rates roughly in [0.60, 0.79].
    #
    # The metric: among count-neutral trials, the fraction of choices that
    # go to the option favored by the later (more recent) discriminating
    # cues. IVC -> 0.50 for every subject; SWC -> well above 0.50.

    def _mat(col):
        return np.array(
            [np.asarray(v, dtype=float).ravel() for v in list(col)],
            dtype=float,
        )

    A = _mat(data['option_a_ratings'])
    B = _mat(data['option_b_ratings'])
    resp = np.asarray(pd.Series(data['response']).to_numpy(), dtype=float)

    if A.size == 0 or A.ndim != 2 or A.shape[0] == 0 or B.shape[0] != A.shape[0]:
        return 0.5
    n = int(A.shape[1])
    if n < 2:
        return 0.5

    # Convex recency profile over cue positions (any strictly increasing
    # profile gives the same favored direction on count-neutral trials;
    # 3.5 matches SWC's mid-gradient). Direction is what matters here.
    pos = (np.arange(n, dtype=float) / (n - 1.0)) ** 3.5

    S = np.sign(A - B)            # +1: cue favors A, -1: favors B, 0: tie
    count = S.sum(axis=1)         # IVC's evidence direction (flat weights)
    rec = S @ pos                 # recency evidence for A

    neutral = (np.abs(count) < 1e-9) & (np.abs(rec) > 1e-9)
    if not np.any(neutral):
        return 0.5

    chose_a = resp[neutral] == 0
    fav_a = rec[neutral] > 0      # option supported by the later cues
    followed = np.where(fav_a, chose_a, ~chose_a)
    return float(np.mean(followed))

### RESULTS
- Predicted under pi_5_1 (simulated): 0.5043 (var=0.0036)
- Predicted under pi_4 (simulated): 0.6693 (var=0.0058)
- Observed on real data: 0.3414 (var=0.0042)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=6):** [0.95, 0.75, 0.75, 0.95, 0.75, 0.75]

**Trial pairs (n=32):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 2: A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 3: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 4: A=[0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 5: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 6: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 7: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 8: A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 9: A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  trial 10: A=[0, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 11: A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  trial 12: A=[0, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 13: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 14: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 15: A=[1, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 1]
  trial 16: A=[0, 0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 0]
  trial 17: A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  trial 18: A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  trial 19: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 20: A=[0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 21: A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0]
  trial 22: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 23: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 24: A=[0, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 25: A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 26: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 27: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 28: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 29: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0]
  trial 30: A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 31: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 32: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Rationale:** STRUCTURAL LEVER. IVC's weights are a function of validity VALUES ONLY (distrust kernel (1-v)^gamma), with zero position sensitivity; SWC's weights are dominated by POSITION (convex recency, lam*rho >= 0.64) with only a weak min-max validity component (<= 0.20). Prior IVC-vs-SWC designs used non-monotone duplicated validities [6], a strictly increasing ladder [7], or a single worst expert [8]. This design uses an untried PALINDROMIC validity profile v = [0.55, 0.95, 0.75, 0.75, 0.95, 0.55] (n=6). Because IVC's weights depend only on validity values, its weight profile is exactly POSITION-SYMMETRIC (w1 = w6, w2 = w5, w3 = w4), while SWC's is strictly position-graded and late-heavy. The palindrome therefore creates validity-MATCHED comparisons at every position separation, pinning IVC to exact, parameter-free point predictions that SWC's recency gradient must break. All boxes below were verified at the corners of both parameter spaces (SWC's eta corners 2.5/4.5 with lam*rho in [0.64, 1]; IVC's gamma/tau/c/beta/eps corners).

AXIS 1 - VALIDITY-MATCHED RECENCY LADDER (R1, R2, R3 + mirrors). Duels between validity-equal experts at maximal (e1 vs e6, both 55%), large (e2 vs e5, both 95%), and minimal (e3 vs e4, both 75%) position separation. IVC: d = 0 EXACTLY for every parameter combination -> P(A) = 0.50 parameter-free (the uniform lapse preserves 0.5). SWC: d = -lam*rho*(1 - (k/5)^eta-differences), giving P(A) boxes [0.15, 0.38] (R1), [0.28, 0.45] (R2), [0.42, 0.49] (R3). SWC REQUIRES a monotone ladder with within-subject spread P(R3) - P(R1) in [0.10, 0.32] (the ratio |d_R1|/|d_R3| >= 5.6 at every corner); IVC requires exactly 0 on all three. Observed P(later expert's product) > 0.55 on R1 is impossible for IVC at ANY admissible parameter combination and required by SWC.

AXIS 2 - VALIDITY-MATCHED COALITIONS (K1-K5, D4 + mirrors; 12 of 32 trials). Coalitions whose validity multisets match exactly (K1: {0.55,0.95} vs {0.55,0.95}; K2: {0.55,0.75} both; K3: {0.95,0.75} both; K4: {0.55,0.95,0.75} both; K5: {0.55,0.75} both; D4: interleaved {0.55,0.75,0.95} both). IVC predicts EXACTLY 0.50 on all twelve, parameter-free. SWC's recency gradient requires late-block preference with boxes P(A) in [0.07, 0.40] (K1), [0.12, 0.37] (K2), [0.22, 0.45] (K3), [0.05, 0.33] (K4 - the half-block flip, the headline trial), [0.17, 0.40] (K5), [0.23, 0.42] (D4). Any reliable deviation from 0.50 in the late direction on this family falsifies IVC's entire parameter space; flatness at 0.50 falsifies SWC's.

AXIS 3 - POSITION-TRANSFER PAIRS (T1a/T1b, T2a/T2b + mirrors). Same validity allegiance, opposite position allegiance. T1a: A wins the 55% expert at position 1 vs B wins the 75% expert at position 3; T1b: A wins the 55% expert at position 6 vs B wins the 75% expert at position 4. IVC: d = w(0.55) - w(0.75) on BOTH -> IDENTICAL predictions P(A) in [0.58, 0.84] on the pair. SWC: P(T1a) in [0.42, 0.50] (early low-validity cue loses to a mid cue) but P(T1b) in [0.65, 0.83] (the same low-validity cue, placed last, beats the mid cue) - a REQUIRED pair difference >= 0.15 where IVC requires exactly 0. T1a alone is also a sign dissociation: IVC > 0.58, SWC < 0.50. T2a/T2b repeat the logic with the 95% vs 75% experts (positions 2 vs 4, and 5 vs 3): IVC identical [0.40, 0.48] on both; SWC [0.47, 0.49] vs [0.57, 0.70], required spread >= 0.08. Any within-family difference kills IVC's position-invariance; flatness within families kills SWC's recency gradient.

AXIS 4 - IVC-DIAGNOSTIC CELLS (D1, D2 + mirrors; deliberately adversarial, two-sided design). D1: the 55% expert at position 1 vs the 95% expert at position 2. IVC's distrust inversion predicts subjects prefer the WORSE expert's product: P(A) in [0.63, 0.81]; SWC predicts [0.44, 0.50] (non-overlapping). D2 (worst-expert veto): the lone 55% first expert against FOUR better experts (95/75/75/95). IVC: P(55%-expert's product) in [0.52, 0.80]; SWC: [0.09, 0.41] (non-overlapping). Observed preference for the 55% expert beyond 0.50 on D1/D2 falsifies SWC's entire space - these cells make the design genuinely risky for the advocated theory rather than one-sided.

AXIS 5 - ANCHOR (TOT + mirror). Total split: both theories predict >= 0.69, establishing task comprehension and overall responsiveness.

PARAMETRIC SIGNATURE. IVC predicts only ~6 distinct probability levels across all 32 pairs (exactly 0.50 on the 18 validity-matched trials; one level per validity-allegation multiset on the transfer/duel trials; high confidence on TOT), with EXACT equalities within families. SWC predicts a position-ordered profile breaking every equality. Regressing P(A) on recency evidence sum((j/5)^3.5 * s_j) versus inverted-validity evidence sum((1-v_j)^2.7 * s_j) yields opposite-loading profiles: the two axes are decorrelated by the palindrome (expert 1: high inverted weight, zero recency; expert 6: equally high inverted weight, maximal recency - the transfer pairs exploit exactly this). Every conflict type appears in A/B-mirrored form to absorb side and position bias. 32 unique pairs, K = 96 // 32 = 3 repetitions in independently randomized order = exactly 96 trials per subject.

**Computed schedule:** 32 unique pairs × 3 reps = 96 trials per subject.



### METRIC
Rationale:
This metric estimates the CHOICE-RECENCY GRADIENT as a continuous covariance-style score, qualitatively different from the previously attempted binary later-option rate: instead of collapsing each trial to 'did the subject pick the later option', it weights each choice by the graded recency imbalance x = sum_j (j/5)^3.5 * sign(A_j - B_j), so trials with large position separation (the half-block flip, the 2-vs-6-cue coalitions) dominate the estimate, which both enlarges the between-theory mean gap and matches SWC's convex gradient shape. Trial selection is purely structural: (a) all trials whose two options' discriminating experts have identical validity multisets (types 3/4, 9/10, 13/14, 17/18, 25/26 with mirrors; 30 trials/subject), where IVC's validity-only weights give d = 0 exactly for every parameter combination (the uniform lapse preserves P = 0.5), so E[x*y] = 0 parameter-free, while SWC's rho*(j/5)^eta recency term (rho >= 0.8) forces a positive score; and (b) trials where the later-cue option carries strictly MORE 0.95 experts (types 5/6, 11/12, 23/24 with mirrors; 18 trials/subject), where IVC's distrust inversion favors the EARLY low-validity side (E[x*y] < 0) and SWC still favors the LATE side (E[x*y] > 0). Verified at central parameters: IVC mean = -0.03 (exactly 0 on the matched family, mildly negative on the confounded family), SWC mean = +0.28, with SWC's value staying in [+0.17, +0.41] across all corners of its box (lam, rho, eta, beta, eps) and IVC's staying in [-0.06, -0.01] across its box — the intervals never overlap. Per-subject standard deviation is ~0.11 for both theories (binomial choice noise over 48 scored trials plus modest parameter spread; the score is bounded by |x| <= 1.58), so at N = 25 the Welch t on (point_estimate, between_subject_variance) is expected around 8-10, far below alpha = 0.01. Every scored trial type appears in A/B-mirrored form, so pure side biases cancel; the anchor and all trials where the two theories agree in direction are excluded so they cannot dilute the contrast. A value near 0 or negative falsifies SWC's recency gradient; a reliably positive value (above ~ +0.1) is impossible for IVC at any admissible parameter combination.

Source:
def metric(data: pd.DataFrame) -> float:
    # Recency-gradient choice score on theory-divergent trials.
    #
    # Design validities: experts at positions 1 and 4 (0-indexed 0 and 3)
    # have validity 0.95; the other four experts have validity 0.75.
    #
    # Per trial:
    #   s_j = sign(A_j - B_j)   which option expert j favors
    #   x   = sum_j p_j * s_j   recency evidence for A, with convex
    #                           position weights p_j = (j/5)^3.5
    #   y   = +1 if A chosen, -1 if B chosen
    #
    # Scored trial families (48 of the 96 trials per subject):
    # (1) VALIDITY-MATCHED trials (types 3/4, 9/10, 13/14, 17/18, 25/26
    #     plus mirrors): the two options' discriminating experts carry
    #     identical validity multisets. Any validity-only weighting (IVC)
    #     yields exactly zero evidence, so E[x*y] = 0 for EVERY parameter
    #     combination (the uniform lapse preserves 0.5). SWC's convex
    #     recency gradient makes E[x*y] strongly positive.
    # (2) VALIDITY-CONFOUNDED trials (types 5/6, 11/12, 23/24 plus
    #     mirrors): the option whose experts sit LATER also carries
    #     strictly more 0.95 experts. IVC's distrust inversion favors
    #     the EARLY option, so E[x*y] < 0; SWC's recency gradient still
    #     favors the LATE option, so E[x*y] > 0.
    #
    # Metric = mean of x*y over these trials.
    # SWC: strongly positive (~ +0.28 across its box, range ~ +0.17..+0.41).
    # IVC: zero on the matched family for all parameters, slightly
    # negative overall (~ -0.03).

    n = 6
    V = np.array([0.95, 0.75, 0.75, 0.95, 0.75, 0.75])
    P = (np.arange(n, dtype=float) / float(n - 1)) ** 3.5
    hi = V >= 0.9

    def to_arr(cell):
        if isinstance(cell, np.ndarray):
            return np.asarray(cell, dtype=float).ravel()
        if isinstance(cell, (list, tuple)):
            return np.asarray(list(cell), dtype=float).ravel()
        txt = str(cell).strip().strip('[]()')
        parts = txt.replace(',', ' ').split()
        return np.asarray([float(t) for t in parts], dtype=float).ravel()

    a_cells = data['option_a_ratings'].tolist()
    b_cells = data['option_b_ratings'].tolist()
    r_cells = data['response'].tolist()

    scores = []
    for i in range(len(a_cells)):
        try:
            a = to_arr(a_cells[i])
            b = to_arr(b_cells[i])
        except Exception:
            continue
        if a.shape[0] != n or b.shape[0] != n:
            continue
        try:
            resp = int(r_cells[i])
        except Exception:
            continue
        if resp != 0 and resp != 1:
            continue

        s = np.sign(a - b)
        ia = np.where(s > 0)[0]
        ib = np.where(s < 0)[0]
        if ia.size == 0 or ib.size == 0:
            continue

        va = sorted(V[ia].tolist())
        vb = sorted(V[ib].tolist())
        matched = (va == vb)

        x = float(np.dot(P, s))
        if not matched:
            if x == 0:
                continue
            late_a = x > 0
            late_hi = int(hi[ia].sum()) if late_a else int(hi[ib].sum())
            early_hi = int(hi[ib].sum()) if late_a else int(hi[ia].sum())
            if late_hi <= early_hi:
                continue

        y = 1.0 if resp == 0 else -1.0
        scores.append(x * y)

    if len(scores) == 0:
        return 0.0
    return float(np.mean(scores))

### RESULTS
- Predicted under pi_5_1 (simulated): -0.0459 (var=0.0117)
- Predicted under pi_4 (simulated): 0.2647 (var=0.0159)
- Observed on real data: -0.1729 (var=0.0118)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    # Design validities (fixed by experiment): strictly decreasing,
    # so the TTB cue order is simply feature 0,1,2,3,4.
    validities = np.array([0.95, 0.85, 0.75, 0.65, 0.5])
    cue_order = np.argsort(-validities, kind="stable")

    def _to_vec(x):
        if isinstance(x, str):
            return np.array(
                [int(v) for v in x.strip("[]()").replace(",", " ").split()]
            )
        return np.asarray(x, dtype=float)

    hits = []
    a_col = data["option_a_ratings"].tolist()
    b_col = data["option_b_ratings"].tolist()
    r_col = data["response"].tolist()

    for a_cell, b_cell, resp in zip(a_col, b_col, r_col):
        a = _to_vec(a_cell)
        b = _to_vec(b_cell)
        if a.size == 0 or a.shape != b.shape:
            continue

        # --- TTB winner: first discriminating cue in validity order ---
        ttb_winner = None
        for j in cue_order:
            if j >= a.size:
                continue
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break
        if ttb_winner is None:
            continue  # no discriminating cue: TTB guesses, uninformative

        # --- Tallying winner: strict feature-wise win counts ---
        a_wins = int(np.sum(a > b))
        b_wins = int(np.sum(b > a))
        if a_wins == b_wins:
            continue  # tally ties -> Tallying guesses at 0.5, dilutes contrast
        tally_winner = 0 if a_wins > b_wins else 1

        if tally_winner == ttb_winner:
            continue  # heuristics agree: uninformative

        # Pure conflict trial: TTB's option vs the tally majority's option.
        hits.append(1.0 if int(resp) == ttb_winner else 0.0)

    if not hits:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.2080 (var=0.0060)
**Predicted under pi_5_1:** 0.1870 (var=0.0059)
**Predicted under pi_4:** 0.2197 (var=0.0050)

### Experiment 4
**Design**
  A=[0, 1, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Tally-vs-TTB choice-allegiance contrast.

    For every trial we determine (i) the TALLYING winner (option with more
    strict feature-wise wins) and (ii) the TAKE-THE-BEST winner (winner on
    the highest-validity discriminating cue, validities fixed by design as
    [0.95, 0.9, 0.85, 0.65, 0.6, 0.5]).  The metric is

        mean( response == tally winner ) - mean( response == TTB winner )

    On the 4 trial types where both heuristics pick the same option
    (trials 1, 3, 5, 7 and mirrors) the two indicators are identical and
    cancel.  On the 12 conflict trial types they are exact complements, so
    the metric reduces to 0.75 * (2*P(tally winner | conflict) - 1): strongly
    positive under Tallying, strongly negative under TTB.
    """
    validities = np.array([0.95, 0.9, 0.85, 0.65, 0.6, 0.5])
    cue_order = np.argsort(-validities, kind="stable")

    a_col = data["option_a_ratings"].tolist()
    b_col = data["option_b_ratings"].tolist()
    r_col = data["response"].tolist()

    tally_match = []
    ttb_match = []

    for ca, cb, cr in zip(a_col, b_col, r_col):
        a = np.asarray(ca, dtype=float).ravel()
        b = np.asarray(cb, dtype=float).ravel()
        r = int(cr)

        # Tallying winner: more strict feature-wise wins (ties on a feature
        # contribute to neither side).
        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        if a_wins > b_wins:
            tally_winner = 0
        elif b_wins > a_wins:
            tally_winner = 1
        else:
            tally_winner = -1  # undiscriminating tally -> excluded

        # TTB winner: first discriminating cue in descending-validity order.
        ttb_winner = -1
        for j in cue_order:
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break

        if tally_winner >= 0:
            tally_match.append(1.0 if r == tally_winner else 0.0)
        if ttb_winner >= 0:
            ttb_match.append(1.0 if r == ttb_winner else 0.0)

    if len(tally_match) == 0 or len(ttb_match) == 0:
        return 0.0

    return float(np.mean(tally_match) - np.mean(ttb_match))
```

**Observed (real) value:** 0.1583 (var=0.0053)
**Predicted under pi_5_1:** 0.2875 (var=0.0062)
**Predicted under pi_4:** 0.2079 (var=0.0051)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd


def _to_vec(x):
    """Robustly convert a cell (list/tuple/ndarray/str) to a float 1-D array."""
    if isinstance(x, np.ndarray):
        return x.astype(float)
    if isinstance(x, (list, tuple)):
        return np.asarray(list(x), dtype=float)
    s = str(x).strip().replace("[", " ").replace("]", " ").replace(",", " ")
    return np.asarray(s.split(), dtype=float)


def metric(data: pd.DataFrame) -> float:
    df = data.reset_index(drop=True)
    n_rows = len(df)
    if n_rows == 0:
        return 0.0

    A = np.vstack([_to_vec(x) for x in df["option_a_ratings"]])
    B = np.vstack([_to_vec(x) for x in df["option_b_ratings"]])
    n_feat = A.shape[1]

    # Design validities [0.95, 0.85, 0.75, 0.65, 0.60]; fall back gracefully.
    v = np.array([0.95, 0.85, 0.75, 0.65, 0.60], dtype=float)
    if v.shape[0] != n_feat:
        v = np.linspace(0.95, 0.60, n_feat)

    # Chose-A indicator (response: 0 = A, 1 = B).
    y = (np.asarray(df["response"], dtype=float) == 0.0).astype(float)

    a_win = A > B
    b_win = B > A

    # Signed tally margin (Tallying's ONLY explanatory variable).
    margin = a_win.sum(axis=1).astype(float) - b_win.sum(axis=1).astype(float)

    # Validity-weighted evidence difference (NVW's driving quantity).
    d = (a_win * v).sum(axis=1) - (b_win * v).sum(axis=1)

    # OLS: y ~ 1 + margin + d. Return the partial coefficient on d.
    X = np.column_stack([np.ones(n_rows), margin, d])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[2])
```

**Observed (real) value:** -0.5756 (var=0.0186)
**Predicted under pi_5_1:** -0.5621 (var=0.0241)
**Predicted under pi_4:** -0.4120 (var=0.0294)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 1, 1, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0, 0, 1, 1]  B=[1, 0, 0, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Anchor-allegiance: proportion of trials with a large tally margin
    (|a_wins - b_wins| >= 6) on which the subject chose the option that
    wins on more features (the tally winner).

    Under Tallying, these are the unambiguous anchors (8-0 and 7-1 splits:
    design trials 7, 8, 21, 22, 23, 24 and their 4 repetitions = 24 trials
    per subject), where the softmax over tallies predicts near-deterministic
    choice for any non-trivial beta. Under NVW, the capacity-interference
    factor delta^4 crushes the effective inverse temperature to
    beta_eff <= 11 * 0.30^4 ~ 0.09, so even the maximum validity-weighted
    evidence difference (~6) yields P ~ sigmoid(0.5) <= 0.64; attention
    lapses and the uniform-mixing epsilon only pull this further toward
    0.5. The two theories therefore make sharply separated predictions on
    this proportion, and the A/B-mirrored anchors cancel side bias.
    """
    a_col = data['option_a_ratings']
    b_col = data['option_b_ratings']
    r_col = data['response']

    n_anchor = 0
    n_hit = 0

    for ra, rb, resp in zip(a_col, b_col, r_col):
        va = np.asarray(ra, dtype=float)
        vb = np.asarray(rb, dtype=float)
        if va.shape != vb.shape:
            continue

        a_wins = float(np.sum(va > vb))
        b_wins = float(np.sum(vb > va))
        margin = a_wins - b_wins

        if abs(margin) >= 6:
            n_anchor += 1
            winner_is_a = margin > 0
            chose_a = (int(resp) == 0)
            if chose_a == winner_is_a:
                n_hit += 1

    if n_anchor == 0:
        return 0.5
    return float(n_hit) / float(n_anchor)

```

**Observed (real) value:** 0.8433 (var=0.0152)
**Predicted under pi_5_1:** 0.8217 (var=0.0065)
**Predicted under pi_4:** 0.8133 (var=0.0065)

### Experiment 7
**Design**
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Validity-adherence rate on VALIDITY-vs-RECENCY conflict trials.
    #
    # For every trial we compute two decision variables from the presented
    # cue patterns:
    #   d_v = sum_j v_j * sign(A_j - B_j)   (validity-weighted evidence for A)
    #   d_r = sum_j (j/(n-1))^3 * sign(A_j - B_j)  (convex cue-order recency
    #                                              evidence for A)
    # A trial is a CONFLICT trial when the two point in opposite directions
    # and both are non-trivial in magnitude. On those trials the metric is
    # the proportion of choices that follow the VALIDITY-implied option.
    #
    # NVW (validity-weighted integration) predicts adherence clearly above
    # chance (~0.6); SWC (recency-dominated subjective weights) predicts
    # adherence well below chance (~0.3), because on these trials its
    # recency gradient favors the opposite option.

    validities = np.array([0.95, 0.90, 0.85, 0.62, 0.50, 0.50], dtype=float)
    n = int(validities.shape[0])
    # Convex recency profile over presentation positions (eta ~ 3).
    recency = (np.arange(n, dtype=float) / float(n - 1)) ** 3.0

    a_col = data["option_a_ratings"]
    b_col = data["option_b_ratings"]
    r_col = data["response"]

    hits = []
    for a_raw, b_raw, resp in zip(a_col, b_col, r_col):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        if a.shape[0] != n or b.shape[0] != n:
            continue
        s = np.sign(a - b)
        if not np.any(s != 0):
            continue

        d_v = float(np.dot(validities, s))
        d_r = float(np.dot(recency, s))

        # Need a real conflict with non-trivial evidence on both axes.
        if abs(d_v) < 0.30 or abs(d_r) < 0.50:
            continue
        if np.sign(d_v) == np.sign(d_r):
            continue

        validity_winner_is_a = (d_v > 0.0)
        chose_a = (int(resp) == 0)
        hits.append(1.0 if (chose_a == validity_winner_is_a) else 0.0)

    if len(hits) == 0:
        return 0.0
    return float(np.mean(hits))
```

**Observed (real) value:** 0.2313 (var=0.0046)
**Predicted under pi_5_1:** 0.2369 (var=0.0070)
**Predicted under pi_4:** 0.2816 (var=0.0052)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Recency-consistency rate: fraction of choices aligned with a convex
    # cue-order recency gradient, restricted to trials where that gradient
    # is strong and its direction is unambiguous across SWC's eta range.
    #
    # SWC (advocated): late cues dominate via convex recency -> rate well
    #   above 0.5 (predicted ~0.65-0.85 across its parameter box).
    # NVW (competing): validity-weighted integration with capacity
    #   interference (beta_eff = beta*delta^3 <= ~0.30 at n=7) -> choices
    #   near chance on nearly every trial, and anti-recency or neutral on
    #   the conflict trials -> rate ~0.50.
    df = data.reset_index(drop=True)
    if len(df) == 0:
        return 0.5

    a_arrs = [np.asarray(list(x), dtype=float) for x in df['option_a_ratings']]
    b_arrs = [np.asarray(list(x), dtype=float) for x in df['option_b_ratings']]
    resp = pd.to_numeric(df['response'], errors='coerce').to_numpy()

    if len(a_arrs) == 0 or a_arrs[0].ndim != 1:
        return 0.5
    n = int(a_arrs[0].shape[0])
    if n < 2:
        return 0.5

    pos = np.arange(n, dtype=float) / (n - 1.0)
    w_lo = pos ** 2.5   # SWC's eta lower corner
    w_mid = pos ** 3.5  # SWC's eta midpoint
    w_hi = pos ** 4.5   # SWC's eta upper corner

    consistent = []
    for a, b, r in zip(a_arrs, b_arrs, resp):
        if a.shape[0] != n or b.shape[0] != n:
            continue
        if not np.isfinite(r):
            continue
        s = np.sign(a - b)
        if not np.any(s != 0):
            continue
        e_lo = float(np.dot(w_lo, s))
        e_mid = float(np.dot(w_mid, s))
        e_hi = float(np.dot(w_hi, s))
        # Require the recency gradient's direction to be invariant across
        # SWC's whole eta range (drops ambiguous trials, e.g. 1 late cue vs
        # 4 mid cues, where the sign flips with eta).
        if e_lo == 0.0 or e_hi == 0.0 or np.sign(e_lo) != np.sign(e_hi):
            continue
        # Require non-trivial gradient magnitude (drops near-degenerate
        # trials where even SWC predicts ~0.52).
        if abs(e_mid) < 0.25:
            continue
        fav_a = e_mid > 0.0
        chose_a = (int(r) == 0)  # response: 0 = chose A, 1 = chose B
        consistent.append(1.0 if (chose_a == fav_a) else 0.0)

    if len(consistent) == 0:
        return 0.5
    return float(np.mean(consistent))
```

**Observed (real) value:** 0.3739 (var=0.0026)
**Predicted under pi_5_1:** 0.4392 (var=0.0036)
**Predicted under pi_4:** 0.7261 (var=0.0028)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Anti-validity vs. recency disagreement score.

    On trials where IVC's distrust-inverted weights and SWC's recency-dominated
    weights point to OPPOSITE options, report the proportion of choices that
    follow the IVC-preferred option. IVC-simulated data -> well above 0.5;
    SWC-simulated data -> well below 0.5.
    """
    data = data.reset_index(drop=True)
    if len(data) == 0:
        return 0.5

    # Build stimulus matrices without hashing any list-valued cells.
    a = np.vstack([np.asarray(r, dtype=float) for r in data["option_a_ratings"]])
    b = np.vstack([np.asarray(r, dtype=float) for r in data["option_b_ratings"]])
    n = a.shape[1]

    # Design validities (fallback mirrors both theories' graceful degradation).
    v = np.array([0.5, 0.9, 0.6, 0.95, 0.5, 0.6], dtype=float)
    if v.shape[0] != n:
        v = np.linspace(0.95, 0.55, n)

    # --- IVC weights at mid-range parameters (tau=0.925, kappa=2.7) ---
    tau, kappa = 0.925, 2.7
    inv = np.power(1.0 - v, kappa)
    Z = float(inv.sum())
    w_inv = inv / Z if Z > 1e-12 else np.full(n, 1.0 / n)
    w_ivc = (1.0 - tau) / n + tau * w_inv

    # --- SWC weights at mid-range parameters (lam=0.9, rho=0.9, eta=3.5) ---
    lam, rho, eta = 0.9, 0.9, 3.5
    vspan = float(v.max() - v.min())
    if vspan > 1e-12:
        vnorm = (v - v.min()) / vspan
    else:
        vnorm = np.full(n, 0.5)
    if n > 1:
        pos = (np.arange(n, dtype=float) / (n - 1.0)) ** eta
    else:
        pos = np.full(n, 0.5)
    g = (1.0 - rho) * vnorm + rho * pos
    w_swc = (1.0 - lam) / n + lam * g

    # Signed discriminating-cue vector per trial.
    s = np.sign(a - b)
    d_ivc = s @ w_ivc
    d_swc = s @ w_swc

    # Keep only trials where the two theories STRICTLY disagree in sign.
    # (d_ivc == 0, e.g. the validity-matched EQ trials, is excluded.)
    mask = (d_ivc * d_swc) < 0.0
    if not np.any(mask):
        return 0.5

    resp = np.asarray(data["response"].to_numpy())
    r = resp[mask]
    ivc_prefers_a = d_ivc[mask] > 0.0
    chose_ivc_preferred = np.where(ivc_prefers_a, r == 0, r == 1)
    return float(chose_ivc_preferred.mean())
```

**Observed (real) value:** 0.7878 (var=0.0086)
**Predicted under pi_5_1:** 0.6478 (var=0.0070)
**Predicted under pi_4:** 0.3622 (var=0.0067)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    """Final-Expert Dominance Index (FEDI).

    Among trials where the LAST-presented expert (feature position n-1) is the
    ONLY cue favoring one option (i.e., that option's rating vector is all
    zeros except a 1 in the final position), compute the proportion of
    choices that follow the final expert's recommendation.

    A convex-recency weighting (SWC) predicts adherence well above 0.5
    (the last cue carries the largest subjective weight at every corner of
    its parameter box). A distrust-inverted validity weighting (IVC)
    predicts adherence below 0.5, because the final expert here has a high
    stated validity (0.90) and is therefore heavily discounted.
    """
    df = data.reset_index(drop=True)

    A = np.array(
        [list(map(int, x)) for x in df['option_a_ratings']], dtype=int
    )
    B = np.array(
        [list(map(int, x)) for x in df['option_b_ratings']], dtype=int
    )
    r = df['response'].to_numpy(dtype=int)

    if A.ndim != 2 or A.shape[0] == 0:
        return float('nan')

    n_feat = A.shape[1]
    last = n_feat - 1

    a_last = A[:, last]
    b_last = B[:, last]

    # Last expert discriminates between the options.
    disc = a_last != b_last

    # The option endorsed by the final expert wins ONLY that final cue
    # (its rating vector sums to 1). This selects the single-cue and
    # multi-cue 'ladder' duels against the last expert (trials 1-14 and
    # their mirrors) and excludes trials where the last-cue option also
    # wins other cues.
    a_sel = disc & (a_last == 1) & (A.sum(axis=1) == 1)
    b_sel = disc & (b_last == 1) & (B.sum(axis=1) == 1)
    sel = a_sel | b_sel

    if not np.any(sel):
        return float('nan')

    # response: 0 = chose A, 1 = chose B.
    # Did the subject follow the final expert's endorsed option?
    followed = np.where(a_sel[sel], r[sel] == 0, r[sel] == 1)

    return float(np.mean(followed))
```

**Observed (real) value:** 0.3957 (var=0.0045)
**Predicted under pi_5_1:** 0.3393 (var=0.0058)
**Predicted under pi_4:** 0.6796 (var=0.0044)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all ten experiments, pi_5_1 (IVC) is the clearly superior theory, while pi_4 (SWC) is degenerate. Scoring absolute prediction error on every experiment: Exp1 (IVC 0.16 vs SWC 0.33), Exp2 (0.13 vs 0.44), Exp3 (0.02 vs 0.01), Exp4 (0.13 vs 0.05), Exp5 (0.01 vs 0.16), Exp6 (0.02 vs 0.03), Exp7 (0.01 vs 0.05), Exp8 (0.07 vs 0.35), Exp9 (0.14 vs 0.43), Exp10 (0.06 vs 0.28). IVC's cumulative error is roughly 0.84 versus SWC's 2.13, and \u2014 decisively \u2014 SWC gets the SIGN wrong on five experiments (1, 2, 8, 9, 10). SWC's core mechanism, a convex cue-order RECENCY gradient (late cues overweighted), is directly and repeatedly falsified: Experiment 1's recency-following rate on count-neutral trials is 0.341 (SWC requires ~0.67, and even IVC's parameter-free 0.50 is too high); Experiment 2's recency-gradient score is -0.173 (SWC requires +0.17..+0.41); Experiment 8's recency-consistency is 0.374 (SWC requires ~0.73); Experiment 10's final-expert dominance is 0.396 (SWC requires ~0.68). The data consistently show the OPPOSITE position gradient: a PRIMACY effect (early cues overweighted). Meanwhile IVC's core mechanism \u2014 distrust inversion of stated validities \u2014 is strongly confirmed: Experiment 5's validity-weighted partial coefficient is -0.576 (IVC: -0.562, nearly exact), Experiment 7's validity adherence on validity-recency conflict trials is 0.231 (IVC: 0.237, nearly exact; well below chance, i.e., subjects follow the WORSE-validated expert), and Experiment 9's IVC-preferred adherence is 0.788 (IVC: 0.648, right direction, under-predicted). However, IVC is not complete: because its weights depend only on validity VALUES, it is structurally incapable of producing any position effect, so it predicts exactly 0.50 wherever validities are equal (Exp1: predicts 0.50, observed 0.34) and under-predicts the strength of the anti-recency signature (Exp2: -0.046 vs -0.173; Exp8: 0.439 vs 0.374; Exp9: 0.648 vs 0.788 \u2014 where a primacy component would reinforce the anti-validity component). No parameter re-tuning within IVC's current description can fix these residuals, since the misses are structural (parameter-free flatness predictions). The correct synthesis is that BOTH observed regularities \u2014 anti-validity (distrust inversion) and anti-recency (primacy) \u2014 must be combined in a single weighting rule. Since SWC is the degenerate theory, it should be replaced by this synthesis rather than merely re-tuning IVC.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace SWC with PRIMACY-WEIGHTED INVERTED-VALIDITY COUNTING (PW-IVC). Core claim: subjects read the stated validities but distrust-invert them (acting as if stated validity is the expert's error rate, exactly as in IVC), AND independently overweight EARLY-presented cues via a convex primacy gradient (the mirror image of SWC's recency). Concretely: subjective weight w_j = (1-tau)/n + tau * [ (1-rho) * ((1-v_j) - c*min_k(1-v_k))^gamma / Z_v + rho * ((n-j)/(n-1))^eta / Z_p ], a mixture of flat counting, a partially-centered inverted-validity kernel, and a normalized convex primacy kernel; evidence d = sum_j w_j * sign(A_j - B_j); choice is logistic in beta*d plus a small uniform lapse epsilon. Key contrasts with the two existing theories: (i) versus SWC \u2014 the position gradient runs from LAST to FIRST (primacy, eta in ~[1.5, 4.5], rho in ~[0.15, 0.5], modest because the anti-validity signal carries most of the variance), so it predicts recency-following BELOW 0.50 on count-neutral equal-validity trials (Exp1: target ~0.34), a negative recency-gradient score (Exp2: target ~-0.17), recency-consistency below 0.5 (Exp8: target ~0.37) and final-expert dominance below 0.5 (Exp10: target ~0.40, since the final expert is both late-positioned and high-validity, so primacy and distrust both discount it); (ii) versus IVC \u2014 the added primacy kernel breaks IVC's parameter-free flatness wherever validities are tied, and on designs where the low-validity expert appears EARLY (as in Exp9's layout [0.5, 0.9, 0.6, 0.95, 0.5, 0.6], where inverted-validity mass and primacy mass are positively correlated on early positions) the two components reinforce, raising IVC-preferred adherence from IVC's 0.648 toward the observed 0.788. Keep tau high (~[0.80, 0.95]) so the anti-validity kernel continues to nail Exp5 (-0.576) and Exp7 (0.231), and keep beta moderate (~[1.8, 5.0]) with epsilon ~[0.08, 0.22] to preserve the Exp6 anchor (~0.84) and the Exp3/Exp4 TTB-vs-tally levels (~0.21, ~0.16). This theory is the only one that can simultaneously explain the anti-validity signature (Exp 5, 7, 9), the anti-recency/primacy signature (Exp 1, 2, 8, 10), and the anchor/tally behavior (Exp 3, 4, 6), making it a genuinely stronger competitor to the surviving IVC rather than a re-parameterization of a falsified recency mechanism."
}
```

## Usage

```json
{
  "input_tokens": 25317,
  "output_tokens": 3555,
  "reasoning_tokens": 2117,
  "provider": "Sail Research"
}
```
