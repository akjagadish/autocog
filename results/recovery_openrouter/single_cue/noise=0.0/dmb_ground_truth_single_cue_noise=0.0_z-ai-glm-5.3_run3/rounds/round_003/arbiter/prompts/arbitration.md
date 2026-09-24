# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_5" and "pi_4") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_5" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_4" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_5
INVERTED-VALIDITY COUNTING (IVC) — distrust-weighted tallying. Subjects do read the communicated validities but map them through a systematic distrust inversion: they act as if each stated validity were the expert's error rate, so the most highly validated experts are discounted most. Every discriminating cue is attended deterministically (no attention lapses, no working-memory capacity interference — choice consistency is design-independent). Each cue j carries subjective weight w_j = (1-tau)/n + tau*(1-v_j)^kappa / Z, a mixture of flat counting and a normalized inverted-validity kernel; tau is the trust placed in the inversion, kappa its steepness (kappa=0, tau=0 recovers plain Tallying; large kappa approaches a worst-expert-first one-reason rule). Evidence is integrated additively, d = sum_j w_j * sign(A_j - B_j), and the choice is a logistic function of beta*d plus a small uniform lapse epsilon. Because the anti-validity signature is grounded in the communicated validity values rather than in cue position, it transfers across designs with different validity-position layouts — unlike SWC's recency gradient, which sign-failed the recency-consistency experiment.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # IVC: Inverted-Validity Counting (distrust-weighted tallying).
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
    kappa = float(parameters["kappa"])
    beta = float(parameters["beta"])
    eps = float(parameters["epsilon"])

    # (1) Distrust inversion: treat the communicated validity as if it
    # were the expert's error rate. (1 - v_j)^kappa is small for highly
    # validated experts and large for weakly validated ones; Z normalizes
    # the kernel so the mixture weights sum to one.
    inv = np.power(1.0 - v, kappa)
    Z = float(inv.sum())
    if Z > 1e-12:
        w_inv = inv / Z
    else:
        w_inv = np.full(n, 1.0 / n)

    # Subjective weights: mixture of flat counting and inverted validity.
    # tau = 0 recovers plain Tallying; tau = 1 with large kappa approaches
    # a 'worst-expert-first' one-reason rule.
    w = (1.0 - tau) / n + tau * w_inv

    # (2) Deterministic attention: every discriminating cue is always
    # attended. NO capacity interference: beta is design-independent
    # (anchor consistency stays high even at n = 8 features).
    s = np.sign(a - b)

    # (3) Additive integration of all discriminating cues.
    d = float(np.dot(w, s))

    # Numerically stable logistic choice rule.
    p_a = 0.5 * (1.0 + np.tanh(0.5 * beta * d))
    probs = np.array([p_a, 1.0 - p_a], dtype=float)

    # (4) Small independent uniform lapse.
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

## EXPERIMENT 1 (proposed by pi_5)

### DESIGN
**Validities (n_features=6):** [0.5, 0.9, 0.6, 0.95, 0.5, 0.6]

**Trial pairs (n=32):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 2: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 3: A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 4: A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 1, 0]
  trial 5: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 6: A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 7: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 8: A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 9: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 11: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 12: A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  trial 13: A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  trial 14: A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0]
  trial 15: A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 16: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 17: A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 18: A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0]
  trial 19: A=[0, 0, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 20: A=[0, 0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  trial 21: A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 22: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 23: A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 24: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 25: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 26: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 27: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 28: A=[0, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 29: A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  trial 30: A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 0]
  trial 31: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 32: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Rationale:** STRUCTURAL LEVER. IVC's weights are a function of the validity VALUES ONLY (w_j proportional to (1-v_j)^kappa: the LESS valid the expert, the MORE weight), with zero position sensitivity. SWC's weights are dominated by POSITION (convex recency, coefficient lam*rho >= 0.64, plus a weak POSITIVE validity component <= 0.20). I therefore use a non-monotone, duplicated validity profile v = [0.55, 0.90, 0.60, 0.95, 0.55, 0.60] (n=6): the two lowest validities sit at positions 1 and 5, duplicated 0.55s at positions 1/5 and duplicated 0.60s at positions 3/6, and the two highest validities at positions 2 and 4. The duplicates create validity-MATCHED comparisons at different positions (killing the validity axis and isolating pure recency), and the low-early/high-late arrangement makes IVC's weight profile (early-heavy) anti-aligned with SWC's (late-heavy). All boxes below were verified at the corners of BOTH parameter spaces.

AXIS 1 - VALIDITY-MATCHED RECENCY BATTERY (EQ1-EQ4 + mirrors). EQ1: A wins expert 1 vs B wins expert 5 (0.55 vs 0.55). EQ2: expert 3 vs expert 6 (0.60 vs 0.60). EQ3: A wins {1,3} vs B wins {5,6} (validity multiset {0.55,0.60} on BOTH sides). EQ4: A wins {1,6} vs B wins {3,5} (same matched multiset). Because IVC's weights depend only on validity values, d = 0 EXACTLY on all four -> IVC predicts P(A) = 0.50 PARAMETER-FREE. SWC's recency gradient predicts a wide spread: P(A) boxes [0.28,0.45] (EQ1), [0.12,0.39] (EQ2), [0.08,0.34] (EQ3), [0.61,0.69] (EQ4). Observed P(EQ3) <= 0.40 or P(EQ4) >= 0.55 is impossible for IVC at ANY parameter combination; observed P(EQ3) >= 0.45 is impossible for SWC. SWC requires an EQ3-vs-EQ4 spread >= 0.27 where IVC requires exactly 0.

AXIS 2 - POSITION-TRANSFER PAIRS (IVC invariance vs SWC flip). TP1: A wins expert 1 (0.55) vs B wins expert 6 (0.60). TP2: A wins expert 5 (0.55) vs B wins expert 3 (0.60). Identical validity allegiances, opposite position allegiances: IVC predicts EXACTLY the same P(A) on both, in [0.52, 0.55]; SWC predicts P(TP1) in [0.12, 0.37] (expert 6's recency dominates) and P(TP2) in [0.54, 0.69] (expert 5 beats expert 3 on recency) - a required pair difference >= 0.17 where IVC requires exactly 0. The same logic at the extreme validity gap: TP3 (A wins expert 1 vs B wins expert 4, 0.55 vs 0.95), TP4 (A wins expert 5 vs B wins expert 4), and M1 (A wins {1,3} vs B wins {4,6}, where the two 0.60 cues cancel) all have d = w(0.55) - w(0.95) under IVC -> P(A) IDENTICAL in [0.59, 0.66] on all three; SWC predicts [0.32,0.47], [0.50,0.57], [0.08,0.33] respectively - a required spread >= 0.17 (TP4 vs M1). TP3'/TP4' repeat this with the 0.90 expert: IVC-equal [0.59,0.66]; SWC [0.42,0.50] vs [0.52,0.69]. Any observed spread within a transfer family (e.g., P(TP1) != P(TP2), or P(TP3) != P(M1)) falsifies IVC's position-invariance; flatness within families falsifies SWC's recency gradient.

AXIS 3 - SIGN-FLIP DUELS. On TP1, TP3, TP3', S3 (A wins expert 3 (0.60) vs B wins expert 4 (0.95) - an ADJACENT-position duel where the later cue is also more valid), M1 and M2 (A wins {1,3,5}, the three least valid experts, vs B wins {2,4,6}), IVC favors the early/low-validity option and SWC favors the late/high-validity option with NON-OVERLAPPING boxes: IVC [0.52,0.55] vs SWC [0.12,0.37] (TP1); IVC [0.59,0.66] vs SWC [0.32,0.47] (TP3); IVC [0.59,0.66] vs SWC [0.42,0.50] (TP3'); IVC [0.57,0.61] vs SWC [0.38,0.48] (S3); IVC [0.59,0.66] vs SWC [0.08,0.33] (M1); IVC [0.63,0.73] vs SWC [0.12,0.37] (M2). M2 is the headline block flip: the option endorsed by the three WORST experts (55/60/55) is chosen confidently only by IVC; SWC confidently picks the option endorsed by the later, more valid experts.

AXIS 4 - CUE MIGRATION (AN2 -> M2). AN2: A wins {1,3,5} vs B wins {2,4}. Adding expert 6 (a mere 60%-valid cue) to B's side yields M2. IVC predicts only a modest drop (AN2 [0.70,0.78] -> M2 [0.63,0.73], still confidently A, since expert 6's inverted weight is only ~0.21 and expert 3's cancels it); SWC predicts a catastrophic flip (AN2 [0.43,0.66] -> M2 [0.12,0.37]) because expert 6 carries the maximal recency weight. A drop exceeding ~0.3 on this single-cue migration is impossible for IVC and required by SWC.

AXIS 5 - REVERSE CONFIDENCE DISSOCIATION. AN2 (A wins {1,3,5} vs B wins {2,4}): IVC [0.70,0.78] vs SWC [0.43,0.66] - IVC more confident (and SWC's box straddles chance). AN3 (A wins {5,6} vs B wins {1,2}): IVC [0.56,0.60] vs SWC [0.66,0.86] - SWC MORE confident. The confidence ordering between AN2 and AN3 reverses between theories: IVC says P(AN2) >> P(AN3); SWC says P(AN3) > P(AN2). TOT (total split) is the agreement anchor both theories accommodate.

PARAMETRIC SIGNATURE. IVC predicts only ~7 distinct probability levels across all 32 pairs (0.50 on all EQ; one level per validity-allegiance multiset for the TP/S/M/AN families; near-1 on TOT), with EXACT equalities within families; SWC predicts a wide, position-ordered spread that breaks every equality. Regressing P(A) on inverted-validity evidence sum((1-v_j)^2.7 * s_j) versus recency evidence sum((j/5)^3.5 * s_j) yields opposite-loading profiles: the two evidence axes are decorrelated by construction (expert 1: maximal inverted weight, zero recency; expert 6: maximal recency, middling inverted weight; experts 2/4: high validity, i.e., low weight for IVC but positive validity weight for SWC). Every trial type appears in A/B-mirrored form to absorb side and position bias. 32 unique pairs, K = 96 // 32 = 3 repetitions in independently randomized order = exactly 96 trials per subject.

**Computed schedule:** 32 unique pairs × 3 reps = 96 trials per subject.



### METRIC
Rationale:
CORE IDEA. The design's duplicated, non-monotone validity profile makes IVC's weights (a function of validity VALUES only: heaviest on the 0.5-validity experts 1/5, then 0.6 experts 3/6, near-zero on 0.9/0.95 experts 2/4) and SWC's weights (dominated by late POSITION: expert 6 >> 5 > 4 > 2 > 3 > 1) point to OPPOSITE options on a fixed, parameter-robust set of trials. The metric recomputes both theories' evidence d = w.s from the raw ratings using mid-range parameters, keeps only trials where sign(d_IVC) and sign(d_SWC) strictly disagree, and returns the proportion of choices following the IVC-preferred option. PARAMETER ROBUSTNESS. I verified the sign of d under BOTH theories at all corners of both parameter spaces: the disagreement set is exactly 12 of the 32 trial types (trials 5, 7, 9, 11, 12, 13, 21, 23, 25, 27, 28, 29 in the design listing), i.e. 36 of each subject's 96 trials, independent of tau/kappa/beta/epsilon (IVC) and lam/rho/eta/beta/epsilon (SWC). For example, w1 < w2 always under SWC (position/validity gradient) while w(0.5) > w(0.9) always under IVC; trial 13 (A wins experts 1,3,5 vs B wins 2,4,6) pits the three least-valid experts against the three latest-position experts — IVC favors the low-validity side, SWC the late side, at every parameter corner. So the mid-parameter weights used inside the metric only need to recover the SIGNS, which are invariant. EXPECTED SEPARATION. Under IVC (mid params) the per-trial P(choose IVC-preferred) on the 12 disagreement types averages ~0.65 (rising to ~0.78 on trials 13/29); under SWC it averages ~0.36. Pooled point estimates: ~0.64 (IVC data) vs ~0.36 (SWC data), a gap of ~0.28. VARIANCE. Per subject the metric is a proportion over 36 Bernoulli trials with p near 0.65 (or 0.36), giving per-subject SD ~0.08; parameter heterogeneity across subjects adds little (the boxes are tight). Welch's t at N=25 with variances ~0.006-0.008 on each side yields t > 9, p far below 0.01. The metric works identically on a single subject's slice (36 trials, no NaN), so the between-subject variance is well defined. INTERPRETATION. A value near 0.5 means neither anti-validity nor recency dominates; >0.5 supports distrust-weighted (inverted-validity) counting; <0.5 supports position/recency-weighted counting. Because agreement trials (e.g., the total splits 16/32, and all validity-matched EQ trials where IVC is exactly at chance) are excluded, the metric carries pure discriminating signal with minimal noise.

Source:
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

### RESULTS
- Predicted under pi_5 (simulated): 0.6389 (var=0.0060)
- Predicted under pi_4 (simulated): 0.3622 (var=0.0067)
- Observed on real data: 0.7878 (var=0.0086)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=6):** [0.85, 0.95, 0.65, 0.75, 0.6, 0.9]

**Trial pairs (n=24):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 2: A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 3: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 4: A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  trial 5: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 6: A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0]
  trial 7: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 8: A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0]
  trial 9: A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 10: A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  trial 11: A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 12: A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0, 0]
  trial 13: A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 14: A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  trial 15: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 16: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 17: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 18: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 19: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 20: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 21: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 22: A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 23: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 24: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Rationale:** STRUCTURAL LEVER. SWC's weights are dominated by POSITION (convex recency, coefficient lam*rho >= 0.64, plus a weak POSITIVE validity component <= 0.20); IVC's weights are a function of the validity VALUES ONLY through the distrust kernel (1-v_j)^kappa, with zero position sensitivity. Prior designs anti-correlated the two profiles via strictly DECREASING validities ([4], vs NVW) or broke them apart with non-monotone duplicated validities ([6]). This design takes the unexplored route: STRICTLY INCREASING validities v = [0.55, 0.60, 0.68, 0.80, 0.90, 0.96]. Now SWC's recency gradient and its weak validity component ALIGN (both favor late cues), so SWC's weight profile is late-heavy at EVERY corner of its box and its predictions are maximally extreme; IVC's inverted kernel is early-heavy (the 55% expert alone carries ~42% of total weight at kappa=2.4). The two weight profiles are therefore near-perfectly ANTI-ALIGNED on every conflict trial. All boxes below were verified at the corners of BOTH parameter spaces; the sign arguments are parameter-free (IVC's d for any 1-vs-1 duel is tau*(inv_A - inv_B)/Z, strictly decreasing in the early cue's validity since the kernel is monotone and flats cancel; SWC's duel d = w_j - w_6 is strictly increasing in j because BOTH vnorm_j and pos_j increase with j for every admissible parameter combination).

AXIS 1 - VALIDITY-GAP LADDER, OPPOSITE-SIGNED SLOPES (L1-L5 + mirrors). A wins exactly one early cue (c1..c5) against B's win of the last cue c6 (96%). IVC predicts a monotone DECREASING series in P(A): 0.64-0.75 (c1, 55%), 0.59-0.68 (c2), 0.55-0.60 (c3), 0.52-0.54 (c4), 0.50-0.51 (c5) - as the early opponent becomes more valid, its inverted weight shrinks toward c6's. SWC predicts a monotone INCREASING series: 0.15-0.35, 0.15-0.34, 0.15-0.36, 0.18-0.40, 0.20-0.44 - as the early opponent becomes later, its recency weight grows. Every adjacent rung has non-overlapping boxes, and the two theories demand opposite-signed slopes across the whole ladder - a parameter-free quantitative signature. L5 is the sharpest single trial: the two BEST experts (90% vs 96%) duel; IVC predicts exactly chance (both inverted weights are tiny), SWC predicts P(later expert's product) in [0.56, 0.80].

AXIS 2 - CROSSOVER COUNT AGAINST THE LAST CUE (L1, M3, M4 + mirrors). How many early-cue wins does it take to flip choice toward the early option? IVC's answer: k* = 1 (P(A) already >= 0.64 with the single 55% cue, L1). SWC's answer: k* >= 5 or never - with FOUR early cues (M3: A wins c1-c4 vs c6) SWC still predicts P(A) in [0.18, 0.44] while IVC predicts [0.82, 0.90]; with ALL FIVE early cues (M4) SWC's box only reaches [0.29, 0.61] (it straddles chance because the last cue's convex recency weight rivals five early cues combined) while IVC predicts [0.82, 0.92]. The measured crossover point separates the theories by 3-4 cues.

AXIS 3 - DOUBLE DISSOCIATION IN CONFIDENCE (W1 vs M4 + mirrors). W1: the product endorsed by the five LATER, MORE valid experts (incl. 90% and 96%) vs the single WORST expert (55%, first position). M4: the product endorsed by the five EARLIER, LESS valid experts vs the single BEST expert (96%, last position). IVC: W1 in [0.52, 0.60] (the lone 55% expert soaks up ~42% of the weight) but M4 in [0.82, 0.92] -> P(W1) - P(M4) is NEGATIVE everywhere (<= -0.21). SWC: W1 in [0.79, 0.97] but M4 in [0.29, 0.61] -> the difference is POSITIVE everywhere (>= +0.18). Opposite-signed differences with disjoint boxes on both trials; note M4 is deliberately SWC-risky (observed P(A) >= 0.65 on M4 falsifies SWC's entire space), making the design genuinely adversarial rather than one-sided.

AXIS 4 - BLOCK FLIP (B1 + mirror). A wins the three early low-validity experts, B the three late high-validity experts. IVC: [0.77, 0.90] for the early block; SWC: [0.05, 0.28]. Non-overlapping.

AXIS 5 - ADJACENT MICRO-DUELS (A1, A2 + mirrors). c3(68%) vs c4(80%) and c4(80%) vs c5(90%): small but sign-clean opposite tilts. IVC: P(earlier/lower-validity product) in [0.55, 0.58] (A1) and [0.50, 0.53] (A2); SWC: [0.42, 0.48] and [0.39, 0.45]. Parameter-free opposite signs on every adjacent duel - a micro-sign series that neither theory can reverse anywhere in its box.

E1/E1m are total-split agreement anchors establishing that subjects can perform the task. Aggregate signature: regressing P(A) on inverted-validity evidence sum((1-v_j)^2.7 * s_j) versus recency evidence sum((j/5)^3.5 * s_j) yields opposite-loading profiles, since the two evidence axes are anti-correlated by construction here. IVC predicts a validity-allegiance-determined probability profile (~8 distinct levels, invariant to which positions carry which allegiances); SWC predicts a position-ordered profile that breaks every such invariance. Every conflict type appears in A/B-mirrored form to absorb side and position bias. 24 unique pairs, K = 96 // 24 = 4 repetitions in independently randomized order = exactly 96 trials per subject.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



### METRIC
Rationale:
FEDI = P(follow the last-presented expert) over the 14 unique 'last-expert duels' (trials 1-14 plus mirrors; 56 trials per subject), where the option endorsed by the final expert wins only that cue (selection rule: last feature discriminates and that option's rating vector sums to 1).

Why this contrast is sharp and sign-clean across the FULL parameter boxes:

SWC weights: w_j = (1-lam)/6 + lam*[(1-rho)*vnorm_j + rho*(j/5)^eta]. At every corner (lam in [0.8,1], rho in [0.8,1], eta in [2.5,4.5]) the last cue's weight w_6 strictly dominates every other single weight and even the sum of the first four (e.g., corner lam=1,rho=1,eta=4.5: w=[0, .001, .016, .106, .366, 1]; corner lam=.8,rho=.8,eta=2.5: w=[.148, .204, .121, .281, .400, .810]). So SWC's P(follow last expert) is above 0.5 on every selected trial type: single-cue duels ~0.58-0.85, M3 (four early cues vs c6) ~0.51-0.74, M4 (five early cues vs c6) ~0.42-0.71. Pooled SWC FEDI ≈ 0.68 (box range ~0.57-0.79).

IVC weights: w_j = (1-tau)/6 + tau*(1-v_j)^kappa/Z with v_6 = 0.90, so the last expert is nearly discarded (w_6 ≈ 0.01-0.04 of ~0.17-0.25 total), while the early low-validity experts (v=0.65, 0.60) soak up most weight. IVC's P(follow last expert) is below 0.5 on every selected trial type: single-cue duels ~0.30-0.51, M3 ~0.23-0.29, M4 ~0.08-0.16. Pooled IVC FEDI ≈ 0.36 (box range ~0.32-0.39, very tight because the inverted kernel is steep and tau is near 1).

Statistical properties at N=25 humans: between-theory mean gap ≈ 0.32; per-subject the metric averages 56 Bernoulli trials (binomial sd ≈ 0.063), plus modest parameter heterogeneity (SWC sd ≈ 0.07, IVC sd ≈ 0.02), giving per-subject sd ≈ 0.09 under SWC and ≈ 0.07 under IVC. Welch t ≈ 13, p << 0.001 — comfortably below alpha = 0.01 even at the most adversarial corners of either box (worst-case gap ≈ 0.18-0.25 still yields t ≈ 7-9). The metric works identically on the pooled DataFrame and on each single subject's 96-trial slice (each subject contributes exactly 56 selected trials), so between_subject_variance is well defined.

Interpretation: FEDI is a direct behavioral measure of how much the final, most-recently-presented expert drives choices when it is opposed by earlier experts. SWC (recency-weighted counting) demands high adherence; IVC (distrust inversion of the communicated 90% validity) demands low adherence. The two theories' predictions are non-overlapping on essentially every selected trial type.

Source:
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

### RESULTS
- Predicted under pi_5 (simulated): 0.3561 (var=0.0033)
- Predicted under pi_4 (simulated): 0.6796 (var=0.0044)
- Observed on real data: 0.3957 (var=0.0045)

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
**Predicted under pi_5:** 0.1807 (var=0.0022)
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
**Predicted under pi_5:** 0.2308 (var=0.0086)
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
**Predicted under pi_5:** -0.5397 (var=0.0203)
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
**Predicted under pi_5:** 0.8083 (var=0.0061)
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
**Predicted under pi_5:** 0.2544 (var=0.0025)
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
**Predicted under pi_5:** 0.4408 (var=0.0038)
**Predicted under pi_4:** 0.7261 (var=0.0028)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all eight experiments, IVC (pi_5) is clearly the more task-invariant theory. On the two adversarial designs (Experiments 1 and 2), each proposed by one theory to kill the other, the observed data land decisively on IVC's side: Experiment 1 (anti-validity vs. recency disagreement score) observed 0.7878 vs. IVC 0.6389 / SWC 0.3622 \u2014 SWC is off by ~0.43 in the wrong direction while IVC is off by only ~0.15; Experiment 2 (FEDI, follow-the-last-expert) observed 0.3957 vs. IVC 0.3561 / SWC 0.6796 \u2014 again SWC misses by ~0.28 and IVC by ~0.04. On the shared experiments, IVC also wins or ties: Experiment 5 (observed -0.5756 vs. IVC -0.5397 / SWC -0.4120), Experiment 7 (observed 0.2313 vs. 0.2544 / 0.2816), Experiment 8 (observed 0.3739 vs. 0.4408 / 0.7261 \u2014 SWC is catastrophically wrong here, predicting strong recency-consistency where subjects show anti-recency), and Experiment 6 is a near-tie (0.8433 observed vs. 0.8083 / 0.8133). SWC's only marginal advantages are Experiments 3 and 4 (differences of ~0.02\u20130.07), where both theories are roughly equally adequate. The pattern is unambiguous: SWC's core mechanism \u2014 convex cue-order recency \u2014 is falsified whenever position is decorrelated from validity (Exp 1) or when recency is directly measured (Exp 8: observed 0.374 vs. SWC 0.726). IVC's core mechanism \u2014 distrust-inverted validity weighting \u2014 is directionally correct on every experiment. However, IVC is systematically miscalibrated in magnitude: in Experiment 1 the observed anti-validity adherence (0.7878) exceeds IVC's prediction (0.6389), and in Experiment 8 the observed anti-recency (0.3739) is more extreme than IVC's (0.4408), while in Experiments 2 and 4 IVC slightly overshoots (0.3561 vs. 0.3957; 0.2308 vs. 0.1583). These are quantitative mis-calibrations of the inversion strength (tau, kappa) and noise (beta, epsilon) rather than structural failures. Since the theory's qualitative signature (validity-value-driven, position-invariant, anti-validity weighting with deterministic attention) survives every test, the right move is not to replace it but to regenerate its parameterization so the inversion kernel and choice-noise ranges capture the observed extremeness \u2014 particularly in designs where validities are high and compressed (Exp 2: v up to 0.95), where the current (1-v)^kappa kernel can saturate too softly.",
  "verdict": "new_model",
  "target_theory_idx": 1,
  "recommendation": "Keep the IVC theory description (distrust-weighted tallying with deterministic attention and additive integration) but regenerate the predict/policy and parameter ranges as follows: (1) Strengthen and re-shape the inversion kernel: use a normalized kernel over the *centered* error rates, e.g. inv_j = ((1 - v_j) - min_k(1 - v_k))^gamma / Z with gamma in [2.0, 6.0] and tau in [0.85, 1.0], so that even in high-validity designs (Exp 2's 0.95/0.90 experts) the most-trusted experts receive essentially zero weight and the least-trusted expert absorbs a large share \u2014 this pushes Exp-1-style disagreement scores from ~0.64 toward the observed ~0.79 and Exp-8-style recency-consistency from ~0.44 toward ~0.37. (2) Widen the choice-noise and lapse ranges: beta in [4, 40], epsilon in [0.02, 0.12], so that the model can simultaneously produce near-deterministic anchor behavior (Exp 6, observed 0.843) and the softer aggregate behavior on Exp 2/Exp 4 (observed FEDI 0.396, tally-vs-TTB 0.158) \u2014 the current ranges force a single compromise. (3) Add subject-level heterogeneity: sample (tau, gamma, beta, epsilon) per subject from the ranges (e.g., tau ~ Beta concentrated near 0.95) so the between-subject variance of metrics (~0.005\u20130.019 observed) is reproduced rather than under-estimated. (4) Optionally include a tiny idiosyncratic per-cue bias term (fixed per subject, mean zero, sd ~0.02) to soak up residual side/cue preferences without reintroducing position sensitivity \u2014 this must remain small enough that IVC's parameter-free equalities (e.g., d = 0 on validity-matched EQ trials) still hold approximately. (5) Verify the regenerated parameter box against all eight archived experiments: the corner predictions should bracket every observed value (0.7878, 0.3957, 0.2080, 0.1583, -0.5756, 0.8433, 0.2313, 0.3739) rather than a single mid-range compromise."
}
```

## Usage

```json
{
  "input_tokens": 21298,
  "output_tokens": 2439,
  "reasoning_tokens": 1195,
  "provider": "Sail Research"
}
```
