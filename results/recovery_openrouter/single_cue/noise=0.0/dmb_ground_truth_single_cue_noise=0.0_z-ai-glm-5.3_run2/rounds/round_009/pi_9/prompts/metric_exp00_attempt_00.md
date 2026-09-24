# metric_exp00_attempt_00

## System Prompt

You are a psychology researcher proposing a metric in the Decision Making (Binary Features) domain.

Your goal is adversarial: propose a metric that DISCRIMINATES the two theories — i.e., its value, computed on data simulated under your advocated theory, should be as far as possible from its value computed on data simulated under the competing theory. The direction of the gap does not matter; what matters is that the two theories produce visibly different numbers on this metric. The metric is computed on the data collected from the experimental design provided in the prompt. Produce a metric where you're prediction will be much more accurate than the competing theory's prediction on human data.

Your metric is a Python function

    metric(data: pd.DataFrame) -> float

Available imports inside `metric`:
- numpy as np
- pandas as pd

The system evaluates your metric in two ways and reports the pair as `point_estimate (var=between_subject_variance)` everywhere downstream:
- `point_estimate` is `metric(data)` applied to the FULL pooled DataFrame (all subjects together) — the canonical scalar;
- `between_subject_variance` is the population variance (`ddof=0`) of `metric(subj_df)` re-applied per `subject_id`, summarising how stable the metric is across subjects. If your metric only makes sense on multi-subject data this will fall back to `n/a` and the metric is rejected (the acceptance test below cannot run without it). Prefer metrics that work both on the pooled DataFrame and on a single subject's slice.

Acceptance rule: the system simulates each theory and runs Welch's two-sample t-test on `(point_estimate_self, between_subject_variance_self, N)` vs. `(point_estimate_adv, between_subject_variance_adv, N)`, where N is the number of HUMAN subjects the experiment will actually be run with (a fixed small number, currently 25). Your metric is admitted iff the two-sided p-value is below the significance level (currently alpha=0.01). Implication: a large between-theory gap is NOT enough — if either theory's metric is also highly variable across subjects, N humans won't reliably distinguish them and the metric will be rejected. Aim for contrasts that are both large in mean AND tight per subject.

Do NOT propose metrics that are trivially true for your theory.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

Each subject completes ~96 trials in a single block, with order randomized independently per subject. On every trial the subject sees two options A and B, each described by `n_features` binary expert ratings (each 0 or 1). The per-feature validities and n_features are fixed per experiment (design-time choices). Validities are communicated to the subject in the instructions. Both `n_features` and `validities` are exposed to your `predict` via the `parameters` dict. The subject chooses A or B; no correctness feedback is provided after the choice.

## CHOSEN EXPERIMENTAL DESIGN
**Validities (n_features=6):** [0.5, 0.6, 0.7, 0.7, 0.8, 0.9]

**Trial pairs (n=16):**
  trial 1: A=[0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 2: A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 0, 0]
  trial 3: A=[0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 4: A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 0]
  trial 5: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 6: A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 8: A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  trial 9: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 10: A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 11: A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 13: A=[0, 0, 1, 1, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  trial 14: A=[0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 15: A=[0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]
  trial 16: A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]

**Rationale:** GOAL: Quantitatively dissociate Diffuse Mixture Frugality with Strong Interior-Gated Validity Leak, sub-linear coherence variant (advocated) from Anti-Validity Frugality with Diffuse Positional-Salience Heterogeneity, momentary-diffusion variant (competing). The two theories share EVERYTHING in their core: the anti-validity one-reason cascade, the A^3-gated diffuse heterogeneity, the gamma tilt range, the tie-anchoring rule, and overlapping (beta, epsilon). The advocated's ONLY unique mechanism relevant here is the INTERIOR-GATED VALIDITY LEAK: on trials where the anti cascade's first discriminating expert (the carrier) carries a STRICTLY interior validity (min+0.10 < val < max-0.10), it mixes toward the TTB (descending-validity) decider with q = rho_leak*sqrt(coherence)*gap_term. The competing has no such mechanism: its follow probability is p_f toward the anti decider on EVERY discriminating trial, flat over carrier validity, opponent identity, margin, and coalition. Experiment [18] tested the leak in a palindromic arrangement where the primacy gate was simultaneously fully open (ambiguity 1.0) and gap_term was 0.5; this design occupies a different corner and adds a new, sharper axis: the INTERIORITY WINDOW LADDER with carriers sitting EXACTLY on both gate boundaries. SETUP: 6 features, validities [0.50, 0.60, 0.70, 0.70, 0.80, 0.90] (f0..f5), monotone ascending with one tied interior pair. Arrangement statistics (both models' own formulas, average-tie ranks [0, 1, 2.5, 2.5, 4, 5]): Spearman(position, validity) = 0.986, so binding ambiguity = 0.0145 — far below the 0.80 threshold, so the advocated's tie-anchored primacy gate is OFF (isolating the leak alone, unlike [18]); the exact ties switch OFF both theories' diffuse distortion (both require all-distinct validities); and every cross-group validity gap (0.10) exceeds the maximum gamma*salience swing per adjacent pair (0.02), so BOTH theories consult the identical deterministic anti cascade f0(50%) -> f1(60%) -> {f2,f3}(70%) -> f4(80%) -> f5(90%), with TTB its exact reverse and tie_break inert (the tied pair is kept concordant in every trial). A bonus of monotonicity: reading order = anti order, so any positional-primacy strategy concurs with the anti core — the design is structurally immune to primacy confounds. FORCED LEAK COMMITMENT: reversal coherence = |Spearman(val, val[::-1])| = 1.0 EXACTLY (symmetric tied ranks), gap_term = mean_gap/range = 0.10/0.40 = 0.25, so q = rho_leak*1.0*0.25 is FORCED into [0.2125, 0.3125] whenever the carrier is strictly interior — no parameter setting can switch it off. The interior window is (0.60, 0.80): ONLY the doubled 0.70 group is interior; the 0.60 and 0.80 experts sit EXACTLY on the window boundaries, excluded by the strict inequality — a parameter-free knife-edge on both sides. CARRIER-RUNG LADDER (each rung = carrier discriminates toward one option while f5 or f4 discriminates toward the other, all lower cues tied): carrier 0.50 (min: no leak), 0.60 (lower boundary: no leak), 0.70 x2 (interior: LEAK, q in [0.21, 0.31]), 0.80 (upper boundary: no leak), 0.90 (max: TTB concurs). SIGNATURE 1 — THE LEAK DIP (primary, two-sided): within-subject anti-decider follow rate on the six leak cells (T1-T4, T14, T15; 36 trials) minus the matched opposition controls (T5-T10; 36 trials). Advocated: dip = q*(2p_f - 1) in [0.02, 0.22], mean ~0.10 — bounded away from zero by its own formulas (q >= 0.2125 forced, p_f > 0.5 forced). Competing: EXACTLY ZERO — it has no mechanism that depends on the carrier's validity, the TTB decider's direction, or anything but the anti decider. SIGNATURE 2 — THE INTERIORITY WINDOW (the sharpest structural test): the dip profile across carrier rungs must be [0, 0, d, d, 0, 0] — zero at 0.50 and 0.60, positive at the two 0.70 carriers, zero at 0.80, zero (concur) at 0.90. T7 is the strongest single no-leak cell in the design: the carrier is the high-validity 0.80 expert opposed by the 0.90 maximum — any monotone validity-respecting contamination (partial TTB, validity-weighted averaging) dips AT LEAST AS MUCH there as at the 0.70 carrier, while the advocated's strict-inequality gate commits to EXACTLY ZERO. Symmetrically, T5 (carrier exactly 0.60, opposed by 0.90) pins the lower boundary. A dip at T5/T7/T9 falsifies the advocated's gate; a dip at T1/T3/T14 falsifies the competing outright. SIGNATURE 3 — TTB-SPECIFICITY CONTROL (T13): the interior carrier and the 0.90 expert CONCUR while f4 (0.80) opposes — the leak fires but mixes toward a concurring decider, so the advocated predicts NO dip despite visible conflict; any 'conflict reduces following' account dips here. Only TTB-decider opposition (T1-T4, T14, T15) may dip. SIGNATURE 4 — LEAK INVARIANCE PROBES: T1 vs T3 (opponent 0.90 vs 0.80), T1 vs T14 (tally margin +1 vs 0), T1 vs T2-carrier-position (f2 vs f3): the advocated predicts the SAME dip everywhere (q depends only on design-level coherence and gap_term, not on the carrier, opponent, or margin) — the competing predicts no dip anywhere; any margin gradient additionally falsifies the one-reason core both share. T1 vs T12 is the cleanest matched pair: identical except the 0.90 expert's direction — advocated demands p(T1) < p(T12) by q*(2p_f-1); competing demands exact equality for any (beta, epsilon). SIGNATURE 5 — SHARED ANTI-VALIDITY CORE: T9/T10 (follow the 50% expert against the 90% expert at p_f) test the distrust-the-weakest claim both theories share — if subjects follow the 90% expert there, both fail together; T11/T12 pin each subject's p_f via lone-discriminator cells; T16 (full tie) anchors lapse and side bias at exactly 0.50. GLOBAL PROFILES: the competing emits a FLAT two-level step function (p_f toward the anti decider on all 15 discriminating cells, 0.50 on the tie) — nothing in it depends on carrier validity or TTB direction; the advocated emits the same step function EXCEPT a graded 'window' notch: ~0.59 toward the anti decider on the six leak cells vs ~0.69 on the nine controls (population means at p_f ~0.69, q ~0.26), with the strict ordering p(T12) = p(T5) = p(T7) = p(T9) > p(T1) = p(T3) = p(T14) that the competing cannot reproduce. HONESTY NOTES: (i) single-cell population intervals overlap (the advocated's p_f spans [0.55, 0.85], the competing's [0.59, 0.84]); the decisive statistics are the within-subject paired contrasts (Signatures 1-3) and the 16-cell profile shape, robust to fitting beta, epsilon, gamma, rho_leak, primacy_weight, sigma_h, kappa, zeta, and tie_break to each theory (gamma, sigma_h, kappa, zeta, primacy_weight, and tie_break are provably inert in this design: distortion is gated off by the ties, the primacy gate is shut by ambiguity 0.0145, and no tilt can flip a 0.10 validity gap). (ii) The dip's lower bound (0.02 at extreme discipline parameters) is small; the population mean commitment is ~0.10, so a well-powered null (CI within +/-0.03) falsifies the advocated's forced q >= 0.2125, while any reliable dip >= 0.05 on interior cells falsifies the competing — a genuinely two-sided test. (iii) The design cannot separately test the sub-linear coherence exponent (coherence = 1 by construction); it instead tests the leak's existence, its interiority window geometry, and its TTB-specificity — the axes on which these two theories' predictions are structurally disjoint. With 16 unique pairs and K = 6 (96 trials exactly), each cell yields 6 repetitions per subject for stable per-subject choice proportions.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Diffuse Mixture Frugality with Strong Interior-Gated Validity Leak (sub-linear coherence variant). People are one-reason decision makers: experts are consulted one at a time and the first strictly discriminating expert decides; no integration across cues, no history use, exact validity ties anchor the consultation order, and a trial with no discriminating expert yields exactly 0.5. The population-central hierarchy is anti-validity (distrust-the-weakest: consult ascending advertised validity). Three graded mechanisms modulate the core. (1) STRONG INTERIOR-GATED VALIDITY LEAK with SUB-LINEAR COHERENCE WEIGHTING: when the anti cascade's first discriminating expert carries a strictly interior validity (min + 0.10 < val < max - 0.10), the decision mixes toward the standard TTB (descending-validity) decider with probability q = rho_leak * sqrt(reversal_coherence) * gap_term, where gap_term is the range-normalized mean adjacent unique-validity gap. The sub-linear coherence exponent leaves fully reversal-coherent designs (coherence ~ 1) exactly on their calibrated anchor values while granting moderately coherent designs substantially more leak mass than the linear weighting did — capturing the empirical fact that leak toward the strongest expert is not confined to perfectly mirror-structured validity lists. (2) DIFFUSE HETEROGENEITY: w_j = -val_j + gamma*salience_j + A^3*(sigma_h*zeta_j + kappa*xi_j), with a small stable per-subject part (zeta) and a large momentary per-trial part (xi), applied only when all validities are distinct. (3) POSITIONAL-PRIMACY MINORITY, TIE-ANCHORED and STEEPLY THRESHOLD-GATED: the reading-order fallback fires only on designs that BOTH contain exact advertised-validity ties AND exhibit near-maximal binding ambiguity (A >= 0.80, linear ramp over 0.20, mixture weight capped at 0.35); all-distinct and monotone/anchored designs remain fully collapsed onto the anti-validity core. Response discipline is heterogeneous: softmax(beta) plus lapse epsilon with beta in [0.3, 2.0] and epsilon in [0.08, 0.30], giving pooled decider-follow ~0.69 with the wide between-subject spread that reproduces the follow-protection tail.

**Parameters:**
- rho_leak: [0.85, 1.25]
- primacy_weight: [0.3, 0.45]
- gamma: [-0.05, 0.12]
- sigma_h: [0.6, 1.2]
- kappa: [8, 12]
- beta: [0.3, 2.0]
- epsilon: [0.08, 0.30]
- tie_break: [0, 1]
- zeta: [(-1, 1)] * n_features
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Diffuse Mixture Frugality with Strong Interior-Gated Validity Leak
    # (sub-linear coherence variant: sqrt(coherence) leak weighting,
    # steepened tie-anchored primacy gate at A >= 0.80).
    #
    # One-reason decision making: experts consulted one at a time, first
    # strictly discriminating expert decides; no integration across cues.
    # Population-central hierarchy: ANTI-VALIDITY (ascending advertised
    # validity; distrust the weakest expert).
    #
    # Mechanisms:
    #   (1) STRONG INTERIOR-GATED VALIDITY LEAK: when the anti cascade's
    #       first discriminating expert j* carries a strictly interior
    #       validity (min+0.10 < val[j*] < max-0.10), mix toward the TTB
    #       (descending-validity) decider with probability
    #       q = rho_leak * coherence^0.5 * gap_term,
    #       where gap_term = mean adjacent unique-validity gap NORMALIZED
    #       by the validity range (vmax - vmin). The SUB-LINEAR coherence
    #       exponent is the single change vs the accepted base: designs
    #       with coherence ~ 1.0 (the leak anchors: leak-dip, knife-edge,
    #       reversed-cascade designs) keep q unchanged (sqrt(1) = 1),
    #       while moderately coherent designs gain leak mass — e.g.
    #       coherence 0.38 -> effective 0.62 (+62%), coherence 0.6 -> 0.77
    #       (+29%) — pulling their over-predicted decider-follow contrasts
    #       down toward the observed values without touching the anchors.
    #   (2) DIFFUSE HETEROGENEITY: w_j = -val_j + gamma*salience_j
    #       + A^3*(sigma_h*zeta_j + kappa*xi_j), applied only when all
    #       validities are distinct; zeta stable per subject, xi re-drawn
    #       every trial.
    #   (3) POSITIONAL-PRIMACY MINORITY, TIE-ANCHORED + STEEP-THRESHOLD-
    #       GATED (retained from the iter-3/4 experiments, per the
    #       critic's confirmation): the reading-order fallback fires ONLY
    #       on designs that both contain exact validity ties AND have
    #       near-maximal binding ambiguity (A >= 0.80, ramp (A-0.80)/0.20),
    #       with the mixture weight capped at 0.35. All-distinct designs
    #       and monotone/anchored designs keep the pure leak-augmented
    #       anti-validity core.
    #
    # Response discipline: softmax(beta) over the binary winner score plus
    # independent lapse epsilon, with the ACCEPTED BASE's ranges verbatim
    # (beta in [0.3, 2.0], epsilon in [0.08, 0.30]) — the discipline axis
    # was sampled at three points in this loop and the base's setting is
    # the optimum; it is not touched again.
    # No discriminating cue -> exactly 0.5. History is ignored.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float).ravel()
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    gamma = float(parameters["gamma"])
    sigma_h = float(parameters["sigma_h"])
    kappa = float(parameters["kappa"])
    rho_leak = float(parameters["rho_leak"])
    primacy_weight = float(parameters["primacy_weight"])
    tie_break = float(parameters["tie_break"])

    zeta = np.asarray(parameters["zeta"], dtype=float).ravel()
    if zeta.shape[0] != n_features:
        raise ValueError(
            f"zeta length {zeta.shape[0]} != n_features {n_features}."
        )

    n_opts = 2
    a, b = stim[0], stim[1]

    if not np.any(a != b):
        # No discriminating cue: pure guess (lapse mixing leaves a
        # uniform unchanged).
        return np.ones(n_opts) / float(n_opts)

    # ---- helper: average-tie ranks ----
    def avg_ranks(x):
        n = x.shape[0]
        srt = np.argsort(x, kind="stable")
        ranks = np.empty(n, dtype=float)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and x[srt[j + 1]] == x[srt[i]]:
                j += 1
            ranks[srt[i:j + 1]] = 0.5 * (i + j)  # 0-indexed average rank
            i = j + 1
        return ranks

    # ---- positional-salience gradient (early-listed experts favored) ----
    if n_features > 1:
        sal = ((n_features - 1 - np.arange(n_features, dtype=float))
               / (n_features - 1.0)) - 0.5
    else:
        sal = np.zeros(1, dtype=float)

    # ---- binding ambiguity: A = 1 - |Spearman(position, validity)| ----
    if n_features > 1:
        ranks = avg_ranks(val)
        pos_r = np.arange(n_features, dtype=float)
        vp = pos_r - pos_r.mean()
        vr = ranks - ranks.mean()
        denom = np.sqrt(float((vp * vp).sum()) * float((vr * vr).sum()))
        if denom > 1e-12:
            rho = float((vp * vr).sum()) / denom
        else:
            rho = 0.0
        ambiguity = 1.0 - abs(rho)
    else:
        ambiguity = 0.0

    # ---- reversal coherence: |Spearman(validity, validity[::-1])| ----
    if n_features > 1:
        ranks_v = avg_ranks(val)
        ranks_r = avg_ranks(val[::-1])
        dv = ranks_v - ranks_v.mean()
        dr = ranks_r - ranks_r.mean()
        denom2 = np.sqrt(float((dv * dv).sum()) * float((dr * dr).sum()))
        if denom2 > 1e-12:
            rho_rev = float((dv * dr).sum()) / denom2
        else:
            rho_rev = 0.0
        coherence = abs(rho_rev)
    else:
        coherence = 0.0

    # ---- mean adjacent gap over UNIQUE validities, RANGE-NORMALIZED ----
    # gap_term = mean_gap / (vmax - vmin): a scale-free measure of how
    # coarse the validity grid is, so coherent small-range designs are
    # no longer starved of leak mass relative to wide-range designs.
    uvals = np.unique(val)
    distinct = (uvals.size == n_features)
    has_ties = (uvals.size < n_features)
    if uvals.size > 1:
        mean_gap = float(np.mean(np.diff(uvals)))
        v_range = float(uvals[-1] - uvals[0])
        gap_term = mean_gap / v_range if v_range > 1e-12 else 0.0
    else:
        gap_term = 0.0

    pos = np.arange(n_features)
    secondary = pos if tie_break < 0.8 else (n_features - 1 - pos)

    # ---- anti-validity weights with diffuse heterogeneity ----
    # Exact validity ties anchor the distrust order (no distortion);
    # all-distinct lists degrade with the CUBE of binding ambiguity.
    # Small stable part (sigma_h * zeta) + large momentary part
    # (kappa * xi, re-drawn fresh on every trial).
    w = -val + gamma * sal
    if distinct and ambiguity > 0.0:
        amb3 = ambiguity * ambiguity * ambiguity
        xi = np.random.uniform(-1.0, 1.0, size=n_features)
        w = w + amb3 * (sigma_h * zeta + kappa * xi)
    order_anti = np.lexsort((secondary, -w))

    def cascade(order):
        for j in order:
            if a[j] > b[j]:
                return 0
            if b[j] > a[j]:
                return 1
        return None

    winner_anti = cascade(order_anti)

    # ---- this trial's anti-validity carrier (for the leak gate) ----
    j_star = None
    for j in order_anti:
        if a[j] != b[j]:
            j_star = int(j)
            break

    def score_prob(winner):
        scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
        z = beta * (scores - scores.max())
        e = np.exp(z)
        return e / e.sum()

    # Start from the (momentary) anti-validity decider.
    p = score_prob(winner_anti)

    # ---- POSITIONAL-PRIMACY MINORITY (tie-anchored, steep-threshold gate) ----
    # The reading-order fallback is elicited ONLY when the design BOTH
    # contains exact advertised-validity ties AND sits in the near-maximal
    # binding-ambiguity regime (A >= 0.80, linear ramp over 0.20). This is
    # the steepened gate confirmed by the critic: it lowered the
    # anti-misbound-decider follow (Exp 13) in both trial iterations
    # without a real cost on the count-vs-primacy contrast (Exp 15),
    # whose all-distinct validities keep w_eff = 0 exactly.
    if has_ties and ambiguity >= 0.80:
        gate = min(1.0, max(0.0, (ambiguity - 0.80) / 0.20))
    else:
        gate = 0.0
    w_eff = min(primacy_weight * gate, 0.35)
    if w_eff > 0.0:
        winner_prim = cascade(pos)  # earliest-listed expert first
        if winner_prim is not None:
            p = (1.0 - w_eff) * p + w_eff * score_prob(winner_prim)

    # ---- STRONG INTERIOR-GATED VALIDITY LEAK ----
    # Fires only when the anti cascade's carrier is strictly interior:
    # min + 0.10 < val[j*] < max - 0.10. Weakest-carrier and near-top
    # carrier trials keep the leak OFF.
    # SINGLE CHANGE vs the accepted base: the coherence factor enters
    # SUB-LINARLY (coherence**0.5) instead of linearly. High-coherence
    # anchor designs are untouched by construction (sqrt(1) = 1);
    # moderately coherent designs gain leak mass, pulling their
    # over-predicted decider-follow contrasts toward the observed values.
    if j_star is not None and n_features > 1:
        vmin = float(val.min())
        vmax = float(val.max())
        if (vmin + 0.10) < val[j_star] < (vmax - 0.10):
            q = rho_leak * (coherence ** 0.5) * gap_term
            # Numerical validity clamp only (keeps mixture weights proper
            # probabilities on extreme small-n designs); far above the
            # substantive target range on the leak-dip design.
            q = min(max(q, 0.0), 0.75)
            if q > 0.0:
                order_ttb = np.lexsort((secondary, -val))
                winner_ttb = cascade(order_ttb)
                if winner_ttb is not None:
                    p = (1.0 - q) * p + q * score_prob(winner_ttb)

    # ---- heterogeneous response discipline: softmax + lapse ----
    # (base ranges verbatim: beta [0.3, 2.0], epsilon [0.08, 0.30])
    return (1.0 - epsilon) * p + epsilon * (np.ones(n_opts) / float(n_opts))
```

**`policy source code`:**
```python
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Anti-Validity Frugality with Diffuse Positional-Salience Heterogeneity (momentary-diffusion variant). People are one-reason decision makers: cues are consulted one at a time in descending subjective weight and the first strictly discriminating cue decides; no integration across cues (flat psychometric profile over tally margins). The population-central hierarchy is anti-validity (distrust-the-weakest, w = -validity). Each subject's weights are a continuous perturbation, w_j = -val_j + gamma*salience_j + A^3*(sigma_h*zeta_j + kappa*xi_j), where (i) salience_j is a positional gradient favoring early-listed experts with gamma drawn asymmetrically from [-0.05, +0.12] — a graded tilt, never a discrete primacy rule, bounded so it can flip nothing on the anchored designs that pin the anti-validity core; (ii) binding ambiguity A = 1 - |Spearman(position, validity)| enters CUBED, a steeply convex, gate-free gradient; and (iii) the idiosyncratic distortion is split into a small stable part (zeta_j, fixed per subject) and a large MOMENTARY part (xi_j, re-drawn every trial): under high binding ambiguity the consultation order is not merely idiosyncratic but diffuse — attention fluctuates from trial to trial, which keeps populations tight around chance-level conformity instead of spreading subjects into fixed opposing deciders. All distortion applies only when all advertised validities are distinct (exact ties anchor the distrust order). Noise is softmax(beta) over the binary winner score plus an independent lapse epsilon (implied follow p_f mean ~0.71, SD ~0.08); no discriminating cue yields exactly 0.5; validity ties are broken by a free per-subject tie-break (~80% early position); history is ignored.

**Parameters:**
- beta: [0.5, 1.9]
- epsilon: [0.08, 0.24]
- gamma: [-0.05, 0.12]
- sigma_h: [0.6, 1.2]
- kappa: [8, 12]
- zeta: [(-1, 1)] * n_features
- tie_break: [0, 1]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Anti-Validity Frugality with Diffuse Positional-Salience Heterogeneity
    # (momentary-diffusion variant).
    #
    # One-reason decision making: cues are consulted one at a time in
    # DESCENDING subjective weight and the first strictly discriminating
    # cue decides; no integration across cues (flat psychometric profile
    # over tally margins).
    #
    # Subjective weights (continuous, per subject):
    #     w_j = -val_j + gamma * salience_j
    #            + A^3 * (sigma_h * zeta_j + kappa * xi_j)
    #
    # - ANTI-VALIDITY CORE: -val_j (distrust the weakest advertised
    #   expert; consult ascending validity).
    # - POSITIONAL SALIENCE: salience_j = (n-1-j)/(n-1) - 1/2, favoring
    #   early-listed experts; gamma is drawn ASYMMETRICALLY from
    #   [-0.05, +0.12]: the negative clamp sits above the anchored-design
    #   flip threshold (-0.0625) and the positive cap below the primacy
    #   type threshold (+0.144), so the tilt remains a graded perturbation
    #   that can never manufacture spurious pure types while retaining a
    #   partial early-position lean on anchored designs.
    # - BINDING AMBIGUITY: A = 1 - |Spearman(display position, validity)|,
    #   now entering CUBED. The steeply convex gradient keeps
    #   low/moderate-ambiguity designs anchored (A^3 ~ 0.01-0.02) while
    #   fully diffusing high-ambiguity all-distinct designs (A^3 ~ 0.3-0.6).
    # - DIFFUSE HETEROGENEITY: the idiosyncratic distortion is split into
    #   (a) a SMALL STABLE part, sigma_h * zeta_j (zeta fixed per subject,
    #   zero-centered), and (b) a LARGE MOMENTARY part, kappa * xi_j,
    #   where xi is RE-DRAWN EVERY TRIAL. Momentary consultation diffusion
    #   is what makes high-ambiguity populations tight around chance-level
    #   conformity: with fixed per-subject orders, scrambling MAXIMIZES
    #   between-subject follow-rate dispersion, whereas the observed
    #   population SD (~0.064 on the conformity-dispersion experiment)
    #   sits barely above the binomial floor. Both parts are applied ONLY
    #   when all advertised validities are distinct (exact ties anchor the
    #   distrust order).
    # NO reversal re-encoding, NO sigmoid gates, NO discrete component
    # draws.
    #
    # Noise: softmax(beta) over the binary winner score mixed with an
    # independent lapse epsilon (implied follow probability
    # p_f = (1-eps)*sigmoid(beta) + eps/2, mean ~0.71, SD ~0.08).
    # No discriminating cue -> exactly 0.5. History is ignored.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float).ravel()
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    gamma = float(parameters["gamma"])
    sigma_h = float(parameters["sigma_h"])
    kappa = float(parameters["kappa"])
    tie_break = float(parameters["tie_break"])

    zeta = np.asarray(parameters["zeta"], dtype=float).ravel()
    if zeta.shape[0] != n_features:
        # NO silent zeros fallback: a shape mismatch means the entire
        # heterogeneity mechanism would be silently disabled, so fail
        # loudly instead (per the arbiter's diagnosis request).
        raise ValueError(
            f"zeta length {zeta.shape[0]} != n_features {n_features}."
        )

    # ---- helper: average-tie ranks ----
    def avg_ranks(x):
        n = x.shape[0]
        srt = np.argsort(x, kind="stable")
        ranks = np.empty(n, dtype=float)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and x[srt[j + 1]] == x[srt[i]]:
                j += 1
            ranks[srt[i:j + 1]] = 0.5 * (i + j)  # 0-indexed average rank
            i = j + 1
        return ranks

    # ---- positional-salience gradient (early-listed experts favored) ----
    # Centered so the tilt is a pure graded perturbation: +0.5 at the
    # first listed expert, -0.5 at the last, linear in between.
    if n_features > 1:
        sal = ((n_features - 1 - np.arange(n_features, dtype=float))
               / (n_features - 1.0)) - 0.5
    else:
        sal = np.zeros(1, dtype=float)

    # ---- binding ambiguity: A = 1 - |Spearman(position, validity)| ----
    if n_features > 1:
        ranks = avg_ranks(val)
        pos_r = np.arange(n_features, dtype=float)
        vp = pos_r - pos_r.mean()
        vr = ranks - ranks.mean()
        denom = np.sqrt(float((vp * vp).sum()) * float((vr * vr).sum()))
        if denom > 1e-12:
            rho = float((vp * vr).sum()) / denom
        else:
            # All validities tied: distrust order undefined.
            rho = 0.0
        ambiguity = 1.0 - abs(rho)
    else:
        ambiguity = 0.0

    # ---- continuous subjective weights ----
    w = -val + gamma * sal

    # ---- diffuse idiosyncratic distortion (distinct lists only) ----
    # Exact validity ties anchor the distrust order (no distortion);
    # all-distinct lists degrade with the CUBE of binding ambiguity.
    # The distortion is split into a small STABLE per-subject part
    # (sigma_h * zeta) and a large MOMENTARY per-trial part (kappa * xi,
    # xi re-drawn fresh on every call), so that high-ambiguity designs
    # show tight, chance-centered conformity rather than fixed
    # idiosyncratic deciders that spread the population apart.
    distinct = (np.unique(val).shape[0] == n_features)
    if distinct and ambiguity > 0.0:
        amb3 = ambiguity * ambiguity * ambiguity
        xi = np.random.uniform(-1.0, 1.0, size=n_features)
        w = w + amb3 * (sigma_h * zeta + kappa * xi)

    # Consult cues in DESCENDING subjective weight. np.lexsort uses the
    # LAST key as primary. Exact ties in w are broken by the free
    # per-subject tie_break parameter (population-consistent: ~80% of
    # subjects break ties by early display position).
    pos = np.arange(n_features)
    secondary = pos if tie_break < 0.8 else (n_features - 1 - pos)
    cue_order = np.lexsort((secondary, -w))

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
        # No discriminating cue: pure guess (lapse mixing leaves a
        # uniform unchanged).
        return np.ones(n_opts) / float(n_opts)

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    # Numerically stable softmax over the binary winner score: the
    # winner's probability is sigmoid(beta). Implied decider-follow
    # probability p_f = (1-eps)*sigmoid(beta) + eps/2 has mean ~0.71
    # with per-subject SD ~0.08 over the sampled (beta, epsilon) ranges.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / float(n_opts))
```

**`policy source code`:**
```python
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return np.random.choice(len(probabilities), p=probabilities)
```


## DATA SCHEMA
Your metric receives a tidy per-trial pandas DataFrame stacking all subjects (rows grouped by `subject_id`, in trial order). Columns:
- subject_id: Subject identifier (one row per trial per subject).
- option_a_ratings: List of n_features binary expert ratings (each 0 or 1) for option A on this trial.
- option_b_ratings: List of n_features binary expert ratings (each 0 or 1) for option B on this trial.
- response: 0 if subject chose A, 1 if subject chose B.

## IMPLEMENTATION GUARDRAILS
Any column in the schema above whose description names a list / tuple / np.ndarray (i.e. a per-trial sequence of values) holds non-scalar cells. Those cells are NOT hashable, so operations that hash row values fail with `TypeError: unhashable type: 'list'`. Treating `<seq_col>` as a placeholder for any such sequence-valued column:
- Avoid: `data.groupby('<seq_col>')`, `data['<seq_col>'].value_counts()`,     `data['<seq_col>'].nunique()`, `data['<seq_col>'].unique()` (returns     an object array but downstream `set()` / `in dict` will crash),     `set(data['<seq_col>'])`, `data['<seq_col>'].isin([...])` against list     values, or using a list cell as a dict key.
- If you need a hashable surrogate, project to one first, e.g.:
    - `data['<seq_col>_key'] = data['<seq_col>'].apply(tuple)` then group by `<seq_col>_key`
    - `data['<seq_col>_str'] = data['<seq_col>'].apply(lambda x: ''.join(map(str, x)))`
    Scalar columns (ints, floats, strings like `subject_id`, integer     responses, etc.) hash fine and can be used directly.
- Generator expressions inside function calls like `map()` or `join()` MUST be     parenthesized. For example:
    - WRONG: `map(str, int(v) for v in x)` → SyntaxError
    - RIGHT: `map(str, (int(v) for v in x))` or use a list comp: `[str(int(v)) for v in x]`
- Always verify your code is syntactically valid Python before returning it.

## METRICS YOU ALREADY TRIED AND FAILED ON
Each entry below is a metric you previously proposed in this round that did NOT discriminate the two theories at the human sample size — either it errored, its between-subject variance was unavailable, or Welch's t-test on `(self mean, self var, N)` vs. `(adv mean, adv var, N)` returned p ≥ alpha. The `outcome` line is the simulation result (means, between-subject variances, t-statistic and p-value at the human N) on the same `data_self` / `data_adv` your next metric will be evaluated on. Use the numbers to see where your hypothesised contrast collapsed — small mean gap, large per-subject variance, or both — and propose something qualitatively different. Don't repeat the same idea with cosmetic tweaks.
(none yet)

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    # Design constants (fixed per experiment): 6 features,\n    # validities [0.50, 0.60, 0.70, 0.70, 0.80, 0.90].\n    val = np.array([0.50, 0.60, 0.70, 0.70, 0.80, 0.90])\n    # Shared anti-validity core: consult experts in ASCENDING advertised\n    # validity (distrust the weakest); the tied 0.70 pair is concordant in\n    # every trial, so any tie-break between f2/f3 is inert.\n    anti_order = np.argsort(val, kind=\"stable\")  # [0,1,2,3,4,5]\n    # TTB order: DESCENDING validity (the leak's target decider).\n    ttb_order = np.argsort(-val, kind=\"stable\")  # [5,4,2,3,1,0]\n    # Interiority window of the advocated leak gate (strict inequalities).\n    lo = float(val.min()) + 0.10  # 0.60\n    hi = float(val.max()) - 0.10  # 0.80\n\n    a_lists = data[\"option_a_ratings\"].tolist()\n    b_lists = data[\"option_b_ratings\"].tolist()\n    resp = data[\"response\"].to_numpy()\n\n    dip_n = 0\n    dip_follow = 0.0\n    ctl_n = 0\n    ctl_follow = 0.0\n\n    for a_raw, b_raw, r in zip(a_lists, b_lists, resp):\n        a = np.asarray(a_raw, dtype=int).ravel()\n        b = np.asarray(b_raw, dtype=int).ravel()\n        if a.shape[0] != val.shape[0] or b.shape[0] != val.shape[0]:\n            continue\n        diff = a != b\n        if not diff.any():\n            # Full tie (T16): no discriminating expert, no anti decider.\n            continue\n\n        # Anti-validity decider: first discriminating expert in ascending\n        # validity order decides.\n        j_star = None\n        for j in anti_order:\n            if diff[j]:\n                j_star = int(j)\n                break\n        anti_choice = 0 if a[j_star] > b[j_star] else 1\n\n        # TTB decider: first discriminating expert in descending order.\n        ttb_choice = None\n        for j in ttb_order:\n            if diff[j]:\n                ttb_choice = 0 if a[j] > b[j] else 1\n                break\n\n        followed = 1.0 if int(r) == anti_choice else 0.0\n\n        # DIP cell: the anti cascade's carrier is STRICTLY interior\n        # (0.60 < val < 0.80, i.e. the doubled 0.70 group) AND the TTB\n        # decider points the other way, so the advocated's leak pulls\n        # choices away from the anti decider. On every other\n        # discriminating trial BOTH theories follow the anti decider at\n        # the same flat rate p_f (boundary carriers 0.60/0.80 are gated\n        # off by the strict inequality; interior carriers with a\n        # concurring TTB, e.g. T12/T13, mix toward the same choice).\n        is_dip = (ttb_choice is not None\n                  and ttb_choice != anti_choice\n                  and lo < val[j_star] < hi)\n\n        if is_dip:\n            dip_n += 1\n            dip_follow += followed\n        else:\n            ctl_n += 1\n            ctl_follow += followed\n\n    if dip_n == 0 or ctl_n == 0:\n        return 0.0\n\n    # Within-subject LEAK DIP: follow-the-anti-decider rate on the six\n    # leak cells (T1-T4, T14, T15; 36 trials) minus the follow rate on\n    # all matched non-leak discriminating cells (T5-T13; 54 trials).\n    # Advocated: negative, approx -q*(2*p_f - 1), mean ~ -0.10.\n    # Competing: exactly 0 (nothing in it depends on carrier validity or\n    # TTB direction).\n    return (dip_follow / dip_n) - (ctl_follow / ctl_n)",
  "rationale": "METRIC: the within-subject LEAK DIP \u2014 the rate of following the (deterministic, shared) anti-validity decider on the six leak cells minus the follow rate on all other discriminating cells. CLASSIFICATION IS COMPUTED FROM THE STIMULI, not hardcoded trial indices: for each trial I reconstruct the anti cascade (ascending validity; the tied 0.70 pair is concordant in every trial so tie-breaks are inert in both theories) and the TTB cascade (descending validity). A trial is a DIP cell iff the anti cascade's carrier (first discriminating expert) carries a strictly interior validity (0.60 < val < 0.80, i.e. only the doubled 0.70 group) AND the TTB decider opposes the anti decider. These are exactly T1-T4, T14, T15 (36 trials). All remaining discriminating trials (T5-T13, 54 trials) serve as controls: boundary carriers (0.60 in T5/T6, 0.80 in T7/T8) are excluded by the strict inequality, and interior carriers with a concurring TTB (T12, T13) mix toward the same choice, so both theories follow the anti decider there at the same flat p_f. WHY IT DISCRIMINATES: the competing theory's follow probability is p_f toward the anti decider on every discriminating trial \u2014 it has no mechanism depending on carrier validity, TTB direction, margin, or coalition \u2014 so its dip is exactly 0 with only binomial noise (per-subject SD ~ 0.10). The advocated theory is FORCED by its own formulas to leak on the interior cells: coherence = 1 exactly (symmetric tied ranks), gap_term = 0.25, so q = rho_leak*0.25 in [0.2125, 0.3125] on every interior-carrier opposed trial, giving dip = q*(2p_f-1) with population mean ~ 0.10 (per-subject SD ~ 0.045 from rho_leak/beta/epsilon heterogeneity). The metric is a within-subject paired contrast, so the shared p_f (and lapse, and any side bias) cancels exactly, leaving the between-theory gap of ~0.10 against a per-subject SD of ~0.11 in both conditions \u2014 at N=25 that yields |t| ~ 3.4, p ~ 0.001, comfortably below alpha = 0.01. The metric works identically on the pooled DataFrame and on a single subject's slice (every subject sees all 16 pairs 6 times, so each subject contributes 36 dip and 54 control trials), satisfying the between-subject-variance requirement. It is also two-sided-honest: a reliably POSITIVE value (higher follow on interior cells) would falsify the advocated's gate just as a zero dip falsifies it, and any dip on the boundary cells T5/T7 is absorbed into the control mean where both theories demand flat p_f."
}
```

## Usage

```json
{
  "input_tokens": 10331,
  "output_tokens": 11292,
  "reasoning_tokens": 9603,
  "provider": "Sail Research"
}
```
