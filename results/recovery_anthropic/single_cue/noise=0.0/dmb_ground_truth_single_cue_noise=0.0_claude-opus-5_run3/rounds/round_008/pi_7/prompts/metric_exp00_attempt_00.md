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
**Validities (n_features=7):** [0.74, 0.6, 0.96, 0.74, 0.86, 0.62, 0.74]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 0, 1, 1, 1, 0]  B=[0, 0, 1, 0, 0, 0, 1]
  trial 2: A=[1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 0, 1, 0]
  trial 3: A=[1, 1, 1, 1, 1, 0, 1]  B=[0, 1, 1, 1, 0, 1, 0]
  trial 4: A=[1, 1, 0, 0, 1, 1, 1]  B=[0, 0, 1, 1, 0, 1, 1]
  trial 5: A=[0, 1, 0, 1, 1, 1, 1]  B=[1, 0, 1, 0, 0, 1, 1]
  trial 6: A=[1, 0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 1, 1]
  trial 7: A=[1, 0, 1, 0, 1, 0, 0]  B=[1, 1, 0, 1, 0, 1, 1]
  trial 8: A=[1, 1, 0, 1, 0, 1, 1]  B=[0, 1, 1, 1, 1, 1, 0]
  trial 9: A=[0, 0, 1, 0, 0, 0, 1]  B=[1, 1, 0, 1, 1, 1, 0]
  trial 10: A=[0, 1, 1, 1, 0, 1, 0]  B=[1, 1, 1, 1, 1, 1, 1]
  trial 11: A=[0, 1, 1, 1, 0, 1, 0]  B=[1, 1, 1, 1, 1, 0, 1]
  trial 12: A=[0, 0, 1, 1, 0, 1, 1]  B=[1, 1, 0, 0, 1, 1, 1]
  trial 13: A=[1, 0, 1, 0, 0, 1, 1]  B=[0, 1, 0, 1, 1, 1, 1]
  trial 14: A=[1, 1, 0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 1, 0, 1]
  trial 15: A=[1, 1, 0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 1, 0, 0]
  trial 16: A=[0, 1, 1, 1, 1, 1, 0]  B=[1, 1, 0, 1, 0, 1, 1]

**Rationale:** GOAL: dissociate the ADVOCATED two-regime reason-counter (near-linear validity vote rho~0.95, ABSOLUTE drive read out with beta~1.2, categorical unanimity bonus ups~1.5, SATURATING majority bonus lam~0.67 that is identical for |dk|=2 and |dk|=5, symmetric first/last-row salience with a BROAD tau~1.0 that persists as a constant residual g_res~0.25 whenever the vote is not deadlocked, SHARP silence sigmoid at f0~0.36 with mutual endorsements weighted omega~0.82, confidence floored at c_min~0.10, small lapse eps~0.035) from DCRI-3s (compressive rho~0.52 with an oracle premium above v_knee~0.91, sqrt numerosity lam~0.56, RELATIVE read-out drive=(V+N+P)/(kap2+Xs) with beta~3.1, steep salience tau~0.75 with g_res~0.10, SMOOTH power-law silence discount with omega~0.63, and a difficulty-scaled lapse up to eps~0.30 that compresses every near-chance display).

LAYOUT: n=7, validities by SCREEN position [0.74, 0.60, 0.96, 0.74, 0.86, 0.62, 0.74]. Design choices: (i) the near-oracle 0.96 sits at pos2, above DCRI's knee so its top-end premium fires (x=1.46 vs 0.92 for the .86 expert; the advocated model rates them 0.92 vs 0.73), and deep enough that its salience is negligible for both models; (ii) THREE experts share validity 0.74 and sit at pos0 (top edge), pos3 (exact middle) and pos6 (bottom edge), which permits a validity-matched EDGE<->MIDDLE swap that leaves the vote of ANY validity-only model exactly unchanged; (iii) the cheap experts (.60, .62) sit at pos1/pos5, which are salience-matched; (iv) in every cell the first discriminating row is pos0 or pos1, whose validity (.48 or .20 in vs units) is far below the later oracle (.92), so BOTH models' authority-checked commitment term is suppressed (q_com <= 0.02) and cannot contaminate any contrast.

8 base cells, each mirrored across sides -> 16 unique pairs, K=6 -> 96 trials; mirroring forces an exact 8A/8B split for BOTH theories, so side bias cannot mimic either model. All mute rows are mutual endorsements (1,1), which is exactly where the two silence rules differ (omega .82 vs .63).

(1) UNANIMITY / SILENCE-CLIFF PAIR - cells 2 and 3 (pairs 2,3 + mirrors). Cell 2: pos0(.74), pos4(.86), pos6(.74) unanimously favour A, four rows mute endorsements. Cell 3 is IDENTICAL except the weak .62 expert at pos5 now DISSENTS (3-1). Advocated: in cell 2 the effective silence fraction .82*4/7=.469 is far above its sharp threshold, c collapses to ~.11 and the big drive (vote 1.73 + unanimity 1.5 + majority .67 + salience .98 = 4.9) is crushed to P(A)~.65; in cell 3 the panel becomes informative (.82*3/7=.351, just BELOW the step), c jumps to ~.65 and P(A) RISES to ~.90 even though a reason was subtracted. So the advocated theory predicts a paradoxical +9 to +24 point INCREASE from adding a dissenter, across its whole box. DCRI's smooth power-law discount plus its relative read-out offset almost exactly: drive falls 1.39->0.95 while c rises .39->.56, giving .801 -> .799, a forced ~zero change, AND in cell 2 DCRI is 15 points ABOVE the advocated model (.80 vs .65), i.e. the ordering of the two models REVERSES between the two cells. This is the flagship crossover.

(2) VALIDITY-MATCHED EDGE<->MIDDLE SWAP - cells 4 and 5 (pairs 4,5 + mirrors). Both cells have the identical speaking-validity multiset {.74,.60,.86,.96,.74}, identical counts (3-2, dk=1, so both salience gates are at their residual), identical silence (two mute endorsements at pos5/pos6) and an identical vote V=+0.025: only WHICH of the two equally valid .74 experts is at the top edge (pos0) versus the exact middle (pos3) changes. Any validity-only model is analytically pinned. The advocated residual gate (mu*g_res=0.455 acting on a symmetric U-shaped, broad-tau salience profile) swings S from +1.28 to -0.53 and predicts P(A)=.67 vs .44 - a 23-point swing that CROSSES chance. DCRI's steeper profile is similar in shape but its salience enters a relative read-out (2.4*0.146*dS/(0.42+Xs)) and is then compressed by its difficulty-scaled lapse, giving .58 vs .52, only ~6 points and never crossing chance.

(3) MAJORITY-BONUS / EDGE CELL PAIR - cells 6 and 7 (pairs 6,7 + mirrors). Cell 6: the two heavyweights pos2(.96) and pos4(.86) favour A, three cheap/mid experts pos1(.60), pos3(.74), pos5(.62) favour B (dk=-1, so the advocated majority bonus is OFF). Cell 7 adds ONE more cheap dissenter at the bottom EDGE, pos6(.74), making it 2-4. Advocated: .62 (A) -> .20 (B), a 42-point swing and a direction reversal, because the single added row simultaneously fires the saturating majority bonus (-0.67) and adds a full edge-salience unit (-0.46). DCRI: .51 -> .40, only ~11 points, because its sqrt numerosity grows slowly, its compression already makes cheap coalitions competitive, and its denominator grows with the extra speaker. Cell 6 alone is also a direction-vs-chance split: the advocated model follows the two heavyweights at .62 while DCRI sits at an exact coin flip (.507) - compression plus numerosity plus the difficulty lapse.

(4) DENSE LANDSLIDE - cell 1 (pair 1 + mirror). Five experts (incl. .74,.60,.74,.86,.62) favour A, the oracle .96 and the bottom-edge .74 favour B, NO mute rows so neither silence rule does any work. Absolute vs relative read-out is isolated: the advocated model's drive (vote .78 + majority .67 + salience .39) at c=1 gives P(A)~.89, whereas DCRI divides an almost identical numerator by kap2+Xs=5.76 and then applies its lapse, giving ~.72. A ~17-point level gap in the cell where both models are maximally confident.

(5) DEADLOCK-GATE CELL - cell 8 (pair 8 + mirror). The two screen extremes (.74,.74) favour A, the two interior heavyweights (.96,.86) favour B, counts deadlocked 2-2 so BOTH gates open fully. Both models overturn the validity vote and choose A, but the advocated model (mu*S=3.09 absolute, silence-discounted c=.55) predicts ~.82 while DCRI (2.4*S/(0.42+Xs) then lapse) predicts ~.74. Included as the anchor showing that the two models agree when the gate is fully open, so the 23-point swap effect in cells 4/5 cannot be attributed to a general position insensitivity of DCRI.

DIAGNOSTIC YIELD: (i) a within-subject ordering CROSSOVER (cell 2: DCRI 15 points above the advocate; cell 3: the advocate 10 points above DCRI) driven by a dissenter that RAISES the advocated model's confidence and leaves DCRI flat; (ii) an analytic validity-matched position swap on which any validity-only account is pinned and where the advocate predicts a chance-crossing 23-point effect vs ~6 for DCRI; (iii) a single-row manipulation (cells 6->7) predicted to reverse the advocate by 42 points and move DCRI by 11; (iv) one 17-point dense-display level gap isolating absolute vs relative read-out plus lapse. Every pair has at least one discriminating cue, every contrast is within-subject, and each cell is mirrored across sides.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** **Reason-counting with a TWO-REGIME display-salience gate (full force only on a deadlocked vote, a constant residual otherwise), near-linear diagnosticity weighting, a saturating majority bonus, unanimity boost, sharp silence-to-confidence collapse (mutual endorsement ≈ silence), and a non-zero floor on read-out gain.**

This keeps every mechanism of the accepted base and changes exactly three calibration commitments (one of them structural-but-local: the shape of the salience gate).

1. **Direction is a vote over discriminating experts, graded near-linearly by stated validity.** Each discriminating expert contributes x_j = ((v_j-0.5)/0.5)^rho with rho ~0.88-1.02. At this exponent a single .95 expert is worth about two upper-middling experts (.72 + .70), so numerosity and diagnosticity are *balanced* rather than one dominating: three mid experts still outvote one top expert, but two mid experts only tie it. Screen position plays no role in this term.

2. **Display salience is a TIE-BREAKER WITH A RESIDUAL, not a quantity that decays with the size of the majority.** s_j = exp(-j/tau)+exp(-(n-1-j)/tau) over the literal display row, multiplied by a gate g = max(exp(-(k_A-k_B)^2/2 sigma^2), g_res) with sigma ~0.60 and g_res ~0.25. The gate has two regimes: when the reason count is exactly deadlocked the first/last rows of the panel decide the choice outright (g = 1); as soon as one side has more reasons, position stops being decisive but does NOT disappear — it persists as a constant, modest attention bonus (g = g_res ≈ 0.25) that is the SAME for a one-reason edge and for a five-against-one landslide. The previous Gaussian-only gate made position vanish geometrically with the majority size, which is unmotivated: there is no reason why a larger majority should erase the extra attention paid to the top and bottom of the screen. The empirical signature is that a three-against-one display whose majority occupies both screen ends is markedly more decisive (~.88) than a three-against-one display whose lone dissenter sits at a screen end (~.82), even when the validities are matched.

3. **Clear majorities are categorical and saturating:** M = lambda*sign(k_A-k_B)*min(max(|k_A-k_B|-1,0),1) with lambda ~0.67. Once one side has at least two more reasons, 'more experts back this one' is itself a discrete reason, and it does not grow further with the size of the majority.

4. **Unanimity among the speaking experts is categorical**: if every expert who noticed a difference points the same way, a constant bonus is added.

5. **Gap inference from the panel is a sharp threshold on how much of the panel was informative, and 'both products are good' is nearly as uninformative as 'neither is'.** Validity is read as sensitivity: a difference only sensitive experts notice implies a marginal true gap. Mutual endorsements count at weight omega ~0.8. c = 1 - eta*sigmoid(k*(f_sil - f0))*vbar.

6. **The collapse of confidence is bounded below:** c is floored at c_min ~0.10, so a panel judged uninformative leaves a small residual tilt rather than a literal coin flip.

7. **A small, display-contingent first-cue commitment survives**: people commit to the first expert who actually discriminates only if no expert further down the list outranks him in stated validity. Individual differences are unimodal jitter in rho, mu, sigma, g_res, lambda, eta, omega, c_min, beta; no strategy subpopulations, no serial stop rule, no learning.

**Parameters:**
- rho: [0.88, 1.02]
- mu: [1.70, 1.95]
- lam: [0.60, 0.74]
- sigma: [0.55, 0.66]
- g_res: [0.22, 0.28]
- tau: [0.85, 1.25]
- ups: [1.25, 1.80]
- beta: [1.05, 1.40]
- eta: [1.45, 1.65]
- sil_k: [30.0, 45.0]
- sil_f0: [0.345, 0.375]
- omega: [0.72, 0.92]
- c_min: [0.07, 0.13]
- phi: [0.95, 1.12]
- s_stop: [6.0, 10.0]
- d_com: [2.20, 2.90]
- epsilon: [0.01, 0.06]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np

    # ------------------------------------------------------------------
    # 1. parse the trial stimulus into two rating vectors (screen order)
    # ------------------------------------------------------------------
    a = None
    b = None
    if isinstance(state, dict):
        if 'option_a_ratings' in state and 'option_b_ratings' in state:
            a = np.asarray(list(state['option_a_ratings']), dtype=float)
            b = np.asarray(list(state['option_b_ratings']), dtype=float)
    if a is None:
        arr = np.asarray(state, dtype=float)
        arr = arr.reshape(2, -1)
        a = arr[0].astype(float)
        b = arr[1].astype(float)

    a = np.ravel(a).astype(float)
    b = np.ravel(b).astype(float)
    n = int(min(a.shape[0], b.shape[0]))
    if n == 0:
        return np.array([0.5, 0.5])
    a = a[:n]
    b = b[:n]
    d = a - b
    disc = np.abs(d) > 1e-12
    if not np.any(disc):
        return np.array([0.5, 0.5])

    # ------------------------------------------------------------------
    # 2. communicated validities -> diagnosticity (= sensitivity)
    # ------------------------------------------------------------------
    val = parameters.get('validities', None)
    try:
        val = np.asarray(val, dtype=float).ravel()
    except Exception:
        val = None
    if val is None or val.shape[0] != n:
        val = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.90])
    val = np.clip(val, 0.5 + 1e-6, 1.0 - 1e-9)
    vs = np.clip((val - 0.5) / 0.5, 1e-6, 1.0)

    # ------------------------------------------------------------------
    # 3. parameters
    # ------------------------------------------------------------------
    rho = float(np.clip(parameters.get('rho', 0.95), 0.05, 4.0))
    mu = float(np.clip(parameters.get('mu', 1.82), 0.0, 8.0))
    lam = float(np.clip(parameters.get('lam', 0.67), 0.0, 3.0))
    sigma = float(np.clip(parameters.get('sigma', 0.60), 0.15, 3.0))
    g_res = float(np.clip(parameters.get('g_res', 0.25), 0.0, 0.6))
    tau = float(np.clip(parameters.get('tau', 1.0), 0.2, 8.0))
    ups = float(np.clip(parameters.get('ups', 1.50), 0.0, 4.0))
    beta = float(np.clip(parameters.get('beta', 1.20), 0.05, 40.0))
    eta = float(np.clip(parameters.get('eta', 1.55), 0.0, 5.0))
    sil_k = float(np.clip(parameters.get('sil_k', 38.0), 1.0, 80.0))
    sil_f0 = float(np.clip(parameters.get('sil_f0', 0.36), 0.05, 0.9))
    omega = float(np.clip(parameters.get('omega', 0.82), 0.0, 1.0))
    c_min = float(np.clip(parameters.get('c_min', 0.10), 0.0, 0.5))
    phi = float(np.clip(parameters.get('phi', 1.05), 0.3, 3.0))
    s_stop = float(np.clip(parameters.get('s_stop', 8.0), 0.5, 60.0))
    d_com = float(np.clip(parameters.get('d_com', 2.50), 0.5, 8.0))
    eps = float(np.clip(parameters.get('epsilon', 0.035), 0.0, 0.5))

    def _sig(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))

    # ------------------------------------------------------------------
    # 4. reason count, graded by stated validity (no position here)
    # ------------------------------------------------------------------
    x = np.power(vs, rho)
    idxs = np.flatnonzero(disc)
    m = int(idxs.shape[0])
    sgn = np.sign(d[idxs])
    xs = x[idxs]
    V = float(np.sum(sgn * xs))

    kA = int(np.count_nonzero(sgn > 0))
    kB = int(np.count_nonzero(sgn < 0))

    # ------------------------------------------------------------------
    # 5. display salience over ABSOLUTE SCREEN POSITION (first & last
    #    rows of the panel), deployed as a TWO-REGIME tie-breaker:
    #    FULL force when the reason count is deadlocked, and a CONSTANT
    #    residual fraction g_res once either side has more reasons --
    #    position stops deciding, but it does not evaporate with the
    #    size of the majority.
    # ------------------------------------------------------------------
    pos = np.arange(n, dtype=float)
    e_sal_full = np.exp(-pos / tau) + np.exp(-(n - 1.0 - pos) / tau)
    e_sal = e_sal_full[idxs]
    S = float(np.sum(sgn * e_sal))
    dk = float(kA - kB)
    g = float(np.exp(-np.clip((dk * dk) / (2.0 * sigma * sigma), 0.0, 60.0)))
    g = float(max(g, g_res))

    # ------------------------------------------------------------------
    # 5b. saturating majority bonus (fires only at |dk| >= 2)
    # ------------------------------------------------------------------
    maj = min(max(abs(dk) - 1.0, 0.0), 1.0)
    M = lam * (1.0 if dk > 0 else (-1.0 if dk < 0 else 0.0)) * maj

    # ------------------------------------------------------------------
    # 6. unanimity among the speakers is categorical
    # ------------------------------------------------------------------
    uni = 0.0
    if kA == 0 or kB == 0:
        uni = ups * (1.0 if kA > 0 else -1.0)

    drive = V + mu * g * S + uni + M

    # ------------------------------------------------------------------
    # 7. gap inference from panel silence (validity = sensitivity).
    #    An expert who endorsed BOTH products has positively asserted the
    #    two are alike on his dimension: he is ALMOST as uninformative
    #    about the quality gap as one who endorsed neither (weight omega).
    #    SHARP collapse, but BOUNDED BELOW at c_min.
    # ------------------------------------------------------------------
    both1 = (~disc) & (a > 0.5) & (b > 0.5)
    both0 = (~disc) & (~both1)
    n_sil_eff = float(np.count_nonzero(both0)) + omega * float(np.count_nonzero(both1))
    f_sil = n_sil_eff / float(n)
    f_sil = float(np.clip(f_sil, 0.0, 1.0))
    tot_x = float(np.sum(xs))
    if tot_x <= 0:
        return np.array([0.5, 0.5])
    vbar = float(np.sum(xs * vs[idxs]) / tot_x)
    c = 1.0 - eta * float(_sig(sil_k * (f_sil - sil_f0))) * vbar
    c = float(np.clip(c, c_min, 1.0))

    p_int = _sig(beta * c * drive)

    # ------------------------------------------------------------------
    # 8. authority-capped first-cue commitment (display-contingent)
    # ------------------------------------------------------------------
    j1 = int(idxs[0])
    if j1 >= n - 1:
        q_com = 1.0
    else:
        v_rest = float(np.max(vs[j1 + 1:]))
        v_rest = max(v_rest, 1e-6)
        ratio = vs[j1] / (phi * v_rest)
        q_com = _sig(s_stop * (ratio - 1.0))
    q_com = float(np.clip(q_com, 0.0, 1.0))

    p_dec = _sig(beta * c * d_com)
    p_com_a = p_dec if d[j1] > 0 else (1.0 - p_dec)

    p_a = q_com * p_com_a + (1.0 - q_com) * p_int

    p = np.array([p_a, 1.0 - p_a], dtype=float)
    p = (1.0 - eps) * p + eps * 0.5
    p = np.clip(p, 1e-12, None)
    p = p / p.sum()
    return p
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if not np.isfinite(s) or s <= 0:
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / s
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** **Dilution-calibrated reason integration with an ORACLE PREMIUM and a SHARPLY-PEAKED edge-salience profile (DCRI-3s): a compressive weighted reason tally whose diagnosticity map is compressive in the mid range but regains a categorical top-end premium near the stated-validity ceiling, read out RELATIVE to the evidence actually on the table, scaled by a smooth, strictly-positive confidence that falls with panel silence and with the coarseness-inference it licenses, plus a two-regime edge-salience tie-breaker whose attention profile is concentrated on the LITERAL first and last rows (steep decay), a slowly-growing numerosity bonus, a difficulty-scaled lapse, and a firm authority-checked top-down stop.**

Mechanism family is unchanged from the accepted base; ONE calibration commitment is sharpened.

1. **Diagnosticity is compressive in the middle but has a TOP-END PREMIUM.** x_j = ((v_j-0.5)/0.5)^rho * (1 + alpha_v*sigmoid(kappa_v*(v_j - v_knee))) with rho ~0.46-0.58, v_knee ~0.91: below ~.88 subjective weight is strongly sub-linear so cheap-cue coalitions routinely outvote a merely-good cue, but a near-certain expert is treated as categorically heavier.

2. **Numerosity is a separate, slowly-growing reason.** N = lambda*sign(k_A-k_B)*sqrt(|k_A-k_B|).

3. **Read-out is RELATIVE:** drive = (V + N + P)/(kappa2 + total speaking diagnosticity).

4. **Confidence = evidence fraction x coarseness inference, never a reversal.** c = c_min + (1-c_min)*frac^psi*max(1 - eta*f_sil^q*vbar, 0), strictly positive and floored, so a lone speaking cue amid a mute panel is still followed above chance and silence only flattens toward chance.

5. **Mutual endorsement is a weaker form of silence than mutual absence** (weight omega ~0.63), giving a small positive processing-load effect.

6. **Edge salience is a two-regime tie-breaker over LITERAL screen rows with a STEEP attention decay.** s_j = exp(-j/tau)+exp(-(n-1-j)/tau) with tau ~0.62-0.88 (previously ~1.05). This is the sharpened commitment: the extra attention paid to the top and bottom of the display is *confined to the first and last rows themselves* and does not leak onto rows 1 and n-2. With a broad profile the mid-list heavyweights that sit just inside the panel edges receive almost as much positional boost as the true screen extremes, so the tie-breaker partially cancels itself in exactly the deadlocked extremes-vs-heavyweights displays it is meant to decide; with a steep profile the tie-breaker actually breaks the tie. Full force only on exactly deadlocked reason counts (narrow sigma), a small constant residual otherwise, symmetric so there is no primacy/recency swing.

7. **Choice difficulty scales the lapse**, so near-deadlocked displays draw genuine guessing while landslides stay near ceiling.

8. **Authority-checked top-down stop**: people commit at the first discriminating row only if nothing below it is more diagnostic; the committed choice is still multiplied by the same confidence c.

Individual differences are unimodal jitter in all parameters; no strategy subpopulations, no learning.

**Parameters:**
- rho: [0.46, 0.58]
- alpha_v: [0.45, 0.85]
- v_knee: [0.895, 0.925]
- kappa_v: [25.0, 45.0]
- lam: [0.48, 0.64]
- mu: [2.15, 2.65]
- tau: [0.62, 0.88]
- sigma: [0.36, 0.46]
- g_res: [0.06, 0.14]
- kappa: [0.45, 0.80]
- kap2: [0.30, 0.55]
- psi: [0.28, 0.45]
- eta: [3.05, 3.65]
- q_sil: [1.05, 1.35]
- omega: [0.55, 0.72]
- c_min: [0.06, 0.11]
- beta: [2.80, 3.50]
- eps0: [0.05, 0.11]
- eps1: [0.16, 0.28]
- q_max: [0.92, 0.99]
- d_com: [2.00, 2.55]
- phi: [0.98, 1.06]
- s_stop: [6.0, 12.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np

    # ------------------------------------------------------------------
    # 1. parse the trial stimulus into two rating vectors (screen order)
    # ------------------------------------------------------------------
    a = None
    b = None
    if isinstance(state, dict):
        if 'option_a_ratings' in state and 'option_b_ratings' in state:
            a = np.asarray(list(state['option_a_ratings']), dtype=float)
            b = np.asarray(list(state['option_b_ratings']), dtype=float)
    if a is None:
        arr = np.asarray(state, dtype=float)
        arr = arr.reshape(2, -1)
        a = arr[0].astype(float)
        b = arr[1].astype(float)

    a = np.ravel(a).astype(float)
    b = np.ravel(b).astype(float)
    n = int(min(a.shape[0], b.shape[0]))
    if n == 0:
        return np.array([0.5, 0.5])
    a = a[:n]
    b = b[:n]
    d = a - b
    disc = np.abs(d) > 1e-12
    if not np.any(disc):
        return np.array([0.5, 0.5])

    # ------------------------------------------------------------------
    # 2. communicated validities -> scaled diagnosticity
    # ------------------------------------------------------------------
    val = parameters.get('validities', None)
    try:
        val = np.asarray(val, dtype=float).ravel()
    except Exception:
        val = None
    if val is None or val.shape[0] != n:
        val = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.90])
    val = np.clip(val, 0.5 + 1e-6, 1.0 - 1e-9)
    vs = np.clip((val - 0.5) / 0.5, 1e-6, 1.0)

    # ------------------------------------------------------------------
    # 3. parameters
    # ------------------------------------------------------------------
    def _p(key, dflt, lo, hi):
        try:
            v = float(parameters.get(key, dflt))
        except Exception:
            v = float(dflt)
        if not np.isfinite(v):
            v = float(dflt)
        return float(np.clip(v, lo, hi))

    rho = _p('rho', 0.52, 0.05, 4.0)
    alpha_v = _p('alpha_v', 0.62, 0.0, 3.0)
    v_knee = _p('v_knee', 0.910, 0.55, 0.999)
    kappa_v = _p('kappa_v', 35.0, 1.0, 200.0)
    lam = _p('lam', 0.56, 0.0, 3.0)
    mu = _p('mu', 2.40, 0.0, 8.0)
    tau = _p('tau', 0.75, 0.2, 8.0)
    sigma = _p('sigma', 0.41, 0.15, 3.0)
    g_res = _p('g_res', 0.10, 0.0, 1.0)
    kappa = _p('kappa', 0.60, 0.05, 5.0)
    kap2 = _p('kap2', 0.42, 0.02, 5.0)
    psi = _p('psi', 0.35, 0.01, 3.0)
    eta = _p('eta', 3.35, 0.0, 10.0)
    q_sil = _p('q_sil', 1.20, 0.3, 8.0)
    omega = _p('omega', 0.63, 0.0, 1.0)
    c_min = _p('c_min', 0.085, 0.0, 0.5)
    beta = _p('beta', 3.10, 0.05, 40.0)
    eps0 = _p('eps0', 0.08, 0.0, 0.5)
    eps1 = _p('eps1', 0.22, 0.0, 0.6)
    q_max = _p('q_max', 0.96, 0.0, 1.0)
    d_com = _p('d_com', 2.25, 0.0, 6.0)
    phi = _p('phi', 1.02, 0.5, 3.0)
    s_stop = _p('s_stop', 9.0, 0.5, 60.0)

    def _sig(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))

    # ------------------------------------------------------------------
    # 4. compressive reason tally WITH A TOP-END (ORACLE) PREMIUM
    #    mid-range validity is read coarsely (rho < 1), but a near-certain
    #    expert regains a categorical extra weight.
    # ------------------------------------------------------------------
    prem = 1.0 + alpha_v * _sig(kappa_v * (val - v_knee))
    x = np.power(vs, rho) * prem
    idxs = np.flatnonzero(disc)
    sgn = np.sign(d[idxs])
    xs = x[idxs]
    V = float(np.sum(sgn * xs))
    Xs = float(np.sum(xs))
    if Xs <= 1e-12:
        return np.array([0.5, 0.5])

    kA = int(np.count_nonzero(sgn > 0))
    kB = int(np.count_nonzero(sgn < 0))
    dk = float(kA - kB)

    # ------------------------------------------------------------------
    # 5. numerosity: a separate reason that grows only as sqrt(|dk|)
    # ------------------------------------------------------------------
    N = lam * (1.0 if dk > 0 else (-1.0 if dk < 0 else 0.0)) * float(np.sqrt(abs(dk)))

    # ------------------------------------------------------------------
    # 6. symmetric edge salience, full force only on deadlocked counts.
    #    STEEP decay (tau < 1): the attention bonus is confined to the
    #    literal first and last rows and does not leak onto rows 1/n-2.
    # ------------------------------------------------------------------
    pos = np.arange(n, dtype=float)
    s_sal = np.exp(-pos / tau) + np.exp(-(n - 1.0 - pos) / tau)
    S = float(np.sum(sgn * s_sal[idxs]))
    g = g_res + (1.0 - g_res) * float(np.exp(-np.clip((dk * dk) / (2.0 * sigma * sigma), 0.0, 60.0)))
    P = mu * g * S

    # ------------------------------------------------------------------
    # 7. RELATIVE read-out: evidence judged against evidence available
    # ------------------------------------------------------------------
    drive = (V + N + P) / (kap2 + Xs)

    # ------------------------------------------------------------------
    # 8. confidence = evidence-fraction dilution x coarseness inference
    #    (smooth, strictly positive, never sign-reversing)
    # ------------------------------------------------------------------
    both1 = (~disc) & (a > 0.5) & (b > 0.5)
    sil_w = np.where(disc, 0.0, np.where(both1, omega, 1.0))
    sil_mass = float(np.sum(sil_w * x))
    f_sil = float(np.sum(sil_w)) / float(n)
    f_sil = float(np.clip(f_sil, 0.0, 1.0))

    frac = Xs / (kappa + Xs + sil_mass)
    frac = float(np.clip(frac, 1e-9, 1.0))
    vbar = float(np.sum(xs * vs[idxs]) / Xs)
    gap = 1.0 - eta * (f_sil ** q_sil) * vbar
    gap = float(max(gap, 0.0))
    c = c_min + (1.0 - c_min) * (frac ** psi) * gap
    c = float(np.clip(c, 0.0, 1.0))

    p_int = float(_sig(beta * c * drive))

    # ------------------------------------------------------------------
    # 9. authority-checked top-down stop (display-contingent, capped)
    # ------------------------------------------------------------------
    j1 = int(idxs[0])
    if j1 >= n - 1:
        q_com = q_max
    else:
        v_rest = float(np.max(vs[j1 + 1:]))
        v_rest = max(v_rest, 1e-6)
        ratio = float(vs[j1]) / (phi * v_rest)
        q_com = q_max * float(_sig(s_stop * (ratio - 1.0)))
    q_com = float(np.clip(q_com, 0.0, 1.0))

    p_dec = float(_sig(beta * c * d_com))
    p_com_a = p_dec if d[j1] > 0 else (1.0 - p_dec)

    p_a = q_com * p_com_a + (1.0 - q_com) * p_int
    p_a = float(np.clip(p_a, 1e-9, 1.0 - 1e-9))

    # ------------------------------------------------------------------
    # 10. choice-difficulty-scaled lapse
    # ------------------------------------------------------------------
    z = float(np.log(p_a / (1.0 - p_a)))
    eps = eps0 + eps1 * float(np.exp(-min(abs(z), 60.0)))
    eps = float(np.clip(eps, 0.0, 0.9))
    p_a = 0.5 + (p_a - 0.5) * (1.0 - eps)

    p = np.array([p_a, 1.0 - p_a], dtype=float)
    p = np.clip(p, 1e-12, None)
    p = p / p.sum()
    return p
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if not np.isfinite(s) or s <= 0:
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / s
    return int(np.random.choice(len(probs), p=probs))
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n\n    K2 = \"01110101111111\"   # cell 2: 3-0 unanimous, 4 mutual-endorsement mute rows\n    K3 = \"01110101111101\"   # cell 3: identical but the weak pos5 expert dissents (3-1)\n    K6 = \"10101011101011\"   # cell 6: oracle+.86 (2 speakers) vs 3 cheap experts\n    K7 = \"10101001101011\"   # cell 7: same, plus one more cheap dissenter at the bottom edge (2-4)\n\n    def canon(a, b):\n        sa = ''.join([str(int(v)) for v in a])\n        sb = ''.join([str(int(v)) for v in b])\n        return min(sa + sb, sb + sa)\n\n    acc = {K2: [0.0, 0.0], K3: [0.0, 0.0], K6: [0.0, 0.0], K7: [0.0, 0.0]}\n\n    for _, row in data.iterrows():\n        try:\n            a = [int(v) for v in row['option_a_ratings']]\n            b = [int(v) for v in row['option_b_ratings']]\n            r = int(row['response'])\n        except Exception:\n            continue\n        if len(a) != len(b) or len(a) == 0:\n            continue\n        k = canon(a, b)\n        if k not in acc:\n            continue\n        kA = sum(1 for i in range(len(a)) if a[i] > b[i])\n        kB = sum(1 for i in range(len(a)) if b[i] > a[i])\n        if kA == kB:\n            continue\n        if k in (K2, K3):\n            # did the subject follow the numerically larger camp?\n            hit = 1.0 if ((kA > kB and r == 0) or (kB > kA and r == 1)) else 0.0\n        else:\n            # did the subject follow the smaller, high-validity (oracle-backed) camp?\n            hit = 1.0 if ((kA < kB and r == 0) or (kB < kA and r == 1)) else 0.0\n        acc[k][0] += hit\n        acc[k][1] += 1.0\n\n    def rate(k):\n        s, n = acc[k]\n        if n <= 0:\n            return None\n        return s / n\n\n    p2, p3, p6, p7 = rate(K2), rate(K3), rate(K6), rate(K7)\n\n    comps = []\n    if (p2 is not None) and (p3 is not None):\n        comps.append(p3 - p2)          # dissenter-raises-confidence effect\n    if (p6 is not None) and (p7 is not None):\n        comps.append(p6 - p7)          # one-extra-cheap-dissenter reversal\n    if len(comps) == 0:\n        return 0.0\n    return float(np.mean(comps))\n",
  "rationale": "Two within-subject, validity-matched contrasts that the two accounts order very differently, averaged to cut binomial noise.\n\n(1) SILENCE-CLIFF / DISSENTER CONTRAST (cell 3 minus cell 2). Cell 2 is a 3-0 unanimous display with four mutual-endorsement (1,1) rows; cell 3 is identical except the cheap pos5 expert now dissents (3-1, three mute rows). The advocated model has a SHARP silence sigmoid (f0~0.36) with mutual endorsements weighted omega~0.82: cell 2 sits above the cliff (effective silence .469) so confidence collapses to ~.11 and P(majority)~.65, whereas cell 3 falls just below the cliff (.351) so confidence jumps to ~.65 and P(majority)~.90. Hand computation at central parameters gives +0.24, and even at the unfavourable corners of the parameter box the sign stays positive. DCRI-3s uses a SMOOTH power-law silence discount plus a RELATIVE read-out, so its drive drop and confidence rise cancel almost exactly: .803 -> .800, i.e. ~0.00. Note the level ordering also reverses (DCRI is 15 pts above the advocate in cell 2 and 10 pts below in cell 3), so no validity-only or side-bias account can mimic this.\n\n(2) SATURATING-MAJORITY / EDGE CONTRAST (cell 6 minus cell 7). Cell 6 pits the oracle (.96) plus .86 against three cheap/mid experts (2-3); cell 7 adds exactly one more cheap dissenter at the bottom screen edge (2-4). For the advocated model this single row simultaneously switches on the categorical saturating majority bonus (-0.67) and adds a full edge-salience unit at the residual gate, flipping P(oracle side) from ~.63 to ~.21 (a 42-point, direction-reversing swing). DCRI's sqrt numerosity, compressive validity map with an oracle premium and growing relative denominator produce only ~.507 -> .399 (11 points). This contrast involves essentially no silence machinery, so it is robust to the parameters that drive contrast (1).\n\nComposite prediction: advocated ~0.33, DCRI ~0.05. Each cell contributes 12 trials per subject, so per-subject binomial variance of the average contrast is ~0.015, tight enough for N=25 despite parameter jitter. The metric is well defined on a single subject's slice and on the pooled frame alike."
}
```

## Usage

```json
{
  "input_tokens": 16446,
  "output_tokens": 18813
}
```
