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
**Validities (n_features=7):** [0.78, 0.95, 0.62, 0.78, 0.62, 0.95, 0.78]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 1, 0]
  trial 2: A=[0, 1, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 1, 1]
  trial 3: A=[1, 0, 1, 1, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1, 1]
  trial 4: A=[0, 1, 1, 0, 0, 1, 1]  B=[1, 0, 1, 1, 1, 1, 0]
  trial 5: A=[0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 1, 0]
  trial 6: A=[1, 1, 1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 1]
  trial 7: A=[0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  trial 9: A=[1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1, 1]
  trial 10: A=[1, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1]
  trial 11: A=[0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0]
  trial 13: A=[1, 0, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 0, 0]
  trial 14: A=[0, 1, 1, 1, 1, 0, 0]  B=[1, 0, 1, 1, 1, 1, 1]
  trial 15: A=[1, 0, 0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0]
  trial 16: A=[0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 1]

**Rationale:** GOAL: dissociate the ADVOCATED two-regime salience-gate reason-counter (SYMMETRIC first/last-row salience that acts at full force only on a deadlocked vote and otherwise persists as a CONSTANT residual g_res~.25; categorical unanimity bonus ups~1.5; SHARP silence threshold at f0~.36 with mutual endorsements down-weighted by omega~.82; confidence FLOORED at c_min~.10; small lapse eps~.035) from the COMPETING top-down reading-attention tally (cue weights multiplied by att_j = 1+alpha_p*exp(-j/tau_p), so the TOP row is worth ~1.65x itself and the bonus is gone by row 4; Gaussian-only edge gate that collapses to ~.08-.11 once the counts are not deadlocked; GRADED, REVERSIBLE silence term c = 1-eta*f_sil^q*vbar that can pass NEGATIVE; no unanimity term; large lapse eps~.16).

LAYOUT: n=7, validities by SCREEN position [.78,.95,.62,.78,.62,.95,.78] 
- deliberately PALINDROMIC, which is the central design move. Because the validity profile is mirror-symmetric and the advocated model's salience s_sal = exp(-j/tau)+exp(-(n-1-j)/tau) = [1.00,.375,.153,.0996,.153,.375,1.00] is also mirror-symmetric, the advocated model is ANALYTICALLY INVARIANT to top/bottom exchanges of validity-matched rows, while the competitor's primacy multiplier ([1.65,1.28,1.12,1.05,1.02,1.01,1.00]) is not. Three rows share v=.78 (pos0 = top edge, pos3 = exact middle, pos6 = bottom edge) so an edge and the middle can be swapped at matched validity; pos1 and pos5 share v=.95 and pos2/pos4 share v=.62. The near-oracle at pos1 (never pos0) keeps the authority-capped first-cue commitment suppressed in the dense cells (ratio .56/(1.05*.90)=.59 -> q_com=.04 advocated, ~.002 competitor), so no contrast is contaminated by the commitment term.

8 base cells, each mirrored across sides -> 16 unique pairs, K=6 -> 96 trials; mirroring forces an exact 8A/8B split for BOTH models, so side bias cannot mimic either.

(1) FLAGSHIP TRIAD X / Y / Z (pairs 1-6). Identical background (pos1->B, pos4->A, two silent mutual endorsements at pos2/pos5) and identical validity multisets; only the sign pattern of the three matched .78 rows changes: X = both EDGES favour A, MIDDLE favours B; Y = top+middle favour A, bottom favours B; Z = middle+bottom favour A, top favours B. All three have kA=3, kB=2 (dk=1), so the advocated gate sits at its residual and the competitor's at .114, and the advocated's vote V is ANALYTICALLY IDENTICAL (-.073) in all three. Advocated salience sums: X = +1.683, Y = Z = -0.122 -> P(A) ~ .70 / .48 / .45, i.e. a large EDGE effect (X-Y = +.22) and an exact TOP-vs-BOTTOM null (Y-Z = +.03, residue of the 4% commitment term only). Competitor: the attention multiplier makes the top row worth 1.65x and the bottom row 1.00x, so V = +.024 / +.080 / -.664 -> P(A) ~ .59 / .51 / .33, i.e. a small edge effect (X-Y = +.08) and a LARGE top/bottom asymmetry (Y-Z = +.18). Two orthogonal contrasts with opposite profiles: symmetric-residual salience predicts X>>Y=Z, primacy attention predicts X~Y>>Z.

(2) LONE-ORACLE SILENCE-COMPOSITION PAIR (pairs 7-10), direction disagreement + forced null. Only pos1 (the .95 expert) discriminates, favouring one option; the other six rows are mutual ABSTENTIONS (ORACLE-0) or mutual ENDORSEMENTS (ORACLE-1). For the advocated model f_sil = 6/7 or .82*6/7, both far above the f0~.36 step, and eta*vbar >= 1.45*.90 > 1, so c is FLOORED at c_min in BOTH cells: it is mathematically forced to predict the same value in both, and (because of the categorical unanimity bonus, kB=0) that value is reliably ABOVE chance, P(A) ~ .54-.63 for its entire box. The competitor's reversible power-law term gives c < 0 for the abstention version at every parameter setting (eta*f^q*vbar >= 2.1*.646*.90 = 1.22 > 1), so it predicts choosing AGAINST the lone near-oracle, P(A) ~ .15-.42 (nominal .31), while the endorsement version lifts c to ~+.19 and it predicts P(A) ~ .61. So: a ~27-point DIRECTION disagreement on ORACLE-0, plus a ~30-point within-subject reversal for the competitor where the advocated theory predicts an exact null.

(3) LONE-WEAK-EXPERT CELL (pairs 11-12). Only pos2 (the weakest .62 expert) discriminates, six mutual abstentions. The advocated model's unanimity bonus (+1.5) plus a high c (low vbar keeps eta*vbar small: c ~ .63) predicts a confident P(A) ~ .79; the competitor has no unanimity term and a similar c, predicting only ~.56. A 22-point level gap on a single-cue display, and (paired with ORACLE-0) two different INVERTED ladders: advocated .79 (lone .62) vs .57 (lone .95), competitor .57 vs .31.

(4) SILENCE-COMPOSITION CONTRAST AT FIXED d (pairs 13-16). Identical difference vector d = [+1,-1,0,0,0,+1,+1] (kA=3, kB=1, so both majority bonuses are saturated and identical) with the THREE silent rows as mutual endorsements (SIL-END) or mutual abstentions (SIL-ABS). n=7 with 3 silent rows is chosen precisely because the advocated model's effective silence fraction straddles its sharp threshold (omega*3/7 = .351 vs 3/7 = .429 against f0 in [.345,.375]): its c collapses from ~.51 to the floor .10, so P(A) drops from ~.83 to ~.58, a ~25-point ordered drop produced by ratings that discriminate nothing. The competitor's smooth f^q term moves c only .886 -> .795, P(A) .85 -> .83, a ~2-point null. This is the exact mirror image of contrast (2): here the advocated model must move a lot and the competitor must not.

DIAGNOSTIC YIELD: (i) an analytic top/bottom null for the advocated model against an 18-point primacy swing for the competitor, together with the complementary 22-vs-8-point edge effect (cells X/Y/Z); (ii) a full direction disagreement on ORACLE-0 (above vs below chance) plus a forced null for the advocated where the competitor must reverse by 30 points; (iii) a 25-vs-2-point silence-composition cliff in the opposite direction; (iv) a 22-point unanimity level gap. Every pair has at least one discriminating cue, every contrast is within-subject, and each cell is mirrored across sides.

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
**Description:** **Noisy compensatory reason-counting with TOP-DOWN READING ATTENTION (primacy-graded cue weights), mildly compressed diagnosticity, a count-contingent edge gate, a saturating numerosity bonus, a reversible gap inference, an authority-capped top-down entry, and a moderated read-out ceiling.**

This is the accepted compensatory-tally base with ONE new commitment: *position is not only a deadlock tie-breaker; it is an attention multiplier on the reasons themselves.*

1. **Direction is a weighted reason tally over discriminating experts.** x_j = ((v_j-.5)/.5)^rho with rho ~.82-1.08: validity is read coarsely enough that numerosity routinely outvotes a single strong expert, but a .95 oracle and a mid cue are not interchangeable.
2. **NEW: people read the panel top-down and the rows they read first carry extra weight beyond their stated validity.** Each cue's contribution to the tally is multiplied by att_j = 1 + alpha_p*exp(-j/tau_p) (alpha_p ~.45-.85, tau_p ~1.0-1.6), i.e. the first screen row is worth ~1.5-1.8 of itself, the second ~1.3, and by the fourth row the bonus is gone. This is a *graded* primacy of reading, not a first-cue capture and not a lottery: every row still contributes, and a mid-list oracle still outweighs a weak top row when the validity gap is large. It is what makes displays look partly lexicographic in designs whose validities descend down the screen, and what makes screen-top cues systematically over-followed in designs that hide the heavyweights in the middle.
3. **'More experts back this one' is a single discrete saturating extra reason** (fires at |k_A-k_B|>=2, never grows). No unanimity term.
4. **Screen position ALSO survives as a symmetric edge tie-breaker on deadlocked counts** (gain g = g_res + (1-g_res)exp(-dk^2/2 sigma^2), narrow sigma, small residual), which is the only thing that can decide displays where the reason counts tie exactly.
5. **Gap inference from panel silence is graded and reversible**: validity is read as sensitivity, c = 1 - eta*f_sil^q*vbar (mutual endorsements count as silence at weight omega<1), allowed to pass slightly negative, which is the sole mechanism that makes people choose against a lone highly-sensitive endorsement.
6. **Authority-capped top-down entry**: people commit to the first discriminating row only if no more diagnostic expert remains below it.
7. **Compressive read-out**: mild dilution by the number of speaking cues, logistic gain beta ~1.45-1.80, fixed lapse eps ~.13-.19 bounding P(choice) in roughly [.12,.88]. Individual differences are unimodal jitter in all parameters; no subpopulations, no learning.

**Parameters:**
- rho: [0.82, 1.08]
- lam: [0.40, 0.65]
- mu: [1.55, 1.95]
- tau: [0.85, 1.25]
- sigma: [0.33, 0.45]
- g_res: [0.05, 0.11]
- eta: [2.10, 2.70]
- q_sil: [2.30, 2.90]
- omega: [0.70, 0.90]
- c_neg: [-0.55, -0.15]
- psi: [0.05, 0.25]
- beta: [1.45, 1.80]
- d_com: [1.80, 2.50]
- phi: [0.92, 1.04]
- s_stop: [12.0, 22.0]
- epsilon: [0.13, 0.19]
- alpha_p: [0.45, 0.85]
- tau_p: [1.00, 1.60]
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

    rho = _p('rho', 0.95, 0.05, 4.0)
    lam = _p('lam', 0.52, 0.0, 3.0)
    mu = _p('mu', 1.75, 0.0, 6.0)
    tau = _p('tau', 1.00, 0.2, 8.0)
    sig_g = _p('sigma', 0.39, 0.15, 3.0)
    g_res = _p('g_res', 0.08, 0.0, 1.0)
    eta = _p('eta', 2.40, 0.0, 8.0)
    q_sil = _p('q_sil', 2.60, 1.0, 8.0)
    om = _p('omega', 0.80, 0.0, 1.0)
    c_neg = _p('c_neg', -0.35, -1.5, 0.5)
    psi = _p('psi', 0.15, 0.0, 1.0)
    beta = _p('beta', 1.60, 0.05, 40.0)
    d_com = _p('d_com', 2.20, 0.2, 8.0)
    phi = _p('phi', 0.98, 0.5, 3.0)
    s_stop = _p('s_stop', 17.0, 0.5, 60.0)
    eps = _p('epsilon', 0.16, 0.0, 0.5)
    alpha_p = _p('alpha_p', 0.65, 0.0, 2.0)
    tau_p = _p('tau_p', 1.20, 0.2, 8.0)

    def _sig(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))

    # ------------------------------------------------------------------
    # 4. reason count with COMPRESSED diagnosticity, now scaled by
    #    TOP-DOWN READING ATTENTION (graded primacy over screen rows)
    # ------------------------------------------------------------------
    x = np.power(vs, rho)
    pos_all = np.arange(n, dtype=float)
    att = 1.0 + alpha_p * np.exp(-pos_all / tau_p)
    xw = x * att

    idxs = np.flatnonzero(disc)
    m = int(idxs.shape[0])
    sgn = np.sign(d[idxs])
    xs = x[idxs]          # raw diagnosticity (used by the silence term)
    ws = xw[idxs]         # reading-attention weighted reasons
    V = float(np.sum(sgn * ws))

    kA = int(np.count_nonzero(sgn > 0))
    kB = int(np.count_nonzero(sgn < 0))
    dk = float(kA - kB)

    # ------------------------------------------------------------------
    # 5. saturating numerosity bonus (fires once, at |dk| >= 2)
    # ------------------------------------------------------------------
    Mb = lam * (1.0 if dk > 0 else (-1.0 if dk < 0 else 0.0)) \
         * float(min(max(abs(dk) - 1.0, 0.0), 1.0))

    # ------------------------------------------------------------------
    # 6. symmetric edge salience, active mainly on deadlocked counts
    # ------------------------------------------------------------------
    pos = np.arange(n, dtype=float)
    s_sal = np.exp(-pos / tau) + np.exp(-(n - 1.0 - pos) / tau)
    S = float(np.sum(sgn * s_sal[idxs]))
    g = g_res + (1.0 - g_res) * float(np.exp(-np.clip((dk * dk) / (2.0 * sig_g * sig_g), 0.0, 60.0)))
    Pterm = mu * g * S

    # ------------------------------------------------------------------
    # 7. graded, REVERSIBLE gap inference (validity = sensitivity)
    # ------------------------------------------------------------------
    both1 = (~disc) & (a > 0.5) & (b > 0.5)
    both0 = (~disc) & (~both1)
    f_sil = (float(np.count_nonzero(both0)) + om * float(np.count_nonzero(both1))) / float(n)
    f_sil = float(np.clip(f_sil, 0.0, 1.0))
    tot = float(np.sum(xs))
    if tot <= 1e-12:
        return np.array([0.5, 0.5])
    vbar = float(np.sum(xs * vs[idxs]) / tot)
    c = 1.0 - eta * (f_sil ** q_sil) * vbar
    c = float(max(c, c_neg))

    # ------------------------------------------------------------------
    # 8. mild dilution by the number of speaking experts + read-out
    # ------------------------------------------------------------------
    denom = float(m) ** psi if m > 0 else 1.0
    drive = (V + Mb + Pterm) / max(denom, 1e-9)
    p_int = float(_sig(beta * c * drive))

    # ------------------------------------------------------------------
    # 9. authority-capped top-down entry
    # ------------------------------------------------------------------
    j1 = int(idxs[0])
    if j1 >= n - 1:
        q_com = 1.0
    else:
        v_rest = float(np.max(vs[j1 + 1:]))
        v_rest = max(v_rest, 1e-6)
        ratio = float(vs[j1]) / (phi * v_rest)
        q_com = float(_sig(s_stop * (ratio - 1.0)))
    q_com = float(np.clip(q_com, 0.0, 1.0))

    p_dec = float(_sig(beta * c * d_com))
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    sums = {'O0': 0.0, 'O1': 0.0, 'SE': 0.0, 'SA': 0.0}\n    cnts = {'O0': 0, 'O1': 0, 'SE': 0, 'SA': 0}\n\n    for _, row in data.iterrows():\n        try:\n            a = np.asarray(list(row['option_a_ratings']), dtype=float)\n            b = np.asarray(list(row['option_b_ratings']), dtype=float)\n            resp = int(row['response'])\n        except Exception:\n            continue\n        n = int(min(a.shape[0], b.shape[0]))\n        if n < 7:\n            continue\n        a = a[:n]\n        b = b[:n]\n        d = a - b\n        disc = [j for j in range(n) if abs(d[j]) > 1e-9]\n\n        cell = None\n        fav_a = None\n\n        if len(disc) == 1 and disc[0] == 1:\n            # lone near-oracle (position 1, validity .95) discriminates\n            silent = [j for j in range(n) if j != 1]\n            all_one = all(a[j] > 0.5 and b[j] > 0.5 for j in silent)\n            all_zero = all(a[j] < 0.5 and b[j] < 0.5 for j in silent)\n            if all_one:\n                cell = 'O1'\n            elif all_zero:\n                cell = 'O0'\n            if cell is not None:\n                fav_a = d[1] > 0\n        elif set(disc) == set([0, 1, 5, 6]):\n            sil = [2, 3, 4]\n            all_one = all(a[j] > 0.5 and b[j] > 0.5 for j in sil)\n            all_zero = all(a[j] < 0.5 and b[j] < 0.5 for j in sil)\n            if all_one:\n                cell = 'SE'\n            elif all_zero:\n                cell = 'SA'\n            if cell is not None:\n                # majority (3 vs 1) side: position 0 sign marks it\n                fav_a = d[0] > 0\n\n        if cell is None or fav_a is None:\n            continue\n\n        chose_a = (resp == 0)\n        chose_fav = 1.0 if (chose_a == bool(fav_a)) else 0.0\n        sums[cell] += chose_fav\n        cnts[cell] += 1\n\n    def prop(k):\n        if cnts[k] == 0:\n            return 0.5\n        return sums[k] / float(cnts[k])\n\n    p_SE = prop('SE')\n    p_SA = prop('SA')\n    p_O1 = prop('O1')\n    p_O0 = prop('O0')\n\n    silence_cliff = p_SE - p_SA          # advocated: large (+.25), competitor: ~0\n    oracle_reversal = p_O1 - p_O0        # advocated: ~0 (floored c), competitor: large (+.30)\n\n    return float(silence_cliff - oracle_reversal)\n",
  "rationale": "Two mechanistically opposite predictions are combined into a single difference-of-differences, so that the two theories land on opposite signs.\n\n(1) Silence-composition cliff at FIXED difference vector (trials 13-16, d=[+1,-1,0,0,0,+1,+1], 3 silent rows either mutual endorsements or mutual abstentions). The advocated model has a SHARP sigmoid silence threshold at f0~.36 with mutual endorsements down-weighted by omega~.82: omega*3/7=.351 sits just below the threshold while 3/7=.429 sits just above, so its read-out gain collapses from c~.51 to the floor c_min=.10 and P(majority option) falls from ~.83 to ~.58 (a ~25-point drop produced by ratings that discriminate nothing). The competitor's smooth, power-law silence term moves c only .89 -> .80, giving a ~2-point drop. So silence_cliff ~ +.25 (advocated) vs ~ +.02 (competitor).\n\n(2) Lone-oracle composition reversal (trials 7-10, only the .95 expert at row 1 discriminates; the other six rows are all mutual 0s or all mutual 1s). For the advocated model f_sil is far above its threshold in BOTH cells and eta*vbar>1, so c is FLOORED at c_min in both: it is analytically forced to give the same choice probability (~.57, above chance thanks to the categorical unanimity bonus). The competitor's REVERSIBLE silence term passes negative for the all-abstain version (c ~ -.35, predicting choice AGAINST the lone oracle, P~.31) but stays positive for the all-endorse version (c ~ +.19, P~.61). So oracle_reversal ~ 0 (advocated) vs ~ +.30 (competitor).\n\nSubtracting gives ~ +0.25 under the advocated theory and ~ -0.29 under the competitor: a ~0.54 gap with opposite signs. Each of the four cells is estimated from 12 trials per subject (2 mirrored pairs x 6 reps), all contrasts are within-subject and side-balanced, so per-subject noise (sd~0.27) is small relative to the between-theory gap at N=25."
}
```

## Usage

```json
{
  "input_tokens": 15415,
  "output_tokens": 8047
}
```
