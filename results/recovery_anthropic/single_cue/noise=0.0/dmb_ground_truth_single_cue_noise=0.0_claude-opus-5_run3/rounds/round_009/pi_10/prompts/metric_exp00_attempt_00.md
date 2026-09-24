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
**Validities (n_features=7):** [0.7, 0.62, 0.96, 0.6, 0.64, 0.84, 0.7]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 0, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1, 1]
  trial 2: A=[1, 0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1, 1, 1]
  trial 3: A=[1, 1, 0, 1, 1, 1, 1]  B=[1, 0, 1, 0, 1, 1, 1]
  trial 4: A=[1, 0, 1, 0, 1, 1, 1]  B=[1, 1, 0, 1, 1, 1, 1]
  trial 5: A=[1, 1, 0, 1, 1, 1, 1]  B=[1, 0, 1, 0, 0, 1, 1]
  trial 6: A=[1, 0, 1, 0, 0, 1, 1]  B=[1, 1, 0, 1, 1, 1, 1]
  trial 7: A=[1, 1, 0, 1, 1, 1, 1]  B=[0, 0, 1, 0, 0, 1, 1]
  trial 8: A=[0, 0, 1, 0, 0, 1, 1]  B=[1, 1, 0, 1, 1, 1, 1]
  trial 9: A=[1, 1, 0, 1, 1, 0, 1]  B=[1, 0, 1, 0, 0, 1, 1]
  trial 10: A=[1, 0, 1, 0, 0, 1, 1]  B=[1, 1, 0, 1, 1, 0, 1]
  trial 11: A=[1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1]
  trial 13: A=[1, 1, 1, 0, 0, 0, 1]  B=[1, 1, 0, 1, 1, 1, 1]
  trial 14: A=[1, 1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 1]
  trial 15: A=[1, 1, 0, 1, 1, 0, 1]  B=[0, 0, 1, 0, 0, 1, 0]
  trial 16: A=[0, 0, 1, 0, 0, 1, 0]  B=[1, 1, 0, 1, 1, 0, 1]

**Rationale:** GOAL: dissociate DCRI-3s (advocated: STRONGLY COMPRESSIVE diagnosticity rho~0.52 with an oracle premium above v_knee~0.91, SQRT numerosity lam*sqrt(|dk|), relative read-out (V+N+P)/(kap2+Xs) with a modest beta~3.1, steep-but-present edge salience with residual g_res~0.10, smooth power-law silence discount times an evidence-fraction term, strong commitment read-out beta*d_com~7c, and a DIFFICULTY-SCALED lapse that runs from ~0.30 on deadlocks down to ~0.08 on landslides) from GIT-AL+P/k (competitor: SUPERLINEAR diagnosticity rho~1.6, LINEAR and tiny numerosity lam~0.135, literal-edge anchor with NO residual, superlinear mass normalisation with a large beta~8.43, and a HARD difficulty-independent lapse eps~0.32 that caps every prediction inside [0.16,0.84]).

LAYOUT: n=7, validities by SCREEN position [0.70, 0.62, 0.96, 0.60, 0.64, 0.84, 0.70]. The near-oracle (.96, above DCRI's knee so its top-end premium fires) sits at pos2 and the second heavyweight (.84) at pos5, both in the display interior where DCRI's steep salience profile is ~0.07-0.26 and the competitor's literal-edge indicator is exactly 0. The cheap experts (.62, .60, .64) occupy pos1/pos3/pos4 (also interior), and the two screen extremes are validity-matched (.70/.70) fillers. Consequence: position machinery is essentially inert in the diagnostic cells for BOTH models, so nothing below can be blamed on salience or primacy. In every conflict cell the first discriminating row is a cheap or mid expert with a far more diagnostic expert printed below it (ratio <= 0.43), so both models' authority-checked commitment term is suppressed (q_com <= 0.01) except in the one cell where it is deliberately the target.

The crucial quantitative fact: with vs=(v-0.5)/0.5, the cheap experts get DCRI weights 0.43-0.52 versus an oracle weight of 1.46 (ratio ~3), but competitor weights 0.08-0.14 versus 0.95 (ratio ~9). DCRI therefore lets a coalition of two-to-three cheap cues outvote the oracle (helped by its sqrt numerosity, +0.79 at |dk|=2), whereas the competitor's expansive weighting plus near-negligible linear numerosity keeps the oracle dominant until FOUR cheap cues oppose it.

8 base cells, each mirrored across sides -> 16 unique pairs, K=6 -> 96 trials; mirroring forces an exact 8A/8B split for both theories.

(1) COALITION LADDER, cells 1-4 (pairs 1-8). The oracle at pos2 opposes a coalition of 1, 2, 3 or 4 cheap/mid experts; all remaining rows are mutual endorsements. Nominal predictions P(coalition side): DCRI .49 / .51 / .58 / .70 (monotone, crossing chance between coalition size 1 and 2); competitor .38 / .28 / .32 / .53 (below chance until size 4, and NON-MONOTONE because adding the second cheap cue adds far more mass to the denominator than to the numerator). The switch point of the ladder is the flagship statistic: compressive weighting + sqrt numerosity puts it at 2-3 cues, superlinear weighting + linear numerosity puts it at 4.

(2) DENSE CONFLICT, cell 5 (pairs 9-10): three cheap experts (.62,.60,.64) against the oracle (.96) plus the .84 expert, edges mute. DCRI lands at a genuine coin flip (~.47, its difficulty-scaled lapse crushing a residual drive of -0.08) while the competitor is emphatically against the coalition (~.19). A ~28-point gap where one model must be at chance and the other near its floor.

(3) FULLY DENSE 5-vs-2, cell 8 (pairs 15-16): d = [+1,+1,-1,+1,+1,-1,+1] with NO mute rows at all, so neither silence rule and neither salience gate can do any work (|dk|=3 closes both gates). DCRI: compressed vote +0.35 plus sqrt-numerosity +0.97 -> P(A) ~ .69. Competitor: vote -0.68, numerosity only +0.41 -> P(A) ~ .39. A ~31-point DIRECTION disagreement that is attributable solely to the shape of the validity-to-weight map and the growth law of numerosity.

(4) LANDSLIDE CEILING, cell 6 (pairs 11-12): a 7-0 unanimous display, no silent rows. DCRI's lapse shrinks to ~0.08 and it predicts ~.94; the competitor's hard eps=0.32 caps it at ~.84 no matter how lopsided the evidence. Combined with cells 1/5 (where DCRI is AT chance and the competitor is at .19-.38) this yields the second flagship signature: DCRI predicts a much WIDER across-cell spread (.47 -> .94) than the competitor (.19 -> .84), i.e. the two models differ in the SLOPE of accuracy against evidence strength, not just in level.

(5) ORACLE-COMMITMENT CELL, cell 7 (pairs 13-14): the oracle is the FIRST discriminating row (pos0/pos1 mute) and is more diagnostic than everything below it, so both models' authority stop fires (q_com ~.91-.92); three lower-validity experts dissent, three rows are mutual endorsements. DCRI's stronger committed read-out (sigmoid(3.1*c*2.25)) plus its small landslide-like lapse gives ~.88; the competitor's weaker commitment (sigmoid(8.43*c*0.50)) and hard lapse give ~.79. Same direction, ~9 points, and it anchors the interpretation of the conflict cells: subjects demonstrably DO follow a leading oracle here, so failure to follow it in cells 1-5 cannot be attributed to inattention to the .96 row.

DIAGNOSTIC YIELD: (i) a four-rung coalition ladder whose crossing point differs by two cues between models, with two rungs showing outright direction disagreement; (ii) two dense-display direction disagreements (~28 and ~31 points) in which silence and position machinery are provably inert; (iii) a ceiling/spread contrast (.94 vs .84) that tests difficulty-scaled versus hard lapse; (iv) a commitment anchor. Every pair has at least one discriminating cue, every contrast is within-subject, and each cell is mirrored across sides.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
**Description:** **Gap-Inference Tally with an AUTHORITY-COINCIDENCE LEAD, a small GRADED PRIMACY CREDIT, SUPERLINEAR EVIDENCE-MASS NORMALISATION, and a MODERATELY WIDENED, NEAR-DEADLOCK-LICENSED ANCHOR (GIT-AL+P/k, sigma-widened, lead-hardened).** Mechanism family is unchanged from the accepted base: a graded (non-ordinal) weighted reason tally, a weak strictly-linear numerosity reason, speaker-rarity as the ONLY confidence term (silence-composition near-blind), a first/last-screen-row anchor tie-breaker licensed by a near-deadlock in reason counts, relative read-out with superlinear mass normalisation, and a hard uniform noise ceiling. TWO calibration commitments change.

1. **Weighted reasons.** Each discriminating expert contributes w_j = ((v_j-0.5)/0.5)^rho, rho ~1.4-1.8 (smooth, no top-end premium, no knee).
2. **Small graded primacy credit** on the tally weights: w_j x (1 + pi_prim*(1 - j/(n-1))), pi_prim ~0.13 (attention multiplier, not a stop rule).
3. **Numerosity** is a separate, strictly linear, non-saturating reason lam*(k_A-k_B), lam ~0.135; it also supplies the baseline drive on 1-vs-0 and |dk|=1 displays.
4. **Speaker-rarity confidence is the only confidence term:** c = max(1 - eta*f_mute^q*s̄_speakers, c_min), with mute rows near composition-blind (omega ~0.7), so every silence-composition contrast stays ~0.
5. **CHANGED — the anchor is licensed by a NEAR-deadlock, not an exact one.** Position enters the direction term only through the literal first and last screen rows, gated by exp(-dk^2/2 sigma^2) with sigma ~0.60 instead of ~0.45. The psychological claim is that the top and bottom rows of the panel are consulted as tie-breakers whenever the reasons are *close to* even, not only when they are exactly even: at sigma=0.60 a one-reason margin still licenses about a quarter of full anchor force (g=0.25) while a two-reason margin is still effectively off (g=0.004), so the anchor remains a tie-breaker and never a majority-overrider.
6. **CHANGED — the authority-coincidence lead is close to categorical.** When the FIRST discriminating screen row is ALSO more diagnostic than anything printed below it, people essentially stop there: the lead's ceiling q_max rises from ~0.91 to ~0.97, so the committed verdict is nearly independent of what the rest of the panel says. This makes first-cue commitment equally strong whether the remaining panel agrees or disagrees with it — the empirical signature being that take-the-best following is as high on tally-conflicting displays as on tally-agreeing ones. The committed choice is still multiplied by the same rarity confidence c, so a lone oracle amid a mute panel stays near chance.
7. **Superlinear evidence-mass normalisation.** drive = (V + N + P)/(k2 + W)^kappa, kappa ~1.17: each extra speaking expert costs slightly more integration capacity than the last.
8. **Hard, difficulty-independent noise ceiling** p = 0.5 + (p-0.5)*(1-eps), eps ~0.32.

**Parameters:**
- rho: [1.40, 1.80]
- lam: [0.11, 0.16]
- pi_prim: [0.08, 0.18]
- mu: [0.85, 1.20]
- sigma: [0.56, 0.64]
- k2: [0.40, 0.58]
- kappa: [1.10, 1.24]
- eta: [4.00, 5.20]
- q: [1.90, 2.50]
- omega: [0.62, 0.80]
- c_min: [0.03, 0.08]
- beta: [6.90, 10.00]
- eps: [0.28, 0.38]
- phi: [0.98, 1.08]
- s_stop: [6.0, 12.0]
- q_max: [0.94, 0.99]
- d_com: [0.42, 0.62]
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
    # 2. communicated validities -> sensitivity / diagnosticity scale
    # ------------------------------------------------------------------
    val = parameters.get('validities', None)
    try:
        val = np.asarray(val, dtype=float).ravel()
    except Exception:
        val = None
    if val is None or val.shape[0] != n:
        val = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.90])
    val = np.clip(val, 0.5 + 1e-6, 1.0 - 1e-9)
    s = np.clip((val - 0.5) / 0.5, 1e-6, 1.0)

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

    rho = _p('rho', 1.60, 0.20, 5.0)
    lam = _p('lam', 0.135, 0.0, 2.0)
    pi_prim = _p('pi_prim', 0.13, 0.0, 1.0)
    mu = _p('mu', 1.00, 0.0, 3.0)
    sigma = _p('sigma', 0.60, 0.15, 3.0)
    k2 = _p('k2', 0.48, 0.05, 5.0)
    kappa = _p('kappa', 1.17, 0.50, 2.50)
    eta = _p('eta', 4.50, 0.0, 15.0)
    q = _p('q', 2.20, 0.5, 8.0)
    omega = _p('omega', 0.70, 0.0, 1.0)
    c_min = _p('c_min', 0.05, 0.0, 0.5)
    beta = _p('beta', 8.43, 0.1, 40.0)
    eps = _p('eps', 0.32, 0.0, 0.8)
    # authority-coincidence lead
    phi = _p('phi', 1.02, 0.5, 3.0)
    s_stop = _p('s_stop', 9.0, 0.5, 60.0)
    q_max = _p('q_max', 0.97, 0.0, 1.0)
    d_com = _p('d_com', 0.50, 0.0, 3.0)

    def _sig(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))

    # ------------------------------------------------------------------
    # 4. smooth (no-premium) diagnosticity tally over speaking experts,
    #    with a SMALL GRADED PRIMACY CREDIT on the weights (attention
    #    multiplier, monotone in screen row, never sign-reversing).
    # ------------------------------------------------------------------
    pos = np.arange(n, dtype=float)
    if n > 1:
        prim = 1.0 + pi_prim * (1.0 - pos / float(n - 1))
    else:
        prim = np.ones(n, dtype=float)
    w = np.power(s, rho) * prim
    idx = np.flatnonzero(disc)
    sgn = np.sign(d[idx])
    ws = w[idx]
    W = float(np.sum(ws))
    if W <= 1e-12:
        return np.array([0.5, 0.5])
    V = float(np.sum(sgn * ws))

    kA = int(np.count_nonzero(sgn > 0))
    kB = int(np.count_nonzero(sgn < 0))
    dk = float(kA - kB)

    # ------------------------------------------------------------------
    # 5. numerosity: linear, graded, no saturation, no unanimity step
    # ------------------------------------------------------------------
    N = lam * dk

    # ------------------------------------------------------------------
    # 6. anchor-row tie-breaker: literal first and last screen rows,
    #    licensed by a NEAR-deadlock in the number of reasons (Gaussian
    #    gate, no residual).  At sigma ~0.60 a one-reason margin retains
    #    ~25% of anchor force; a two-reason margin is still ~0.
    # ------------------------------------------------------------------
    edge = np.zeros(n, dtype=float)
    edge[0] = 1.0
    edge[n - 1] = 1.0
    S = float(np.sum(sgn * edge[idx]))
    g = float(np.exp(-min((dk * dk) / (2.0 * sigma * sigma), 60.0)))
    P = mu * g * S

    # ------------------------------------------------------------------
    # 7. read-out relative to the evidence actually on the table, with
    #    SUPERLINEAR mass normalisation.
    # ------------------------------------------------------------------
    den = float(k2 + W)
    den = max(den, 1e-9)
    drive = (V + N + P) / float(np.power(den, kappa))

    # ------------------------------------------------------------------
    # 8. rarity / sensitivity gap-inference: the ONLY confidence term.
    # ------------------------------------------------------------------
    both1 = (~disc) & (a > 0.5) & (b > 0.5)
    mute_w = np.where(disc, 0.0, np.where(both1, omega, 1.0))
    f_mute = float(np.sum(mute_w)) / float(n)
    f_mute = float(np.clip(f_mute, 0.0, 1.0))
    sbar = float(np.sum(ws * s[idx]) / W)
    c = 1.0 - eta * (f_mute ** q) * sbar
    c = float(np.clip(c, c_min, 1.0))

    # ------------------------------------------------------------------
    # 9. integrative read-out
    # ------------------------------------------------------------------
    z = float(np.clip(beta * c * drive, -60.0, 60.0))
    p_int = 1.0 / (1.0 + np.exp(-z))

    # ------------------------------------------------------------------
    # 10. AUTHORITY-COINCIDENCE LEAD: if the topmost discriminating row is
    #     also more diagnostic than anything printed below it, its verdict
    #     dominates (near-categorically).  Still scaled by the same rarity
    #     confidence c, so a lone oracle amid a mute panel stays near chance.
    # ------------------------------------------------------------------
    j1 = int(idx[0])
    if j1 >= n - 1:
        q_com = q_max
    else:
        v_rest = float(np.max(s[j1 + 1:]))
        v_rest = max(v_rest, 1e-6)
        ratio = float(s[j1]) / (phi * v_rest)
        q_com = q_max * float(_sig(s_stop * (ratio - 1.0)))
    q_com = float(np.clip(q_com, 0.0, 1.0))

    p_dec = float(_sig(beta * c * d_com))
    p_com_a = p_dec if d[j1] > 0 else (1.0 - p_dec)

    p_a = q_com * p_com_a + (1.0 - q_com) * p_int

    # ------------------------------------------------------------------
    # 11. HARD uniform noise ceiling
    # ------------------------------------------------------------------
    p_a = 0.5 + (p_a - 0.5) * (1.0 - eps)
    p_a = float(np.clip(p_a, 1e-9, 1.0 - 1e-9))

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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    # Cheap-cue-coalition cells: the difference vectors (A - B) in which a\n    # coalition of low/mid-validity experts (positions 1,3,4 and, in the dense\n    # cell, 0 and 6) opposes the near-oracle expert at position 2 (v=.96)\n    # (and, in two cells, the .84 expert at position 5).\n    # Orientation: +1 entries mark the COALITION side.\n    pats = [\n        (0, 1, -1, 1, 0, 0, 0),    # pairs 3/4  : 2 cheap cues vs oracle\n        (0, 1, -1, 1, 1, 0, 0),    # pairs 5/6  : 3 cheap cues vs oracle\n        (0, 1, -1, 1, 1, -1, 0),   # pairs 9/10 : 3 cheap cues vs oracle + .84\n        (1, 1, -1, 1, 1, -1, 1),   # pairs 15/16: dense 5-vs-2\n    ]\n    lookup = {}\n    for p in pats:\n        lookup[p] = 1\n        lookup[tuple(-x for x in p)] = -1\n\n    hits = 0\n    tot = 0\n    a_col = data['option_a_ratings'].tolist()\n    b_col = data['option_b_ratings'].tolist()\n    r_col = data['response'].tolist()\n    for a, b, r in zip(a_col, b_col, r_col):\n        try:\n            av = [int(x) for x in a]\n            bv = [int(x) for x in b]\n        except Exception:\n            continue\n        if len(av) != 7 or len(bv) != 7:\n            continue\n        d = tuple(av[i] - bv[i] for i in range(7))\n        s = lookup.get(d)\n        if s is None:\n            continue\n        try:\n            resp = int(r)\n        except Exception:\n            continue\n        tot += 1\n        chose_a = (resp == 0)\n        if (s == 1 and chose_a) or (s == -1 and not chose_a):\n            hits += 1\n    if tot == 0:\n        return float('nan')\n    return float(hits) / float(tot)",
  "rationale": "The two theories differ mainly in the SHAPE of the validity-to-weight map and the growth law of numerosity, and this shows up as the location of the switch point on the cheap-cue-coalition ladder. I pool the four cells in which a coalition of low/mid-validity experts opposes the near-oracle (.96) expert (and, in two cells, also the .84 expert): design pairs 3/4, 5/6, 9/10 and 15/16, each of which appears mirrored across sides and 6 times, so each subject contributes 48 scored trials. The metric is the proportion of those trials on which the subject sided with the numerous cheap-cue coalition rather than with the high-validity expert. Under DCRI-3s (compressive diagnosticity, ratio ~3 between oracle and cheap cue, plus a sqrt numerosity bonus of +0.79 at |dk|=2) the coalition is followed at about chance or above in every one of these cells (nominal .51/.58/.47/.69, mean ~.56). Under GIT-AL+P/k (superlinear diagnosticity, ratio ~9, essentially negligible linear numerosity, hard eps=.32 floor) the oracle dominates throughout (nominal .28/.32/.19/.39, mean ~.30). That is a ~.27 mean gap in the same statistic, with a direction disagreement relative to chance. Positional/salience machinery is inert in all four cells (the diagnostic rows are interior, extremes are validity-matched), and the mirroring cancels any side bias, so the contrast is attributable to weighting shape alone. Averaging 48 trials per subject keeps binomial noise small (SD ~.07) so the per-subject spread should be dominated only by mild parameter jitter, giving a tight, highly separable contrast at N=25."
}
```

## Usage

```json
{
  "input_tokens": 15871,
  "output_tokens": 3647
}
```
