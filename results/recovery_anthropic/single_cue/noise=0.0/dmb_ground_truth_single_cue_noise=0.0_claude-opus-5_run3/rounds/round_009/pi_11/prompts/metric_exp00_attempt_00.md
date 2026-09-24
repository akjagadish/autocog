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
**Validities (n_features=7):** [0.72, 0.6, 0.95, 0.72, 0.62, 0.66, 0.72]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1]
  trial 3: A=[1, 1, 0, 1, 1, 1, 1]  B=[1, 0, 1, 1, 0, 0, 1]
  trial 4: A=[1, 0, 1, 1, 0, 0, 1]  B=[1, 1, 0, 1, 1, 1, 1]
  trial 5: A=[1, 1, 0, 1, 1, 1, 0]  B=[1, 0, 1, 0, 0, 0, 1]
  trial 6: A=[1, 0, 1, 0, 0, 0, 1]  B=[1, 1, 0, 1, 1, 1, 0]
  trial 7: A=[1, 1, 0, 0, 1, 1, 1]  B=[0, 0, 1, 1, 1, 1, 0]
  trial 8: A=[0, 0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 1, 1, 1]
  trial 9: A=[0, 1, 0, 1, 1, 1, 1]  B=[1, 0, 1, 0, 1, 1, 0]
  trial 10: A=[1, 0, 1, 0, 1, 1, 0]  B=[0, 1, 0, 1, 1, 1, 1]
  trial 11: A=[1, 1, 1, 0, 1, 1, 1]  B=[0, 1, 0, 1, 1, 1, 0]
  trial 12: A=[0, 1, 0, 1, 1, 1, 0]  B=[1, 1, 1, 0, 1, 1, 1]
  trial 13: A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 1, 1, 0]
  trial 14: A=[1, 1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  trial 15: A=[1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1, 1]
  trial 16: A=[1, 1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0]

**Rationale:** GOAL: dissociate GIT-AL+P/k (advocated: SUPERLINEAR diagnosticity rho~1.6, tiny LINEAR numerosity lam~0.135, literal first/last-row anchor with NO residual but a WIDENED near-deadlock gate sigma~0.60 (g=1 at dk=0, ~0.25 at |dk|=1, ~0.004 at |dk|=2), superlinear mass normalisation (k2+W)^1.17 with beta~8.4, near-categorical authority lead q_max~0.97, and a HARD difficulty-independent lapse eps~0.32 that caps every prediction inside [0.16,0.84]) from DCRI-3s (competitor: COMPRESSIVE rho~0.52 with an oracle premium above v_knee~0.91, SQRT numerosity lam~0.56, steep edge salience with a CONSTANT residual g_res~0.10 and large mu~2.40, linear relative read-out (V+N+P)/(0.42+Xs) with beta~3.1, and a DIFFICULTY-SCALED lapse running from ~0.30 on deadlocks to ~0.08 on landslides).

LAYOUT: n=7, validities by SCREEN position [0.72, 0.60, 0.95, 0.72, 0.62, 0.66, 0.72]. THREE experts share v=0.72 and sit at pos0 (top edge), pos3 (deep interior) and pos6 (bottom edge): this permits validity-matched EDGE<->INTERIOR swaps that leave the validity vote of ANY validity-only model analytically unchanged (GIT's 13% primacy credit moves V by <0.05) while flipping the anchor/salience term. The oracle (0.95, above DCRI's knee so its premium fires) sits at pos2, deep enough that both position profiles are near zero there; cheap experts (0.60, 0.62, 0.66) fill pos1/pos4/pos5. In every cell except cell 8 the first discriminating row is a 0.72 or 0.60 expert with the 0.95 expert printed below it (ratio <=0.48), so BOTH models' authority-checked commitment is suppressed (q_com<=0.01) and cannot contaminate any contrast. All mute rows are mutual ENDORSEMENTS, the case where the two silence rules differ least, so silence cannot be blamed for the direction disagreements. Derived weights: GIT w=[.304,.084,.919,.287,.106,.165,.269] (oracle/cheap ratio ~11); DCRI x=[.652,.433,1.418,.652,.476,.553,.652] (ratio ~3.3).

8 base cells, each mirrored across sides -> 16 unique pairs, K=6 -> 96 trials; mirroring forces an exact 8A/8B split for both theories.

(1) LANDSLIDE CEILING, pairs 1-2: 7-0 unanimous, NO mute rows, so both silence rules and both gates are provably inert. GIT's hard lapse caps it at P(A)=.840 no matter how lopsided; DCRI's difficulty-scaled lapse shrinks to .084 and it predicts .941. A ~10-point ceiling gap and an engagement check.

(2) CHEAP-COALITION vs ORACLE (sparse), pairs 3-4: pos1(.60), pos4(.62), pos5(.66) favour A, the .95 oracle favours B, three mute endorsements. GIT's superlinear weights make the coalition worth .356 against .919 and its linear numerosity adds only +.27: P(A)=.345. DCRI's compression makes the same coalition worth 1.462 against 1.418 and its sqrt numerosity adds +.79: P(A)=.595. A ~25-point DIRECTION disagreement straddling chance.

(3) CHEAP-COALITION (near-dense), pairs 5-6: d=[0,+1,-1,+1,+1,+1,-1], only one mute row, so silence machinery is essentially off. GIT .361 vs DCRI .580 - a second ~22-point direction disagreement that cannot be attributed to silence or position.

(4-5) FLAGSHIP ANCHOR-WIDTH SWAP AT |dk|=1, pairs 7-10. Both cells have the identical validity multisets (A side {.72,.72,.60}, B side {.95,.72}), identical counts (3-2), identical silence (two mute endorsements at pos4/pos5) and an analytically identical vote; the ONLY change is whether the second 0.72 supporter sits at the TOP EDGE (pos0, cell ON) or in the DEEP INTERIOR (pos3, cell OFF), with the corresponding dissenter moved the other way. GIT's widened near-deadlock gate licenses a quarter of full anchor force at |dk|=1 (g=.25), so its anchor term swings by mu*g*dS = 0.50 and it predicts .545 (ON) vs .308 (OFF) - a ~24-point swing that CROSSES chance, and cell OFF alone is a ~22-point DIRECTION disagreement (GIT .31 vs DCRI .53). DCRI's gate at |dk|=1 is only .146 but its salience is multiplied by mu=2.4 and then divided by 0.42+Xs=4.23 and compressed by its near-deadlock lapse, giving only .601 vs .526, ~7 points, never crossing chance.

(6-7) SAME SWAP AT |dk|=2, pairs 11-14: identical construction (A side {.72,.72,.95}, B side {.72}, three mute endorsements) but the reason margin is now 3-1. GIT's Gaussian gate is effectively SHUT (g=.004), so it is mathematically forced to a null: .825 vs .817, <1 point. DCRI's constant residual g_res=.10 survives: .772 vs .733, ~4 points. The diagnostic statistic is the INTERACTION across cells 4-7: GIT predicts the edge effect to collapse from ~24 points to ~1 point as the margin grows from 1 to 2 reasons (a factor ~30), DCRI predicts a shallow decline from ~7.5 to ~4 (a factor ~2). No parameter setting of the residual-salience model can produce that collapse, and no setting of the Gaussian-only model can produce a reliable edge effect at |dk|=2.

(8) ORACLE-FIRST COMMITMENT, pairs 15-16: pos0/pos1 mute, the .95 oracle is the FIRST discriminating row and is more diagnostic than everything below it (ratio 2.01), so both authority stops fire (q_com=.97 / .96) while four lower-validity experts dissent. Both models follow the oracle against the tally, but GIT's weaker committed read-out sigmoid(8.43c*0.50) and hard lapse cap it at .811 whereas DCRI's sigmoid(3.1c*2.25) plus a small landslide-like lapse gives .922. This cell anchors interpretation: subjects demonstrably DO follow a leading oracle, so failure to follow it in cells 2-3 cannot be attributed to inattention to the .95 row, and paired with cell 1 it isolates hard vs difficulty-scaled lapse (GIT's whole predicted range across the design is [.31,.84], DCRI's is [.53,.94]).

DIAGNOSTIC YIELD: (i) two direction disagreements driven purely by the shape of the validity-to-weight map and the growth law of numerosity (cells 2, 3; ~22-25 points each), one of them in a near-dense display; (ii) a validity-matched edge<->interior swap where the advocated theory predicts a chance-crossing 24-point effect and the competitor ~7, plus a third direction disagreement on the OFF cell; (iii) the same swap at |dk|=2 where the advocated theory is analytically pinned to a null and the competitor must retain a residual - the gate-width interaction is the flagship; (iv) two ceiling cells (1 and 8) testing the hard [0.16,0.84] cap against a difficulty-scaled lapse. Every pair has at least one discriminating cue, every contrast is within-subject, and each cell is mirrored across sides.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n\n    # Six target displays (ordered A,B rating strings), all mirrored pairs of\n    # three base cells:\n    #   cell 2 (sparse cheap-coalition vs oracle):      trials 3 / 4\n    #   cell 3 (near-dense cheap-coalition vs oracle):  trials 5 / 6\n    #   cell 5 (edge-OFF |dk|=1 swap):                  trials 9 / 10\n    sel_keys = {\n        (\"1101111\", \"1011001\"), (\"1011001\", \"1101111\"),\n        (\"1101110\", \"1010001\"), (\"1010001\", \"1101110\"),\n        (\"0101111\", \"1010110\"), (\"1010110\", \"0101111\"),\n    }\n\n    def key(seq):\n        return \"\".join([str(int(round(float(x)))) for x in list(seq)])\n\n    hits = 0\n    tot = 0\n    for a, b, r in zip(data[\"option_a_ratings\"], data[\"option_b_ratings\"], data[\"response\"]):\n        try:\n            av = [int(round(float(x))) for x in list(a)]\n            bv = [int(round(float(x))) for x in list(b)]\n        except Exception:\n            continue\n        if len(av) != len(bv):\n            continue\n        k = (key(av), key(bv))\n        if k not in sel_keys:\n            continue\n        kA = sum(1 for i in range(len(av)) if av[i] > bv[i])\n        kB = sum(1 for i in range(len(av)) if bv[i] > av[i])\n        if kA == kB:\n            continue\n        target_is_A = kA > kB\n        try:\n            resp = int(r)\n        except Exception:\n            continue\n        chose_A = (resp == 0)\n        tot += 1\n        if chose_A == target_is_A:\n            hits += 1\n\n    if tot == 0:\n        return float(\"nan\")\n    return float(hits) / float(tot)\n",
  "rationale": "The design's sharpest theory split is the *direction* of choice on displays where a coalition of several low/mid-validity experts (0.60, 0.62, 0.66, 0.72) outnumbers a single near-certain 0.95 oracle plus one helper. My advocated GIT-AL+P/k uses a SUPERLINEAR validity->weight map (rho~1.6) and only a tiny linear numerosity term (lam~0.135), so the oracle's weight (~0.92) dwarfs the whole cheap coalition (~0.36) and the count majority is followed well BELOW chance. The competing DCRI-3s uses a COMPRESSIVE map (rho~0.52) plus a sqrt numerosity bonus (lam~0.56), so the same coalition (~1.46) narrowly outweighs the premium-boosted oracle (~1.42) and the count majority is followed ABOVE chance. I restrict to the three base cells (each mirrored across sides, so side bias cancels exactly) where this contrast is cleanest and where no anchoring/commitment machinery can contaminate it: the sparse coalition cell (trials 3/4), the near-dense coalition cell (trials 5/6, only one mute row so silence rules are inert), and the edge-OFF |dk|=1 swap (trials 9/10, where the two edge rows support opposite sides so GIT's anchor term is exactly zero and only the weight map and numerosity drive the choice). I deliberately EXCLUDE the edge-ON cell, the |dk|=2 cells, the landslides and the oracle-first commitment cell, because on those the two theories agree in direction (or disagree in the opposite direction), which would dilute the contrast. The statistic is the simple proportion of choices for the count-majority side over these 36 trials per subject: GIT predicts ~0.34, DCRI ~0.57 -- a ~23-point gap that straddles chance. Because it is a plain within-subject proportion over 36 mirrored trials, its per-subject spread is dominated by binomial noise (sd~0.08) plus modest parameter jitter, so it should be tight enough to separate the theories at N=25."
}
```

## Usage

```json
{
  "input_tokens": 16310,
  "output_tokens": 7706
}
```
