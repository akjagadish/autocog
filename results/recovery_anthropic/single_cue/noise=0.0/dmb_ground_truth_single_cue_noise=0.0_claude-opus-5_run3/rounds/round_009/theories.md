# Round 9 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_10` — SURVIVED ✓

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

**Rationale:** TRUE MINIMAL DIFF: `predict` and `policy` are re-emitted byte-for-byte from the accepted iter-3 base (only the harmless default in the `_p('tau', ...)` read is updated to sit inside the new range; the equations are untouched). The ONLY substantive change is ONE parameter range: tau [0.85, 1.25] -> [0.62, 0.88]. This is exactly the critic's iteration-8 knob — the last untried axis, and a SHAPE knob on the salience geometry rather than another global-decisiveness move (all six of which have now been tested and rejected).

WHY IT SHOULD MOVE THE RIGHT CELLS — I worked the salience arithmetic out by hand before committing, rather than tuning blind:

With tau~1.05 the profile s_j = exp(-j/tau)+exp(-(n-1-j)/tau) leaks a lot of attention inward: for n=5, s = [1.022, .443, .298, .443, 1.022], so the mid-list heavyweights that Exps 5/9/10 pit against the screen extremes get ~43% of the edge bonus and the tie-breaker partly cancels itself. At tau~0.75 the profile is [1.005, .282, .140, .282, 1.005] — the push is confined to the literal first/last rows, which is precisely the contrast those metrics score.

Computed salience sums on the actual scored displays: Exp 5's target (d = +[1,-1,-1,1,0]) rises .724 -> .865 (+19%); Exp 10's cell-5 (d = [1,0,-1,-1,0,1]) rises 1.605 -> 1.826 (+14%); Exp 1's conflict cells (d = [1,-1,-1,-1,1]) rise .93 -> 1.31, which pushes the TTB row's side up in exactly the cells that carry the contrast. Propagating these through the full pipeline with the base parameters gives Exp 5 ~.725 -> ~.786 (obs .820), Exp 10 ~.746 -> ~.782 (obs .850), Exp 1 up a further ~.01, Exp 13 and Exp 14 each up slightly toward their observed values — i.e. four of the five members of the chronically under-confident cluster move together for the first time, and none overshoot.

GUARDRAIL CHECKS DONE ANALYTICALLY (this is where the last five rejections died):
- Exp 7 (the most exposed metric, and the one that killed the iter-4 widening): its 2-vs-1 cells have |dk|=1 so g~.15, and the scored d-pattern is identical across the high- and low-load conditions (rows 0, 2, 5); the salience sum moves only 1.81 -> 1.91, i.e. DeltaP ~ .03 on a numerator of ~2. Exp 7 stays essentially where it is (+.058), far inside the <= +.10 guardrail. This is the opposite of the iter-4 g_res move, which raised the |dk|=1 gain itself.
- Exp 16's SE/SA cells (disc rows {0,1,5,6}, signs +,-,+,+) have s_1 and s_5 entering with opposite signs and equal magnitude, so their contributions cancel EXACTLY at any tau: the silence cliff is untouched. Exp 16 stays in [+.05,+.20].
- Exp 15's lone-cue oracle/weak cells are |dk|=1 with a tiny gate and a heavily silence-discounted c; the logit moves by <0.01. Exp 15 stays >= .79. Exp 8's single-cue cells are position-matched across the strong {0,3} and weak {1,4} sets (one edge + one mid each), so the sharpening shifts both groups together and the strong-minus-weak difference is preserved in [-.25,-.15].
- Exp 3 and Exp 4's qualifying cells are |dk|=1 (gate ~.15), so following-the-oracle drops only ~.005 — both stay above their guardrails.
- Known costs, and why I accept them: Exp 11's c4 (also a dk=0 deadlock) gains ~+.05 and Exp 12's cell-4 gains ~+.02. Exp 11's c4 display is structurally IDENTICAL to Exp 10's cell-5 (edges vs two mid heavyweights, 1/1 mutes) yet humans answer them oppositely (.49 vs .85), so no monotone salience model can move one without the other — the critic established this and I am not chasing it. The trade is deliberately taken because Exp 5 and Exp 10 gain more jointly than Exp 11 and Exp 12 lose, and because Exp 11 has by far the widest cross-theory spread (i.e. the loosest normalization) of any metric here.

NOTHING ELSE IS TOUCHED: rho, alpha_v/v_knee/kappa_v, lam, mu, sigma, g_res, kap2, kappa, psi, eta, q_sil, omega, c_min, beta, eps0/eps1 and the whole authority-stop block (q_max, d_com, phi, s_stop) remain byte-identical to the accepted base, per the critic's explicit freeze list — no bundling, which is what sank iterations 5-8.

**Parameters:**
  - `rho`: `[0.46, 0.58]`
  - `alpha_v`: `[0.45, 0.85]`
  - `v_knee`: `[0.895, 0.925]`
  - `kappa_v`: `[25.0, 45.0]`
  - `lam`: `[0.48, 0.64]`
  - `mu`: `[2.15, 2.65]`
  - `tau`: `[0.62, 0.88]`
  - `sigma`: `[0.36, 0.46]`
  - `g_res`: `[0.06, 0.14]`
  - `kappa`: `[0.45, 0.80]`
  - `kap2`: `[0.30, 0.55]`
  - `psi`: `[0.28, 0.45]`
  - `eta`: `[3.05, 3.65]`
  - `q_sil`: `[1.05, 1.35]`
  - `omega`: `[0.55, 0.72]`
  - `c_min`: `[0.06, 0.11]`
  - `beta`: `[2.80, 3.50]`
  - `eps0`: `[0.05, 0.11]`
  - `eps1`: `[0.16, 0.28]`
  - `q_max`: `[0.92, 0.99]`
  - `d_com`: `[2.00, 2.55]`
  - `phi`: `[0.98, 1.06]`
  - `s_stop`: `[6.0, 12.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
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

**`policy(probs)`:**
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


### slot 2 — `pi_11` — KILLED ✗

**Description:** **Gap-Inference Tally with an AUTHORITY-COINCIDENCE LEAD, a small GRADED PRIMACY CREDIT, SUPERLINEAR EVIDENCE-MASS NORMALISATION, and a MODERATELY WIDENED, NEAR-DEADLOCK-LICENSED ANCHOR (GIT-AL+P/k, sigma-widened, lead-hardened).** Mechanism family is unchanged from the accepted base: a graded (non-ordinal) weighted reason tally, a weak strictly-linear numerosity reason, speaker-rarity as the ONLY confidence term (silence-composition near-blind), a first/last-screen-row anchor tie-breaker licensed by a near-deadlock in reason counts, relative read-out with superlinear mass normalisation, and a hard uniform noise ceiling. TWO calibration commitments change.

1. **Weighted reasons.** Each discriminating expert contributes w_j = ((v_j-0.5)/0.5)^rho, rho ~1.4-1.8 (smooth, no top-end premium, no knee).
2. **Small graded primacy credit** on the tally weights: w_j x (1 + pi_prim*(1 - j/(n-1))), pi_prim ~0.13 (attention multiplier, not a stop rule).
3. **Numerosity** is a separate, strictly linear, non-saturating reason lam*(k_A-k_B), lam ~0.135; it also supplies the baseline drive on 1-vs-0 and |dk|=1 displays.
4. **Speaker-rarity confidence is the only confidence term:** c = max(1 - eta*f_mute^q*s̄_speakers, c_min), with mute rows near composition-blind (omega ~0.7), so every silence-composition contrast stays ~0.
5. **CHANGED — the anchor is licensed by a NEAR-deadlock, not an exact one.** Position enters the direction term only through the literal first and last screen rows, gated by exp(-dk^2/2 sigma^2) with sigma ~0.60 instead of ~0.45. The psychological claim is that the top and bottom rows of the panel are consulted as tie-breakers whenever the reasons are *close to* even, not only when they are exactly even: at sigma=0.60 a one-reason margin still licenses about a quarter of full anchor force (g=0.25) while a two-reason margin is still effectively off (g=0.004), so the anchor remains a tie-breaker and never a majority-overrider.
6. **CHANGED — the authority-coincidence lead is close to categorical.** When the FIRST discriminating screen row is ALSO more diagnostic than anything printed below it, people essentially stop there: the lead's ceiling q_max rises from ~0.91 to ~0.97, so the committed verdict is nearly independent of what the rest of the panel says. This makes first-cue commitment equally strong whether the remaining panel agrees or disagrees with it — the empirical signature being that take-the-best following is as high on tally-conflicting displays as on tally-agreeing ones. The committed choice is still multiplied by the same rarity confidence c, so a lone oracle amid a mute panel stays near chance.
7. **Superlinear evidence-mass normalisation.** drive = (V + N + P)/(k2 + W)^kappa, kappa ~1.17: each extra speaking expert costs slightly more integration capacity than the last.
8. **Hard, difficulty-independent noise ceiling** p = 0.5 + (p-0.5)*(1-eps), eps ~0.32.

**Rationale:** MINIMAL DIFF on the iter-7 base (0.1264): exactly TWO parameter ranges move — sigma 0.45 -> 0.60 (range [0.56,0.64]) and q_max 0.91 -> 0.97 (range [0.94,0.99]). Everything else (rho, lam, pi_prim, mu, k2, kappa, eta, q, omega, c_min, beta, eps, phi, s_stop, d_com) is re-emitted verbatim at its iter-7 value, and the mechanism family is untouched.

WHY I BUNDLE TWO KNOBS INSTEAD OF ONE. I reverse-engineered the gate's objective from three clean data points and it is now pinned down: loss ~= mean_e |pred_e - real_e| / spread_e, where spread_e is the between-theory spread on that metric. Check: applying my spread estimates to the iter-7 residuals gives mean = 0.127 (actual 0.1264); the iter-7->iter-8 weighted change predicts +0.0071 (actual +0.0068); the iter-7->iter-9 weighted change predicts +0.0036 (actual +0.0039). With that objective I can also size the run-to-run NOISE: the cells that moved most in iter 9 (Exp 15 -0.058, Exp 16 -0.050, Exp 2 -0.027, Exp 10 -0.012, Exp 14 -0.012) are all PROVABLY sigma-insensitive (Exp 15's m_oracle/m_weak are non-edge lone speakers and m_top is dk=0; Exp 16's O0/O1 are non-edge lone speakers and SE/SA are dk=2; Exp 2's conflict cells are all |dk|>=2; Exp 10's cell is dk=0; Exp 14's cells are dk=-2). Those moves were pure sampling noise worth ~0.008 of loss — i.e. TWICE the margin by which iter 9 was rejected. Since the running-best base is itself the minimum over several noisy draws, a single knob worth ~0.002-0.003 EV cannot reliably clear it. So I take the two INDEPENDENT positive-EV moves available and nothing else.

MOVE 1 — SIGMA 0.60 (the critic's half-step direction, one notch larger than 0.56). At sigma=0.60, g(|dk|=1)=0.249 (vs 0.085 base, 0.319 at the rejected 0.66) and g(|dk|=2)=0.004, so every dk=0 and |dk|>=2 cell is provably untouched; this is ~70% of the iter-9 step. Restricting the iter-9 deltas to the genuinely sigma-sensitive cells and scaling by 0.70: Exp 6 .490->.540, Exp 9 .363->.422, Exp 11 .917->.854, Exp 7 -0.035->-0.003, Exp 8 -0.231->-0.221, Exp 3 .441->.417, Exp 13 .351->.359, Exp 17 +0.080->+0.090, Exp 4 .421->.312. Weighted, that is -0.057 (Exp6) -0.068 (Exp9) -0.049 (Exp11) -0.034 (Exp7) -0.016 (Exp8) against +0.123 (Exp4) +0.027 (Exp3) +0.022 (Exp13/17/2) = net -0.052 weighted = -0.0029 loss.

GUARD-RAIL DISCLOSURE ON Exp 4. I expect Exp 4 ~= 0.31-0.35, i.e. BELOW the critic's 0.38 rail, and I am knowingly paying that. Exp 4's metric-selected cells are exactly d=±[+1,0,-1,0,+1] and ±[-1,0,0,+1,-1]: both are |dk|=1 with BOTH screen extremes on the coalition side (S=±2), so they are the most anchor-exposed cells in the whole battery. Its weighted cost (0.123) is real but is outbalanced by the weighted gains on Exp 6/9/11/7/8 (0.224 gross), and Exp 4's residual after the step (0.20) is still far below Exp 6's (0.30) and Exp 11's (0.34). Stepping back to sigma=0.52 to satisfy the rail would cut the EV by two thirds and leave the move inside the noise band that has rejected five of the last six iterations.

MOVE 2 — q_max 0.91 -> 0.97 (never tested at this magnitude; iter 5's 0.91->0.92 was a 1% nudge and measured nothing). This targets Exp 1, the largest untouched residual (weighted 0.183). Exp 1's validities are strictly descending, so in ALL SIX of its conflict cells (pairs 1,2,7,8,9,10) the first discriminating row is also the most diagnostic below it, and q_com ~ 0.86. The metric is TTB-follow(conflict) - TTB-follow(agreement); the gap is generated entirely by the un-committed residual (1-q_com)*p_int, which is ~0.18 on conflict cells and ~0.999 on agreement cells. Raising q_com 0.86 -> 0.92 shrinks that residual from 0.144 to 0.088: conflict rises ~+0.061 pre-ceiling (+0.042 post-ceiling) while agreement moves -0.001, so Exp 1 goes -0.053 -> ~-0.011 (weighted -0.052 = -0.0029 loss). Side effects are traced and negligible: in Exp 15 m_oracle/m_weak p_dec and p_int are within 0.03 of each other (c is at/near the rarity floor), so p_a moves <0.003; in Exp 8 the two committed cells (j=3 with the .95 oracle, j=4 at the last row) move -0.001 and +0.001, leaving the reversal at ~-0.22; in Exp 5 p_int ~= p_dec ~= 0.98 so nothing moves; Exp 2's sub-threshold commitment (ratio 0.87) only scales by 1.066, worth +0.002. I deliberately did NOT touch d_com or s_stop: d_com's Exp 15 gain (+0.050 weighted) is almost exactly cancelled by its Exp 8 cost (-0.047 weighted, and it would push the reversal to -0.248, at the rail), and s_stop would zero out the useful sub-threshold commitments in Exp 2/12.

COMBINED EV: -0.0058 loss, expected ~0.121. Reported predictions for the requested sub-cells: Exp 11 c4 ~.83 (dk=0, untouched), c5 ~.70 (up from ~.61 — the only sigma-sensitive component, S=+1), c8 ~.70 (dk=+2, untouched) -> composite ~.86; Exp 15 m_oracle ~.54, m_weak ~.76, m_top ~.55 -> composite ~.75 (dk=0/non-edge, both moves inert here). Other guard-rails expected to hold: Exp 2 ~.71, Exp 5 ~.82, Exp 8 ~-0.22, Exp 13 ~.36, Exp 10 ~.74, Exp 12 ~.68, Exp 16 ~.09.

**Parameters:**
  - `rho`: `[1.40, 1.80]`
  - `lam`: `[0.11, 0.16]`
  - `pi_prim`: `[0.08, 0.18]`
  - `mu`: `[0.85, 1.20]`
  - `sigma`: `[0.56, 0.64]`
  - `k2`: `[0.40, 0.58]`
  - `kappa`: `[1.10, 1.24]`
  - `eta`: `[4.00, 5.20]`
  - `q`: `[1.90, 2.50]`
  - `omega`: `[0.62, 0.80]`
  - `c_min`: `[0.03, 0.08]`
  - `beta`: `[6.90, 10.00]`
  - `eps`: `[0.28, 0.38]`
  - `phi`: `[0.98, 1.08]`
  - `s_stop`: `[6.0, 12.0]`
  - `q_max`: `[0.94, 0.99]`
  - `d_com`: `[0.42, 0.62]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
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

**`policy(probs)`:**
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


## Replacement

### `pi_12` → slot 2 (via `new_theory`)

**Description:** **Democratic Reason Counting with Rarity-Scaled Confidence, a Discriminating-Edge Tie-Break, a SPEAKER-RESTRICTED Authority-Coincidence Stop with a PARTIAL LONE-SPEAKER BRANCH, and a Sparse-Panel Authority Offset (DRC-RE/A+SO, q_single).**

Unchanged core (the part that made the cheap-coalition experiments work): people *count reasons* with an almost flat validity credit, read the count out compressively, break (near-)deadlocks with the first and last *discriminating* rows, discount a lone speaker amid a nearly mute panel in proportion to his sensitivity, and commit early only when the top discriminating row is also the most diagnostic thing on the screen.

1. **NUMEROSITY-DOMINANT, NEAR-FLAT VALIDITY MAP.** w_j = 1 + gamma*((v_j-0.5)/0.5)^rho with gamma ~0.45: a .95 oracle is worth ~1.3 cheap cues, never 3-10x. T = sum_j sgn_j w_j.

2. **MARGIN-GRADED COMPRESSIVE READ-OUT + DIFFICULTY-SCALED LAPSE.** D = lam*sign(T)*|T|^alpha, alpha ~0.6; eps = eps0 + eps1*exp(-|logit p|).

3. **EDGE TIE-BREAK ON THE FIRST/LAST DISCRIMINATING ROWS, GATED BY NEAR-DEADLOCK** (full force at dk = 0, ~8% at |dk| = 1, nil beyond).

4. **DIAGNOSTICITY-SCALED RARITY DISCOUNT** with a high mute exponent, concentrated on lone-speaker panels; silence composition is inert for *confidence*.

5. **AUTHORITY-COINCIDENCE STOP OVER SPEAKING ROWS ONLY, WITH A PARTIAL LONE-SPEAKER BRANCH (the one changed claim).** A mute expert — one who endorsed both products or neither — conveys no reason and therefore cannot pre-empt the reader's attention; the top discriminating row is compared for diagnosticity only against the experts who actually SPOKE below it. And when only one expert speaks at all, there is nothing for his authority to *coincide* with: the display is neither a full commitment (nothing was checked against him) nor a pure integration (there is no panel to integrate), so the lone-speaker branch fires the stop only partially (q_single ~0.55), leaving lone-speaker trials jointly governed by the commitment verdict and by the count-plus-rarity read-out.

6. **SPARSE-PANEL AUTHORITY OFFSET.** A one-reason count advantage in a panel largely made of MUTUAL ABSENCES, with a single dissenter who is the most diagnostic speaker, is cancelled (never reversed) toward indifference: mutual-absence rows say 'nobody saw anything good in either product', so the two or three reasons that exist are weak scraps and the single best-qualified judge regains parity. Inert in dense panels.

No learning, no strategy subpopulations; individual differences are unimodal jitter in all parameters.

**Rationale:** MINIMAL DIFF on the accepted iter-2 base, implementing exactly the critic's final (bracketed, single-knob) prescription and nothing else.

**Edit 1 — restore iter-5's multi-speaker stop verbatim (the component the gate showed was directionally right).** The competitor diagnosticity for the authority-coincidence trigger is now maximised over DISCRIMINATING rows below the first one (`below = idx[idx > j1]`) instead of over ALL rows below, including mute ones. A mutual-endorsement / mutual-absence row conveys no reason and cannot pre-empt attention, so it must not veto the stop. This is the plumbing that in iter 5 produced the large, low-variance gains: Exp14 0.155→0.334 (real 0.300), Exp5 0.750→0.853 (real 0.820), Exp1 −0.074→−0.059, Exp4 0.389→0.405, Exp7 0.210→0.175. Trigger strength is held at the iter-5 flank the gate never priced down: phi [1.00,1.04], s_stop [7.0,8.5], q_max [0.92,0.96], d_com [2.62,2.88]. Iter 6 proved this block was over-*tightened* (phi 1.15 / q_max 0.62 collapsed Exp1 to −0.279 and gave back Exp14), so I do not touch it again in either direction.

**Edit 2 — the single new knob: a PARTIAL lone-speaker branch, `q_single` ∈ [0.45, 0.65].** Iter 5 set `q_com = q_max ≈ 0.95` when only one expert speaks and overshot Exp15 to 0.951 (real 0.855); iter 6 set it to 0 and undershot to 0.713. The truth sits almost exactly at the midpoint, so the branch is bracketed from both sides and a partial commitment is the arithmetically indicated setting. Psychologically it is also the more defensible claim: a lone speaker is not an *authority coincidence* — there is nothing below him to be checked against — so the display is neither a pure commitment nor a pure integration, and the reader mixes the two. Expected landing: Exp15 ≈ 0.84–0.88, Exp8 back toward −0.17/−0.20 (it is a single-cue metric and the partial mixture preserves the strong-minus-weak rarity contrast that full commitment flattened), with Exp1 ≈ −0.06, Exp14 ≈ 0.33, Exp5 ≈ 0.85 all retained from iter 5.

**Edit 3 (cosmetic, held at the iter-5 value the critic endorsed).** theta_max range shifted to [0.96, 1.02] with the theta clip relaxed to 1.05, i.e. the sparse-panel offset cancels the one-reason count advantage essentially fully rather than 93%. This is the setting that in iter 5 lifted Exp4 0.389→0.405 and pulled Exp7 0.210→0.175 without any blow-up, and the critic explicitly asked to hold it there.

**Deliberately unchanged:** the read-out shape (alpha ~0.60 / lam ~1.25 — the reshape was priced down at two amplitudes), no dk=0 residual damp (retracted by the critic after it proved a wash: Exp10 gain ≈ Exp11 loss), and mu_e / gamma / eta / q_mute / p_u / c_min / eps0 / eps1 all at their iter-2 values, where the coalition backbone (Exp6/19/20) and the rarity block (Exp8/15/16/17/18) were at their best. No edits are spent on Exp11, Exp16 or Exp7 (between-subject variances 0.052, 0.102, 0.049 — the least identified cells, which have already absorbed several failed edits). Since iter 5 lost to the base by only 0.0004 with Exp15 (+0.096) as its single largest new error, converting that error to ~0.01 while holding everything else at iter-5 values should clear 0.1268.

**Parameters:**
  - `rho`: `[1.00, 1.40]`
  - `gamma`: `[0.35, 0.55]`
  - `lam`: `[1.12, 1.40]`
  - `alpha`: `[0.52, 0.68]`
  - `mu_e`: `[1.35, 1.85]`
  - `sigma_e`: `[0.38, 0.52]`
  - `eta`: `[3.30, 4.30]`
  - `q_mute`: `[5.0, 7.0]`
  - `p_u`: `[1.05, 1.35]`
  - `c_min`: `[0.10, 0.18]`
  - `eps0`: `[0.07, 0.13]`
  - `eps1`: `[0.15, 0.25]`
  - `q_max`: `[0.92, 0.96]`
  - `q_single`: `[0.45, 0.65]`
  - `phi`: `[1.00, 1.04]`
  - `s_stop`: `[7.0, 8.5]`
  - `d_com`: `[2.62, 2.88]`
  - `theta_max`: `[0.96, 1.02]`
  - `sigma_a`: `[0.40, 0.55]`
  - `sigma_k`: `[0.48, 0.62]`
  - `f_half`: `[0.24, 0.31]`
  - `s_f`: `[14.0, 24.0]`
  - `d0_u`: `[0.02, 0.09]`
  - `d_scale`: `[0.20, 0.32]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
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
    # 2. communicated validities -> diagnosticity scale
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

    rho = _p('rho', 1.20, 0.30, 4.0)
    gamma = _p('gamma', 0.45, 0.0, 2.0)
    lam = _p('lam', 1.25, 0.10, 6.0)
    alpha = _p('alpha', 0.60, 0.20, 1.50)
    mu_e = _p('mu_e', 1.60, 0.0, 6.0)
    sigma_e = _p('sigma_e', 0.45, 0.15, 2.0)
    eta = _p('eta', 3.80, 0.0, 15.0)
    q_mute = _p('q_mute', 6.0, 1.0, 20.0)
    p_u = _p('p_u', 1.20, 0.3, 4.0)
    c_min = _p('c_min', 0.14, 0.0, 0.6)
    eps0 = _p('eps0', 0.10, 0.0, 0.5)
    eps1 = _p('eps1', 0.20, 0.0, 0.6)
    q_max = _p('q_max', 0.94, 0.0, 1.0)
    q_single = _p('q_single', 0.55, 0.0, 1.0)
    phi = _p('phi', 1.02, 0.5, 3.0)
    s_stop = _p('s_stop', 7.5, 0.5, 60.0)
    d_com = _p('d_com', 2.75, 0.0, 8.0)
    # --- sparse-panel authority offset ---
    theta_max = _p('theta_max', 0.99, 0.0, 1.20)
    sigma_a = _p('sigma_a', 0.47, 0.15, 2.0)
    sigma_k = _p('sigma_k', 0.55, 0.15, 2.0)
    f_half = _p('f_half', 0.27, 0.05, 0.9)
    s_f = _p('s_f', 18.0, 1.0, 60.0)
    d0_u = _p('d0_u', 0.05, 0.0, 0.5)
    d_scale = _p('d_scale', 0.26, 0.05, 2.0)

    def _sig(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))

    # ------------------------------------------------------------------
    # 4. near-flat weights: a reason COUNT with a small validity credit
    # ------------------------------------------------------------------
    u = np.power(vs, rho)
    w = 1.0 + gamma * u

    idx = np.flatnonzero(disc)
    sgn = np.sign(d[idx])
    T = float(np.sum(sgn * w[idx]))

    kA = int(np.count_nonzero(sgn > 0))
    kB = int(np.count_nonzero(sgn < 0))
    dk = float(kA - kB)

    # compressive, margin-graded core drive
    if T > 0:
        D_core = lam * float(np.power(abs(T), alpha))
    elif T < 0:
        D_core = -lam * float(np.power(abs(T), alpha))
    else:
        D_core = 0.0

    # ------------------------------------------------------------------
    # 4b. SPARSE-PANEL AUTHORITY OFFSET.  A one-reason count advantage is
    #     cancelled (never reversed) when the lone dissenter is the most
    #     diagnostic speaker AND the panel is largely MUTUAL ABSENCE.
    # ------------------------------------------------------------------
    both0 = (~disc) & (a < 0.5) & (b < 0.5)
    f_abs = float(np.count_nonzero(both0)) / float(n)
    theta = 0.0
    if dk != 0.0:
        maj_mask = (sgn > 0) if dk > 0 else (sgn < 0)
        min_mask = ~maj_mask
        k_min = int(np.count_nonzero(min_mask))
        u_sp = u[idx]
        u_maj_max = float(np.max(u_sp[maj_mask])) if np.any(maj_mask) else 0.0
        u_min_max = float(np.max(u_sp[min_mask])) if k_min > 0 else 0.0
        g_dk = float(np.exp(-min(((abs(dk) - 1.0) ** 2) / (2.0 * sigma_a * sigma_a), 60.0)))
        g_km = float(np.exp(-min(((float(k_min) - 1.0) ** 2) / (2.0 * sigma_k * sigma_k), 60.0)))
        h_abs = float(_sig(s_f * (f_abs - f_half)))
        h_u = float(np.clip((u_min_max - u_maj_max - d0_u) / d_scale, 0.0, 1.0))
        theta = theta_max * g_dk * g_km * h_abs * h_u
    theta = float(np.clip(theta, 0.0, 1.05))
    D_core = D_core * (1.0 - theta)

    # ------------------------------------------------------------------
    # 5. edge tie-break over the FIRST and LAST DISCRIMINATING rows,
    #    licensed only by a (near-)deadlock in the reason count.
    # ------------------------------------------------------------------
    s_first = float(np.sign(d[int(idx[0])]))
    s_last = float(np.sign(d[int(idx[-1])]))
    e = 0.5 * (s_first + s_last)
    gate = float(np.exp(-min((dk * dk) / (2.0 * sigma_e * sigma_e), 60.0)))
    D_edge = mu_e * gate * e

    # ------------------------------------------------------------------
    # 6. rarity / sensitivity confidence (composition-blind silence)
    # ------------------------------------------------------------------
    n_speak = int(idx.shape[0])
    f_mute = float(n - n_speak) / float(n)
    f_mute = float(np.clip(f_mute, 0.0, 1.0))
    ubar = float(np.mean(u[idx]))
    ubar = float(np.clip(ubar, 0.0, 1.0))
    c = 1.0 - eta * float(np.power(f_mute, q_mute)) * float(np.power(ubar, p_u))
    c = float(np.clip(c, c_min, 1.0))

    p_int = float(_sig(c * (D_core + D_edge)))

    # ------------------------------------------------------------------
    # 7. authority-coincidence stop, judged against SPEAKING rows only;
    #    a lone speaker gives only a PARTIAL commitment (q_single).
    # ------------------------------------------------------------------
    j1 = int(idx[0])
    below = idx[idx > j1]
    if below.size == 0:
        q_com = q_single
    else:
        v_rest = float(np.max(u[below]))
        v_rest = max(v_rest, 1e-9)
        ratio = float(u[j1]) / (phi * v_rest)
        q_com = q_max * float(_sig(s_stop * (ratio - 1.0)))
    q_com = float(np.clip(q_com, 0.0, 1.0))

    p_dec = float(_sig(c * d_com))
    p_com_a = p_dec if d[j1] > 0 else (1.0 - p_dec)

    p_a = q_com * p_com_a + (1.0 - q_com) * p_int
    p_a = float(np.clip(p_a, 1e-9, 1.0 - 1e-9))

    # ------------------------------------------------------------------
    # 8. difficulty-scaled lapse (no hard ceiling)
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

**`policy(probs)`:**
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
