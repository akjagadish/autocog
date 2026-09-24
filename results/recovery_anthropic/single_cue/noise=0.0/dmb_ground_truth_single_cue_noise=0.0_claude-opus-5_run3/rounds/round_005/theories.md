# Round 5 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_6` — KILLED ✗

**Description:** **Read-until-sufficient integration with mildly compressed diagnosticity, silence-calibrated evidence, and an S-SHAPED (convex) read-out of relative evidence.**

People do not run a fixed heuristic (TTB, tally, or weighted-additive). They read the expert ratings in the order the screen presents them, accumulate weighted differences, and stop as soon as their running lead is safe from any single expert they have not yet read. Six claims:

1. **Cue weights = MILDLY COMPRESSED stated diagnosticity x a modest reading-position bonus.** w_j = ((v_j-0.5)/0.5)^rho * (1 + alpha*exp(-j/tau) + gamma*exp(-(n-1-j)/tau)), with rho ~1.0-1.3, i.e. subjective weight is essentially linear-to-slightly-convex in (v-0.5). A .95 expert is only ~2x a .72 expert and ~5-8x a .56 expert -- clearly heavier, and roughly *balanced* by a pair of upper-middling experts (.78 + .72), but routinely outvotable by three mid-validity experts ('tally-like without unit-weight tallying'). This exponent sits at the crossover point where the two screen-extreme flankers of a mid-list .95 expert neither dominate it nor are dominated by it, which is what turns double-dissociation displays into genuine coin flips. Position is an attention multiplier, never the causal driver.

2. **Serial reading with a graded sufficiency stop.** After each rating that actually discriminates, the reader asks whether any single unread expert could still overturn the current lead; the bar is theta = phi * max weight among unread positions and stopping is graded, q = sigmoid(s*(|E|/theta - 1)). This is the only place where screen order has structural force, and it is display-contingent. Because the bar is set by the strongest unread expert, a merely salient first cue stops the read only when its weighted lead genuinely exceeds what any single remaining expert could reverse -- and as diagnosticity compression relaxes (higher rho) a salient-but-mediocre first cue loses that race to an unread .95 expert.

3. **Evidence is read FULLY relatively, with no load machinery.** R = sum(w*d)/sum(w*|d|) over the cues read so far; features on which both products are endorsed (or both unendorsed) are simply dropped, so any dense-vs-sparse contrast is exactly zero.

4. **Silence calibration.** Validity is treated as sensitivity: a difference only a top expert notices while the rest of the panel is silent implies a marginal quality gap and is acted on hesitantly. Gain is multiplied by c = 1 - eta*(fraction silent)^q * (weighted mean diagnosticity of the speaking experts), which bites only when silence is overwhelming, producing a *reversed* single-cue validity ladder. This term is a function of raw diagnosticity, not of the compressed weights, so the ladder is invariant to rho.

5. **Decisiveness is a CONVEX (S-shaped) function of the evidence balance, not a linear one.** A near-balanced set of reasons does not merely produce a slightly tilted preference; it produces genuine indifference. Read-out is logistic(beta * c * sign(R)*|R|^kappa) with kappa ~1.4-1.9. A small residual imbalance (|R| ~ 0.1-0.2, e.g. two screen-extreme cues almost exactly offsetting one top-validity cue) is discounted to near-nothing and the choice is a coin flip; a clear weighted majority (|R| ~ 0.5) is still followed at ~0.78-0.82; and a categorical conclusion (|R| = 1, i.e. a sufficiency stop, or a single discriminating cue) is unchanged, so committed reads stay decisive.

6. **Choice** = logistic(beta * c * sign(R)|R|^kappa) with lapse epsilon. Individual differences are unimodal jitter in rho, alpha, phi, s, kappa, beta, eta and lapse -- no strategy subpopulations, no primacy/recency reader types, no learning (there is no feedback).

**Rationale:** **Minimal diff.** The iter-5 base (loop best, 0.0929) is re-emitted BYTE-IDENTICAL except for two parameter-range lines: rho [0.85,1.20] -> [1.00,1.32] (the critic's requested small upward shift of the compression exponent, same range WIDTH as before so Exp4's between-subject variance cannot inflate) and beta [2.6,3.3] -> [2.8,3.5] (the critic's own explicitly sanctioned guardrail #1, applied pre-emptively rather than after the fact). Nothing structural moves: the attended-unit sufficiency check, the fully-relative R, the zero-load commitment, kappa, alpha, gamma, tau, phi, s, eta and sil_pow are untouched. I did not repeat the validity-unit stop (iter 6, rejected), the alpha trim (iter 3, rejected) or any phi raise.

**Why rho, with the exact arithmetic.** I worked out the Exp4 cells the metric actually selects. They are (i) A=[1,0,0,0,1] vs B=[0,0,1,0,0] and (ii) A=[0,0,0,1,0] vs B=[1,0,0,0,1], with VAL=[.78,.60,.95,.88,.72]. Cell (i) at the current mean rho=1.02: w0 = 0.56*1.95 = 1.092, bar = phi*w2 = 1.05*(0.90*1.203) = 1.137, so |E|/bar = 0.96 and q = sigmoid(10*(-0.04)) = 0.40 — the reader stops on the salient first cue 40% of the time with a categorical |R|=1, and the exhausted branch gives only 0.547; total flanker rate 0.71, i.e. TTB rate 0.29, exactly the observed 0.303. At mean rho = 1.16 the same arithmetic gives w0 = 1.001 vs bar = 1.119, so |E|/bar = 0.895 and q falls to 0.26, while the exhausted-read |R| falls 0.178 -> 0.138 and kappa crushes it further (p 0.547 -> 0.531): TTB rate rises to ~0.36. The identical calculation on cell (ii) gives 0.254 -> 0.313. So rho attacks the loop's single largest residual (Exp4, -0.205) through the stop probability WITHOUT globally suppressing stopping — the move that the gate killed three times.

**Why I am also pre-applying the beta guardrail.** I reverse-engineered the loop's scoring: loss is very close to an RMS over per-experiment residuals (loss^2 vs mean delta^2 tracks within ~10% across all six iterations), with Exp4 apparently somewhat down-weighted and Exp5/Exp6/Exp7 somewhat up-weighted. Under that metric the rho move alone is a net win only if the Exp5/Exp6 cost stays near the iter2->iter4 measured rate (-0.031/-0.050 per +0.20 of mean rho); if kappa amplifies that cost (plausible, since iter 6 showed Exp5's exhausted reads collapse to ~0.53 once stops vanish), the edit lands at ~0.094 and the gate rejects. A modest beta lift is the cheap insurance: every residual except Exp8 is NEGATIVE, so extra gain moves Exp5, Exp6 and Exp2 the right way, while it barely touches Exp3/Exp4 (their exhausted-read drive is already crushed by kappa to |drive| ~0.04, so beta 2.95 -> 3.15 shifts them by <0.006).

**Exp8 and Exp7 are provably safe.** Exp8's metric uses only single-discriminating-cue trials, where |R| = 1 exactly, so the cue weights (and therefore rho) cannot enter at all — the ladder is a pure function of c = 1 - eta*(0.8^q)*vs and beta. I checked beta arithmetically: at beta 2.95 the strong-minus-weak gap is (0.582+0.841)/2 - (0.905+0.931)/2 = -0.207; at beta 3.15 it is (0.588+0.856)/2 - (0.917+0.941)/2 = -0.207. Invariant, so the pool's only correctly-signed reversed ladder is preserved. Exp7 stays ~0 because there is still no load machinery anywhere in the code.

**Expected ledger:** Exp4 +0.05 to +0.07, Exp3 +0.04, Exp1 +0.013, Exp5 -0.021 then +0.014 back from beta, Exp6 -0.034 then +0.018 back, Exp2 ~-0.010, Exp7/Exp8 unchanged. Weighted RMS lands near 0.080-0.085, and even under a pessimistic doubling of the Exp5/Exp6 rho cost the beta offset keeps it at ~0.089, below the 0.0929 floor.

**Parameters:**
  - `rho`: `[1.00, 1.32]`
  - `alpha`: `[0.7, 1.2]`
  - `gamma`: `[0.0, 0.15]`
  - `tau`: `[1.0, 1.6]`
  - `phi`: `[0.95, 1.20]`
  - `s`: `[7.0, 14.0]`
  - `beta`: `[2.80, 3.50]`
  - `kappa`: `[1.35, 1.85]`
  - `eta`: `[1.5, 2.1]`
  - `sil_pow`: `[2.4, 3.0]`
  - `epsilon`: `[0.0, 0.06]`
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
        if arr.ndim == 1:
            arr = arr.reshape(2, -1)
        else:
            arr = arr.reshape(2, -1)
        if arr.shape[0] != 2:
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
    # 2. communicated validities -> diagnosticity
    # ------------------------------------------------------------------
    val = parameters.get('validities', None)
    try:
        val = np.asarray(val, dtype=float).ravel()
    except Exception:
        val = None
    if val is None or val.shape[0] != n:
        val = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.90])
    val = np.clip(val, 0.5 + 1e-6, 1.0 - 1e-9)
    vs = np.clip((val - 0.5) / 0.5, 1e-9, 1.0)      # scaled diagnosticity in (0,1]

    # ------------------------------------------------------------------
    # 3. parameters
    # ------------------------------------------------------------------
    rho = float(np.clip(parameters.get('rho', 1.15), 0.1, 6.0))
    alpha = float(np.clip(parameters.get('alpha', 0.9), 0.0, 4.0))
    gamma = float(np.clip(parameters.get('gamma', 0.05), 0.0, 3.0))
    tau = float(np.clip(parameters.get('tau', 1.3), 0.2, 8.0))
    phi = float(np.clip(parameters.get('phi', 1.05), 0.3, 3.0))
    s = float(np.clip(parameters.get('s', 10.0), 0.5, 80.0))
    beta = float(np.clip(parameters.get('beta', 3.15), 0.05, 40.0))
    kappa = float(np.clip(parameters.get('kappa', 1.6), 0.5, 5.0))
    eta = float(np.clip(parameters.get('eta', 1.8), 0.0, 5.0))
    q_sil = float(np.clip(parameters.get('sil_pow', 2.7), 1.0, 8.0))
    eps = float(np.clip(parameters.get('epsilon', 0.03), 0.0, 0.5))

    # ------------------------------------------------------------------
    # 4. cue weights: compressed diagnosticity x modest reading-position
    #    attention.  rho ~1.0-1.3 places a .95 cue at roughly the combined
    #    strength of two upper-middling cues, so screen-extreme flankers
    #    neither dominate nor are dominated by a mid-list top expert.
    # ------------------------------------------------------------------
    pos = np.arange(n, dtype=float)
    att = 1.0 + alpha * np.exp(-pos / tau) + gamma * np.exp(-(n - 1.0 - pos) / tau)
    w = np.power(vs, rho) * att
    w = np.clip(w, 1e-12, None)

    # ------------------------------------------------------------------
    # 5. silence calibration: validity = sensitivity.  A difference that
    #    only the most sensitive experts can see, while the rest of the
    #    panel sees none, implies a marginal quality gap -> discounted.
    #    A difference an insensitive expert notices implies a large gap.
    # ------------------------------------------------------------------
    m = int(np.count_nonzero(disc))
    f_sil = float(n - m) / float(n)
    wd = w[disc]
    tot_wd = float(np.sum(wd))
    if tot_wd <= 0:
        return np.array([0.5, 0.5])
    vbar = float(np.sum(wd * vs[disc]) / tot_wd)
    c = 1.0 - eta * (f_sil ** q_sil) * vbar
    c = float(np.clip(c, 0.02, 1.0))

    def _sig(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))

    # convex (S-shaped) read-out of the relative balance: |R| = 1 is
    # untouched, small residual imbalances are discounted toward chance.
    def _drive(R):
        R = float(np.clip(R, -1.0, 1.0))
        return float(np.sign(R) * (abs(R) ** kappa))

    # ------------------------------------------------------------------
    # 6. serial reading in screen order with graded sufficiency stopping;
    #    evidence is read fully relatively (R in [-1, 1]).
    # ------------------------------------------------------------------
    idxs = np.flatnonzero(disc)
    E = 0.0
    M = 0.0
    remain = 1.0
    p_a = 0.0
    last = int(idxs.shape[0]) - 1
    for t in range(idxs.shape[0]):
        j = int(idxs[t])
        E += w[j] * (1.0 if d[j] > 0 else -1.0)
        M += w[j]
        R = (E / M) if M > 1e-12 else 0.0
        p_here = _sig(beta * c * _drive(R))
        if t >= last:
            q = 1.0
        else:
            tail = w[j + 1:]
            bar = phi * float(np.max(tail)) if tail.size > 0 else 0.0
            if bar <= 1e-12:
                q = 1.0
            else:
                q = _sig(s * (abs(E) / bar - 1.0))
        p_a += remain * q * p_here
        remain *= (1.0 - q)
        if remain <= 1e-12:
            break
    if remain > 1e-12:
        R = (E / M) if M > 1e-12 else 0.0
        p_a += remain * _sig(beta * c * _drive(R))

    p = np.array([p_a, 1.0 - p_a], dtype=float)
    p = (1.0 - eps) * p + eps * 0.5
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


### slot 2 — `pi_7` — SURVIVED ✓

**Description:** **Reason-counting with a TWO-REGIME display-salience gate (full force only on a deadlocked vote, a constant residual otherwise), near-linear diagnosticity weighting, a saturating majority bonus, unanimity boost, sharp silence-to-confidence collapse (mutual endorsement ≈ silence), and a non-zero floor on read-out gain.**

This keeps every mechanism of the accepted base and changes exactly three calibration commitments (one of them structural-but-local: the shape of the salience gate).

1. **Direction is a vote over discriminating experts, graded near-linearly by stated validity.** Each discriminating expert contributes x_j = ((v_j-0.5)/0.5)^rho with rho ~0.88-1.02. At this exponent a single .95 expert is worth about two upper-middling experts (.72 + .70), so numerosity and diagnosticity are *balanced* rather than one dominating: three mid experts still outvote one top expert, but two mid experts only tie it. Screen position plays no role in this term.

2. **Display salience is a TIE-BREAKER WITH A RESIDUAL, not a quantity that decays with the size of the majority.** s_j = exp(-j/tau)+exp(-(n-1-j)/tau) over the literal display row, multiplied by a gate g = max(exp(-(k_A-k_B)^2/2 sigma^2), g_res) with sigma ~0.60 and g_res ~0.25. The gate has two regimes: when the reason count is exactly deadlocked the first/last rows of the panel decide the choice outright (g = 1); as soon as one side has more reasons, position stops being decisive but does NOT disappear — it persists as a constant, modest attention bonus (g = g_res ≈ 0.25) that is the SAME for a one-reason edge and for a five-against-one landslide. The previous Gaussian-only gate made position vanish geometrically with the majority size, which is unmotivated: there is no reason why a larger majority should erase the extra attention paid to the top and bottom of the screen. The empirical signature is that a three-against-one display whose majority occupies both screen ends is markedly more decisive (~.88) than a three-against-one display whose lone dissenter sits at a screen end (~.82), even when the validities are matched.

3. **Clear majorities are categorical and saturating:** M = lambda*sign(k_A-k_B)*min(max(|k_A-k_B|-1,0),1) with lambda ~0.67. Once one side has at least two more reasons, 'more experts back this one' is itself a discrete reason, and it does not grow further with the size of the majority.

4. **Unanimity among the speaking experts is categorical**: if every expert who noticed a difference points the same way, a constant bonus is added.

5. **Gap inference from the panel is a sharp threshold on how much of the panel was informative, and 'both products are good' is nearly as uninformative as 'neither is'.** Validity is read as sensitivity: a difference only sensitive experts notice implies a marginal true gap. Mutual endorsements count at weight omega ~0.8. c = 1 - eta*sigmoid(k*(f_sil - f0))*vbar.

6. **The collapse of confidence is bounded below:** c is floored at c_min ~0.10, so a panel judged uninformative leaves a small residual tilt rather than a literal coin flip.

7. **A small, display-contingent first-cue commitment survives**: people commit to the first expert who actually discriminates only if no expert further down the list outranks him in stated validity. Individual differences are unimodal jitter in rho, mu, sigma, g_res, lambda, eta, omega, c_min, beta; no strategy subpopulations, no serial stop rule, no learning.

**Rationale:** **Three-line diff on the accepted iter-7 base: (i) the parity gate gets a constant residual, `g = max(exp(-dk^2/2 sigma^2), g_res)` with g_res~0.25 (new parameter); (ii) rho [0.60,0.78] -> [0.88,1.02]; (iii) lam [0.62,0.85] -> [0.60,0.74]. sigma stays .60, c_min stays .10, nothing else moves.**

**Why I trust the arithmetic this time.** I rebuilt the cell-level algebra of the accepted base and validated it against the simulator before proposing: it reproduces Exp1 = +.106 EXACTLY (the metric is carried entirely by the commitment branch, since in Exp1 every |dk|>=2 cell has d0!=0 and therefore q_com=1), Exp8 = -.207 vs simulated -.205, Exp10 = .89 vs .887, Exp5 = .80 vs .806, Exp2 within .03, and it predicts iter-9's realised deltas (Exp6 +.024 predicted / +.013 observed; Exp2 +.007 / +.005). It also correctly retro-predicts why iter-8's sigma-trim failed (Exp9 cell 8 has c=1 and loses .06 when g(1) shrinks, swamping the +.02/+.03 available in Exp3/Exp4). I use that same machinery below.

**Edit 1 - residual floor on the salience gate (the one surgical lever the battery still offers).** Exp6-class-3 and Exp2's conflict cells are the SAME configuration to the model (three agreeing cues vs one opposing cue, |dk|=2, disc rows {0,1,2,5}); humans put them at ~.88 and .775. The only thing that distinguishes them is WHERE the majority sits on the screen: in Exp6 the three agreeing cues own both display ends (S = +2.22 toward the majority), in Exp2 the lone dissenter owns an end (S = -0.57, i.e. only .57 toward the majority). The current Gaussian gate kills salience at |dk|=2 (g=.004) and therefore cannot use this 4:1 asymmetry at all. Flooring the gate at g_res=.25 - exactly the value the Gaussian already takes at |dk|=1, so EVERY |dk|<=1 cell is bit-for-bit unchanged (Exp3, Exp4, Exp7 both classes, Exp8's ladder, Exp9 both cells, Exp6-class-2) and every dk=0 cell is unchanged (Exp5, Exp10, g=1 either way) - buys Exp6-class-3 +.09 while costing Exp2 only +.05, a 2:1 ratio. Exp1 is provably untouched because its |dk|>=2 cells are commitment-dominated (q_com=1). Theoretically this is the cleaner claim: there is no reason why a larger majority should erase the extra attention paid to the top and bottom rows; position stops deciding, it does not evaporate.

**Edit 2 - rho to ~0.95 (the critic's direction, at the upper end of its request).** This is the only lever that moves Exp2, Exp3, Exp4, Exp9 and Exp10 the same way at once, and it neatly pays for the Exp2 cost of edit 1. Cell arithmetic at rho .95: Exp2's three conflict cells go .852 -> .853 AFTER absorbing edit 1 (i.e. rho cancels the salience-floor cost almost exactly); Exp3 .470 -> .476; Exp4 .436 -> .450; Exp9 .453 -> .457; Exp10 .887 -> .876; Exp8's ladder lands at -.203 (real -.203) because the ladder is carried by vbar, not by rho; Exp5 -.004; Exp7 unchanged (both load classes move together). The known cost is Exp6-class-3 (V drops .55 -> .28) but edit 1 over-compensates it by 4x. I am going past the critic's .83 because the regime has changed since the last rho experiment: with c floored at c_min and salience now indexed by absolute display row, the sign of the rho effect on Exp2/Exp9 has flipped relative to iteration 4.

**Edit 3 - lam .75 -> .67.** The one component of iterations 8 and 9 that paid in both runs. At the new operating point d(sum|delta|)/dlam = +.07 (Exp6-class-3 is saturating at p~.89 so it loses only .105/unit, while Exp2 gains .122/unit), so the trim is unconditionally positive and it keeps Exp2 off the ceiling.

**Projected cell-by-cell change in |delta| (base -> new):** Exp6 .180 -> .143, Exp4 .090 -> .076, Exp3 .063 -> .057, Exp10 .037 -> .026, Exp9 .037 -> .033, Exp2 .083 -> .084, Exp1 .013 -> .016, Exp5/Exp7/Exp8 unchanged. Sum |delta| .559 -> .494, i.e. loss ~.0695 -> ~.061 - roughly 1.6 standard deviations of the Monte-Carlo noise I estimated from the iter-7/iter-9 comparison, which is the smallest edit I could find that clears the noise floor the critic (correctly) flagged.

**Parameters:**
  - `rho`: `[0.88, 1.02]`
  - `mu`: `[1.70, 1.95]`
  - `lam`: `[0.60, 0.74]`
  - `sigma`: `[0.55, 0.66]`
  - `g_res`: `[0.22, 0.28]`
  - `tau`: `[0.85, 1.25]`
  - `ups`: `[1.25, 1.80]`
  - `beta`: `[1.05, 1.40]`
  - `eta`: `[1.45, 1.65]`
  - `sil_k`: `[30.0, 45.0]`
  - `sil_f0`: `[0.345, 0.375]`
  - `omega`: `[0.72, 0.92]`
  - `c_min`: `[0.07, 0.13]`
  - `phi`: `[0.95, 1.12]`
  - `s_stop`: `[6.0, 10.0]`
  - `d_com`: `[2.20, 2.90]`
  - `epsilon`: `[0.01, 0.06]`
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

### `pi_8` → slot 1 (via `new_theory`)

**Description:** ATTENTION LOTTERY WITH LIMITED-SAMPLE COMPARISON, AUTHORITY-CONTINGENT CAPTURE, PANEL-VOLUME CONFIDENCE, AND A CATEGORICAL (STEP-LIKE) PANEL-INFORMATIVENESS GATE. (1) Each display row enters an attention lottery with weight att_j = s_j * x_j * pop_j, where s_j = exp(-j/tau) + gamma*exp(-(n-1-j)/tau) is literal screen salience (asymmetric: top of panel stronger than bottom), x_j = ((v_j-.5)/.5)^rho is mildly compressed stated diagnosticity (rho ~ .55-.85), and pop_j is a pop-out factor: rows on which the two columns are identical do not pop out and keep only a residual share delta, while the FIRST row at which the columns disagree gets a bonus. (2) That first-mismatch bonus is AUTHORITY-CONTINGENT: a row captures attention outright only if it is both the first thing that discriminates AND the most diagnostic discriminating row on the screen. Position is always active, never gated by a deadlock rule. (3) The subject does not integrate the panel: only k = clip(ceil(k_frac*m), 2, 3) rows are actually compared, drawn without replacement with probability proportional to att. Choice = the attention-weighted sign of the sampled differences plus a small within-sample count term; a 1-1 sample is decided by the more salient sampled row. Numerosity helps only because more rows on one side makes the sample more likely to be dominated by that side; there is no unanimity bonus, no additive majority bonus, no serial stop rule. Because the sample is small and random, the map from true evidence to P(choice) is compressive with a ceiling near .85-.90 even for landslides. (4) NEW COMMITMENT (this iteration): THE READER'S JUDGEMENT OF WHETHER THE PANEL IS INFORMATIVE AT ALL IS CATEGORICAL, NOT GRADED. Silence is read as sensitivity, but the subject does not compute a continuous 'how much of the panel was mute' discount; they make a yes/no assessment — 'did most of this panel have nothing to say?' — with the boundary at about two-fifths of the panel (f0 ~ .39) and an essentially step-like transition (kappa ~ 100). Below the boundary the sampled verdict is taken at full face value (c ~ 1); above it the verdict is essentially voided and the choice approaches a coin flip (c floored near .03). There is no intermediate regime in which a panel is 'somewhat' informative. Mutual endorsements (both columns say yes) count toward silence at weight omega ~ .84, i.e. they are nearly, but not quite, as uninformative as mutual abstentions; because the gate is now a step, this small composition weight decides on which SIDE of the categorical boundary a mixed panel falls, which is where the whole behavioural signature lives. (5) Confidence also scales with the VOLUME of the panel, not with the amount actually read: beta_eff = beta*(n/6)^zeta (illusion of knowledge — short panels are systematically less decisive than structurally identical cells on wide panels). (6) Individual differences are unimodal jitter in k_frac, tau, gamma, rho, phi, phi_auth, s_auth, beta, zeta, eta, omega, c_min and a lapse rate; no strategy subpopulations, no learning.

**Rationale:** MINIMAL DIFF on the accepted iter-8 base (loss 0.1152). Only the silence-gate constants change: kap_s [18,32]->[70,130] (and its clip ceiling raised from 80 to 400 so the range is reachable), f0 [0.345,0.385]->[0.380,0.396], omega [0.80,0.94]->[0.80,0.88]. Every other line - the authority-contingent pop-out, beta_eff = beta*(n/6)^zeta, beta, eta, c_min, delta_sil, rho, tau, gamma_rec, lam_w, k_frac, epsilon, the lottery and read-out code - is re-emitted verbatim. No new mechanism, no unanimity/majority bonus, no deadlock gate, no serial stop.

WHY I DID NOT TAKE THE PRESCRIBED OMEGA CUT. I checked the critic's arithmetic on the two cells it names and it inverts. Exp7's hi-load cells (trials 6 and 14) have THREE both-1 mutes of six, so f_sil = omega/2, which drops below f0 = 0.365 as soon as omega < 0.73. Exp6's class-2 cells (disc {0,2,5}) have TWO both-1 plus ONE both-0 mute, so f_sil = (1+2*omega)/6, which only drops below f0 when omega < 0.595. The prescribed range [0.60,0.75] therefore frees the GUARDRAIL cell (Exp7 hi-load) from the discount while leaving the TARGET cell (Exp6 class-2) still discounted - exactly backwards. Exp7's metric is hi minus lo; its lo-load cells are three both-0 mutes (f_sil = 0.50, permanently floored at chance), so freeing hi-load alone takes the contrast from +0.06 to roughly +0.25 to +0.34 against a real value of +0.053, a ~0.25 regression that dwarfs the at-most +0.10 available on Exp6. Exp6 class-2 and Exp7 hi-load are in genuine mechanism conflict here (identical silence composition, opposite required decisiveness), so Exp6 is not addressable by omega and I leave it.

WHAT I DID INSTEAD, AND WHY IT IS THE SAME MECHANISM. The critic's deeper point is right - the silence term is where the leverage is - but the productive knob is the SHAPE and LOCATION of the gate, not the both-1 weight. At kap_s = 25 the sigmoid is soft, so a large set of cells sit in a mushy intermediate regime with c ~ 0.3-0.9, and those are precisely the mis-predicted ones. Turning the gate into a step at f0 ~ 0.39 sorts every diagnostic cell cleanly onto one side, and I verified the sorting cell-by-cell (f_sil computed from the actual designs):
  FREED (c -> 1, becomes decisive): Exp10 cell-5 f_sil = 2w/6 = 0.29 (0.793 -> ~0.83, real 0.850); Exp11 c5 f_sil = 2w/5 = 0.34 - and c5 enters the index NEGATIVELY, so index 0.842 -> ~0.78, the first gain on that cell that does not cost elsewhere; Exp6 class-3 f_sil = (1+w)/6 = 0.31 (small gain); Exp12 cell-4 f_sil = 3w/7 = 0.343-0.377, which is why omega's TOP end is trimmed to 0.88 - it keeps this cell cleanly below f0 for every subject (cell-4 0.675 -> ~0.79, Exp12 0.411 -> ~0.47); Exp1 pairs 7/9 f_sil = 2w/5 = 0.34.
  VOIDED (c -> c_min, becomes chance): Exp4's four qualifying cells, all f_sil = 2/5 = 0.40 exactly, just above the new f0 ceiling of 0.396 - this is the big one, 0.379 -> ~0.48 against real 0.508, because those cells' entire residual was the soft-sigmoid c ~ 0.32 leaving a residual extremes tilt; Exp1's agreement pairs 6/12 (f_sil = 0.40), which lifts the conflict-minus-agreement contrast toward the real +0.093.
  STRUCTURALLY UNCHANGED (already saturated or already free): Exp3 (f_sil = 0.50, all both-0, stays at chance 0.469 - the cell that proves the step is in the right place), Exp8 (single-cue trials, f_sil >= 0.64, reversed ladder untouched at -0.19), Exp9 (cell-5 f_sil = 0.48 floored, cell-8 f_sil = 0.15 free - both already on the correct side), Exp5 (one mute, f_sil = 0.20), Exp6 class-2 and Exp7 both cells (f_sil 0.43-0.50, all floored, so Exp7's contrast stays near zero as it must).

NET EXPECTATION: gains of ~0.11 on Exp4, ~0.06 on Exp11, ~0.05 on Exp12, ~0.04 on Exp10, ~0.02 each on Exp1/Exp6, against small costs of ~0.01-0.04 on Exp2 (its one partially-discounted conflict pair becomes decisive) and ~0.04 on Exp7 (its already-small contrast compresses further). Eight of twelve experiments move by less than 0.02. Theoretically this is a sharpening, not a fudge: the reader does not compute a graded silence discount, they make a categorical call on whether the panel said anything worth reading, and only near that boundary does the both-1 versus both-0 composition of the mute rows matter at all.

**Parameters:**
  - `tau`: `[0.72, 1.00]`
  - `gamma_rec`: `[0.38, 0.62]`
  - `rho`: `[0.55, 0.85]`
  - `delta_sil`: `[0.04, 0.11]`
  - `phi_first`: `[0.30, 0.70]`
  - `phi_auth`: `[4.0, 9.0]`
  - `s_auth`: `[6.0, 16.0]`
  - `lam_w`: `[0.82, 0.95]`
  - `beta`: `[2.10, 2.70]`
  - `zeta_n`: `[1.0, 2.0]`
  - `eta`: `[1.30, 1.60]`
  - `f0`: `[0.380, 0.396]`
  - `kap_s`: `[70.0, 130.0]`
  - `omega`: `[0.80, 0.88]`
  - `c_min`: `[0.015, 0.055]`
  - `k_frac`: `[0.40, 0.55]`
  - `epsilon`: `[0.02, 0.06]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    from itertools import combinations, permutations

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
    # 2. communicated validities -> diagnosticity
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
    def _pget(key, dflt, lo, hi):
        try:
            v = float(parameters.get(key, dflt))
        except Exception:
            v = float(dflt)
        if not np.isfinite(v):
            v = float(dflt)
        return float(np.clip(v, lo, hi))

    tau = _pget('tau', 0.85, 0.2, 8.0)
    gam = _pget('gamma_rec', 0.50, 0.0, 1.5)
    rho = _pget('rho', 0.70, 0.05, 4.0)
    dlt = _pget('delta_sil', 0.07, 0.0, 1.0)
    phi = _pget('phi_first', 0.50, 0.0, 3.0)
    phi_auth = _pget('phi_auth', 6.0, 0.0, 20.0)
    s_auth = _pget('s_auth', 10.0, 0.5, 60.0)
    lam = _pget('lam_w', 0.88, 0.0, 1.0)
    beta = _pget('beta', 2.40, 0.05, 40.0)
    zet = _pget('zeta_n', 1.50, 0.0, 4.0)
    eta = _pget('eta', 1.45, 0.0, 5.0)
    f0 = _pget('f0', 0.388, 0.0, 1.0)
    kap = _pget('kap_s', 100.0, 1.0, 400.0)
    ome = _pget('omega', 0.84, 0.0, 1.0)
    cmin = _pget('c_min', 0.03, 0.0, 0.5)
    kfrac = _pget('k_frac', 0.47, 0.05, 1.0)
    eps = _pget('epsilon', 0.035, 0.0, 0.5)

    def sig(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))

    # ------------------------------------------------------------------
    # 4. attention weights: screen salience x compressed diagnosticity,
    #    matched rows do not pop out (residual share delta), first
    #    mismatching row of the top-down scan gets a pop-out bonus that
    #    is AUTHORITY-CONTINGENT: it captures attention outright only if
    #    no more diagnostic expert further down the panel also speaks.
    # ------------------------------------------------------------------
    pos = np.arange(n, dtype=float)
    s = np.exp(-pos / tau) + gam * np.exp(-(n - 1.0 - pos) / tau)
    x = np.power(vs, rho)
    att = s * x
    att = np.where(disc, att, dlt * att)
    idxs = np.flatnonzero(disc)
    att = np.array(att, dtype=float)
    j1 = int(idxs[0])
    if idxs.shape[0] > 1:
        x_rest = float(np.max(x[idxs[1:]]))
        ratio = float(x[j1]) / max(x_rest, 1e-9)
        q_auth = float(sig(s_auth * (ratio - 1.0)))
    else:
        q_auth = 1.0
    att[j1] = att[j1] * (1.0 + phi + phi_auth * q_auth)
    att = np.clip(att, 1e-12, None)

    # ------------------------------------------------------------------
    # 5. CATEGORICAL panel-informativeness gate (silence-as-sensitivity).
    #    The reader makes a yes/no judgement 'did most of the panel have
    #    nothing to say?' with an essentially step-like boundary at f0;
    #    below it the verdict is taken at face value, above it it is
    #    voided (floored near zero).  Mutual endorsements count toward
    #    silence at weight omega, which decides which SIDE of the
    #    boundary a mixed panel falls on.
    # ------------------------------------------------------------------
    both1 = (~disc) & (a > 0.5) & (b > 0.5)
    n_both0 = float(np.count_nonzero((~disc) & (~both1)))
    n_both1 = float(np.count_nonzero(both1))
    f_sil = float(np.clip((n_both0 + ome * n_both1) / float(n), 0.0, 1.0))
    xw = x[idxs]
    sxw = float(np.sum(xw))
    vbar = float(np.sum(xw * vs[idxs]) / sxw) if sxw > 1e-12 else 0.5
    c = 1.0 - eta * float(sig(kap * (f_sil - f0))) * vbar
    c = float(np.clip(c, cmin, 1.0))

    # ------------------------------------------------------------------
    # 5b. PANEL-VOLUME CONFIDENCE: the sampled comparison stays the same
    #     size, but the felt informativeness of its verdict grows with
    #     how many experts were displayed (illusion of knowledge).
    # ------------------------------------------------------------------
    beta_eff = beta * float(np.power(float(n) / 6.0, zet))

    # ------------------------------------------------------------------
    # 6. the lottery: draw k rows without replacement, prob ~ attention
    # ------------------------------------------------------------------
    m = int(idxs.shape[0])
    k = int(np.clip(int(np.ceil(kfrac * m)), 2, 3))
    k = int(min(k, n))
    if k < 1:
        k = 1

    p = att / float(np.sum(att))
    p = 0.998 * p + 0.002 / float(n)
    p = p / float(np.sum(p))

    tot = 0.0
    pa = 0.0
    for S in combinations(range(n), k):
        ps = 0.0
        for perm in permutations(S):
            q = 1.0
            rem = 1.0
            good = True
            for t in perm:
                if rem <= 1e-9:
                    good = False
                    break
                q *= p[t] / rem
                rem -= p[t]
            if good:
                ps += q
        if ps <= 1e-14:
            continue
        sel = [j for j in S if disc[j]]
        if len(sel) == 0:
            E = 0.0
        else:
            ws = att[sel]
            sg = np.sign(d[sel])
            wsum = float(np.sum(ws))
            Ew = float(np.sum(ws * sg) / wsum) if wsum > 1e-12 else 0.0
            nA = float(np.sum(sg > 0))
            nB = float(np.sum(sg < 0))
            Ec = (nA - nB) / (nA + nB) if (nA + nB) > 0 else 0.0
            E = lam * Ew + (1.0 - lam) * Ec
        pa += ps * float(sig(beta_eff * c * E))
        tot += ps

    p_a = (pa / tot) if tot > 1e-12 else 0.5

    pr = np.array([p_a, 1.0 - p_a], dtype=float)
    pr = (1.0 - eps) * pr + eps * 0.5
    pr = np.clip(pr, 1e-12, None)
    pr = pr / pr.sum()
    return pr
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
