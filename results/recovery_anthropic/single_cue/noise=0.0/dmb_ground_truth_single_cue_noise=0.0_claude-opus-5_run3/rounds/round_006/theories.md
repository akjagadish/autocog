# Round 6 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_8` — KILLED ✗

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

### `pi_9` → slot 1 (via `new_theory`)

**Description:** **Noisy compensatory reason-counting with TOP-DOWN READING ATTENTION (primacy-graded cue weights), mildly compressed diagnosticity, a count-contingent edge gate, a saturating numerosity bonus, a reversible gap inference, an authority-capped top-down entry, and a moderated read-out ceiling.**

This is the accepted compensatory-tally base with ONE new commitment: *position is not only a deadlock tie-breaker; it is an attention multiplier on the reasons themselves.*

1. **Direction is a weighted reason tally over discriminating experts.** x_j = ((v_j-.5)/.5)^rho with rho ~.82-1.08: validity is read coarsely enough that numerosity routinely outvotes a single strong expert, but a .95 oracle and a mid cue are not interchangeable.
2. **NEW: people read the panel top-down and the rows they read first carry extra weight beyond their stated validity.** Each cue's contribution to the tally is multiplied by att_j = 1 + alpha_p*exp(-j/tau_p) (alpha_p ~.45-.85, tau_p ~1.0-1.6), i.e. the first screen row is worth ~1.5-1.8 of itself, the second ~1.3, and by the fourth row the bonus is gone. This is a *graded* primacy of reading, not a first-cue capture and not a lottery: every row still contributes, and a mid-list oracle still outweighs a weak top row when the validity gap is large. It is what makes displays look partly lexicographic in designs whose validities descend down the screen, and what makes screen-top cues systematically over-followed in designs that hide the heavyweights in the middle.
3. **'More experts back this one' is a single discrete saturating extra reason** (fires at |k_A-k_B|>=2, never grows). No unanimity term.
4. **Screen position ALSO survives as a symmetric edge tie-breaker on deadlocked counts** (gain g = g_res + (1-g_res)exp(-dk^2/2 sigma^2), narrow sigma, small residual), which is the only thing that can decide displays where the reason counts tie exactly.
5. **Gap inference from panel silence is graded and reversible**: validity is read as sensitivity, c = 1 - eta*f_sil^q*vbar (mutual endorsements count as silence at weight omega<1), allowed to pass slightly negative, which is the sole mechanism that makes people choose against a lone highly-sensitive endorsement.
6. **Authority-capped top-down entry**: people commit to the first discriminating row only if no more diagnostic expert remains below it.
7. **Compressive read-out**: mild dilution by the number of speaking cues, logistic gain beta ~1.45-1.80, fixed lapse eps ~.13-.19 bounding P(choice) in roughly [.12,.88]. Individual differences are unimodal jitter in all parameters; no subpopulations, no learning.

**Rationale:** MINIMAL DIFF on the accepted base (iter 2, loss 0.1308): the oracle premium is gone (per the critic's directive 0 — it was rejected twice), beta/epsilon/eta/q_sil/omega/psi/c_neg/lam/phi/s_stop/mu/sigma/g_res/tau are ALL left at the accepted base values, and the only change is three added lines that multiply the tally weights by a top-down reading-attention profile att_j = 1 + alpha_p*exp(-j/tau_p), plus the two new parameters.

WHY I IMPLEMENTED THE CRITIC'S PRIMACY IDEA AS AN UNGATED WEIGHT MULTIPLIER RATHER THAN AS `delta` INSIDE s_sal. I simulated the critic's literal edit (s_sal = exp(-pos/tau) + delta*exp(-(n-1-pos)/tau)) cell by cell and it is nearly a no-op, for a structural reason the critic overlooked: the edge term is multiplied by the count gate g, and g = g_res + (1-g_res)exp(-dk^2/2sigma^2) is only ~0.08-0.11 whenever |dk|>=1. Every one of the cells the critic wanted primacy to fix is a NON-deadlock cell — Exp13 EXT/AUTH (dk=-2,-3), Exp13 TAIL (dk=1), Exp14 both cells (dk=-2), Exp6 class2/class3 (dk=1,2), Exp12 c8 (dk=2), Exp3/Exp4 (dk=1) — so the salience term contributes < 0.1 logits there and halving its bottom half changes nothing. The delta edit acts ONLY on the four deadlock cells (Exp5, Exp10 c5, Exp11 c4, Exp12 c4), and there my sim shows S falls by a near-constant factor (0.67-0.87) across all of them, so rescaling mu to protect Exp10 (guard rail >= .78) cancels the gain on Exp11/Exp12: net ≈ 0, with a real risk of a rejection for nothing. The critic's PSYCHOLOGY, however — 'people read the panel top-down; the first rows carry extra weight beyond their stated validity' — is exactly what an ungated attention multiplier on the reason weights implements, and it reaches all the non-deadlock cells.

CELL-BY-CELL SIM (alpha_p≈.65, tau_p≈1.2, so row0 x1.65, row1 x1.30, row2 x1.12, row3 x1.05, rows 4+ ≈1): Exp13 — all three target cells (EXT, AUTH, TAIL) have their top speaking row backing the under-chosen option, so all three rise together: .292 -> ~.36 (obs .353). Exp14 — pos0 (.74) is the only cue backing the focal against mid-list .95/.86: .179 -> ~.22 (obs .300). Exp6 — pos0 is part of the crowd opposing the mid-list heavyweight in both classes: .646 -> ~.72+ (obs .841). Exp12 — c8's pos0/pos1 oppose the tally winner, so it falls from .793 toward .665; c4 roughly unchanged. Exp9 — cell 1 (extremes vs mid) .57 -> .61 in sim, so the mean moves .445 -> ~.47 (obs .490). Exp1 — validities descend down the screen, so primacy is quasi-lexicographic and the TTB-consistency contrast rises from +.006 toward pi_7's +.09 (obs .093) without touching the twice-failed s_stop/phi knob. Exp8 — the single-cue ladder gets a bigger boost at pos0 (.72, strong) than at pos1 (.56, weak) and almost none at pos3/pos4, pulling the contrast from -.255 back toward the observed -.203. Exp10 — pos0/pos5 are the WEAK extremes (.62/.58) so multiplying them changes little (V: -1.205 -> -1.186) and the deadlock edge term still carries the cell: stays ~.82 (guard rail >= .78 met). Exp5 — deadlock, dominated by the edge term, essentially unchanged at ~.75. Exp7 — primacy is orthogonal to the shared-endorsement load manipulation, so the load contrast stays in [0.00, 0.08].

KNOWN COSTS, accepted deliberately: Exp2 rises ~.02 in sim (.777 -> ~.797, obs .775, still inside the .76-.79 guard rail's immediate neighbourhood) and Exp3/Exp4 lose ~.03/.04 because their anti-oracle crowd contains the top screen row. Summed over the corpus my sim gives ~+.35 of L1 gain against ~.10 of loss, concentrated in the largest residuals (Exp6 -.195, Exp14 -.121, Exp1 -.088, Exp13 -.061, Exp12 +.128), which is why I expect this to clear the 0.1308 floor where the literal delta edit would not. No mechanism was removed and no other parameter range was touched, so the arbiter's compensatory-tally prescription (flattened validity weighting, saturating majority, no unanimity, graded silence inference, explicit lapse ceiling) and the two empirically-earned deviations (count-contingent edge gate, authority-capped entry) are all intact.

**Parameters:**
  - `rho`: `[0.82, 1.08]`
  - `lam`: `[0.40, 0.65]`
  - `mu`: `[1.55, 1.95]`
  - `tau`: `[0.85, 1.25]`
  - `sigma`: `[0.33, 0.45]`
  - `g_res`: `[0.05, 0.11]`
  - `eta`: `[2.10, 2.70]`
  - `q_sil`: `[2.30, 2.90]`
  - `omega`: `[0.70, 0.90]`
  - `c_neg`: `[-0.55, -0.15]`
  - `psi`: `[0.05, 0.25]`
  - `beta`: `[1.45, 1.80]`
  - `d_com`: `[1.80, 2.50]`
  - `phi`: `[0.92, 1.04]`
  - `s_stop`: `[12.0, 22.0]`
  - `epsilon`: `[0.13, 0.19]`
  - `alpha_p`: `[0.45, 0.85]`
  - `tau_p`: `[1.00, 1.60]`
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
