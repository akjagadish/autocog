# Round 9 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_11` — KILLED ✗

**Description:** **SSG-6 — Salience-Share Gating with an opposition-graded adjudication limb and a leakier integration limb.**

The architecture is exactly SSG-5's; one quantitative commitment is revised, none is reversed.

**(1) Two modes: adjudication vs. integration.** On every trial the subject either lets ONE expert adjudicate the choice or integrates all disagreeing rows. Mode is decided by a RELATIVE quantity — the share of the display's total *oddness* owned by genuinely near-chance experts — not by display length nor by an absolute knife edge. Each discriminating row has oddness o_j = (1-v_j)^gamma_s (gamma_s ≈ 6.4-8.6, a contrast-enhanced, nearly ordinal 'who is the strangest expert here' read-out) and a scandal value g_j = sigmoid((v_crit - v_j)/w_h) with a crisp boundary (w_h ≈ .008-.011, v_crit ≈ .576-.586). The gate is share = Σ_j g_j·o_j / Σ_j o_j and pi_one = (max_j g_j)·[f + (1-f)·sigmoid((share - c)/w)]: adjudication requires BOTH that some expert be near-chance AND that near-chance experts own most of the display's oddness.

**(2) Adjudication is faithful but NOT flat in opposition; it is flat only in display length.** When a near-chance expert commands the trial, subjects follow its verdict with high fidelity (eps_t ≈ .19-.29) and scanning extra, silent or agreeing rows costs nothing at all — one-reason choice is free in sheer list length, which is why anchor-follow rates stay high on six-row displays. What it is NOT free in is genuine CONTRADICTION: each additional row that argues the other way beyond the first charges rho_opp ≈ .82-.90. A scandalous dissenter standing alone against one opposing voice is followed almost as often as a scandalous dissenter in a unanimous display, but once two or three experts are lined up the other way the adjudication itself becomes contested and the verdict is dragged appreciably toward chance. Among co-scandalous rows the consulted verdict is drawn from softmax(log g_j + mu·j) with a WEAK positional gain (mu ≈ .35-.62); reading order can never lift an ordinary expert over a near-chance one.

**(3) REVISED — integration is an intrinsically effortful, frequently-abandoned operation, and it is fragile under DISAGREEMENT.** When the gate is shut, evidence is integrated with w_j = (1-v_j)^gamma_w·exp(lam·j) (gamma_w ≈ 1.32-1.58, lam ≈ .10-.17, so doubt, not position, buys weight), divisively normalised to the loudest dissent, E = Σ sign_j·ŵ_j/(sigma + Σ ŵ_j), p = sigmoid(beta·E). The sharpened claim is that the BASELINE abandonment rate of integration is appreciably higher than previously assumed (eps0 ≈ .26-.34, not ≈ .22-.30): weighing several probabilistically-labelled experts against one another is an operation subjects decline to complete on roughly a third of trials even before any conflict or load penalty, so integrated verdicts are systematically less decided than the weights alone imply. On top of this baseline the retained evidence leaks further with display load (rho_m) and much more steeply with every additional row that argues the other way (rho_c ≈ .62-.75 per minority row). Because the baseline lapse applies to EVERY integration trial — including the unanimous/dominance displays where there is nothing to adjudicate — it predicts that even fully consistent multi-row displays stay clearly below ceiling (~.74-.75), while conflict displays are pulled nearer chance still and never reach a near-deterministic anti-validity verdict.

The theory's signature is therefore that BOTH modes are degraded by contradiction and NEITHER is degraded by list length — they differ in how steeply contradiction bites (mild for adjudication, severe for integration) and in their baseline completion rates (adjudication is cheap and usually completed; integration is expensive and often abandoned). When all discriminating experts agree there is nothing to adjudicate (pi_one = 0) and only the load/baseline leak applies. If two printed validities are indistinguishable (gap < .002) oddness has no referent and choice collapses to a graded unweighted row tally. No feedback is given, so the rule is stationary across the block.

**Rationale:** MINIMAL DIFF: `predict` and `policy` are re-emitted byte-for-byte. Exactly ONE parameter range moves — the critic's item 1, and nothing else.

Edit — eps0 [0.22, 0.30] -> [0.26, 0.34].

Why this is the right single lever now. eps0 enters the model only through keep_c = (1-eps0)·rho_m^(m-2)·rho_c^k_conf, i.e. it scales the INTEGRATION verdict's distance from chance and is structurally inert on the adjudication limb (p_one_a does not reference it). That makes its sign profile exactly aligned with the whole of the residual cluster the critic labelled group (i), which is uniformly too far from chance in the anti-validity / decided direction:

- Exp 5 (dominance accuracy) is the cleanest test: k_conf = 0 there, so pi_one = 0 identically and the trial is decided purely by keep_c. Currently .769 vs real .743; a ~.04 rise in eps0 scales (p_comp-.5) down by ~5-6% and moves it to ~.75. No other untouched knob reaches dominance displays at all (rho_c^0 = 1, rho_opp inert, gate closed).
- Exps 1 (-.028), 3 (-.020) and 12 (-.055) each retain a substantial (1-pi_one) integration component even though the gate is often open there; shrinking that component pulls all three toward .5, which is the correct direction for all three simultaneously.
- Exp 7 (+.023, too late-row-decided) and Exp 14 (+.039, above chance where subjects are below) are pulled DOWN toward their targets by the same shrink — the only two over-shot metrics in the group, and they move the right way.

Debits are small and bounded: Exp 13 (.521, real .546) drifts down by roughly .003-.005 and Exp 10 (.420, real .418) by ~.005; Exp 4 (.132, real .150) actually improves slightly since shrinking toward .5 raises it. Exps 15, 16, 18-T, 8, 9, 11 and 17 are adjudication-dominated or gate-shut and are essentially untouched.

Why this is not a repeat of the failed sigma/rho_c attempts (iters 2-3): those knobs act only on the WITHIN-integration evidence shape or on the conflict-count exponent, so they were structurally unable to reach (a) the dominance displays (k_conf = 0 kills rho_c) and (b) they were swamped on Exps 3/12 by the simultaneous eps_t drop. eps0 is a multiplicative floor on every integration trial regardless of m, k_conf, or the weight profile — the one dimension of integration determinism that has never been spent in this loop.

Per the critic I explicitly did NOT touch: eps_t and rho_opp (moving either now would be pure oscillation, and both push Exp 15 and Exp 8 the same way so their tension is unresolvable there), c_share / w_share / gate_floor (they buy Exps 2 and 4, pi_10's clearest failures), gamma_w / lam (delivering Exps 10/11/14/17), w_h / v_crit (Exp 16 now on target), mu (further spreading would hurt Exp 9, already .013 low), rho_m, rho_c, sigma, beta, kappa. Exps 13 and 14 remain treated as a jointly irreducible lone-dissenter pair straddling chance; no new term is introduced to separate them.

**Parameters:**
  - `gamma_s`: `[6.4, 8.6]`
  - `v_crit`: `[0.576, 0.586]`
  - `w_h`: `[0.008, 0.011]`
  - `c_share`: `[0.69, 0.75]`
  - `w_share`: `[0.040, 0.065]`
  - `gate_floor`: `[0.33, 0.47]`
  - `mu`: `[0.35, 0.62]`
  - `eps_t`: `[0.19, 0.29]`
  - `rho_opp`: `[0.82, 0.90]`
  - `gamma_w`: `[1.32, 1.58]`
  - `lam`: `[0.10, 0.17]`
  - `sigma`: `[0.58, 0.78]`
  - `beta`: `[2.15, 2.65]`
  - `eps0`: `[0.26, 0.34]`
  - `rho_m`: `[0.87, 0.93]`
  - `rho_c`: `[0.62, 0.75]`
  - `tally_kappa`: `[0.7, 1.3]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # SSG: Salience-Share Gating
    import numpy as np

    # ---------- parse the (2, n_features) stimulus robustly ----------
    a = None
    b = None
    if isinstance(state, dict):
        if 'option_a_ratings' in state and 'option_b_ratings' in state:
            a = np.asarray(state['option_a_ratings'], dtype=float).ravel()
            b = np.asarray(state['option_b_ratings'], dtype=float).ravel()
        else:
            vals = list(state.values())
            if len(vals) >= 2:
                a = np.asarray(vals[0], dtype=float).ravel()
                b = np.asarray(vals[1], dtype=float).ravel()
    if a is None or b is None:
        arr = np.asarray(state, dtype=float)
        if arr.ndim == 2 and arr.shape[0] == 2:
            a, b = arr[0].ravel(), arr[1].ravel()
        elif arr.ndim == 2 and arr.shape[1] == 2:
            a, b = arr[:, 0].ravel(), arr[:, 1].ravel()
        elif arr.ndim == 1:
            h = arr.shape[0] // 2
            a, b = arr[:h], arr[h:2 * h]
        else:
            f = np.asarray(arr, dtype=float).reshape(2, -1)
            a, b = f[0].ravel(), f[1].ravel()

    n = int(min(np.size(a), np.size(b)))
    if n == 0:
        return np.array([0.5, 0.5])
    a = np.asarray(a, dtype=float)[:n]
    b = np.asarray(b, dtype=float)[:n]

    # ---------- printed validities ----------
    v_raw = parameters.get('validities', None) if hasattr(parameters, 'get') else None
    if v_raw is None:
        v = np.linspace(0.95, 0.55, n)
    else:
        v = np.asarray(v_raw, dtype=float).ravel()
        if v.size < n:
            v = np.concatenate([v, np.full(n - v.size, 0.55)])
        v = v[:n]
    v = np.clip(v, 0.5, 0.9995)

    # ---------- parameters ----------
    gamma_s = float(parameters['gamma_s'])
    v_crit = float(parameters['v_crit'])
    w_h = float(max(float(parameters['w_h']), 1e-4))
    c_share = float(parameters['c_share'])
    w_share = float(max(float(parameters['w_share']), 1e-4))
    floor = float(np.clip(float(parameters['gate_floor']), 0.0, 1.0))
    mu = float(parameters['mu'])
    eps_t = float(np.clip(float(parameters['eps_t']), 0.0, 1.0))
    rho_opp = float(np.clip(float(parameters['rho_opp']), 1e-6, 1.0))
    gamma_w = float(parameters['gamma_w'])
    lam = float(parameters['lam'])
    sigma = float(max(float(parameters['sigma']), 1e-6))
    beta = float(parameters['beta'])
    eps0 = float(np.clip(float(parameters['eps0']), 0.0, 1.0))
    rho_m = float(np.clip(float(parameters['rho_m']), 1e-6, 1.0))
    rho_c = float(np.clip(float(parameters['rho_c']), 1e-6, 1.0))
    kappa = float(parameters['tally_kappa'])

    def sig(x):
        x = float(np.clip(x, -60.0, 60.0))
        return 1.0 / (1.0 + np.exp(-x))

    # ---------- discriminating rows ----------
    diff = a - b
    disc = np.nonzero(diff != 0.0)[0]
    m = int(disc.size)
    if m == 0:
        return np.array([0.5, 0.5])
    signs = np.sign(diff[disc])
    n_a = float(np.sum(signs > 0.0))
    n_b = float(m) - n_a
    k_conf = int(min(n_a, n_b))

    # ---------- leak on the integration limb ----------
    keep_c = (1.0 - eps0) * (rho_m ** max(m - 2, 0)) * (rho_c ** k_conf)
    keep_c = float(min(max(keep_c, 0.0), 1.0))

    # ---------- label-degeneracy gate ----------
    degenerate = False
    if n > 1:
        vs = np.sort(v)
        if bool(np.any(np.diff(vs) < 2.0e-3)):
            degenerate = True

    if degenerate:
        p_core = sig(kappa * (n_a - n_b))
        p_a = 0.5 + keep_c * (p_core - 0.5)
        p_a = float(min(max(p_a, 1e-9), 1.0 - 1e-9))
        p = np.array([p_a, 1.0 - p_a], dtype=float)
        return p / p.sum()

    vd = v[disc]
    jd = disc.astype(float)

    # ---------- scandal values and sharpened oddness ----------
    zz = np.clip((v_crit - vd) / w_h, -60.0, 60.0)
    g = 1.0 / (1.0 + np.exp(-zz))                       # near-chance "scandal" weight
    log_g = -np.logaddexp(0.0, -zz)                     # stable log sigmoid
    log_o = gamma_s * np.log(np.clip(1.0 - vd, 1e-12, None))
    log_o_shift = log_o - np.max(log_o)
    o = np.exp(np.clip(log_o_shift, -700.0, 0.0))
    o_sum = float(np.sum(o))
    if (not np.isfinite(o_sum)) or o_sum <= 0.0:
        o = np.ones(m, dtype=float)
        o_sum = float(m)

    share = float(np.sum(g * o) / o_sum)
    g_share = sig((share - c_share) / w_share)
    P_trig = float(np.clip(np.max(g), 0.0, 1.0))

    # unanimous displays need no adjudication
    if k_conf == 0:
        pi_one = 0.0
    else:
        pi_one = P_trig * (floor + (1.0 - floor) * g_share)
    pi_one = float(np.clip(pi_one, 0.0, 1.0))

    # ---------- which row adjudicates (recency only breaks scandal ties) ----------
    sc = log_g + mu * jd
    sc = sc - np.max(sc)
    ww = np.exp(np.clip(sc, -700.0, 0.0))
    ws = float(np.sum(ww))
    if (not np.isfinite(ws)) or ws <= 0.0:
        ww = np.full(m, 1.0 / m)
    else:
        ww = ww / ws
    p_or_a = float(np.sum(ww[signs > 0.0]))
    fid = (1.0 - eps_t) * (rho_opp ** max(k_conf - 1, 0))
    fid = float(min(max(fid, 0.0), 1.0))
    p_one_a = 0.5 + fid * (p_or_a - 0.5)

    # ---------- integration limb ----------
    logw = gamma_w * np.log(np.clip(1.0 - vd, 1e-12, None)) + lam * jd
    logw = logw - np.max(logw)
    wi = np.exp(np.clip(logw, -700.0, 0.0))
    wsum = float(np.sum(wi))
    if (not np.isfinite(wsum)) or wsum <= 0.0:
        wi = np.ones(m, dtype=float)
        wsum = float(m)
    E = float(np.sum(signs * wi)) / (sigma + wsum)
    p_comp = sig(beta * E)
    p_c_a = 0.5 + keep_c * (p_comp - 0.5)

    # ---------- mixture ----------
    p_a = pi_one * p_one_a + (1.0 - pi_one) * p_c_a
    p_a = float(min(max(p_a, 1e-9), 1.0 - 1e-9))
    p = np.array([p_a, 1.0 - p_a], dtype=float)
    return p / p.sum()
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if not np.isfinite(s) or s <= 0.0:
        return int(np.random.randint(len(probs)))
    probs = probs / s
    return int(np.random.choice(len(probs), p=probs))
```


### slot 2 — `pi_10` — SURVIVED ✓

**Description:** DTG-2R/OC-sharp (load-graded compensatory limb) — Doubt-Threshold-Gated one-reason choice with compensatory fallback, where the doubt gate is an ABSOLUTE and SHARP near-chance criterion, the one-reason limb is degraded by OPPOSITION rather than by raw display length, and the compensatory limb is degraded by DISPLAY LOAD more steeply than previously assumed. (1) One-reason adjudication is commanded only by an expert whose PRINTED validity is essentially a coin flip. The trigger is g_j = sigmoid((theta - v_j)/tau) evaluated only over rows that actually discriminate (P_trig = max_{j in disc} g_j), with theta ~ .558-.588 and tau ~ .006-.013. An expert at .52-.55 is 'newsworthy' (g >= .9) while an expert at .62-.68 is NOT (g <= .02): a merely mediocre expert is ordinary evidence, not a scandal demanding adjudication. Silent near-chance experts are irrelevant, so no null-verdict capture and no doubt-gradient leak are needed. (2) When a genuinely sub-threshold expert speaks, its verdict decides the trial essentially independently of how MANY higher-validity rows oppose it; among discriminating rows the consulted verdict is drawn from softmax(log g_j + mu*j), so the recency tilt mu only arbitrates BETWEEN near-chance rows. (3) Because a single row is consulted, merely scanning extra rows costs the one-reason limb almost nothing (rho_t ~ .985-1.0), but holding a genuine RIVAL COALITION does cost it: each minority row beyond the first charges rho_tc (~.62-.86). Hence a near-chance dissenter standing ALONE against 1-5 opposing rows is followed at ~.82-.86 flat in coalition size (Exps 15, 16), whereas BALANCED splits drift toward chance. (4) When no near-chance expert discriminates — including displays whose least valid dissenter is only mid-doubt (v ~ .58-.68) — evidence is compensatory: anti-validity, weakly recency-tilted weights w_j = (1-v_j)^gamma * exp(lam*j), divisively normalised to the loudest dissent, E = sum(s_j*what_j)/(sigma + sum(what_j)), p = sigmoid(beta*E). NEW, SHARPENED CLAIM: the compensatory rule is a genuinely effortful integration whose thread is lost quickly as the display lengthens — every discriminating row beyond the second multiplies the retained evidence by rho_m ~ .84-.93, appreciably steeper than the one-reason limb's near-free scan. This asymmetry is the theory's signature: one-reason choice is cheap to run on long displays and expensive only under opposition, whereas compensatory integration is expensive in sheer list length. Consequently long unanimous displays stay confident but clearly below ceiling (~.74-.76), and long conflict displays settle appreciably closer to chance than short ones, while short displays (m=2-3) are essentially unaffected. (5) If any two printed validities are within .002 the experts cannot be individuated, the doubt gate has no referent, and choice collapses to a graded unweighted row tally (exact coin flip on 1-1 splits). No feedback is given, so the rule is stationary across the block.

**Rationale:** MINIMAL DIFF on the ACCEPTED iter-4 base (loss .0549). `predict` and `policy` are re-emitted byte-identical apart from a comment line; sigma is restored to the accepted base's [0.40, 0.72] (undoing the iter-5 raise the gate rejected), and EXACTLY ONE range is changed: rho_m [0.89, 0.97] -> [0.84, 0.93].

Why this single knob, and why it should clear the .0549 floor:

(1) The critic's iter-5 post-mortem identified a clean LOAD-SCALING signature in the residual, not a scale/temperature one. Short-display compensatory cells are already essentially exact (Exp4 m=2-3 at .173 vs .150; Exp13 at .531 vs .546; Exp10 at .404 vs .418), while LONG-display compensatory cells are systematically too far from chance in whichever direction the fallback points: Exp5 (unanimous, m up to 6) at .785 vs .743 (+.042), Exp1 at .244 vs .282 (-.039), Exp12 at .220 vs .261 (-.040), Exp14 at .551 vs .468 (+.084), Exp3 at .273 vs .286. rho_m is the only parameter that enters as rho_m^(m-2) and ONLY on the compensatory branch, so it is silent exactly where the fit is already good and acts exactly where the residual lives.

(2) Effect size is above the measured noise floor (~.02-.03/cell), which the critic stressed is the binding constraint. At mid-box rho_m goes .93 -> .885, so at m=6 the retained deviation from chance is scaled by (.885/.93)^4 = .82, and at m=5 by .86 — an .04-.05 movement on the m=5-6 cells. Predicted: Exp5 .785 -> ~.75 (real .743), Exp1 .244 -> ~.27 (real .282), Exp12 .220 -> ~.24 (real .261), Exp14 .551 -> ~.52 (real .468), Exp3 .273 -> ~.29 (real .286). Bounded costs on the short-display cells: Exp4 ~+.01-.02 (toward .19 vs real .150), Exp13 ~-.01, Exp10 ~+.01-.02. Net clearly positive on paper across five cells.

(3) The trigger limb is untouched by construction, so the two edits the gate has already rewarded are fully preserved: theta [.558,.588] / tau [.006,.013] (the .52/.545/.55 anchors keep g>=.9; the .62/.68 dissenters keep g<=.02) and rho_tc [.62,.86] / rho_t [.985,1.00]. Exps 15, 16, 8, 9, 11 therefore move only through whatever small compensatory mass they carry.

(4) I explicitly do NOT repeat anything the gate has punished: no sigma raise (iter-5 REJECTED, and the per-cell evidence shows sigma is near-inert here — on dominance displays sum(w) is large so sigma barely shifts E), no blanket eps_t cut (iter-2 REJECTED), no beta-cut/eps0-raise bundle (iter-2 REJECTED), and no gamma/lam edit (iter-1 showed it is inert on Exps 13/14, and the human data show no anti-validity gradient between v=.62 and v=.68, so any gamma cut that helps Exp14 costs Exp13 about as much).

(5) rho_c is deliberately left at [.74,.90]. The critic listed it only as a conditional SECOND step, and the loop has twice punished multi-knob compensatory edits; a single knob with a >=.04 predicted movement on multiple cells is the edit most likely to be distinguishable from noise by the accept gate.

(6) Exp2 is again not chased — a documented family-level cost the arbiter accepted when prescribing DTG-2R; every fix would either undo rho_tc (re-breaking Exp1/Exp3) or weaken the near-chance gate Exps 15/16 depend on.

**Parameters:**
  - `theta`: `[0.558, 0.588]`
  - `tau`: `[0.006, 0.013]`
  - `mu`: `[0.45, 0.95]`
  - `gamma`: `[1.5, 2.3]`
  - `lam`: `[0.0, 0.22]`
  - `sigma`: `[0.40, 0.72]`
  - `beta`: `[1.8, 2.6]`
  - `eps_t`: `[0.20, 0.36]`
  - `rho_t`: `[0.985, 1.00]`
  - `rho_tc`: `[0.62, 0.86]`
  - `eps0`: `[0.13, 0.27]`
  - `rho_m`: `[0.84, 0.93]`
  - `rho_c`: `[0.74, 0.90]`
  - `tally_kappa`: `[0.6, 1.3]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # DTG-2R/OC: Doubt-Threshold-Gated one-reason choice with compensatory fallback;
    # one-reason limb degraded by OPPOSITION (rho_tc) rather than by raw row count,
    # compensatory limb degraded by DISPLAY LOAD (rho_m)
    import numpy as np

    # ---------- parse the (2, n_features) stimulus robustly ----------
    a = None
    b = None
    if isinstance(state, dict):
        if 'option_a_ratings' in state and 'option_b_ratings' in state:
            a = np.asarray(state['option_a_ratings'], dtype=float).ravel()
            b = np.asarray(state['option_b_ratings'], dtype=float).ravel()
        else:
            vals = list(state.values())
            if len(vals) >= 2:
                a = np.asarray(vals[0], dtype=float).ravel()
                b = np.asarray(vals[1], dtype=float).ravel()
    if a is None or b is None:
        arr = np.asarray(state, dtype=float)
        if arr.ndim == 2 and arr.shape[0] == 2:
            a, b = arr[0].ravel(), arr[1].ravel()
        elif arr.ndim == 2 and arr.shape[1] == 2:
            a, b = arr[:, 0].ravel(), arr[:, 1].ravel()
        elif arr.ndim == 1:
            h = arr.shape[0] // 2
            a, b = arr[:h], arr[h:2 * h]
        else:
            f = np.asarray(arr, dtype=float).reshape(2, -1)
            a, b = f[0].ravel(), f[1].ravel()

    n = int(min(np.size(a), np.size(b)))
    if n == 0:
        return np.array([0.5, 0.5])
    a = np.asarray(a, dtype=float)[:n]
    b = np.asarray(b, dtype=float)[:n]

    # ---------- printed validities ----------
    v_raw = parameters.get('validities', None) if hasattr(parameters, 'get') else None
    if v_raw is None:
        v = np.linspace(0.95, 0.55, n)
    else:
        v = np.asarray(v_raw, dtype=float).ravel()
        if v.size < n:
            v = np.concatenate([v, np.full(n - v.size, 0.55)])
        v = v[:n]
    v = np.clip(v, 0.5, 0.9995)

    # ---------- parameters ----------
    theta = float(parameters['theta'])
    tau = float(max(float(parameters['tau']), 1e-4))
    mu = float(parameters['mu'])
    gamma = float(parameters['gamma'])
    lam = float(parameters['lam'])
    sigma = float(max(float(parameters['sigma']), 1e-6))
    beta = float(parameters['beta'])
    eps_t = float(np.clip(float(parameters['eps_t']), 0.0, 1.0))
    rho_t = float(np.clip(float(parameters['rho_t']), 1e-6, 1.0))
    rho_tc = float(np.clip(float(parameters['rho_tc']), 1e-6, 1.0))
    eps0 = float(np.clip(float(parameters['eps0']), 0.0, 1.0))
    rho_m = float(np.clip(float(parameters['rho_m']), 1e-6, 1.0))
    rho_c = float(np.clip(float(parameters['rho_c']), 1e-6, 1.0))
    kappa = float(parameters['tally_kappa'])

    def sig(x):
        x = float(np.clip(x, -60.0, 60.0))
        return 1.0 / (1.0 + np.exp(-x))

    # ---------- discriminating rows ----------
    diff = a - b
    disc = np.nonzero(diff != 0.0)[0]
    m = int(disc.size)
    if m == 0:
        return np.array([0.5, 0.5])
    signs = np.sign(diff[disc])
    n_a = float(np.sum(signs > 0.0))
    n_b = float(m) - n_a
    k_conf = int(min(n_a, n_b))

    # ---------- lapses ----------
    expo = max(m - 2, 0)
    keep_c = (1.0 - eps0) * (rho_m ** expo) * (rho_c ** k_conf)
    keep_c = float(min(max(keep_c, 0.0), 1.0))
    # one-reason limb: scanning extra rows is nearly free, holding a RIVAL
    # coalition is not -> charge rho_tc per minority row beyond the first
    keep_t = (1.0 - eps_t) * (rho_t ** expo) * (rho_tc ** max(k_conf - 1, 0))
    keep_t = float(min(max(keep_t, 0.0), 1.0))

    # ---------- label-degeneracy gate ----------
    degenerate = False
    if n > 1:
        vs = np.sort(v)
        if bool(np.any(np.diff(vs) < 2.0e-3)):
            degenerate = True

    if degenerate:
        p_core = sig(kappa * (n_a - n_b))
        p_a = 0.5 + keep_c * (p_core - 0.5)
        p_a = float(min(max(p_a, 1e-9), 1.0 - 1e-9))
        p = np.array([p_a, 1.0 - p_a], dtype=float)
        return p / p.sum()

    # ---------- absolute doubt trigger over DISCRIMINATING rows ----------
    zz = np.clip((v - theta) / tau, -60.0, 60.0)
    gtrig = 1.0 / (1.0 + np.exp(zz))          # = sigmoid((theta - v)/tau)
    gd = gtrig[disc]
    P_trig = float(np.clip(np.max(gd), 0.0, 1.0))

    # verdict of the consulted (near-chance) expert; recency tilt among them
    logs = np.log(np.clip(gd, 1e-300, None)) + mu * disc.astype(float)
    logs = logs - np.max(logs)
    s = np.exp(np.clip(logs, -700.0, 0.0))
    ssum = float(np.sum(s))
    if (not np.isfinite(ssum)) or ssum <= 0.0:
        s = np.full(m, 1.0 / m)
    else:
        s = s / ssum
    p_or_a = float(np.sum(s[signs > 0.0]))
    p_t = 0.5 + keep_t * (p_or_a - 0.5)

    # ---------- compensatory fallback (divisively normalised anti-validity) ----------
    logw = gamma * np.log(np.clip(1.0 - v, 1e-12, None)) + lam * np.arange(n, dtype=float)
    ld = logw[disc]
    ld = ld - np.max(ld)
    w = np.exp(np.clip(ld, -700.0, 0.0))
    wsum = float(np.sum(w))
    if (not np.isfinite(wsum)) or wsum <= 0.0:
        w = np.ones(m, dtype=float)
        wsum = float(m)
    E = float(np.sum(signs * w)) / (sigma + wsum)
    p_comp = sig(beta * E)
    p_c = 0.5 + keep_c * (p_comp - 0.5)

    # ---------- doubt-threshold-gated mixture ----------
    p_a = P_trig * p_t + (1.0 - P_trig) * p_c
    p_a = float(min(max(p_a, 1e-9), 1.0 - 1e-9))
    p = np.array([p_a, 1.0 - p_a], dtype=float)
    return p / p.sum()
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if not np.isfinite(s) or s <= 0.0:
        return int(np.random.randint(len(probs)))
    probs = probs / s
    return int(np.random.choice(len(probs), p=probs))
```


## Replacement

### `pi_12` → slot 1 (via `new_theory`)

**Description:** **AIG-4 — Authority-Inversion with a near-zero ordinary positional tilt and a stronger co-scandal recency gain.** (Identical architecture to AIG-3; two quantitative commitments sharpened, none reversed.)

One graded sampling rule governs every trial; there is no mode switch and no compensatory integration anywhere in the architecture.

**(1) Salience = an absolute near-chance bonus plus a weak doubt tilt.** Each printed expert j carries a *scandal* value g_j = sigmoid((theta - v_j)/tau) with an ABSOLUTE, SHARP criterion placed so that a printed .55 is unambiguously a coin flip (theta ~ .562-.572, tau ~ .003-.006 gives g(.55) >= .92, g(.58) <= .12, g(.60) ~ 0). Salience is log u_j = g_j*(kappa + mu*j) + gamma*log(1-v_j) + lam*j. The scandal term is an enormous additive log-bonus (kappa ~ 5-7.5), so whenever a genuinely near-chance expert is present it owns essentially all salience; the ordinary doubt tilt gamma is weak and only arbitrates among ordinary experts.

**(2) SHARPENED — reading position is NOT an ordinary source of salience.** The free-standing positional tilt is pushed to essentially zero (lam ~ .00-.10). Among ordinary, non-scandalous experts, where a row happens to sit in the list buys it almost no extra chance of being consulted: what makes a row consultable is its *credibility*, not its recency. The apparent 'recency' effects in the corpus are therefore carried almost entirely by the scandal term's own positional gain mu, which is now raised (mu ~ .90-1.40) and rides ONLY on g_j. Reading order thus arbitrates exclusively BETWEEN co-scandalous experts — in alternating high/near-chance designs the LAST near-chance row is the one consulted — and it can never lift an ordinary expert over a near-chance one, nor can it decide between two ordinary experts. This is a genuine, falsifiable commitment: displays with exactly one scandal row (or none) should show no recency advantage at all, whereas displays with several near-chance rows should show a strong one.

**(3) A single row is consulted — never a sum.** The trial is settled by the verdict of ONE row drawn from softmax(log u) — a probabilistic one-reason rule. Coalitions buy almost nothing: adding agreeing rows does not add evidence, it only adds lottery tickets. Hence tally-vs-TTB gaps stay small and validity-weighted metrics sit far below chance.

**(4) Authority inversion is adopted wholesale, flat in display length.** A discriminating near-chance expert wins the lottery with probability ~1 no matter how many rows precede or oppose it. Its verdict is retained with fidelity (1-eps) ~ .70-.77 and is charged only a very mild opposition tax rho_opp per minority row beyond the first. Scanning extra rows costs the authority limb NOTHING: anchor-follow is ~.86 alike on one-row, three-row and six-row displays and with 1-5 opponents.

**(5) Pre-attentive NULL-VERDICT CAPTURE is the DOMINANT mode of responding.** Scandal is a property of the *expert*, not of the disagreement, so attention is captured pre-attentively by the least credible expert in the WHOLE display before the subject checks whether it actually discriminates. Subjects commit to whoever captured attention on a clear majority of trials (q ~ .58-.72): the decision to re-scan the display for a speaking cue is effortful and usually declined. Consequence: when the scandalous expert SPEAKS, capture and re-targeting agree, so anchor-follow stays high and flat; when it is MUTE, most response mass goes to a verdictless expert and the subject guesses, so every anchor-silent cell is pulled toward chance.

**(6) The fallback lottery is expensive in sheer list length.** When no scandal is present the subject must hold the whole list in view to run the lottery over ordinary rows, and the thread is lost quickly: retained evidence is multiplied by rho_len ~ .87-.95 per discriminating row beyond the second — a penalty the authority limb never pays, because the length exponent is gated by (1 - A_speak).

**(7) Degeneracy and stationarity.** If two printed validities are within .002 the experts cannot be individuated, salience has no referent, and choice collapses to a graded unweighted row tally (exact coin flip on 1-1 splits). No feedback is given, so the rule is stationary across the block.

**Rationale:** MINIMAL DIFF: `predict` and `policy` are re-emitted VERBATIM from the running-best iter-3 base. Only TWO parameter ranges change — exactly the critic's primary and secondary knobs, both previously untouched in any iteration, so there is zero oscillation risk.

(1) **lam [0.05, 0.20] -> [0.00, 0.10]** (PRIMARY). lam is the free-standing positional tilt; in the salience equation it enters as `lam * j_all` OUTSIDE the scandal gate, so it only ever arbitrates among NON-scandalous rows. That confines its effect precisely to the anchor-silent family where every remaining residual lives, and it resolves the one structural tension the critic identified — Exp 13 (early .68 lone dissenter, model .483 vs obs .546, too LOW) versus Exp 17 (late .66 lone dissenter, model .533 vs obs .493, too HIGH) — in the SAME direction for both, which no other parameter can do. Sign check: weaker recency raises the early dissenter's lottery share (Exp 13 up toward .546), lowers the late dissenter's (Exp 17 down toward .493), lowers last-read-row follow (Exp 7 .447 -> toward obs .438), and raises Exp 1 (whose late rows are its least valid; model .265 vs obs .282). Exps 15/16/19/20 are structurally immune because a single scandal row owns essentially all salience there via the kappa bonus; Exps 10/11/12 place their scandal row at j=0 where lam*j contributes nothing.

(2) **mu [0.60, 1.00] -> [0.90, 1.40]** (SECONDARY). mu rides INSIDE the scandal gate (`g * (kappa + mu*j)`), so it arbitrates only BETWEEN co-scandalous experts. Exps 8 and 9 are exactly the alternating .955/.545 and .958/.546 designs in which several near-chance rows compete and the metric rewards the LATEST of them; raising mu buys back the -.018 and -.024 those cells lost last round. Every single-anchor experiment (15/16/17/19/20) has exactly one row with g>0 and is therefore mathematically unaffected by mu. This also partially compensates the lam cut in the one place the corpus genuinely demands a recency effect, making the theoretical claim sharper rather than weaker: recency is a property of scandal competition, not of ordinary rows.

I deliberately did NOT take the critic's optional third step (a conflict-specific leak for Exp 2) and did NOT touch q_capture, theta/tau, eps, rho_opp or rho_len. The critic explicitly flagged q as being at its optimum and warned that any cut to authority-limb fidelity would drag Exps 15/19/20 off their currently excellent .86-.87; stacking a third structural edit risks over-shooting and tripping the gate. Architecture, capture branch, flat-in-length authority limb, degeneracy clause and stationarity are all unchanged.

**Parameters:**
  - `theta`: `[0.562, 0.572]`
  - `tau`: `[0.003, 0.006]`
  - `kappa`: `[5.0, 7.5]`
  - `mu`: `[0.90, 1.40]`
  - `gamma`: `[0.30, 0.46]`
  - `lam`: `[0.00, 0.10]`
  - `q_capture`: `[0.58, 0.72]`
  - `eps`: `[0.23, 0.30]`
  - `rho_opp`: `[0.90, 0.99]`
  - `rho_len`: `[0.87, 0.95]`
  - `tally_kappa`: `[0.7, 1.3]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # AIG: Authority-Inversion with null-verdict capture and a single-cue fallback lottery
    import numpy as np

    # ---------- parse the (2, n_features) stimulus robustly ----------
    a = None
    b = None
    if isinstance(state, dict):
        if 'option_a_ratings' in state and 'option_b_ratings' in state:
            a = np.asarray(state['option_a_ratings'], dtype=float).ravel()
            b = np.asarray(state['option_b_ratings'], dtype=float).ravel()
        else:
            vals = list(state.values())
            if len(vals) >= 2:
                a = np.asarray(vals[0], dtype=float).ravel()
                b = np.asarray(vals[1], dtype=float).ravel()
    if a is None or b is None:
        arr = np.asarray(state, dtype=float)
        if arr.ndim == 2 and arr.shape[0] == 2:
            a, b = arr[0].ravel(), arr[1].ravel()
        elif arr.ndim == 2 and arr.shape[1] == 2:
            a, b = arr[:, 0].ravel(), arr[:, 1].ravel()
        elif arr.ndim == 1:
            h = arr.shape[0] // 2
            a, b = arr[:h], arr[h:2 * h]
        else:
            f = np.asarray(arr, dtype=float).reshape(2, -1)
            a, b = f[0].ravel(), f[1].ravel()

    n = int(min(np.size(a), np.size(b)))
    if n == 0:
        return np.array([0.5, 0.5])
    a = np.asarray(a, dtype=float)[:n]
    b = np.asarray(b, dtype=float)[:n]

    # ---------- printed validities ----------
    v_raw = parameters.get('validities', None) if hasattr(parameters, 'get') else None
    if v_raw is None:
        v = np.linspace(0.95, 0.55, n)
    else:
        v = np.asarray(v_raw, dtype=float).ravel()
        if v.size < n:
            v = np.concatenate([v, np.full(n - v.size, 0.55)])
        v = v[:n]
    v = np.clip(v, 0.5, 0.9995)

    # ---------- parameters ----------
    theta = float(parameters['theta'])
    tau = float(max(float(parameters['tau']), 1e-4))
    kappa = float(parameters['kappa'])
    mu = float(parameters['mu'])
    gamma = float(parameters['gamma'])
    lam = float(parameters['lam'])
    q = float(np.clip(float(parameters['q_capture']), 0.0, 1.0))
    eps = float(np.clip(float(parameters['eps']), 0.0, 1.0))
    rho_opp = float(np.clip(float(parameters['rho_opp']), 1e-6, 1.0))
    rho_len = float(np.clip(float(parameters['rho_len']), 1e-6, 1.0))
    tkap = float(parameters['tally_kappa'])

    def sig(x):
        x = float(np.clip(x, -60.0, 60.0))
        return 1.0 / (1.0 + np.exp(-x))

    def softmax(logv):
        logv = np.asarray(logv, dtype=float)
        if logv.size == 0:
            return logv
        z = logv - np.max(logv)
        e = np.exp(np.clip(z, -700.0, 0.0))
        s = float(np.sum(e))
        if (not np.isfinite(s)) or s <= 0.0:
            return np.full(logv.size, 1.0 / logv.size)
        return e / s

    # ---------- discriminating rows ----------
    diff = a - b
    disc = np.nonzero(diff != 0.0)[0]
    m = int(disc.size)
    if m == 0:
        return np.array([0.5, 0.5])
    signs = np.sign(diff[disc])
    fav_a = signs > 0.0
    n_a = float(np.sum(fav_a))
    n_b = float(m) - n_a
    k_conf = int(min(n_a, n_b))

    # ---------- label-degeneracy gate ----------
    degenerate = False
    if n > 1:
        vs = np.sort(v)
        if bool(np.any(np.diff(vs) < 2.0e-3)):
            degenerate = True

    if degenerate:
        keep = (1.0 - eps) * (rho_len ** max(m - 2, 0))
        keep = float(min(max(keep, 0.0), 1.0))
        p_core = sig(tkap * (n_a - n_b))
        p_a = 0.5 + keep * (p_core - 0.5)
        p_a = float(min(max(p_a, 1e-9), 1.0 - 1e-9))
        p = np.array([p_a, 1.0 - p_a], dtype=float)
        return p / p.sum()

    # ---------- salience: absolute scandal bonus + weak doubt/recency tilt ----------
    j_all = np.arange(n, dtype=float)
    zz = np.clip((theta - v) / tau, -60.0, 60.0)
    g = 1.0 / (1.0 + np.exp(-zz))                      # scandal (near-chance) value
    logu = g * (kappa + mu * j_all) \
        + gamma * np.log(np.clip(1.0 - v, 1e-12, None)) \
        + lam * j_all

    # global salience over ALL rows (capture is pre-attentive)
    s_all = softmax(logu)
    # restricted lottery over rows that actually speak
    t_disc = softmax(logu[disc])

    # (i) capture branch: go with whoever grabbed attention; guess if it is silent
    mask = np.zeros(n, dtype=bool)
    mask[disc] = True
    s_silent = float(np.sum(s_all[~mask]))
    s_d = s_all[disc]
    p_glob_a = float(np.sum(s_d[fav_a])) + 0.5 * s_silent

    # (ii) re-targeted branch: lottery restricted to discriminating rows
    p_rest_a = float(np.sum(t_disc[fav_a]))

    core_a = q * p_glob_a + (1.0 - q) * p_rest_a

    # ---------- fidelity: authority limb is free in length, fallback is not ----------
    A_speak = float(np.clip(np.max(g[disc]), 0.0, 1.0))
    len_pen = rho_len ** ((1.0 - A_speak) * float(max(m - 2, 0)))
    opp_pen = rho_opp ** float(max(k_conf - 1, 0))
    fid = (1.0 - eps) * opp_pen * len_pen
    fid = float(min(max(fid, 0.0), 1.0))

    p_a = 0.5 + fid * (core_a - 0.5)
    p_a = float(min(max(p_a, 1e-9), 1.0 - 1e-9))
    p = np.array([p_a, 1.0 - p_a], dtype=float)
    return p / p.sum()
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if not np.isfinite(s) or s <= 0.0:
        return int(np.random.randint(len(probs)))
    probs = probs / s
    return int(np.random.choice(len(probs), p=probs))
```
