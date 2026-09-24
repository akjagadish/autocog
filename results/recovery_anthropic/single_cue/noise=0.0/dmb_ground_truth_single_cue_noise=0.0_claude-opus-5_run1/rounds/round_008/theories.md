# Round 8 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_7` — KILLED ✗

**Description:** **MDSC-NV (sharpened) — Maximum-Doubt Single-Cue selection with Null-Verdict capture.**

Four claims; the mechanism family is unchanged from the accepted base, but one quantitative commitment is sharpened.

**(1) One-reason choice by MAXIMUM DOUBT (no summation).** On each trial the subject settles A-vs-B with the verdict of a SINGLE expert. Which expert is consulted is decided by *doubt*: a disagreement voiced by a near-chance expert (v≈.53) is a strange, newsworthy fact that has to be adjudicated, whereas a disagreement voiced by a near-certain expert (v≈.97) is discounted as a mere echo of the impression the subject already has. Salience is log u_j = delta*log(1-v_j) + lam*j and the consulted row is drawn from softmax(u). **The doubt gain is LARGER than previously assumed (delta ≈ 13-17, not 8-12): the doubt rule is essentially argmax, not a graded share.** This is the sharpened commitment. It matters wherever two moderately-doubtful rows are within ~0.4 log-validity of one another (e.g. v=.55 vs v=.70 rows two columns apart): a share-based rule lets the second-most-doubtful row steal ~18% of the verdict and blunts the choice, whereas the near-argmax rule leaves the display fully decided by the single most doubtful row. Coalitions still buy nothing: five agreeing rows never outvote one more-doubtful row, which is why choice is systematically anti-validity in every design where the printed labels are usable.

**(2) Reading position is a WEAK tie-breaker.** lam = pos_ratio*delta with pos_ratio ≈ 0.09-0.14 — an order of magnitude below the doubt term's dynamic range. Position therefore only arbitrates between rows whose stated validities are (nearly) *equal* — exactly the .955/.950/.945 and .958/.954/.950 designs, where it makes the last low-validity row win. Whenever validities genuinely differ, doubt beats recency: a low-validity EARLY column defeats a late .93/.99 column. Because delta is raised and pos_ratio held fixed, the *ratio* of doubt-range to positional gain is preserved, so all orderings in the corpus are unchanged.

**(3) NULL-VERDICT CAPTURE.** Doubt is a property of the *expert*, not of the disagreement, so attention is captured pre-attentively by the most doubtful expert **in the whole display**, before the subject checks whether that expert actually discriminates. With probability q (≈ .41, slightly higher than previously estimated) the subject simply goes with whoever captured attention; if that expert gave both products the same rating it returns no verdict, the subject has nothing to adjudicate and guesses. With probability 1-q the subject re-targets deliberately, restricting the doubt competition to the rows that differ. Sharp prediction: *holding the discriminating rows fixed, adding agreement on a MORE doubtful (lower-validity) row pushes choice toward chance*, whereas adding agreement on a high-validity row does nothing. Displays in which only two high/mid-validity rows disagree while the display's most doubtful expert stays silent are therefore near chance — no compensatory or one-reason model predicts that non-discriminating rows matter at all.

**(4) LABEL-DEGENERACY GATE (all-or-none) and LOAD LAPSE.** The doubt rule presupposes that each printed validity picks out one expert. If any label is printed twice (tolerance .002 — it must NOT fire for the .004/.005-gap designs) the experts are interchangeable, the doubt ordering is unavailable, and choice collapses to a graded unweighted row tally (exact coin flip on 1-1 splits). Independently, the more rows disagree the more often the subject loses the thread: p(guess) = 1-(1-eps0)*rho^(m-2), which compresses conflict displays toward chance and keeps even unanimous 6-row displays well below ceiling. Individual subjects differ mainly in how leaky (eps0, rho) and how capture-prone (q) they are, not in the doubt rule itself.

No feedback is given, so the rule is stationary across the block.

**Rationale:** MINIMAL DIFF: `predict`/`policy` are byte-identical to the accepted base; only sampling ranges changed. Two knobs move (doubt_gain 8-12 -> 13-17, capture .30-.40 -> .33-.49) and three are widened around their old centres (eps0, rho, tally_kappa) for between-subject dispersion. Every change was chosen after hand-simulating the residual experiments.

Why doubt_gain up (the critic's Exp4 suggestion, but the real payoff is Exp2). The single largest residual is Exp2 (contrast .251 vs .174 = 51% of the squared loss). I traced it trial-by-trial. The contrast is p_tally(lopsided conflict trials) - p_ttb(tally-tie trials). The tie trial A=[0,0,1,1,0,0]/B=[0,0,0,0,1,1] has disc rows {2(A,v=.55), 3(A,v=.87), 4(B,v=.70), 5(B,v=.78)}. At delta=10 the doubt leader col2 only holds t=.82 of the softmax because col4 (v=.70, two columns later) is just 1.65 log-units behind, so p_core_a=.816 and the model returns .658 where the data need ~.72. At delta=15 the same gap becomes 2.48 (col2 holds .92), p rises to .709, and p_ttb_tie rises by .026, dropping the contrast with essentially no side effects: the doubt ORDERING is unchanged in all ten designs (I re-checked Exp1 col4>col3 gap 3.77+lam, Exp8/9 col5>col3>col1 gaps ~2.5-3.7, Exp10 col0 still leader), because lam = pos_ratio*delta scales with delta so the doubt/position ratio is preserved. The same edit makes the v=.53 column strictly decisive in Exp4's two tally-tie designs, moving Exp4 from .188 toward .15 exactly as the critic asked.

Why capture up only to ~.41. The other half of Exp2's residual is the 1-vs-1 tie A=[0,1,0,0,0,0]/B=[1,0,0,0,0,0], where the display's doubt leader (col2, v=.55) is SILENT: only null-verdict capture can move it off .25 toward the ~.34-.43 the data require. dContrast/dq = -.19, so q=.41 buys ~.026. I explicitly rejected the critic-adjacent temptation to push q to .7-.85: hand-simulation shows q=.85 lifts Exp1 to .33 (real .282), Exp3 to .32 (real .286) and Exp4 to .22 (real .150) - a net LOSS of ~.007 in squared error versus a .006 gain on Exp2. I also tested and rejected an (n-m)-scaled capture (q_eff = 1-(1-q)^(silent rows)): it fixes Exp2 outright but destroys Exp5 (.66), Exp8 (.72) and Exp9 (.75). q=.41 is the local optimum where the Exp2 gain (~.004 squared) still exceeds the pooled Exp5/8/9 cost (~.0015).

What I deliberately did NOT do. (a) The critic suggested raising pos_ratio for Exp8/9. Hand-simulation shows this is counter-productive: at pos_ratio=.20 the Exp2 tie trial's doubt mass splits between col2 and col4, p_ttb collapses to .47 and the Exp2 contrast RISES to .25; meanwhile Exp8/9 are already ordering-decisive (col5 leads by 2.5-3.7 log units), so their residual is pure lapse ceiling, not ordering. (b) The critic suggested shaving eps0 for Exp9. Global lapse reduction raises Exp9 (+) but simultaneously pushes Exp5 (dominance, real .743) to ~.79, Exp8 above .77, and Exp1/Exp3/Exp10 further below chance - a net loss. Exp9's ~.05 shortfall is a genuine structural ceiling of this family, and the corpus-wide lapse level is pinned by Exp5/Exp10; I left it alone rather than trade three fits for one.

Hand-computed predictions after the edit (real in parentheses): .283 (.282), .226 (.174), .285 (.286), .156 (.150), .738 (.743), .500 (.507), ~.45 (.438), .761 (.770), .779 (.819), ~.415 (.418). Sum of squared deviations drops from ~.0113 to ~.0067, i.e. an expected aggregate loss near .026 versus the .0322 floor. Widening eps0/rho/capture around unchanged centres leaves those point estimates intact while roughly doubling simulated between-subject spread, addressing the variance criticism.

**Parameters:**
  - `doubt_gain`: `[13.0, 17.0]`
  - `pos_ratio`: `[0.09, 0.14]`
  - `capture`: `[0.33, 0.49]`
  - `eps0`: `[0.19, 0.29]`
  - `rho`: `[0.77, 0.85]`
  - `tally_kappa`: `[1.00, 1.40]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # MDSC-NV: Maximum-Doubt Single-Cue selection with Null-Verdict capture
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

    # ---------- communicated validities (the printed labels) ----------
    v_raw = parameters.get('validities', None) if hasattr(parameters, 'get') else None
    if v_raw is None:
        v = np.linspace(0.95, 0.55, n)
    else:
        v = np.asarray(v_raw, dtype=float).ravel()
        if v.size < n:
            v = np.concatenate([v, np.full(n - v.size, 0.55)])
        v = v[:n]
    v = np.clip(v, 0.5, 0.9975)

    # ---------- parameters ----------
    delta = float(parameters['doubt_gain'])
    lam = float(parameters['pos_ratio']) * delta        # weak positional gain
    q = float(np.clip(float(parameters['capture']), 0.0, 1.0))
    eps0 = float(np.clip(float(parameters['eps0']), 0.0, 1.0))
    rho = float(np.clip(float(parameters['rho']), 1e-6, 1.0))
    kap = float(parameters['tally_kappa'])

    # ---------- discriminating rows ----------
    diff = a - b
    disc = np.nonzero(diff != 0.0)[0]
    m = int(disc.size)
    if m == 0:
        return np.array([0.5, 0.5])
    fav_a = diff[disc] > 0.0
    n_a = float(np.sum(fav_a))
    n_b = float(m) - n_a

    # ---------- label-degeneracy gate (all-or-none, exact repetition) ----------
    degenerate = False
    if n > 1:
        vs = np.sort(v)
        if bool(np.any(np.diff(vs) < 2.0e-3)):
            degenerate = True

    if degenerate:
        # experts cannot be individuated -> unweighted graded row tally
        z = float(np.clip(kap * (n_a - n_b), -60.0, 60.0))
        p_core_a = 1.0 / (1.0 + np.exp(-z))
    else:
        # doubt salience over ALL rows
        logu = delta * np.log(np.clip(1.0 - v, 1e-12, None)) + lam * np.arange(n, dtype=float)

        lg = logu - np.max(logu)
        s = np.exp(lg)
        ssum = float(np.sum(s))
        s = s / ssum if (np.isfinite(ssum) and ssum > 0.0) else np.full(n, 1.0 / n)

        ld = logu[disc]
        ld = ld - np.max(ld)
        t = np.exp(ld)
        tsum = float(np.sum(t))
        t = t / tsum if (np.isfinite(tsum) and tsum > 0.0) else np.full(m, 1.0 / m)

        # (i) deliberate re-targeting: doubt competition among disagreeing rows
        p_rest_a = float(np.sum(t[fav_a]))

        # (ii) null-verdict capture: attention grabbed by the most doubtful
        #      expert in the whole display; if it is silent -> guess
        mask = np.zeros(n, dtype=bool)
        mask[disc] = True
        s_nondisc = float(np.sum(s[~mask]))
        s_disc = s[disc]
        p_glob_a = float(np.sum(s_disc[fav_a])) + 0.5 * s_nondisc

        p_core_a = q * p_glob_a + (1.0 - q) * p_rest_a

    # ---------- load-dependent lapse ----------
    expo = max(m - 2, 0)
    g = 1.0 - (1.0 - eps0) * (rho ** expo)
    g = float(min(max(g, 0.0), 1.0))

    p_a = 0.5 * g + (1.0 - g) * p_core_a
    p_a = float(min(max(p_a, 1e-9), 1.0 - 1e-9))
    p = np.array([p_a, 1.0 - p_a], dtype=float)
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

### `pi_11` → slot 1 (via `new_theory`)

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
