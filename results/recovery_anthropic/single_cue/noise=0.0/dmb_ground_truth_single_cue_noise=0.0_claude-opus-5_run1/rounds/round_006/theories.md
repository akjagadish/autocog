# Round 6 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_7` — SURVIVED ✓

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


### slot 2 — `pi_8` — KILLED ✗

**Description:** **DTV-RG/CL — Doubt-Triggered one-reason with Rank-Geometric weighting, under a two-component (load + conflict) leak and a DECISIVE degenerate-label tally.**

The evidence rule is unchanged from the accepted base; two claims about *how the rule degrades* are sharpened.

**(1) Rank-geometric evidence, decided from the LEAST valid dissenting expert.** On each trial the subject lines the *discriminating* rows up in order of printed validity, least valid first, and sums their verdicts with a geometric discount: E = Σ_r s_r · φ^r (r = 0 for the least valid discriminating expert, s_r = ±1 its verdict); p(A) = σ(βE). Only the ORDER of the printed labels enters, never the size of the gaps, which is why the .955/.950/.945 and .958/.954/.950 families are still resolved almost deterministically in favour of the strictly least valid dissenter, and why coalitions of more valid experts normally buy nothing (though a large enough coalition can still overturn one doubtful voice).

**(2) The discount rate is set by the ABSOLUTE doubtfulness of the decisive voice.** One-reason choice is a *reaction to a near-chance expert*: φ = φ_lo + (φ_hi − φ_lo)·σ((v₀ − v_c)/w), with v₀ the validity of the least valid DISCRIMINATING expert, φ_lo ≈ .25 (take-the-least), φ_hi ≈ .88 (near tally), v_c ≈ .575, w ≈ .015. Below the trigger the display is settled by one strange dissent; above it nothing is newsworthy and the rows are weighted nearly equally.

**(3) No null-verdict capture.** Non-discriminating rows are strictly irrelevant whatever their validity; all pull toward chance comes from the leak in (4).

**(4) NEW — the leak has TWO separable sources, and dominance displays leak only through the first.** Losing the thread is driven (i) by sheer display load — how many rows have to be held while the ordering is run — and (ii) independently by *opposition*: every additional row that argues the other way adds a fresh occasion to lose track of which side one was on. p(guess) = 1 − (1−ε₀)·ρ_m^max(m−2,0)·ρ_c^min(n_A,n_B), with ρ_m ≈ .79 (load) and ρ_c ≈ .97 (opposition). The two terms are close to orthogonal in the corpus: unanimous (dominance) displays have min(n_A,n_B)=0 and therefore leak only by load, so they stay well above the balanced-conflict displays of the same size, whereas a 3-vs-3 display of six rows is markedly noisier than a 6-vs-0 display of six rows. This is a direct, testable clash both with a pure-load leak (which charges a 6-0 display exactly what it charges a 3-3 display) and with a pure-opposition leak (which would make unanimous displays near-perfect).

**(5) NEW — when labels are degenerate the fallback tally is DECISIVE, not tepid.** The doubt ordering presupposes that each printed validity picks out one expert. If any label is repeated (gap < .002) the columns are interchangeable and choice collapses to an unweighted row tally — but that tally is used with a steep slope (κ ≈ 1.6, not ≈ 0.85): having nothing else to go on, the subject commits hard to the vote margin. Hence 1-vs-1 splits under duplicated labels are an exact coin flip (no anti-validity, no recency), while 2-vs-0 and 4-vs-0 splits under duplicated labels are chosen at ~.83 and ~.95 before the leak — which is what keeps unanimous displays high and what makes a 2-vs-1 display under duplicated labels clearly favour the majority rather than hovering near chance.

No feedback is given, so the rule is stationary across the block.

**Rationale:** MINIMAL DIFF on the accepted base: (i) the lapse line becomes g = 1-(1-eps0)*rho_m^max(m-2,0)*rho_c^min(nA,nB) exactly as the critic prescribed (one extra factor, one extra parameter, nothing else in `predict` touched); (ii) the ranges of eps0/rho_m/rho_c are re-fit; (iii) tally_kappa is raised from [0.60,1.10] to [1.30,1.90]. The evidence rule, the rank-geometric sum, the phi(v0) doubt trigger and the degeneracy gate are untouched, per the critic's instruction.

Diagnosis behind the kappa move (the critic's residuals had a hidden cause). I back-solved the base model trial-by-trial and found that Exp 5 and Exp 7 are DEGENERATE-LABEL designs: with the base parameters the degenerate branch reproduces the reported values to three decimals (Exp5 hand-value .699 vs reported .6987; Exp7 .455 vs .4564), whereas the non-degenerate branch would have given .738 / much higher. So the two experiments the critic flagged as 'too noisy' are NOT governed by the lapse at all in the way assumed - they are governed by tally_kappa, which at ~0.85 makes a 2-0 unanimous display only sigmoid(1.7)=.85 confident before any lapse. Raising kappa to ~1.6 lifts Exp5 from .699 to ~.745 (real .743) and simultaneously pulls Exp7 from .456 to ~.434 (real .438) - it fixes BOTH of the critic's structured residuals with one parameter, and it cannot disturb Exp6 (whose metric trials are 1-1 splits, exactly .5 for any kappa).

Why I still adopt the conflict factor but keep it mild (rho_c ~ .965 rather than .72-.88). Exp 4 pins the 2-cue conflict retention at q(2,1)=P*rho_c=.733 (real .150 = .5-.477q) and Exp 3 pins the 4-cue/2-opposed retention at q(4,2)=P*rho_m^2*rho_c^2=.455; together these force rho_m^2*rho_c=.62, so a strong conflict term (rho_c<.9) can only be bought by pushing rho_m toward 1, which then makes 5- and 6-row displays too clean and blows up Exp 2 (+.03). A mild rho_c is what the data support: it buys the dominance-versus-conflict dissociation the critic wanted (min(nA,nB)=0 displays leak only by load) without breaking the two experiments that pin the leak. Hand-computed values with the midpoint parameters (real in brackets): Exp1 .280 [.282], Exp2 .191 [.174], Exp3 .288 [.286], Exp4 .150 [.150], Exp5 .745 [.743], Exp6 .50 [.507], Exp7 .434 [.438], Exp8 .768 [.770], Exp9 .753 [.819], Exp10 .448 [.418], Exp11 .625 [.593], Exp12 .267 [.261]; SSE ~.0066 versus ~.0090 for the accepted base, with the gains concentrated exactly where the critic diagnosed failure (Exp5 -.044 -> +.002, Exp7 +.018 -> -.004) and the four discriminating cells (Exp1, Exp3, Exp8, Exp12) all moving by <0.01 as demanded. Exp9 is left alone, per the critic's advice that it is irreducible (its near-twin Exp8 is hit exactly while real Exp9 sits .05 higher). I also verified that the critic's expectation for Exp10 is sign-inverted - its model value .446 is BELOW .5, so extra conflict noise would push it up toward .5, away from the real .418; keeping rho_c mild avoids that regression.

**Parameters:**
  - `phi_lo`: `[0.21, 0.29]`
  - `phi_hi`: `[0.84, 0.92]`
  - `v_crit`: `[0.570, 0.580]`
  - `v_width`: `[0.013, 0.018]`
  - `beta`: `[4.5, 5.5]`
  - `eps0`: `[0.20, 0.28]`
  - `rho_m`: `[0.76, 0.82]`
  - `rho_c`: `[0.93, 1.00]`
  - `tally_kappa`: `[1.30, 1.90]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # DTV-RG/CL: Doubt-Triggered one-reason with Rank-Geometric weighting,
    # two-component (load + conflict) leak, decisive degenerate-label tally
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
    v = np.clip(v, 0.5, 0.9995)

    # ---------- parameters ----------
    phi_lo = float(np.clip(float(parameters['phi_lo']), 1e-3, 0.999))
    phi_hi = float(np.clip(float(parameters['phi_hi']), 1e-3, 0.999))
    v_crit = float(parameters['v_crit'])
    v_width = float(max(float(parameters['v_width']), 1e-4))
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

    # ---------- label-degeneracy gate (repeated printed labels) ----------
    degenerate = False
    if n > 1:
        vs = np.sort(v)
        if bool(np.any(np.diff(vs) < 2.0e-3)):
            degenerate = True

    if degenerate:
        # experts cannot be individuated -> decisive unweighted row tally
        p_core_a = sig(kappa * (n_a - n_b))
    else:
        # order discriminating rows by printed validity (ascending);
        # ties broken by reading position (later read = more decisive)
        order = np.lexsort((-disc.astype(float), v[disc]))
        idx_sorted = disc[order]
        s_sorted = signs[order]

        v0 = float(v[idx_sorted[0]])
        # doubt trigger: only a near-chance dissenter buys one-reason choice
        phi = phi_lo + (phi_hi - phi_lo) * sig((v0 - v_crit) / v_width)
        phi = float(np.clip(phi, 1e-4, 0.9999))

        r = np.arange(m, dtype=float)
        w = np.power(phi, r)
        E = float(np.sum(s_sorted * w))
        p_core_a = sig(beta * E)

    # ---------- two-component leak: display load x opposition ----------
    expo_m = max(m - 2, 0)
    k_conf = int(min(n_a, n_b))
    g = 1.0 - (1.0 - eps0) * (rho_m ** expo_m) * (rho_c ** k_conf)
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


## Replacement

### `pi_9` → slot 2 (via `new_theory`)

**Description:** **NAVA-HLp — Normalized Anti-Validity Accumulation with a HALF-LOUDNESS load lapse and a PARTIAL unanimity discount.**

Six claims. The evidence rule is unchanged from the accepted base: graded, compensatory, anti-validity, divisively normalized, with no one-reason step, no argmax, no verdict capture and no doubt trigger. Only the *price charged for holding rival voices* is revised, and only partially.

**(1) Every dissenting row speaks, and doubt is what makes it loud.** A printed validity close to chance makes a disagreement *newsworthy* (it demands adjudication); a printed validity close to certainty makes the same disagreement a mere echo of the impression the subject already has. Each row carries weight w_j = (1 − v_j)^γ · exp(λ·j): doubt raised to a moderate power γ ≈ 1.9–2.5, times a modest reading-position gain λ ≈ 0.28–0.45. All discriminating rows contribute simultaneously, so coalitions matter monotonically, while the global direction of choice stays systematically anti-validity in every design whose labels are usable.

**(2) Divisive normalization relative to the loudest dissent.** Weights enter only as ratios to the loudest discriminating voice, ŵ_j = w_j / max_{k∈disc} w_k, and E = Σ_{j∈disc} s_j ŵ_j / (σ + Σ_{j∈disc} ŵ_j), p(A) = σ(βE). Absolute scale is discarded (dominance displays decided only by high-validity experts are still resolved confidently) while the denominator grows with each speaking row, so a coalition erodes a doubtful dissenter's lead toward, but rarely far past, chance.

**(3) The displacement leak is a DOUBT GRADIENT.** Doubt is a property of the expert, not of the disagreement, so the most doubtful expert *in the whole display* competes for attention before the subject checks whether it discriminates. What costs accuracy is the *doubt gradient* between the loudest silent expert and the loudest speaking one: attention is pulled off the adjudication only when the silent expert is appreciably MORE doubtful than the expert currently carrying the argument. Formally d = clip((v_loudest-disc − v_loudest-silent)/Δ, 0, 1) · w_silent,max/(w_silent,max + w_disc,max) with fixed Δ = .05, and the leak multiplies the retention term by (1 − κ·d), κ ≈ .40–.95. Silent rows never contribute evidence; the term is purely subtractive and can only shrink |p − .5|.

**(4) Losing the thread is charged HALF PER ROW and HALF PER LOUD VOICE.** What has to be held in mind while the comparison is run is neither the raw number of disagreeing rows (which over-charges a display in which one towering doubtful voice drowns out five whispers) nor purely the effective number of loud voices (which under-charges a long unanimous list that still has to be scanned). Both costs are real and roughly equally weighted: L = ½·(m − 2) + ½·c·(Σ_{j∈disc} ŵ_j − 1), and p(guess) = 1 − (1−ε₀)·ρ^max(L,0)·(1 − κ·d) with ρ ≈ .64–.80 and floor ε₀ ≈ .07–.15.

**(5) NEW — unanimity is CHEAPER BUT NOT FREE.** The loud-voice component of the load is an *adjudication* cost: it is the price of holding rival loud voices against one another. When every speaking row agrees there is no adjudication, but the rows still have to be scanned and their loudness still occupies the comparison, so the cost is discounted rather than waived: c = 1 when both options are favoured by at least one row, and c = c₀ ≈ .45–.75 when the display is unanimous. The row-scan half, ½·(m − 2), is charged in full regardless, so a 6-vs-0 dominance display remains well below ceiling while a 3-vs-3 display of the same size is markedly noisier. This is the model's sharp prediction about dominance: accuracy on unanimous displays falls with list length but strictly more slowly than on conflict displays of matched length and matched weight concentration.

**(6) Degenerate printed labels collapse the weighting.** If any two printed validities are within .002 the experts cannot be individuated; all weights become equal and the same normalized rule reduces to an unweighted tally, E = (n_A − n_B)/(σ + m) with the same β and σ — exact chance on 1-1 splits, no anti-validity and no recency signature. The displacement leak is switched off, because no row is 'the doubtful one'.

No feedback is given, so the rule is stationary across the block.

**Rationale:** **Minimal diff: two added lines in the load exponent plus one new bounded parameter.** The accepted base (NAVA-HL, loss 0.0486) is re-emitted verbatim — same evidence rule (γ, λ, σ, β, max-normalized divisive form), same ±.002 degeneracy gate, same doubt-gradient displacement leak with fixed Δ=.05 and κ∈[.40,.95], same half-row/half-loud-voice lapse shape, same `policy`. Changed lines:

* compute `n_a`, `n_b` from `signs` (they were already implicit) and `c = 1.0 if min(n_a,n_b) >= 1 else c0`;
* `expo = 0.5*(m-2) + 0.5*(Σŵ - 1)` → `expo = 0.5*(m-2) + 0.5*c*(Σŵ - 1)`;
* one new parameter `c0 ∈ [0.48, 0.75]`.

**Why exactly this, and why c₀ is NOT 0.** The loop has now bracketed the answer with two measurements on the same cell: charging the loud-voice half in full on unanimous displays (c=1, the accepted base) gives Exp 5 = .701; waiving it entirely (c=0, iter 8) gives .822; the human value is .743. Both extremes sit outside the hard guardrail [.73,.78] — the base 4 points low, iter 8 8 points high — so neither a revert nor a repeat of the binary gate is admissible. Linear interpolation between the two bracketing points puts the target at c ≈ .65, i.e. roughly a one-third discount of the adjudication cost on unanimous displays. I therefore place c₀ in [0.48, 0.75] (mean ≈ .62, predicted Exp 5 ≈ .745–.755) rather than fixing it at the critic's suggested .50, which by the same interpolation would land ≈.762 — inside the band but hugging its upper edge, and the guardrail has now failed on BOTH sides. A single bounded parameter centred on the interpolated optimum is the smaller risk; the critic explicitly authorised one bounded c0 as the alternative to the fixed constant.

**Cognitive reading (unchanged family).** The loud-voice term was always an *adjudication* cost: the price of holding rival loud voices against one another. When every speaking row agrees there is nothing to adjudicate, but the rows still have to be scanned and their loudness still occupies the comparison — so the cost is discounted, not waived. The row-scan half, ½(m−2), remains charged in full on unanimous displays, which is what keeps a 6-vs-0 list away from ceiling.

**Predicted trace.** Exp 5's dominance cells lose roughly 0.4·(Σŵ−1)/2 ≈ 0.3–0.6 of exponent relative to the base: .701 → ≈.745 (real .743), back inside [.73,.78] from the correct side. Every conflict-based metric (Exp 1, 2, 3, 4, 7, 9, 10, 11, 12, 13, 14) is arithmetically **bit-identical** to the accepted base, because min(n_A,n_B) ≥ 1 on every trial those metrics retain, so c = 1 and the exponent is unchanged; any movement there is pure simulation noise. Exp 6 and Exp 8 contain unanimous trials but their metrics filter them out (Exp 6 requires exactly 2 opposed rows; Exp 8 requires m ≥ 2 with the tally not agreeing with the salient row), so <1 point movement. Expected gain ≈ the entire Exp 5 residual (.042² ≈ .0018 of squared error) with essentially no offsetting risk.

**Deliberate omissions, learned from this loop's rejections.** No ρ reduction bundled in (the critic's conditional second step is explicitly conditional on Exp 5 landing ≥ .77, and bundling is what sank iters 2, 3 and 6); no re-centring of β, λ, σ, ε₀ or κ; no coalition-erosion term for the Exp 13/14 sign conflict (rejected twice); no mechanism-family change (no argmax, no one-reason step, no verdict capture); the ±.002 degeneracy gate and sign convention untouched. I also do not read the ±.03–.06 swings on Exp 2/3/4/6/12/13 between iters 7 and 8 as signal — with between-subject variances of .004–.016 those are within noise, and the edit above leaves them mathematically unchanged in any case.

**Parameters:**
  - `gamma`: `[1.9, 2.5]`
  - `lam`: `[0.28, 0.45]`
  - `sigma`: `[0.12, 0.30]`
  - `beta`: `[2.8, 3.8]`
  - `eps0`: `[0.07, 0.15]`
  - `rho`: `[0.64, 0.80]`
  - `kappa`: `[0.40, 0.95]`
  - `c0`: `[0.48, 0.75]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # NAVA-HLp: Normalized Anti-Validity Accumulation with doubt-gradient displacement leak,
    # HALF-ROW / HALF-LOUD-VOICE load lapse, and a PARTIAL unanimity discount on the loud-voice half
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
    v = np.clip(v, 0.5, 0.9995)

    # ---------- parameters ----------
    gamma = float(parameters['gamma'])
    lam = float(parameters['lam'])
    sigma = float(max(float(parameters['sigma']), 1e-6))
    beta = float(parameters['beta'])
    eps0 = float(np.clip(float(parameters['eps0']), 0.0, 1.0))
    rho = float(np.clip(float(parameters['rho']), 1e-6, 1.0))
    kappa = float(np.clip(float(parameters['kappa']), 0.0, 1.0))
    c0 = float(np.clip(float(parameters['c0']), 0.0, 1.0))

    DV = 0.05  # fixed doubt-gradient scale (not fitted)

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

    # ---------- label-degeneracy gate (printed labels not individuable) ----------
    degenerate = False
    if n > 1:
        vs = np.sort(v)
        if bool(np.any(np.diff(vs) < 2.0e-3)):
            degenerate = True

    # ---------- doubt x reading-position weights ----------
    if degenerate:
        w = np.ones(n, dtype=float)
    else:
        logw = gamma * np.log(np.clip(1.0 - v, 1e-12, None)) + lam * np.arange(n, dtype=float)
        logw = logw - np.max(logw)
        w = np.exp(np.clip(logw, -700.0, 0.0))

    wd = w[disc]
    wmax = float(np.max(wd))
    if (not np.isfinite(wmax)) or wmax <= 0.0:
        what = np.ones(m, dtype=float)
        wmax = 1.0
    else:
        what = wd / wmax

    # ---------- divisive normalization ----------
    num = float(np.sum(signs * what))
    den = sigma + float(np.sum(what))
    if den <= 0.0:
        E = 0.0
    else:
        E = num / den
    p_core_a = sig(beta * E)

    # ---------- doubt-gradient displacement leak (bounded, evidence-free) ----------
    d = 0.0
    if (not degenerate) and m < n:
        mask = np.ones(n, dtype=bool)
        mask[disc] = False
        sil_idx = np.nonzero(mask)[0]
        if sil_idx.size > 0:
            k_s = int(sil_idx[int(np.argmax(w[sil_idx]))])
            ws = float(w[k_s])
            k_l = int(disc[int(np.argmax(wd))])
            gap = float(v[k_l] - v[k_s])          # positive iff silent row is MORE doubtful
            mult = float(np.clip(gap / DV, 0.0, 1.0))
            if np.isfinite(ws) and (ws + wmax) > 0.0:
                d = mult * ws / (ws + wmax)
    d = float(min(max(d, 0.0), 1.0))

    # ---------- HALF-ROW / HALF-LOUD-VOICE effective load, with partial unanimity discount ----------
    sum_what = float(np.sum(what))
    if not np.isfinite(sum_what):
        sum_what = float(m)
    n_a = float(np.sum(signs > 0.0))
    n_b = float(m) - n_a
    c = 1.0 if min(n_a, n_b) >= 1.0 else c0   # unanimity: adjudication cost discounted, not waived
    expo = 0.5 * float(m - 2) + 0.5 * c * (sum_what - 1.0)
    expo = float(max(expo, 0.0))

    keep = (1.0 - eps0) * (rho ** expo) * (1.0 - kappa * d)
    keep = float(min(max(keep, 0.0), 1.0))
    g = 1.0 - keep

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
