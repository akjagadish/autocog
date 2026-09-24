# Round 5 — Theories

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


### slot 2 — `pi_6` — KILLED ✗

**Description:** SSI-DS+ (Strategy Selection by Label-Code Injectivity, with Distinctive-Cue Summation and Competition-Scaled Interference).

(1) STRATEGY SWITCH. Before the block the subject reads the printed validity list as a naming code. If all labels are distinct printed symbols (0.955 vs 0.950 counts as distinct) the experts are individuable and the subject adopts a WEIGHTED mode; if any label is repeated the experts are interchangeable and the subject adopts an unweighted MAJORITY TALLY mode. The switch is all-or-none at the design level (only a small per-trial slip probability omega), not a graded function of numeric gap size.

(2) WEIGHTED MODE = COMPENSATORY SUMMATION over all discriminating rows: E = sum_j sign(a_j-b_j) * (1-v_j)^gamma * exp(lam*j), rescaled so the strongest discriminating row has magnitude 1, then p(A) = sigmoid(beta*E). Validity enters with INVERTED sign (near-certain experts are redundant echoes of the general impression; near-chance disagreements are the news that must be adjudicated) and position enters as a reading-order recency gain. Coalitions ADD, so several early distinctive rows can outvote a single later row.

(3) TALLY MODE = graded majority p(A) = sigmoid(kappa*(n_A - n_B)): 1-1 splits give exactly 0.50, 2-1 splits ~0.77 (harder than probability matching, softer than a hard majority rule).

(4) CONFLICT INTERFERENCE IS COMPETITION-SCALED (the refinement). Guessing is driven by the load of holding several discrepancies in mind while ADJUDICATING BETWEEN THEM. The load is therefore not the raw number of discriminating rows but the number of rows that actually have to be pitted against one another: when the display is unanimous (min(n_A,n_B) = 0, i.e. every discrepancy points the same way) there is nothing to adjudicate and the extra rows impose only a discounted reading load eta*(m-2) rather than the full (m-2). Formally g = 1 - (1-eps0) * rho^L with L = (m-2) on opposed displays and L = eta*(m-2) on unanimous displays, and the core probability is mixed toward 0.5 by g. This predicts that dominance accuracy degrades with the number of agreeing experts only mildly, whereas conflict displays degrade steeply with the number of opposed experts - the two are dissociated, not a single monotone function of display size.

No feedback, hence no learning; the rule is stationary across the block.

**Rationale:** MINIMAL DIFF on the accepted iter-1 base: I added exactly one parameter (eta) and three lines in the lapse block; gamma, lam, beta, kappa, eps0, rho, omega ranges and every other line are re-emitted verbatim, per the critic's instruction not to retune lam/gamma and not to widen beta/eps0 again.

What changed and why. The critic's key diagnosis is correct: Exp5 (unanimous displays) needs LESS guessing while the sub-chance conflict metrics need MORE, and a single g = 1-(1-eps0)*rho^(m-2) cannot deliver both. I implement the prescribed decoupling, but with the conflict/dominance asymmetry placed on the DOMINANCE side rather than on the conflict side. Load L = (m-2) on opposed displays and L = eta*(m-2) on unanimous displays: adjudication load arises only from rows that must be pitted against one another, whereas agreeing extra rows impose mere reading effort. This is the same 'conflict interference' clause, just stated in terms of competing rather than total rows.

Why not the critic's literal theta^min(n_A,n_B) with rho pushed to 0.85-0.95: I worked the algebra through the actual designs and that form regresses Exp8. The Exp8 metric is computed on OPPOSED trials only (7 trial types, three of them m=2, c=1, with core probabilities 0.80-0.99), so any extra conflict-driven guessing pulls Exp8 toward 0.5 - and Exp8 is already 0.019 BELOW the data (0.751 vs 0.770), i.e. it needs less guessing, not more. With theta=0.84, rho=0.82, eps0=0.23 my hand calculation gives Exp8 ~0.718 (error -0.052) in exchange for a +0.03 gain on Exp1: a net loss the gate would reject, exactly as in iteration 2. Exp1 and Exp8 have essentially the same joint (m, min(n_A,n_B)) distribution over their diagnostic trials, so no count-based conflict term can separate them; chasing Exp1 through the lapse is a dead end.

The eta edit is instead strictly monotone in the right direction and provably touches only Exp5. Every other metric filters to opposed displays: Exp1 keeps only trials where the first cue disagrees with the tally (never unanimous), Exp2 keeps tally ties and TTB/tally disagreements, Exp3/Exp4 keep tally ties, Exp6 keeps exactly two opposed cues, Exp7 explicitly skips unanimous displays, Exp8 drops trials where the majority agrees with the salient row (which includes all dominance trials). So min(n_A,n_B) >= 1 there and g is bit-for-bit unchanged: Exp1 0.253, Exp2 0.170, Exp3 0.283, Exp4 0.137, Exp6 0.508 and Exp7 0.439 (the two currently exact values the critic warned are most sensitive) and Exp8 0.751 are all preserved.

Calibration of eta. Exp5's dominance trials are m=2, 4, 6 in equal proportion. Inverting the base output (0.7193 at mean g = 0.501) gives an effective p_core ~0.94 and dAcc/dg ~ -0.44, so reaching the observed 0.743 requires mean g to fall by ~0.055, i.e. exp(-0.5034*eta) + exp(-1.0068*eta) = 1.186 at rho = 0.7775, giving eta ~ 0.71-0.73. The declared range [0.62, 0.84] brackets that point symmetrically (predicted Exp5 ~0.732-0.753, mean ~0.742) and, as a bonus, is the only parameter that varies dominance accuracy across subjects, which addresses the too-low Exp5 between-subject variance without touching beta or eps0 as the gate demanded. Expected result: seven metrics unchanged and Exp5's 0.023 error reduced to ~0.003, lowering the RMS point error from ~0.0155 to ~0.0133 and so clearing the 0.0232 floor.

**Parameters:**
  - `gamma`: `[0.86, 0.94]`
  - `lam`: `[1.45, 1.55]`
  - `beta`: `[3.2, 4.0]`
  - `kappa`: `[1.10, 1.30]`
  - `eps0`: `[0.22, 0.26]`
  - `rho`: `[0.755, 0.80]`
  - `omega`: `[0.0, 0.04]`
  - `eta`: `[0.62, 0.84]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
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
            flat = np.asarray(arr, dtype=float).reshape(2, -1)
            a, b = flat[0].ravel(), flat[1].ravel()

    n = int(min(np.size(a), np.size(b)))
    if n == 0:
        return np.array([0.5, 0.5])
    a = np.asarray(a, dtype=float)[:n]
    b = np.asarray(b, dtype=float)[:n]

    # ---------- communicated validities (the printed label code) ----------
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
    beta = float(parameters['beta'])
    kappa = float(parameters['kappa'])
    eps0 = float(np.clip(float(parameters['eps0']), 0.0, 1.0))
    rho = float(np.clip(float(parameters['rho']), 1e-6, 1.0))
    omega = float(np.clip(float(parameters['omega']), 0.0, 0.5))
    eta = float(np.clip(float(parameters['eta']), 0.0, 1.0))

    # ---------- STRATEGY SWITCH: injectivity of the printed label code ----------
    LABEL_TOL = 2.5e-3   # two printed labels closer than this are the same symbol
    if n <= 1:
        injective = True
    else:
        vs = np.sort(v)
        gaps = np.diff(vs)
        injective = bool(np.all(gaps > LABEL_TOL))

    # ---------- discriminating rows ----------
    diff = a - b
    disc = np.nonzero(diff != 0.0)[0]
    m = int(disc.size)
    if m == 0:
        return np.array([0.5, 0.5])
    signs = np.sign(diff[disc])

    # ---------- weighted mode: distinctive-cue SUMMATION ----------
    # log weight = gamma*log(1 - v_j) + lam*j ; rescaled so strongest row = 1
    logw = gamma * np.log(np.clip(1.0 - v[disc], 1e-9, None)) + lam * disc.astype(float)
    logw = logw - np.max(logw)
    w = np.exp(logw)
    E = float(np.sum(signs * w))
    z = float(np.clip(beta * E, -60.0, 60.0))
    p_weighted_a = 1.0 / (1.0 + np.exp(-z))

    # ---------- tally mode: graded majority ----------
    n_a = float(np.sum(signs > 0.0))
    n_b = float(m) - n_a
    zt = float(np.clip(kappa * (n_a - n_b), -60.0, 60.0))
    p_tally_a = 1.0 / (1.0 + np.exp(-zt))

    if injective:
        p_core_a = (1.0 - omega) * p_weighted_a + omega * p_tally_a
    else:
        p_core_a = (1.0 - omega) * p_tally_a + omega * p_weighted_a

    # ---------- conflict interference / lapse (competition-scaled load) ----------
    # load counts only rows that must be ADJUDICATED against one another:
    # unanimous (dominance) displays impose only a discounted reading load.
    conflict = float(min(n_a, n_b))
    expo = float(max(m - 2, 0))
    if conflict <= 0.0:
        expo = eta * expo
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


## Replacement

### `pi_8` → slot 2 (via `new_theory`)

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
