# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_7" and "pi_8") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_7" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_8" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_7
**MDSC-NV (sharpened) — Maximum-Doubt Single-Cue selection with Null-Verdict capture.**

Four claims; the mechanism family is unchanged from the accepted base, but one quantitative commitment is sharpened.

**(1) One-reason choice by MAXIMUM DOUBT (no summation).** On each trial the subject settles A-vs-B with the verdict of a SINGLE expert. Which expert is consulted is decided by *doubt*: a disagreement voiced by a near-chance expert (v≈.53) is a strange, newsworthy fact that has to be adjudicated, whereas a disagreement voiced by a near-certain expert (v≈.97) is discounted as a mere echo of the impression the subject already has. Salience is log u_j = delta*log(1-v_j) + lam*j and the consulted row is drawn from softmax(u). **The doubt gain is LARGER than previously assumed (delta ≈ 13-17, not 8-12): the doubt rule is essentially argmax, not a graded share.** This is the sharpened commitment. It matters wherever two moderately-doubtful rows are within ~0.4 log-validity of one another (e.g. v=.55 vs v=.70 rows two columns apart): a share-based rule lets the second-most-doubtful row steal ~18% of the verdict and blunts the choice, whereas the near-argmax rule leaves the display fully decided by the single most doubtful row. Coalitions still buy nothing: five agreeing rows never outvote one more-doubtful row, which is why choice is systematically anti-validity in every design where the printed labels are usable.

**(2) Reading position is a WEAK tie-breaker.** lam = pos_ratio*delta with pos_ratio ≈ 0.09-0.14 — an order of magnitude below the doubt term's dynamic range. Position therefore only arbitrates between rows whose stated validities are (nearly) *equal* — exactly the .955/.950/.945 and .958/.954/.950 designs, where it makes the last low-validity row win. Whenever validities genuinely differ, doubt beats recency: a low-validity EARLY column defeats a late .93/.99 column. Because delta is raised and pos_ratio held fixed, the *ratio* of doubt-range to positional gain is preserved, so all orderings in the corpus are unchanged.

**(3) NULL-VERDICT CAPTURE.** Doubt is a property of the *expert*, not of the disagreement, so attention is captured pre-attentively by the most doubtful expert **in the whole display**, before the subject checks whether that expert actually discriminates. With probability q (≈ .41, slightly higher than previously estimated) the subject simply goes with whoever captured attention; if that expert gave both products the same rating it returns no verdict, the subject has nothing to adjudicate and guesses. With probability 1-q the subject re-targets deliberately, restricting the doubt competition to the rows that differ. Sharp prediction: *holding the discriminating rows fixed, adding agreement on a MORE doubtful (lower-validity) row pushes choice toward chance*, whereas adding agreement on a high-validity row does nothing. Displays in which only two high/mid-validity rows disagree while the display's most doubtful expert stays silent are therefore near chance — no compensatory or one-reason model predicts that non-discriminating rows matter at all.

**(4) LABEL-DEGENERACY GATE (all-or-none) and LOAD LAPSE.** The doubt rule presupposes that each printed validity picks out one expert. If any label is printed twice (tolerance .002 — it must NOT fire for the .004/.005-gap designs) the experts are interchangeable, the doubt ordering is unavailable, and choice collapses to a graded unweighted row tally (exact coin flip on 1-1 splits). Independently, the more rows disagree the more often the subject loses the thread: p(guess) = 1-(1-eps0)*rho^(m-2), which compresses conflict displays toward chance and keeps even unanimous 6-row displays well below ceiling. Individual subjects differ mainly in how leaky (eps0, rho) and how capture-prone (q) they are, not in the doubt rule itself.

No feedback is given, so the rule is stationary across the block.

`predict(parameters, state, history) -> np.ndarray`:
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

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if not np.isfinite(s) or s <= 0.0:
        return int(np.random.randint(len(probs)))
    probs = probs / s
    return int(np.random.choice(len(probs), p=probs))

## THEORY 2 — pi_8
**DTV-RG/CL — Doubt-Triggered one-reason with Rank-Geometric weighting, under a two-component (load + conflict) leak and a DECISIVE degenerate-label tally.**

The evidence rule is unchanged from the accepted base; two claims about *how the rule degrades* are sharpened.

**(1) Rank-geometric evidence, decided from the LEAST valid dissenting expert.** On each trial the subject lines the *discriminating* rows up in order of printed validity, least valid first, and sums their verdicts with a geometric discount: E = Σ_r s_r · φ^r (r = 0 for the least valid discriminating expert, s_r = ±1 its verdict); p(A) = σ(βE). Only the ORDER of the printed labels enters, never the size of the gaps, which is why the .955/.950/.945 and .958/.954/.950 families are still resolved almost deterministically in favour of the strictly least valid dissenter, and why coalitions of more valid experts normally buy nothing (though a large enough coalition can still overturn one doubtful voice).

**(2) The discount rate is set by the ABSOLUTE doubtfulness of the decisive voice.** One-reason choice is a *reaction to a near-chance expert*: φ = φ_lo + (φ_hi − φ_lo)·σ((v₀ − v_c)/w), with v₀ the validity of the least valid DISCRIMINATING expert, φ_lo ≈ .25 (take-the-least), φ_hi ≈ .88 (near tally), v_c ≈ .575, w ≈ .015. Below the trigger the display is settled by one strange dissent; above it nothing is newsworthy and the rows are weighted nearly equally.

**(3) No null-verdict capture.** Non-discriminating rows are strictly irrelevant whatever their validity; all pull toward chance comes from the leak in (4).

**(4) NEW — the leak has TWO separable sources, and dominance displays leak only through the first.** Losing the thread is driven (i) by sheer display load — how many rows have to be held while the ordering is run — and (ii) independently by *opposition*: every additional row that argues the other way adds a fresh occasion to lose track of which side one was on. p(guess) = 1 − (1−ε₀)·ρ_m^max(m−2,0)·ρ_c^min(n_A,n_B), with ρ_m ≈ .79 (load) and ρ_c ≈ .97 (opposition). The two terms are close to orthogonal in the corpus: unanimous (dominance) displays have min(n_A,n_B)=0 and therefore leak only by load, so they stay well above the balanced-conflict displays of the same size, whereas a 3-vs-3 display of six rows is markedly noisier than a 6-vs-0 display of six rows. This is a direct, testable clash both with a pure-load leak (which charges a 6-0 display exactly what it charges a 3-3 display) and with a pure-opposition leak (which would make unanimous displays near-perfect).

**(5) NEW — when labels are degenerate the fallback tally is DECISIVE, not tepid.** The doubt ordering presupposes that each printed validity picks out one expert. If any label is repeated (gap < .002) the columns are interchangeable and choice collapses to an unweighted row tally — but that tally is used with a steep slope (κ ≈ 1.6, not ≈ 0.85): having nothing else to go on, the subject commits hard to the vote margin. Hence 1-vs-1 splits under duplicated labels are an exact coin flip (no anti-validity, no recency), while 2-vs-0 and 4-vs-0 splits under duplicated labels are chosen at ~.83 and ~.95 before the leak — which is what keeps unanimous displays high and what makes a 2-vs-1 display under duplicated labels clearly favour the majority rather than hovering near chance.

No feedback is given, so the rule is stationary across the block.

`predict(parameters, state, history) -> np.ndarray`:
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

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if not np.isfinite(s) or s <= 0.0:
        return int(np.random.randint(len(probs)))
    probs = probs / s
    return int(np.random.choice(len(probs), p=probs))

## EXPERIMENT 1 (proposed by pi_7)

### DESIGN
**Validities (n_features=5):** [0.68, 0.95, 0.8, 0.52, 0.88]

**Trial pairs (n=18):**
  trial 1: A=[1, 0, 1, 1, 0]  B=[1, 0, 1, 0, 0]
  trial 2: A=[1, 0, 1, 0, 0]  B=[1, 0, 1, 1, 0]
  trial 3: A=[1, 1, 0, 1, 1]  B=[0, 1, 0, 1, 1]
  trial 4: A=[0, 1, 0, 1, 1]  B=[1, 1, 0, 1, 1]
  trial 5: A=[0, 1, 1, 1, 1]  B=[0, 0, 1, 0, 1]
  trial 6: A=[0, 0, 1, 0, 1]  B=[0, 1, 1, 1, 1]
  trial 7: A=[1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 8: A=[0, 0, 1, 0, 0]  B=[1, 1, 1, 0, 0]
  trial 9: A=[1, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0]
  trial 10: A=[0, 1, 1, 1, 0]  B=[1, 1, 0, 1, 0]
  trial 11: A=[1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1]
  trial 12: A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 1, 0]
  trial 13: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 1, 1]
  trial 14: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 1, 0]
  trial 15: A=[0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 16: A=[0, 1, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 17: A=[0, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 18: A=[0, 1, 1, 0, 1]  B=[0, 0, 0, 1, 0]

**Rationale:** GOAL: separate MDSC-NV (advocated) from DTV-RG/CL (competitor). The two accounts share the anti-validity ('doubt') direction and a load-based leak, so the design is built so that they AGREE about WHICH row is decisive and differ only on the three mechanisms that actually distinguish them: (A) NULL-VERDICT CAPTURE (MDSC: non-discriminating rows matter, because attention is grabbed pre-attentively by the globally most doubtful expert and a silent capture target forces a guess with prob q; DTV: silent rows are strictly irrelevant); (B) SINGLE-CUE vs RANK-GEOMETRIC SUMMATION with DTV's ABSOLUTE-DOUBT TRIGGER (DTV's phi jumps from ~.25 to ~.88 as soon as the least valid DISCRIMINATING expert is above v_c~.575, turning it into a near-tally in which coalitions overturn the doubtful voice; MDSC is near-argmax one-reason always, so coalitions buy nothing); (C) leak structure.

VALIDITIES [0.68, 0.95, 0.80, 0.52, 0.88] at columns c0..c4. Three properties are deliberately engineered. (i) One expert, c3 = .52, sits BELOW DTV's trigger (v_c ~ .575); every other expert (.68, .80, .88, .95) sits well above it. So whenever c3 discriminates DTV is in take-the-least mode (phi~.27), and whenever c3 is silent DTV flips into near-tally mode (phi~.88). (ii) c3 = .52 is also MDSC's global doubt anchor: with delta in [13,17] and lam = pos_ratio*delta, log-salience is c3 >> c0 > c2 > c4 > c1 across the ENTIRE parameter box (margin of c3 over c0 is 8-11 log units), and this ordering is exactly the ascending-validity ordering DTV uses, so the models can never be confused about which row is decisive. (iii) Validity does NOT track column order (.68, .95, .80, .52, .88), so 'use the first/last row' cannot masquerade as either model, and all gaps (>=.07) are far above both degeneracy tolerances (.002/.0025), keeping both models out of tally/degenerate mode.

9 base structures, each with its exact A/B mirror (18 unique pairs, K=5 -> 90 trials), so both models' predicted responses are exactly key-balanced and side bias cannot mimic either. Silent rows carry matched 1s AND matched 0s so displays are dense and cannot be solved by counting 1s. Predictions below use MDSC (delta=15, pos_ratio=.115, q=.41, eps0=.24, rho=.81) and DTV (phi_lo=.25, phi_hi=.88, v_c=.575, w=.015, beta=5, eps0=.24, rho_m=.79, rho_c=.97).

BLOCK 1 - CROWN JEWEL: NULL-VERDICT CAPTURE ON DOMINANCE DISPLAYS (bases 1-4, pairs 1-8). Four unanimous displays that differ ONLY in WHICH experts are silent.
* B1 (m=1, A wins c3 = the .52 anchor): MDSC .88, DTV .875.
* B2 (m=1, A wins c0 = .68; the .52 anchor is SILENT): MDSC .724, DTV .875.
* B3 (m=2, A wins c3 and c1; anchor speaks): MDSC .88, DTV .879.
* B4 (m=2, A wins c0 and c1; anchor SILENT): MDSC .724, DTV .880.
DTV (like every compensatory or standard one-reason model) must call B1=B2 and B3=B4 EXACTLY - a flat ~.88 profile for any parameter setting, because E is computed over discriminating rows only and the leak sees only m and min(n_A,n_B), both identical within each pair. MDSC must produce a 13-17 point DROP (robust over its whole box) purely as a function of whether the .52 expert happens to be the one that speaks. A 2-0 (or 1-0) dominance display becoming measurably harder because a NON-discriminating row is the most doubtful one is the signature no competitor can generate.

BLOCK 2 - CROWN JEWEL: COALITIONS vs ONE REASON WITH THE ANCHOR SILENT (bases 2, 5, 6, 7; pairs 3-4, 9-14). A always wins c0 (.68, the least valid DISCRIMINATING row) while B's opposing coalition grows, and the .52 anchor stays silent throughout, so DTV is locked in near-tally mode (phi~.88):
* B2 1-vs-0: MDSC .724 / DTV .875
* B5 1-vs-1 (B wins c2): MDSC .712 / DTV .608
* B6 1-vs-2 (B wins c2,c4): MDSC .669 / DTV .230  <- 44-point DIRECTIONAL REVERSAL
* B7 1-vs-3 (B wins c2,c4,c1): MDSC .637 / DTV .271  <- 37-point DIRECTIONAL REVERSAL
MDSC predicts a shallow, monotone decline driven only by load (p_core pinned at ~.78 whatever the coalition size; never crossing chance). DTV predicts a CRASH THROUGH CHANCE between the 1-vs-1 and 1-vs-2 displays as the geometric sum flips sign. Shallow-and-always-above-chance vs steep-crossing is a parameter-robust quantitative signature (checked at the corners of both parameter boxes: MDSC B6 stays in [.63,.72], DTV B6 stays in [.21,.26]).

BLOCK 3 - THE ABSOLUTE-DOUBT TRIGGER (bases 8-9 vs 6-7; pairs 11-18). Structurally identical conflict displays in which only the IDENTITY of the lone dissenter changes, from c0 (.68, above DTV's trigger) to c3 (.52, below it):
* 1-vs-2: B6 (lone .68) vs B8 (lone .52). DTV: .230 -> .771, a 54-point swing that crosses chance - exactly its claim (2). MDSC: .669 -> .808, a 14-point swing, both above chance (the extra rise is capture being satisfied rather than a change of rule).
* 1-vs-3: B7 (lone .68) vs B9 (lone .52). DTV: .271 -> .713 (44 points). MDSC: .637 -> .749 (11 points).
A large, chance-crossing swing keyed to the absolute validity of the dissenter confirms DTV's trigger; a small swing with both cells above chance confirms MDSC's near-argmax rule plus capture. Bases 1, 3, 8 and 9 also double as the shared lapse/load baseline (both models within 4 points there), so any divergence in Blocks 1-2 cannot be written off as global noise.

INTERPRETATION: if 1-0 and 2-0 dominance accuracy drops ~15 points purely because the .52 expert is silent, and if the single .68 dissenter still beats two- and three-row coalitions of more valid experts, MDSC-NV is confirmed and DTV-RG/CL is falsified. If instead dominance displays are flat at ~.88 regardless of who is silent, and the coalition displays invert (majority chosen ~.75 when the dissenter is .68, but the lone dissenter chosen ~.77 when it is .52), rank-geometric summation with the absolute-doubt trigger wins and null-verdict capture is dead.

**Computed schedule:** 18 unique pairs × 5 reps = 90 trials per subject.



### METRIC
Rationale:
This isolates the design's Block-2/Block-3 coalition displays in which the single .68 expert (feature 0) dissents against a coalition of two or three MORE valid experts (features 1, 2, 4) while the .52 anchor (feature 3) is silent \u2014 i.e. base structures 6 and 7 (trial pairs 11-14), 20 trials per subject.

MDSC-NV is a near-argmax maximum-doubt one-reason rule: the least valid DISCRIMINATING expert (.68) decides regardless of how many more valid experts oppose it, and capture by the silent .52 anchor only pulls part of the way toward chance. It therefore predicts the lone .68 dissenter's option is still chosen well ABOVE chance (~.64-.67).

DTV-RG/CL, because the least valid dissenter (.68) is ABOVE its absolute-doubt trigger (v_c \u2248 .575), flips into near-tally mode (\u03c6 \u2248 .88), so a 2- or 3-member coalition of more valid experts overturns the lone dissenter: it predicts this same quantity CRASHES BELOW chance (~.23-.27).

So the predicted values are ~.65 vs ~.25 \u2014 a ~40-point gap that straddles chance, with each subject contributing 20 trials, giving tight per-subject estimates (binomial SD \u2248 .10) and a clean Welch separation at N=25. It is not trivially true of MDSC: a compensatory model, a pure tally, a recency rule, or any ordinary take-the-best (which would favour the most valid expert's side) all give distinct values, and the contrast is purely about whether coalitions of more valid experts can overturn one doubtful dissenter.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    hits = []
    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()
    for a_raw, b_raw, r in zip(a_col, b_col, r_col):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        n = int(min(a.size, b.size))
        if n < 5:
            continue
        a = a[:n]
        b = b[:n]
        d = a - b
        disc = [int(i) for i in np.nonzero(d != 0.0)[0]]
        # require feature 0 (validity .68) to discriminate
        if 0 not in disc:
            continue
        # require the global doubt anchor, feature 3 (validity .52), to be SILENT
        if 3 in disc:
            continue
        fav_a = [i for i in disc if d[i] > 0.0]
        fav_b = [i for i in disc if d[i] < 0.0]
        if d[0] > 0.0:
            lone, other = fav_a, fav_b
            lone_is_a = True
        else:
            lone, other = fav_b, fav_a
            lone_is_a = False
        # exactly one dissenter (feature 0) against a coalition of >=2 more valid experts
        if len(lone) != 1 or len(other) < 2:
            continue
        try:
            resp = int(r)
        except Exception:
            continue
        chose_lone = (resp == 0) if lone_is_a else (resp == 1)
        hits.append(1.0 if chose_lone else 0.0)
    if len(hits) == 0:
        return float('nan')
    return float(np.mean(hits))

### RESULTS
- Predicted under pi_7 (simulated): 0.6410 (var=0.0093)
- Predicted under pi_8 (simulated): 0.2680 (var=0.0082)
- Observed on real data: 0.5460 (var=0.0126)

## EXPERIMENT 2 (proposed by pi_8)

### DESIGN
**Validities (n_features=6):** [0.62, 0.78, 0.86, 0.93, 0.53, 0.97]

**Trial pairs (n=24):**
  trial 1: A=[1, 0, 1, 1, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  trial 2: A=[1, 0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1, 0]
  trial 3: A=[1, 1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 1, 1]
  trial 4: A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  trial 5: A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 6: A=[0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 1, 0]
  trial 7: A=[1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 1]
  trial 8: A=[1, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1]
  trial 9: A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  trial 10: A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  trial 11: A=[1, 0, 0, 1, 0, 1]  B=[0, 1, 1, 1, 0, 1]
  trial 12: A=[0, 1, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0, 1]
  trial 13: A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0, 1]
  trial 14: A=[0, 1, 1, 1, 0, 1]  B=[1, 0, 0, 0, 0, 1]
  trial 15: A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0]
  trial 16: A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  trial 17: A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 1, 1]
  trial 18: A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 0, 1, 1, 1]
  trial 19: A=[0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 1]
  trial 20: A=[0, 1, 1, 0, 0, 1]  B=[0, 0, 0, 1, 1, 1]
  trial 21: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 22: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  trial 23: A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1]
  trial 24: A=[1, 1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 0]

**Rationale:** GOAL: separate DTV-RG/CL (advocated) from MDSC-NV (competitor). The two share the anti-validity ('doubt') direction and a load leak, so the design is built so that they always AGREE about WHICH row is the decisive/doubt-anchor row, and differ only on the three mechanisms that genuinely distinguish them: (A) NULL-VERDICT CAPTURE (MDSC: silent rows matter, because attention is grabbed pre-attentively by the globally most doubtful expert and a silent capture target forces a guess with prob q~.41; DTV: non-discriminating rows are strictly irrelevant); (B) ABSOLUTE-DOUBT TRIGGER + RANK-GEOMETRIC SUMMATION (DTV: phi jumps from ~.28 to ~.85 as soon as the least valid DISCRIMINATING expert is above v_c~.575, so with the sub-trigger expert silent DTV becomes a near-tally in which coalitions of more valid experts overturn the lone doubtful dissenter, crossing chance; MDSC is near-argmax one-reason always, so coalitions buy nothing and it can never cross chance); (C) DTV's two-component leak (load x opposition, rho_c) versus MDSC's pure-load leak.

VALIDITIES [0.62, 0.78, 0.86, 0.93, 0.53, 0.97] at columns c0..c4,c5. Three engineered properties. (i) Exactly one expert, c4 = .53, sits BELOW DTV's trigger (v_c ~ .575); every other expert (.62, .78, .86, .93, .97) sits well above it, so DTV is in take-the-least mode whenever c4 speaks and in near-tally mode whenever c4 is silent. (ii) c4 = .53 is placed at column 4 so that doubt and reading position are ALIGNED for it: MDSC's log-salience (delta 13-17, lam = pos_ratio*delta) makes c4 the global argmax by >7 log units at every corner of its parameter box, and the second-most-doubtful discriminating row (c0 = .62) beats the next candidate (c1 = .78) by >5 log units, so MDSC is effectively pure argmax and cannot hide behind graded sharing. Crucially MDSC's doubt ordering over the discriminating rows (.53 < .62 < .78 < .86 < .93 < .97) is IDENTICAL to DTV's ascending-validity ordering, so the models never disagree about which row is decisive \u2014 only about what is done with it. (iii) Validity does not track column index (.62,.78,.86,.93,.53,.97), so 'take the first/last row' cannot masquerade as either model, and all label gaps (>= .04) are far above both degeneracy tolerances (.002/.0025), keeping both models out of tally/degenerate mode. Silent rows carry matched 1s AND matched 0s so displays are dense and cannot be solved by counting 1s.

12 base structures, each with its exact A/B mirror (24 unique pairs, K = 4 -> 96 trials), so both models' predicted responses are perfectly key-balanced and side bias cannot mimic either. Predictions below use DTV (phi_lo=.25, phi_hi=.88, v_c=.575, w=.015, beta=5, eps0=.24, rho_m=.79, rho_c=.97) and MDSC (delta=15, pos_ratio=.115, q=.41, eps0=.24, rho=.81).

BLOCK 1 \u2014 CROWN JEWEL: NULL-VERDICT CAPTURE ON DOMINANCE DISPLAYS (bases 1-4, pairs 1-8). Four unanimous displays that differ ONLY in WHICH experts are silent, holding m and min(n_A,n_B) exactly constant within each matched pair.
* Base 1 (m=2 unanimous, A wins c4 [.53] and c3; the anchor SPEAKS): DTV .878, MDSC .880.
* Base 2 (m=2 unanimous, A wins c0 [.62] and c3; the .53 anchor is SILENT): DTV .878, MDSC .724.
* Base 4 (m=4 unanimous, A wins c4,c1,c2,c3; anchor speaks): DTV .737, MDSC .749.
* Base 3 (m=4 unanimous, A wins c0,c1,c2,c3; anchor SILENT): DTV .737, MDSC .647.
DTV must call base1 = base2 and base3 = base4 EXACTLY for every parameter setting (E is computed over discriminating rows only and the leak sees only m and min(n_A,n_B)). MDSC must produce a 10-16 point DROP purely because a NON-discriminating row happens to be the most doubtful one. Flat-within-pair versus stepped is a parameter-free read-out of capture, and base 1 simultaneously pins eps0 for both models.

BLOCK 2 \u2014 CROWN JEWEL: COALITION CRASH WITH THE ANCHOR SILENT (bases 2, 5, 6, 7; pairs 3-4, 9-14). A always wins c0 (.62, the least valid DISCRIMINATING row) while B's opposing coalition of more valid experts grows, with the .53 anchor silent throughout, so DTV is locked in near-tally mode (phi~.85):
* 1-vs-0 (base 2): DTV .878 / MDSC .724
* 1-vs-1, B wins c1 (base 5): DTV .632 / MDSC .724
* 1-vs-2, B wins c1,c2 (base 6): DTV .240 / MDSC .682  <- 44-point DIRECTIONAL REVERSAL
* 1-vs-3, B wins c1,c2,c3 (base 7): DTV .271 / MDSC .647  <- 38-point DIRECTIONAL REVERSAL
MDSC predicts a shallow, monotone, always-above-chance decline driven only by load (its core is pinned at ~.795 whatever the coalition size); DTV predicts a CRASH THROUGH CHANCE between the 1-vs-1 and 1-vs-2 displays as the rank-geometric sum flips sign. Checked at the corners of DTV's box (phi_hi .84-.92): E at 1-vs-2 is -.55 to -.77, always negative.

BLOCK 3 \u2014 THE ABSOLUTE-DOUBT TRIGGER (bases 6 vs 8, 9 vs 10). Structurally identical conflict displays in which only the IDENTITY of the lone dissenter changes, from c0 (.62, above the trigger) to c4 (.53, below it):
* 1-vs-2: base 6 (lone .62 dissenter) DTV .240 -> base 8 (A wins c4, B wins c0,c1) DTV .769: a 53-point swing CROSSING chance. MDSC: .682 -> .808, a 13-point swing with both cells above chance.
* 2-vs-2 at an exact row-count tie: base 9 (A wins c0,c3; B wins c1,c2; anchor silent) DTV .523 (essentially CHANCE \u2014 the geometric sum nearly cancels despite a lopsided validity profile) vs base 10 (A wins c4,c3; B wins c1,c2; anchor speaks) DTV .708, a 19-point swing. MDSC: .647 -> .749, and it can never sit at chance on base 9 because its argmax row (c0) decides. Base 9 is the sharpest single cell: DTV ~.52, MDSC ~.65 at a 2-2 tally tie (no tally account can produce either value systematically).

BLOCK 4 \u2014 LOAD VERSUS OPPOSITION LEAK (bases 11, 12; pairs 21-24). Base 11 is a 6-vs-0 unanimous display (DTV .648, MDSC .663); base 12 is a 3-vs-3 display of the same six rows with the .53 anchor favouring A (DTV .626, MDSC .663). DTV's second leak component (rho_c^min(n_A,n_B)) says the balanced-conflict display is measurably noisier than the unanimous display of the SAME size even though the decisive row is equally decisive; MDSC's pure-load leak says they must be identical. Base 12 also shows both models above chance at an exact 3-3 tally, which kills any vote-counting account outright.

INTERPRETATION: if dominance displays are flat within matched pairs (base1 = base2, base3 = base4) regardless of who is silent, and if a single .62 dissenter is beaten by two- and three-row coalitions of more valid experts (bases 6-7 below chance) while a single .53 dissenter beats the same coalitions (base 8 ~.77), the absolute-doubt trigger plus rank-geometric summation is confirmed and null-verdict capture is dead. If instead 1-0/2-0 and 4-0 dominance accuracy drops 10-16 points purely because the .53 expert happens to be silent, and the lone .62 dissenter stays above chance against every coalition, MDSC-NV wins. No pair is degenerate: on 8 of the 12 bases the models disagree in direction or by more than 10 points, and bases 1, 4 and 11 double as the shared lapse/load baseline (models within 2 points there), so divergences elsewhere cannot be written off as global noise.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



### METRIC
Rationale:
This isolates Block 2's 'coalition crash' cells (trial pairs 11-14: 1-vs-2 and 1-vs-3 displays in which the lone dissenter is c0 = .62, the least valid DISCRIMINATING expert, while the globally most doubtful expert c4 = .53 stays SILENT). The metric is the proportion of those trials on which the subject sides with the lone least-valid dissenter.

DTV-RG/CL: with the sub-trigger .53 expert silent, v0 = .62 > v_c, so phi jumps to ~.85 and evidence becomes a near-tally rank-geometric sum: E = 1 - phi - phi^2 (1-vs-2) and 1 - phi - phi^2 - phi^3 (1-vs-3), both clearly NEGATIVE at every corner of the parameter box. The coalition of more valid experts overturns the lone dissenter, so choice falls BELOW chance: predicted ~.24-.27.

MDSC-NV: the doubt rule is near-argmax over discriminating rows, and c0 (.62) is by far the most doubtful discriminating row, so the deliberate route almost always sides with the lone dissenter; only the capture-to-silent-row component (q ~ .41 x 0.5) and the load lapse pull it toward chance, leaving it clearly ABOVE chance: predicted ~.65-.68.

Hence a ~0.41 directional gap that straddles chance, with 16 trials per subject and parameter ranges that barely move either prediction, so between-subject variance is dominated by (small) binomial noise. The metric uses no theory-specific machinery: it is just 'do people follow the lone least-valid dissenter against a coalition?'

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    V = np.array([0.62, 0.78, 0.86, 0.93, 0.53, 0.97], dtype=float)
    ANCHOR = 4  # the .53 expert (below DTV's doubt trigger)
    hits = 0.0
    tot = 0.0
    for a_raw, b_raw, resp in zip(data['option_a_ratings'].tolist(),
                                  data['option_b_ratings'].tolist(),
                                  data['response'].tolist()):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        n = int(min(a.size, b.size))
        if n != 6:
            continue
        diff = a[:6] - b[:6]
        disc = np.nonzero(diff != 0.0)[0]
        m = int(disc.size)
        if m < 3:
            continue
        # the globally most doubtful expert (.53) must be SILENT
        if ANCHOR in set(disc.tolist()):
            continue
        d = diff[disc]
        na = int(np.sum(d > 0.0))
        nb = m - na
        if min(na, nb) != 1 or max(na, nb) < 2:
            continue
        if na == 1:
            lone = int(disc[d > 0.0][0])
            fav = 0  # option A favoured by the lone dissenter
        else:
            lone = int(disc[d < 0.0][0])
            fav = 1
        # the lone dissenter must be the LEAST valid discriminating expert
        if abs(V[lone] - float(np.min(V[disc]))) > 1e-9:
            continue
        tot += 1.0
        if int(resp) == fav:
            hits += 1.0
    if tot < 1.0:
        return 0.5
    return float(hits / tot)

### RESULTS
- Predicted under pi_7 (simulated): 0.6388 (var=0.0149)
- Predicted under pi_8 (simulated): 0.2313 (var=0.0086)
- Observed on real data: 0.4675 (var=0.0144)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """One-reason adherence on model-disagreement trials.

    For every trial we compute (a) the option favoured by the FIRST
    discriminating cue in validity order (features are listed in
    descending validity: 0.95, 0.85, 0.75, 0.65, 0.55) and (b) the option
    favoured by the unit-weight tally of feature-wise wins.  We keep only
    trials where these two disagree in direction OR where the tally is tied
    (so a counting rule must guess).  The metric is the proportion of such
    trials on which the subject picked the option favoured by the first
    discriminating cue.
    """
    hits = []
    for a_raw, b_raw, resp in zip(data['option_a_ratings'],
                                  data['option_b_ratings'],
                                  data['response']):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        n = min(a.shape[0], b.shape[0])
        a = a[:n]
        b = b[:n]

        # first discriminating cue (features already in descending validity)
        first_winner = None
        for j in range(n):
            if a[j] > b[j]:
                first_winner = 0
                break
            if b[j] > a[j]:
                first_winner = 1
                break
        if first_winner is None:
            continue  # no cue discriminates: uninformative trial

        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        if a_wins > b_wins:
            tally_winner = 0
        elif b_wins > a_wins:
            tally_winner = 1
        else:
            tally_winner = None  # counting rule must guess

        if tally_winner is not None and tally_winner == first_winner:
            continue  # agreement control trial: not diagnostic

        try:
            r = int(resp)
        except (TypeError, ValueError):
            continue
        hits.append(1.0 if r == first_winner else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.2820 (var=0.0035)
**Predicted under pi_7:** 0.2878 (var=0.0023)
**Predicted under pi_8:** 0.2730 (var=0.0019)

### Experiment 4
**Design**
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.6, 0.93, 0.55, 0.87, 0.7, 0.78], dtype=float)

    def analyze(row):
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        n = min(a.shape[0], b.shape[0])
        a = a[:n]
        b = b[:n]
        val = validities[:n] if validities.shape[0] >= n else np.arange(n, 0, -1, dtype=float)
        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        margin = a_wins - b_wins
        if margin > 0:
            tally_winner = 0
        elif margin < 0:
            tally_winner = 1
        else:
            tally_winner = -1
        order = np.argsort(-val, kind='stable')
        ttb_winner = -1
        for j in order:
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break
        return margin, tally_winner, ttb_winner

    info = [analyze(r) for _, r in data.iterrows()]
    margins = np.array([x[0] for x in info], dtype=float)
    tally_w = np.array([x[1] for x in info], dtype=int)
    ttb_w = np.array([x[2] for x in info], dtype=int)
    resp = data['response'].to_numpy().astype(int)

    # Group 1: tally-tie trials where TTB is decisive (parameter-free for Tallying)
    tie_mask = (tally_w == -1) & (ttb_w >= 0)
    # Group 2: trials where tally margin is >= 2 AND TTB points the other way
    dis_mask = (np.abs(margins) >= 2) & (tally_w >= 0) & (ttb_w >= 0) & (tally_w != ttb_w)

    if tie_mask.sum() == 0 and dis_mask.sum() == 0:
        return float('nan')

    if dis_mask.sum() > 0:
        p_tally = float(np.mean(resp[dis_mask] == tally_w[dis_mask]))
    else:
        p_tally = 0.5
    if tie_mask.sum() > 0:
        p_ttb_tie = float(np.mean(resp[tie_mask] == ttb_w[tie_mask]))
    else:
        p_ttb_tie = 0.5

    return p_tally - p_ttb_tie

```

**Observed (real) value:** 0.1744 (var=0.0127)
**Predicted under pi_7:** 0.2300 (var=0.0149)
**Predicted under pi_8:** 0.2117 (var=0.0142)

### Experiment 5
**Design**
  A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    V = np.array([0.68, 0.96, 0.52, 0.88, 0.58, 0.78], dtype=float)

    hits = 0.0
    n = 0.0
    for _, row in data.iterrows():
        try:
            a = np.asarray([float(x) for x in row['option_a_ratings']], dtype=float)
            b = np.asarray([float(x) for x in row['option_b_ratings']], dtype=float)
        except Exception:
            continue
        k = int(min(a.shape[0], b.shape[0]))
        if k == 0:
            continue
        a = a[:k]
        b = b[:k]
        if V.shape[0] >= k:
            v = V[:k]
        else:
            v = np.concatenate([V, np.full(k - V.shape[0], 0.55)])[:k]

        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        # keep only unit-tally TIES with at least one discriminating cue
        if a_wins != b_wins or a_wins == 0:
            continue

        w = v ** 2.0
        d = float(np.sum(w * (a - b)))
        # discard tie trials whose validity-weighted margin is itself
        # near zero (no clear "higher-validity side" to predict)
        if abs(d) < 0.15:
            continue

        resp = row['response']
        try:
            resp = int(resp)
        except Exception:
            continue
        if resp not in (0, 1):
            continue

        chose_a = (resp == 0)
        picked_high_validity = chose_a if d > 0 else (not chose_a)
        hits += 1.0 if picked_high_validity else 0.0
        n += 1.0

    if n == 0:
        return 0.5
    return float(hits / n)
```

**Observed (real) value:** 0.2856 (var=0.0160)
**Predicted under pi_7:** 0.2778 (var=0.0058)
**Predicted under pi_8:** 0.2794 (var=0.0061)

### Experiment 6
**Design**
  A=[1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    v = np.array([0.58, 0.97, 0.53, 0.80, 0.66, 0.90], dtype=float)
    w = v ** 2.0  # AVAI-style amplified weights (gamma=2)

    num = 0.0
    den = 0.0
    for a_raw, b_raw, resp in zip(data['option_a_ratings'],
                                  data['option_b_ratings'],
                                  data['response']):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        n = int(min(a.shape[0], b.shape[0], w.shape[0]))
        if n == 0:
            continue
        a = a[:n]
        b = b[:n]
        ww = w[:n]
        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        if a_wins != b_wins:
            continue  # not a tally tie
        d = float(np.sum(ww * (a - b)))
        if abs(d) < 0.30:
            continue  # tie trials whose weighted margin is negligible
        den += 1.0
        if d > 0:
            num += 1.0 if int(resp) == 0 else 0.0
        else:
            num += 1.0 if int(resp) == 1 else 0.0

    if den == 0:
        return 0.5
    return num / den
```

**Observed (real) value:** 0.1500 (var=0.0194)
**Predicted under pi_7:** 0.1500 (var=0.0133)
**Predicted under pi_8:** 0.1217 (var=0.0081)

### Experiment 7
**Design**
  A=[1, 1, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 0, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 1]
  A=[1, 1, 0, 1, 0, 1]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    correct = []
    for _, row in data.iterrows():
        try:
            a = np.asarray(list(row['option_a_ratings']), dtype=float).ravel()
            b = np.asarray(list(row['option_b_ratings']), dtype=float).ravel()
        except Exception:
            continue
        n = int(min(a.size, b.size))
        if n == 0:
            continue
        d = a[:n] - b[:n]
        nz = d[d != 0.0]
        if nz.size == 0:
            continue
        # dominance trial: every discriminating cue favours the same option
        if not (np.all(nz > 0) or np.all(nz < 0)):
            continue
        dominant_is_a = bool(np.all(nz > 0))
        try:
            resp = int(row['response'])
        except Exception:
            continue
        chose_a = (resp == 0)
        correct.append(1.0 if (chose_a == dominant_is_a) else 0.0)

    if len(correct) == 0:
        return 0.5
    return float(np.mean(correct))
```

**Observed (real) value:** 0.7427 (var=0.0173)
**Predicted under pi_7:** 0.7467 (var=0.0048)
**Predicted under pi_8:** 0.7487 (var=0.0072)

### Experiment 8
**Design**
  A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[1, 0, 1, 0, 0, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    # design validities (fixed by the experimental design)
    v_design = np.array([0.95, 0.70, 0.55, 0.95, 0.70, 0.55], dtype=float)
    hits = 0
    n = 0
    for a_raw, b_raw, r in zip(data['option_a_ratings'],
                               data['option_b_ratings'],
                               data['response']):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        k = int(min(a.size, b.size))
        if k == 0:
            continue
        if v_design.size >= k:
            vv = v_design[:k]
        else:
            vv = np.linspace(0.95, 0.55, k)
        d = a[:k] - b[:k]
        idx = np.nonzero(d != 0.0)[0]
        if idx.size != 2:
            continue
        j1 = int(idx[0])
        j2 = int(idx[1])
        # the two discriminating cues must point to OPPOSITE options
        if d[j1] * d[j2] > 0:
            continue
        # and the LATER cue must be the LESS valid one
        if not (vv[j2] < vv[j1]):
            continue
        late_choice = 0 if d[j2] > 0 else 1  # 0 = A, 1 = B
        try:
            resp = int(r)
        except Exception:
            continue
        n += 1
        if resp == late_choice:
            hits += 1
    if n == 0:
        return float('nan')
    return float(hits) / float(n)
```

**Observed (real) value:** 0.5067 (var=0.0112)
**Predicted under pi_7:** 0.4817 (var=0.0130)
**Predicted under pi_8:** 0.4725 (var=0.0109)

### Experiment 9
**Design**
  A=[1, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 1, 0, 1, 0, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 1, 0]  B=[1, 0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 0, 1]  B=[0, 1, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    hits = []
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float).ravel()
        b = np.asarray(row['option_b_ratings'], dtype=float).ravel()
        n = int(min(a.size, b.size))
        if n == 0:
            continue
        d = a[:n] - b[:n]
        disc = np.nonzero(d != 0.0)[0]
        m = int(disc.size)
        if m < 2:
            continue
        fav_a = d[disc] > 0.0
        # skip unanimous (dominance) displays: both theories agree there
        if fav_a.all() or (~fav_a).all():
            continue
        # option favoured by the LAST-READ (highest-index) discriminating row
        last_idx = disc[-1]
        late_option = 0 if d[last_idx] > 0.0 else 1
        try:
            resp = int(row['response'])
        except Exception:
            continue
        hits.append(1.0 if resp == late_option else 0.0)

    if len(hits) == 0:
        return float('nan')
    return float(np.mean(hits))
```

**Observed (real) value:** 0.4383 (var=0.0023)
**Predicted under pi_7:** 0.4508 (var=0.0038)
**Predicted under pi_8:** 0.4408 (var=0.0032)

### Experiment 10
**Design**
  A=[1, 0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Design validities (fixed by the experiment); fallback handles other lengths.
    V_DESIGN = np.array([0.955, 0.545, 0.950, 0.540, 0.945, 0.535], dtype=float)
    LAM = 1.6
    DELTA = 0.9

    hits = []
    for _, row in data.iterrows():
        try:
            a = np.asarray(list(row['option_a_ratings']), dtype=float).ravel()
            b = np.asarray(list(row['option_b_ratings']), dtype=float).ravel()
        except Exception:
            continue
        n = int(min(a.size, b.size))
        if n == 0:
            continue
        a = a[:n]
        b = b[:n]

        if V_DESIGN.size >= n:
            v = V_DESIGN[:n]
        else:
            v = np.concatenate([V_DESIGN, np.full(n - V_DESIGN.size, 0.545)])[:n]

        d = a - b
        disc = np.nonzero(d != 0.0)[0]
        m = int(disc.size)
        if m < 2:
            continue

        # salience of each discriminating row: recency x (1 - validity)
        j = disc.astype(float)
        logu = LAM * j + DELTA * np.log(np.clip(1.0 - v[disc], 1e-9, None))
        logu = logu - np.max(logu)
        u = np.exp(logu)
        top = disc[int(np.argmax(u))]
        top_owner_a = bool(d[top] > 0.0)  # True -> the salient row favours A

        n_a = int(np.sum(d[disc] > 0.0))
        n_b = m - n_a

        # include only trials where the row-count tally does NOT strictly
        # agree with the most salient (late / low-validity) discriminating row
        if n_a > n_b:
            majority_a = True
            tie = False
        elif n_b > n_a:
            majority_a = False
            tie = False
        else:
            majority_a = None
            tie = True
        if (not tie) and (majority_a == top_owner_a):
            continue

        try:
            resp = int(row['response'])
        except Exception:
            continue
        chose_a = (resp == 0)
        hits.append(1.0 if (chose_a == top_owner_a) else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.7697 (var=0.0057)
**Predicted under pi_7:** 0.7686 (var=0.0034)
**Predicted under pi_8:** 0.7611 (var=0.0025)

### Experiment 11
**Design**
  A=[1, 0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 0, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    V_DESIGN = np.array([0.958, 0.546, 0.954, 0.542, 0.950, 0.538], dtype=float)
    LAM = 1.5      # reading-order recency gain (shared by both accounts)
    GAM = 0.9      # oddness exponent on (1 - validity) (shared by both accounts)

    hits = 0.0
    tot = 0.0

    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()

    for a_raw, b_raw, resp in zip(a_col, b_col, r_col):
        try:
            a = np.asarray(a_raw, dtype=float).ravel()
            b = np.asarray(b_raw, dtype=float).ravel()
        except Exception:
            continue
        n = int(min(a.size, b.size))
        if n == 0:
            continue
        a = a[:n]
        b = b[:n]

        if n == V_DESIGN.size:
            v = V_DESIGN.copy()
        else:
            v = np.linspace(0.95, 0.55, n)
        v = np.clip(v, 0.5, 0.995)

        diff = a - b
        disc = np.nonzero(diff != 0.0)[0]
        m = int(disc.size)
        if m < 2:
            continue
        signs = np.sign(diff[disc])
        # keep only CONFLICT displays (both options win at least one row)
        if np.all(signs > 0) or np.all(signs < 0):
            continue

        logw = GAM * np.log(np.clip(1.0 - v[disc], 1e-9, None)) + LAM * disc.astype(float)
        k = int(np.argmax(logw))
        favored_a = signs[k] > 0.0

        try:
            r = int(resp)
        except Exception:
            continue
        chose_a = (r == 0)
        hits += 1.0 if (chose_a == favored_a) else 0.0
        tot += 1.0

    if tot == 0.0:
        return float('nan')
    return float(hits / tot)
```

**Observed (real) value:** 0.8189 (var=0.0074)
**Predicted under pi_7:** 0.7714 (var=0.0022)
**Predicted under pi_8:** 0.7536 (var=0.0025)

### Experiment 12
**Design**
  A=[1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 0]
  A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    # Design validities (fixed by the experiment)
    v_default = np.array([0.53, 0.72, 0.86, 0.93, 0.99], dtype=float)

    def weights(n):
        if n <= v_default.size:
            v = v_default[:n]
        else:
            v = np.concatenate([v_default, np.full(n - v_default.size, 0.99)])
        v = np.clip(v, 0.5, 0.9995)
        logw = 0.9 * np.log(1.0 - v) + 1.5 * np.arange(n, dtype=float)
        logw = logw - np.max(logw)
        return np.exp(logw)

    wcache = {}
    hits = []
    for a_raw, b_raw, resp in zip(data['option_a_ratings'].values,
                                  data['option_b_ratings'].values,
                                  data['response'].values):
        try:
            a = np.asarray(list(a_raw), dtype=float).ravel()
            b = np.asarray(list(b_raw), dtype=float).ravel()
        except Exception:
            continue
        n = int(min(a.size, b.size))
        if n == 0:
            continue
        a = a[:n]
        b = b[:n]
        d = a - b
        nA = int(np.sum(d > 0))
        nB = int(np.sum(d < 0))
        # keep only CONFLICT displays (both options favoured by at least one expert)
        if nA == 0 or nB == 0:
            continue
        if n not in wcache:
            wcache[n] = weights(n)
        w = wcache[n]
        S = float(np.sum(np.sign(d) * w))
        if S == 0.0:
            continue
        fav = 0 if S > 0 else 1  # response coding: 0 = A, 1 = B
        try:
            r = int(resp)
        except Exception:
            continue
        hits.append(1.0 if r == fav else 0.0)

    if len(hits) == 0:
        return float('nan')
    return float(np.mean(hits))
```

**Observed (real) value:** 0.4178 (var=0.0051)
**Predicted under pi_7:** 0.3878 (var=0.0029)
**Predicted under pi_8:** 0.4492 (var=0.0019)

### Experiment 13
**Design**
  A=[1, 1, 0, 1, 1, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[1, 1, 0, 1, 0, 1]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0, 1]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 1, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 0, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 1, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 1]  B=[1, 1, 0, 1, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[1, 1, 1, 1, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Design-fixed printed validities for this experiment
    v = np.array([0.55, 0.90, 0.75, 0.95, 0.68, 0.85], dtype=float)
    n = v.size
    j = np.arange(n, dtype=float)

    # MDSC-NV salience: near-argmax doubt rule (delta=15, lam=0.115*15)
    logu = 15.0 * np.log(np.clip(1.0 - v, 1e-12, None)) + 1.725 * j
    # SSI-DS+ compensatory summation weights (gamma=0.9, lam=1.5)
    w_ssi = np.power(np.clip(1.0 - v, 1e-12, None), 0.9) * np.exp(1.5 * j)

    hits = 0
    tot = 0
    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()

    for a_raw, b_raw, r in zip(a_col, b_col, r_col):
        try:
            a = np.asarray(a_raw, dtype=float).ravel()
            b = np.asarray(b_raw, dtype=float).ravel()
        except Exception:
            continue
        if a.size != n or b.size != n:
            continue
        d = a - b
        disc = np.nonzero(d != 0.0)[0]
        if disc.size == 0:
            continue

        # MDSC-NV predicted winner: verdict of the single most doubtful
        # discriminating expert
        k = disc[int(np.argmax(logu[disc]))]
        pred_mdsc = 0 if d[k] > 0 else 1

        # SSI-DS+ predicted winner: sign of the weighted sum over all
        # discriminating rows (coalitions add, late rows dominate)
        E = float(np.sum(np.sign(d[disc]) * w_ssi[disc]))
        if E == 0.0:
            continue
        pred_ssi = 0 if E > 0 else 1

        if pred_ssi == pred_mdsc:
            continue  # non-diagnostic trial type

        tot += 1
        try:
            resp = int(r)
        except Exception:
            continue
        if resp == pred_mdsc:
            hits += 1

    if tot == 0:
        return 0.5
    return float(hits) / float(tot)
```

**Observed (real) value:** 0.5925 (var=0.0026)
**Predicted under pi_7:** 0.6833 (var=0.0038)
**Predicted under pi_8:** 0.6367 (var=0.0043)

### Experiment 14
**Design**
  A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 1, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 0, 1]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[1, 0, 1, 1, 0, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 1]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    V_DESIGN = np.array([0.51, 0.72, 0.80, 0.90, 0.95, 0.985], dtype=float)

    GAMMA = 0.90
    LAM = 1.50
    DELTA = 15.0
    LAM_M = 1.725

    hits = 0.0
    tot = 0.0

    for a_raw, b_raw, resp in zip(data['option_a_ratings'],
                                  data['option_b_ratings'],
                                  data['response']):
        try:
            a = np.asarray(list(a_raw), dtype=float).ravel()
            b = np.asarray(list(b_raw), dtype=float).ravel()
        except Exception:
            continue
        n = int(min(a.size, b.size))
        if n == 0:
            continue
        a = a[:n]
        b = b[:n]

        if n <= V_DESIGN.size:
            v = V_DESIGN[:n]
        else:
            v = np.concatenate([V_DESIGN, np.full(n - V_DESIGN.size, 0.985)])
        v = np.clip(v, 0.5, 0.9995)

        diff = a - b
        disc = np.nonzero(diff != 0.0)[0]
        if disc.size == 0:
            continue
        signs = np.sign(diff[disc])

        # --- SSI-DS+ side: max-normalised weighted SUM, late/high-validity dominated
        logw = GAMMA * np.log(np.clip(1.0 - v[disc], 1e-12, None)) + LAM * disc.astype(float)
        logw = logw - np.max(logw)
        w = np.exp(logw)
        E = float(np.sum(signs * w))
        if abs(E) < 1e-9:
            continue
        ssi_side = 1.0 if E > 0 else -1.0

        # --- MDSC-NV side: single most-doubtful DISCRIMINATING row decides
        sal = DELTA * np.log(np.clip(1.0 - v[disc], 1e-12, None)) + LAM_M * disc.astype(float)
        k = int(np.argmax(sal))
        mdsc_side = float(signs[k])
        if mdsc_side == 0.0:
            continue

        # keep only trials where the two accounts point to OPPOSITE options
        if mdsc_side == ssi_side:
            continue

        try:
            r = int(resp)
        except Exception:
            continue
        chose_a = (r == 0)
        ssi_favours_a = (ssi_side > 0)
        hits += 1.0 if (chose_a == ssi_favours_a) else 0.0
        tot += 1.0

    if tot == 0.0:
        return 0.5
    return float(hits / tot)
```

**Observed (real) value:** 0.2608 (var=0.0079)
**Predicted under pi_7:** 0.2592 (var=0.0042)
**Predicted under pi_8:** 0.2729 (var=0.0043)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the 12 legacy experiments the two theories are nearly indistinguishable: both reproduce the anti-validity signatures (Exp 3: .288/.273 vs .282; Exp 5: .278/.279 vs .286; Exp 6: .150/.122 vs .150; Exp 14: .259/.273 vs .261), the dominance level (Exp 7: .747/.749 vs .743), the near-degenerate-label recency designs (Exp 10: .769/.761 vs .770; Exp 11: .771/.754 vs .819), and the two-cue conflict cell (Exp 8: ~.48 vs .507). Those experiments simply do not separate them \\u2014 they only confirm the shared core (choice is driven by low-validity/late rows, with a load-dependent leak toward chance).\n\nThe two new experiments were built precisely to separate them, and both give the SAME answer: neither theory is right, and the truth lies between them. In Exp 1 the lone .68 dissenter against a 2\\u20133 member coalition of more valid experts was followed on .546 of trials; in Exp 2 the lone .62 dissenter against the same kind of coalition was followed on .468. pi_7 (near-argmax one-reason plus null-verdict capture) predicted .64 in both cases \\u2014 too high by .09 and .17, i.e. it over-commits to the doubtful dissenter and wrongly asserts that coalitions buy nothing. pi_8 (rank-geometric with the absolute-doubt trigger flipping \\u03c6 to ~.88) predicted a crash to .27 and .23 \\u2014 wrong by .28 and .24, and wrong in DIRECTION in Exp 1. So pi_8's two sharpest quantitative commitments (the v_c \\u2248 .575 trigger and the resulting chance-crossing coalition crash) are decisively falsified, while pi_7's commitment (coalitions are strictly irrelevant, choice stays ~.64\\u2013.67) is falsified more gently, in the same direction in both experiments.\n\nThe real pattern is graded compensation with a strong anti-validity weight profile: one near-chance/low-validity dissenter is worth roughly as much as two-to-three more valid experts combined \\u2014 producing near-chance behaviour in exactly these cells \\u2014 rather than either 'one reason always wins' or 'the tally always wins'. Neither existing theory encodes that balance point; pi_7 has no summation at all and pi_8's summation is switched on/off by a threshold that the data do not support. Also relevant: pi_8's Exp-12 prediction (.449 vs observed .418) and pi_7's Exp-13 over-prediction (.683 vs .593) both point the same way \\u2014 the decisive row's dominance is graded and partially erodable, not all-or-none.\n\npi_7 is the better of the two overall (smaller absolute error in both diagnostic experiments and no directional failure), so it should be retained as the incumbent; pi_8 is the theory that should be replaced, and it should be replaced by a genuinely different competitor \\u2014 a graded, normalized anti-validity accumulator \\u2014 rather than by another variant of triggered one-reason choice.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace pi_8 with a GRADED ANTI-VALIDITY COMPENSATORY ACCUMULATOR with divisive normalization (call it 'NAVA' \\u2014 Normalized Anti-Validity Accumulation). Core claims, which differ from BOTH incumbents:\n\n(1) NO one-reason step and NO trigger. All discriminating rows contribute simultaneously with weight w_j = (1 \\u2212 v_j)^\\u03b3 \\u00b7 exp(\\u03bb\\u00b7j), i.e. doubt (distance of the printed validity from chance) raised to a moderate power \\u03b3 \\u2248 1.5\\u20132.5, times a modest reading-position/recency gain \\u03bb \\u2248 0.15\\u20130.6. This keeps the anti-validity direction that every legacy experiment demands (Exp 3, 5, 6, 12, 14) and the late-row advantage that the near-degenerate designs demand (Exp 10, 11) without asserting argmax.\n\n(2) DIVISIVE NORMALIZATION (the new mechanism). Evidence is E = \\u03a3_{j\\u2208disc} s_j w_j / (\\u03c3 + \\u03a3_{j\\u2208disc} w_j), p(A) = \\u03c3(\\u03b2E). Because the denominator grows with every row that speaks, a coalition of more valid experts CAN erode a doubtful dissenter's lead but rarely overturns it decisively \\u2014 exactly the near-chance values observed in the two diagnostic cells (.546 with the .68 dissenter in a 5-cue display, .468 with the .62 dissenter in a 6-cue display). Calibrate \\u03b3, \\u03bb, \\u03b2 so that the 1-vs-2 and 1-vs-3 coalition cells of BOTH new designs land in [.44, .58] while 1-vs-0 and 2-vs-0 dominance cells stay near .85 and 6-vs-0 near .72\\u2013.75 (Exp 7 observed .743).\n\n(3) NO null-verdict capture: silent rows are irrelevant (pi_7's capture mechanism was never independently supported and is what makes it over/under-shoot inconsistently). Instead, all pull toward chance comes from a single load-based lapse p(guess) = 1 \\u2212 (1\\u2212\\u03b5\\u2080)\\u00b7\\u03c1^max(m\\u22122,0), \\u03c1 \\u2248 .80\\u2013.90, \\u03b5\\u2080 \\u2248 .10\\u2013.25 \\u2014 enough to hold 6-row dominance at ~.74.\n\n(4) Degenerate printed labels (gap < .002): weights collapse to equal, so behaviour becomes a normalized unweighted tally with the SAME \\u03b2 (no special \\u03ba), giving exact chance on 1-1 splits and ~.8 on 2-0 splits after the leak.\n\nKey testable contrasts against pi_7: NAVA says coalition size matters monotonically and can drive choice to (but not far past) chance, whereas pi_7 says coalitions buy nothing; NAVA says non-discriminating rows are strictly irrelevant, whereas pi_7 predicts a 13\\u201317 point dominance drop whenever the globally most doubtful expert is silent \\u2014 a contrast that neither new experiment cleanly isolated and that the next design should target directly (matched dominance displays that differ only in which row is silent)."
}
```

## Usage

```json
{
  "input_tokens": 38194,
  "output_tokens": 4756
}
```
