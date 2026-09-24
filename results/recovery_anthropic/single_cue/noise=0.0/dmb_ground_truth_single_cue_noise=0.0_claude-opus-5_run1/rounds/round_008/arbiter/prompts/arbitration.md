# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_7" and "pi_10") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_7" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_10" (THEORY 2 below). Justify your choice.


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

## THEORY 2 — pi_10
DTG-2R/OC-sharp (load-graded compensatory limb) — Doubt-Threshold-Gated one-reason choice with compensatory fallback, where the doubt gate is an ABSOLUTE and SHARP near-chance criterion, the one-reason limb is degraded by OPPOSITION rather than by raw display length, and the compensatory limb is degraded by DISPLAY LOAD more steeply than previously assumed. (1) One-reason adjudication is commanded only by an expert whose PRINTED validity is essentially a coin flip. The trigger is g_j = sigmoid((theta - v_j)/tau) evaluated only over rows that actually discriminate (P_trig = max_{j in disc} g_j), with theta ~ .558-.588 and tau ~ .006-.013. An expert at .52-.55 is 'newsworthy' (g >= .9) while an expert at .62-.68 is NOT (g <= .02): a merely mediocre expert is ordinary evidence, not a scandal demanding adjudication. Silent near-chance experts are irrelevant, so no null-verdict capture and no doubt-gradient leak are needed. (2) When a genuinely sub-threshold expert speaks, its verdict decides the trial essentially independently of how MANY higher-validity rows oppose it; among discriminating rows the consulted verdict is drawn from softmax(log g_j + mu*j), so the recency tilt mu only arbitrates BETWEEN near-chance rows. (3) Because a single row is consulted, merely scanning extra rows costs the one-reason limb almost nothing (rho_t ~ .985-1.0), but holding a genuine RIVAL COALITION does cost it: each minority row beyond the first charges rho_tc (~.62-.86). Hence a near-chance dissenter standing ALONE against 1-5 opposing rows is followed at ~.82-.86 flat in coalition size (Exps 15, 16), whereas BALANCED splits drift toward chance. (4) When no near-chance expert discriminates — including displays whose least valid dissenter is only mid-doubt (v ~ .58-.68) — evidence is compensatory: anti-validity, weakly recency-tilted weights w_j = (1-v_j)^gamma * exp(lam*j), divisively normalised to the loudest dissent, E = sum(s_j*what_j)/(sigma + sum(what_j)), p = sigmoid(beta*E). NEW, SHARPENED CLAIM: the compensatory rule is a genuinely effortful integration whose thread is lost quickly as the display lengthens — every discriminating row beyond the second multiplies the retained evidence by rho_m ~ .84-.93, appreciably steeper than the one-reason limb's near-free scan. This asymmetry is the theory's signature: one-reason choice is cheap to run on long displays and expensive only under opposition, whereas compensatory integration is expensive in sheer list length. Consequently long unanimous displays stay confident but clearly below ceiling (~.74-.76), and long conflict displays settle appreciably closer to chance than short ones, while short displays (m=2-3) are essentially unaffected. (5) If any two printed validities are within .002 the experts cannot be individuated, the doubt gate has no referent, and choice collapses to a graded unweighted row tally (exact coin flip on 1-1 splits). No feedback is given, so the rule is stationary across the block.

`predict(parameters, state, history) -> np.ndarray`:
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
**Validities (n_features=6):** [0.7, 0.96, 0.64, 0.88, 0.52, 0.66]

**Trial pairs (n=24):**
  trial 1: A=[1, 0, 1, 1, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  trial 2: A=[1, 0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1, 0]
  trial 3: A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 4: A=[0, 0, 1, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0]
  trial 5: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  trial 7: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 1, 0]
  trial 8: A=[0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 1, 1]
  trial 9: A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0, 1]
  trial 10: A=[1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  trial 11: A=[1, 0, 0, 1, 1, 0]  B=[1, 0, 1, 1, 0, 1]
  trial 12: A=[1, 0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  trial 13: A=[0, 0, 0, 1, 1, 0]  B=[1, 0, 1, 1, 0, 1]
  trial 14: A=[1, 0, 1, 1, 0, 1]  B=[0, 0, 0, 1, 1, 0]
  trial 15: A=[0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 0, 1]
  trial 16: A=[1, 1, 1, 1, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  trial 17: A=[1, 0, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  trial 18: A=[1, 0, 1, 1, 1, 0]  B=[1, 0, 0, 1, 1, 1]
  trial 19: A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  trial 20: A=[1, 0, 1, 1, 1, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 21: A=[0, 1, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0, 0]
  trial 22: A=[1, 1, 1, 1, 0, 0]  B=[0, 1, 0, 0, 0, 1]
  trial 23: A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  trial 24: A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 0, 1, 1, 1]

**Rationale:** GOAL: separate MDSC-NV (advocated) from DTG-2R/OC-sharp (competitor). Both are anti-validity with a weak position gain and a load leak, so they agree about WHICH row is decisive in every display here. They differ on exactly three mechanisms, and the design isolates each: (A) RELATIVE (near-argmax doubt, delta=13-17) vs ABSOLUTE-SHARP doubt trigger (theta~.558-.588, tau~.006-.013); (B) SINGLE-CUE always vs COMPENSATORY SUMMATION whenever no sub-threshold expert speaks; (C) load leak on EVERY display (MDSC, rho^(m-2)) vs a nearly-free scan in the one-reason limb (DTG rho_t~.99, cost only for a rival coalition, rho_tc^(k-1)).

VALIDITIES [0.70, 0.96, 0.64, 0.88, 0.52, 0.66] at columns c0..c5. Engineering: (i) EXACTLY ONE expert, c4 = .52, lies below DTG's trigger; every other expert (.64, .66, .70, .88, .96) is comfortably above it (g(.64) <= .018, g(.66) <= .004 at the worst corner theta=.588/tau=.013), so DTG is in take-the-least mode iff c4 discriminates and compensatory otherwise. (ii) c4 is ALSO MDSC's global doubt anchor at every corner of its box: with score = log(1-v)+p*j (p = pos_ratio in [.09,.14]) the ordering is c4 (-0.27) > c5 (-0.50) > c2 (-0.79) > c0 (-1.20) > c3 > c1, with the c4-c5 margin 0.205-0.255 log-units (x delta = 2.7-4.3) and the c5-c2 margin 0.21-0.36 (x delta = 2.8-6.2). So MDSC is effectively 'whoever wins c4; if c4 is silent, whoever wins c5; and if c4 is silent, capture drags the trial ~q/2 toward chance'. Validity does not track column index, so 'read the first/last row' cannot masquerade as either model, and the minimum label gap (.02) is 10x both degeneracy tolerances.

12 base structures, each with its exact A/B mirror (24 unique pairs, K=4 -> 96 trials), perfectly key-balanced. Silent rows carry matched 1s AND matched 0s so displays are dense. Predictions use MDSC (delta=15, pos_ratio=.115, q=.41, eps0=.24, rho=.81) and DTG (theta=.573, tau=.010, mu=.7, gamma=1.9, lam=.11, sigma=.55, beta=2.2, eps_t=.28, rho_t=.99, rho_tc=.74, eps0=.20, rho_m=.885, rho_c=.82).

BLOCK 1 - CROWN JEWEL: FLAT-vs-DECLINING ANCHOR LADDER (bases 5,6,7,8 = pairs 9-16). A always wins c4 (.52, the sub-threshold anchor) while B's opposing coalition of MORE valid experts grows 1 -> 2 -> 3 -> 5 rows. DTG is locked in its one-reason limb with k_conf = 1 throughout, so rho_tc never fires and rho_t~.99 makes extra rows nearly free: it must predict an essentially FLAT profile (.854, .849, .846, .839). MDSC pays its raw-m load leak on every added row: (.857, .786, .732, .650). A ~21-point monotone decline vs a <2-point flat line, with the decisive row and its verdict held constant, is the parameter-robust signature. The unanimity ladder (bases 1,2,3 = pairs 1-6: 2-0, 4-0, 6-0 all carried by the anchor) repeats the same test without any conflict at all: DTG .860/.853/.845 (flat) vs MDSC .875/.746/.663 (steep). Base 1 doubles as the shared lapse baseline (both ~.86-.88).

BLOCK 2 - CROWN JEWEL: THE ABSOLUTE THRESHOLD (bases 7 vs 11, 6 vs 10, 5 vs 9). Structurally identical 1-vs-1, 1-vs-2 and 1-vs-3 displays in which ONLY the identity of the lone dissenter changes, from c4 (.52, below DTG's trigger) to c5 (.66, above it). DTG must swing across chance: 1-vs-1 .854 -> .530, 1-vs-2 .849 -> .471, 1-vs-3 .846 -> .468 (35-38 point, chance-crossing swings), because a merely mediocre .66 dissenter is ordinary evidence that the summed .64/.70/.88 coalition overturns. MDSC must swing only ~10-13 points and must stay ABOVE chance in every cell: .857 -> .723, .786 -> .678, .732 -> .644 (the whole swing is null-verdict capture on the silent .52 row, not a change of rule). Small-swing/always-above-chance vs large-swing/chance-crossing is readable with no free parameters.

BLOCK 3 - COALITIONS BUY NOTHING vs SUMMATION (bases 9,10,11 = pairs 17-22). With the anchor silent, A wins c5 (.66, MDSC's doubt argmax among the discriminating rows, by 0.21-0.36 log-units) and B's coalition grows c2 -> c2+c0 -> c2+c0+c3. MDSC: .723, .678, .644 - a shallow load-only decline that never crosses chance, because only the argmax row's verdict is used. DTG's compensatory limb sums the max-normalised weights (c5=1, c2=.80, c0=.46, c3=.08) and crashes: .530, .471, .468. Note base 9 (1-vs-1, .66 vs .64) is the sharpest single cell for the SHARPENED doubt gain: the two rows are within 0.06 log-validity, so a share-based rule (or any summation rule) is pinned near chance (DTG .48-.56 across its lam range) whereas near-argmax hands the display entirely to c5 (MDSC .67-.74 across its box).

BLOCK 4 - TALLY TIE AND COMPENSATORY LOAD (bases 12, 4 = pairs 23-24, 7-8). Base 12 is an exact 2-vs-2 row tie (A wins c5,c1; B wins c2,c0): MDSC .644 (the argmax decides), DTG .480 (summed weights nearly cancel, then the k_conf=2 opposition/conflict leak flattens it) - a 16-point split at a tally tie that no vote-counting account can produce in either direction. Base 4 is a unanimous 5-0 display with the anchor SILENT, to be read against base 3 (unanimous 6-0, anchor speaks): DTG says ADDING the sixth agreeing row makes the display much EASIER (.697 -> .845, because the .52 row flips it into the cheap one-reason limb), MDSC says the gain is small (.622 -> .663, capture satisfied but load increased).

INTERPRETATION: if accuracy is flat in display length whenever the .52 expert speaks (~.84-.86 from 1-vs-0 through 1-vs-5 and from 2-0 through 6-0) but collapses to chance as soon as the lone dissenter is .66 instead of .52, the absolute sharp trigger plus load-graded compensatory integration is confirmed and MDSC-NV is falsified. If instead accuracy declines steadily with the raw number of discriminating rows even when the verdict-bearing row is unchanged (.875 -> .66 unanimous; .857 -> .650 at 1-vs-5), and the option holding the single most doubtful row is still chosen ~.64-.72 when two or three more valid experts oppose it, MDSC-NV's near-argmax doubt rule with null-verdict capture wins.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



### METRIC
Rationale:
This isolates the 'coalitions buy nothing vs. compensatory summation' cell (design bases 9-12, pairs 17-24; 32 trials/subject). Selection rule: the .52 anchor expert (index 4) is silent, the mid-doubt .66 expert (index 5) discriminates, and at least one MORE valid expert (indices 0-3, validities .70/.96/.64/.88) opposes it. The metric is the rate of choosing the option backed by the single .66 expert against that opposing coalition.

MDSC-NV: with the anchor silent, the doubt argmax among discriminating rows is c5 (.66), leading by 0.21-0.36 log-units, so with a near-argmax doubt gain the verdict of c5 decides the trial; extra opposing rows only add a load leak and null-verdict capture drags things partway toward chance. Predicted ~.64-.72 pooled (~.67), reliably ABOVE chance in every included cell.

DTG-2R/OC: .66 is above the absolute sharp trigger (g<=.004), so all these displays fall into the compensatory limb, where the normalised anti-validity weights of c2 (.80), c0 (.46), c3 (.08) sum against c5 (1.0) and the k_conf/load leaks flatten the result. Predicted ~.47-.53 pooled (~.49), i.e. essentially AT or BELOW chance.

So the two theories straddle chance with a ~.18-.19 gap, while each subject contributes 32 trials (binomial SD ~.09) and the parameter boxes produce only ~.03 of extra between-subject spread, giving a tight, large-N-robust contrast. Neither value is trivially fixed by my model's machinery: it is an empirical question whether a lone mediocre dissenter beats a coalition of more valid experts.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    hits = 0.0
    tot = 0.0
    for a_raw, b_raw, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        try:
            a = [int(x) for x in a_raw]
            b = [int(x) for x in b_raw]
        except Exception:
            continue
        if len(a) < 6 or len(b) < 6:
            continue
        # expert 4 (validity .52, the near-chance 'anchor') must be SILENT
        if a[4] != b[4]:
            continue
        # expert 5 (validity .66, mid-doubt) must discriminate
        d5 = a[5] - b[5]
        if d5 == 0:
            continue
        # there must be a genuine opposing coalition among the more valid experts 0..3
        opposed = False
        for j in range(4):
            dj = a[j] - b[j]
            if dj != 0 and (dj * d5) < 0:
                opposed = True
                break
        if not opposed:
            continue
        fav = 0 if d5 > 0 else 1  # option favoured by the .66 expert
        tot += 1.0
        try:
            rr = int(r)
        except Exception:
            continue
        if rr == fav:
            hits += 1.0
    if tot <= 0:
        return 0.5
    return float(hits / tot)

### RESULTS
- Predicted under pi_7 (simulated): 0.6956 (var=0.0072)
- Predicted under pi_10 (simulated): 0.4906 (var=0.0058)
- Observed on real data: 0.4925 (var=0.0074)

## EXPERIMENT 2 (proposed by pi_10)

### DESIGN
**Validities (n_features=6):** [0.64, 0.96, 0.72, 0.9, 0.52, 0.68]

**Trial pairs (n=24):**
  trial 1: A=[1, 0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0, 1]
  trial 2: A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0, 1]
  trial 3: A=[0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 0, 1]
  trial 4: A=[0, 1, 0, 1, 1, 0]  B=[1, 0, 1, 0, 0, 1]
  trial 5: A=[0, 1, 0, 1, 1, 0]  B=[1, 0, 1, 1, 0, 0]
  trial 6: A=[0, 1, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0, 1]
  trial 7: A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  trial 8: A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  trial 9: A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  trial 10: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 1, 0]
  trial 11: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 12: A=[1, 0, 1, 1, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  trial 13: A=[1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 1]
  trial 14: A=[1, 1, 1, 0, 0, 1]  B=[0, 1, 0, 0, 1, 1]
  trial 15: A=[1, 1, 1, 1, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  trial 16: A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 0, 1, 1, 0]
  trial 17: A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  trial 18: A=[1, 1, 1, 0, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  trial 19: A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1]
  trial 20: A=[1, 0, 1, 1, 1, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 21: A=[1, 1, 1, 1, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  trial 22: A=[0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 1, 1]
  trial 23: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  trial 24: A=[1, 0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1, 0]

**Rationale:** GOAL: separate DTG-2R/OC-sharp (advocated) from MDSC-NV (competitor). Both are anti-validity accounts with a weak recency tilt and a load leak, and this design is deliberately built so they AGREE about which row carries the verdict in every display; they differ only on three mechanisms, each of which is isolated by a matched-load contrast.

VALIDITIES [0.64, 0.96, 0.72, 0.90, 0.52, 0.68] at columns c0..c5. Engineering: (i) EXACTLY ONE expert, c4 = .52, lies below DTG's absolute sharp gate (theta .558-.588, tau .006-.013): g(.52) = .95-1.00, whereas g(.64) <= .018, g(.68) <= .001, g(.72) <= 4e-5. So DTG runs its one-reason limb iff c4 discriminates and its compensatory limb otherwise. (ii) c4 is ALSO MDSC's global doubt anchor at every corner of its box: score_j = log(1-v_j) + pos_ratio*j gives c4 (-0.40 at pos_ratio .09, -0.20 at .14) ahead of the runner-up c5 (.68, last column: -0.69 / -0.44) by .24-.29 log-(1-v) units, i.e. 3.2-4.9 delta units (share .96-.99); with c4 silent the secondary MDSC anchor is unambiguously c5 (ahead of c0 by .33-.58 units). MDSC therefore reduces to 'whoever wins c4; if c4 is silent, whoever wins c5, with capture q dragging the trial toward chance'. (iii) Validity does not track column index, so 'read the first/last row' cannot masquerade as either model; the minimum label gap (.04) is 20x both degeneracy tolerances. (iv) DTG's compensatory weights (gamma 1.9, lam .11) over the non-anchor rows are c5=.199, c0=.144, c2=.111, c3=.018, c1=.002, so c0+c2 (.255) OVERTURNS a lone c5 dissenter: coalitions of merely mediocre experts beat the most doubtful discriminating row for DTG but buy nothing for MDSC.

12 base structures, each with its exact A/B mirror (24 unique pairs, K=4 -> 96 trials), perfectly key-balanced. Silent rows carry matched 1s AND matched 0s so displays are dense. Predictions use DTG (theta=.573, tau=.010, mu=.7, gamma=1.9, lam=.11, sigma=.55, beta=2.2, eps_t=.28, rho_t=.99, rho_tc=.74, eps0=.20, rho_m=.885, rho_c=.82) and MDSC (delta=15, pos_ratio=.115, q=.41, eps0=.24, rho=.81).

BLOCK 1 - CROWN JEWEL: FLAT-vs-DECLINING ANCHOR LADDER (bases 1,2,3 = pairs 1-6). A always wins c4 (.52) while B's coalition of MORE valid experts grows 0 -> 2 -> 5 rows (m = 1, 3, 6). DTG stays in the cheap one-reason limb with k_conf <= 1, so rho_tc never fires and rho_t~.99 makes extra rows nearly free: .860, .856, .846 (FLAT within 2 points). MDSC pays rho^(m-2) on every added row: .880, .808, .664 (a 22-point monotone decline). Same decisive row, same verdict, matched direction: flat vs steep is parameter-robust. Base 1 also pins the shared lapse baseline (~.86-.88 for both).

BLOCK 2 - OPPOSITION-COST vs RAW-LOAD LEAK AT MATCHED m (bases 3 vs 4, 6 vs 5). Base 4 is a 3-vs-3 split at m=6 with the anchor on A's side; base 3 is 1-vs-5 at the same m=6. DTG charges rho_tc per minority row beyond the first, so it must predict a ~16-point DROP (.846 -> .689) purely from the balance of the split, whereas MDSC's leak sees only m and must call them equal (.664 vs .653). The same test at m=4: base 6 (1-vs-3, .853) versus base 5 (2-vs-2, .761) for DTG, against .739/.749 (no ordering) for MDSC. This dissociates a coalition-driven from a length-driven leak with display length held exactly constant.

BLOCK 3 - CROWN JEWEL: THE ABSOLUTE THRESHOLD / COMPENSATORY REVERSAL (bases 7,8,9 = pairs 13-18, read against bases 2 and 6). With the .52 anchor SILENT, A wins c5 (.68 - MDSC's secondary argmax, but ORDINARY evidence for DTG) and B's coalition grows: 1-vs-1 (c0), 1-vs-2 (c0,c2), 1-vs-3 (c0,c2,c3). DTG's compensatory limb sums max-normalised anti-validity weights and crashes through chance: .544, .469, .465. MDSC's near-argmax rule keeps the c5-holder above chance throughout, with only its load leak moving: .722, .683, .648. Base 8 versus base 2 is the sharpest single comparison: two structurally IDENTICAL 1-vs-2 displays against the same coalition (c0,c2) in which only the identity of the lone dissenter changes, .52 -> .68. DTG must swing 39 points ACROSS chance (.856 -> .469); MDSC must swing only ~12 points and stay above chance in both cells (.808 -> .683).

BLOCK 4 - COMPENSATORY LOAD vs CHEAP SCAN ON UNANIMOUS DISPLAYS (bases 10, 11, 12 = pairs 19-24). Base 12 is a unanimous 2-0 with the anchor speaking (DTG .860 / MDSC .880, shared baseline). Base 10 is a unanimous 5-0 with the anchor SILENT and base 11 the unanimous 6-0 obtained by letting the anchor speak too - the two displays differ in exactly one row. DTG says ADDING the sixth agreeing row makes the display markedly EASIER (.698 -> .846, +15 points), because the .52 row flips the trial out of the effortful, rho_m-degraded compensatory integration into the near-free one-reason limb; MDSC says the sixth row barely helps (.622 -> .664, +4) because capture is only partially satisfied while load increases. DTG also predicts long unanimous displays sit confidently at ~.85 when the anchor speaks, well above MDSC's ~.66.

INTERPRETATION: if accuracy is essentially FLAT in display length whenever the .52 expert speaks (~.85 from 1-vs-0 through 1-vs-5 and from 2-0 through 6-0), drops by ~15 points when the split is BALANCED at the same m, and collapses to or below chance as soon as the lone dissenter is .68 instead of .52 (bases 8, 9), the absolute sharp doubt gate with an opposition-costed one-reason limb and a load-costed compensatory limb is confirmed and MDSC-NV is falsified. If instead accuracy declines steadily with the raw number of discriminating rows even when the verdict-bearing row is unchanged (.88 -> .66), balance of the split makes no difference at matched m, and the option holding the most doubtful DISCRIMINATING row is still chosen ~.65-.72 when two or three more valid experts oppose it, MDSC-NV's near-argmax relative doubt rule with null-verdict capture wins.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



### METRIC
Rationale:
The two theories agree on WHICH row carries the verdict in every display of this design; they disagree about (i) whether long displays are cheap when the .52 anchor speaks, and (ii) whether a merely mid-doubt lone dissenter (.68) still dictates the choice when the anchor is silent.

The metric is a within-subject difference of two rates computed from the stimulus structure alone (no theory-specific fitting):

T = P(choose the option favoured by the .52 anchor) on LONG, low-opposition displays (anchor discriminates, m>=4 discriminating rows, at most one row in the minority) -> trials 3, 6, 11, 15, 18, 23 (24 trials/subject).
C = P(choose the option favoured by the .68 last-column row) when the .52 anchor is SILENT and the .68 row stands ALONE against a coalition of 1-3 more valid rows -> trials 7, 8, 9, 19, 20, 21 (24 trials/subject).

DTG-2R/OC-sharp: the one-reason limb is charged for opposition, not for list length, so T stays high (~.85) even at m=6; and with no sub-threshold expert discriminating, the compensatory limb (anti-validity weights normalised to the loudest dissent, load-degraded) pulls C to or below chance (~.47-.52). Predicted T - C ~ +0.35.

MDSC-NV: the load leak rho^(m-2) drags long displays down (T ~ .66-.74) while near-argmax doubt keeps the .68-holder chosen well above chance (C ~ .65-.72). Predicted T - C ~ 0.00.

Both components use 24 trials per subject, so binomial noise is modest, and both theories' predicted rates are driven by parameters with narrow ranges, keeping between-subject variance small relative to the ~0.35 mean gap. A near-zero value falsifies the absolute sharp doubt gate + opposition-costed one-reason limb; a large positive value falsifies MDSC-NV.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    ANCHOR = 4   # printed validity .52  (the only sub-.558 expert)
    SECOND = 5   # printed validity .68, last column (secondary doubt row)

    t_num = 0.0; t_den = 0.0   # anchor-follow rate on LONG, low-opposition displays
    c_num = 0.0; c_den = 0.0   # c5-follow rate when the .52 anchor is SILENT

    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()

    for a_, b_, r_ in zip(a_col, b_col, r_col):
        try:
            a = np.asarray(a_, dtype=float).ravel()
            b = np.asarray(b_, dtype=float).ravel()
            r = int(r_)
        except Exception:
            continue
        if a.size < 6 or b.size < 6:
            continue
        d = a[:6] - b[:6]
        disc = np.nonzero(d != 0.0)[0]
        m = int(disc.size)
        if m == 0:
            continue
        nA = int(np.sum(d[disc] > 0.0))
        nB = m - nA
        kconf = min(nA, nB)
        chose_a = (r == 0)

        anchor_speaks = bool(np.any(disc == ANCHOR))
        second_speaks = bool(np.any(disc == SECOND))

        if anchor_speaks and m >= 4 and kconf <= 1:
            # long display, anchor either alone against a coalition or with everyone
            fav_a = (d[ANCHOR] > 0.0)
            t_den += 1.0
            if chose_a == fav_a:
                t_num += 1.0
        elif (not anchor_speaks) and second_speaks and kconf == 1:
            # anchor silent; is the .68 row the LONE dissenter on its side?
            fav_a_5 = (d[SECOND] > 0.0)
            side_count = nA if fav_a_5 else nB
            if side_count == 1:
                c_den += 1.0
                if chose_a == fav_a_5:
                    c_num += 1.0

    t_rate = (t_num / t_den) if t_den > 0 else 0.5
    c_rate = (c_num / c_den) if c_den > 0 else 0.5
    return float(t_rate - c_rate)

### RESULTS
- Predicted under pi_7 (simulated): 0.0125 (var=0.0220)
- Predicted under pi_10 (simulated): 0.3550 (var=0.0214)
- Observed on real data: 0.3183 (var=0.0405)

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
**Predicted under pi_10:** 0.2522 (var=0.0026)

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
**Predicted under pi_10:** 0.3175 (var=0.0088)

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
**Predicted under pi_10:** 0.2650 (var=0.0064)

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
**Predicted under pi_10:** 0.1600 (var=0.0080)

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
**Predicted under pi_10:** 0.7707 (var=0.0058)

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
**Predicted under pi_10:** 0.5017 (var=0.0113)

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
**Predicted under pi_10:** 0.4500 (var=0.0038)

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
**Predicted under pi_10:** 0.7960 (var=0.0025)

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
**Predicted under pi_10:** 0.7953 (var=0.0023)

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
**Predicted under pi_10:** 0.4236 (var=0.0018)

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
**Predicted under pi_10:** 0.5946 (var=0.0036)

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
**Predicted under pi_10:** 0.2013 (var=0.0047)

### Experiment 15
**Design**
  A=[1, 0, 1, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[1, 1, 0, 1, 1]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 1]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[0, 0, 0, 1, 0]

**Metric**
```python
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
```

**Observed (real) value:** 0.5460 (var=0.0126)
**Predicted under pi_7:** 0.6410 (var=0.0093)
**Predicted under pi_10:** 0.5080 (var=0.0118)

### Experiment 16
**Design**
  A=[1, 0, 1, 1, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 0, 1]  B=[0, 1, 1, 1, 0, 1]
  A=[0, 1, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0, 1]
  A=[0, 1, 1, 1, 0, 1]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 1]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 0]

**Metric**
```python
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
```

**Observed (real) value:** 0.4675 (var=0.0144)
**Predicted under pi_7:** 0.6388 (var=0.0149)
**Predicted under pi_10:** 0.5413 (var=0.0121)

### Experiment 17
**Design**
  A=[1, 0, 1, 1, 0, 1]  B=[1, 0, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0]
  A=[1, 1, 0, 1, 0, 0]  B=[1, 1, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 1, 1, 0, 1, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 1, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    hits = 0.0
    tot = 0.0
    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()
    ANCHOR = 3  # column with the lowest printed validity (0.55)
    for a_raw, b_raw, r in zip(a_col, b_col, r_col):
        a = [int(x) for x in list(a_raw)]
        b = [int(x) for x in list(b_raw)]
        k = min(len(a), len(b))
        if k <= ANCHOR:
            continue
        if a[ANCHOR] == b[ANCHOR]:
            continue  # anchor silent -> not a coalition test
        n_a = 0
        n_b = 0
        for i in range(k):
            if a[i] > b[i]:
                n_a += 1
            elif b[i] > a[i]:
                n_b += 1
        anchor_favors_a = a[ANCHOR] > b[ANCHOR]
        opposing = n_b if anchor_favors_a else n_a
        if opposing < 1:
            continue  # pure dominance for the anchor side -> no coalition
        try:
            resp = int(r)
        except Exception:
            continue
        chose_a = (resp == 0)
        if chose_a == anchor_favors_a:
            hits += 1.0
        tot += 1.0
    if tot == 0.0:
        return float('nan')
    return float(hits / tot)
```

**Observed (real) value:** 0.8620 (var=0.0155)
**Predicted under pi_7:** 0.7610 (var=0.0035)
**Predicted under pi_10:** 0.8087 (var=0.0029)

### Experiment 18
**Design**
  A=[1, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 1, 0, 1, 1, 1]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    hits = 0.0
    n = 0.0
    for a_raw, b_raw, resp in zip(data['option_a_ratings'],
                                  data['option_b_ratings'],
                                  data['response']):
        a = np.asarray(list(a_raw), dtype=float).ravel()
        b = np.asarray(list(b_raw), dtype=float).ravel()
        k = int(min(a.size, b.size))
        if k < 6:
            continue
        # the near-chance anchor expert (column 0, validity .52) must discriminate
        if a[0] == b[0]:
            continue
        anchor_side = 0 if a[0] > b[0] else 1  # 0 == option A, 1 == option B
        # count late, mid-doubt rows (columns 3,4,5) that oppose the anchor
        opp = 0
        for j in (3, 4, 5):
            if a[j] == b[j]:
                continue
            fav = 0 if a[j] > b[j] else 1
            if fav != anchor_side:
                opp += 1
        if opp < 2:
            continue
        n += 1.0
        try:
            r = int(resp)
        except Exception:
            continue
        if r == anchor_side:
            hits += 1.0

    if n == 0.0:
        return 0.5
    return float(hits / n)

```

**Observed (real) value:** 0.8260 (var=0.0180)
**Predicted under pi_7:** 0.7245 (var=0.0054)
**Predicted under pi_10:** 0.8075 (var=0.0040)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all 18 experiments the two accounts are nearly indistinguishable on the 'generic anti-validity' metrics (Exps 3, 5, 6, 9, 10, 11, 12, 14 \u2014 both within a few points of the observed value, with pi_7 marginally closer in several), but they diverge sharply on exactly the cells that were engineered to discriminate them, and on every one of those cells pi_10 wins.\n\n(1) Exp 1 (proposed by pi_7, its own crown-jewel cell): a lone mid-doubt .66 dissenter against a coalition of more valid experts. pi_7 predicted .696 (coalitions buy nothing, near-argmax doubt); pi_10 predicted .491 (compensatory summation because .66 is above the absolute gate). Observed .4925 \u2014 essentially exact for pi_10, a ~20-point miss for pi_7. This directly falsifies pi_7's sharpened claim (1) and (2).\n\n(2) Exp 2: T\u2212C contrast. pi_7 predicted ~.01 (load leak kills long displays while the .68 holder stays above chance); pi_10 predicted .355. Observed .318. Again pi_10 is close and pi_7 is catastrophically wrong on both limbs simultaneously: long displays in which a near-chance (.52) expert speaks stay confident (no raw-length leak), and a merely mid-doubt lone dissenter does NOT dictate choice.\n\n(3) The same pattern replicates in earlier experiments: Exp 15 (lone .68 vs coalition, obs .546; pi_7 .641, pi_10 .508), Exp 16 (obs .4675; pi_7 .639, pi_10 .541), Exp 13 (obs .593; pi_7 .683, pi_10 .595). And where the dissenter IS near chance (.52\u2013.55), accuracy is high and flat in coalition size: Exp 17 obs .862, Exp 18 obs .826 \u2014 pi_10 (.809/.808) again closer than pi_7 (.761/.725).\n\nSo pi_7's two distinctive commitments \u2014 relative near-argmax doubt (so the least-valid discriminating row always wins, regardless of its absolute validity) and a raw display-length lapse \u2014 are both refuted by the data. Its remaining fits on the generic metrics come from machinery it shares with pi_10 (anti-validity weighting, recency tilt, degeneracy/tally gate), not from its own claims.\n\npi_10 is clearly the better theory, but it is not perfect: it overpredicts the swing in Exp 4 (obs .174 vs .318; pi_7 .230 closer), underpredicts the anti-validity extremity of Exp 14 (obs .261 vs .201 \u2014 actually it overshoots in the anti-validity direction there), overpredicts Exp 16 by ~.07, and underpredicts the very high anchor-follow rates in Exps 17/18. Its weakest structural commitment is the ABSOLUTE, razor-sharp gate at theta\u2248.573 with tau\u2248.01: an expert at .60 behaves like an expert at .95, which is psychologically implausible and is what drives the Exp 4 miss (there the least-valid experts are .55/.60, tightly spaced, and the observed swing is half what the absolute gate predicts).\n\nTherefore the productive move is not to patch pi_7's parameters \u2014 its prose claims are the thing that failed \u2014 but to replace it with a genuinely different competitor that reproduces everything pi_10 gets right while replacing the absolute knife-edge gate with a relative, graded mechanism that can also fix Exp 4 and the underprediction in Exps 17/18.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace pi_7 (MDSC-NV) entirely. Its two signature claims are falsified: (a) 'coalitions buy nothing / near-argmax relative doubt' is refuted by Exps 1 (.49), 15 (.55), 16 (.47) and by Exp 2's C-term; (b) the raw display-length lapse rho^(m-2) is refuted by the flat, high anchor-follow rates on long displays (Exp 2 T-term, Exps 17/18).\n\nPropose instead a RELATIVE-SHARE GATING theory \u2014 call it SSG (Salience-Share Gating) \u2014 that is a strong competitor to pi_10 rather than a weak one:\n\n1. Row salience over DISCRIMINATING rows only: s_j = (1-v_j)^gamma * exp(lam*j), gamma ~ 1.5-2.5, lam ~ 0.1-0.5 (weak recency; large lam only needed in the near-tied-validity designs of Exps 10/11).\n2. GRADED MODE GATE driven by the salience SHARE of the top row, not an absolute validity threshold: pi_one = sigmoid((s_max/sum_j s_j - c)/w), with c ~ 0.55-0.75, w ~ 0.05-0.15. Intuition: a subject abandons integration only when one dissent is *conspicuously odder than everything else on the screen*. This reproduces pi_10's key successes \u2014 a .52 expert among .64-.96 experts has share ~.6-.8 -> one-reason, high and flat in coalition size (Exps 2-T, 17, 18); a .66/.68 expert among .64/.70/.72 experts has share ~.3-.4 -> compensatory, hence at/below chance against a coalition (Exps 1, 15, 16) \u2014 but WITHOUT the implausible knife edge, and it makes a distinct, testable prediction pi_10 cannot make: a .68 dissenter standing alone against a single .96 row should still be followed (its share is high), whereas pi_10 says compensatory there. It should also shrink the Exp-4 swing (least-valid .55 sits next to .60, so share is low, mode is mostly compensatory) which is pi_10's clearest quantitative miss.\n3. One-reason limb: follow the top-salience row with fidelity (1-eps_t) degraded only by the size of the opposing coalition (rho_opp^(k_conf-1)), NOT by display length \u2014 matching the observed flatness.\n4. Compensatory limb: divisively normalised anti-validity sum E = sum(sign_j * s_j)/(sigma + sum s_j), p = sigmoid(beta*E), with a mild conflict (not length) cost; calibrate so unanimous 5-6 row displays land near .74 (Exp 7) rather than at ceiling.\n5. Keep the label-degeneracy/tally fallback only for exact repeats (tolerance .002) so the .004/.005-gap designs (Exps 10, 11) still run the salience rule with strong recency.\n\nTarget quantitative anchors for the new model: Exp 1 ~.49, Exp 2 (T-C) ~.32, Exp 15 ~.55, Exp 16 ~.47, Exp 17 ~.86, Exp 18 ~.83, Exp 7 ~.74, Exp 4 ~.17, Exp 13 ~.59, Exps 3/5/6/12/14 in the .15-.42 anti-validity range."
}
```

## Usage

```json
{
  "input_tokens": 44992,
  "output_tokens": 4737
}
```
