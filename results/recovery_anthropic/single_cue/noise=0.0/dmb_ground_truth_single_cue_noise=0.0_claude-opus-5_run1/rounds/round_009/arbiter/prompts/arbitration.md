# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_11" and "pi_10") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_11" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_10" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_11
**SSG-6 — Salience-Share Gating with an opposition-graded adjudication limb and a leakier integration limb.**

The architecture is exactly SSG-5's; one quantitative commitment is revised, none is reversed.

**(1) Two modes: adjudication vs. integration.** On every trial the subject either lets ONE expert adjudicate the choice or integrates all disagreeing rows. Mode is decided by a RELATIVE quantity — the share of the display's total *oddness* owned by genuinely near-chance experts — not by display length nor by an absolute knife edge. Each discriminating row has oddness o_j = (1-v_j)^gamma_s (gamma_s ≈ 6.4-8.6, a contrast-enhanced, nearly ordinal 'who is the strangest expert here' read-out) and a scandal value g_j = sigmoid((v_crit - v_j)/w_h) with a crisp boundary (w_h ≈ .008-.011, v_crit ≈ .576-.586). The gate is share = Σ_j g_j·o_j / Σ_j o_j and pi_one = (max_j g_j)·[f + (1-f)·sigmoid((share - c)/w)]: adjudication requires BOTH that some expert be near-chance AND that near-chance experts own most of the display's oddness.

**(2) Adjudication is faithful but NOT flat in opposition; it is flat only in display length.** When a near-chance expert commands the trial, subjects follow its verdict with high fidelity (eps_t ≈ .19-.29) and scanning extra, silent or agreeing rows costs nothing at all — one-reason choice is free in sheer list length, which is why anchor-follow rates stay high on six-row displays. What it is NOT free in is genuine CONTRADICTION: each additional row that argues the other way beyond the first charges rho_opp ≈ .82-.90. A scandalous dissenter standing alone against one opposing voice is followed almost as often as a scandalous dissenter in a unanimous display, but once two or three experts are lined up the other way the adjudication itself becomes contested and the verdict is dragged appreciably toward chance. Among co-scandalous rows the consulted verdict is drawn from softmax(log g_j + mu·j) with a WEAK positional gain (mu ≈ .35-.62); reading order can never lift an ordinary expert over a near-chance one.

**(3) REVISED — integration is an intrinsically effortful, frequently-abandoned operation, and it is fragile under DISAGREEMENT.** When the gate is shut, evidence is integrated with w_j = (1-v_j)^gamma_w·exp(lam·j) (gamma_w ≈ 1.32-1.58, lam ≈ .10-.17, so doubt, not position, buys weight), divisively normalised to the loudest dissent, E = Σ sign_j·ŵ_j/(sigma + Σ ŵ_j), p = sigmoid(beta·E). The sharpened claim is that the BASELINE abandonment rate of integration is appreciably higher than previously assumed (eps0 ≈ .26-.34, not ≈ .22-.30): weighing several probabilistically-labelled experts against one another is an operation subjects decline to complete on roughly a third of trials even before any conflict or load penalty, so integrated verdicts are systematically less decided than the weights alone imply. On top of this baseline the retained evidence leaks further with display load (rho_m) and much more steeply with every additional row that argues the other way (rho_c ≈ .62-.75 per minority row). Because the baseline lapse applies to EVERY integration trial — including the unanimous/dominance displays where there is nothing to adjudicate — it predicts that even fully consistent multi-row displays stay clearly below ceiling (~.74-.75), while conflict displays are pulled nearer chance still and never reach a near-deterministic anti-validity verdict.

The theory's signature is therefore that BOTH modes are degraded by contradiction and NEITHER is degraded by list length — they differ in how steeply contradiction bites (mild for adjudication, severe for integration) and in their baseline completion rates (adjudication is cheap and usually completed; integration is expensive and often abandoned). When all discriminating experts agree there is nothing to adjudicate (pi_one = 0) and only the load/baseline leak applies. If two printed validities are indistinguishable (gap < .002) oddness has no referent and choice collapses to a graded unweighted row tally. No feedback is given, so the rule is stationary across the block.

`predict(parameters, state, history) -> np.ndarray`:
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

## EXPERIMENT 1 (proposed by pi_11)

### DESIGN
**Validities (n_features=6):** [0.66, 0.62, 0.64, 0.9, 0.94, 0.52]

**Trial pairs (n=24):**
  trial 1: A=[1, 0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 1, 0]
  trial 2: A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 1]
  trial 3: A=[0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 1, 0]
  trial 4: A=[0, 1, 0, 1, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  trial 5: A=[1, 0, 1, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 6: A=[1, 0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  trial 7: A=[1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  trial 8: A=[1, 0, 1, 1, 1, 0]  B=[1, 0, 1, 0, 0, 1]
  trial 9: A=[1, 1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0, 1]
  trial 10: A=[0, 1, 0, 0, 0, 1]  B=[1, 1, 0, 1, 1, 1]
  trial 11: A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  trial 12: A=[0, 0, 1, 1, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  trial 13: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 14: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  trial 15: A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 16: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0]
  trial 17: A=[1, 1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 1, 0]
  trial 18: A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1]
  trial 19: A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 1, 0, 0]
  trial 20: A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  trial 21: A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  trial 22: A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 1, 0]
  trial 23: A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  trial 24: A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]

**Rationale:** GOAL: separate SSG-6 (advocated) from DTG-2R/OC-sharp (competitor). The two accounts share almost everything: an absolute, razor-sharp near-chance trigger (v_crit .576-.586 / theta .558-.588 with widths .008-.013), an anti-validity, divisively-normalised integration limb with a weak recency tilt, a nearly-length-free adjudication limb, and an opposition-graded cost on adjudication. They differ on exactly two mechanisms, and BOTH are isolated here at matched display length with the decisive row and its verdict held constant: (A) SSG-6 sets pi_one = 0 whenever the discriminating rows are UNANIMOUS ('when all discriminating experts agree there is nothing to adjudicate'), so dominance displays are forced through the expensive, frequently-abandoned integration limb (eps0 .26-.34) and must sit clearly below ceiling, whereas DTG keeps P_trig = max_j g_j and therefore runs its cheap one-reason limb on unanimous displays too; (B) SSG-6 adds a relative SALIENCE-SHARE gate (share = sum g_j o_j / sum o_j, o_j = (1-v_j)^gamma_s, c_share ~ .72, floor ~ .40) so that adjudication is only partly available when other mildly-doubtful rows own a chunk of the display's oddness, whereas DTG's gate is purely absolute.

VALIDITIES [0.66, 0.62, 0.64, 0.90, 0.94, 0.52] at columns c0..c5. Engineering: (i) EXACTLY ONE expert, c5 = .52, is sub-threshold for both models (g ~ 1.00), while c1 = .62 gives g <= .08 even at the worst DTG corner (theta=.588, tau=.013) and <= .02 for SSG, and c2=.64, c0=.66, c3=.90, c4=.94 are effectively non-scandalous for both. So both models adjudicate iff c5 discriminates and integrate otherwise, and they always AGREE which row is decisive. (ii) c5 sits in the LAST column, so both models' positional gains (mu, lam) reinforce rather than threaten the anchor's selection in the softmax - no corner of either parameter box can let an ordinary expert be consulted instead. (iii) Validity does not track column index, so 'read the first/last row' cannot masquerade as either model. (iv) Minimum label gap is .02, ten times both degeneracy tolerances (.002), so neither model can retreat into tally mode. (v) The three mid rows (.62,.64,.66) carry appreciable oddness relative to .52 ((1-v) ratios .79/.75/.71, i.e. o-shares of .17/.12/.08 at gamma_s=7.5) so that when they are the opposition SSG's share gate partially shuts, while the two high rows (.90,.94) have utterly negligible oddness (o-share < 1e-5) so that share ~ 1 and SSG's gate is wide open. (vi) Integration weights are nearly identical across the two models (normalised to the anchor: SSG .31/.42/.44/.08/.04/1.00, DTG .30/.46/.46/.05/.05/1.00), so the integration limbs never disagree in DIRECTION - the whole contrast is quantitative.

12 base structures, each with its exact A/B mirror (24 unique pairs, K = 4 -> 96 trials), so both models' predicted responses are perfectly key-balanced and side bias cannot mimic either. Silent rows carry matched 1s AND matched 0s so displays are dense and cannot be solved by counting 1s. Predictions below use SSG-6 (gamma_s=7.5, v_crit=.581, w_h=.0095, c_share=.72, w_share=.05, floor=.40, mu=.5, eps_t=.25, rho_opp=.86, gamma_w=1.45, lam=.135, sigma=.68, beta=2.4, eps0=.30, rho_m=.90, rho_c=.68) and DTG (theta=.573, tau=.010, mu=.70, gamma=1.9, lam=.11, sigma=.55, beta=2.2, eps_t=.28, rho_t=.99, rho_tc=.74, eps0=.20, rho_m=.885, rho_c=.82).

BLOCK 1 - CROWN JEWEL: DOES A DISSENTER MAKE THE DISPLAY EASIER? (bases 1/2, 3/4, 7/12 = pairs 1-8, 13-14, 23-24). Three matched cells in which the .52 anchor favours A in BOTH members and only the DIRECTION of the remaining rows flips from agreeing to opposing. For DTG these members are EXACTLY equal for every parameter setting, because keep_t = (1-eps_t)*rho_t^max(m-2,0)*rho_tc^max(k_conf-1,0) is identical when k_conf goes 0 -> 1 at fixed m (rho_tc^0 = 1 either way) and P_trig = 1 in all six displays: base1 = base2 (.860), base3 = base4 (.856), base7 = base12 (.846). SSG-6 must instead predict a large ADVANTAGE for the conflict member, because unanimity switches adjudication off entirely: base1 (1-0 anchor alone) .715 vs base2 (anchor vs .94) .875 (+16); base3 (anchor + two highs, 3-0) .699 vs base4 (anchor vs two highs, 1-2) .875 (+18); base7 (6-0 all agree) .667 vs base12 (anchor vs all five others, 1-5) .774 (+11). A display becoming ~15 points EASIER when a high-validity expert switches from agreeing to contradicting the .52 expert is producible by no compensatory, one-reason or tally account other than SSG-6; three exact zeros are the DTG signature. Robustness: at SSG's most favourable-to-DTG corner (eps0=.26, rho_m=.93, beta=2.65, sigma=.58, eps_t=.19) base3 rises only to ~.74 while base4 stays ~.90, so the sign and a >=10-point magnitude survive the whole box.

BLOCK 2 - CROWN JEWEL 2: DOES THE SCANDALOUS ROW HELP A DOMINANCE DISPLAY? (bases 3 vs 5, 7 vs 8 = pairs 5-6, 9-10, 13-16). Matched unanimous displays differing ONLY in whether the lowest-validity speaking row is the .52 anchor or a .66/.62 ordinary expert. DTG must predict a jump, because the anchor flips the trial out of the load-degraded compensatory limb into the near-free one-reason limb: base3 (.52 speaks, 3-0) .856 vs base5 (.66 speaks instead, 3-0) .725 (+13); base7 (6-0, anchor speaks) .846 vs base8 (the identical 5-0 display with the anchor row silenced) .701 (+14.5). SSG-6 predicts NO jump at all - both members are integration trials (pi_one = 0) and differ only in weights: .699 vs .710 (-1) and .667 vs .691 (-2.4), i.e. if anything the anchor-present display is marginally WORSE because it is one row longer. Flat-or-slightly-negative versus a +13 to +15 step is a second parameter-robust read-out, and base8 vs base7 is a one-row minimal contrast (the two displays differ in a single cell).

BLOCK 3 - SHARED INTEGRATION BASELINES THAT BLOCK THE 'GLOBAL NOISE' ESCAPE (bases 5, 6, 8, 11 = pairs 9-12, 15-16, 21-22). With the anchor silent both models integrate and agree within ~3-7 points: base5 (3-0) .710/.725; base8 (5-0) .691/.701; base6 (1-2, .66 dissenter vs two highs) .572/.637; base11 (exact 2-2 split, A wins .66+.90, B wins .62+.64) .454/.423 - both models BELOW chance at a row-count tie, which no vote-counting account can produce. These cells pin eps0, rho_m and rho_c for both models free of the contrast, so Block-1/2 divergences cannot be re-described as overall lapse-rate differences. They also give a graded test of SSG's steeper conflict leak (rho_c .62-.75 vs .74-.90): SSG must be ~6 points closer to chance than DTG on base6 and ~3 points on base11.

BLOCK 4 - MANIPULATION CHECK (base4 vs base6, base2 vs base6). Structurally identical 1-vs-2 displays in which only the identity of the lone dissenter changes, .52 -> .66. BOTH models predict a large advantage for the .52 dissenter (SSG .875 -> .572, +30; DTG .856 -> .637, +22), so if subjects show no effect here the printed validities are simply being ignored and neither theory is in play - this guards the interpretation of the null cells in Block 1.

BLOCK 5 - OPPOSITION COST AND THE SALIENCE-SHARE GATE (bases 9, 10 = pairs 17-20). Base 9 (2-vs-2: A wins .52 + .66, B wins .90 + .94; k_conf = 2) charges rho_opp once for SSG (.86) but rho_tc once for DTG (.74): SSG .820 vs DTG .761, and read against base4 the decline is -5.5 (SSG) vs -9.5 (DTG) points. Base 10 (A wins .52 alone, B wins the three mid rows .66/.62/.64) is the share-gate probe: DTG stays fully in one-reason mode (.853) whereas SSG's near-chance expert now owns only ~73% of the display's oddness, half-shutting the gate and mixing in an integration verdict that is itself near zero (the three mids' summed normalised weight 1.16 slightly exceeds the anchor's 1.00), giving .775 and ranging .71-.87 across gamma_s = 6.4-8.6. This cell is the design's estimator for gamma_s/c_share if SSG wins.

INTERPRETATION: if a 1-0 (or 3-0, or 6-0) display carried by the .52 expert is answered ~.67-.72 while the SAME display with the other rows flipped into opposition is answered ~.87 - i.e. contradiction makes adjudication available and therefore makes the trial EASIER - and if silencing the .52 row in a unanimous display costs essentially nothing, SSG-6's unanimity-kills-adjudication claim plus its high baseline integration lapse is confirmed and DTG-2R/OC-sharp is falsified. If instead all six Block-1 displays sit flat at ~.85 regardless of agreement/opposition, and unanimous displays drop ~14 points the moment the .52 expert is silenced, DTG's purely absolute trigger with a length-costed compensatory limb wins and the salience-share gate plus the pi_one = 0 unanimity rule is dead.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



### METRIC
Rationale:
Isolates the single mechanism on which SSG-6 and DTG-2R disagree most sharply at maximal trial count: UNANIMOUS displays in which the near-chance expert (v=0.52, the design's anchor) is among the discriminating rows. In this design those are trial pairs 1/2 (only the .52 row speaks), 5/6 (.52 plus the two high-validity rows, all agreeing) and 13/14 (full 6-row dominance) — 6 unique pairs x 4 reps = 24 trials per subject, so the per-subject estimate is tight (binomial SD ~.09).

SSG-6 sets pi_one = 0 whenever the discriminating rows are unanimous ('nothing to adjudicate'), forcing these displays through the expensive, frequently-abandoned integration limb with baseline lapse eps0 = .26-.34 plus load leak rho_m. Across its whole parameter box SSG predicts an accuracy on this cell of roughly .64-.76 (point predictions .715, .699, .667; mean ~.69).

DTG-2R keeps P_trig = max_j g_j regardless of unanimity, so the .52 row triggers the cheap one-reason limb (keep_t = (1-eps_t)*rho_t^(m-2), rho_t >= .985, no opposition charge since k_conf = 0). It predicts ~.82-.90 (point predictions .860, .856, .846; mean ~.85).

So the metric should read ~.69 under SSG-6 and ~.85 under DTG, a ~16-point gap with per-subject SD near .10 — comfortably separable at N=25 — while being non-trivial (it is a plain behavioural follow-the-dominant-option rate on dominance displays, computable without reference to either model). Crucially, the conflict-anchor cells (where the two theories nearly coincide) are deliberately excluded, since pooling them would wash the contrast out.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    base_vals = [0.66, 0.62, 0.64, 0.90, 0.94, 0.52]
    hits = 0
    tot = 0
    for a_, b_, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        try:
            a = np.asarray(list(a_), dtype=float).ravel()
            b = np.asarray(list(b_), dtype=float).ravel()
        except Exception:
            continue
        n = int(min(a.size, b.size))
        if n == 0:
            continue
        a = a[:n]
        b = b[:n]
        if n <= len(base_vals):
            v = np.asarray(base_vals[:n], dtype=float)
        else:
            v = np.asarray(base_vals + [0.55] * (n - len(base_vals)), dtype=float)
        anchor = int(np.argmin(v))
        d = a - b
        disc = np.nonzero(d != 0.0)[0]
        if disc.size == 0:
            continue
        if anchor not in set(disc.tolist()):
            continue
        s = np.sign(d[disc])
        # keep only UNANIMOUS displays (every discriminating expert agrees)
        if not (bool(np.all(s > 0.0)) or bool(np.all(s < 0.0))):
            continue
        fav_a = bool(d[anchor] > 0.0)
        try:
            chose_a = (int(r) == 0)
        except Exception:
            continue
        tot += 1
        if chose_a == fav_a:
            hits += 1
    if tot == 0:
        return 0.5
    return float(hits) / float(tot)

### RESULTS
- Predicted under pi_11 (simulated): 0.6958 (var=0.0069)
- Predicted under pi_10 (simulated): 0.8450 (var=0.0051)
- Observed on real data: 0.8617 (var=0.0175)

## EXPERIMENT 2 (proposed by pi_10)

### DESIGN
**Validities (n_features=7):** [0.635, 0.93, 0.615, 0.645, 0.97, 0.52, 0.625]

**Trial pairs (n=24):**
  trial 1: A=[1, 0, 1, 0, 1, 1, 0]  B=[1, 0, 1, 0, 1, 0, 0]
  trial 2: A=[1, 1, 0, 1, 1, 1, 0]  B=[1, 0, 0, 1, 0, 0, 0]
  trial 3: A=[1, 1, 1, 1, 0, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 1, 1, 0]  B=[1, 0, 1, 1, 1, 0, 1]
  trial 5: A=[0, 1, 1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 1, 0, 1]
  trial 6: A=[1, 0, 0, 1, 0, 1, 0]  B=[1, 1, 0, 1, 1, 0, 0]
  trial 7: A=[1, 0, 0, 1, 0, 1, 1]  B=[1, 0, 1, 1, 0, 0, 1]
  trial 8: A=[0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0, 0, 1]
  trial 9: A=[1, 0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 1, 0, 1]
  trial 10: A=[1, 1, 1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1, 1, 0]
  trial 11: A=[1, 1, 1, 1, 0, 1, 1]  B=[0, 0, 0, 0, 0, 1, 0]
  trial 12: A=[1, 0, 1, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1, 0, 1]
  trial 13: A=[1, 0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1, 0]
  trial 14: A=[1, 0, 0, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1, 0]
  trial 15: A=[0, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 1, 1]
  trial 16: A=[1, 0, 1, 1, 1, 0, 1]  B=[0, 0, 0, 0, 1, 1, 0]
  trial 17: A=[0, 1, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 0, 1, 1]
  trial 18: A=[1, 1, 0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0, 1, 0]
  trial 19: A=[1, 0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 1, 1]
  trial 20: A=[0, 1, 1, 1, 0, 0, 1]  B=[0, 1, 0, 1, 0, 1, 0]
  trial 21: A=[1, 1, 1, 0, 1, 0, 1]  B=[1, 0, 0, 0, 0, 1, 0]
  trial 22: A=[0, 1, 0, 0, 1, 1, 0]  B=[1, 1, 1, 0, 1, 1, 0]
  trial 23: A=[0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 0, 1, 1]
  trial 24: A=[0, 0, 0, 1, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 0]

**Rationale:** GOAL: separate DTG-2R/OC-sharp (advocated) from SSG-6 (competitor). The two accounts share an absolute, razor-sharp near-chance trigger, an anti-validity divisively-normalised integration limb with a weak recency tilt, and a nearly length-free adjudication limb, so they always AGREE about WHICH row carries the verdict here. They differ on exactly two mechanisms, and the design isolates both while holding the decisive row, its verdict, and (in the key ladders) the conflict count k_conf = 1 exactly constant: (A) SSG-6 adds a RELATIVE salience-share gate (share = sum g_j*o_j / sum o_j with o_j = (1-v_j)^gamma_s, c_share ~ .72, floor ~ .40), so adjudication is only partly available when several mildly-doubtful (but non-scandalous) experts own a chunk of the display's oddness; DTG's gate is purely ABSOLUTE (P_trig = max_j g_j), so the identity of the OPPOSING rows is irrelevant as long as the near-chance expert speaks. (B) SSG-6 sets pi_one = 0 whenever the discriminating rows are UNANIMOUS ('nothing to adjudicate'), forcing dominance displays through its expensive, frequently-abandoned integration limb (eps0 .26-.34), whereas DTG runs its cheap one-reason limb on unanimous displays too.

VALIDITIES [0.635, 0.93, 0.615, 0.645, 0.97, 0.52, 0.625] at columns c0..c6. Engineering: (i) EXACTLY ONE expert, c5 = .52, is sub-threshold for both models (g ~ 1.00 over both boxes); the four mids (.615,.625,.635,.645) are non-scandalous for both (g <= .11 even at DTG's worst corner theta=.588/tau=.013, <= .03 for SSG) yet carry appreciable ODDNESS relative to the anchor ((1-v) ratios .74-.80, i.e. o-shares .10-.19 each at gamma_s = 7.5), while the two highs (.93,.97) have utterly negligible oddness (< 1e-4), so share ~ 1 when they are the opposition. (ii) The anchor sits LATE (c5, j=5) so both models' positional gains (mu) REINFORCE rather than threaten its selection in the adjudication softmax: even at DTG's worst corner (theta=.588, tau=.013, mu=.95) the anchor still owns ~.87 of the softmax against all four mids. (iii) Validity does not track column index (a high sits last), so 'read the first/last row' cannot masquerade as either model; minimum label gap is .01, five times both degeneracy tolerances. (iv) The integration limbs of the two models use nearly identical weights (gamma 1.9 vs gamma_w 1.45), so they never disagree in DIRECTION - the whole contrast is quantitative.

12 base structures, each with its exact A/B mirror (24 unique pairs, K = 4 -> 96 trials), perfectly key-balanced. Silent rows carry matched 1s AND matched 0s so every display is dense and cannot be solved by counting 1s. Predictions use DTG (theta=.573, tau=.010, mu=.70, gamma=1.9, lam=.11, sigma=.55, beta=2.2, eps_t=.28, rho_t=.99, rho_tc=.74, eps0=.20, rho_m=.885, rho_c=.82) and SSG-6 (gamma_s=7.5, v_crit=.581, w_h=.0095, c_share=.72, w_share=.05, floor=.40, mu=.5, eps_t=.25, rho_opp=.86, gamma_w=1.45, lam=.135, sigma=.68, beta=2.4, eps0=.30, rho_m=.90, rho_c=.68).

BLOCK 1 - CROWN JEWEL: OPPONENT-TYPE x OPPONENT-COUNT LADDER WITH THE ANCHOR ALONE (bases 4,6,7,8,9; pairs 4,6,7,8,9 + mirrors). In every cell the .52 anchor favours A alone (k_conf = 1 throughout, so neither model's opposition cost fires) and only the NUMBER and TYPE of opposing rows change:
  * vs 1 mid (.615, m=2): DTG .860 / SSG .861
  * vs 2 mids (.615,.625, m=3): DTG .857 / SSG .781
  * vs 4 mids (all, m=5): DTG .850 / SSG .657
  * vs 1 high (.97, m=2): DTG .860 / SSG .875
  * vs 2 highs (.93,.97, m=3): DTG .857 / SSG .874
  * vs 2 mids + 2 highs (m=5): DTG .850 / SSG .780
DTG must predict a FLAT ~.85-.86 profile across all six cells for every parameter setting (P_trig = 1, rho_t ~ 1, rho_tc never fires at k_conf = 1) - 'a near-chance dissenter standing alone is followed at ~.85 flat in coalition size'. SSG-6 must instead produce a two-way INTERACTION: flat ~.87 when the opposition is high-validity (share stays ~1, gate wide open) but a monotone 20-point slide .861 -> .781 -> .657 as mid-doubt rows accumulate and steal oddness share, plus the matched-m contrast 4 mids (.657) versus 2 mids + 2 highs (.780) which is a pure TYPE effect at identical display length and identical row count. Robustness: across gamma_s = 6.4-8.6 the 4-mid cell lies in [.62,.74] for SSG while DTG stays in [.76,.86] even at its own worst corner. Flat-in-everything versus graded-in-oddness-share is the decisive parameter-light read-out.

BLOCK 2 - UNANIMITY: DOES THE SCANDALOUS ROW HELP A DOMINANCE DISPLAY? (bases 1,2,3 vs 10,11; pairs 1,2,3,10,11 + mirrors). Unanimous displays carried by the anchor: m=1 (anchor alone) DTG .860 / SSG .715; m=3 (anchor + two highs) DTG .857 / SSG .697; m=5 (anchor + four mids) DTG .850 / SSG .693. Matched anchor-SILENT unanimous displays: m=2 (two mids) DTG .773 / SSG .744; m=5 (four mids + .93) DTG .701 / SSG .690. Read at matched m=5, DTG says adding/substituting the anchor row makes the display ~15 points EASIER (.701 -> .850) because it flips the trial out of the load-degraded compensatory limb into the near-free one-reason limb, whereas SSG says it is worth exactly nothing (.690 -> .693) since unanimity kills adjudication. Equally, base 1 versus base 7 (anchor alone unanimous vs anchor contradicted by one mid) is flat for DTG (.860 -> .860) but a +15-point jump for SSG (.715 -> .861): contradiction MAKING a display easier is SSG's signature, invariance is DTG's.

BLOCK 3 - SHARED INTEGRATION BASELINES THAT BLOCK THE 'GLOBAL NOISE' ESCAPE (bases 10,11,12; pairs 10,11,12 + mirrors). With the anchor silent both models integrate and must agree within ~3 points: 2-0 mids .773/.744, 5-0 .701/.690, and the exact 2-vs-2 mid split .467/.473 - both BELOW chance at a row-count tie, which no vote-counting account can produce in either direction. These cells pin eps0, rho_m and rho_c for both models free of the contrast, so any Block-1/2 divergence cannot be re-described as a lapse-rate difference; they also give a mild test of SSG's steeper conflict leak.

INTERPRETATION: if the lone .52 dissenter is followed at ~.85 regardless of whether it is opposed by one high, two highs, one mid, two mids or four mids, and if unanimous displays carried by the .52 expert stay at ~.85 while the same-length displays with the .52 row silenced fall to ~.70, the purely ABSOLUTE sharp gate with a cheap, opposition-only-costed one-reason limb and a load-costed compensatory limb (DTG-2R/OC-sharp) is confirmed and the salience-share gate plus the pi_one = 0 unanimity rule is falsified. If instead follow rates slide monotonically with the number of mid-doubt (but not high-validity) opponents, and unanimous anchor displays sit ~15 points BELOW the conflict displays that contain the very same anchor, SSG-6 wins.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



### METRIC
Rationale:
The metric is the anchor-follow rate on exactly the trials where the two accounts make maximally divergent commitments: displays in which the lone near-chance expert (v=.52, column 5) discriminates, is never opposed by a high-validity expert (.93/.97), and is either unopposed (unanimous dominance displays: base pairs 1/13, 2/14, 3/15) or opposed by a coalition of two or four mid-doubt experts (pairs 4/16, 8/20). Under DTG-2R/OC-sharp the absolute sharp gate sets P_trig=1 on all of these, the one-reason limb is nearly free in list length, and rho_tc does not compound at k_conf<=1 for the unanimous cells, so the follow rate must be a flat ~.85 in every one of these cells regardless of the number/type of opponents and regardless of unanimity (predicted pooled value ~.855). Under SSG-6 two distinct mechanisms pull the same trials down: (a) pi_one = 0 on unanimous displays forces them through the high-baseline-lapse integration limb (~.69-.72), and (b) the relative salience-share gate is stolen by accumulating mid-doubt opponents, driving the 4-mid cell to ~.66 and the 2-mid cell to ~.78 (predicted pooled value ~.71). The 40 trials per subject entering the average make the per-subject binomial noise small (~.003-.005 variance), and the cells chosen are those whose predictions are stable across each theory's parameter box, so the ~.15 mean gap should be highly reliable at N=25. Trials where the anchor faces a single mid opponent, or where high-validity experts oppose it, are deliberately excluded because both theories agree there (~.86), which would only dilute the contrast; anchor-silent displays are excluded for the same reason.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    # printed validities of this design (fixed at design time)
    V = np.array([0.635, 0.93, 0.615, 0.645, 0.97, 0.52, 0.625])
    anchor = int(np.argmin(V))  # the lone near-chance expert, v = .52 (column 5)

    follows = []
    for a_r, b_r, resp in zip(data['option_a_ratings'],
                              data['option_b_ratings'],
                              data['response']):
        try:
            a = np.asarray(list(a_r), dtype=float)
            b = np.asarray(list(b_r), dtype=float)
        except Exception:
            continue
        n = int(min(a.size, b.size, V.size))
        if n <= anchor:
            continue
        d = a[:n] - b[:n]
        if d[anchor] == 0.0:
            continue  # anchor silent -> not a test trial
        s = np.sign(d[anchor])
        opp = [j for j in range(n) if j != anchor and d[j] != 0.0 and np.sign(d[j]) != s]
        # exclude displays where a high-validity expert opposes the anchor
        if any(V[j] >= 0.9 for j in opp):
            continue
        # keep only unanimous-with-anchor displays (0 opponents) and displays with
        # a COALITION of >=2 mid-doubt opponents; drop the single-mid-opponent cell
        # where the two theories coincide
        if len(opp) == 1:
            continue
        try:
            r = int(resp)
        except Exception:
            continue
        chose_a = (r == 0)
        follows.append(1.0 if ((s > 0) == chose_a) else 0.0)

    if len(follows) == 0:
        return 0.5
    return float(np.mean(follows))

### RESULTS
- Predicted under pi_11 (simulated): 0.7045 (var=0.0056)
- Predicted under pi_10 (simulated): 0.8570 (var=0.0034)
- Observed on real data: 0.8750 (var=0.0100)

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
**Predicted under pi_11:** 0.2610 (var=0.0020)
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
**Predicted under pi_11:** 0.2156 (var=0.0155)
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
**Predicted under pi_11:** 0.2578 (var=0.0035)
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
**Predicted under pi_11:** 0.1267 (var=0.0084)
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
**Predicted under pi_11:** 0.7587 (var=0.0089)
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
**Predicted under pi_11:** 0.4925 (var=0.0142)
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
**Predicted under pi_11:** 0.4725 (var=0.0025)
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
**Predicted under pi_11:** 0.8106 (var=0.0027)
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
**Predicted under pi_11:** 0.8128 (var=0.0023)
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
**Predicted under pi_11:** 0.4203 (var=0.0023)
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
**Predicted under pi_11:** 0.6012 (var=0.0039)
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
**Predicted under pi_11:** 0.2096 (var=0.0037)
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
**Predicted under pi_11:** 0.4950 (var=0.0133)
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
**Predicted under pi_11:** 0.4950 (var=0.0100)
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
**Predicted under pi_11:** 0.8033 (var=0.0034)
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
**Predicted under pi_11:** 0.8455 (var=0.0028)
**Predicted under pi_10:** 0.8075 (var=0.0040)

### Experiment 19
**Design**
  A=[1, 0, 1, 1, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 1, 0]  B=[1, 0, 1, 1, 0, 1]
  A=[1, 0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[1, 0, 1, 1, 0, 1]
  A=[1, 0, 1, 1, 0, 1]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 0, 1]
  A=[1, 1, 1, 1, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 0, 1, 1, 1]

**Metric**
```python
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
```

**Observed (real) value:** 0.4925 (var=0.0074)
**Predicted under pi_11:** 0.5025 (var=0.0063)
**Predicted under pi_10:** 0.4906 (var=0.0058)

### Experiment 20
**Design**
  A=[1, 0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0, 1]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 0, 1]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 1]  B=[0, 1, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 0, 1, 1, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1, 0]

**Metric**
```python
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
```

**Observed (real) value:** 0.3183 (var=0.0405)
**Predicted under pi_11:** 0.3325 (var=0.0167)
**Predicted under pi_10:** 0.3550 (var=0.0214)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "The two decisive new experiments were both engineered to isolate SSG-6's two unique commitments: (a) pi_one = 0 on unanimous displays (unanimity kills adjudication) and (b) the relative salience-share gate (mid-doubt opponents steal oddness share and half-shut the gate). Both commitments were falsified, and falsified in the same direction, twice.\n\nExperiment 1 metric (anchor-follow rate on UNANIMOUS displays where the .52 expert discriminates): SSG-6 predicted .696, DTG predicted .845, observed .862. Experiment 2 metric (anchor-follow on unanimous displays plus displays where the anchor faces 2 or 4 mid-doubt opponents): SSG-6 predicted .704, DTG predicted .857, observed .875. In both cases the observed value sits at or slightly above DTG's prediction and 16-17 points away from SSG-6's, far outside the between-subject scatter (var ~ .01-.018, i.e. SD ~ .10-.13 across subjects, and the point estimates pool 24-40 trials per subject over N subjects). So: (i) unanimity does NOT kill adjudication \ns the display carried by the near-chance expert stays at ~.86 whether or not anyone contradicts it; (ii) accumulating mid-doubt opponents does NOT erode the near-chance expert's authority \ns there is no salience-share dilution. SSG-6's two distinguishing mechanisms are both dead, and its high baseline integration lapse (eps0 .26-.34) applied to dominance displays is too pessimistic for anchor-carried dominance displays specifically (though Exp 7's generic dominance rate of .743 shows the ~.74 sub-ceiling claim IS right when the anchor is not the carrier \ns exactly DTG's asymmetry).\n\nAcross the historical corpus (Exps 3-20) the two theories are nearly indistinguishable and both are good: Exp 3 (.282 obs vs .261/.252), Exp 5 (.286 vs .258/.265), Exp 6 (.150 vs .127/.160), Exp 7 (.743 vs .759/.771), Exp 8 (.507 vs .493/.502), Exp 9 (.438 vs .473/.450), Exp 10 (.770 vs .811/.796), Exp 11 (.819 vs .813/.795), Exp 12 (.418 vs .420/.424), Exp 13 (.593 vs .601/.595), Exp 14 (.261 vs .210/.201), Exp 15 (.546 vs .495/.508), Exp 16 (.468 vs .495/.541), Exp 17 (.862 vs .803/.809), Exp 18 (.826 vs .846/.808), Exp 19 (.493 vs .503/.491), Exp 20 (.318 vs .333/.355). Average absolute error is comparable (SSG slightly better on Exps 16, 18, 20; DTG slightly better on 15, 19). Only Exp 4 shows a visible split (obs .174 vs pi_11 .216 / pi_10 .318) favouring SSG \ns but that is a single derived difference metric with high between-subject variance (.009-.016), and it is dwarfed by the two 16-point misses on Exps 1 and 2.\n\nA couple of residual misfits are shared by BOTH theories and are worth noting for whatever replaces SSG-6: the anchor-follow rate under genuine opposition is being UNDER-predicted (Exp 17: obs .862 vs .803/.809; Exp 18: obs .826 vs .846/.808; Exp 1/2: obs .862/.875 vs DTG .845/.857), i.e. the near-chance expert's authority is even flatter and even more robust to opposition than DTG's rho_tc allows. Conversely both theories slightly over-predict Exp 10 (.770 obs vs .811/.796) and under-predict the ceiling-ish conflict cells of Exp 14/15. So the true picture appears to be: an absolute, sharp near-chance trigger; a one-reason limb that is essentially flat in BOTH list length AND opposition (rho_tc ~ .9-1.0, not .62-.86); and a compensatory limb that is load-degraded and sits around .70-.75 on long unanimous non-anchor displays.\n\nVerdict: SSG-6 (pi_11) is the loser. Its two signature mechanisms are directly refuted by the two experiments designed to test them, and it has no compensating advantage elsewhere. Rather than re-parameterising SSG-6 (which would require gutting its defining claims \ns setting pi_one > 0 on unanimous displays and c_share to nothing, which just turns it into DTG), it should be replaced by a genuinely different competitor that can beat DTG on the cells where DTG is still off (the under-predicted opposition-anchor cells and Exp 4's tally-vs-TTB contrast).",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace SSG-6 entirely. Its two defining commitments \ns (a) pi_one = 0 whenever the discriminating rows are unanimous, and (b) the relative salience-share gate share = sum g_j o_j / sum o_j \ns are both directly falsified: observed anchor-follow on unanimous anchor-carried displays is .86-.88 (SSG predicted .67-.72), and it does not decline as mid-doubt opponents accumulate. Re-parameterising cannot save it without erasing the description.\n\nPropose a NEW theory that is a strong competitor to DTG-2R/OC-sharp rather than a clone. Suggested sketch \ns call it 'AUTHORITY-INVERSION with a graded cost-of-integration' (AIG):\n\n1. Keep what is overwhelmingly supported: an ABSOLUTE, sharp near-chance criterion on PRINTED validity. When some discriminating expert prints v near .50-.55, that expert's verdict is adopted essentially wholesale. Crucially, make the adoption flat in BOTH display length AND opposition: the retained fidelity should be ~.86-.90 with at most a very mild opposition charge (rho_opp in .90-1.0), because the observed anchor-follow is .862 (Exp 1), .875 (Exp 2), .862 (Exp 17, all coalition trials), .826 (Exp 18, >=2 late mid-doubt opponents) \ns all AT or ABOVE DTG's predictions, which currently under-shoot Exps 17 and 1/2. This is the one place where DTG is systematically biased and where a new theory can win.\n\n2. Differ from DTG in the FALLBACK. Instead of DTG's divisively-normalised anti-validity sum with a load leak, make the no-anchor case a SINGLE-CUE lottery over rows sampled with probability proportional to salience s_j = (1-v_j)^gamma * exp(lam*j) (a probabilistic one-reason / 'sampling the oddest late row' rule) rather than a compensatory sum. A sampling rule naturally reproduces (i) the ~.70-.75 ceiling on long unanimous non-anchor displays (Exp 7 obs .743) without needing a large explicit eps0, because samples from a unanimous display all agree but an occasional lapse/guess remains; (ii) the strongly below-chance validity-weighted metrics of Exps 3, 5, 6, 12, 14 (obs .28, .29, .15, .42, .26) because the sampled row is almost always the LEAST valid / latest one; (iii) crucially it predicts a SMALLER tally-vs-TTB gap than a compensatory sum on Exp 4 (obs .174, DTG over-predicts at .318), because a sampler does not add coalition members the way a weighted sum does. That Exp-4 discrepancy is the clearest remaining crack in DTG and should be the new theory's target.\n\n3. Make the near-chance trigger and the sampler ONE mechanism rather than two limbs: p(consult row j) = softmax over delta*log(1-v_j) + lam*j with a LARGE contrast exponent delta (so a .52 row wins the lottery with probability ~1 whenever it speaks, reproducing the .86 anchor-follow automatically), and a moderate delta-driven spread among mid rows when no anchor speaks. Add a single lapse/guess rate eps (.12-.20) and, optionally, a mild length-dependent guess increment to match the .74 dominance ceiling. This gives a parsimonious, genuinely different architecture (one graded sampling rule, no mode-switch gate, no compensatory integration) that makes distinguishable predictions from DTG on: coalition-size effects with the anchor silent (Exps 15, 16, 19, where observed values hug .47-.55 \ns a pure sampler predicts near-chance there naturally), the Exp-4 style tally-vs-TTB difference, and the exact ceiling of long unanimous displays.\n\n4. Retain the degeneracy clause (validities within .002 -> unweighted graded row tally) and stationarity, since nothing in the data contradicts them."
}
```

## Usage

```json
{
  "input_tokens": 51005,
  "output_tokens": 3119
}
```
