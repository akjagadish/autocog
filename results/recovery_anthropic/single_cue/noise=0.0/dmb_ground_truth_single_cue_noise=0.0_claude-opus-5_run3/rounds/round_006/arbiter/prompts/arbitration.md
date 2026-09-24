# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_8" and "pi_7") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_8" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_7" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_8
ATTENTION LOTTERY WITH LIMITED-SAMPLE COMPARISON, AUTHORITY-CONTINGENT CAPTURE, PANEL-VOLUME CONFIDENCE, AND A CATEGORICAL (STEP-LIKE) PANEL-INFORMATIVENESS GATE. (1) Each display row enters an attention lottery with weight att_j = s_j * x_j * pop_j, where s_j = exp(-j/tau) + gamma*exp(-(n-1-j)/tau) is literal screen salience (asymmetric: top of panel stronger than bottom), x_j = ((v_j-.5)/.5)^rho is mildly compressed stated diagnosticity (rho ~ .55-.85), and pop_j is a pop-out factor: rows on which the two columns are identical do not pop out and keep only a residual share delta, while the FIRST row at which the columns disagree gets a bonus. (2) That first-mismatch bonus is AUTHORITY-CONTINGENT: a row captures attention outright only if it is both the first thing that discriminates AND the most diagnostic discriminating row on the screen. Position is always active, never gated by a deadlock rule. (3) The subject does not integrate the panel: only k = clip(ceil(k_frac*m), 2, 3) rows are actually compared, drawn without replacement with probability proportional to att. Choice = the attention-weighted sign of the sampled differences plus a small within-sample count term; a 1-1 sample is decided by the more salient sampled row. Numerosity helps only because more rows on one side makes the sample more likely to be dominated by that side; there is no unanimity bonus, no additive majority bonus, no serial stop rule. Because the sample is small and random, the map from true evidence to P(choice) is compressive with a ceiling near .85-.90 even for landslides. (4) NEW COMMITMENT (this iteration): THE READER'S JUDGEMENT OF WHETHER THE PANEL IS INFORMATIVE AT ALL IS CATEGORICAL, NOT GRADED. Silence is read as sensitivity, but the subject does not compute a continuous 'how much of the panel was mute' discount; they make a yes/no assessment — 'did most of this panel have nothing to say?' — with the boundary at about two-fifths of the panel (f0 ~ .39) and an essentially step-like transition (kappa ~ 100). Below the boundary the sampled verdict is taken at full face value (c ~ 1); above it the verdict is essentially voided and the choice approaches a coin flip (c floored near .03). There is no intermediate regime in which a panel is 'somewhat' informative. Mutual endorsements (both columns say yes) count toward silence at weight omega ~ .84, i.e. they are nearly, but not quite, as uninformative as mutual abstentions; because the gate is now a step, this small composition weight decides on which SIDE of the categorical boundary a mixed panel falls, which is where the whole behavioural signature lives. (5) Confidence also scales with the VOLUME of the panel, not with the amount actually read: beta_eff = beta*(n/6)^zeta (illusion of knowledge — short panels are systematically less decisive than structurally identical cells on wide panels). (6) Individual differences are unimodal jitter in k_frac, tau, gamma, rho, phi, phi_auth, s_auth, beta, zeta, eta, omega, c_min and a lapse rate; no strategy subpopulations, no learning.

`predict(parameters, state, history) -> np.ndarray`:
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

`policy(probs) -> int`:
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

## THEORY 2 — pi_7
**Reason-counting with a TWO-REGIME display-salience gate (full force only on a deadlocked vote, a constant residual otherwise), near-linear diagnosticity weighting, a saturating majority bonus, unanimity boost, sharp silence-to-confidence collapse (mutual endorsement ≈ silence), and a non-zero floor on read-out gain.**

This keeps every mechanism of the accepted base and changes exactly three calibration commitments (one of them structural-but-local: the shape of the salience gate).

1. **Direction is a vote over discriminating experts, graded near-linearly by stated validity.** Each discriminating expert contributes x_j = ((v_j-0.5)/0.5)^rho with rho ~0.88-1.02. At this exponent a single .95 expert is worth about two upper-middling experts (.72 + .70), so numerosity and diagnosticity are *balanced* rather than one dominating: three mid experts still outvote one top expert, but two mid experts only tie it. Screen position plays no role in this term.

2. **Display salience is a TIE-BREAKER WITH A RESIDUAL, not a quantity that decays with the size of the majority.** s_j = exp(-j/tau)+exp(-(n-1-j)/tau) over the literal display row, multiplied by a gate g = max(exp(-(k_A-k_B)^2/2 sigma^2), g_res) with sigma ~0.60 and g_res ~0.25. The gate has two regimes: when the reason count is exactly deadlocked the first/last rows of the panel decide the choice outright (g = 1); as soon as one side has more reasons, position stops being decisive but does NOT disappear — it persists as a constant, modest attention bonus (g = g_res ≈ 0.25) that is the SAME for a one-reason edge and for a five-against-one landslide. The previous Gaussian-only gate made position vanish geometrically with the majority size, which is unmotivated: there is no reason why a larger majority should erase the extra attention paid to the top and bottom of the screen. The empirical signature is that a three-against-one display whose majority occupies both screen ends is markedly more decisive (~.88) than a three-against-one display whose lone dissenter sits at a screen end (~.82), even when the validities are matched.

3. **Clear majorities are categorical and saturating:** M = lambda*sign(k_A-k_B)*min(max(|k_A-k_B|-1,0),1) with lambda ~0.67. Once one side has at least two more reasons, 'more experts back this one' is itself a discrete reason, and it does not grow further with the size of the majority.

4. **Unanimity among the speaking experts is categorical**: if every expert who noticed a difference points the same way, a constant bonus is added.

5. **Gap inference from the panel is a sharp threshold on how much of the panel was informative, and 'both products are good' is nearly as uninformative as 'neither is'.** Validity is read as sensitivity: a difference only sensitive experts notice implies a marginal true gap. Mutual endorsements count at weight omega ~0.8. c = 1 - eta*sigmoid(k*(f_sil - f0))*vbar.

6. **The collapse of confidence is bounded below:** c is floored at c_min ~0.10, so a panel judged uninformative leaves a small residual tilt rather than a literal coin flip.

7. **A small, display-contingent first-cue commitment survives**: people commit to the first expert who actually discriminates only if no expert further down the list outranks him in stated validity. Individual differences are unimodal jitter in rho, mu, sigma, g_res, lambda, eta, omega, c_min, beta; no strategy subpopulations, no serial stop rule, no learning.

`predict(parameters, state, history) -> np.ndarray`:
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

`policy(probs) -> int`:
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

## EXPERIMENT 1 (proposed by pi_8)

### DESIGN
**Validities (n_features=8):** [0.82, 0.6, 0.95, 0.88, 0.72, 0.58, 0.68, 0.76]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 0, 0, 0, 1, 0, 1]  B=[0, 1, 1, 1, 1, 1, 1, 0]
  trial 2: A=[0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1, 0, 1]
  trial 3: A=[1, 0, 1, 1, 0, 1, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  trial 4: A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 0, 1, 0, 0]
  trial 5: A=[0, 1, 1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  trial 6: A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0, 1, 1, 1]
  trial 7: A=[1, 1, 1, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 1, 1, 1, 0]
  trial 8: A=[0, 1, 0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 1, 0, 1]
  trial 9: A=[1, 0, 1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0, 1, 0]
  trial 10: A=[0, 0, 0, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0, 0, 0, 1]
  trial 11: A=[1, 1, 0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 1, 1]
  trial 13: A=[1, 1, 1, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  trial 14: A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 1, 1]
  trial 15: A=[1, 0, 1, 1, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  trial 16: A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 0, 0, 0, 1]

**Rationale:** GOAL: dissociate the ADVOCATED attention-lottery model (asymmetric screen salience with tau~0.85 so only rows 0,1,6,7 are really visible; authority-contingent first-mismatch capture gated by the most diagnostic DISCRIMINATING row only; limited-sample comparison of k=2-3 rows; panel-volume confidence beta_eff=beta*(n/6)^zeta; and above all the NEW commitment that the panel-informativeness judgement is CATEGORICAL, a step at f0~0.388 with kappa~100, with mutual endorsements counted at omega~0.84) from the COMPETING reason-counting model (near-linear validity vote over ALL discriminating experts, categorical unanimity bonus ups~1.5, saturating majority bonus lam~0.67, symmetric first/last-row salience entering at the constant residual g_res~0.25 whenever the vote is not deadlocked, a SHALLOW silence sigmoid sil_k~38 centred at 0.36, floor c_min~0.10, and a first-cue commitment whose authority cap is computed over ALL later rows including silent ones).

n=8, validities by SCREEN position [.82,.60,.95,.88,.72,.58,.68,.76]: the two heavyweights (.95,.88) sit mid-list at pos2/pos3 where the advocated model's salience is ~0.10/0.03 of pos0's, the screen extremes are mid-high (.82 at pos0, .76 at pos7), and the weakest experts (.60,.58) fill pos1/pos5. All displays are deliberately DENSE (>=4 discriminating rows): with n=8 any panel with 5+ mute rows is voided by BOTH theories, so sparse cells are uninformative by construction. Because no cell is vote-deadlocked, the competitor's salience term always sits at its constant residual g_res, which makes its predictions tight. 8 base cells, each mirrored across sides (16 unique pairs, K=6 -> 96 trials); mirroring forces an exact 8A/8B split for both theories.

(1) EXT (pairs 1-2): A=[1,1,0,0,0,1,0,1] vs B=[0,1,1,1,1,1,1,0]. The two screen extremes (.82,.76) favour A; the four mid-list rows (.95,.88,.72,.68) favour B; 2 mute rows are mutual endorsements (f_sil=.21, both theories at full confidence). Advocated: att(pos0)=1.57 and att(pos7)=0.32 against 0.13 for ALL four mid rows together, so the 3-row lottery almost never sees the mid-list -> P(A)~.93. Competitor: V=-1.35, majority bonus -0.67, residual salience +0.62 -> P(A)~.22. ~70-point DIRECTION disagreement over both full parameter boxes.

(2) AUTH (pairs 3-4): A=[1,0,1,1,0,1,0,0] vs B=[0,1,1,1,1,1,1,1]. pos0 (.82) alone favours A against four lower-validity rows (.60,.72,.68,.76); crucially the .95 and .88 experts are MUTE (mutual endorsements, f_sil=.315). This isolates the two theories' different authority gates: the advocated model's first-mismatch capture asks only whether pos0 is the most diagnostic DISCRIMINATING row (it is: ratio 1.16 -> q_auth~.83, capture factor ~6.5, att(pos0)=4.7 vs 0.5 for all four opponents) -> P(A)~.90; the competitor's commitment gate is computed against the max validity of ALL later rows, i.e. the silent .95 expert, so q_com~.07 and it falls back on the vote (V=-0.94, M=-0.67, salience -0.37) -> P(A)~.18. Again ~70 points, opposite direction, and the mechanism is the silent-vs-speaking composition of the gate.

(3) TAIL (pairs 5-6): A=[0,1,1,1,0,1,1,1] vs B=[1,1,1,1,1,0,0,0]. pos0 (.82) and pos4 (.72) favour B; the tail rows pos5/pos6/pos7 (.58,.68,.76) favour A; f_sil=.315. Validities were chosen so the competitor's vote is almost exactly null (V=-0.02 to +0.01 across rho in [.88,1.02]) and only its residual salience tilts it (+0.18 to +0.30), giving P(A)~.51-.55, i.e. a coin flip. The advocated model captures on pos0 (q_auth~.83, att 4.7 vs 0.41 for the whole tail) and predicts P(A)~.08. This cell also reverses the side that pos0 favours relative to cells 1-2, so 'follow the top row' cannot be confounded with a side bias.

(4-5) SILENCE-COMPOSITION CONTRAST at fixed d, SIL-LO (pairs 7-8) vs SIL-HI (pairs 9-10). Identical difference vector (pos0,pos2,pos7 favour A; pos4,pos6 favour B); the three mute rows are mutual ENDORSEMENTS in SIL-LO (f_sil=.84*3/8=.315) and mutual ABSTENTIONS in SIL-HI (f_sil=.375). Both values lie BELOW the advocated step (f0 in [.380,.396]) but straddle the competitor's shallow sigmoid centre (.36): advocated c goes 1.00 -> .81 (P .95 -> .92, and >=.87 anywhere in its box), whereas competitor c goes .85 -> .37 (P .88 -> .71). A ~17-point graded drop vs a ~3-point near-null, produced entirely by ratings that discriminate nothing, is the direct test of categorical-vs-graded panel informativeness.

(6-7) INVISIBLE-DISSENTER / UNANIMITY CONTRAST at fixed silence, UNI41 (pairs 11-12) vs UNI50 (pairs 13-14). Both have the same three mute both-0 rows (f_sil=.375) and the same five speaking experts (pos0,pos1,pos2,pos6,pos7); the ONLY change is the .95 expert at pos2 flipping from favouring B (4-1) to favouring A (5-0). The advocated model's lottery samples pos2 with probability ~.005 (its salience is 0.096 while pos0 carries the capture bonus), and its silence/vbar terms are identical because the speaker set is identical, so it is FORCED to predict the same probability in both cells (P(A)~.94 in each, difference <1 point). The competitor gains the categorical unanimity bonus (+1.5) plus ~1.9 of vote, moving from P(A)~.76 to ~.91. A 15-point within-subject jump where the advocated theory predicts an exact null.

(8) VOID (pairs 15-16): A=[1,0,1,1,0,0,0,1] vs B=[0,0,0,0,0,0,0,0]. Four experts (incl. the .95 and .88) unanimously endorse A, four endorse neither (f_sil=.50), i.e. the strongest verdict a half-mute panel can deliver. Both theories judge the panel uninformative, but the advocated model voids it essentially completely (c floored at .015-.055) and therefore predicts a LITERAL coin flip, P(A)=.51-.55 across its whole box, despite a 4-0 landslide, whereas the competitor's floor c_min~.10 multiplied by a large drive (V=2.89 + unanimity 1.5 + majority .67 + salience 1.01) leaves a reliable residual tilt, P(A)~.62-.72. Paired with UNI50 (same all-zero competitor, one more mute row) this is a 41-point cliff for the advocated model vs 26 points for the competitor, and the absolute level at VOID (.53 vs .67) is the decisive statistic.

DIAGNOSTIC YIELD: three full direction/level disagreements (EXT, AUTH ~70 points each; TAIL, near-determinism vs chance, ~43 points); one forced null for the competitor's silence sigmoid turned into a 17-point graded drop (SIL-LO vs SIL-HI); one forced null for the advocated lottery turned into a 15-point unanimity jump (UNI41 vs UNI50); and a coin-flip-vs-residual-tilt test of the two floors (VOID). Every pair has at least one discriminating cue and no cell is degenerate.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
The design's three 'direction-disagreement' cells (EXT, AUTH, TAIL) were constructed so that the advocated attention-lottery model, via asymmetric screen salience plus authority-contingent first-mismatch capture, follows the TOP screen row (position 0, validity .82) almost deterministically (P ~ .90-.93 per cell), while the competing reason-counting model integrates the full validity-weighted vote over all discriminating experts and therefore goes the OTHER way in EXT and AUTH (P(top-row option) ~ .18-.22) and is essentially at chance in TAIL (~ .47). The metric is simply the proportion of choices, restricted to those six displays (3 cells x 2 mirror orders, 36 trials per subject), that agree with the option favoured by the first discriminating screen row. Predicted values: ~0.90 under the advocated theory, ~0.29 under the competitor -- a ~60-point gap. Because it pools 36 trials per subject and both theories make near-extreme, parameter-robust predictions in these cells (the design rationale states the direction holds across both full parameter boxes), the per-subject sampling variance is small (binomial var ~0.002-0.007 plus modest parameter jitter), so the contrast should survive Welch's test at N=25. Mirroring A/B across the six displays removes any side-bias confound, and the metric is well defined on both pooled and single-subject slices.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # The six target displays (EXT, AUTH, TAIL cells and their mirrors).
    targets = {
        # EXT
        ("11000101", "01111110"),
        ("01111110", "11000101"),
        # AUTH
        ("10110100", "01111111"),
        ("01111111", "10110100"),
        # TAIL
        ("01110111", "11111000"),
        ("11111000", "01110111"),
    }

    def as_str(x):
        try:
            return "".join([str(int(round(float(v)))) for v in x])
        except Exception:
            return ""

    hits = 0
    tot = 0
    for _, row in data.iterrows():
        try:
            a = [int(round(float(v))) for v in row["option_a_ratings"]]
            b = [int(round(float(v))) for v in row["option_b_ratings"]]
        except Exception:
            continue
        ka = "".join([str(v) for v in a])
        kb = "".join([str(v) for v in b])
        if (ka, kb) not in targets:
            continue
        # first (top-most) screen row on which the two columns disagree
        j = -1
        for i in range(min(len(a), len(b))):
            if a[i] != b[i]:
                j = i
                break
        if j < 0:
            continue
        try:
            resp = int(row["response"])
        except Exception:
            continue
        if resp not in (0, 1):
            continue
        favors_a = a[j] > b[j]
        followed = (favors_a and resp == 0) or ((not favors_a) and resp == 1)
        hits += 1 if followed else 0
        tot += 1

    if tot == 0:
        return 0.5
    return float(hits) / float(tot)

### RESULTS
- Predicted under pi_8 (simulated): 0.9261 (var=0.0024)
- Predicted under pi_7 (simulated): 0.2983 (var=0.0068)
- Observed on real data: 0.3533 (var=0.0101)

## EXPERIMENT 2 (proposed by pi_7)

### DESIGN
**Validities (n_features=6):** [0.74, 0.95, 0.62, 0.86, 0.58, 0.74]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 1, 1, 1, 0]
  trial 2: A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 1, 0, 1, 1]
  trial 3: A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1, 0]
  trial 4: A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1, 1]
  trial 5: A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 6: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0]
  trial 7: A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 1, 0, 1, 1]
  trial 8: A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  trial 9: A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 1, 0, 1, 1]
  trial 10: A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  trial 11: A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 1, 0]
  trial 12: A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 13: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 1, 0, 1, 0]
  trial 14: A=[0, 0, 1, 0, 1, 0]  B=[1, 1, 1, 1, 1, 1]
  trial 15: A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 16: A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 1, 0, 0]

**Rationale:** GOAL: dissociate the ADVOCATED reason-counting model (near-linear validity vote over ALL discriminating experts, SYMMETRIC first/last-row salience that is at FULL force only when the reason count is deadlocked and at a constant residual g_res~.25 otherwise, saturating majority bonus lam~.67, categorical unanimity bonus, authority cap computed over ALL later rows INCLUDING silent ones) from the COMPETING attention-lottery model (ASYMMETRIC salience with top >> bottom, huge authority-contingent first-mismatch capture gated only by later SPEAKING rows, limited sample of k=2-3 rows, no unanimity/majority bonuses, compressive ceiling ~.90).

VALIDITIES BY SCREEN POSITION: [.74, .95, .62, .86, .58, .74]. n=6 makes the competitor's panel-volume term exactly neutral (beta*(6/6)^zeta = beta). The two SCREEN EXTREMES are given IDENTICAL validity (.74) - this is the key design move (see SWAP below). The oracle (.95) sits at pos1 and the second heavyweight (.86) at pos3, i.e. in the rows the lottery can barely see (competitor attention s*x = [.60,.29,.04,.06,.05,.30] before capture; advocated salience is the symmetric U [1.01,.39,.19,.19,.39,1.01]). Because pos1 (.95) is always present later in the list, the advocated model's first-cue commitment is uniformly suppressed whenever pos0 speaks (ratio .48/(1.05*.90)=.51 -> q_com<=.07 even at its worst corner), so it cannot contaminate any contrast. Every cell has >=4 speaking rows and <=2 silent rows, and all silent rows are MUTUAL ENDORSEMENTS, so f_sil<=.33 for both theories and both silence gates are provably OFF everywhere - no result can be blamed on silence machinery.

8 base cells, each mirrored across sides -> 16 unique pairs, K=6 -> 96 trials; mirroring forces an exact 8A/8B split for BOTH theories.

(1) FLAGSHIP FORCED NULL: TOP/BOTTOM SWAP, cells SWAP-A (pairs 9-10) and SWAP-B (pairs 11-12). SWAP-A has d=[+1,-1,0,+1,0,-1], SWAP-B has d=[-1,-1,0,+1,0,+1]: the two extremes simply trade sides while pos1 and pos3 are held fixed. Because pos0 and pos5 have EQUAL validity and EQUAL (symmetric) salience, the advocated model's vote V=-.173 and its salience sum S=-.201 are ANALYTICALLY IDENTICAL in the two cells for EVERY (rho, mu, tau, sigma, g_res) - both are vote-deadlocked (kA=kB=2, g=1), so it is mathematically forced to predict the same P(A)~.35 in both (residual <=3-6 points from the suppressed commitment term). The lottery is anchored on the top row (att(pos0)/att(pos5) = 1.6-2.6 from gamma alone, times a >=1.5 first-mismatch factor) and therefore REVERSES: P(A)~.71 in SWAP-A vs ~.20 in SWAP-B, a ~50-point swing. This isolates symmetric-vs-asymmetric salience with a forced null on one side.

(2) DEADLOCK-GATE REVERSAL, cells D1 (pairs 1-2) and D2 (pairs 3-4). D1: d=[+1,-1,0,-1,0,+1] - the two extremes (.74,.74) favour A, the two mid heavyweights (.95,.86) favour B, kA=kB=2, so the advocated gate opens to FULL force (drive = -.64 + mu*1.44 = +1.98) -> P(A)~.89. D2 adds ONE extra weak dissenter at pos2 (.62): the count is no longer deadlocked, the gate collapses to g_res=.25 and the drive flips to -.33 -> P(A)~.41, a 48-point DIRECTION REVERSAL produced by a single low-validity mid-list row. The lottery barely samples pos2 (att .04 out of ~1.7) and is forced to a near-null: P(A) ~.81 -> ~.75. A large within-subject reversal vs an approximate null.

(3) INVISIBLE-MAJORITY CONTRAST, cells V1 (pairs 5-6) and V2 (pairs 7-8). Both have pos0 (.74) favouring A and pos5 (.74) favouring B; the ONLY change is the direction of the two nearly invisible mid-list heavyweights (pos1 .95, pos3 .86). V1 (they oppose A): advocated V=-1.64 plus majority bonus -.67 -> P(A)~.08; V2 (they back A): drive +2.57 -> P(A)~.95, i.e. an ~87-point within-subject swing. The lottery's sampling probability for pos1/pos3 is ~.21 combined and its evidence is normalised within a 2-row sample, so it predicts only ~.70 -> ~.81 (11 points) and, crucially, chooses A in BOTH cells: V1 alone is a ~62-point DIRECTION disagreement.

(4) AUTHORITY-GATE CELL, AUTH (pairs 15-16): d=[+1,0,-1,0,-1,-1] with pos1 (.95) and pos3 (.86) MUTE (mutual endorsements). pos0 (.74) is the most diagnostic SPEAKING row, so the lottery's authority-contingent capture fires almost fully (q_auth~.5-1 -> att(pos0)=2.7 of 3.1) and it follows the top row: P(A)~.82-.85. The advocated model's authority cap is computed against ALL later rows including the silent .95 expert, so q_com~.02; it counts three cheap reasons plus the saturating majority bonus and chooses the other option: P(A)~.14-.20. ~65 points, opposite direction, over both full parameter boxes, with the silence gate provably off.

(5) CEILING ANCHOR, LAND (pairs 13-14): d=[+1,+1,0,+1,0,+1], a 4-0 landslide. The advocated model adds unanimity (+1.5) and majority (+.67) bonuses to a large vote -> P(A)~.96-.97; the lottery's small-sample read-out is capped at sigmoid(beta_eff)~.89 no matter how lopsided the panel -> P(A)~.87-.89. This cell also serves as an engagement check and calibrates the compressive-ceiling claim.

DIAGNOSTIC YIELD: (i) one analytic forced null for the advocated theory (SWAP-A vs SWAP-B) against a ~50-point reversal for the lottery; (ii) one near-null for the lottery (D1 vs D2) against a 48-point direction reversal for the advocated theory; (iii) an 87-vs-11-point invisible-majority contrast plus two outright direction disagreements (V1, AUTH, ~60-65 points each); (iv) a ceiling test of limited-sample compression. No cell is degenerate and no contrast depends on either model's silence machinery.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
The metric is the proportion of trials, pooled over two hand-picked stimulus cells (identified by content so that mirrored A/B presentations are collapsed), on which the subject chooses a designated 'focal' option.

Cell V1 ([1,0,1,0,1,0] vs [0,1,1,1,1,1]): the focal option is backed only by the top screen row (.74) while the two most valid experts (.95 at row 2 and .86 at row 4) plus a numerical majority oppose it. The advocated reason-counting model integrates ALL discriminating experts with near-linear validity weights and adds a saturating majority bonus, so it decisively rejects the focal option (P(focal) ~ .08). The attention-lottery model samples only k=2-3 rows drawn with weights strongly biased to the top of the panel and heavily discounted for the low-salience mid rows, so it mostly never reads the .95/.86 evidence and chooses the focal option most of the time (P(focal) ~ .70).

Cell AUTH ([1,1,0,1,0,0] vs [0,1,1,1,1,1]): the .95 and .86 experts are mute (mutual endorsements), so the top speaking row (.74) is the most diagnostic speaker and triggers the competitor's authority-contingent first-mismatch capture, driving it to follow that top row and pick the focal (P(focal) ~ .83). The advocated model computes its authority cap over ALL later rows including the silent .95 expert, so its first-cue commitment is suppressed; it instead counts three cheap dissenting reasons plus the majority bonus and rejects the focal (P(focal) ~ .17).

Both cells push in the SAME direction across theories (advocated low, competitor high), so averaging them roughly doubles the contrast while averaging over 24 trials per subject keeps binomial noise small (~.07 SD). Expected values: ~0.13 under the advocated theory versus ~0.77 under the attention lottery — a ~0.64 gap with tight per-subject spread. Crucially the metric is not trivially true for either model: it is a plain choice proportion on two ordinary stimulus pairs, and either model could in principle land anywhere in [0,1].

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Two diagnostic cells (each appears mirrored across sides, 6 reps each).
    # Cell V1  : focal = [1,0,1,0,1,0]  vs  [0,1,1,1,1,1]
    #            (two invisible mid-list heavyweights .95/.86 oppose the focal;
    #             screen extremes .74/.74 split)
    # Cell AUTH: focal = [1,1,0,1,0,0]  vs  [0,1,1,1,1,1]
    #            (pos1 .95 and pos3 .86 are mute mutual endorsements; the top
    #             speaking row .74 backs the focal, three cheaper rows oppose it)
    cells = [
        ((1, 0, 1, 0, 1, 0), (0, 1, 1, 1, 1, 1)),
        ((1, 1, 0, 1, 0, 0), (0, 1, 1, 1, 1, 1)),
    ]

    def _tup(v):
        try:
            return tuple(int(round(float(z))) for z in v)
        except Exception:
            return None

    if len(data) == 0:
        return float('nan')

    av = data['option_a_ratings'].apply(_tup).values
    bv = data['option_b_ratings'].apply(_tup).values
    try:
        rv = data['response'].astype(float).values
    except Exception:
        return float('nan')

    hits = []
    for i in range(len(av)):
        A = av[i]
        B = bv[i]
        if A is None or B is None:
            continue
        r = rv[i]
        if not np.isfinite(r):
            continue
        r = int(round(r))
        for focal, other in cells:
            if (A == focal and B == other) or (A == other and B == focal):
                chosen = A if r == 0 else B
                hits.append(1.0 if chosen == focal else 0.0)
                break

    if len(hits) == 0:
        return float('nan')
    return float(np.mean(hits))

### RESULTS
- Predicted under pi_8 (simulated): 0.7658 (var=0.0055)
- Predicted under pi_7 (simulated): 0.1350 (var=0.0054)
- Observed on real data: 0.3000 (var=0.0112)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    dis_hits = 0.0
    dis_n = 0.0
    agr_hits = 0.0
    agr_n = 0.0

    for a_raw, b_raw, resp in zip(data['option_a_ratings'],
                                  data['option_b_ratings'],
                                  data['response']):
        a = np.asarray(a_raw, dtype=float)
        b = np.asarray(b_raw, dtype=float)
        if a.shape != b.shape or a.size == 0:
            continue

        # Take-The-Best winner: features are listed in descending validity
        # order, so scan left to right for the first discriminating cue.
        ttb_winner = None
        for j in range(a.size):
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break
        if ttb_winner is None:
            continue

        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        if a_wins == b_wins:
            continue  # tally-tie pairs excluded (uninformative for the contrast)
        tally_winner = 0 if a_wins > b_wins else 1

        try:
            r = int(resp)
        except (TypeError, ValueError):
            continue
        hit = 1.0 if r == ttb_winner else 0.0

        if tally_winner != ttb_winner:
            dis_hits += hit
            dis_n += 1.0
        else:
            agr_hits += hit
            agr_n += 1.0

    if dis_n == 0.0 or agr_n == 0.0:
        return float('nan')

    return float(dis_hits / dis_n - agr_hits / agr_n)
```

**Observed (real) value:** 0.0933 (var=0.0064)
**Predicted under pi_8:** 0.0794 (var=0.0108)
**Predicted under pi_7:** 0.0921 (var=0.0072)

### Experiment 4
**Design**
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[0, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    validities = np.array([0.9, 0.6, 0.8, 0.55, 0.7, 0.95], dtype=float)

    n_match = 0.0
    n_total = 0.0

    for _, row in data.iterrows():
        a = np.asarray(list(row['option_a_ratings']), dtype=float)
        b = np.asarray(list(row['option_b_ratings']), dtype=float)
        n = a.shape[0]
        if validities.shape[0] == n:
            val = validities
        else:
            # fallback: assume given order is already descending validity
            val = np.arange(n, 0, -1, dtype=float)
        order = np.argsort(-val, kind='stable')

        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        if a_wins == b_wins:
            continue
        tally_winner = 0 if a_wins > b_wins else 1
        margin = abs(a_wins - b_wins)
        if margin < 2:
            continue

        ttb_winner = None
        for j in order:
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break
        if ttb_winner is None:
            continue
        if ttb_winner == tally_winner:
            continue  # only conflict trials

        try:
            resp = int(row['response'])
        except (TypeError, ValueError):
            continue
        n_total += 1.0
        if resp == tally_winner:
            n_match += 1.0

    if n_total == 0:
        return float('nan')
    return float(n_match / n_total)
```

**Observed (real) value:** 0.7750 (var=0.0072)
**Predicted under pi_8:** 0.7965 (var=0.0057)
**Predicted under pi_7:** 0.8510 (var=0.0036)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Rate of choosing the option favoured by the HIGHEST-VALIDITY cue (screen
    # position 3, validity .95) on the 'double-dissociation' pairs, i.e. trials
    # where the two screen-extreme cues (positions 0 and 5) jointly favour one
    # option while the middle top-validity cue favours the other.
    num = 0.0
    den = 0.0
    num_all = 0.0
    den_all = 0.0
    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()
    for a_raw, b_raw, r in zip(a_col, b_col, r_col):
        try:
            a = np.asarray(list(a_raw), dtype=float)
            b = np.asarray(list(b_raw), dtype=float)
        except Exception:
            continue
        if a.shape[0] < 6 or b.shape[0] < 6:
            continue
        d3 = a[3] - b[3]
        if d3 == 0:
            continue
        top_side = 0 if d3 > 0 else 1
        try:
            resp = int(r)
        except Exception:
            continue
        chose_top = 1.0 if resp == top_side else 0.0
        num_all += chose_top
        den_all += 1.0
        d0 = a[0] - b[0]
        d5 = a[5] - b[5]
        if d0 != 0 and d5 != 0:
            s0 = 1.0 if d0 > 0 else -1.0
            s5 = 1.0 if d5 > 0 else -1.0
            s3 = 1.0 if d3 > 0 else -1.0
            if s0 == s5 and s0 == -s3:
                num += chose_top
                den += 1.0
    if den > 0:
        return float(num / den)
    if den_all > 0:
        return float(num_all / den_all)
    return float('nan')
```

**Observed (real) value:** 0.5175 (var=0.0233)
**Predicted under pi_8:** 0.4700 (var=0.0191)
**Predicted under pi_7:** 0.4537 (var=0.0136)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    VAL = np.array([0.78, 0.60, 0.95, 0.88, 0.72], dtype=float)

    def ttb_winner(a, b, val):
        order = np.argsort(-val, kind='stable')
        for j in order:
            if a[j] > b[j]:
                return 0
            if b[j] > a[j]:
                return 1
        return None

    def scan_sign(d, lam):
        n = d.shape[0]
        pos = np.arange(n, dtype=float)
        z = -lam * (pos - pos.mean())
        z = z - z.max()
        w = np.exp(z)
        num = float(np.sum(w * d))
        den = float(np.sum(w * np.abs(d)))
        if den <= 1e-12:
            return 0.0
        return num / den

    hits = []
    for _, row in data.iterrows():
        try:
            a = np.asarray(list(row['option_a_ratings']), dtype=float)
            b = np.asarray(list(row['option_b_ratings']), dtype=float)
        except Exception:
            continue
        n = a.shape[0]
        if b.shape[0] != n or n == 0:
            continue
        val = VAL if n == VAL.shape[0] else np.linspace(0.95, 0.55, n)
        w_ttb = ttb_winner(a, b, val)
        if w_ttb is None:
            continue
        d = a - b
        s_prim = scan_sign(d, 2.0)    # primacy reader (lambda > 0)
        s_rec = scan_sign(d, -2.0)    # recency reader (lambda < 0)
        if abs(s_prim) < 0.15 or abs(s_rec) < 0.15:
            continue
        # scanner choice: S > 0 -> A (0), S < 0 -> B (1)
        c_prim = 0 if s_prim > 0 else 1
        c_rec = 0 if s_rec > 0 else 1
        # keep only cells where BOTH scanning orientations oppose TTB
        if c_prim == c_rec and c_prim != w_ttb:
            try:
                r = int(row['response'])
            except Exception:
                continue
            hits.append(1.0 if r == w_ttb else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))

```

**Observed (real) value:** 0.5075 (var=0.0075)
**Predicted under pi_8:** 0.3950 (var=0.0070)
**Predicted under pi_7:** 0.4238 (var=0.0074)

### Experiment 7
**Design**
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    hits = 0
    tot = 0
    for _, row in data.iterrows():
        try:
            a = np.asarray([int(v) for v in row['option_a_ratings']], dtype=float)
            b = np.asarray([int(v) for v in row['option_b_ratings']], dtype=float)
        except Exception:
            continue
        if a.shape[0] != 5 or b.shape[0] != 5:
            continue
        d = a - b
        # target cell: d = +-[1,-1,-1,1,0]  (pairs 1 and 2)
        pattern_ok = (
            abs(d[4]) < 1e-9
            and abs(abs(d[0]) - 1.0) < 1e-9
            and abs(abs(d[1]) - 1.0) < 1e-9
            and abs(abs(d[2]) - 1.0) < 1e-9
            and abs(abs(d[3]) - 1.0) < 1e-9
            and abs(d[0] - d[3]) < 1e-9
            and abs(d[1] - d[2]) < 1e-9
            and abs(d[0] + d[1]) < 1e-9
        )
        if not pattern_ok:
            continue
        try:
            resp = int(row['response'])
        except Exception:
            continue
        if resp not in (0, 1):
            continue
        # side supported by expert positions 0 and 3 (the first- and last-
        # discriminating cues); opposed by positions 1 and 2
        favored_is_a = d[0] > 0
        chose_favored = (resp == 0) if favored_is_a else (resp == 1)
        hits += 1 if chose_favored else 0
        tot += 1

    if tot == 0:
        return float('nan')
    return float(hits) / float(tot)

```

**Observed (real) value:** 0.8200 (var=0.0126)
**Predicted under pi_8:** 0.7863 (var=0.0111)
**Predicted under pi_7:** 0.8175 (var=0.0092)

### Experiment 8
**Design**
  A=[1, 0, 1, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 0, 1]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 1, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0, 1]  B=[1, 1, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    hits = 0
    n = 0
    for _, row in data.iterrows():
        try:
            a = np.asarray([int(v) for v in row['option_a_ratings']], dtype=float)
            b = np.asarray([int(v) for v in row['option_b_ratings']], dtype=float)
        except Exception:
            continue
        if a.shape[0] != b.shape[0] or a.shape[0] < 6:
            continue
        d = a - b
        disc = [int(i) for i in np.nonzero(np.abs(d) > 1e-9)[0]]
        target = None
        # Class (2): d nonzero at {0,2,5}, first and last agree, mid-list top cue opposes
        if disc == [0, 2, 5]:
            if d[0] == d[5] and d[2] == -d[0]:
                target = 0 if d[0] > 0 else 1
        # Class (3): d nonzero at {0,1,2,5}, three cues agree, single top cue opposes
        elif disc == [0, 1, 2, 5]:
            if d[0] == d[1] == d[5] and d[2] == -d[0]:
                target = 0 if d[0] > 0 else 1
        if target is None:
            continue
        n += 1
        try:
            resp = int(row['response'])
        except Exception:
            continue
        if resp == target:
            hits += 1
    if n == 0:
        return 0.5
    return float(hits) / float(n)

```

**Observed (real) value:** 0.8413 (var=0.0134)
**Predicted under pi_8:** 0.6850 (var=0.0040)
**Predicted under pi_7:** 0.6869 (var=0.0059)

### Experiment 9
**Design**
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[1, 1, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[0, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 1, 1, 0]  B=[1, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    hi_load_hits, hi_load_n = 0, 0
    lo_load_hits, lo_load_n = 0, 0

    for a_raw, b_raw, resp in zip(data['option_a_ratings'],
                                  data['option_b_ratings'],
                                  data['response']):
        try:
            a = np.asarray(list(a_raw), dtype=float)
            b = np.asarray(list(b_raw), dtype=float)
        except Exception:
            continue
        n = int(min(a.shape[0], b.shape[0]))
        if n == 0:
            continue
        a = a[:n]
        b = b[:n]
        d = a - b

        pos_idx = np.where(d > 0.5)[0]   # features favouring A
        neg_idx = np.where(d < -0.5)[0]  # features favouring B
        k_a = pos_idx.size
        k_b = neg_idx.size

        # keep only the 3-cue conflict cells: two cues on one side, one on the other
        if not ((k_a == 2 and k_b == 1) or (k_a == 1 and k_b == 2)):
            continue

        # did the subject pick the side backed by the LARGER NUMBER of cues?
        if k_a == 2:
            chose_many = 1 if int(resp) == 0 else 0
        else:
            chose_many = 1 if int(resp) == 1 else 0

        # processing load = shared positive endorsements (non-discriminating 1s)
        L = int(np.count_nonzero((a > 0.5) & (b > 0.5)))

        if L > 0:
            hi_load_hits += chose_many
            hi_load_n += 1
        else:
            lo_load_hits += chose_many
            lo_load_n += 1

    if hi_load_n == 0 or lo_load_n == 0:
        return 0.0

    return float(hi_load_hits / hi_load_n - lo_load_hits / lo_load_n)

```

**Observed (real) value:** 0.0533 (var=0.0494)
**Predicted under pi_8:** 0.0183 (var=0.0473)
**Predicted under pi_7:** -0.0733 (var=0.0466)

### Experiment 10
**Design**
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 1]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 1, 1, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 1]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    weak_idx = {1, 4}      # validities .56 and .62
    strong_idx = {0, 3}    # validities .72 and .95

    weak_hits = []
    strong_hits = []

    for _, row in data.iterrows():
        try:
            a = np.asarray(list(row['option_a_ratings']), dtype=float)
            b = np.asarray(list(row['option_b_ratings']), dtype=float)
        except Exception:
            continue
        if a.shape[0] != b.shape[0] or a.shape[0] == 0:
            continue
        d = a - b
        nz = np.flatnonzero(np.abs(d) > 1e-9)
        if nz.size != 1:
            continue  # only single-discriminating-cue trials
        j = int(nz[0])
        # 1 if subject chose the option endorsed by the single discriminating expert
        try:
            resp = int(row['response'])
        except Exception:
            continue
        chose_a = (resp == 0)
        endorsed_a = d[j] > 0
        hit = 1.0 if (chose_a == endorsed_a) else 0.0
        if j in weak_idx:
            weak_hits.append(hit)
        elif j in strong_idx:
            strong_hits.append(hit)

    if len(weak_hits) == 0 or len(strong_hits) == 0:
        return float('nan')

    return float(np.mean(strong_hits) - np.mean(weak_hits))

```

**Observed (real) value:** -0.2025 (var=0.0220)
**Predicted under pi_8:** -0.2113 (var=0.0237)
**Predicted under pi_7:** -0.2100 (var=0.0192)

### Experiment 11
**Design**
  A=[1, 1, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[1, 0, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0, 1]  B=[0, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 1, 1, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Cells where BOTH scanning orientations (primacy and recency) predict the
    # option favoured by the two screen-extreme cues (pos0 = .72, pos5 = .70),
    # while the advocated model predicts a coin flip (cell 5) or a reversal
    # toward the mid-list crowd (cell 8).
    #   cell 5: d = (+1, 0, -1, 0, 0, +1)  and its mirror
    #   cell 8: d = (+1, -1, -1, -1, 0, +1) and its mirror
    targets = {
        (1, 0, -1, 0, 0, 1),
        (-1, 0, 1, 0, 0, -1),
        (1, -1, -1, -1, 0, 1),
        (-1, 1, 1, 1, 0, -1),
    }

    hits = []
    for _, row in data.iterrows():
        try:
            a = [int(v) for v in row['option_a_ratings']]
            b = [int(v) for v in row['option_b_ratings']]
        except Exception:
            continue
        if len(a) != 6 or len(b) != 6:
            continue
        d = tuple(int(x) - int(y) for x, y in zip(a, b))
        if d not in targets:
            continue
        resp = row['response']
        try:
            resp = int(resp)
        except Exception:
            continue
        # option favoured by the screen-extreme cues (positions 0 and 5 agree)
        fav_a = (d[0] > 0)
        chose_a = (resp == 0)
        hits.append(1.0 if (chose_a == fav_a) else 0.0)

    if len(hits) == 0:
        return float('nan')
    return float(np.mean(hits))
```

**Observed (real) value:** 0.4900 (var=0.0105)
**Predicted under pi_8:** 0.7292 (var=0.0071)
**Predicted under pi_7:** 0.4583 (var=0.0105)

### Experiment 12
**Design**
  A=[1, 1, 1, 1, 0, 1]  B=[1, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 1, 0, 1]  B=[1, 1, 1, 1, 0, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 1, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Cell-5 pairs (trials 9 and 10 of the design), mirrored across sides:
    #   d = a - b = [+1, 0, -1, -1, 0, +1]   -> extremes (pos0=.62, pos5=.58) favour A
    #   d = a - b = [-1, 0, +1, +1, 0, -1]   -> extremes favour B
    # Score = fraction of these trials on which the subject chose the option
    # endorsed by the two SCREEN-EXTREME (weakest) experts, i.e. against the
    # two mid-list heavyweight experts (.95 and .86).
    target_A = (1, 0, -1, -1, 0, 1)
    target_B = (-1, 0, 1, 1, 0, -1)

    hits = 0
    n = 0
    for a_raw, b_raw, resp in zip(data['option_a_ratings'],
                                  data['option_b_ratings'],
                                  data['response']):
        try:
            a = [int(v) for v in a_raw]
            b = [int(v) for v in b_raw]
        except Exception:
            continue
        if len(a) != len(b) or len(a) != 6:
            continue
        d = tuple(int(x - y) for x, y in zip(a, b))
        if d == target_A:
            extreme_side = 0  # option A
        elif d == target_B:
            extreme_side = 1  # option B
        else:
            continue
        n += 1
        try:
            r = int(resp)
        except Exception:
            continue
        if r == extreme_side:
            hits += 1

    if n == 0:
        return 0.5
    return float(hits) / float(n)
```

**Observed (real) value:** 0.8500 (var=0.0111)
**Predicted under pi_8:** 0.8183 (var=0.0177)
**Predicted under pi_7:** 0.8817 (var=0.0095)

### Experiment 13
**Design**
  A=[1, 0, 1, 0, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 1, 1, 0, 1]
  A=[1, 1, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 1]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 1]
  A=[1, 1, 0, 1, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    def key(row):
        a = ''.join([str(int(round(float(v)))) for v in list(row['option_a_ratings'])])
        b = ''.join([str(int(round(float(v)))) for v in list(row['option_b_ratings'])])
        return a + '|' + b

    # cell -> {pair_key: target_response}  (target = the option predicted pro-A-side)
    cells = {
        'c4': {  # d = [+1,0,-1,-1,+1] : screen extremes vs two mid heavyweights
            '11001|01110': 0,
            '01110|11001': 1,
        },
        'c5': {  # d = [+1,+1,0,-1,0] : early cues favour, later heavyweight opposes
            '11101|00111': 0,
            '00111|11101': 1,
        },
        'c8': {  # d = [+1,+1,0,-1,+1] : three mid experts vs one oracle
            '11101|00110': 0,
            '00110|11101': 1,
        },
    }

    if len(data) == 0:
        return float('nan')

    k = data.apply(key, axis=1)
    resp = pd.to_numeric(data['response'], errors='coerce')

    rates = {}
    for cname, mapping in cells.items():
        hits = []
        for pk, tgt in mapping.items():
            sel = (k == pk)
            if sel.any():
                r = resp[sel]
                hits.append(float(np.mean((r.values == tgt).astype(float))))
        rates[cname] = float(np.mean(hits)) if len(hits) > 0 else 0.5

    # competitor (reason-counting + salience gate) : high on c4, high on c8, lowish on c5
    # advocated (read-until-sufficient, convex read-out) : low on c4, mid on c8, high on c5
    idx = rates['c4'] + rates['c8'] - rates['c5']
    return float(idx)

```

**Observed (real) value:** 0.5133 (var=0.0520)
**Predicted under pi_8:** 0.7650 (var=0.0380)
**Predicted under pi_7:** 1.1533 (var=0.0393)

### Experiment 14
**Design**
  A=[1, 0, 1, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0]  B=[0, 1, 0, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 1, 0, 0, 1]
  A=[1, 0, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0]
  A=[0, 1, 0, 1, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 1]  B=[1, 0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Difference-vector signatures for the two full-direction-disagreement cells
    # (cell 4: deadlock/extremes cell, pairs 4 & 12;  cell 8: four-against-the-oracle, pairs 8 & 16)
    # value = index of the option favoured by the reason-counting (advocated) account
    targets = {
        (1, -1, -1, 0, 0, 0, 1): 0,     # pair 4  -> A
        (-1, 1, 1, 0, 0, 0, -1): 1,     # pair 12 -> B
        (-1, -1, 1, 1, 1, 0, 1): 0,     # pair 8  -> A
        (1, 1, -1, -1, -1, 0, -1): 1,   # pair 16 -> B
    }

    hits = []
    for a_r, b_r, resp in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        try:
            a = [int(v) for v in a_r]
            b = [int(v) for v in b_r]
        except Exception:
            continue
        if len(a) != len(b):
            continue
        d = tuple(int(ai - bi) for ai, bi in zip(a, b))
        if d in targets:
            fav = targets[d]
            try:
                r = int(resp)
            except Exception:
                continue
            hits.append(1.0 if r == fav else 0.0)

    if len(hits) == 0:
        return float('nan')
    return float(np.mean(hits))
```

**Observed (real) value:** 0.6650 (var=0.0058)
**Predicted under pi_8:** 0.4883 (var=0.0049)
**Predicted under pi_7:** 0.7775 (var=0.0103)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the two head-to-head experiments the observed data land squarely on pi_7's side and decisively falsify pi_8's core commitment. Experiment 1's metric (rate of following the first/top discriminating screen row) was 0.353 observed: pi_7 predicted 0.298 (off by .055) while pi_8 predicted 0.926 (off by .573) \\u2014 an outright direction failure for the attention-lottery/top-row-capture account. Experiment 2 repeats this: observed 0.300, pi_7 0.135 (off .165), pi_8 0.766 (off .466). Again pi_8 gets the sign of the effect wrong: subjects do NOT follow the salient top row against the mid-list heavyweights and the numerical majority; they integrate the whole panel. Experiment 11 shows the same signature (obs 0.49; pi_7 0.458, pi_8 0.729). So pi_8's asymmetric salience + authority-contingent first-mismatch capture + k=2\\u20133 sampling is systematically wrong wherever the design puts the diagnostic cues out of the 'visible' rows \\u2014 exactly the cases the experiments were designed to probe. On the legacy experiments the two are near-indistinguishable (Exp 3, 5, 7, 10, 12 are ties or near-ties; pi_8 is modestly better on Exp 4, 9, 13; pi_7 better on Exp 3, 6, 7, 11, 14). Weighted overall, pi_7 is clearly the better task-invariant account.\\n\\nHowever pi_7 is not itself adequate: it is systematically OVER-extreme. It predicted 0.135 where 0.300 was observed (Exp 2), 0.778 vs 0.665 (Exp 14), 1.153 vs 0.513 (Exp 13, a gross miss), 0.851 vs 0.775 (Exp 4), and it, like pi_8, badly under-predicts Exp 8 (0.687 vs observed 0.841, where three cheap cues beat one top cue). The recurring pattern in the real data is: (i) compensatory integration over ALL cues with weakly differentiated validity weights \\u2014 numerosity beats diagnosticity more often than either model allows (Exp 8, 12, 14, and the negative strong-minus-weak contrast in Exp 10); (ii) a compressive read-out \\u2014 real choice proportions rarely exceed ~.85 or fall below ~.15 even for decisive panels; (iii) essentially no silence/mutual-endorsement confidence machinery (Exp 9's load contrast is ~0.05, a near-null); (iv) only weak, tie-breaking position effects. pi_7 has the right skeleton but its unanimity boost, sharp silence gate, deadlock gate and first-cue commitment add unneeded extremity and unneeded moving parts; pi_8's central mechanism is simply wrong. The right move is to retire pi_8 and replace it with a sharper competitor that keeps compensatory integration but fixes pi_7's over-extremity and its Exp-8/13 misses.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace pi_8 entirely. Do NOT propose another attention-lottery / top-row-capture / limited-sample-reading model: the observed top-row-following rates (.35, .30, .49 where it predicted .93, .77, .73) refute that family across three independent designs.\\n\\nProposed replacement \\u2014 'Noisy compensatory tally with compressed diagnosticity, saturating numerosity bonus, and a hard read-out ceiling':\\n1. Evidence = sum over ALL discriminating cues of sign * w_j, with w_j = ((v_j-.5)/.5)^rho and rho SMALL (~0.3\\u20130.6), i.e. validities are read only coarsely ('credible' vs 'not'), so numerosity routinely outweighs a single oracle. This is what produces Exp 8 (obs .84), Exp 12 (.85), Exp 14 (.67) and the negative strong-minus-weak contrast of Exp 10, all of which pi_7 under- or over-shoots.\\n2. A discrete, saturating 'more experts back this one' bonus at |k_A-k_B|>=2, but NO unanimity boost (the unanimity term is what makes pi_7 over-extreme on landslide cells: Exp 13 index 1.15 vs observed 0.51, Exp 2 .135 vs .30).\\n3. Compressive read-out: logistic on the drive PLUS a substantial fixed lapse/noise term (epsilon ~ .10\\u2013.20) so that P(choice) is bounded roughly in [.14, .86] regardless of how lopsided the panel is. Every observed extreme in the corpus sits at .82\\u2013.85, never above; this single constraint fixes most of pi_7's residual error.\\n4. Position enters only as a weak symmetric edge bonus that is applied uniformly (no deadlock gate, no majority-dependent gating) and is small enough that it can flip only near-tied panels \\u2014 enough for Exp 7 (.82) and Exp 12 (.85) but not enough to produce pi_8-style top-row following.\\n5. No categorical silence gate and no first-cue commitment. Replace the silence machinery with, at most, a mild graded dilution: divide the drive by (number of speaking cues)^psi with psi in [0, .5], so sparse panels are slightly less decisive but never voided. Exp 9's load contrast (obs .053, both models ~0/\\u22120.07) and Exp 1's VOID cell show the silence mechanism carries almost no behavioural weight.\\n6. Individual differences: unimodal jitter in rho, majority-bonus weight, edge-salience weight, beta and the lapse rate; no subpopulations, no learning.\\n\\nThis theory is a genuine competitor to pi_7 (same compensatory family, but flatter validity weighting, no unanimity/silence/first-cue terms, and an explicit noise ceiling) and it is designed to cure precisely the cells where pi_7 currently misses (Exp 2, 8, 13, 14). Future discriminating experiments should contrast many-weak-cues-vs-one-oracle displays (tests rho) and landslide displays (tests the read-out ceiling / unanimity)."
}
```

## Usage

```json
{
  "input_tokens": 37299,
  "output_tokens": 5321
}
```
