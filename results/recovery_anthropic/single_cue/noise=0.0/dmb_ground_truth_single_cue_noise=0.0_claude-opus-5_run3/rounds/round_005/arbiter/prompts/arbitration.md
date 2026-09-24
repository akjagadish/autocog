# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_6" and "pi_7") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_6" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_7" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_6
**Read-until-sufficient integration with mildly compressed diagnosticity, silence-calibrated evidence, and an S-SHAPED (convex) read-out of relative evidence.**

People do not run a fixed heuristic (TTB, tally, or weighted-additive). They read the expert ratings in the order the screen presents them, accumulate weighted differences, and stop as soon as their running lead is safe from any single expert they have not yet read. Six claims:

1. **Cue weights = MILDLY COMPRESSED stated diagnosticity x a modest reading-position bonus.** w_j = ((v_j-0.5)/0.5)^rho * (1 + alpha*exp(-j/tau) + gamma*exp(-(n-1-j)/tau)), with rho ~1.0-1.3, i.e. subjective weight is essentially linear-to-slightly-convex in (v-0.5). A .95 expert is only ~2x a .72 expert and ~5-8x a .56 expert -- clearly heavier, and roughly *balanced* by a pair of upper-middling experts (.78 + .72), but routinely outvotable by three mid-validity experts ('tally-like without unit-weight tallying'). This exponent sits at the crossover point where the two screen-extreme flankers of a mid-list .95 expert neither dominate it nor are dominated by it, which is what turns double-dissociation displays into genuine coin flips. Position is an attention multiplier, never the causal driver.

2. **Serial reading with a graded sufficiency stop.** After each rating that actually discriminates, the reader asks whether any single unread expert could still overturn the current lead; the bar is theta = phi * max weight among unread positions and stopping is graded, q = sigmoid(s*(|E|/theta - 1)). This is the only place where screen order has structural force, and it is display-contingent. Because the bar is set by the strongest unread expert, a merely salient first cue stops the read only when its weighted lead genuinely exceeds what any single remaining expert could reverse -- and as diagnosticity compression relaxes (higher rho) a salient-but-mediocre first cue loses that race to an unread .95 expert.

3. **Evidence is read FULLY relatively, with no load machinery.** R = sum(w*d)/sum(w*|d|) over the cues read so far; features on which both products are endorsed (or both unendorsed) are simply dropped, so any dense-vs-sparse contrast is exactly zero.

4. **Silence calibration.** Validity is treated as sensitivity: a difference only a top expert notices while the rest of the panel is silent implies a marginal quality gap and is acted on hesitantly. Gain is multiplied by c = 1 - eta*(fraction silent)^q * (weighted mean diagnosticity of the speaking experts), which bites only when silence is overwhelming, producing a *reversed* single-cue validity ladder. This term is a function of raw diagnosticity, not of the compressed weights, so the ladder is invariant to rho.

5. **Decisiveness is a CONVEX (S-shaped) function of the evidence balance, not a linear one.** A near-balanced set of reasons does not merely produce a slightly tilted preference; it produces genuine indifference. Read-out is logistic(beta * c * sign(R)*|R|^kappa) with kappa ~1.4-1.9. A small residual imbalance (|R| ~ 0.1-0.2, e.g. two screen-extreme cues almost exactly offsetting one top-validity cue) is discounted to near-nothing and the choice is a coin flip; a clear weighted majority (|R| ~ 0.5) is still followed at ~0.78-0.82; and a categorical conclusion (|R| = 1, i.e. a sufficiency stop, or a single discriminating cue) is unchanged, so committed reads stay decisive.

6. **Choice** = logistic(beta * c * sign(R)|R|^kappa) with lapse epsilon. Individual differences are unimodal jitter in rho, alpha, phi, s, kappa, beta, eta and lapse -- no strategy subpopulations, no primacy/recency reader types, no learning (there is no feedback).

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

## EXPERIMENT 1 (proposed by pi_6)

### DESIGN
**Validities (n_features=5):** [0.7, 0.6, 0.95, 0.82, 0.66]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 1, 0, 1]  B=[0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1]
  trial 3: A=[1, 1, 1, 1, 1]  B=[0, 1, 0, 1, 0]
  trial 4: A=[0, 1, 0, 1, 0]  B=[1, 1, 1, 1, 1]
  trial 5: A=[1, 1, 1, 0, 1]  B=[0, 1, 0, 0, 0]
  trial 6: A=[0, 1, 0, 0, 0]  B=[1, 1, 1, 0, 1]
  trial 7: A=[1, 1, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  trial 8: A=[0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  trial 9: A=[1, 1, 1, 0, 1]  B=[0, 0, 1, 1, 1]
  trial 10: A=[0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 1]
  trial 11: A=[1, 1, 0, 1, 1]  B=[0, 1, 1, 1, 0]
  trial 12: A=[0, 1, 1, 1, 0]  B=[1, 1, 0, 1, 1]
  trial 13: A=[1, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0]
  trial 14: A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  trial 15: A=[1, 1, 1, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 16: A=[0, 0, 1, 1, 0]  B=[1, 1, 1, 0, 1]

**Rationale:** GOAL: dissociate READ-UNTIL-SUFFICIENT INTEGRATION (advocated: serial read in screen order, compressed diagnosticity x position attention, sufficiency stop, FULLY RELATIVE evidence R, silence discount c = 1 - eta*(fraction silent)^q*vbar where EVERY non-discriminating feature counts identically, convex kappa read-out) from REASON-COUNTING WITH A TWO-REGIME SALIENCE GATE (competitor: validity-graded vote, unanimity boost, saturating majority bonus, position salience at FULL force only when the reason count is deadlocked and at a constant residual g_res~.25 otherwise, sharp silence threshold in which mutual endorsements count only at weight omega~.82, c floored at c_min).

VALIDITIES BY SCREEN POSITION: [.70, .60, .95, .82, .66]. The near-oracle expert (95%) sits mid-list at pos2, the second heavyweight (.82) sits at pos3, and the two SCREEN EXTREMES are deliberately mid-low (.70, .66) so that (i) the advocated model's attention multiplier makes the extremes roughly, but not quite, balance the mid-list heavyweight (w = [.68, .23, 1.08, .68, .30] at nominal rho=1.15, alpha=.95, tau=1.3), and (ii) the competitor's salience term e_sal = [1.02, .42, .27, .42, 1.02] loads precisely on the two weakest experts, so its deadlock gate can overturn the validity vote. n = 5 is chosen because with exactly 2 of 5 features silent the competitor's effective silence fraction is .40 for both-0 padding but .82*.40 = .33 for both-1 padding, which straddles its sharp threshold sil_f0 ~ .36 -- while the advocated model's f_sil is .40 in BOTH cases, i.e. a mathematically forced null.

8 base cells, each mirrored across sides -> 16 unique pairs, K = 6 -> 96 trials. Mirroring makes the predicted-choice split exactly 8 A / 8 B for BOTH theories, so side bias cannot mimic anything.

(1) SILENCE-COMPOSITION TRIAD, cells 1-3 (pairs 1-6). All three have the IDENTICAL difference vector d = [+1,0,+1,0,+1] (unanimous 3-0 for A; pos0 .70, pos2 .95, pos4 .66 speak, pos1 and pos3 are silent). They differ ONLY in what the two silent experts said: cell 1 both-0 (A=[1,0,1,0,1] vs B=[0,0,0,0,0]), cell 2 both-1 (A=[1,1,1,1,1] vs B=[0,1,0,1,0]), cell 3 mixed (one both-1, one both-0). The advocated model's silence term counts non-discriminating features regardless of level, and its evidence uses only discriminating cues, so it predicts EXACTLY THE SAME probability (~.93-.95, R = 1 after a certain stop at pos2) in all three cells for every parameter setting. The competitor's omega-weighted, threshold-sharp silence term gives c = .16 / .77 / .45 respectively, hence P(A) ~ .72 / .99 / .93 -- a ~27-point ordered spread driven by ratings that discriminate nothing. This within-subject triad is the flagship: one theory is pinned to a flat line, the other to a monotone ladder.

(2) SECOND SILENCE-COMPOSITION CONTRAST IN A CONFLICT DISPLAY, cells 6-7 (pairs 11-14). Both have d = [+1,0,-1,0,+1] (the two screen extremes favour A, the mid-list 95% expert favours B), with the two silent experts both-1 in cell 6 and both-0 in cell 7. Advocated: identical in both (~.45; it stops at pos2 with R ~ -.23, which the convex read-out crushes toward chance -- a mild preference for B). Competitor: P(A) = .65 (both-1) vs .53 (both-0), and it chooses the OPPOSITE option from the advocated model in the both-1 version because mu*g_res*S = +.80 overturns the validity vote V = -.15. Direction disagreement plus a second forced-null test.

(3) DEADLOCK-GATE CELL, cell 4 (pairs 7-8) -- the sharpest cell. d = [+1,0,-1,-1,+1]: the two screen extremes (.70, .66) favour A, the two mid-list heavyweights (.95, .82) favour B, one silent expert. Now kA = kB = 2, so the competitor's gate opens to FULL force (g = 1): drive = V + mu*S = -.80 + 2.45 = +1.65 and P(A) ~ .88. The advocated model reads past pos0, stops at pos3 with R ~ -.44 and predicts P(A) ~ .28-.35 across its whole (rho, alpha, tau, kappa, beta) box (checked at both corners: rho=1.32/alpha=0.7 -> .28; rho=1.0/alpha=1.2/tau=1.6 -> .35). Opposite direction, ~55 points, over both full parameter boxes.

CRUCIALLY, cell 4 and cell 6 differ by a SINGLE feature (whether the .82 expert at pos3 also opposes A). The competitor must become MORE pro-A when that extra expert is added against A (.65 -> .88, because the added reason deadlocks the count and unleashes the salience gate); the advocated model must move the other way (.45 -> .32). A non-monotonicity of exactly this sign is a competitor-only signature; monotone decline is advocated-only.

(4) SUFFICIENCY-STOP CELL, cell 5 (pairs 9-10). d = [+1,+1,0,-1,0]: the two leading cues (.70, .60) favour A, the later .82 expert favours B, two silent both-1. The advocated model's running lead after pos0+pos1 (E = .91) clears the bar set by the strongest unread expert (phi*w3 ~ .71), so it stops and commits categorically on R = +1: P(A) ~ .85-.95 across its box. The competitor never stops here (its first-cue commitment is authority-capped and pos0 is outranked by pos2/pos3), integrates everything to a near-null vote V = -.02 rescued only by the residual salience term, giving P(A) ~ .61. A 25-35 point confidence gap in the same direction, driven purely by stopping vs full integration.

(5) THREE-MID-EXPERTS-vs-ONE-ORACLE CELL, cell 8 (pairs 15-16). d = [+1,+1,0,-1,+1]: pos0 (.70), pos1 (.60) and pos4 (.66) favour A while the mid-list 95% expert favours B; only one expert is silent, so silence machinery is essentially OFF in both models and cannot be blamed for anything. The competitor fires its saturating majority bonus (|dk| = 2 -> M = .67) plus residual salience and predicts a decisive P(A) ~ .89. The advocated model's compressed weights make this an almost exact standoff (sum of the three mid weights 1.21 vs 1.08 for the oracle), the final R ~ +.06 is crushed by the convex read-out, and P(A) ~ .47-.68 across its box -- near indifference where the competitor is near ceiling.

DIAGNOSTIC YIELD: (i) two forced-null contrasts (cells 1/2/3 and cells 6/7) where the advocated theory is mathematically committed to zero difference and the competitor predicts 12-27 point ordered shifts produced by non-discriminating ratings; (ii) one full direction disagreement (cell 4, ~55 points) plus a sign-reversed single-feature manipulation (cell 4 vs cell 6); (iii) two magnitude dissociations (cells 5 and 8, 25-35 points, opposite in sign to each other) that separate sufficiency-stopping + convex read-out from reason-counting with majority/unanimity bonuses. Every pair has at least one discriminating cue; no cell is degenerate.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
The design contains three cells whose predicted choice proportions move in opposite directions under the two theories. Cell 4 (pairs 7/8, d=[+1,0,-1,-1,+1]): the competitor's deadlock salience gate opens at full force and drives P(pro-extreme)~.88, while the read-until-sufficient model stops on the mid-list .95 expert and gives ~.31. Cell 8 (pairs 15/16): the competitor fires its saturating majority bonus (~.89) while the convex read-out crushes a near-balanced R to near indifference (~.47-.68). Cell 5 (pairs 9/10): the sufficiency stop makes the advocated model commit categorically (~.90) whereas the competitor integrates to a near-null vote (~.61). Summing the two cells where the competitor is higher and subtracting the cell where the advocated model is higher yields an index of roughly -0.0 for the advocated theory versus ~+1.16 for the competitor - a gap of over one unit. Each component rate is estimated from 12 trials per subject and all three components are strongly parameter-insensitive within each theory's stated parameter box, so per-subject variance stays small (dominated by binomial noise, ~0.02-0.03 per component), giving a very large Welch t at N=25. Crucially the index is not trivially true for either theory: it is a signed combination of directional choice proportions that either theory could in principle produce at any value.

Source:
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


### RESULTS
- Predicted under pi_6 (simulated): 0.4233 (var=0.0611)
- Predicted under pi_7 (simulated): 1.1533 (var=0.0393)
- Observed on real data: 0.5133 (var=0.0520)

## EXPERIMENT 2 (proposed by pi_7)

### DESIGN
**Validities (n_features=7):** [0.7, 0.95, 0.86, 0.7, 0.6, 0.78, 0.7]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 1, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[1, 1, 1, 1, 1, 1, 0]  B=[0, 1, 0, 1, 0, 0, 0]
  trial 3: A=[1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 1, 0, 0, 1]
  trial 4: A=[1, 0, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 0]
  trial 5: A=[1, 0, 1, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 1]
  trial 6: A=[1, 0, 1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0, 0, 0]
  trial 7: A=[1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1]
  trial 8: A=[0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 1, 0]
  trial 9: A=[0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 1, 0]
  trial 10: A=[0, 1, 0, 1, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0]
  trial 11: A=[0, 1, 0, 1, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 1]
  trial 12: A=[0, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 1, 1, 1, 1]
  trial 13: A=[0, 1, 0, 0, 0, 0, 1]  B=[1, 0, 1, 1, 0, 0, 0]
  trial 14: A=[0, 1, 0, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0, 1]
  trial 15: A=[0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0]
  trial 16: A=[1, 1, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 1]

**Rationale:** GOAL: dissociate the advocated REASON-COUNTING WITH A TWO-REGIME SALIENCE GATE (validity-graded vote, symmetric first/last-row salience at FULL force only when the reason count is deadlocked and at a CONSTANT residual g_res~.25 otherwise, saturating majority bonus, unanimity bonus, sharp silence threshold in which mutual endorsements count only at weight omega~.82, c floored at c_min) from READ-UNTIL-SUFFICIENT INTEGRATION (competitor: compressed diagnosticity x a PRIMACY-dominated attention multiplier, serial read with a sufficiency stop, FULLY RELATIVE evidence R, convex kappa read-out, and a silence discount in which EVERY non-discriminating feature counts identically).

VALIDITIES BY SCREEN POSITION: [.70, .95, .86, .70, .60, .78, .70]. Seven experts is chosen deliberately: with exactly 3 of 7 features silent, the advocated model's effective silence fraction is 3/7 = .429 when the silent experts endorsed NEITHER product but .82*3/7 = .351 when they endorsed BOTH -- values that STRADDLE its sharp threshold sil_f0 ~ .36 (sigmoid weights .93 vs .42 at k=38), while the competitor's f_sil is 3/7 in BOTH cases, i.e. a mathematically forced null. Three experts share validity .70 and sit at pos0, pos3 and pos6 (both screen ends plus the exact middle), which permits validity-matched swaps of screen position. The oracle (.95) is deliberately placed at pos1 rather than pos0, and in EVERY cell the first discriminating cue is the .70 expert at pos0, so the advocated model's authority-capped first-cue commitment is uniformly suppressed (ratio .40/(1.05*.90) = .42 -> q_com ~ .01) and cannot contaminate any contrast. Advocated salience e_sal (tau=1) = [1.00,.37,.15,.10,.15,.37,1.00] (symmetric U); competitor attention (alpha~.9,tau~1.3,gamma<=.15) = [1.90,1.42,1.20,1.09,1.05,1.04,1.06] (primacy only, essentially flat over pos2-pos6), giving competitor weights w = [.66,1.26,.82,.38,.17,.54,.37] -- dominated by the mid-list .95 expert.

8 base cells, each mirrored across sides -> 16 unique pairs, K = 6 -> 96 trials. Mirroring makes the predicted-choice split exactly 8 A / 8 B for BOTH theories, so side bias cannot mimic anything.

(1) SILENCE-COMPOSITION TRIAD, cells 1-3 (pairs 1-3 + mirrors 9-11). FLAGSHIP. All three carry the IDENTICAL difference vector d = [+1,0,+1,0,+1,+1,0] (unanimous 4-0 for A; .70,.86,.60,.78 speak; pos1, pos3, pos6 silent). They differ ONLY in what the three silent experts said: cell 1 all both-0, cell 2 two both-1 + one both-0, cell 3 all both-1. The competitor's silence term counts non-discriminating features regardless of level and its evidence uses only discriminating cues, so it is FORCED to predict exactly the same probability in all three (stop at pos2 with R = 1, P(A) ~ .94) for every parameter setting. The advocated model's omega-weighted, threshold-sharp term gives c = .21 / .45 / .65 and hence P(A) ~ .78 / .93 / .98 -- a ~20-point MONOTONE ORDERED ladder produced by ratings that discriminate nothing at all.

(2) DEADLOCK-GATE CELL, cell 4 (pairs 4, 12). d = [+1,-1,-1,0,0,0,+1]: the two screen EXTREMES (.70, .70) favour A, the two mid-list heavyweights (.95, .86) favour B, three silent both-1. Because kA = kB = 2 the advocated gate opens to FULL force (g = 1): drive = V + mu*S = -.80 + 2.66 = +1.86, and with c ~ .56 it predicts P(A) ~ .73-.85 across its whole box. The competitor reads on, stops at pos2 with R ~ -.52 and predicts P(A) ~ .29. Opposite direction, ~50 points.

(3) VALIDITY-MATCHED SALIENCE SWAP AT |dk| = 1, cells 5 & 6 (pairs 5-6, 13-14). The residual-gate signature. Cell 5: A supported by pos0(.70), pos2(.86), pos3(.70), opposed by pos1(.95) and pos6(.70). Cell 6: the two .70 experts trade places (A supported by pos0, pos2, pos6; opposed by pos1, pos3). The validity multiset on each side is IDENTICAL, the silence composition is identical (two both-0), so the vote term V = +.25 is identical and any validity-based model must be flat. The advocated model's constant residual salience (g_res ~ .25, mu ~ 1.8) still reads the screen ends: S goes from -.12 to +1.68, so P(A) goes from ~.55 to ~.76 (delta logit ~ .8-1.0 across mu, g_res, tau). The competitor's position attention is primacy-shaped and nearly identical at pos3 (w = .382) and pos6 (w = .370), and its relative evidence gives P(A) ~ .54 vs ~ .53 -- a ~1-point null, if anything in the opposite direction. A 20-point swap effect vs a flat line isolates SYMMETRIC (first-and-last) salience surviving as a constant bonus outside the deadlock regime.

(4) SUFFICIENCY-STOP vs FULL INTEGRATION, cell 7 (pairs 7, 15). d = [+1,+1,0,0,0,-1,-1]: pos0(.70) and pos1(.95) favour A, pos5(.78) and pos6(.70) favour B, three silent both-1. The competitor's running lead after pos1 (|E| = 1.92) clears the bar set by the strongest unread expert (1.05*.535 = .56), so it stops and commits categorically on R = +1: P(A) ~ .94. The advocated model is deadlocked (kA = kB = 2) so its gate opens, but the salience term cancels EXACTLY (e_sal at pos0+pos1 equals e_sal at pos5+pos6), leaving only the small vote V = +.32 discounted by c ~ .59: P(A) ~ .56. A ~38-point confidence gap in the same direction, driven purely by stopping-plus-normalisation vs. full graded integration -- and a cell where the advocated model's salience machinery is provably inert, so the gap cannot be attributed to position.

(5) FOUR-AGAINST-THE-ORACLE CELL, cell 8 (pairs 8, 16). d = [-1,-1,+1,+1,+1,0,+1]: the leading .70 expert AND the oracle .95 expert favour B, while four later experts (.86,.70,.60,.70) favour A; only ONE expert is silent, so both models' silence machinery is essentially off and cannot be blamed. The competitor accumulates -1.92 over the first two positions, clears the .86 bar and commits on R = -1: P(A) ~ .05. The advocated model's near-linear vote (V = +.40 to +.53 across rho) plus the saturating majority bonus (|dk| = 2 -> M ~ .67) gives drive ~ 1.0-1.2 and P(A) ~ .75-.82. Opposite direction, ~70 points, over both full parameter boxes.

DIAGNOSTIC YIELD: (i) a three-cell forced-null triad where the competitor is pinned to one number and the advocated theory predicts a 20-point ordered ladder produced by non-discriminating ratings; (ii) a validity-matched position swap (cells 5 vs 6) that isolates symmetric residual salience against a primacy-only, fully normalising alternative, where the competitor is again forced to a null; (iii) two full direction disagreements (cells 4 and 8, 50 and 70 points) covering both the deadlock regime and the majority-bonus regime; (iv) one same-direction magnitude dissociation (cell 7, 38 points) in which the advocated model's salience term cancels by construction. Every pair has at least one discriminating cue; no cell is degenerate.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
The design contains two cells where the two accounts predict OPPOSITE choices, and each cell is presented twice (base + side-mirror) x 6 reps = 24 trials per subject, so a simple proportion has small binomial noise.

Cell 4 (d = [+1,-1,-1,0,0,0,+1]; the two .70 screen-extreme experts vs. the mid-list .95 and .86 heavyweights, three both-1 silent): the advocated reason-counting model is at an exact vote deadlock (kA = kB = 2), which opens its salience gate to full force, so the first/last screen rows decide and it picks the extremes-backed option with P ~ .73-.85. The read-until-sufficient competitor weights by compressed validity x a primacy-only attention profile, reads on past the deadlock and lands on a clearly negative relative evidence R ~ -.5, giving P ~ .29 for that same option.

Cell 8 (d = [-1,-1,+1,+1,+1,0,+1]; leading .70 expert and the .95 oracle favour one side, four later experts the other, only one silent feature so neither model's silence machinery is engaged): the advocated model's near-linear vote plus saturating majority bonus favours the four-expert side at P ~ .75-.82; the competitor accumulates a large early lead from the first two (heavily attention-weighted) positions, clears its sufficiency bar, commits on R = -1 and gives P ~ .05 for the four-expert side.

The metric is the proportion of choices, over those 24 trials, that agree with the option the advocated theory favours. Predicted values: ~0.78 under reason-counting-with-salience-gate vs. ~0.17 under read-until-sufficient integration -- a ~0.6 gap. Because the cells are mirrored across sides, side bias cannot inflate or deflate the score, and because both cells sit far from indifference in both models the per-subject variance is dominated by binomial noise over 24 trials (~0.007) plus modest parameter jitter, which should keep Welch's t large at N = 25.

Source:
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

### RESULTS
- Predicted under pi_6 (simulated): 0.1783 (var=0.0069)
- Predicted under pi_7 (simulated): 0.7775 (var=0.0103)
- Observed on real data: 0.6650 (var=0.0058)

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
**Predicted under pi_6:** 0.0110 (var=0.0029)
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
**Predicted under pi_6:** 0.7530 (var=0.0025)
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
**Predicted under pi_6:** 0.4450 (var=0.0192)
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
**Predicted under pi_6:** 0.3663 (var=0.0121)
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
**Predicted under pi_6:** 0.7913 (var=0.0231)
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
**Predicted under pi_6:** 0.7238 (var=0.0070)
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
**Predicted under pi_6:** -0.0050 (var=0.0446)
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
**Predicted under pi_6:** -0.1838 (var=0.0208)
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
**Predicted under pi_6:** 0.4325 (var=0.0102)
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
**Predicted under pi_6:** 0.2300 (var=0.0174)
**Predicted under pi_7:** 0.8817 (var=0.0095)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all twelve experiments pi_7 (reason-counting + two-regime salience gate) is the clearly more task-invariant account, while pi_6 (read-until-sufficient with convex read-out) fails catastrophically on the experiments that carry the most diagnostic weight.\n\nWhere pi_6 fails badly: Exp12 (extremes-vs-two-mid-heavyweights: observed .85, pi_6 .23, pi_7 .88 \" a 62-point miss), Exp2 (observed .665, pi_6 .178, pi_7 .778), Exp3 (observed +.093, pi_6 +.011, pi_7 +.092), Exp6 (.508 vs pi_6 .366). pi_6's serial-stop-on-the-oracle + fully-relative evidence machinery systematically predicts that a mid-list high-validity expert wins; the data repeatedly say it does not when the display extremes are lined up against it under a deadlocked reason count.\n\nStrikingly, the single most counter-intuitive pattern in the whole corpus is exactly pi_7's signature: at a 2-vs-1 reason count with the extremes on the majority side (Exp11 cell5/cell8 pooled = .49) people are at chance, whereas at an exact 2-vs-2 deadlock with the extremes on one side and the .95/.86 heavyweights on the other (Exp12 = .85) they overwhelmingly follow the screen extremes. pi_7 predicted .458 and .882 respectively. That non-monotonicity in majority size is a strong confirmation of the deadlock-gate idea and a direct falsification of pi_6's monotone weighted integration.\n\nWhere pi_6 wins it is mostly by accident or by being closer to the middle: Exp1 (observed .513; pi_6 .423, pi_7 1.153) and Exp4 (.775 vs .851), Exp8 (.841 vs pi_6 .724, pi_7 .687), Exp9 (sign). The Exp1 miss is informative about pi_7's weak points rather than about pi_6's strengths: pi_7's unanimity + saturating-majority bonuses push it to near-ceiling (.89 on the three-mid-vs-oracle cell, .99 on the all-both-1 silence cell) while the real data cap out around .80-.85 in almost every cell (Exp7 .82, Exp8 .84, Exp12 .85, Exp4 .775, Exp2 .665). Simultaneously pi_7 *under*-predicts Exp8 (.687 vs .841), where the extremes plus a majority oppose the mid-list oracle. So pi_7 is directionally right almost everywhere but mis-calibrated in the gain/ceiling regime: it is too categorical when bonuses stack and not decisive enough when extremes plus numerosity agree.\n\nBoth theories are also jointly weak on the pure-position/load nulls (Exp5 .5175, Exp6 .5075, Exp9 +.053), where both sit 5-14 points low. Given this, pi_6 is the degenerate member of the pair and should be replaced by a genuinely different competitor that can (a) reproduce the deadlock/extremes reversal, (b) reproduce the reversed single-cue validity ladder (Exp10 = -.20, both models already get this), and (c) deliver the ~.85 soft ceiling and the compressive gain that pi_7 lacks.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace pi_6 entirely with an ATTENTION-LOTTERY / LIMITED-SAMPLE COMPARISON account that is a real rival to pi_7 rather than another serial-stop integrator.\n\nCore claims to propose:\n1. On each trial the subject does NOT integrate all rows. They draw a small, noisy sample of rows to actually compare (effective sample size k ~ 2-3 of n), with sampling probability p_j proportional to a salience term (U-shaped over the literal screen: exp(-j/tau)+exp(-(n-1-j)/tau)) TIMES a mildly compressed diagnosticity term ((v_j-.5)/.5)^rho with rho ~ 0.6-1.0. Position is therefore always active, not gated \" but its behavioural *consequence* is largest exactly when the validity-weighted evidence is near-balanced, because that is when the identity of the sampled rows flips the verdict. This reproduces Exp12's .85 (2-vs-2 deadlock resolved by the salient extremes) and Exp11's ~.49 (2-vs-1 with a .95 opponent: the oracle is very likely to be sampled and cancels the extremes) WITHOUT any discrete deadlock gate, and it predicts the Exp11/Exp12 non-monotonicity as a smooth consequence of evidence balance rather than of |k_A-k_B| exactly equalling zero.\n2. Choice = sign of the sampled weighted difference, with ties broken by the most salient sampled row. Because the sample is small and random, the mapping from true evidence to P(choice) is COMPRESSIVE with a ceiling near .82-.88 even for landslides: this is the key calibration difference from pi_7, which over-predicts Exp1 (1.153 vs .513), Exp4 (.851 vs .775) and Exp2 (.778 vs .665) whenever its unanimity and majority bonuses stack. No unanimity bonus, no additive majority bonus \" numerosity helps only because more rows on one side means the sample is more likely to be dominated by that side (which already yields Exp4 ~.77 and Exp3 ~+.09).\n3. Silence-as-sensitivity is retained but as a *sampling* effect plus a gain discount: rows on which both products agree are still sampled (they compete for attention), so when most of the panel is silent the sample often contains no discriminating row and the subject guesses. This reproduces the reversed single-cue validity ladder (Exp10 = -.20: a lone .95 speaker amid silence is discounted more than a lone .60 speaker) and predicts a near-zero density/load effect (Exp9 ~ 0 to +.05), and it should be made indifferent to whether the silent rows are both-1 or both-0 only up to a small weight omega, so the theory is not forced to an exact null on the silence-composition triads.\n4. Individual differences are unimodal jitter in k (sample size), tau (salience width), rho, and a lapse rate; no strategy subpopulations, no learning.\n\nCalibration targets the new model must hit simultaneously: Exp1 ~0.51, Exp2 ~0.67, Exp3 ~+0.09, Exp4 ~0.78, Exp5 ~0.52, Exp6 ~0.51, Exp7 ~0.82, Exp8 ~0.84, Exp9 ~+0.05, Exp10 ~-0.20, Exp11 ~0.49, Exp12 ~0.85. In particular it must be MORE decisive than pi_7 on Exp8 (extremes + numerosity vs the mid-list oracle) and LESS decisive than pi_7 on Exp1/Exp4 (stacked unanimity/majority cells) \" that pair of constraints is what the compressive small-sample read-out buys and is the sharpest empirical wedge against pi_7."
}
```

## Usage

```json
{
  "input_tokens": 34045,
  "output_tokens": 7219
}
```
