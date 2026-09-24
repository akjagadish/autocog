# experiment_attempt_00

## System Prompt

You are a renowned cognitive scientist designing an experiment in the Decision Making (Binary Features) domain.

Your goal is to be an adversarial collaborator: propose a design whose outcomes would be predicted by your advocated theory but NOT by the competing theory. Both are provided below.

A useful proposal targets a *quantitative* dissociation between the two theories — how they respond differently to specific stimuli in addition to differences in overall performance.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

A multi-attribute decision-making experiment. On each trial the subject sees two options (A, B), each described by `n_features` integer expert ratings (`n_features` is set by the length of `validities` you propose). Choose `validities` — one per feature, each in [0.5, 1.0], order free — to fix each expert's advertised accuracy; subjects are told these values up front. Then choose `trial_a_ratings/trial_b_ratings` (each rating value in [0, 1]) so that the intended decision-making theories make distinguishable predictions: avoid degenerate pairs where every theory agrees, and prefer pairs that maximally dissociate the candidate theories. Validities and the trial ratings together define the design; they are fixed across all trials in this experiment. No trial-by-trial correctness feedback. The total number of trials per subject is held at roughly 96: each unique pair is repeated K = max(1, 96 // n_unique_pairs) times in an independently-randomized order per subject.

Subjects see the following instructions:
In this experiment you will repeatedly choose between two fictitious products, A and B. On every trial you will see `n_features` expert ratings for each product (the number of experts is fixed across all trials and is set by the length of `validities`).

Each rating is an integer in [0, 1]. The ratings are displayed as a horizontal filled bar with the numeric value (e.g. "0/1") shown next to it. Higher = more positive.

The same experts (in the same order) provide ratings for both products on every trial. Each expert's accuracy (their validity expressed as a percentage, e.g. "Expert 1 (80%)") is shown next to their rating on every trial AND is also listed up front in an "Expert accuracies" panel.

On each trial, decide which product is of higher quality and press A for product A or B for product B. There is no time limit and no feedback. Note that for the first ~`min_rt_ms` of each trial the answer prompt is hidden and the keys are locked, so subjects first see the full ratings and can answer once the A / B prompt appears — design pairs that actually require comparing the ratings.

Total trials per subject is roughly `MAX_TRIALS`: each unique pair you propose is repeated `K = max(1, MAX_TRIALS // n_unique_pairs)` times in an independently-randomized order per subject.

## ADVOCATED THEORY
**Description:** **Reason-counting with a TWO-REGIME display-salience gate (full force only on a deadlocked vote, a constant residual otherwise), near-linear diagnosticity weighting, a saturating majority bonus, unanimity boost, sharp silence-to-confidence collapse (mutual endorsement ≈ silence), and a non-zero floor on read-out gain.**

This keeps every mechanism of the accepted base and changes exactly three calibration commitments (one of them structural-but-local: the shape of the salience gate).

1. **Direction is a vote over discriminating experts, graded near-linearly by stated validity.** Each discriminating expert contributes x_j = ((v_j-0.5)/0.5)^rho with rho ~0.88-1.02. At this exponent a single .95 expert is worth about two upper-middling experts (.72 + .70), so numerosity and diagnosticity are *balanced* rather than one dominating: three mid experts still outvote one top expert, but two mid experts only tie it. Screen position plays no role in this term.

2. **Display salience is a TIE-BREAKER WITH A RESIDUAL, not a quantity that decays with the size of the majority.** s_j = exp(-j/tau)+exp(-(n-1-j)/tau) over the literal display row, multiplied by a gate g = max(exp(-(k_A-k_B)^2/2 sigma^2), g_res) with sigma ~0.60 and g_res ~0.25. The gate has two regimes: when the reason count is exactly deadlocked the first/last rows of the panel decide the choice outright (g = 1); as soon as one side has more reasons, position stops being decisive but does NOT disappear — it persists as a constant, modest attention bonus (g = g_res ≈ 0.25) that is the SAME for a one-reason edge and for a five-against-one landslide. The previous Gaussian-only gate made position vanish geometrically with the majority size, which is unmotivated: there is no reason why a larger majority should erase the extra attention paid to the top and bottom of the screen. The empirical signature is that a three-against-one display whose majority occupies both screen ends is markedly more decisive (~.88) than a three-against-one display whose lone dissenter sits at a screen end (~.82), even when the validities are matched.

3. **Clear majorities are categorical and saturating:** M = lambda*sign(k_A-k_B)*min(max(|k_A-k_B|-1,0),1) with lambda ~0.67. Once one side has at least two more reasons, 'more experts back this one' is itself a discrete reason, and it does not grow further with the size of the majority.

4. **Unanimity among the speaking experts is categorical**: if every expert who noticed a difference points the same way, a constant bonus is added.

5. **Gap inference from the panel is a sharp threshold on how much of the panel was informative, and 'both products are good' is nearly as uninformative as 'neither is'.** Validity is read as sensitivity: a difference only sensitive experts notice implies a marginal true gap. Mutual endorsements count at weight omega ~0.8. c = 1 - eta*sigmoid(k*(f_sil - f0))*vbar.

6. **The collapse of confidence is bounded below:** c is floored at c_min ~0.10, so a panel judged uninformative leaves a small residual tilt rather than a literal coin flip.

7. **A small, display-contingent first-cue commitment survives**: people commit to the first expert who actually discriminates only if no expert further down the list outranks him in stated validity. Individual differences are unimodal jitter in rho, mu, sigma, g_res, lambda, eta, omega, c_min, beta; no strategy subpopulations, no serial stop rule, no learning.

**Parameters:**
- rho: [0.88, 1.02]
- mu: [1.70, 1.95]
- lam: [0.60, 0.74]
- sigma: [0.55, 0.66]
- g_res: [0.22, 0.28]
- tau: [0.85, 1.25]
- ups: [1.25, 1.80]
- beta: [1.05, 1.40]
- eta: [1.45, 1.65]
- sil_k: [30.0, 45.0]
- sil_f0: [0.345, 0.375]
- omega: [0.72, 0.92]
- c_min: [0.07, 0.13]
- phi: [0.95, 1.12]
- s_stop: [6.0, 10.0]
- d_com: [2.20, 2.90]
- epsilon: [0.01, 0.06]
- validities: validities

**`predict source code`:**
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

**`policy source code`:**
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


## COMPETING THEORY
**Description:** **Dilution-calibrated reason integration with an ORACLE PREMIUM and a SHARPLY-PEAKED edge-salience profile (DCRI-3s): a compressive weighted reason tally whose diagnosticity map is compressive in the mid range but regains a categorical top-end premium near the stated-validity ceiling, read out RELATIVE to the evidence actually on the table, scaled by a smooth, strictly-positive confidence that falls with panel silence and with the coarseness-inference it licenses, plus a two-regime edge-salience tie-breaker whose attention profile is concentrated on the LITERAL first and last rows (steep decay), a slowly-growing numerosity bonus, a difficulty-scaled lapse, and a firm authority-checked top-down stop.**

Mechanism family is unchanged from the accepted base; ONE calibration commitment is sharpened.

1. **Diagnosticity is compressive in the middle but has a TOP-END PREMIUM.** x_j = ((v_j-0.5)/0.5)^rho * (1 + alpha_v*sigmoid(kappa_v*(v_j - v_knee))) with rho ~0.46-0.58, v_knee ~0.91: below ~.88 subjective weight is strongly sub-linear so cheap-cue coalitions routinely outvote a merely-good cue, but a near-certain expert is treated as categorically heavier.

2. **Numerosity is a separate, slowly-growing reason.** N = lambda*sign(k_A-k_B)*sqrt(|k_A-k_B|).

3. **Read-out is RELATIVE:** drive = (V + N + P)/(kappa2 + total speaking diagnosticity).

4. **Confidence = evidence fraction x coarseness inference, never a reversal.** c = c_min + (1-c_min)*frac^psi*max(1 - eta*f_sil^q*vbar, 0), strictly positive and floored, so a lone speaking cue amid a mute panel is still followed above chance and silence only flattens toward chance.

5. **Mutual endorsement is a weaker form of silence than mutual absence** (weight omega ~0.63), giving a small positive processing-load effect.

6. **Edge salience is a two-regime tie-breaker over LITERAL screen rows with a STEEP attention decay.** s_j = exp(-j/tau)+exp(-(n-1-j)/tau) with tau ~0.62-0.88 (previously ~1.05). This is the sharpened commitment: the extra attention paid to the top and bottom of the display is *confined to the first and last rows themselves* and does not leak onto rows 1 and n-2. With a broad profile the mid-list heavyweights that sit just inside the panel edges receive almost as much positional boost as the true screen extremes, so the tie-breaker partially cancels itself in exactly the deadlocked extremes-vs-heavyweights displays it is meant to decide; with a steep profile the tie-breaker actually breaks the tie. Full force only on exactly deadlocked reason counts (narrow sigma), a small constant residual otherwise, symmetric so there is no primacy/recency swing.

7. **Choice difficulty scales the lapse**, so near-deadlocked displays draw genuine guessing while landslides stay near ceiling.

8. **Authority-checked top-down stop**: people commit at the first discriminating row only if nothing below it is more diagnostic; the committed choice is still multiplied by the same confidence c.

Individual differences are unimodal jitter in all parameters; no strategy subpopulations, no learning.

**Parameters:**
- rho: [0.46, 0.58]
- alpha_v: [0.45, 0.85]
- v_knee: [0.895, 0.925]
- kappa_v: [25.0, 45.0]
- lam: [0.48, 0.64]
- mu: [2.15, 2.65]
- tau: [0.62, 0.88]
- sigma: [0.36, 0.46]
- g_res: [0.06, 0.14]
- kappa: [0.45, 0.80]
- kap2: [0.30, 0.55]
- psi: [0.28, 0.45]
- eta: [3.05, 3.65]
- q_sil: [1.05, 1.35]
- omega: [0.55, 0.72]
- c_min: [0.06, 0.11]
- beta: [2.80, 3.50]
- eps0: [0.05, 0.11]
- eps1: [0.16, 0.28]
- q_max: [0.92, 0.99]
- d_com: [2.00, 2.55]
- phi: [0.98, 1.06]
- s_stop: [6.0, 12.0]
- validities: validities

**`predict source code`:**
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

    rho = _p('rho', 0.52, 0.05, 4.0)
    alpha_v = _p('alpha_v', 0.62, 0.0, 3.0)
    v_knee = _p('v_knee', 0.910, 0.55, 0.999)
    kappa_v = _p('kappa_v', 35.0, 1.0, 200.0)
    lam = _p('lam', 0.56, 0.0, 3.0)
    mu = _p('mu', 2.40, 0.0, 8.0)
    tau = _p('tau', 0.75, 0.2, 8.0)
    sigma = _p('sigma', 0.41, 0.15, 3.0)
    g_res = _p('g_res', 0.10, 0.0, 1.0)
    kappa = _p('kappa', 0.60, 0.05, 5.0)
    kap2 = _p('kap2', 0.42, 0.02, 5.0)
    psi = _p('psi', 0.35, 0.01, 3.0)
    eta = _p('eta', 3.35, 0.0, 10.0)
    q_sil = _p('q_sil', 1.20, 0.3, 8.0)
    omega = _p('omega', 0.63, 0.0, 1.0)
    c_min = _p('c_min', 0.085, 0.0, 0.5)
    beta = _p('beta', 3.10, 0.05, 40.0)
    eps0 = _p('eps0', 0.08, 0.0, 0.5)
    eps1 = _p('eps1', 0.22, 0.0, 0.6)
    q_max = _p('q_max', 0.96, 0.0, 1.0)
    d_com = _p('d_com', 2.25, 0.0, 6.0)
    phi = _p('phi', 1.02, 0.5, 3.0)
    s_stop = _p('s_stop', 9.0, 0.5, 60.0)

    def _sig(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))

    # ------------------------------------------------------------------
    # 4. compressive reason tally WITH A TOP-END (ORACLE) PREMIUM
    #    mid-range validity is read coarsely (rho < 1), but a near-certain
    #    expert regains a categorical extra weight.
    # ------------------------------------------------------------------
    prem = 1.0 + alpha_v * _sig(kappa_v * (val - v_knee))
    x = np.power(vs, rho) * prem
    idxs = np.flatnonzero(disc)
    sgn = np.sign(d[idxs])
    xs = x[idxs]
    V = float(np.sum(sgn * xs))
    Xs = float(np.sum(xs))
    if Xs <= 1e-12:
        return np.array([0.5, 0.5])

    kA = int(np.count_nonzero(sgn > 0))
    kB = int(np.count_nonzero(sgn < 0))
    dk = float(kA - kB)

    # ------------------------------------------------------------------
    # 5. numerosity: a separate reason that grows only as sqrt(|dk|)
    # ------------------------------------------------------------------
    N = lam * (1.0 if dk > 0 else (-1.0 if dk < 0 else 0.0)) * float(np.sqrt(abs(dk)))

    # ------------------------------------------------------------------
    # 6. symmetric edge salience, full force only on deadlocked counts.
    #    STEEP decay (tau < 1): the attention bonus is confined to the
    #    literal first and last rows and does not leak onto rows 1/n-2.
    # ------------------------------------------------------------------
    pos = np.arange(n, dtype=float)
    s_sal = np.exp(-pos / tau) + np.exp(-(n - 1.0 - pos) / tau)
    S = float(np.sum(sgn * s_sal[idxs]))
    g = g_res + (1.0 - g_res) * float(np.exp(-np.clip((dk * dk) / (2.0 * sigma * sigma), 0.0, 60.0)))
    P = mu * g * S

    # ------------------------------------------------------------------
    # 7. RELATIVE read-out: evidence judged against evidence available
    # ------------------------------------------------------------------
    drive = (V + N + P) / (kap2 + Xs)

    # ------------------------------------------------------------------
    # 8. confidence = evidence-fraction dilution x coarseness inference
    #    (smooth, strictly positive, never sign-reversing)
    # ------------------------------------------------------------------
    both1 = (~disc) & (a > 0.5) & (b > 0.5)
    sil_w = np.where(disc, 0.0, np.where(both1, omega, 1.0))
    sil_mass = float(np.sum(sil_w * x))
    f_sil = float(np.sum(sil_w)) / float(n)
    f_sil = float(np.clip(f_sil, 0.0, 1.0))

    frac = Xs / (kappa + Xs + sil_mass)
    frac = float(np.clip(frac, 1e-9, 1.0))
    vbar = float(np.sum(xs * vs[idxs]) / Xs)
    gap = 1.0 - eta * (f_sil ** q_sil) * vbar
    gap = float(max(gap, 0.0))
    c = c_min + (1.0 - c_min) * (frac ** psi) * gap
    c = float(np.clip(c, 0.0, 1.0))

    p_int = float(_sig(beta * c * drive))

    # ------------------------------------------------------------------
    # 9. authority-checked top-down stop (display-contingent, capped)
    # ------------------------------------------------------------------
    j1 = int(idxs[0])
    if j1 >= n - 1:
        q_com = q_max
    else:
        v_rest = float(np.max(vs[j1 + 1:]))
        v_rest = max(v_rest, 1e-6)
        ratio = float(vs[j1]) / (phi * v_rest)
        q_com = q_max * float(_sig(s_stop * (ratio - 1.0)))
    q_com = float(np.clip(q_com, 0.0, 1.0))

    p_dec = float(_sig(beta * c * d_com))
    p_com_a = p_dec if d[j1] > 0 else (1.0 - p_dec)

    p_a = q_com * p_com_a + (1.0 - q_com) * p_int
    p_a = float(np.clip(p_a, 1e-9, 1.0 - 1e-9))

    # ------------------------------------------------------------------
    # 10. choice-difficulty-scaled lapse
    # ------------------------------------------------------------------
    z = float(np.log(p_a / (1.0 - p_a)))
    eps = eps0 + eps1 * float(np.exp(-min(abs(z), 60.0)))
    eps = float(np.clip(eps, 0.0, 0.9))
    p_a = 0.5 + (p_a - 0.5) * (1.0 - eps)

    p = np.array([p_a, 1.0 - p_a], dtype=float)
    p = np.clip(p, 1e-12, None)
    p = p / p.sum()
    return p
```

**`policy source code`:**
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


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] Five experts with clearly ordered, well-spread validities (0.95, 0.85, 0.75, 0.65, 0.55) so that Take-The-Best's cue cascade is unambiguous. The 12 unique pairs (each repeated 8 times, ~96 trials) are chosen to create four distinct diagnostic classes:

(1) CONFLICT / non-compensatory pairs (trials 1, 2, 8, 10): the option favored by the single highest-validity discriminating cue is beaten on the tally (e.g. A=[1,0,0,0,0] vs B=[0,1,1,1,1], tally 1-4 for B). TTB predicts a confident choice of A; Tallying predicts a confident choice of B, with margin-graded confidence (4-cue margin in trials 1/8 vs 1-cue margin in trials 2/10).

(2) TALLY-TIE pairs (trials 3, 4): A wins two cues, B wins two cues, so Tallying is exactly at chance (50/50 regardless of beta and epsilon), whereas TTB decides confidently on the top cue. Any systematic deviation from 50% here is impossible for Tallying at any parameter setting and is therefore the sharpest single-trial discriminator.

(3) LOW-CUE-DECIDES pairs (trials 7, 9): the top two cues tie, the third cue (validity .75) discriminates in one direction while the two lowest cues both favor the other option, so TTB and Tallying make opposite predictions in a case where TTB has to descend the cascade rather than read the first cue.

(4) AGREEMENT pairs with different tally margins (trials 5, 6, 11, 12): both theories pick the same option, but Tallying's softmax makes accuracy strongly margin-dependent (4-1 vs 2-1), while TTB predicts identical choice probability sigmoid(beta) irrespective of margin. Comparing accuracy on trials 5/11 versus 6/12 gives a purely quantitative dissociation that does not depend on which option is 'correct'.

All pairs are mirrored across A and B to avoid side bias, and no pair is degenerate (every pair has at least one discriminating cue and the two theories either disagree in direction or in predicted confidence).

[1] Six experts with well-spread but deliberately *non-monotone* validity order across screen positions (0.90, 0.60, 0.80, 0.55, 0.70, 0.95) so that Take-The-Best's cue cascade (f5 > f0 > f2 > f4 > f1 > f3) cannot be confused with a simple left-to-right reading strategy, while Tallying (which ignores order entirely) is unaffected. Twelve unique pairs, each repeated 8 times (96 trials), fall into four diagnostic classes.

(1) NON-COMPENSATORY CONFLICT with graded tally margin (trials 1-4). The single highest-validity cue (f5, 95%) favours one option while the tally favours the other, by margin 4 (trials 1, 2), margin 2 (trial 3) or margin 1 (trial 4). TTB predicts the SAME confidence sigmoid(beta) for all four, always siding with the f5 winner; Tallying predicts the opposite direction with confidence strictly increasing in the margin (sigmoid(4*beta) > sigmoid(2*beta) > sigmoid(beta)). Direction AND the margin gradient jointly identify the model.

(2) TALLY-TIE pairs (trials 5-8). Feature-wise wins are exactly balanced (3-3 in trials 5/6 with no ties; 2-2 with two tied cues in trials 7/8), so Tallying is pinned at exactly 50% for every (beta, epsilon) - no parameter setting can move it. TTB decides confidently: on trials 5/6 the top cue f5 discriminates immediately; on trials 7/8 f5 ties and TTB must descend to f0 (90%). Any systematic departure from chance here is impossible for Tallying, making these the sharpest single-cell discriminators, and trials 7/8 additionally verify that subjects use validity rank rather than screen position.

(3) LOW-CUE-DECIDES CONFLICT (trials 9, 10). The two most valid cues (f5, f0) tie; the third-ranked cue f2 (80%) discriminates one way while the three lower cues (f4, f1, f3) all favour the other option (tally 1-3). TTB must walk the cascade and then choose against the majority; Tallying chooses the majority. Opposite directions in a case where TTB's one-reason logic is hardest to mimic.

(4) AGREEMENT pairs with extreme margin contrast (trials 11, 12). Both theories favour the same option, but with tally margin 5 (trial 11) versus margin 1 (trial 12). TTB predicts identical accuracy on the two; Tallying predicts near-ceiling on trial 11 and markedly lower accuracy on trial 12. This is a purely quantitative dissociation that is independent of choice direction and of any objective 'correct' answer.

All classes are mirrored across sides (TTB predicts A on 6 pairs and B on 6; Tallying's non-tie predictions are likewise 4/4), so side bias cannot masquerade as either model. No pair is degenerate: every pair has at least one discriminating cue, and in every pair the two theories differ either in predicted direction or in predicted confidence.

[2] GOAL: dissociate Take-The-Best (validity-ordered lexicographic search) from the 'read-order scanning' model (geometric weighting by SCREEN position, primacy for ~65% of subjects, recency for ~35%, with only a weak validity pull gamma<=0.25).

KEY DESIGN MOVE: the display order is deliberately NOT validity-ordered, and crucially the top-validity cue is placed in the MIDDLE of the screen (position 3, 95%), while the second-best cue sits at the END (position 5, 85%) and a mid-validity cue sits FIRST (position 0, 70%). Validities by screen position are [.70, .55, .60, .95, .65, .85], so TTB's cue cascade is pos3 > pos5 > pos0 > pos4 > pos2 > pos1. A primacy scanner anchors on pos0, a recency scanner anchors on pos5, and TTB anchors on pos3 -- three mutually distinguishable anchors. I verified numerically that this ordering is robust across the scanning model's whole parameter range: even at the maximum validity pull (gamma=.25) and the shallowest gradient (lam=1.2), pos0 still has the largest primacy weight and pos5 the largest recency weight, so the predictions below do not depend on lucky parameter values. (This is why pos0 was given 0.70 rather than the lowest validity: with a low-validity first cue the gamma=.25 pull would have flipped the primacy anchor onto the top-validity cue and destroyed the dissociation.)

TRIAL CLASSES (12 unique pairs, each repeated 8x = 96 trials; every pair mirrored across sides so side bias cannot mimic either model; TTB predicts A on 6 and B on 6, and so do both scanning orientations).

(1) DOUBLE-DISSOCIATION pairs 1-2 (A=[1,0,0,0,0,1] vs B=[0,0,0,1,0,0] and mirror). Only pos0, pos3, pos5 discriminate; the two screen-EXTREME cues favour one option and the top-validity MIDDLE cue favours the other. TTB must choose the pos3 option; BOTH primacy and recency scanners choose the other option (checked numerically for lam in [1.2,3.5] x gamma in [0,.25]: |S|>0.78 in every case). These are the sharpest cells -- no setting of the scanning model's parameters, of either orientation, can reproduce TTB here.

(2) PRIMACY-vs-TTB pairs 3-4 and 7-8. Pair 3: pos0 favours A, pos3 favours B, nothing else discriminates -> TTB=B, recency=B, primacy=A. Pair 7: pos3 ties so TTB must DESCEND to pos5 (A) while pos0 favours B -> TTB=A, recency=A, primacy=B. These isolate primacy readers and simultaneously verify that subjects use validity rank, not left-to-right position, when the leading screen cue conflicts with the cascade.

(3) RECENCY-vs-TTB pairs 5-6. pos3 (95%) favours B, pos5 (85%) favours A, nothing else discriminates -> TTB=B, primacy=B, recency=A. These isolate recency readers.

(4) COMPENSATORY / GRADED-CONFIDENCE pairs 9-12. Pair 9 (A=[1,0,0,0,0,0] vs B=[0,1,1,0,0,0]): one early cue against two slightly later, lower-validity cues -> TTB=A confidently, primacy=A but with markedly reduced relative evidence (S ~= 0.44-0.59 depending on lam), recency=B. Pair 11 (A=[0,0,0,0,0,1] vs B=[1,1,1,0,0,0]): the single last-position 85% cue against the three leading cues -> TTB=A, recency=A (S ~= 0.57-1.0), primacy=B with near-maximal evidence. Because TTB's choice probability is the SAME sigmoid(beta) on every non-tied pair, any systematic accuracy gradient across pairs 3/7 (near-|S|=1 for the scanner) versus pairs 9/11 (intermediate |S|) is a purely quantitative signature of the scanning model that TTB cannot produce at any (beta, epsilon).

DIAGNOSTIC YIELD: for a primacy subject TTB and the scanning model disagree in DIRECTION on 8 of 12 pairs (1,2,3,4,7,8,11,12); for a recency subject they disagree on 6 of 12 (1,2,5,6,9,10). Under TTB, choices should track pos3 (then pos5) regardless of screen position; under the competitor, choices should track pos0 (or pos5) and show margin-graded confidence. Every pair has at least one discriminating cue, so no cell is degenerate.

[3] GOAL: dissociate Take-The-Best (validity-ordered lexicographic search, one reason, constant confidence sigmoid(beta) on every discriminating pair) from 'read-order scanning' (geometric weighting by SCREEN position, ~65% primacy / ~35% recency readers, only a weak validity pull gamma<=0.25, and relative/normalised evidence that is graded when several later cues oppose the leading one).

DESIGN MOVE (new relative to previously explored designs): five experts, display order NOT validity-ordered, with the top-validity cue at screen position 2 and, crucially, the two screen EXTREMES given adjacent mid-high validities (pos0 = .78, pos4 = .72). Validities by screen position are [.78, .60, .95, .88, .72], so TTB's cascade is pos2 > pos3 > pos0 > pos4 > pos1. Three mutually distinct anchors: TTB anchors on pos2, a primacy scanner on pos0, a recency scanner on pos4. I verified the anchor assignment is robust over the scanning model's whole parameter box: with rank = gamma*vrank + (1-gamma)*pos, at the maximum validity pull gamma = .25 the ranks are [0.50, 1.75, 1.50, 2.50, 3.75], so pos0 is still the minimum (largest primacy weight) and pos4 still the maximum (largest recency weight); at gamma = 0 this holds trivially. Deliberately giving pos0 the third-highest validity (rather than the lowest) prevents the gamma pull from flipping the primacy anchor onto pos2, which would destroy the dissociation.

TRIAL CLASSES (12 unique pairs, each repeated 8x = 96 trials; every class mirrored across sides, so TTB predicts A on 6 and B on 6, and so do BOTH scanning orientations \u2014 side bias cannot mimic any model).

(1) DOUBLE DISSOCIATION, pairs 1-2: A=[1,0,0,0,1] vs B=[0,0,1,0,0]. Only the two screen extremes (favouring A) and the top-validity middle cue (favouring B) discriminate. TTB must choose the pos2 option; BOTH primacy and recency scanners choose the other option (worst-case relative evidence checked numerically: primacy S = .55 at gamma=.25, lam=1.2, rising to .94 at lam=3.5; recency S = .84 at gamma=0, lam=1.2). No parameter setting of either orientation reproduces TTB here \u2014 the sharpest cells.

(2) RECENCY-ISOLATING, pairs 3-4: A=[1,0,0,0,0] vs B=[0,0,0,0,1]. pos2 and pos3 tie, so TTB descends to pos0 (.78) and picks A; primacy also picks A (S ~ 1); recency picks B (S ~ -1). Under TTB the across-subject distribution of P(A) is unimodal near sigmoid(beta); under scanning it must be BIMODAL, with roughly one third of subjects locked on B. Subject-level bimodality on a cell where TTB is uniform is a signature TTB cannot produce at any (beta, epsilon).

(3) PRIMACY-ISOLATING, pairs 5-6: A=[1,0,0,0,0] vs B=[0,0,1,0,1]. TTB (pos2) and recency (pos4) both pick B; primacy picks A (S = .52 worst case at gamma=.25/lam=1.2, .82 at gamma=0). Complementary bimodality with the opposite majority direction.

(4) CASCADE-DESCENT DOUBLE DISSOCIATION, pairs 7-8: A=[0,0,0,1,0] vs B=[1,0,0,0,1]. The top cue pos2 ties, so TTB must walk to pos3 (.88) and choose A, while both scanners are dominated by the extremes and choose B (primacy S = -.84; recency S = -.54 to -.94 across the box). This verifies that any TTB-looking behaviour survives when the deciding validity cue is not the first-listed one, and doubles the direction-disagreement yield.

(5) COMPENSATORY / GRADED-CONFIDENCE, pairs 9-12. Pair 9: A=[1,0,0,0,0] vs B=[0,1,1,1,0] \u2014 one leading screen cue against the three next-read cues. TTB says B with its usual full confidence; a recency reader says B with near-maximal evidence (S = -.96); a primacy reader says A but with markedly REDUCED relative evidence (S = .24-.41 at lam=1.2, near 1 only at lam=3.5). Pair 11: A=[0,0,0,0,1] vs B=[0,1,1,1,0] \u2014 the mirror-image compensatory case for recency readers (recency S = +.41 weak, primacy S = -.96 strong, TTB = B). Because TTB predicts the SAME choice probability on all twelve pairs, any systematic accuracy gradient between the extreme-anchored pairs 1-8 (|S| near 1) and the compensatory pairs 9-12 (|S| intermediate for whichever orientation the subject has) is a purely quantitative signature of the scanning model, independent of direction and of any objective 'correct' answer.

DIAGNOSTIC YIELD: TTB and the scanning model disagree in DIRECTION on 8 of 12 pairs for a primacy subject (1,2,5,6,7,8,9,10) and on 8 of 12 for a recency subject (1,2,3,4,7,8,11,12). Under TTB choices track pos2 (then pos3, then pos0) regardless of screen location and show flat confidence; under the competitor they track pos0 or pos4, split bimodally across subjects, and show margin-graded confidence on the compensatory pairs. Every pair has at least one discriminating cue, so no cell is degenerate.

[4] GOAL: dissociate threshold-terminated compensatory integration with conclusion-graded commitment (weights fixed by validity, UNNORMALISED accumulation, early stop when no single unread expert could reverse the lead, amplified commitment after a stop vs. shallow readout of an unresolved residual) from read-order scanning (geometric weighting by SCREEN position, ~65% primacy / ~35% recency, evidence NORMALISED by the discriminating weight so one early cue is as confident as many).

DESIGN MOVE: five experts with validities by screen position [.78, .74, .70, .58, .90] \u2014 the single most valid expert (90%) is placed LAST, three mid-validity experts occupy the leading positions, and a near-useless expert (58%) sits at position 3. Under the advocated model the compressed weights are w \u2248 [.235, .191, .149, .043, .381] (stable across rho in [1.25,1.45]). Because the largest weight belongs to the last-read expert, the sufficiency bar for every earlier screen position is phi*w4 \u2248 .38-.44: NO single early cue can ever clear it, but a crowd of two-to-three leading mid cues can. The model therefore switches qualitatively as leading cues are added: 1 leading cue -> never stops, reads the 90% expert, shallow residual readout; 2 leading cues -> borderline/likely stop; 3 leading cues -> confident stop that never even reads the best expert (tallying-looking behaviour, choosing AGAINST the 90% expert). The competitor's weights are position-based, so its normalised evidence is near |S| = 1 on all of these, flat.

CORE LADDER (pairs 7/8, 3/4, 11/12, 5/6): the same target option is opposed by the last-listed 90% expert while it is supported by, respectively, (a) only pos0, (b) pos1+pos2, (c) pos0+pos1, (d) pos0+pos1+pos2. Advocated model predicts a steep MONOTONE gradient with a DIRECTION REVERSAL: P(support-side) \u2248 .25 (one cue, unresolved, best expert wins the residual), \u2248 .49 (two mid cues, unresolved and near chance), \u2248 .85 (pos0+pos1, stop typically triggered, amplified commitment despite a total weighted margin of only +.05), \u2248 .97 (three cues, stop certain). Read-order scanning predicts a FLAT ceiling: for a primacy reader |S| = .96-1.00 on all four (choose the leading-cue side every time), for a recency reader |S| = .78-1.00 in the opposite direction on all four. No (lam, gamma, beta, epsilon) can produce a monotone gradient plus a reversal across cells whose normalised evidence is essentially constant, and no accumulator with normalisation can put pair 3/4 near chance while pair 5/6 is at ceiling.

BOTH-ORIENTATIONS-AGREE CELL (pairs 1/2): d = [+1,-1,-1,+1,0] with the 90% expert tied. The earliest AND the latest discriminating cue favour the same option, so primacy AND recency scanners both choose it with S \u2248 +.45 to +.94 (P \u2248 .97-1.00) \u2014 no orientation escape hatch. The advocated model never clears the bar (max partial |E| = .235 < .38), so it exhausts the list and reads out a residual of only -.06, predicting a slight preference for the OPPOSITE option (P \u2248 .35-.43). This is the sharpest single cell: opposite direction, and the competitor is pinned near ceiling for its whole parameter box.

CASCADE-INVERSION CELL (pairs 9/10): d = [0,0,-1,0,+1] \u2014 the 70% expert at position 2 against the 90% expert at position 4. The advocated model cannot stop on the small pos2 cue, reads to the end and picks the 90% side (P \u2248 .76-.91); a primacy scanner picks the pos2 side robustly (pos2 keeps the lower inspection rank even at gamma = .25), a recency scanner picks the pos4 side. Direction disagreement with the 65% majority orientation, and a graded (not ceiling) confidence for the advocated model.

Every class is mirrored across sides (advocated model favours A on 6 pairs and B on 6; primacy scanning likewise 6/6), so side bias cannot mimic either model. 12 unique pairs x 8 repetitions = 96 trials. No pair is degenerate: every pair has at least one discriminating cue and the two theories differ in direction, in confidence level, or in the shape of the within-subject gradient across the ladder.

[5] GOAL: dissociate read-order scanning (position-geometric weights, ~65% primacy / ~35% recency readers, evidence NORMALISED by the discriminating weight so a one-cue decision is as confident as a six-cue decision) from threshold-terminated compensatory integration with conclusion-graded commitment (validity-only weights, UNNORMALISED accumulation, early stop only when no single unread expert can reverse the lead, kappa-amplified commitment after a stop vs. shallow beta readout of an unresolved residual).

DESIGN MOVE (new): six experts whose validities by SCREEN position are [.78, .66, .95, .86, .70, .56]. The two heaviest experts (95%, 86%) sit at positions 2 and 3, a mid expert (78%) leads, and the WEAKEST expert (56%) is last. Two consequences that the two theories read completely differently:
  (a) Competing model's compressed weights (rho in [1.25,1.45]) are w ~ [.18, .08, .34, .25, .11, .02]. Because w2=.34 is the largest, the sufficiency bar for positions 0 and 1 is phi*w2 ~ .33-.39, so NO combination of the two leading cues (max .27) can ever clear it; the model is forced to read the 95% expert, and if that expert opposes the leaders the residual is tiny and the trial ends UNRESOLVED with a shallow beta readout (near chance). Conversely a lone pos2 (or pos2+pos3) cue clears the bar at position 2 (.34/.28 = 1.2-1.4) and is committed with beta*kappa (ceiling).
  (b) Scanning anchors are robust and unambiguous: vrank = [2,4,0,1,3,5], so even at the maximum validity pull gamma=.25 the inspection ranks are [0.5, 1.75, 1.5, 2.5, 3.75, 5.0] - position 0 still has the smallest rank (primacy anchor) and position 5 (the 56% expert, whose low validity pushes it even later) still has the largest (recency anchor, gap 1.25 to pos4). Anchors hold for every lam in [1.2,3.5].

TRIAL CLASSES (12 unique pairs x 8 repetitions = 96 trials; every class mirrored, and the predicted-choice split is exactly 6 A / 6 B for the competing model AND for both scanning orientations, so side bias cannot mimic anything).

(1) SINGLE-CUE WEIGHT LADDER, pairs 1-4. Exactly one expert discriminates: pos5 (w=.02) in pair 1, pos1 (.08) in pair 2, pos0 (.18) in pair 3, pos2 (.34) in pair 4. For scanning, one discriminating cue gives |S| = 1 EXACTLY, for every lam, gamma and orientation, so all four cells must sit at the same flat ceiling sigmoid(beta) ~ .95-1.00. For the competing model a single discriminating cue can never trigger the reversal check (the list is exhausted at that cue), so it is read out with the shallow beta and accuracy is a steep MONOTONE ladder in the cue's validity weight: P ~ .52-.56 (pos5), .59-.74 (pos1), .67-.90 (pos0), .80-.98 (pos2). Flat ceiling vs. a four-step ladder rising from chance is a purely quantitative signature that no (beta, epsilon) of the normalising model can produce.

(2) BOTH-ORIENTATIONS-AGREE DIRECTION REVERSAL, pairs 7-8 (sharpest cells). d = [+1, 0, -1, 0, 0, +1]: the first-read 78% cue and the last-read 56% cue favour A, the mid-list 95% cue favours B. Primacy readers pick A (S = .54 worst case at gamma=.25/lam=1.2, .94 at lam=3.5); recency readers pick A even harder (S = .97-1.00 across the whole box) because the last position dominates. So scanning predicts P(A) ~ .90-1.00 with NO orientation escape hatch. The competing model cannot stop on pos0 (.18 < .33 bar), cannot stop on pos2 (|E| = .16 vs a .25-.29 bar), and exhausts the list with a residual of about -.14, predicting P(A) ~ .18-.35 - the OPPOSITE option. Opposite direction over both full parameter boxes.

(3) UNRESOLVED-RESIDUAL vs NORMALISED CEILING, pairs 5-6. d = [+1, +1, -1, 0, 0, +1]: three cues favour A but the single 95% cue opposes, leaving a final weighted margin of only -.055. Scanning: primacy S = .61-.87, recency S = .97-.99 - both orientations at ceiling for A (P >= .90) because normalisation makes the sign, not the size, of the margin decisive. Competing: the leaders may occasionally clear the pos1 bar (only at the shallowest s), otherwise the trial is unresolved and read out shallowly, giving P(A) ~ .36-.50. Same stimulus family as class (2) but here the dissociation is confidence-level rather than direction, which pins down normalisation independently of anchor identity.

(4) COMPETING-AT-CEILING / SCANNING-BIMODAL, pairs 9-10. d = [0, 0, +1, +1, 0, -1]: the 95% and 86% mid-list experts favour A, the last-read 56% expert favours B. The competing model stops at pos2 (ratio 1.16-1.41, stop prob .87-1.00) or certainly at pos3 (ratio > 4) and commits with beta*kappa: P(A) ~ .97-.99, unimodal across subjects. Scanning splits: primacy S = +.89 to +.98 (A), recency S = -.88 to -.99 (B), so the subject-level distribution of P(A) must be BIMODAL with ~35% of subjects locked on B and an aggregate near .65. Bimodality where the competitor is pinned at ceiling.

(5) MID-CUE STOP WITH A TINY MARGIN, pairs 11-12. d = [-1, 0, 0, +1, 0, 0]: the leading 78% expert favours B, the 86% expert at position 3 favours A. Competing: pos0 cannot stop, and after pos3 the margin (+.073) still fails the .11-.13 bar, so the trial ends unresolved and only weakly tilted, P(A) ~ .59-.67. Scanning: primacy S = -.83 to -.95 (B, ceiling), recency S = +.95 (A, ceiling) - again an extreme bimodal split with an aggregate near .35, i.e. the opposite direction from the competitor's modal prediction, plus within-subject near-determinism where the competitor predicts near-chance hesitation.

DIAGNOSTIC YIELD: direction disagreement with BOTH orientations on pairs 7/8; direction disagreement with the 65% majority on 9/10 and 11/12; confidence-level disagreement (ceiling vs near-chance) on 5/6; and a flat-vs-monotone contrast across the four single-cue cells 1-4 that no normalising model can produce and no unnormalised model can avoid. Every pair has at least one discriminating cue, so no cell is degenerate.

[6] GOAL: dissociate anchored compensatory integration with LOAD-GATED weight flattening + supra-additive corroboration + PARTIAL (psi~0.3) margin normalisation from read-order scanning (screen-position geometric weights, FULL per-trial normalisation, ~65% primacy / ~35% recency readers).

VALIDITIES BY SCREEN POSITION: [.68, .55, .95, .60, .86, .72]. The single best expert (95%) sits in the MIDDLE (pos2), a mid expert leads (pos0 = .68) and a mid expert closes (pos5 = .72). Under the competitor (rank = gamma*vrank + (1-gamma)*pos, gamma<=.25) the ranks are [0.75, 2.0, 1.5, 3.25, 3.25, 4.25], so pos0 is the primacy anchor and pos5 the recency anchor for EVERY (lam, gamma) in its box, while the advocated model's diagnosticity term keeps pos2 dominant at low load. Three distinct anchors, robustly.

THE CENTRAL ADVERSARIAL MOVE: several cells come in PAIRS with an IDENTICAL difference vector d = a - b, differing only in how many features carry a shared positive endorsement (L = both options rated 1). The competitor's evidence S = sum(w*d)/sum(w*|d|) depends only on d and screen position, so it predicts EXACTLY THE SAME choice probability within each such pair u2014 no parameter setting can produce any load effect. The advocated model predicts large, signed load effects (rho_eff = rho/(1+cL) flattens the validity gradient; delta_eff = delta*L/(1+L) turns on counting).

CELLS (8 base designs, each mirrored across sides -> 16 unique pairs, K = 6 -> 96 trials; advocated favours A on 8 and B on 8 overall, so side bias cannot mimic anything).

(1) SINGLE-CUE LADDER, trials 1-2 (+mirrors 9-10). Exactly one expert discriminates and L = 0: pos2 (95%) in trial 1, pos3 (60%) in trial 2. The competitor is pinned at |S| = 1 in BOTH cells for every lam, gamma and orientation -> a FLAT ceiling (~.95-1.00). The advocated model reads a single cue against a partially attenuated yardstick, E = w_j^(1-psi), giving a steep ladder: P ~ .99 for the 95% cue but only ~.55-.60 for the 60% cue. Flat ceiling vs. a 40-point ladder is a purely quantitative signature the normalising model cannot produce.

(2) LOAD x CUE-STRENGTH CROSSOVER, trials 3-4 (+11-12). Trial 3 repeats trial 2's d (only pos3 discriminates) but adds three shared positive endorsements (L = 3); trial 4 repeats trial 1's d (only pos2 discriminates) with L = 4. Competitor: identical to trials 2 and 1 respectively (zero effect). Advocated: flattening RAISES the weak cue's relative weight (P .55-.60 -> .72-.80, about +18 pts) and LOWERS the strong cue's (P ~.99 -> ~.92-.93, about -6 pts). A crossover interaction of ~25 points produced by ratings that discriminate nothing at all is impossible for any model that only reads d.

(3) LOAD-INDUCED DIRECTION REVERSAL, trials 5-6 (+13-14) u2014 the sharpest cells. Both have d = [+1, 0, -1, 0, 0, +1]: the first-read 68% cue and the last-read 72% cue favour A, the mid-list 95% cue favours B. Trial 5 has L = 0, trial 6 pads three shared positives (L = 3). The competitor predicts A in BOTH and in both orientations (primacy S ~ +.43, recency S ~ +.93; P(A) ~ .85-.99) with no difference between them u2014 no orientation escape hatch, no load sensitivity. The advocated model predicts B at L = 0 (P(A) ~ .07-.10, checked at the extreme primacy corner alpha=2.1/lam=0.9/rho=2.3) and A at L = 3 (P(A) ~ .84-.92, checked across c in [.65,1.10] and rho in [2.3,2.7]) because flattening plus the 2-vs-1 corroboration bonus flips the sign of E. Same d, opposite direction: within-subject reversal across load is diagnostic no matter what the individual's parameters are.

(4) UNIMODAL vs BIMODAL, trial 7 (+15). d = [0,0,+1,0,0,-1]: the middle 95% expert against the last-read 72% expert, L = 0. Advocated: unimodal P(A) ~ .95-.98 for every subject. Competitor: primacy readers S ~ +.93 (P(A) ~ .98) but the ~35% recency readers S ~ -.93 (P(A) ~ .02) -> a strongly BIMODAL subject-level distribution with aggregate ~ .65. Both the aggregate level and the shape of the between-subject distribution differ.

(5) MAJORITY-DIRECTION DISAGREEMENT, trial 8 (+16). d = [+1,0,-1,0,0,0]: leading 68% cue vs. middle 95% cue, L = 0. Advocated: B, confidently (P(A) ~ .04-.09) because diagnosticity beats the primacy multiplier when rho is convex and load is zero. Competitor: the 65% primacy majority chooses A (P(A) ~ .85-.99), the recency minority chooses B; aggregate ~ .60-.65, bimodal. This cell also anchors the interpretation of trials 5/6: if subjects show the primacy-anchored A response here AND no load effect in 5-vs-6, the competitor wins outright.

Every pair has at least one discriminating cue, no cell is degenerate, and each of the three load-matched contrasts (2 vs 3, 1 vs 4, 5 vs 6) is a within-subject test on which the competing theory is mathematically forced to predict a null.

[7] GOAL: dissociate READ-ORDER SCANNING (screen-position geometric weights, ~65% primacy / ~35% recency readers, FULL per-trial normalisation of evidence S = sum(w*d)/sum(w*|d|)) from ANCHORED COMPENSATORY INTEGRATION (convex diagnosticity (v-.5)^rho, primacy attention MULTIPLIER, load-gated flattening + supra-additive corroboration, and only PARTIAL margin normalisation psi~0.3).

VALIDITIES BY SCREEN POSITION: [.72, .56, .80, .95, .62]. Deliberately non-monotone: the near-oracle expert (95%) sits at position 3, a mid expert leads (pos0 = .72) and a weak expert closes (pos4 = .62). Anchor check for the scanning model (rank = gamma*vrank + (1-gamma)*pos, vrank = [2,4,1,0,3]): at gamma = 0 ranks are [0,1,2,3,4]; at the maximum validity pull gamma = .25 ranks are [0.50, 1.75, 1.75, 2.25, 3.75]. So pos0 is the primacy anchor and pos4 the recency anchor for EVERY (lam, gamma) in the box, while the competitor's weights (rho=2.5, alpha=1.8, lam=1.1, L=0) are w = [.232, .005, .216, .529, .019], i.e. dominated by pos3. Three distinct anchors, robustly, and crucially the competitor's dominant cue is neither screen extreme.

CELLS (12 base designs, each mirrored across sides -> 24 unique pairs, K = 4 -> 96 trials; mirroring makes the predicted-choice split exactly 12 A / 12 B for the competitor AND for both scanning orientations, so side bias cannot mimic anything).

(1) SINGLE-CUE LADDER, cells 1-4 (pairs 1-8, L = 0). Exactly one expert discriminates: pos1 (.56, w=.005), pos4 (.62, w=.019), pos0 (.72, w=.232), pos3 (.95, w=.529). For scanning, one discriminating cue gives |S| = 1 EXACTLY for every lam, gamma and orientation, so all four cells sit at the same flat ceiling sigmoid(beta) ~ .93-1.00. For the competitor E = w^(1-psi) and accuracy is a steep MONOTONE ladder: P ~ .55 (pos1), ~.61 (pos4), ~.94 (pos0), ~.99 (pos3). Flat ceiling vs a 45-point ladder is a purely quantitative signature that no fully normalising model can produce and no partially normalising model can avoid. The pos3 rung also serves as an attention/calibration anchor: it shows the low pos1 value is not general inattention.

(2) LOAD-MATCHED NULL #1, cells 5 & 6 (pairs 9-12). Both have d = [+1,0,0,-1,0] (leading 72% cue vs the 95% cue); cell 5 has L = 0 (a=[1,0,0,0,0] vs b=[0,0,0,1,0]), cell 6 pads three shared positive endorsements at pos1/pos2/pos4 (a=[1,1,1,0,1] vs b=[0,1,1,1,1]). Scanning depends only on d and screen position, so it is MATHEMATICALLY FORCED to predict identical behaviour in both: primacy readers A (S ~ +.95, P(A) ~ .98), recency readers B, aggregate ~ .65 in BOTH cells. The competitor predicts a full DIRECTION REVERSAL driven by ratings that discriminate nothing: rho_eff = rho/(1+cL) flattens (w -> [.380,.087,.203,.240,.091]) so P(A) goes from ~.04-.08 (L=0) to ~.61-.77 (L=3) across its whole (rho, c, alpha, lam, beta) box.

(3) BOTH-ORIENTATIONS-AGREE DIRECTION CELL, cell 7 (pairs 13-14) - the sharpest single cell. d = [+1,0,0,-1,+1]: the first-read 72% cue and the last-read 62% cue favour A, the 95% cue favours B. Primacy readers choose A (S = .64 worst case at gamma=.25/lam=1.2, .95 at gamma=0), recency readers choose A even more strongly (S = .54-.99 across the box) - NO orientation escape hatch, scanning predicts P(A) ~ .90-1.00 for every subject. The competitor chooses B for its entire box (P(A) ~ .10 at nominal parameters, ~.22-.29 even at the extreme primacy corner alpha=2.1/lam=0.9/rho=2.3). Opposite direction over full parameter boxes.

(4) LOAD-MATCHED NULL #2, cell 8 (pairs 15-16). Same d as cell 7 but with L = 2 shared positives (a=[1,1,1,0,1] vs b=[0,1,1,1,0]). Scanning: identical to cell 7 (P(A) ~ .95, both orientations). Competitor: flattening plus the 2-vs-1 corroboration bonus (delta_eff = delta*L/(1+L)) flips it to P(A) ~ .90 - a ~80-point within-subject reversal on a cell where the advocated theory predicts an exact null.

(5) COMPENSATORY / GRADED-EVIDENCE CELL, cell 9 (pairs 17-18). a=[1,0,0,0,0] vs b=[0,1,1,1,0]: the leading cue against the three next-read cues, L = 0. Competitor: B at near-ceiling (P(A) ~ .02). Scanning: primacy readers still choose A but with markedly REDUCED relative evidence (S = .41 at lam=1.2 rising to .94 at lam=3.5), recency readers choose B at S ~ -.96. The predicted within-subject accuracy DROP between cell 3 (|S| = 1) and cell 9 (|S| intermediate) for primacy readers, alongside direction disagreement with the majority, is a scanning-specific gradient.

(6) UNIMODAL vs BIMODAL, cell 10 (pairs 19-20). a=[0,0,0,1,0] vs b=[0,0,0,0,1]: the 95% expert (pos3) against the last-read 62% expert (pos4). Competitor: unimodal ceiling, P(A) ~ .99 for every subject. Scanning: primacy S = +.54 to +.94 (A), recency S = -.54 to -.99 (B) -> a strongly BIMODAL subject-level distribution with ~35% of subjects locked on B and an aggregate near .65. Both the aggregate level and the SHAPE of the between-subject distribution differ.

(7) ANCHOR-vs-DIAGNOSTICITY BALANCE + its load partner, cells 11 & 12 (pairs 21-24). d = [+1,0,-1,0,0]: leading 72% cue vs the 80% cue at pos2. At L = 0 the competitor's primacy multiplier almost exactly cancels the diagnosticity gap (P(A) ~ .54, low between-subject spread); with L = 3 shared positives (cell 12: a=[1,1,0,1,1] vs b=[0,1,1,1,1]) flattening pushes it to P(A) ~ .83. Scanning predicts the same strongly bimodal .97/.03 split (aggregate ~.65) in both, with zero load effect.

DIAGNOSTIC YIELD: (i) flat-vs-ladder across four single-cue cells tests full vs partial normalisation; (ii) three load-matched contrasts (5vs6, 7vs8, 11vs12) are within-subject tests on which the advocated theory is mathematically forced to predict an exact null while the competitor predicts reversals of up to 80 points; (iii) cell 7 gives direction disagreement with BOTH scanning orientations; (iv) cells 9-12 give bimodality where the competitor is unimodal, plus direction disagreement with the 65% primacy majority. Every pair has at least one discriminating cue, so no cell is degenerate.

[8] GOAL: dissociate the advocated model (mildly compressed diagnosticity weights x reading-position attention, serial read with a sufficiency stop, FULLY relative evidence R in [-1,1], SILENCE CALIBRATION c = 1 - eta*(fraction silent)^q * mean diagnosticity of the speaking experts, and a CONVEX kappa read-out that turns small residual imbalances into coin flips) from read-order scanning (geometric weights in SCREEN position, ~65% primacy / ~35% recency readers, evidence FULLY normalised so any unanimous cue set gives |S| = 1 exactly).

VALIDITIES BY SCREEN POSITION: [.72, .60, .95, .86, .56, .70]. The near-oracle expert (95%) sits mid-list at pos2, a mid expert leads (pos0 = .72) and a mid expert closes (pos5 = .70), with a weak expert (.56) at pos4 and a near-useless one (.60) at pos1. Scanning anchors are robust: vrank = [2,4,0,1,5,3], so with rank = gamma*vrank + (1-gamma)*pos the ranks at the maximum validity pull gamma = .25 are [0.50, 1.75, 1.50, 2.50, 4.25, 4.50] and at gamma = 0 they are [0,1,2,3,4,5]; pos0 is the primacy anchor and pos5 the recency anchor for EVERY (lam, gamma) in the competitor's box. The advocated model's weights at nominal parameters (rho = 1.15, alpha = 0.9, tau = 1.3) are w = [.74, .22, 1.06, .75, .09, .37], i.e. dominated by the mid-list 95% cue but with the two screen extremes summing to almost exactly that value (1.11 vs 1.06) - the crossover the theory advertises.

8 base cells, each mirrored across sides (16 unique pairs, K = 6 -> 96 trials). Mirroring makes the predicted-choice split exactly 8 A / 8 B for the advocated model AND for both scanning orientations, so side bias cannot mimic anything.

(1) REVERSED SINGLE-CUE VALIDITY LADDER + SILENCE DILUTION, cells 1-4 (pairs 1-8). Cell 1: only pos2 (.95) discriminates. Cell 2: only pos4 (.56). Cell 3: only pos0 (.72). Cell 4: pos0, pos2 and pos5 ALL favour the same option. In every one of these four cells the discriminating cues are unanimous, so the competitor's normalised evidence is |S| = 1 EXACTLY for every lam, gamma, orientation - it is mathematically forced to predict the SAME probability, sigmoid(beta) ~ .95-1.00, in all four cells. The advocated model's silence term makes them wildly different and, crucially, INVERTED with respect to cue quality: with 5 of 6 experts silent, c = 1 - eta*(5/6)^q*vbar collapses for a lone .95 cue (c ~ .02-.22 -> P ~ .52-.67) but barely touches a lone .56 cue (c ~ .84-.90 -> P ~ .93-.95), with the lone .72 cue in between (P ~ .75-.90); adding two agreeing cues to the .95 cue (cell 4, silence only 3/6) restores near-ceiling commitment (P ~ .90-.94). A monotone reversed ladder (worst expert alone MORE decisive than the best expert alone) plus a large dilution gain from adding redundant cues, across cells where the competitor is pinned to a single number, is the flagship quantitative signature.

(2) BOTH-ORIENTATIONS-AGREE INDIFFERENCE CELL, cell 5 (pairs 9-10). d = [+1,0,-1,0,0,+1]: the first-read 72% cue and the last-read 70% cue favour A, the mid-list 95% cue favours B. Primacy readers choose A (S = .54-.96 across the box), recency readers choose A even harder (S = .78-1.00): scanning predicts P(A) ~ .90-1.00 with NO orientation escape hatch. The advocated model lands on a residual |R| ~ 0.02-0.09 which the convex kappa read-out crushes to nothing: P(A) ~ .47-.57 across its whole (rho, alpha, tau, kappa) box, i.e. a genuine coin flip where the competitor is at ceiling. Sharpest single cell.

(3) UNIMODAL-NEAR-CHANCE vs BIMODAL-EXTREME, cell 6 (pairs 11-12). d = [+1,0,-1,0,0,0]: leading 72% cue against the mid-list 95% cue, four experts silent. Advocated: unresolved read, |R| ~ .18 discounted by kappa and by the silence factor (c ~ .57) -> P(A) ~ .46-.52 for every subject. Scanning: primacy readers P(A) ~ .90-1.00, recency readers P(A) ~ .00-.10 -> strongly BIMODAL subject-level distribution with aggregate ~ .64. Aggregate level AND between-subject shape differ.

(4) SUFFICIENCY-STOP CELL, cell 7 (pairs 13-14). d = [0,0,+1,0,0,-1]: the 95% expert (pos2) against the last-read 70% expert. The advocated model clears the sufficiency bar at pos2 (|E|/phi*max unread = 1.34) and commits on a categorical R = +1, but with the 4/6-silence discount: P(A) ~ .78-.88, unimodal. Scanning splits again: primacy P(A) ~ .99, recency P(A) ~ .01, aggregate ~ .65 with maximal between-subject variance. Same aggregate ballpark, opposite distributional shape - and, paired with cell 6, it shows the advocated model moving where scanning cannot (cells 6 and 7 differ by 30+ points for the advocated model but are both ~.65 aggregate for scanning).

(5) DENSE CROWD-BEATS-EXTREMES DIRECTION CELL, cell 8 (pairs 15-16). d = [+1,-1,-1,-1,0,+1]: both screen extremes favour A while the three mid-list cues (.60, .95, .86) favour B; only one expert is silent so the silence discount is ~zero. The advocated model accumulates past the extremes, triggers the stop at pos3 with R ~ -.47 and predicts P(A) ~ .28-.35. Scanning predicts A for BOTH orientations (primacy S = .41-.73, recency S = .78-.96 -> P(A) ~ .84-1.00). Direction disagreement with both orientations again, this time in a dense display where no silence machinery is doing the work - so it cannot be explained away as a silence artefact.

DIAGNOSTIC YIELD: (i) four cells where the competitor is mathematically pinned to one identical probability while the advocated model predicts a 35-point INVERTED validity ladder and a large redundancy/dilution gain; (ii) two cells (5 and 8) where scanning predicts near-ceiling A for both primacy and recency readers while the advocated model predicts a coin flip and a reversal respectively; (iii) two cells (6, 7) where scanning must be bimodal at the subject level and the advocated model unimodal. Every pair has at least one discriminating cue; no cell is degenerate.

[9] GOAL: dissociate READ-ORDER SCANNING (advocated: geometric weights in SCREEN position, ~65% primacy / ~35% recency readers, only a weak validity pull gamma<=.25, evidence FULLY normalised so S = sum(w*d)/sum(w*|d|) and any unanimous discriminating set gives |S| = 1 exactly) from READ-UNTIL-SUFFICIENT INTEGRATION (competitor: mildly compressed validity weights x a position attention multiplier, sufficiency stop, SILENCE CALIBRATION c = 1 - eta*(fraction silent)^q*mean diagnosticity of the speakers, and a CONVEX kappa read-out that crushes small residual imbalances).

VALIDITIES BY SCREEN POSITION: [.62, .70, .95, .86, .66, .58]. New layout relative to everything explored so far: the two heaviest experts sit at the CENTRE-RIGHT of the list (pos2 = .95, pos3 = .86), the two screen EXTREMES are the two WEAKEST experts (pos0 = .62, pos5 = .58), and two mid experts fill pos1 (.70) and pos4 (.66). Anchor check for scanning: vrank = [4,2,0,1,3,5], so rank = gamma*vrank + (1-gamma)*pos gives pos0 = 4*gamma (minimum for every gamma < 1/3, hence the primacy anchor for the whole gamma in [0,.25] box) and pos5 = 5 exactly (maximum, since the largest competitor rank is pos4 = 4-gamma), so pos5 is the recency anchor for every (lam, gamma). Crucially the extremes are the LEAST valid experts, so a scanning reader of either orientation anchors on a cue the competitor treats as nearly negligible: competitor weights at nominal parameters (rho = 1.15, alpha = .9, tau = 1.3, gamma_att = .05) are w = [.37, .50, 1.06, .75, .29, .13], i.e. dominated by the mid-list .95 expert, with the two extremes together (.50) worth less than half of it.

8 base cells, each mirrored across sides -> 16 unique pairs, K = 6 -> 96 trials. Mirroring makes the predicted-choice split exactly 8 A / 8 B for the competitor AND for both scanning orientations, so side bias cannot mimic anything. Sparse cells are padded with SHARED positive endorsements on non-discriminating features (both theories depend only on d, so the padding changes nothing but makes the displays natural).

(1) INVERTED SINGLE-CUE LADDER + CORROBORATION DILUTION, cells 1-4 (pairs 1-8). In all four the discriminating cues are unanimous, so scanning's normalised evidence is |S| = 1 EXACTLY for every lam, gamma and orientation: it is mathematically pinned to one number, sigmoid(beta) ~ .95-1.00, in all four cells. The competitor's silence term makes them wildly different and INVERTED with respect to cue quality: cell 1 (only pos2, the .95 expert, discriminates; 5/6 silent) gives c ~ .02-.22 -> P ~ .52-.67, a near coin flip; cell 2 (only pos5, the .58 expert) gives c ~ .78-.86 -> P ~ .92-.94; cell 3 (only pos1, .70) is the intermediate rung, P ~ .81-.89; cell 4 adds the .70 and .86 experts AGREEING with the .95 expert (3/6 silent) and restores near-ceiling commitment, P ~ .90-.94. So the competitor predicts (i) the single WORST expert alone to be ~35 points MORE decisive than the single BEST expert alone, and (ii) a ~40-point gain from adding two strictly weaker corroborating experts to cell 1's cue. Scanning predicts an exact null across all four cells. This flat-vs-inverted-ladder contrast is the flagship quantitative signature and no (beta, epsilon, lam, gamma, orientation) of the scanning model can produce any spread here.

(2) BOTH-ORIENTATIONS-AGREE DIRECTION CELL, cell 5 (pairs 9-10), sharpest cell. d = [+1,0,-1,-1,0,+1]: the first-read .62 cue and the last-read .58 cue favour A, the two mid-list heavyweights (.95, .86) favour B. Scanning chooses A for BOTH orientations across its whole box (primacy S = +.17 at the worst corner gamma=.25/lam=1.2, +.70 to +.79 elsewhere; recency S = +.88 to +.93), so P(A) ~ .67-1.00 with no orientation escape hatch. The competitor cannot stop on pos0 (.37 vs a 1.11 bar), reads the two heavyweights, stops at pos3 with R ~ -.66 and, because only 2/6 experts are silent, the silence discount is negligible (c ~ .93): P(A) ~ .18-.25. Opposite direction over both full parameter boxes, in a dense display where no silence machinery is doing the work.

(3) SPARSE EXTREMES DUEL: UNIMODAL vs BIMODAL, cell 6 (pairs 11-12). d = [+1,0,0,0,0,-1]: the leading .62 expert against the last-read .58 expert, 4/6 silent. The competitor clears the sufficiency bar immediately at pos0 (ratio 2.7) and commits on R = +1 with only a mild silence discount (the speakers are both low-validity, so vbar ~ .22 and c ~ .87): P(A) ~ .93-.95, UNIMODAL for every subject. Scanning splits hard: primacy readers S ~ +1 (P(A) ~ .99), recency readers S ~ -1 (P(A) ~ .01), giving a strongly BIMODAL subject-level distribution with aggregate ~ .65. Same ballpark aggregate is impossible to confuse with the competitor's tight unimodal .94, and the between-subject distribution shape is the decisive statistic.

(4) MAJORITY-DIRECTION DISAGREEMENT, cell 7 (pairs 13-14). d = [-1,0,+1,0,0,0]: the leading .62 cue favours B, the mid-list .95 cue favours A, 4/6 silent. The competitor never stops on pos0, ends with R ~ +.48 which the convex read-out and the silence factor (c ~ .56) discount to P(A) ~ .60-.68, unimodal. Scanning: the 65% primacy majority chooses B (P(A) ~ .02-.10), the 35% recency minority chooses A (P(A) ~ .90-.98); aggregate ~ .35, i.e. the OPPOSITE side from the competitor, with maximal between-subject variance.

(5) DENSE 3-vs-3 SPLIT, cell 8 (pairs 15-16). d = [+1,+1,+1,-1,-1,-1]: no expert is silent, so the competitor's silence machinery is exactly off; it accumulates .37+.50+1.06 = 1.93 by pos2, clears the bar set by the strongest unread expert (1.05*.75 = .79) and commits on R = +1: P(A) ~ .95-.97, unimodal. Scanning: primacy readers also choose A at ceiling, but recency readers are dominated by pos5 and choose B at ceiling, so the subject-level distribution must again be BIMODAL with ~35% of subjects locked on B and an aggregate near .65 -- a 30-point aggregate gap plus a categorical shape difference, in the one cell where neither silence calibration nor normalisation-of-sparse-evidence can be blamed.

DIAGNOSTIC YIELD: (i) four cells where scanning is mathematically pinned to a single probability while the competitor predicts a 40-point INVERTED validity ladder plus a large corroboration gain; (ii) one cell (5) with direction disagreement against BOTH scanning orientations over the full boxes; (iii) three cells (6, 7, 8) where scanning must be bimodal at the subject level and the competitor unimodal, one of them (7) with an aggregate direction reversal. Every pair has at least one discriminating cue; no cell is degenerate.

[10] GOAL: dissociate READ-UNTIL-SUFFICIENT INTEGRATION (advocated: serial read in screen order, compressed diagnosticity x position attention, sufficiency stop, FULLY RELATIVE evidence R, silence discount c = 1 - eta*(fraction silent)^q*vbar where EVERY non-discriminating feature counts identically, convex kappa read-out) from REASON-COUNTING WITH A TWO-REGIME SALIENCE GATE (competitor: validity-graded vote, unanimity boost, saturating majority bonus, position salience at FULL force only when the reason count is deadlocked and at a constant residual g_res~.25 otherwise, sharp silence threshold in which mutual endorsements count only at weight omega~.82, c floored at c_min).

VALIDITIES BY SCREEN POSITION: [.70, .60, .95, .82, .66]. The near-oracle expert (95%) sits mid-list at pos2, the second heavyweight (.82) sits at pos3, and the two SCREEN EXTREMES are deliberately mid-low (.70, .66) so that (i) the advocated model's attention multiplier makes the extremes roughly, but not quite, balance the mid-list heavyweight (w = [.68, .23, 1.08, .68, .30] at nominal rho=1.15, alpha=.95, tau=1.3), and (ii) the competitor's salience term e_sal = [1.02, .42, .27, .42, 1.02] loads precisely on the two weakest experts, so its deadlock gate can overturn the validity vote. n = 5 is chosen because with exactly 2 of 5 features silent the competitor's effective silence fraction is .40 for both-0 padding but .82*.40 = .33 for both-1 padding, which straddles its sharp threshold sil_f0 ~ .36 -- while the advocated model's f_sil is .40 in BOTH cases, i.e. a mathematically forced null.

8 base cells, each mirrored across sides -> 16 unique pairs, K = 6 -> 96 trials. Mirroring makes the predicted-choice split exactly 8 A / 8 B for BOTH theories, so side bias cannot mimic anything.

(1) SILENCE-COMPOSITION TRIAD, cells 1-3 (pairs 1-6). All three have the IDENTICAL difference vector d = [+1,0,+1,0,+1] (unanimous 3-0 for A; pos0 .70, pos2 .95, pos4 .66 speak, pos1 and pos3 are silent). They differ ONLY in what the two silent experts said: cell 1 both-0 (A=[1,0,1,0,1] vs B=[0,0,0,0,0]), cell 2 both-1 (A=[1,1,1,1,1] vs B=[0,1,0,1,0]), cell 3 mixed (one both-1, one both-0). The advocated model's silence term counts non-discriminating features regardless of level, and its evidence uses only discriminating cues, so it predicts EXACTLY THE SAME probability (~.93-.95, R = 1 after a certain stop at pos2) in all three cells for every parameter setting. The competitor's omega-weighted, threshold-sharp silence term gives c = .16 / .77 / .45 respectively, hence P(A) ~ .72 / .99 / .93 -- a ~27-point ordered spread driven by ratings that discriminate nothing. This within-subject triad is the flagship: one theory is pinned to a flat line, the other to a monotone ladder.

(2) SECOND SILENCE-COMPOSITION CONTRAST IN A CONFLICT DISPLAY, cells 6-7 (pairs 11-14). Both have d = [+1,0,-1,0,+1] (the two screen extremes favour A, the mid-list 95% expert favours B), with the two silent experts both-1 in cell 6 and both-0 in cell 7. Advocated: identical in both (~.45; it stops at pos2 with R ~ -.23, which the convex read-out crushes toward chance -- a mild preference for B). Competitor: P(A) = .65 (both-1) vs .53 (both-0), and it chooses the OPPOSITE option from the advocated model in the both-1 version because mu*g_res*S = +.80 overturns the validity vote V = -.15. Direction disagreement plus a second forced-null test.

(3) DEADLOCK-GATE CELL, cell 4 (pairs 7-8) -- the sharpest cell. d = [+1,0,-1,-1,+1]: the two screen extremes (.70, .66) favour A, the two mid-list heavyweights (.95, .82) favour B, one silent expert. Now kA = kB = 2, so the competitor's gate opens to FULL force (g = 1): drive = V + mu*S = -.80 + 2.45 = +1.65 and P(A) ~ .88. The advocated model reads past pos0, stops at pos3 with R ~ -.44 and predicts P(A) ~ .28-.35 across its whole (rho, alpha, tau, kappa, beta) box (checked at both corners: rho=1.32/alpha=0.7 -> .28; rho=1.0/alpha=1.2/tau=1.6 -> .35). Opposite direction, ~55 points, over both full parameter boxes.

CRUCIALLY, cell 4 and cell 6 differ by a SINGLE feature (whether the .82 expert at pos3 also opposes A). The competitor must become MORE pro-A when that extra expert is added against A (.65 -> .88, because the added reason deadlocks the count and unleashes the salience gate); the advocated model must move the other way (.45 -> .32). A non-monotonicity of exactly this sign is a competitor-only signature; monotone decline is advocated-only.

(4) SUFFICIENCY-STOP CELL, cell 5 (pairs 9-10). d = [+1,+1,0,-1,0]: the two leading cues (.70, .60) favour A, the later .82 expert favours B, two silent both-1. The advocated model's running lead after pos0+pos1 (E = .91) clears the bar set by the strongest unread expert (phi*w3 ~ .71), so it stops and commits categorically on R = +1: P(A) ~ .85-.95 across its box. The competitor never stops here (its first-cue commitment is authority-capped and pos0 is outranked by pos2/pos3), integrates everything to a near-null vote V = -.02 rescued only by the residual salience term, giving P(A) ~ .61. A 25-35 point confidence gap in the same direction, driven purely by stopping vs full integration.

(5) THREE-MID-EXPERTS-vs-ONE-ORACLE CELL, cell 8 (pairs 15-16). d = [+1,+1,0,-1,+1]: pos0 (.70), pos1 (.60) and pos4 (.66) favour A while the mid-list 95% expert favours B; only one expert is silent, so silence machinery is essentially OFF in both models and cannot be blamed for anything. The competitor fires its saturating majority bonus (|dk| = 2 -> M = .67) plus residual salience and predicts a decisive P(A) ~ .89. The advocated model's compressed weights make this an almost exact standoff (sum of the three mid weights 1.21 vs 1.08 for the oracle), the final R ~ +.06 is crushed by the convex read-out, and P(A) ~ .47-.68 across its box -- near indifference where the competitor is near ceiling.

DIAGNOSTIC YIELD: (i) two forced-null contrasts (cells 1/2/3 and cells 6/7) where the advocated theory is mathematically committed to zero difference and the competitor predicts 12-27 point ordered shifts produced by non-discriminating ratings; (ii) one full direction disagreement (cell 4, ~55 points) plus a sign-reversed single-feature manipulation (cell 4 vs cell 6); (iii) two magnitude dissociations (cells 5 and 8, 25-35 points, opposite in sign to each other) that separate sufficiency-stopping + convex read-out from reason-counting with majority/unanimity bonuses. Every pair has at least one discriminating cue; no cell is degenerate.

[11] GOAL: dissociate the advocated REASON-COUNTING WITH A TWO-REGIME SALIENCE GATE (validity-graded vote, symmetric first/last-row salience at FULL force only when the reason count is deadlocked and at a CONSTANT residual g_res~.25 otherwise, saturating majority bonus, unanimity bonus, sharp silence threshold in which mutual endorsements count only at weight omega~.82, c floored at c_min) from READ-UNTIL-SUFFICIENT INTEGRATION (competitor: compressed diagnosticity x a PRIMACY-dominated attention multiplier, serial read with a sufficiency stop, FULLY RELATIVE evidence R, convex kappa read-out, and a silence discount in which EVERY non-discriminating feature counts identically).

VALIDITIES BY SCREEN POSITION: [.70, .95, .86, .70, .60, .78, .70]. Seven experts is chosen deliberately: with exactly 3 of 7 features silent, the advocated model's effective silence fraction is 3/7 = .429 when the silent experts endorsed NEITHER product but .82*3/7 = .351 when they endorsed BOTH -- values that STRADDLE its sharp threshold sil_f0 ~ .36 (sigmoid weights .93 vs .42 at k=38), while the competitor's f_sil is 3/7 in BOTH cases, i.e. a mathematically forced null. Three experts share validity .70 and sit at pos0, pos3 and pos6 (both screen ends plus the exact middle), which permits validity-matched swaps of screen position. The oracle (.95) is deliberately placed at pos1 rather than pos0, and in EVERY cell the first discriminating cue is the .70 expert at pos0, so the advocated model's authority-capped first-cue commitment is uniformly suppressed (ratio .40/(1.05*.90) = .42 -> q_com ~ .01) and cannot contaminate any contrast. Advocated salience e_sal (tau=1) = [1.00,.37,.15,.10,.15,.37,1.00] (symmetric U); competitor attention (alpha~.9,tau~1.3,gamma<=.15) = [1.90,1.42,1.20,1.09,1.05,1.04,1.06] (primacy only, essentially flat over pos2-pos6), giving competitor weights w = [.66,1.26,.82,.38,.17,.54,.37] -- dominated by the mid-list .95 expert.

8 base cells, each mirrored across sides -> 16 unique pairs, K = 6 -> 96 trials. Mirroring makes the predicted-choice split exactly 8 A / 8 B for BOTH theories, so side bias cannot mimic anything.

(1) SILENCE-COMPOSITION TRIAD, cells 1-3 (pairs 1-3 + mirrors 9-11). FLAGSHIP. All three carry the IDENTICAL difference vector d = [+1,0,+1,0,+1,+1,0] (unanimous 4-0 for A; .70,.86,.60,.78 speak; pos1, pos3, pos6 silent). They differ ONLY in what the three silent experts said: cell 1 all both-0, cell 2 two both-1 + one both-0, cell 3 all both-1. The competitor's silence term counts non-discriminating features regardless of level and its evidence uses only discriminating cues, so it is FORCED to predict exactly the same probability in all three (stop at pos2 with R = 1, P(A) ~ .94) for every parameter setting. The advocated model's omega-weighted, threshold-sharp term gives c = .21 / .45 / .65 and hence P(A) ~ .78 / .93 / .98 -- a ~20-point MONOTONE ORDERED ladder produced by ratings that discriminate nothing at all.

(2) DEADLOCK-GATE CELL, cell 4 (pairs 4, 12). d = [+1,-1,-1,0,0,0,+1]: the two screen EXTREMES (.70, .70) favour A, the two mid-list heavyweights (.95, .86) favour B, three silent both-1. Because kA = kB = 2 the advocated gate opens to FULL force (g = 1): drive = V + mu*S = -.80 + 2.66 = +1.86, and with c ~ .56 it predicts P(A) ~ .73-.85 across its whole box. The competitor reads on, stops at pos2 with R ~ -.52 and predicts P(A) ~ .29. Opposite direction, ~50 points.

(3) VALIDITY-MATCHED SALIENCE SWAP AT |dk| = 1, cells 5 & 6 (pairs 5-6, 13-14). The residual-gate signature. Cell 5: A supported by pos0(.70), pos2(.86), pos3(.70), opposed by pos1(.95) and pos6(.70). Cell 6: the two .70 experts trade places (A supported by pos0, pos2, pos6; opposed by pos1, pos3). The validity multiset on each side is IDENTICAL, the silence composition is identical (two both-0), so the vote term V = +.25 is identical and any validity-based model must be flat. The advocated model's constant residual salience (g_res ~ .25, mu ~ 1.8) still reads the screen ends: S goes from -.12 to +1.68, so P(A) goes from ~.55 to ~.76 (delta logit ~ .8-1.0 across mu, g_res, tau). The competitor's position attention is primacy-shaped and nearly identical at pos3 (w = .382) and pos6 (w = .370), and its relative evidence gives P(A) ~ .54 vs ~ .53 -- a ~1-point null, if anything in the opposite direction. A 20-point swap effect vs a flat line isolates SYMMETRIC (first-and-last) salience surviving as a constant bonus outside the deadlock regime.

(4) SUFFICIENCY-STOP vs FULL INTEGRATION, cell 7 (pairs 7, 15). d = [+1,+1,0,0,0,-1,-1]: pos0(.70) and pos1(.95) favour A, pos5(.78) and pos6(.70) favour B, three silent both-1. The competitor's running lead after pos1 (|E| = 1.92) clears the bar set by the strongest unread expert (1.05*.535 = .56), so it stops and commits categorically on R = +1: P(A) ~ .94. The advocated model is deadlocked (kA = kB = 2) so its gate opens, but the salience term cancels EXACTLY (e_sal at pos0+pos1 equals e_sal at pos5+pos6), leaving only the small vote V = +.32 discounted by c ~ .59: P(A) ~ .56. A ~38-point confidence gap in the same direction, driven purely by stopping-plus-normalisation vs. full graded integration -- and a cell where the advocated model's salience machinery is provably inert, so the gap cannot be attributed to position.

(5) FOUR-AGAINST-THE-ORACLE CELL, cell 8 (pairs 8, 16). d = [-1,-1,+1,+1,+1,0,+1]: the leading .70 expert AND the oracle .95 expert favour B, while four later experts (.86,.70,.60,.70) favour A; only ONE expert is silent, so both models' silence machinery is essentially off and cannot be blamed. The competitor accumulates -1.92 over the first two positions, clears the .86 bar and commits on R = -1: P(A) ~ .05. The advocated model's near-linear vote (V = +.40 to +.53 across rho) plus the saturating majority bonus (|dk| = 2 -> M ~ .67) gives drive ~ 1.0-1.2 and P(A) ~ .75-.82. Opposite direction, ~70 points, over both full parameter boxes.

DIAGNOSTIC YIELD: (i) a three-cell forced-null triad where the competitor is pinned to one number and the advocated theory predicts a 20-point ordered ladder produced by non-discriminating ratings; (ii) a validity-matched position swap (cells 5 vs 6) that isolates symmetric residual salience against a primacy-only, fully normalising alternative, where the competitor is again forced to a null; (iii) two full direction disagreements (cells 4 and 8, 50 and 70 points) covering both the deadlock regime and the majority-bonus regime; (iv) one same-direction magnitude dissociation (cell 7, 38 points) in which the advocated model's salience term cancels by construction. Every pair has at least one discriminating cue; no cell is degenerate.

[12] GOAL: dissociate the ADVOCATED attention-lottery model (asymmetric screen salience with tau~0.85 so only rows 0,1,6,7 are really visible; authority-contingent first-mismatch capture gated by the most diagnostic DISCRIMINATING row only; limited-sample comparison of k=2-3 rows; panel-volume confidence beta_eff=beta*(n/6)^zeta; and above all the NEW commitment that the panel-informativeness judgement is CATEGORICAL, a step at f0~0.388 with kappa~100, with mutual endorsements counted at omega~0.84) from the COMPETING reason-counting model (near-linear validity vote over ALL discriminating experts, categorical unanimity bonus ups~1.5, saturating majority bonus lam~0.67, symmetric first/last-row salience entering at the constant residual g_res~0.25 whenever the vote is not deadlocked, a SHALLOW silence sigmoid sil_k~38 centred at 0.36, floor c_min~0.10, and a first-cue commitment whose authority cap is computed over ALL later rows including silent ones).

n=8, validities by SCREEN position [.82,.60,.95,.88,.72,.58,.68,.76]: the two heavyweights (.95,.88) sit mid-list at pos2/pos3 where the advocated model's salience is ~0.10/0.03 of pos0's, the screen extremes are mid-high (.82 at pos0, .76 at pos7), and the weakest experts (.60,.58) fill pos1/pos5. All displays are deliberately DENSE (>=4 discriminating rows): with n=8 any panel with 5+ mute rows is voided by BOTH theories, so sparse cells are uninformative by construction. Because no cell is vote-deadlocked, the competitor's salience term always sits at its constant residual g_res, which makes its predictions tight. 8 base cells, each mirrored across sides (16 unique pairs, K=6 -> 96 trials); mirroring forces an exact 8A/8B split for both theories.

(1) EXT (pairs 1-2): A=[1,1,0,0,0,1,0,1] vs B=[0,1,1,1,1,1,1,0]. The two screen extremes (.82,.76) favour A; the four mid-list rows (.95,.88,.72,.68) favour B; 2 mute rows are mutual endorsements (f_sil=.21, both theories at full confidence). Advocated: att(pos0)=1.57 and att(pos7)=0.32 against 0.13 for ALL four mid rows together, so the 3-row lottery almost never sees the mid-list -> P(A)~.93. Competitor: V=-1.35, majority bonus -0.67, residual salience +0.62 -> P(A)~.22. ~70-point DIRECTION disagreement over both full parameter boxes.

(2) AUTH (pairs 3-4): A=[1,0,1,1,0,1,0,0] vs B=[0,1,1,1,1,1,1,1]. pos0 (.82) alone favours A against four lower-validity rows (.60,.72,.68,.76); crucially the .95 and .88 experts are MUTE (mutual endorsements, f_sil=.315). This isolates the two theories' different authority gates: the advocated model's first-mismatch capture asks only whether pos0 is the most diagnostic DISCRIMINATING row (it is: ratio 1.16 -> q_auth~.83, capture factor ~6.5, att(pos0)=4.7 vs 0.5 for all four opponents) -> P(A)~.90; the competitor's commitment gate is computed against the max validity of ALL later rows, i.e. the silent .95 expert, so q_com~.07 and it falls back on the vote (V=-0.94, M=-0.67, salience -0.37) -> P(A)~.18. Again ~70 points, opposite direction, and the mechanism is the silent-vs-speaking composition of the gate.

(3) TAIL (pairs 5-6): A=[0,1,1,1,0,1,1,1] vs B=[1,1,1,1,1,0,0,0]. pos0 (.82) and pos4 (.72) favour B; the tail rows pos5/pos6/pos7 (.58,.68,.76) favour A; f_sil=.315. Validities were chosen so the competitor's vote is almost exactly null (V=-0.02 to +0.01 across rho in [.88,1.02]) and only its residual salience tilts it (+0.18 to +0.30), giving P(A)~.51-.55, i.e. a coin flip. The advocated model captures on pos0 (q_auth~.83, att 4.7 vs 0.41 for the whole tail) and predicts P(A)~.08. This cell also reverses the side that pos0 favours relative to cells 1-2, so 'follow the top row' cannot be confounded with a side bias.

(4-5) SILENCE-COMPOSITION CONTRAST at fixed d, SIL-LO (pairs 7-8) vs SIL-HI (pairs 9-10). Identical difference vector (pos0,pos2,pos7 favour A; pos4,pos6 favour B); the three mute rows are mutual ENDORSEMENTS in SIL-LO (f_sil=.84*3/8=.315) and mutual ABSTENTIONS in SIL-HI (f_sil=.375). Both values lie BELOW the advocated step (f0 in [.380,.396]) but straddle the competitor's shallow sigmoid centre (.36): advocated c goes 1.00 -> .81 (P .95 -> .92, and >=.87 anywhere in its box), whereas competitor c goes .85 -> .37 (P .88 -> .71). A ~17-point graded drop vs a ~3-point near-null, produced entirely by ratings that discriminate nothing, is the direct test of categorical-vs-graded panel informativeness.

(6-7) INVISIBLE-DISSENTER / UNANIMITY CONTRAST at fixed silence, UNI41 (pairs 11-12) vs UNI50 (pairs 13-14). Both have the same three mute both-0 rows (f_sil=.375) and the same five speaking experts (pos0,pos1,pos2,pos6,pos7); the ONLY change is the .95 expert at pos2 flipping from favouring B (4-1) to favouring A (5-0). The advocated model's lottery samples pos2 with probability ~.005 (its salience is 0.096 while pos0 carries the capture bonus), and its silence/vbar terms are identical because the speaker set is identical, so it is FORCED to predict the same probability in both cells (P(A)~.94 in each, difference <1 point). The competitor gains the categorical unanimity bonus (+1.5) plus ~1.9 of vote, moving from P(A)~.76 to ~.91. A 15-point within-subject jump where the advocated theory predicts an exact null.

(8) VOID (pairs 15-16): A=[1,0,1,1,0,0,0,1] vs B=[0,0,0,0,0,0,0,0]. Four experts (incl. the .95 and .88) unanimously endorse A, four endorse neither (f_sil=.50), i.e. the strongest verdict a half-mute panel can deliver. Both theories judge the panel uninformative, but the advocated model voids it essentially completely (c floored at .015-.055) and therefore predicts a LITERAL coin flip, P(A)=.51-.55 across its whole box, despite a 4-0 landslide, whereas the competitor's floor c_min~.10 multiplied by a large drive (V=2.89 + unanimity 1.5 + majority .67 + salience 1.01) leaves a reliable residual tilt, P(A)~.62-.72. Paired with UNI50 (same all-zero competitor, one more mute row) this is a 41-point cliff for the advocated model vs 26 points for the competitor, and the absolute level at VOID (.53 vs .67) is the decisive statistic.

DIAGNOSTIC YIELD: three full direction/level disagreements (EXT, AUTH ~70 points each; TAIL, near-determinism vs chance, ~43 points); one forced null for the competitor's silence sigmoid turned into a 17-point graded drop (SIL-LO vs SIL-HI); one forced null for the advocated lottery turned into a 15-point unanimity jump (UNI41 vs UNI50); and a coin-flip-vs-residual-tilt test of the two floors (VOID). Every pair has at least one discriminating cue and no cell is degenerate.

[13] GOAL: dissociate the ADVOCATED reason-counting model (near-linear validity vote over ALL discriminating experts, SYMMETRIC first/last-row salience that is at FULL force only when the reason count is deadlocked and at a constant residual g_res~.25 otherwise, saturating majority bonus lam~.67, categorical unanimity bonus, authority cap computed over ALL later rows INCLUDING silent ones) from the COMPETING attention-lottery model (ASYMMETRIC salience with top >> bottom, huge authority-contingent first-mismatch capture gated only by later SPEAKING rows, limited sample of k=2-3 rows, no unanimity/majority bonuses, compressive ceiling ~.90).

VALIDITIES BY SCREEN POSITION: [.74, .95, .62, .86, .58, .74]. n=6 makes the competitor's panel-volume term exactly neutral (beta*(6/6)^zeta = beta). The two SCREEN EXTREMES are given IDENTICAL validity (.74) - this is the key design move (see SWAP below). The oracle (.95) sits at pos1 and the second heavyweight (.86) at pos3, i.e. in the rows the lottery can barely see (competitor attention s*x = [.60,.29,.04,.06,.05,.30] before capture; advocated salience is the symmetric U [1.01,.39,.19,.19,.39,1.01]). Because pos1 (.95) is always present later in the list, the advocated model's first-cue commitment is uniformly suppressed whenever pos0 speaks (ratio .48/(1.05*.90)=.51 -> q_com<=.07 even at its worst corner), so it cannot contaminate any contrast. Every cell has >=4 speaking rows and <=2 silent rows, and all silent rows are MUTUAL ENDORSEMENTS, so f_sil<=.33 for both theories and both silence gates are provably OFF everywhere - no result can be blamed on silence machinery.

8 base cells, each mirrored across sides -> 16 unique pairs, K=6 -> 96 trials; mirroring forces an exact 8A/8B split for BOTH theories.

(1) FLAGSHIP FORCED NULL: TOP/BOTTOM SWAP, cells SWAP-A (pairs 9-10) and SWAP-B (pairs 11-12). SWAP-A has d=[+1,-1,0,+1,0,-1], SWAP-B has d=[-1,-1,0,+1,0,+1]: the two extremes simply trade sides while pos1 and pos3 are held fixed. Because pos0 and pos5 have EQUAL validity and EQUAL (symmetric) salience, the advocated model's vote V=-.173 and its salience sum S=-.201 are ANALYTICALLY IDENTICAL in the two cells for EVERY (rho, mu, tau, sigma, g_res) - both are vote-deadlocked (kA=kB=2, g=1), so it is mathematically forced to predict the same P(A)~.35 in both (residual <=3-6 points from the suppressed commitment term). The lottery is anchored on the top row (att(pos0)/att(pos5) = 1.6-2.6 from gamma alone, times a >=1.5 first-mismatch factor) and therefore REVERSES: P(A)~.71 in SWAP-A vs ~.20 in SWAP-B, a ~50-point swing. This isolates symmetric-vs-asymmetric salience with a forced null on one side.

(2) DEADLOCK-GATE REVERSAL, cells D1 (pairs 1-2) and D2 (pairs 3-4). D1: d=[+1,-1,0,-1,0,+1] - the two extremes (.74,.74) favour A, the two mid heavyweights (.95,.86) favour B, kA=kB=2, so the advocated gate opens to FULL force (drive = -.64 + mu*1.44 = +1.98) -> P(A)~.89. D2 adds ONE extra weak dissenter at pos2 (.62): the count is no longer deadlocked, the gate collapses to g_res=.25 and the drive flips to -.33 -> P(A)~.41, a 48-point DIRECTION REVERSAL produced by a single low-validity mid-list row. The lottery barely samples pos2 (att .04 out of ~1.7) and is forced to a near-null: P(A) ~.81 -> ~.75. A large within-subject reversal vs an approximate null.

(3) INVISIBLE-MAJORITY CONTRAST, cells V1 (pairs 5-6) and V2 (pairs 7-8). Both have pos0 (.74) favouring A and pos5 (.74) favouring B; the ONLY change is the direction of the two nearly invisible mid-list heavyweights (pos1 .95, pos3 .86). V1 (they oppose A): advocated V=-1.64 plus majority bonus -.67 -> P(A)~.08; V2 (they back A): drive +2.57 -> P(A)~.95, i.e. an ~87-point within-subject swing. The lottery's sampling probability for pos1/pos3 is ~.21 combined and its evidence is normalised within a 2-row sample, so it predicts only ~.70 -> ~.81 (11 points) and, crucially, chooses A in BOTH cells: V1 alone is a ~62-point DIRECTION disagreement.

(4) AUTHORITY-GATE CELL, AUTH (pairs 15-16): d=[+1,0,-1,0,-1,-1] with pos1 (.95) and pos3 (.86) MUTE (mutual endorsements). pos0 (.74) is the most diagnostic SPEAKING row, so the lottery's authority-contingent capture fires almost fully (q_auth~.5-1 -> att(pos0)=2.7 of 3.1) and it follows the top row: P(A)~.82-.85. The advocated model's authority cap is computed against ALL later rows including the silent .95 expert, so q_com~.02; it counts three cheap reasons plus the saturating majority bonus and chooses the other option: P(A)~.14-.20. ~65 points, opposite direction, over both full parameter boxes, with the silence gate provably off.

(5) CEILING ANCHOR, LAND (pairs 13-14): d=[+1,+1,0,+1,0,+1], a 4-0 landslide. The advocated model adds unanimity (+1.5) and majority (+.67) bonuses to a large vote -> P(A)~.96-.97; the lottery's small-sample read-out is capped at sigmoid(beta_eff)~.89 no matter how lopsided the panel -> P(A)~.87-.89. This cell also serves as an engagement check and calibrates the compressive-ceiling claim.

DIAGNOSTIC YIELD: (i) one analytic forced null for the advocated theory (SWAP-A vs SWAP-B) against a ~50-point reversal for the lottery; (ii) one near-null for the lottery (D1 vs D2) against a 48-point direction reversal for the advocated theory; (iii) an 87-vs-11-point invisible-majority contrast plus two outright direction disagreements (V1, AUTH, ~60-65 points each); (iv) a ceiling test of limited-sample compression. No cell is degenerate and no contrast depends on either model's silence machinery.

[14] GOAL: dissociate NOISY COMPENSATORY REASON-COUNTING WITH TOP-DOWN READING ATTENTION (advocated: cue weights multiplied by att_j = 1+alpha_p*exp(-j/tau_p), so screen-TOP rows are worth up to ~1.8x themselves and by row 4 the bonus is gone; Gaussian-only edge gate that collapses to ~0.08-0.11 as soon as the reason counts are not deadlocked; GRADED, REVERSIBLE silence inference c = 1-eta*f_sil^q*vbar that may pass NEGATIVE; no unanimity term; lapse eps~.16 capping P in [.12,.88]) from REASON-COUNTING WITH A TWO-REGIME SALIENCE GATE (competitor: position enters only as a SYMMETRIC first/last-row salience with a CONSTANT residual g_res~.25; no attention multiplier on the vote; categorical unanimity bonus ups~1.5; SHARP silence sigmoid centred at f0~.36 with mutual endorsements down-weighted by omega~.82; c FLOORED at c_min~.10; eps~.035).

VALIDITIES BY SCREEN POSITION: [.72, .80, .95, .62, .58, .80, .72]. Two design constraints drive this layout. (a) pos0 and pos6 have IDENTICAL validity (.72) and pos1 and pos5 have IDENTICAL validity (.80); since the competitor's salience is symmetric (s_sal = [1.00,.37,.15,.10,.15,.37,1.00]), any exchange of the sign pattern of {pos0,pos1} with {pos5,pos6} leaves its vote V, its salience sum S and its reason counts ANALYTICALLY unchanged. The advocated model's primacy multiplier (att = [1.65,1.28,1.12,1.05,1.02,1.01,1.00] at nominal parameters) does not. (b) The near-oracle (.95) sits mid-list at pos2, so whenever pos0 or pos1 is the first discriminating row both models' authority-capped first-cue commitment is provably suppressed (ratio <= .64 -> q_com <= .014 for the competitor, <= .002 for the advocated), and it cannot contaminate any contrast.

8 base cells, each mirrored across sides -> 16 unique pairs, K = 6 -> 96 trials. Mirroring forces an exact 8A/8B split for BOTH theories, so side bias cannot mimic either model.

(1) FLAGSHIP FORCED NULL #1 - TOP/BOTTOM READING SWAP (pairs 1-4). SWAP-TOP d = [+1,+1,0,+1,-1,-1,-1]; SWAP-BOT d = [-1,-1,0,+1,-1,+1,+1]. The two designs differ ONLY by exchanging the signs of the two TOP rows with the two BOTTOM rows (validity-matched pairwise), with pos2 a mutual endorsement and pos3/pos4 held fixed so that both models sit almost exactly at drive ~ 0 at the midpoint. Counts are 3-3 in both (deadlock, so both gates are at full force and both majority bonuses are off). For the competitor V = x3 - x4 = +.082 and S = -.054 in BOTH cells, so it is MATHEMATICALLY FORCED to predict the same number, ~.50 (computed .501 vs .490 including the 1.4% commitment residue), for every (rho, mu, sigma, g_res, tau, lam, ups, beta) in its box. The advocated model's reading-attention multiplier makes V = +.556 vs -.372, i.e. P(A) = .615 vs .384 at nominal parameters and a 14-34 point swing across its ENTIRE (alpha_p, tau_p, rho, beta) box. A ~23-point within-subject swing where the competitor predicts an exact null isolates graded primacy attention against symmetric edge salience.

(2) FLAGSHIP FORCED NULL #2 - REVERSIBLE SILENCE / LONE-ORACLE CELLS (pairs 9-12). Only pos2 (the .95 expert) discriminates, favouring A; the other six rows are mutual ABSTENTIONS (ORACLE-0) or mutual ENDORSEMENTS (ORACLE-1). Because the lone speaker has vs = .90 and f_sil = 6/7, the competitor's sharp sigmoid is fully saturated in BOTH cells (omega*6/7 >= .617 >> f0 <= .375) and eta*vbar >= 1.3 > 1, so its c hits the floor c_min in BOTH: it is FORCED to predict the identical value, P(A) ~ .53-.63 (nominal .573), i.e. mildly following the oracle, in both cells. The advocated model's power-law, reversible silence term gives c = max(1-eta*(6/7)^q*.90, c_neg) < 0 GUARANTEED for the abstention version (the minimum of eta*f^q*vbar over its box is 1.21 > 1) but c ~ +.19 for the endorsement version, so it predicts P(A) ~ .27 (choosing AGAINST the lone near-oracle endorsement) vs ~ .64 - a ~36-point within-subject reversal that crosses chance, produced entirely by ratings that discriminate nothing. Any systematic below-chance following of the lone .95 expert is impossible for the competitor at any parameter setting.

(3) COMPETITOR-ONLY SILENCE CLIFF (pairs 5-8). Identical difference vector d = [+1,+1,0,-1,0,0,+1] (4 speakers, 3-1 for A, so the majority bonus is saturated and identical), with the three silent rows mutual endorsements (SIL-HI) or mutual abstentions (SIL-LO). n = 7 with 3 silent rows is chosen precisely because the competitor's effective silence fraction straddles its threshold (omega*3/7 = .351 vs 3/7 = .429, with f0 in [.345,.375]): its c goes .70 -> .33 and P(A) .909 -> .754, a ~16-point ordered drop. The advocated model's smooth f_sil^q term moves c only .931 -> .877, P(A) .884 -> .877, a <1-point null. This is the mirror image of contrast (2): here the competitor must move and the advocated must not.

(4) UNANIMITY-BONUS / INVERTED-LADDER CELL (pairs 13-14). Only pos4 (the WEAKEST expert, .58) discriminates, six rows mutually abstain. The competitor adds its categorical unanimity bonus (+1.5) to a panel whose low vbar keeps c high (.75), predicting a confident P(A) ~ .80; the advocated model has no unanimity term, and its tiny lone reason (xw = .18) read out through a mild silence discount gives P(A) ~ .545. A 26-point level gap on a single-cue cell. Paired with ORACLE-0 this also gives the two models different INVERTED ladders: the advocated predicts the lone .58 expert is followed at .545 while the lone .95 expert is followed at .27 (crossing chance), whereas the competitor predicts .80 vs .57 (both above chance).

(5) CEILING / LAPSE ANCHOR (pairs 15-16). A 7-0 landslide with no silent rows: both silence gates are provably off and both models are at their maximum. The competitor's small lapse (eps ~ .035) puts it at ~.98; the advocated model's large fixed lapse (eps ~ .13-.19) caps it at ~.92. This cell calibrates the read-out ceiling, serves as an engagement check, and makes the absolute levels in cells (1)-(4) interpretable.

DIAGNOSTIC YIELD: two analytically forced nulls for the competitor (a 23-point reading-order swing and a 36-point silence-composition reversal that crosses chance), one forced near-null for the advocated model where the competitor must drop 16 points, one 26-point unanimity level gap, and a 6-point ceiling contrast. Every pair has at least one discriminating cue; no cell is degenerate; every contrast is within-subject and mirrored across sides.

[15] GOAL: dissociate the ADVOCATED two-regime salience-gate reason-counter (SYMMETRIC first/last-row salience that acts at full force only on a deadlocked vote and otherwise persists as a CONSTANT residual g_res~.25; categorical unanimity bonus ups~1.5; SHARP silence threshold at f0~.36 with mutual endorsements down-weighted by omega~.82; confidence FLOORED at c_min~.10; small lapse eps~.035) from the COMPETING top-down reading-attention tally (cue weights multiplied by att_j = 1+alpha_p*exp(-j/tau_p), so the TOP row is worth ~1.65x itself and the bonus is gone by row 4; Gaussian-only edge gate that collapses to ~.08-.11 once the counts are not deadlocked; GRADED, REVERSIBLE silence term c = 1-eta*f_sil^q*vbar that can pass NEGATIVE; no unanimity term; large lapse eps~.16).

LAYOUT: n=7, validities by SCREEN position [.78,.95,.62,.78,.62,.95,.78] 
- deliberately PALINDROMIC, which is the central design move. Because the validity profile is mirror-symmetric and the advocated model's salience s_sal = exp(-j/tau)+exp(-(n-1-j)/tau) = [1.00,.375,.153,.0996,.153,.375,1.00] is also mirror-symmetric, the advocated model is ANALYTICALLY INVARIANT to top/bottom exchanges of validity-matched rows, while the competitor's primacy multiplier ([1.65,1.28,1.12,1.05,1.02,1.01,1.00]) is not. Three rows share v=.78 (pos0 = top edge, pos3 = exact middle, pos6 = bottom edge) so an edge and the middle can be swapped at matched validity; pos1 and pos5 share v=.95 and pos2/pos4 share v=.62. The near-oracle at pos1 (never pos0) keeps the authority-capped first-cue commitment suppressed in the dense cells (ratio .56/(1.05*.90)=.59 -> q_com=.04 advocated, ~.002 competitor), so no contrast is contaminated by the commitment term.

8 base cells, each mirrored across sides -> 16 unique pairs, K=6 -> 96 trials; mirroring forces an exact 8A/8B split for BOTH models, so side bias cannot mimic either.

(1) FLAGSHIP TRIAD X / Y / Z (pairs 1-6). Identical background (pos1->B, pos4->A, two silent mutual endorsements at pos2/pos5) and identical validity multisets; only the sign pattern of the three matched .78 rows changes: X = both EDGES favour A, MIDDLE favours B; Y = top+middle favour A, bottom favours B; Z = middle+bottom favour A, top favours B. All three have kA=3, kB=2 (dk=1), so the advocated gate sits at its residual and the competitor's at .114, and the advocated's vote V is ANALYTICALLY IDENTICAL (-.073) in all three. Advocated salience sums: X = +1.683, Y = Z = -0.122 -> P(A) ~ .70 / .48 / .45, i.e. a large EDGE effect (X-Y = +.22) and an exact TOP-vs-BOTTOM null (Y-Z = +.03, residue of the 4% commitment term only). Competitor: the attention multiplier makes the top row worth 1.65x and the bottom row 1.00x, so V = +.024 / +.080 / -.664 -> P(A) ~ .59 / .51 / .33, i.e. a small edge effect (X-Y = +.08) and a LARGE top/bottom asymmetry (Y-Z = +.18). Two orthogonal contrasts with opposite profiles: symmetric-residual salience predicts X>>Y=Z, primacy attention predicts X~Y>>Z.

(2) LONE-ORACLE SILENCE-COMPOSITION PAIR (pairs 7-10), direction disagreement + forced null. Only pos1 (the .95 expert) discriminates, favouring one option; the other six rows are mutual ABSTENTIONS (ORACLE-0) or mutual ENDORSEMENTS (ORACLE-1). For the advocated model f_sil = 6/7 or .82*6/7, both far above the f0~.36 step, and eta*vbar >= 1.45*.90 > 1, so c is FLOORED at c_min in BOTH cells: it is mathematically forced to predict the same value in both, and (because of the categorical unanimity bonus, kB=0) that value is reliably ABOVE chance, P(A) ~ .54-.63 for its entire box. The competitor's reversible power-law term gives c < 0 for the abstention version at every parameter setting (eta*f^q*vbar >= 2.1*.646*.90 = 1.22 > 1), so it predicts choosing AGAINST the lone near-oracle, P(A) ~ .15-.42 (nominal .31), while the endorsement version lifts c to ~+.19 and it predicts P(A) ~ .61. So: a ~27-point DIRECTION disagreement on ORACLE-0, plus a ~30-point within-subject reversal for the competitor where the advocated theory predicts an exact null.

(3) LONE-WEAK-EXPERT CELL (pairs 11-12). Only pos2 (the weakest .62 expert) discriminates, six mutual abstentions. The advocated model's unanimity bonus (+1.5) plus a high c (low vbar keeps eta*vbar small: c ~ .63) predicts a confident P(A) ~ .79; the competitor has no unanimity term and a similar c, predicting only ~.56. A 22-point level gap on a single-cue display, and (paired with ORACLE-0) two different INVERTED ladders: advocated .79 (lone .62) vs .57 (lone .95), competitor .57 vs .31.

(4) SILENCE-COMPOSITION CONTRAST AT FIXED d (pairs 13-16). Identical difference vector d = [+1,-1,0,0,0,+1,+1] (kA=3, kB=1, so both majority bonuses are saturated and identical) with the THREE silent rows as mutual endorsements (SIL-END) or mutual abstentions (SIL-ABS). n=7 with 3 silent rows is chosen precisely because the advocated model's effective silence fraction straddles its sharp threshold (omega*3/7 = .351 vs 3/7 = .429 against f0 in [.345,.375]): its c collapses from ~.51 to the floor .10, so P(A) drops from ~.83 to ~.58, a ~25-point ordered drop produced by ratings that discriminate nothing. The competitor's smooth f^q term moves c only .886 -> .795, P(A) .85 -> .83, a ~2-point null. This is the exact mirror image of contrast (2): here the advocated model must move a lot and the competitor must not.

DIAGNOSTIC YIELD: (i) an analytic top/bottom null for the advocated model against an 18-point primacy swing for the competitor, together with the complementary 22-vs-8-point edge effect (cells X/Y/Z); (ii) a full direction disagreement on ORACLE-0 (above vs below chance) plus a forced null for the advocated where the competitor must reverse by 30 points; (iii) a 25-vs-2-point silence-composition cliff in the opposite direction; (iv) a 22-point unanimity level gap. Every pair has at least one discriminating cue, every contrast is within-subject, and each cell is mirrored across sides.

[16] GOAL: dissociate DCRI-3s (advocated: compressive diagnosticity with an oracle premium, sqrt-numerosity, RELATIVE read-out drive=(V+N+P)/(kappa2+Xs), smooth power-law silence discount with omega=0.63, steep edge salience with g_res~0.10, strong commitment read-out beta*d_com~7c, difficulty-scaled lapse up to eps~0.30) from the two-regime salience-gate reason counter (competitor: near-linear vote, ABSOLUTE drive, categorical unanimity +1.5 and saturating majority +0.67, constant residual salience g_res~0.25 with mu*g in absolute units, SHARP silence sigmoid at f0~0.36 with omega=0.82 and floor c_min~0.10, small fixed lapse 0.035, weaker commitment read-out beta*d_com~3c).

LAYOUT: n=6, validities by SCREEN position [.74,.62,.96,.66,.86,.74]. The near-oracle (.96, above the advocated model's v_knee~.91, so its top-end premium fires) sits at pos2, deep in the display where BOTH salience profiles are minimal (adv s=.088, comp s=.206), and the second heavyweight (.86) sits at pos4. pos0 and pos5 are validity-matched (.74) AND salience-matched, so edge-vs-interior contrasts can be built without confounding validity; pos2 and pos3 are also salience-matched, so the lone-oracle cells carry ZERO salience term for both models. Because the oracle sits at pos2, any cell in which pos0 or pos1 speaks has its authority-checked commitment suppressed for both models (ratio <=.51 -> q_com<=.02), so commitment contaminates only the two cells where it is deliberately the target.

8 base cells, each mirrored across sides -> 16 unique pairs, K=6 -> 96 trials; mirroring forces an exact 8A/8B split for both theories.

(1) LONE-ORACLE SILENCE-COMPOSITION PAIR (cells A and B, pairs 1-4): identical difference vector d=[0,0,+1,-1,0,0] (the .96 expert vs the .66 expert, salience exactly cancelling, deadlocked counts), with the four mute rows mutual ENDORSEMENTS (A) or mutual ABSTENTIONS (B). For the competitor f_sil = .82*4/6 = .547 or 4/6 = .667; BOTH saturate its sharp sigmoid (even at omega=.72 the effective fraction .48 is far above f0<=.375) and eta*vbar = 1.45-1.65 * .76 > 1, so c is FLOORED at c_min in both cells: it is mathematically forced to predict the identical value, ~.56, in both. The advocated model's smooth power-law term gives gap ~.11 (endorsements) vs 0 (abstentions), and its much stronger commitment read-out (sigmoid(6.98c) vs sigmoid(3.0c) on the oracle-first row, q_com~.87) converts that into P(A) ~ .69 vs ~ .60. So: a within-subject forced null for the competitor against a ~9-point ordered drop for the advocate, plus a ~12-point LEVEL gap on cell A.

(2) DENSE-MAJORITY CEILING CELL (cell C, pairs 5-6): 4-1 for A with only one mute row. The competitor's absolute read-out (V'=2.08 + majority .67 + residual salience .92, c~1) puts it at P(A)~.97; the advocate's relative read-out (drive = 4.39/4.34 = 1.01) plus its difficulty-scaled lapse caps it at ~.89. An ~8-point gap in the OPPOSITE direction from cells A/B.

(3) EDGE-DEADLOCK CELL (cell E, pairs 7-8): the two screen extremes (.74,.74) favour A, the two deep-interior rows (.96,.66) favour B, counts deadlocked so BOTH gates open fully. Both models choose A (salience overturns the vote), but the competitor is at ~.94 while the advocate, dividing by total speaking diagnosticity, lands at ~.83. Same direction, ~11 points, with no silence machinery doing the work.

(4) NUMEROSITY-AT-dk=1 CELL (cell F, pairs 9-10) - the direction-split cell: three cheap/mid reasons (.74,.62,.66) against two better ones (.86,.74), dk=+1 so the competitor's majority bonus is OFF and its near-linear vote is -.135 (net salience +.21) -> P(A)~.50, i.e. a coin flip or slightly against A. The advocate's compressive weights make the vote +.11 and, decisively, its sqrt-numerosity term adds +.56 with no counterpart in the competitor -> P(A)~.59. Predictions straddle chance.

(5) UNANIMITY CONTRAST (cells G1, G2, pairs 11-14): G1 = the two .74 edge experts for A against the .62 expert (dk=1, three mute endorsements); G2 = the same two supporters with the .62 dissenter turned mute (2-0 unanimous, four mute endorsements). The competitor fires its categorical unanimity bonus (+1.5) plus the saturating majority bonus (+.67), predicting a ~9-point rise (.67 -> .77) despite the extra silence; the advocate has no unanimity term, its numerosity grows only from .56 to .79, and its own silence discount eats the gain, predicting a ~2-point NULL (.78 -> .80). Note also the level reversal: on G1 the advocate is ~10 points ABOVE the competitor, on G2 they nearly coincide.

(6) CHEAP-COALITION vs ORACLE AT dk=2 (cell H, pairs 15-16): three cheap/mid reasons (.74,.62,.66) against the lone .96 oracle, two mute endorsements. Here the competitor's majority bonus fires (+.67) on top of a positive vote and a large residual salience term, giving a decisive P(A)~.84; the advocate's oracle premium (x=1.46 vs .92 for a .86 expert) keeps the vote nearly balanced (+.25), and after relative normalisation, the .63 confidence and the difficulty-scaled lapse it predicts only ~.64. The largest single gap in the design, ~19 points, and in the opposite direction from cells A/B/F/G1.

DIAGNOSTIC YIELD: the signature is a CROSSOVER in the sign of the between-model difference across cells - the advocate is relatively MORE decisive than the competitor in sparse/heavily-mute displays that turn on commitment (A: +12, G1: +10) and relatively LESS decisive in dense, high-mass displays (C: -8, E: -11, H: -19) because its read-out is relative and its lapse is difficulty-scaled; plus one analytic forced null for the competitor (A vs B) where the advocate must show an ordered silence-composition drop, one forced near-null for the advocate (G1 vs G2) where the competitor must show a unanimity jump, and one cell (F) where the two predictions straddle chance. No parameter setting of either model can reproduce both the sparse-cell ordering and the dense-cell ordering. Every pair has at least one discriminating cue; no cell is degenerate; every cell is mirrored across sides.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "GOAL: dissociate the ADVOCATED two-regime reason-counter (near-linear validity vote rho~0.95, ABSOLUTE drive read out with beta~1.2, categorical unanimity bonus ups~1.5, SATURATING majority bonus lam~0.67 that is identical for |dk|=2 and |dk|=5, symmetric first/last-row salience with a BROAD tau~1.0 that persists as a constant residual g_res~0.25 whenever the vote is not deadlocked, SHARP silence sigmoid at f0~0.36 with mutual endorsements weighted omega~0.82, confidence floored at c_min~0.10, small lapse eps~0.035) from DCRI-3s (compressive rho~0.52 with an oracle premium above v_knee~0.91, sqrt numerosity lam~0.56, RELATIVE read-out drive=(V+N+P)/(kap2+Xs) with beta~3.1, steep salience tau~0.75 with g_res~0.10, SMOOTH power-law silence discount with omega~0.63, and a difficulty-scaled lapse up to eps~0.30 that compresses every near-chance display).\n\nLAYOUT: n=7, validities by SCREEN position [0.74, 0.60, 0.96, 0.74, 0.86, 0.62, 0.74]. Design choices: (i) the near-oracle 0.96 sits at pos2, above DCRI's knee so its top-end premium fires (x=1.46 vs 0.92 for the .86 expert; the advocated model rates them 0.92 vs 0.73), and deep enough that its salience is negligible for both models; (ii) THREE experts share validity 0.74 and sit at pos0 (top edge), pos3 (exact middle) and pos6 (bottom edge), which permits a validity-matched EDGE<->MIDDLE swap that leaves the vote of ANY validity-only model exactly unchanged; (iii) the cheap experts (.60, .62) sit at pos1/pos5, which are salience-matched; (iv) in every cell the first discriminating row is pos0 or pos1, whose validity (.48 or .20 in vs units) is far below the later oracle (.92), so BOTH models' authority-checked commitment term is suppressed (q_com <= 0.02) and cannot contaminate any contrast.\n\n8 base cells, each mirrored across sides -> 16 unique pairs, K=6 -> 96 trials; mirroring forces an exact 8A/8B split for BOTH theories, so side bias cannot mimic either model. All mute rows are mutual endorsements (1,1), which is exactly where the two silence rules differ (omega .82 vs .63).\n\n(1) UNANIMITY / SILENCE-CLIFF PAIR - cells 2 and 3 (pairs 2,3 + mirrors). Cell 2: pos0(.74), pos4(.86), pos6(.74) unanimously favour A, four rows mute endorsements. Cell 3 is IDENTICAL except the weak .62 expert at pos5 now DISSENTS (3-1). Advocated: in cell 2 the effective silence fraction .82*4/7=.469 is far above its sharp threshold, c collapses to ~.11 and the big drive (vote 1.73 + unanimity 1.5 + majority .67 + salience .98 = 4.9) is crushed to P(A)~.65; in cell 3 the panel becomes informative (.82*3/7=.351, just BELOW the step), c jumps to ~.65 and P(A) RISES to ~.90 even though a reason was subtracted. So the advocated theory predicts a paradoxical +9 to +24 point INCREASE from adding a dissenter, across its whole box. DCRI's smooth power-law discount plus its relative read-out offset almost exactly: drive falls 1.39->0.95 while c rises .39->.56, giving .801 -> .799, a forced ~zero change, AND in cell 2 DCRI is 15 points ABOVE the advocated model (.80 vs .65), i.e. the ordering of the two models REVERSES between the two cells. This is the flagship crossover.\n\n(2) VALIDITY-MATCHED EDGE<->MIDDLE SWAP - cells 4 and 5 (pairs 4,5 + mirrors). Both cells have the identical speaking-validity multiset {.74,.60,.86,.96,.74}, identical counts (3-2, dk=1, so both salience gates are at their residual), identical silence (two mute endorsements at pos5/pos6) and an identical vote V=+0.025: only WHICH of the two equally valid .74 experts is at the top edge (pos0) versus the exact middle (pos3) changes. Any validity-only model is analytically pinned. The advocated residual gate (mu*g_res=0.455 acting on a symmetric U-shaped, broad-tau salience profile) swings S from +1.28 to -0.53 and predicts P(A)=.67 vs .44 - a 23-point swing that CROSSES chance. DCRI's steeper profile is similar in shape but its salience enters a relative read-out (2.4*0.146*dS/(0.42+Xs)) and is then compressed by its difficulty-scaled lapse, giving .58 vs .52, only ~6 points and never crossing chance.\n\n(3) MAJORITY-BONUS / EDGE CELL PAIR - cells 6 and 7 (pairs 6,7 + mirrors). Cell 6: the two heavyweights pos2(.96) and pos4(.86) favour A, three cheap/mid experts pos1(.60), pos3(.74), pos5(.62) favour B (dk=-1, so the advocated majority bonus is OFF). Cell 7 adds ONE more cheap dissenter at the bottom EDGE, pos6(.74), making it 2-4. Advocated: .62 (A) -> .20 (B), a 42-point swing and a direction reversal, because the single added row simultaneously fires the saturating majority bonus (-0.67) and adds a full edge-salience unit (-0.46). DCRI: .51 -> .40, only ~11 points, because its sqrt numerosity grows slowly, its compression already makes cheap coalitions competitive, and its denominator grows with the extra speaker. Cell 6 alone is also a direction-vs-chance split: the advocated model follows the two heavyweights at .62 while DCRI sits at an exact coin flip (.507) - compression plus numerosity plus the difficulty lapse.\n\n(4) DENSE LANDSLIDE - cell 1 (pair 1 + mirror). Five experts (incl. .74,.60,.74,.86,.62) favour A, the oracle .96 and the bottom-edge .74 favour B, NO mute rows so neither silence rule does any work. Absolute vs relative read-out is isolated: the advocated model's drive (vote .78 + majority .67 + salience .39) at c=1 gives P(A)~.89, whereas DCRI divides an almost identical numerator by kap2+Xs=5.76 and then applies its lapse, giving ~.72. A ~17-point level gap in the cell where both models are maximally confident.\n\n(5) DEADLOCK-GATE CELL - cell 8 (pair 8 + mirror). The two screen extremes (.74,.74) favour A, the two interior heavyweights (.96,.86) favour B, counts deadlocked 2-2 so BOTH gates open fully. Both models overturn the validity vote and choose A, but the advocated model (mu*S=3.09 absolute, silence-discounted c=.55) predicts ~.82 while DCRI (2.4*S/(0.42+Xs) then lapse) predicts ~.74. Included as the anchor showing that the two models agree when the gate is fully open, so the 23-point swap effect in cells 4/5 cannot be attributed to a general position insensitivity of DCRI.\n\nDIAGNOSTIC YIELD: (i) a within-subject ordering CROSSOVER (cell 2: DCRI 15 points above the advocate; cell 3: the advocate 10 points above DCRI) driven by a dissenter that RAISES the advocated model's confidence and leaves DCRI flat; (ii) an analytic validity-matched position swap on which any validity-only account is pinned and where the advocate predicts a chance-crossing 23-point effect vs ~6 for DCRI; (iii) a single-row manipulation (cells 6->7) predicted to reverse the advocate by 42 points and move DCRI by 11; (iv) one 17-point dense-display level gap isolating absolute vs relative read-out plus lapse. Every pair has at least one discriminating cue, every contrast is within-subject, and each cell is mirrored across sides.",
  "validities": [
    0.74,
    0.6,
    0.96,
    0.74,
    0.86,
    0.62,
    0.74
  ],
  "trial_a_ratings": [
    [
      1,
      1,
      0,
      1,
      1,
      1,
      0
    ],
    [
      1,
      1,
      1,
      1,
      1,
      1,
      1
    ],
    [
      1,
      1,
      1,
      1,
      1,
      0,
      1
    ],
    [
      1,
      1,
      0,
      0,
      1,
      1,
      1
    ],
    [
      0,
      1,
      0,
      1,
      1,
      1,
      1
    ],
    [
      1,
      0,
      1,
      0,
      1,
      0,
      1
    ],
    [
      1,
      0,
      1,
      0,
      1,
      0,
      0
    ],
    [
      1,
      1,
      0,
      1,
      0,
      1,
      1
    ],
    [
      0,
      0,
      1,
      0,
      0,
      0,
      1
    ],
    [
      0,
      1,
      1,
      1,
      0,
      1,
      0
    ],
    [
      0,
      1,
      1,
      1,
      0,
      1,
      0
    ],
    [
      0,
      0,
      1,
      1,
      0,
      1,
      1
    ],
    [
      1,
      0,
      1,
      0,
      0,
      1,
      1
    ],
    [
      1,
      1,
      0,
      1,
      0,
      1,
      1
    ],
    [
      1,
      1,
      0,
      1,
      0,
      1,
      1
    ],
    [
      0,
      1,
      1,
      1,
      1,
      1,
      0
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      0,
      1,
      0,
      0,
      0,
      1
    ],
    [
      0,
      1,
      1,
      1,
      0,
      1,
      0
    ],
    [
      0,
      1,
      1,
      1,
      0,
      1,
      0
    ],
    [
      0,
      0,
      1,
      1,
      0,
      1,
      1
    ],
    [
      1,
      0,
      1,
      0,
      0,
      1,
      1
    ],
    [
      1,
      1,
      0,
      1,
      0,
      1,
      1
    ],
    [
      1,
      1,
      0,
      1,
      0,
      1,
      1
    ],
    [
      0,
      1,
      1,
      1,
      1,
      1,
      0
    ],
    [
      1,
      1,
      0,
      1,
      1,
      1,
      0
    ],
    [
      1,
      1,
      1,
      1,
      1,
      1,
      1
    ],
    [
      1,
      1,
      1,
      1,
      1,
      0,
      1
    ],
    [
      1,
      1,
      0,
      0,
      1,
      1,
      1
    ],
    [
      0,
      1,
      0,
      1,
      1,
      1,
      1
    ],
    [
      1,
      0,
      1,
      0,
      1,
      0,
      1
    ],
    [
      1,
      0,
      1,
      0,
      1,
      0,
      0
    ],
    [
      1,
      1,
      0,
      1,
      0,
      1,
      1
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 51703,
  "output_tokens": 57354
}
```
