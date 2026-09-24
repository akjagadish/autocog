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
**Description:** **Read-until-sufficient integration with mildly compressed diagnosticity, silence-calibrated evidence, and an S-SHAPED (convex) read-out of relative evidence.**

People do not run a fixed heuristic (TTB, tally, or weighted-additive). They read the expert ratings in the order the screen presents them, accumulate weighted differences, and stop as soon as their running lead is safe from any single expert they have not yet read. Six claims:

1. **Cue weights = MILDLY COMPRESSED stated diagnosticity x a modest reading-position bonus.** w_j = ((v_j-0.5)/0.5)^rho * (1 + alpha*exp(-j/tau) + gamma*exp(-(n-1-j)/tau)), with rho ~1.0-1.3, i.e. subjective weight is essentially linear-to-slightly-convex in (v-0.5). A .95 expert is only ~2x a .72 expert and ~5-8x a .56 expert -- clearly heavier, and roughly *balanced* by a pair of upper-middling experts (.78 + .72), but routinely outvotable by three mid-validity experts ('tally-like without unit-weight tallying'). This exponent sits at the crossover point where the two screen-extreme flankers of a mid-list .95 expert neither dominate it nor are dominated by it, which is what turns double-dissociation displays into genuine coin flips. Position is an attention multiplier, never the causal driver.

2. **Serial reading with a graded sufficiency stop.** After each rating that actually discriminates, the reader asks whether any single unread expert could still overturn the current lead; the bar is theta = phi * max weight among unread positions and stopping is graded, q = sigmoid(s*(|E|/theta - 1)). This is the only place where screen order has structural force, and it is display-contingent. Because the bar is set by the strongest unread expert, a merely salient first cue stops the read only when its weighted lead genuinely exceeds what any single remaining expert could reverse -- and as diagnosticity compression relaxes (higher rho) a salient-but-mediocre first cue loses that race to an unread .95 expert.

3. **Evidence is read FULLY relatively, with no load machinery.** R = sum(w*d)/sum(w*|d|) over the cues read so far; features on which both products are endorsed (or both unendorsed) are simply dropped, so any dense-vs-sparse contrast is exactly zero.

4. **Silence calibration.** Validity is treated as sensitivity: a difference only a top expert notices while the rest of the panel is silent implies a marginal quality gap and is acted on hesitantly. Gain is multiplied by c = 1 - eta*(fraction silent)^q * (weighted mean diagnosticity of the speaking experts), which bites only when silence is overwhelming, producing a *reversed* single-cue validity ladder. This term is a function of raw diagnosticity, not of the compressed weights, so the ladder is invariant to rho.

5. **Decisiveness is a CONVEX (S-shaped) function of the evidence balance, not a linear one.** A near-balanced set of reasons does not merely produce a slightly tilted preference; it produces genuine indifference. Read-out is logistic(beta * c * sign(R)*|R|^kappa) with kappa ~1.4-1.9. A small residual imbalance (|R| ~ 0.1-0.2, e.g. two screen-extreme cues almost exactly offsetting one top-validity cue) is discounted to near-nothing and the choice is a coin flip; a clear weighted majority (|R| ~ 0.5) is still followed at ~0.78-0.82; and a categorical conclusion (|R| = 1, i.e. a sufficiency stop, or a single discriminating cue) is unchanged, so committed reads stay decisive.

6. **Choice** = logistic(beta * c * sign(R)|R|^kappa) with lapse epsilon. Individual differences are unimodal jitter in rho, alpha, phi, s, kappa, beta, eta and lapse -- no strategy subpopulations, no primacy/recency reader types, no learning (there is no feedback).

**Parameters:**
- rho: [1.00, 1.32]
- alpha: [0.7, 1.2]
- gamma: [0.0, 0.15]
- tau: [1.0, 1.6]
- phi: [0.95, 1.20]
- s: [7.0, 14.0]
- beta: [2.80, 3.50]
- kappa: [1.35, 1.85]
- eta: [1.5, 2.1]
- sil_pow: [2.4, 3.0]
- epsilon: [0.0, 0.06]
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

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "GOAL: dissociate READ-UNTIL-SUFFICIENT INTEGRATION (advocated: serial read in screen order, compressed diagnosticity x position attention, sufficiency stop, FULLY RELATIVE evidence R, silence discount c = 1 - eta*(fraction silent)^q*vbar where EVERY non-discriminating feature counts identically, convex kappa read-out) from REASON-COUNTING WITH A TWO-REGIME SALIENCE GATE (competitor: validity-graded vote, unanimity boost, saturating majority bonus, position salience at FULL force only when the reason count is deadlocked and at a constant residual g_res~.25 otherwise, sharp silence threshold in which mutual endorsements count only at weight omega~.82, c floored at c_min).\n\nVALIDITIES BY SCREEN POSITION: [.70, .60, .95, .82, .66]. The near-oracle expert (95%) sits mid-list at pos2, the second heavyweight (.82) sits at pos3, and the two SCREEN EXTREMES are deliberately mid-low (.70, .66) so that (i) the advocated model's attention multiplier makes the extremes roughly, but not quite, balance the mid-list heavyweight (w = [.68, .23, 1.08, .68, .30] at nominal rho=1.15, alpha=.95, tau=1.3), and (ii) the competitor's salience term e_sal = [1.02, .42, .27, .42, 1.02] loads precisely on the two weakest experts, so its deadlock gate can overturn the validity vote. n = 5 is chosen because with exactly 2 of 5 features silent the competitor's effective silence fraction is .40 for both-0 padding but .82*.40 = .33 for both-1 padding, which straddles its sharp threshold sil_f0 ~ .36 -- while the advocated model's f_sil is .40 in BOTH cases, i.e. a mathematically forced null.\n\n8 base cells, each mirrored across sides -> 16 unique pairs, K = 6 -> 96 trials. Mirroring makes the predicted-choice split exactly 8 A / 8 B for BOTH theories, so side bias cannot mimic anything.\n\n(1) SILENCE-COMPOSITION TRIAD, cells 1-3 (pairs 1-6). All three have the IDENTICAL difference vector d = [+1,0,+1,0,+1] (unanimous 3-0 for A; pos0 .70, pos2 .95, pos4 .66 speak, pos1 and pos3 are silent). They differ ONLY in what the two silent experts said: cell 1 both-0 (A=[1,0,1,0,1] vs B=[0,0,0,0,0]), cell 2 both-1 (A=[1,1,1,1,1] vs B=[0,1,0,1,0]), cell 3 mixed (one both-1, one both-0). The advocated model's silence term counts non-discriminating features regardless of level, and its evidence uses only discriminating cues, so it predicts EXACTLY THE SAME probability (~.93-.95, R = 1 after a certain stop at pos2) in all three cells for every parameter setting. The competitor's omega-weighted, threshold-sharp silence term gives c = .16 / .77 / .45 respectively, hence P(A) ~ .72 / .99 / .93 -- a ~27-point ordered spread driven by ratings that discriminate nothing. This within-subject triad is the flagship: one theory is pinned to a flat line, the other to a monotone ladder.\n\n(2) SECOND SILENCE-COMPOSITION CONTRAST IN A CONFLICT DISPLAY, cells 6-7 (pairs 11-14). Both have d = [+1,0,-1,0,+1] (the two screen extremes favour A, the mid-list 95% expert favours B), with the two silent experts both-1 in cell 6 and both-0 in cell 7. Advocated: identical in both (~.45; it stops at pos2 with R ~ -.23, which the convex read-out crushes toward chance -- a mild preference for B). Competitor: P(A) = .65 (both-1) vs .53 (both-0), and it chooses the OPPOSITE option from the advocated model in the both-1 version because mu*g_res*S = +.80 overturns the validity vote V = -.15. Direction disagreement plus a second forced-null test.\n\n(3) DEADLOCK-GATE CELL, cell 4 (pairs 7-8) -- the sharpest cell. d = [+1,0,-1,-1,+1]: the two screen extremes (.70, .66) favour A, the two mid-list heavyweights (.95, .82) favour B, one silent expert. Now kA = kB = 2, so the competitor's gate opens to FULL force (g = 1): drive = V + mu*S = -.80 + 2.45 = +1.65 and P(A) ~ .88. The advocated model reads past pos0, stops at pos3 with R ~ -.44 and predicts P(A) ~ .28-.35 across its whole (rho, alpha, tau, kappa, beta) box (checked at both corners: rho=1.32/alpha=0.7 -> .28; rho=1.0/alpha=1.2/tau=1.6 -> .35). Opposite direction, ~55 points, over both full parameter boxes.\n\nCRUCIALLY, cell 4 and cell 6 differ by a SINGLE feature (whether the .82 expert at pos3 also opposes A). The competitor must become MORE pro-A when that extra expert is added against A (.65 -> .88, because the added reason deadlocks the count and unleashes the salience gate); the advocated model must move the other way (.45 -> .32). A non-monotonicity of exactly this sign is a competitor-only signature; monotone decline is advocated-only.\n\n(4) SUFFICIENCY-STOP CELL, cell 5 (pairs 9-10). d = [+1,+1,0,-1,0]: the two leading cues (.70, .60) favour A, the later .82 expert favours B, two silent both-1. The advocated model's running lead after pos0+pos1 (E = .91) clears the bar set by the strongest unread expert (phi*w3 ~ .71), so it stops and commits categorically on R = +1: P(A) ~ .85-.95 across its box. The competitor never stops here (its first-cue commitment is authority-capped and pos0 is outranked by pos2/pos3), integrates everything to a near-null vote V = -.02 rescued only by the residual salience term, giving P(A) ~ .61. A 25-35 point confidence gap in the same direction, driven purely by stopping vs full integration.\n\n(5) THREE-MID-EXPERTS-vs-ONE-ORACLE CELL, cell 8 (pairs 15-16). d = [+1,+1,0,-1,+1]: pos0 (.70), pos1 (.60) and pos4 (.66) favour A while the mid-list 95% expert favours B; only one expert is silent, so silence machinery is essentially OFF in both models and cannot be blamed for anything. The competitor fires its saturating majority bonus (|dk| = 2 -> M = .67) plus residual salience and predicts a decisive P(A) ~ .89. The advocated model's compressed weights make this an almost exact standoff (sum of the three mid weights 1.21 vs 1.08 for the oracle), the final R ~ +.06 is crushed by the convex read-out, and P(A) ~ .47-.68 across its box -- near indifference where the competitor is near ceiling.\n\nDIAGNOSTIC YIELD: (i) two forced-null contrasts (cells 1/2/3 and cells 6/7) where the advocated theory is mathematically committed to zero difference and the competitor predicts 12-27 point ordered shifts produced by non-discriminating ratings; (ii) one full direction disagreement (cell 4, ~55 points) plus a sign-reversed single-feature manipulation (cell 4 vs cell 6); (iii) two magnitude dissociations (cells 5 and 8, 25-35 points, opposite in sign to each other) that separate sufficiency-stopping + convex read-out from reason-counting with majority/unanimity bonuses. Every pair has at least one discriminating cue; no cell is degenerate.",
  "validities": [
    0.7,
    0.6,
    0.95,
    0.82,
    0.66
  ],
  "trial_a_ratings": [
    [
      1,
      0,
      1,
      0,
      1
    ],
    [
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1,
      1
    ],
    [
      0,
      1,
      0,
      1,
      0
    ],
    [
      1,
      1,
      1,
      0,
      1
    ],
    [
      0,
      1,
      0,
      0,
      0
    ],
    [
      1,
      1,
      0,
      0,
      1
    ],
    [
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
      0,
      1
    ],
    [
      0,
      0,
      1,
      1,
      1
    ],
    [
      1,
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
      0
    ],
    [
      1,
      0,
      0,
      0,
      1
    ],
    [
      0,
      0,
      1,
      0,
      0
    ],
    [
      1,
      1,
      1,
      0,
      1
    ],
    [
      0,
      0,
      1,
      1,
      0
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      1,
      0,
      1
    ],
    [
      0,
      1,
      0,
      1,
      0
    ],
    [
      1,
      1,
      1,
      1,
      1
    ],
    [
      0,
      1,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      0,
      1
    ],
    [
      0,
      1,
      1,
      1,
      0
    ],
    [
      1,
      1,
      0,
      0,
      1
    ],
    [
      0,
      0,
      1,
      1,
      1
    ],
    [
      1,
      1,
      1,
      0,
      1
    ],
    [
      0,
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
      1
    ],
    [
      0,
      0,
      1,
      0,
      0
    ],
    [
      1,
      0,
      0,
      0,
      1
    ],
    [
      0,
      0,
      1,
      1,
      0
    ],
    [
      1,
      1,
      1,
      0,
      1
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 31566,
  "output_tokens": 31197
}
```
