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
**Description:** **Chorus-Threshold Novelty Weighting, Balanced-Gradient variant (CTN-BG).**

People told the accuracies of a panel of binary expert ratings do not weight a rating by its accuracy; they weight it by how much *news* it carries, and news is loud only when it stands against a genuine CHORUS. Six commitments, all smooth and continuous (no mute band, no solo release, no discrete count switch).

**(1) The count comparator is a parameter-free gate.** Before any expert identity is consulted, an unweighted feature-wise comparison asks: does one product win on strictly more attributes? If not — equal win counts, including wholly non-discriminating displays — the display is non-diagnostic and the subject flips a mental coin: exactly 0.5, with NO validity tie-break and NO take-the-best fallback, at every panel composition and size.

**(2) Seniority is panel-relative; novelty is a smooth convex, monotonically DECREASING function of it, and the descent is STEEP.** s_j = (v_j − v_min)/(v_max − v_min); nov_j = (1 − s_j)^kappa with kappa ≈ 14–16.5. Nothing is muted by fiat and nothing is released by fiat, but the curve is sharp: only the panel's genuinely idiosyncratic BOTTOM voice speaks with nov ≈ 1; the second-from-bottom voice retains a real but small share; a mid voice essentially nothing; a star nothing at all. Novelty is never negative — a star is ignored, never counted against its own side.

**(3) THE CORE CLAIM — a novel voice is loud only in proportion to the CHORUS it speaks into, and a chorus is measured in *mid-expert equivalents*.** Each other discriminating voice contributes a subjective unit u_i = (1 + tau·s_i)/(1 + tau/2): a mid-panel voice counts as exactly one unit, a star ≈ 1.4, a bottom voice ≈ 0.6. The chorus a cue j faces is Q_j = Σ_{opposing} u_i + lam_own·Σ_{own side, i≠j} u_i — voices it contradicts counted in full, voices it merely stands beside echoed at a discount. Loudness is a smooth logistic of that chorus, g_j = Gmax·σ((Q_j − q0)/wid), with a narrow but continuous threshold at roughly two-and-a-half mid-equivalents. BOTH how many voices and how senior they are matter, in one scalar: two mid voices are not yet a chorus, two stars just barely are, three upper voices fully are.

**(4) The ungated panel gradient is THIN and only MILDLY convex — a bare ordinal pull that does not, by itself, give the panel's bottom voice an edge over a pair of mid voices.** Every discriminating cue contributes r_j = eta + c·(1 − s_j)^nu with small eta, small c and a moderate exponent nu ≈ 1.7–2.05. The scale is small enough that all-upper displays drift near chance at any count margin, and the exponent is flat enough that a single floor cue's residual is worth about the same as two mid cues' residuals — so a floor voice opposed by a mid PAIR is a dead heat (it needs the chorus gate, not the gradient, to win), while a floor voice opposed by a three-cue upper bloc is carried past chance by the gate.

**(5) Integration is Luce-like and saturating.** w_j = r_j + nov_j·g_j; E = Σ dir_j w_j; z = beta·E/((c_n + Σ_j w_j)·D^gamma). Evidence is judged relative to the total weight of evidence on the table plus a constant of indecision, and is diluted by the number of mutually discriminating attributes with gamma ≈ 1, so a lone cue and an unopposed pair of the same tier feel about equally compelling. An attention lapse epsilon mixes the result with a coin flip.

**(6) Capacity is a near-total collapse of the chorus construal past ~7 voices, with only mild extra noise.** For n > 7 the amplification essentially vanishes (Gmax × (1 − psi·(n−7))) while integration noise grows only slightly, so eight-expert star-backed majorities sit near chance rather than being reversed, and eight-expert plain tally adherence survives on the residual gradient.

**Cross-design signature:** exact chance on tally ties at any composition; near-chance on displays whose every discriminating voice is upper-band, at every margin; moderate (not ceiling) adherence to majorities containing no bottom voice; a bottom voice facing two MID voices lands at chance, facing two STARS reverses substantially, facing a three-cue upper bloc reverses to ≈.12–.18; near-ceiling following of majorities that contain the bottom voice; attenuated integration on eight-expert panels.

**Parameters:**
- eta: [0.030, 0.048]
- c_res: [0.10, 0.14]
- nu: [1.70, 2.05]
- kappa: [14.0, 16.5]
- tau: [1.0, 1.5]
- lam_own: [0.50, 0.70]
- gmax: [1.70, 2.10]
- q0: [2.38, 2.52]
- wid: [0.11, 0.17]
- c_n: [2.70, 3.05]
- beta: [34, 48]
- gamma: [0.88, 1.05]
- epsilon: [0.09, 0.15]
- psi: [0.80, 0.95]
- chi: [0.02, 0.08]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np

    # ---------------- parse the stimulus ----------------
    a = None
    b = None
    if isinstance(state, dict):
        for ka, kb in (('option_a_ratings', 'option_b_ratings'),
                       ('a_ratings', 'b_ratings'),
                       ('A', 'B'), ('a', 'b')):
            if ka in state and kb in state:
                try:
                    a = np.asarray(list(state[ka]), dtype=float).ravel()
                    b = np.asarray(list(state[kb]), dtype=float).ravel()
                except Exception:
                    a = None
                    b = None
                break
    if a is None or b is None:
        try:
            arr = np.asarray(state, dtype=float).reshape(2, -1)
            a = np.asarray(arr[0], dtype=float).ravel()
            b = np.asarray(arr[1], dtype=float).ravel()
        except Exception:
            return np.array([0.5, 0.5], dtype=float)

    if a is None or b is None or a.size == 0 or a.size != b.size:
        return np.array([0.5, 0.5], dtype=float)

    n = int(a.size)

    # ---------- Commitment 1: parameter-free count gate ----------
    d = np.sign(a - b)
    iA = np.where(d > 0)[0]
    iB = np.where(d < 0)[0]
    mA = int(iA.size)
    mB = int(iB.size)
    if mA == mB:                      # includes wholly non-discriminating displays
        return np.array([0.5, 0.5], dtype=float)

    D = float(mA + mB)

    # ---------------- parameters ----------------
    def _g(name, default, lo, hi):
        try:
            x = float(parameters.get(name, default))
        except Exception:
            x = float(default)
        if not np.isfinite(x):
            x = float(default)
        return float(min(max(x, lo), hi))

    eta = _g('eta', 0.040, 0.0, 0.5)
    c_res = _g('c_res', 0.12, 0.0, 1.5)
    nu = _g('nu', 1.88, 0.5, 5.0)
    kappa = _g('kappa', 15.0, 1.0, 30.0)
    tau = _g('tau', 1.2, 0.0, 4.0)
    lam_own = _g('lam_own', 0.60, 0.0, 1.0)
    gmax = _g('gmax', 1.90, 0.0, 5.0)
    q0 = _g('q0', 2.45, 0.20, 8.0)
    wid = _g('wid', 0.14, 0.02, 1.0)
    c_n = _g('c_n', 2.9, 0.10, 12.0)
    beta = _g('beta', 41.0, 0.0, 150.0)
    gamma = _g('gamma', 0.96, 0.0, 2.0)
    eps = _g('epsilon', 0.12, 0.0, 0.6)
    psi = _g('psi', 0.88, 0.0, 1.0)
    chi = _g('chi', 0.05, 0.0, 3.0)

    # ---------------- stated validities ----------------
    v = None
    try:
        vr = parameters.get('validities', None) if hasattr(parameters, 'get') else None
        if vr is not None:
            v = np.asarray(vr, dtype=float).ravel()
    except Exception:
        v = None
    if v is None or v.size != n:
        v = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.75], dtype=float)

    v_min = float(np.min(v))
    v_max = float(np.max(v))
    span = v_max - v_min
    if (not np.isfinite(span)) or span <= 1e-9:
        s = np.zeros(n, dtype=float)
    else:
        s = np.clip((v - v_min) / span, 0.0, 1.0)

    one_minus = np.clip(1.0 - s, 0.0, 1.0)

    # ---------- Commitment 2: smooth convex novelty, steep descent ----------
    nov = np.power(one_minus, kappa)
    nov = np.where(np.isfinite(nov), nov, 0.0)

    # ---------- Commitment 4: THIN, mildly convex panel gradient ----------
    resid = eta + c_res * np.power(one_minus, nu)
    resid = np.where(np.isfinite(resid), resid, eta)

    # ---------- Commitment 3: chorus measured in MID-EXPERT EQUIVALENTS ----------
    denom_u = 1.0 + 0.5 * tau
    if (not np.isfinite(denom_u)) or denom_u <= 0.0:
        denom_u = 1.0
    u = (1.0 + tau * s) / denom_u        # mid voice == 1 unit, star ~1.4, floor ~0.6
    u = np.where(np.isfinite(u), u, 1.0)

    U_A = float(np.sum(u[iA])) if mA > 0 else 0.0
    U_B = float(np.sum(u[iB])) if mB > 0 else 0.0

    Q = np.zeros(n, dtype=float)
    if mA > 0:
        Q[iA] = U_B + lam_own * (U_A - u[iA])
    if mB > 0:
        Q[iB] = U_A + lam_own * (U_B - u[iB])
    Q = np.clip(Q, 0.0, None)

    # ---------- Commitment 6: capacity limit past ~7 voices ----------
    over = max(0.0, float(n) - 7.0)
    gmax_eff = gmax * max(0.0, 1.0 - psi * over)
    noise = 1.0 + chi * over
    if (not np.isfinite(noise)) or noise <= 0.0:
        noise = 1.0

    # smooth logistic threshold in the chorus size
    xg = (Q - q0) / max(wid, 1e-6)
    xg = np.clip(xg, -60.0, 60.0)
    gate = gmax_eff / (1.0 + np.exp(-xg))
    gate = np.where(np.isfinite(gate), gate, 0.0)

    # ---------- weights ----------
    w = resid + nov * gate
    w = np.where(np.isfinite(w), w, 0.0)
    w = np.clip(w, 0.0, None)

    E = 0.0
    if mA > 0:
        E += float(np.sum(w[iA]))
    if mB > 0:
        E -= float(np.sum(w[iB]))
    if not np.isfinite(E):
        E = 0.0

    W_tot = 0.0
    if mA > 0:
        W_tot += float(np.sum(w[iA]))
    if mB > 0:
        W_tot += float(np.sum(w[iB]))
    if (not np.isfinite(W_tot)) or W_tot < 0.0:
        W_tot = 0.0

    # ---------- Commitment 5: Luce-like saturating integration ----------
    den = (c_n + W_tot) * (float(D) ** gamma) * noise
    if (not np.isfinite(den)) or den <= 0.0:
        den = 1.0

    z = beta * E / den
    if not np.isfinite(z):
        z = 0.0
    z = float(min(max(z, -30.0), 30.0))

    p_a = 1.0 / (1.0 + np.exp(-z))
    p_a = (1.0 - eps) * p_a + eps * 0.5
    p_a = float(min(max(p_a, 1e-9), 1.0 - 1e-9))

    p = np.array([p_a, 1.0 - p_a], dtype=float)
    p = np.clip(p, 1e-12, 1.0)
    return p / p.sum()
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if (not np.isfinite(s)) or s <= 0:
        probs = np.ones_like(probs) / float(len(probs))
    else:
        probs = probs / s
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** **Flat-Maverick Inverse-Seniority Integration with a Count-Confidence Gate and Mild Count Dilution (FM-ISN-CC5).**

When people are told the accuracies of a panel of binary raters, they do not weight a rating by its accuracy and they do not weight it by the crowd it happens to face. They weight it by one thing only: *how far down the displayed panel's accuracy range that voice sits*. Five commitments.

**(1) CONTEXT-FREE INVERSE-SENIORITY WEIGHT (the decisive claim), NEARLY CATEGORICAL about the true panel floor.** Seniority is panel-relative: s_j = (v_j − v_min)/(v_max − v_min) computed over the panel shown in the instructions. Each discriminating cue gets w_j = eta + (1 − s_j)^kappa with kappa ≈ 15–17 and a very thin flat floor eta ≈ 0.008–0.018. Only the panel's genuinely bottom voice speaks with full loudness; the SECOND-from-bottom voice already retains only about a quarter of it, and mid and star voices collapse onto the near-zero flat residual and are effectively interchangeable and mute. Crucially the weight is **completely independent of context**: it does not depend on how many voices oppose the cue, how senior those opponents are, whether the cue stands alone, or whether it sits inside a unanimous bloc. There is NO chorus amplification of any kind. A floor voice opposed by a pair, a floor voice opposed by a trio, and a floor voice inside a unanimous bloc are all read with exactly the same loudness — their observed rates differ only through the ordinary arithmetic of the weighted balance.

**(2) NORMALISED (LUCE-LIKE), SATURATING INTEGRATION WITH MILD COUNT DILUTION.** Evidence is the signed weighted balance E = Σ_j dir_j w_j, and confidence is that balance measured against the total subjective weight actually on the table plus a constant of indecision, further diluted by the number of mutually discriminating attributes: z = beta·E/((c_n + Σ_j w_j)·D^gamma) with gamma ≈ 0.34–0.46. Because a display made entirely of upper-band voices puts almost no subjective weight on the table, such displays sit near chance *at every count margin*, and the mild dilution makes a lone thin voice at least as decisive as an unopposed thin pair. A display containing the panel's floor voice immediately supplies enough weight to drive the choice near ceiling in the floor's direction.

**(3) THE COUNT COMPARATOR GATES CONFIDENCE, NOT DIRECTION.** An unweighted feature-wise count is computed first. It never supplies a direction of its own and never acts as a tie-break by validity rank. What it does is license confidence: when one option wins on strictly more attributes the weighted reading is taken at full gain; when the counts are equal (including wholly non-discriminating displays) the very same weighted reading is taken at a moderately reduced gain rho_tie ≈ 0.60–0.78, so tie displays hover close to chance — slightly *anti*-validity, never pro-validity.

**(4) CAPACITY IS AT MOST A WHISPER.** Holding a whole validity ordering in mind gets marginally harder past about seven voices; the integration slope is divided by 1 + chi·(n − 7) with chi ≈ 0–0.10, i.e. essentially absent. Eight-expert panels retain their tally adherence, and star-backed dense majorities stay at chance for the independent reason that their weights are all near eta.

**(5) LAPSE.** A small attention lapse epsilon ≈ 0.05–0.10 mixes the result with a coin flip.

**Cross-design signature:** near-chance (slightly anti-validity) on tally ties at any composition; chance on displays whose every discriminating voice is upper-band, at any margin, with solo ≥ multi-cue upper; a null on the two contrasts that falsify chorus theories — an opposed floor voice and a floor voice inside a unanimous bloc come out at the same rate, and titrating the seniority of an opposing pair produces a flat line; hard reversal of upper-backed majorities by a lone floor voice; near-ceiling following of majorities containing the floor voice; and only MODERATE anti-validity when the weak side is carried merely by the second-from-bottom voice rather than by the true floor.

**Parameters:**
- eta: [0.008, 0.018]
- kappa: [15.0, 17.0]
- beta: [8.3, 9.2]
- c_n: [1.2, 1.5]
- gamma: [0.34, 0.46]
- rho_tie: [0.60, 0.78]
- epsilon: [0.05, 0.10]
- chi: [0.00, 0.10]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np

    # ---------------- parse the stimulus ----------------
    a = None
    b = None
    if isinstance(state, dict):
        for ka, kb in (('option_a_ratings', 'option_b_ratings'),
                       ('a_ratings', 'b_ratings'),
                       ('A', 'B'), ('a', 'b')):
            if ka in state and kb in state:
                try:
                    a = np.asarray(list(state[ka]), dtype=float).ravel()
                    b = np.asarray(list(state[kb]), dtype=float).ravel()
                except Exception:
                    a = None
                    b = None
                break
    if a is None or b is None:
        try:
            arr = np.asarray(state, dtype=float).reshape(2, -1)
            a = np.asarray(arr[0], dtype=float).ravel()
            b = np.asarray(arr[1], dtype=float).ravel()
        except Exception:
            return np.array([0.5, 0.5], dtype=float)

    if a is None or b is None or a.size == 0 or a.size != b.size:
        return np.array([0.5, 0.5], dtype=float)

    n = int(a.size)

    # ---------------- parameters ----------------
    def _g(name, default, lo, hi):
        try:
            x = float(parameters.get(name, default))
        except Exception:
            x = float(default)
        if not np.isfinite(x):
            x = float(default)
        return float(min(max(x, lo), hi))

    eta = _g('eta', 0.013, 0.0, 0.50)        # thin flat listening floor
    kappa = _g('kappa', 16.0, 1.0, 30.0)     # steepness of inverse-seniority descent
    beta = _g('beta', 8.75, 0.0, 40.0)       # integration slope
    c_n = _g('c_n', 1.35, 0.05, 20.0)        # constant of indecision
    gamma = _g('gamma', 0.40, 0.0, 2.0)      # mild count dilution
    rho_tie = _g('rho_tie', 0.69, 0.0, 1.0)  # confidence gain when counts are tied
    eps = _g('epsilon', 0.075, 0.0, 0.60)    # attention lapse
    chi = _g('chi', 0.05, 0.0, 2.0)          # (near-absent) capacity flattening past 7 voices

    # ---------------- stated validities ----------------
    v = None
    try:
        vr = parameters.get('validities', None) if hasattr(parameters, 'get') else None
        if vr is not None:
            v = np.asarray(vr, dtype=float).ravel()
    except Exception:
        v = None
    if v is None or v.size != n:
        v = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.75], dtype=float)

    v_min = float(np.min(v))
    v_max = float(np.max(v))
    span = v_max - v_min
    if (not np.isfinite(span)) or span <= 1e-9:
        s = np.zeros(n, dtype=float)
    else:
        s = np.clip((v - v_min) / span, 0.0, 1.0)

    # ---------- Commitment 1: context-free inverse-seniority weights ----------
    base = np.clip(1.0 - s, 0.0, 1.0)
    try:
        w = eta + np.power(base, kappa)
    except Exception:
        w = eta + base
    w = np.where(np.isfinite(w), w, eta)
    w = np.clip(w, 0.0, None)

    # ---------------- feature-wise comparison ----------------
    d = np.sign(a - b)
    iA = np.where(d > 0)[0]
    iB = np.where(d < 0)[0]
    mA = int(iA.size)
    mB = int(iB.size)
    D = mA + mB
    if D == 0:
        return np.array([0.5, 0.5], dtype=float)

    sumA = float(np.sum(w[iA])) if mA > 0 else 0.0
    sumB = float(np.sum(w[iB])) if mB > 0 else 0.0
    E = sumA - sumB
    W = sumA + sumB
    if not np.isfinite(E):
        E = 0.0
    if (not np.isfinite(W)) or W < 0.0:
        W = 0.0

    # ---------- Commitment 3: count comparator gates CONFIDENCE only ----------
    gain = rho_tie if mA == mB else 1.0

    # ---------- Commitment 4: (near-absent) capacity flattening ----------
    cap = 1.0 + chi * max(0.0, float(n) - 7.0)
    if (not np.isfinite(cap)) or cap <= 0.0:
        cap = 1.0

    # ---------- Commitment 2: Luce-like saturating integration + mild dilution ----------
    dil = float(D) ** gamma
    if (not np.isfinite(dil)) or dil <= 0.0:
        dil = 1.0

    den = (c_n + W) * dil * cap
    if (not np.isfinite(den)) or den <= 0.0:
        den = 1.0

    z = beta * E * gain / den
    if not np.isfinite(z):
        z = 0.0
    z = float(min(max(z, -30.0), 30.0))

    p_a = 1.0 / (1.0 + np.exp(-z))

    # ---------- Commitment 5: lapse ----------
    p_a = (1.0 - eps) * p_a + eps * 0.5
    p_a = float(min(max(p_a, 1e-9), 1.0 - 1e-9))

    p = np.array([p_a, 1.0 - p_a], dtype=float)
    p = np.clip(p, 1e-12, 1.0)
    return p / p.sum()
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if (not np.isfinite(s)) or s <= 0:
        probs = np.ones_like(probs) / float(len(probs))
    else:
        probs = probs / s
    return int(np.random.choice(len(probs), p=probs))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] Five experts with a wide validity spread (0.95, 0.85, 0.75, 0.62, 0.55) make the cue hierarchy salient, so Take The Best (TTB) has an unambiguous search order, while Tallying is indifferent to that order. The 12 unique pairs (repeated 8x each, ~96 trials) fall into three families, all of which dissociate the two models.

(1) Direct-conflict pairs (T1-T6, T11-T12): one option wins the highest discriminating cue while the other wins the majority of the remaining lower-validity cues. TTB predicts choice of the top-cue winner on every one of these; Tallying predicts the opposite option. Critically, the tally margin is parametrically varied: margin 3 (T1/T2: 1 vs 4 cue wins), margin 2 (T5/T6: 1 vs 3; T11/T12: 1 vs 3 with the top cue tied so TTB's deciding cue is expert 2), and margin 1 (T3/T4: 2 vs 3). This yields the key *quantitative* dissociation: TTB's softmax operates on a binary 1/0 score, so its predicted choice probability for the top-cue winner is identical (sigmoid(beta), lapse-corrected) across all conflict trials regardless of tally margin, whereas Tallying's softmax probability grows monotonically with the tally margin (exp(beta*1) vs exp(beta*2) vs exp(beta*3) contrasts). Thus a flat, margin-invariant preference for the top-cue winner is uniquely TTB, while a graded, margin-increasing preference for the majority-cue winner is uniquely Tallying.

(2) Tally-tie pairs (T7-T10): the two options split cue wins evenly (1-1 or 2-2), so Tallying is forced to guess at exactly 50/50 irrespective of its parameters, while TTB still decides confidently via the highest discriminating cue. Systematic above-chance choice of the top-cue winner here cannot be produced by Tallying under any beta/epsilon, making these trials a parameter-free falsification test.

(3) Left/right and direction counterbalancing: every conflict and tie configuration appears in both A-favoring and B-favoring versions, so any side bias cancels and cannot masquerade as evidence for either heuristic.

T11/T12 additionally probe whether the deciding cue must be the very best cue: expert 1 ties, so TTB's decision rests on expert 2 while the tally strongly favors the other option; this separates TTB from any 'only-the-single-best-cue-matters' caricature as well as from Tallying.

[1] Goal: dissociate Tallying (advocated) from Take The Best by factorially crossing (i) which option wins the highest-validity discriminating cue \u2014 the only thing TTB is sensitive to \u2014 with (ii) the tally margin, the only thing Tallying is sensitive to. Six experts with a wide, deliberately scrambled validity spread [0.93, 0.57, 0.79, 0.86, 0.62, 0.71] give the descending cue order f0 > f3 > f2 > f5 > f4 > f1, so validity rank is decoupled from screen position (unlike a monotone left-to-right layout, this prevents a positional-ordering confound from mimicking TTB).

Core manipulation (trials 1,3,5,7,9 and their exact A/B mirrors 2,4,6,8,10): option A always wins expert f0 (validity .93), so TTB's deciding cue and predicted winner are IDENTICAL across all five cells; its predicted P(choose A) is the constant (1-eps)*sigmoid(beta) + eps/2 regardless of the rest of the profile. Meanwhile the tally margin is parametrically swept M = -3, -1, 0, +2, +4 by re-allocating the five lower-validity cues. Tallying's softmax gives P(A) = sigmoid(beta*M) (lapse-corrected), i.e. a strictly monotone, sign-crossing psychometric function of M: strong B-preference at M=-3, weak B at M=-1, exactly 50/50 at M=0, and increasingly strong A at +2 and +4. A flat line across the five margin cells is a unique TTB signature; a monotone increasing line crossing chance at M=0 is a unique Tallying signature. Crucially the design contains both conflict cells (M=-3, -1: TTB says A, Tallying says B) and agreement cells (M=+2, +4: both say A), so the same subject's data pin down beta from the agreement cells and then test whether the conflict cells reverse (Tallying) or not (TTB) \u2014 a within-subject, parameter-anchored test rather than a mere direction contrast.

Tally-tie cells (trials 5, 6): three cue wins each. Tallying MUST be exactly 50/50 for every possible beta and epsilon, while TTB confidently picks the f0 winner. Systematic above-chance choice here parameter-free falsifies Tallying; chance performance parameter-free falsifies TTB.

Low-rank deciding cue (trials 11, 12): the two best experts (f0=.93 and f3=.86) are tied, so TTB must descend to f2 (.79), which favors A, while the tally favors B by 2 (B wins f5, f4, f1). This checks whether the one-reason rule really cascades down the validity hierarchy or whether subjects instead aggregate the three lower experts \u2014 an independent conflict test whose TTB prediction is the same magnitude as in trials 1/3 (binary score) but whose Tallying prediction is intermediate (|M|=2).

Every configuration appears in both an A-favoring and a B-favoring mirror, so side bias cancels exactly and cannot masquerade as support for either heuristic. Tie cues are encoded sometimes as 0-0 and sometimes as 1-1 to prevent a 'count the 1s' surface strategy from being confounded with either model. 12 unique pairs \u00d7 8 repetitions \u2248 96 trials.

[2] GOAL: dissociate Limited-Sample Noisy Cue Counting (LSNC, advocated) from softmax Tallying (competing). Tallying's choice probability is a function of the tally margin M = nA - nB ALONE: any two pairs with the same M must produce exactly the same P(choose A) for every possible beta and epsilon. LSNC instead depends on the full composition (nA, nB, nT): tied/uninformative cues occupy sampling slots (tie dilution) and only the PROPORTION of sampled informative cues won matters, so two profiles with identical M but different numbers of tied cues yield different adherence. The design is therefore a 3 (margin: M = 1, 2, 3) x 2 (informative density: SPARSE, all discriminating cues won by one option but many tied cues; DENSE, no/few tied cues but the loser also wins several cues) factorial, fully mirrored for side. Tallying predicts a pure main effect of margin and EXACTLY ZERO density effect (parameter-free); LSNC predicts sparse > dense at every margin.

CELLS (A-favouring version; the mirror swaps A and B):
- S1 sparse M=+1: (nA,nB,nT) = (1,0,5)
- D1 dense  M=+1: (3,2,1)
- S2 sparse M=+2: (2,0,4)
- D2 dense  M=+2: (4,2,0)
- S3 sparse M=+3: (3,0,3)
- D3 dense  M=+3: (4,1,1)

QUANTITATIVE LSNC PREDICTIONS (k=2.5, nu=0.20, epsilon=0, exact enumeration of the hypergeometric sample compositions): M=1: 0.625 (sparse) vs 0.572 (dense); M=2: 0.702 vs 0.630; M=3: 0.754 vs 0.707. So LSNC predicts a ~5-7 point sparse-minus-dense advantage at EVERY margin (and the gap grows with k), while adherence stays capped well below 1 even at M=3. Tallying predicts 0 at every margin. Aggregated over the 3 margins each subject contributes 48 sparse and 48 dense trials, so the parameter-free density contrast is powerfully estimated within subject.

CONFOUND CONTROL (critical): (i) In every A-favouring cell option A wins the highest-validity discriminating cue f0 (0.92), which also discriminates in every cell, so Take-The-Best (or any 'best cue' strategy) predicts a CONSTANT P(A) across all six cells and cannot generate a density effect. (ii) Validity-weighted additive evidence (log-odds weights w = [2.44, 0.41, 1.59, 0.90, 0.20, 1.21]) was deliberately matched between the sparse and dense member of each margin pair: M=1: +2.44 vs +2.13; M=2: +2.64 vs +2.53; M=3: +3.05 vs +3.16. Hence a WADD-type subject also cannot produce the sparse>dense pattern; only sample-limited counting with tie dilution can. (iii) Validity order is scrambled relative to screen position (descending order is f0, f2, f5, f3, f1, f4) so positional heuristics do not mimic validity-ordered search. (iv) Tied cues are encoded sometimes as 1-1 and sometimes as 0-0 so that the number of displayed 1s is not a constant cue to density. (v) Every configuration appears in both an A-favouring and a B-favouring mirror (12 unique pairs x 8 repetitions = 96 trials), so side bias cancels exactly.

INTERPRETATION: a graded, margin-monotone function that is IDENTICAL for sparse and dense cells supports Tallying; a margin-monotone function that is systematically shifted upward in the sparse (tie-padded, proportion-1.0) cells relative to the dense cells with the same margin, together with adherence ceilings around 0.70-0.80 at M=3, uniquely supports LSNC and cannot be produced by Tallying under any beta/epsilon.

[3] GOAL: dissociate softmax Tallying (advocated) from Limited-Sample Noisy Cue Counting (LSNC, competing). Tallying's choice probability is a function of the tally margin M = nA - nB ALONE: P(A) = (1-eps)*sigmoid(beta*M) + eps/2. Therefore any two profiles with the same M MUST yield exactly the same P(A) for every beta and epsilon -- a parameter-free invariance. LSNC instead depends on the full composition (nA, nB, nT): because the inspected subset is drawn uniformly from ALL features, tied cues occupy sampling slots (tie dilution) and only the PROPORTION of sampled informative cues won matters, so adherence must fall monotonically as a fixed margin is realized with fewer ties and more loser-wins.

DESIGN: 8 experts, scrambled validity spread [0.93, 0.55, 0.78, 0.86, 0.60, 0.71, 0.66, 0.82] (descending rank f0 > f3 > f7 > f2 > f5 > f6 > f4 > f1, so validity order is decoupled from screen position). 2 (margin: M = 2, 4) x 3 (composition: SPARSE / MID / DENSE) factorial, each cell fully mirrored for side => 12 unique pairs x 8 repetitions ~ 96 trials.

Cells (A-favouring version; mirror swaps A and B):
- S2 sparse M=2: (nA,nB,nT) = (2,0,6)
- M2 mid    M=2: (3,1,4)
- D2 dense  M=2: (5,3,0)
- S4 sparse M=4: (4,0,4)
- M4 mid    M=4: (5,1,2)
- D4 dense  M=4: (6,2,0)

QUANTITATIVE PREDICTIONS. Tallying: a pure margin main effect and EXACTLY ZERO composition effect -- the three M=2 cells must be statistically identical, the three M=4 cells must be statistically identical, and the M=4 level must exceed M=2 by the sigmoid gap sigmoid(4b) - sigmoid(2b). LSNC (exact enumeration, k=2.5, nu=0.20, eps=0): M=2 -> 0.661 (sparse) vs ~0.62 (mid) vs 0.598 (dense); M=4 -> 0.750 (sparse) vs ~0.72 (mid) vs 0.689 (dense). I.e. LSNC predicts a monotone ~6-point decline across the composition ladder at BOTH margins, plus adherence capped near 0.75-0.80 even at M=4. Note that at sample size K=1 all same-margin cells coincide (P = 0.5 + 0.3*M/n), so any composition effect in the data is diagnostic of multi-cue but sub-total sampling; a flat ladder is diagnostic of full integration (Tallying).

CONFOUND CONTROLS. (i) In every A-favouring cell option A wins the highest-validity cue f0 (0.93), which discriminates in all six cells, so Take-The-Best / any 'best cue' rule predicts a CONSTANT P(A) across all cells and cannot generate either the margin effect or a composition ladder. (ii) Validity-weighted additive evidence (log-odds weights) was deliberately matched WITHIN each margin level: M=2 cells give net +2.99 (sparse), +3.04 (mid), +3.12 (dense); M=4 cells give +4.55, +4.78, +4.39. Hence WADD-type integration also predicts no composition effect, so a sparse>mid>dense ordering can only come from sample-limited counting with tie dilution, while a flat ladder plus a clear margin effect uniquely supports Tallying. (iii) Tied cues are encoded sometimes 1-1 and sometimes 0-0 so that display density of 1s is not itself a cue to composition. (iv) Every configuration appears in both an A-favouring and a B-favouring mirror, so side bias cancels exactly.

INTERPRETATION: margin-graded but composition-invariant adherence (with the M=4 cells approaching the ceiling implied by the beta estimated from the M=2 cells) supports Tallying; a systematic sparse > mid > dense decline at each margin, with adherence capped around 0.75 at M=4, supports LSNC.

[4] GOAL: dissociate Limited-Sample Noisy Cue Counting (LSNC, advocated) from Count-Direction with Conflict-Graded Confidence (CDCC, competing). Both theories agree on the count-based DIRECTION of choice and both cap adherence below 1, so the classic tally-vs-TTB contrasts are useless here. They differ on exactly two things, and this design isolates both.

KEY DISSOCIATION 1 (parameter-free for LSNC): COHERENCE. LSNC's predicted probability is a function of the composition (nA, nB, nT) ALONE 
cue identity/validity is irrelevant because the inspected subset is drawn uniformly at random. CDCC's confidence is gated by the coherence term C = (W_win - W_lose)/(W_win + W_lose) with convex weights w = (v-0.5)^rho, so it must show a large drop when the highest-validity cue is won by the count LOSER, holding (nA, nB, nT) exactly fixed. Every composition therefore appears in a COHERENT version (count winner also wins the top-validity cues, loser wins only the weakest ones) and a CONFLICT version (count winner wins only the weakest cues, loser wins the top cue(s)), with the identical (nA, nB, nT).

KEY DISSOCIATION 2 (composition ordering reversal): with n = 6 fixed, CDCC's dilution term is S = M/D^gamma (D = nA+nB, ties irrelevant), whereas LSNC's dilution comes from tied cues occupying sampling slots and from the PROPORTION of sampled informative cues won. These give opposite orderings for two of the three cells: LSNC ranks (4,2,0) ABOVE (2,1,3) (0.630 vs 0.589), while CDCC ranks (2,1,3) ABOVE (4,2,0) (0.633 vs 0.602 averaged over coherence; 0.707 vs 0.651 in the coherent versions), because CDCC penalises D = 6 heavily via D^1.5 while LSNC penalises the 3 tied slots.

DESIGN: 6 experts, scrambled validity spread [0.94, 0.57, 0.86, 0.62, 0.70, 0.78] (descending rank f0 > f2 > f5 > f4 > f3 > f1, so validity rank is decoupled from screen position and positional heuristics cannot mimic validity-based coherence). One very high (0.94) and two low (0.57, 0.62) validities make the convex weighting in CDCC bite hard (w = (v-.5)^3 gives f0 = .085 vs f1 = .0003, a 250:1 ratio). 3 compositions x 2 coherence levels x 2 side mirrors = 12 unique pairs x 8 repetitions = 96 trials.

CELLS (A-favouring version listed; the mirror swaps A and B):
- T1/T2  C1-COHERENT (nA,nB,nT) = (2,1,3): A wins f0(.94), f2(.86); B wins f1(.57); f3,f4,f5 tied.
- T3/T4  C1-CONFLICT (2,1,3): A wins f1(.57), f3(.62); B wins f0(.94); f2,f4,f5 tied.
- T5/T6  C2-COHERENT (3,1,2): A wins f0,f2,f5; B wins f1; f3,f4 tied.
- T7/T8  C2-CONFLICT (3,1,2): A wins f1,f3,f4; B wins f0; f2,f5 tied.
- T9/T10 C3-COHERENT (4,2,0): A wins f0,f2,f4,f5; B wins f1,f3.
- T11/T12 C3-CONFLICT (4,2,0): A wins f1,f3,f4,f5; B wins f0,f2.

QUANTITATIVE PREDICTIONS. LSNC (exact enumeration, k = 2.5, nu = 0.20, eps = 0): P(count winner) = 0.589 for (2,1,3), 0.656 for (3,1,2), 0.630 for (4,2,0) 
 IDENTICAL for the coherent and conflict member of each pair (a strictly zero coherence effect, for every k, nu, eps). CDCC (beta = 2.6, gamma = 1.5, kappa = 0.7, rho = 3.0, eps = 0.10): coherent 0.707 / 0.757 / 0.651 versus conflict 0.544 / 0.576 / 0.553 
 a 10-18 point coherence effect that cannot be switched off (kappa > 0.4 by hypothesis) and that is largest for the sparse (2,1,3) and (3,1,2) cells where the loser's single win is the top expert.

Each subject contributes 48 coherent and 48 conflict trials, so the coherence contrast is estimated within subject with high power. INTERPRETATION: adherence that depends only on the (nA,nB,nT) composition, is flat across the coherence manipulation, and shows the (3,1,2) > (4,2,0) > (2,1,3) ordering with a ceiling near 0.65, uniquely supports LSNC. A systematic coherent > conflict gap (never dropping below 0.5) together with the (3,1,2) > (2,1,3) > (4,2,0) ordering uniquely supports CDCC. Below-chance adherence in the conflict cells would falsify BOTH theories in favour of validity-weighted / one-reason integration, so the design is also self-diagnostic.

CONTROLS: (i) every configuration appears in an A-favouring and a B-favouring mirror, so side bias cancels exactly; (ii) tied cues are encoded sometimes 1-1 and sometimes 0-0 so the number of displayed 1s does not covary with composition; (iii) validity order is scrambled relative to display position; (iv) no pair is degenerate 
 every trial has a strict count winner and every trial requires reading more than one rating.

[5] GOAL: dissociate Count-Direction with Conflict-Graded Confidence (CDCC, advocated) from Limited-Sample Noisy Cue Counting (LSNC, competing). Both theories agree that the DIRECTION of choice is the count winner and both cap adherence below 1, so tally-vs-TTB style contrasts are useless. They differ on exactly what modulates adherence: LSNC's probability is a function of the composition (nA, nB, nT) ALONE (uniform sampling makes cue identity/validity irrelevant), whereas CDCC's confidence is (i) gated by validity COHERENCE C = (W_win - W_lose)/(W_win + W_lose) with convex weights w = (v-0.5)^rho, and (ii) diluted by S = M/D^gamma where D = nA+nB, so tied cues are completely irrelevant to CDCC but occupy sampling slots for LSNC. This design attacks both differences and, crucially, contains two ORDERING REVERSALS.

DESIGN: 7 experts, scrambled validity spread [0.94, 0.58, 0.86, 0.63, 0.72, 0.80, 0.55] (descending rank f0 > f2 > f5 > f4 > f3 > f1 > f6, so validity rank is decoupled from screen position and positional heuristics cannot mimic validity coherence). One very high (0.94) and two very low (0.58, 0.55) validities make the convex weighting bite: w = (v-.5)^3 gives f0 = .0852 vs f6 = .000125, a ~680:1 ratio. 6 configurations x 2 side mirrors = 12 unique pairs x 8 repetitions = 96 trials.

CELLS (A-favouring version listed; the even-numbered mirror swaps A and B):
- T1/T2  SPARSE-COHERENT (nA,nB,nT) = (2,1,4): A wins f0(.94), f2(.86); B wins f6(.55); rest tied.
- T3/T4  SPARSE-CONFLICT  (2,1,4): A wins f1(.58), f6(.55); B wins f0(.94); rest tied. IDENTICAL composition to T1/T2.
- T5/T6  DENSE-COHERENT   (4,2,1): A wins f0,f2,f5,f4; B wins f3,f1; f6 tied.
- T7/T8  DENSE-CONFLICT   (4,2,1): A wins f1,f3,f4,f6; B wins f0,f2; f5 tied. IDENTICAL composition to T5/T6.
- T9/T10 UNANIMOUS-SPARSE (2,0,5): A wins f2,f4; five tied cues, no cue opposes.
- T11/T12 UNANIMOUS-DENSE (4,0,3): A wins f0,f3,f5,f6; three tied cues, no cue opposes.

QUANTITATIVE PREDICTIONS.
CDCC (beta=2.6, gamma=1.5, kappa=0.7, rho=3.0, eps=0.10): (2,1,4) coherent 0.708 vs conflict 0.541; (4,2,1) coherent 0.651 vs conflict 0.540; (2,0,5) 0.928; (4,0,3) 0.888.
LSNC (exact enumeration, k=2.5, nu=0.20, eps=0): (2,1,4) 0.579 for BOTH coherent and conflict; (4,2,1) 0.621 for BOTH; (2,0,5) 0.680; (4,0,3) 0.770.

THREE ORTHOGONAL DIAGNOSTICS:
(1) COHERENCE (parameter-free for LSNC): holding (nA,nB,nT) exactly fixed, CDCC predicts a 11-17 point coherent-minus-conflict gap that cannot be switched off (kappa >= 0.4 by hypothesis), while LSNC predicts EXACTLY ZERO for every k, nu, epsilon. Each subject contributes 32 coherent and 32 conflict trials on matched compositions, so this contrast is estimated within subject with high power.
(2) SPARSITY REVERSAL among the coherent cells: CDCC's D^gamma dilution makes (2,1,4) [D=2 winners vs 1 loser, D=3] MORE confident than (4,2,1) [D=6] despite the smaller raw margin (0.708 > 0.651); LSNC's tie dilution predicts the OPPOSITE order (0.579 < 0.621) because the sparse cell wastes 4 of 7 sampling slots on tied cues. Same reversal for the unanimous cells: CDCC predicts (2,0,5) > (4,0,3) (0.928 > 0.888, and 0.911 > 0.826 at gamma=1.75), LSNC predicts (4,0,3) > (2,0,5) (0.770 > 0.680). A margin-2 profile beating a margin-4 profile is impossible for LSNC under any parameters.
(3) CEILING: on the clean unanimous-sparse cell CDCC allows 0.90-0.95 adherence, whereas LSNC is capped near 0.68 (and at most ~0.79 even at k=3.5, nu=0.12) because a small noisy sample easily reverses the sign. Adherence above ~0.85 on T9/T10 is outside LSNC's reachable range.

CONTROLS: (i) every configuration appears in an A-favouring and a B-favouring mirror, so side bias cancels exactly; (ii) tied cues are encoded sometimes 1-1 and sometimes 0-0 so the number of displayed 1s does not covary with composition or coherence; (iii) validity rank is scrambled relative to display position; (iv) no pair is degenerate: every trial has a strict count winner and requires reading several ratings; (v) systematic BELOW-chance adherence in the conflict cells (i.e. following the single best cue against the tally) would falsify BOTH theories in favour of one-reason/WADD integration, so the design is self-diagnostic.

INTERPRETATION: a large coherent > conflict gap that never dips below 0.5, together with sparse > dense ordering at matched or even smaller margins and near-0.9 adherence on unanimous sparse profiles, uniquely supports CDCC. Flat adherence across the coherence manipulation with a (4,0,3) > (2,0,5) > (4,2,1) > (2,1,4) composition ordering and a ceiling near 0.7 uniquely supports LSNC.

[6] GOAL: dissociate RDTI-MCR (advocated) from Count-Direction with Conflict-Graded Confidence (CDCC, competing). The two models agree on nothing important here: CDCC makes DIRECTION purely count-based (ties => exactly 0.5, parameter-free) and lets validity affect only confidence, never sign, and never drops adherence to the count-winner below 0.5. RDTI-MCR instead (i) median-splits the panel into a discounted 'star' tier and a full-weight 'marginal' tier, (ii) applies a contrarian discount proportional to the count margin, and (iii) releases the star discount only when the discriminating set is essentially a single voice (D=1).

PANEL: 6 experts, validities [0.95, 0.57, 0.88, 0.60, 0.66, 0.83] (screen order deliberately scrambled). Descending rank f0(.95) > f2(.88) > f5(.83) > f4(.66) > f3(.60) > f1(.57), so the RDTI median split is unambiguous: STAR = {f0,f2,f5}, MARGINAL = {f4,f3,f1}, with a wide .83/.66 gap at the split. CDCC's convex weights w=(v-.5)^3 are .091/.066/.036 for the stars versus .0041/.0010/.0003 for the marginals (a >20:1 ratio), so its coherence gate is driven almost entirely by the stars. n=6 makes RDTI's panel-size dilution term exactly 1, removing a nuisance factor. 6 configurations x 2 side mirrors = 12 unique pairs x 8 repetitions = 96 trials.

CELLS (A-favouring version listed; the following trial is the exact A/B mirror):
C1 STAR-SOLO: only f0(.95) discriminates, A wins it (D=1).
C2 MARGINAL-SOLO: only f1(.57) discriminates, A wins it (D=1).
C3 TIE-1v1 MISMATCHED: A wins f1(.57), B wins f0(.95); tally tied.
C4 TIE-2v2 MISMATCHED: A wins f3(.60) and f4(.66), B wins f0(.95) and f2(.88); tally tied.
C5 DENSE STAR-MAJORITY: A wins f0,f2,f5 (all stars), B wins f3,f4 (marginals); nA=3, nB=2, f1 tied.
C6 DENSE MARGINAL-MAJORITY: A wins f1,f3,f4 (all marginals), B wins f0,f2 (stars); nA=3, nB=2, f5 tied.

QUANTITATIVE PREDICTIONS, P(choose A) at nominal parameters.
RDTI-MCR (eps_w=.18, beta=2.9, delta=.075, tau=.9, lapse=.115, count_bias=.45, psi=.60, kappa_d=.30): C1 = 0.79; C2 = 0.87; C3 = 0.86; C4 = 0.93; C5 = 0.07; C6 = 0.94.
CDCC (beta=2.6, gamma=1.5, kappa=0.7, rho=3.0, eps=0.10): C1 = 0.945; C2 = 0.945; C3 = 0.500; C4 = 0.500; C5 = 0.600; C6 = 0.521.

THREE ORTHOGONAL DIAGNOSTICS:
(1) TALLY-TIED, TIER-MISMATCHED CELLS (C3, C4) are parameter-free for CDCC: nA = nB so it MUST be exactly 0.5 for every beta, gamma, kappa, rho, epsilon. RDTI predicts strong, systematic choice of the option backed only by the LOW-validity marginal experts (0.78-0.93 across its whole parameter box for C3, 0.85-0.97 for C4). Each subject contributes 32 such trials, so even a 10-point deviation from 0.5 is decisive. Note the sign is also diagnostic against WADD/TTB, which would predict the opposite (below 0.5), so the cell is self-diagnostic three ways.
(2) DIRECTION REVERSAL ON DENSE CONFLICT (C5 vs C6). Both cells have the same count margin (+1 for A) and the same D=5, so CDCC must put BOTH above 0.5 for A (0.52-0.70 depending on gamma/beta; conflict can only erode confidence toward 0.5, never past it). RDTI's redundancy discount plus contrarian majority discount makes C5 collapse to ~0.07 while C6 rises to ~0.94 -- an ~0.87 swing between two profiles CDCC treats as same-direction. Observing sub-chance adherence to the count winner in C5 falsifies CDCC outright; observing C5 and C6 both mildly above 0.5 with C5 > C6 falsifies RDTI.
(3) GRADED REDUNDANCY RELEASE (C1 vs C2). Both are single-discriminating-cue trials with the count winner also winning the only cue, so CDCC's S = 1, C = +1 and its prediction is EXACTLY IDENTICAL for the two cells regardless of parameters (~0.94). RDTI predicts C2 > C1 by 0.08-0.23 (the solo star is only partially released from its redundancy discount: eps_eff = eps_w + (1-eps_w)*psi < 1), and predicts C1 to sit noticeably below CDCC's ~0.94 ceiling. Comparing C1 (0.79-ish, star alone) with C5 (0.07, same star embedded in a dense display) also isolates RDTI's claim that a star is discounted only when there is a chorus to be redundant with.

CONTROLS: every configuration appears in an A-favouring and a B-favouring mirror so side bias cancels exactly; tied cues are coded sometimes 1-1 and sometimes 0-0 so the number of displayed 1s does not track composition; validity rank is decoupled from screen position; no pair is degenerate (every trial requires reading several ratings and no cell has the two models making the same prediction).

[7] GOAL: dissociate Count-Direction with Conflict-Graded Confidence (CDCC, advocated) from Redundancy-Discounted Tiered Integration with Majority Discounting (RDTI-MCR, competing). CDCC makes DIRECTION purely count-based (exact 0.5 on tally ties, parameter-free; never sub-chance adherence to the count winner) and lets validity enter only as a confidence gate, with dilution S = M/D^gamma. RDTI-MCR median-splits the panel into a discounted 'star' tier and a full-weight 'marginal' tier, adds a contrarian discount proportional to the count margin, and releases the star discount only for essentially solo displays (D=1). This design attacks the three points where their signs, not just their magnitudes, diverge.

PANEL: 8 experts, validities in scrambled screen order [0.95, 0.56, 0.85, 0.60, 0.65, 0.90, 0.53, 0.80]. Descending rank f0(.95) > f5(.90) > f2(.85) > f7(.80) >> f4(.65) > f3(.60) > f1(.56) > f6(.53), so RDTI's median split is unambiguous (STAR = {f0,f5,f2,f7}, MARGINAL = {f4,f3,f1,f6}) with a wide .80/.65 gap at the split, and CDCC's convex weights w=(v-.5)^3 are .091/.064/.043/.027 for the stars versus .0034/.0010/.0002/.00003 for the marginals (>25:1), so its coherence gate is star-driven. n=8 (rather than 6) also puts RDTI's capacity term (n/6)^tau above 1, which shrinks its evidence term relative to the count-margin discount and sharpens the predicted sign reversals. 6 configurations x 2 side mirrors = 12 unique pairs x 8 repetitions = 96 trials.

CELLS (A-favouring version listed; the next trial is its exact A/B mirror):
C1 (T1/T2) TALLY-TIED 3v3, TIER-MISMATCHED: A wins marginals f4,f3,f1; B wins stars f0,f5,f2; f6,f7 tied.
C2 (T3/T4) TALLY-TIED 2v2, TIER-MISMATCHED: A wins the two weakest experts f6(.53), f1(.56); B wins the two best f0(.95), f5(.90); f2,f3,f4,f7 tied.
C3 (T5/T6) DENSE STAR-MAJORITY (D=6, M=+2): A wins all four stars f0,f5,f2,f7; B wins marginals f4,f3; f1,f6 tied.
C4 (T7/T8) DENSE MARGINAL-MAJORITY (D=6, M=+2): A wins all four marginals f4,f3,f1,f6; B wins stars f0,f5; f2,f7 tied.
C5 (T9/T10) SPARSE UNANIMOUS STAR (D=2, M=+2): A wins only f0(.95) and f5(.90); six cues tied.
C6 (T11/T12) MID STAR-MAJORITY (D=4, M=+2): A wins stars f0,f5,f2; B wins marginal f4; f1,f3,f6,f7 tied.

QUANTITATIVE PREDICTIONS, P(choose A).
CDCC (beta=2.6, gamma=1.5, kappa=0.7, rho=3.0, eps=0.10): C1 = 0.500, C2 = 0.500, C3 = 0.650, C4 = 0.532, C5 = 0.928, C6 = 0.754.
RDTI-MCR (eps_w=.18, beta=2.9, delta=.075, tau=.9, lapse=.115, count_bias=.45, psi=.60, kappa_d=.30): C1 = 0.935, C2 = 0.911, C3 = 0.086, C4 = 0.941, C5 = 0.487, C6 = 0.180.

THREE ORTHOGONAL DIAGNOSTICS:
(1) TALLY-TIED TIER-MISMATCHED CELLS (C1, C2). CDCC MUST be exactly 0.5 for every beta, gamma, kappa, rho, epsilon (nA = nB returns the coin flip before any validity computation). RDTI predicts strong, systematic choice of the option backed only by the LOW-validity marginal experts (0.85-0.97 across its whole parameter box, since E = k(1 - eps_w) > 0 with eps_w <= 0.34 and the count-margin term is exactly zero). Each subject contributes 32 such trials, so even a 10-point deviation from 0.5 is decisive; note the sign also separates RDTI from WADD/TTB, which would predict below 0.5 here, so the cell is self-diagnostic three ways.
(2) SIGN REVERSAL ON STAR-BACKED MAJORITIES (C3 and C6 versus C4). C3, C4 and C6 all have count margin +2 for A. CDCC must put ALL THREE above 0.5 (conflict can only erode confidence toward chance, never past it) and predicts the coherent star-backed cells ABOVE the conflict cell: C3 0.65 and C6 0.75 versus C4 0.53. RDTI predicts C3 = 0.09 and C6 = 0.18 (sub-chance: 4*eps_w <= 1.36 < 2 so the tier-weighted evidence is negative whenever the count winner's wins are the stars, and the contrarian majority discount pushes it further down), while C4 = 0.94 -- an ~0.85 swing between profiles CDCC treats as same-direction and similarly signed. Observing systematic sub-chance adherence to the count winner in C3/C6 falsifies CDCC outright; observing C3 > C4 > 0.5 falsifies RDTI.
(3) DILUTION LADDER AT CONSTANT MARGIN AND CONSTANT COHERENCE (C5, C6, C3: D = 2, 4, 6 with M = +2 and the count winner always holding the stars). CDCC's S = M/D^gamma predicts a steep monotone decline 0.93 -> 0.75 -> 0.65, with near-ceiling (0.90-0.95) adherence on the sparse unanimous star cell C5 -- the classic CDCC signature that sparse clean evidence beats dense larger-D evidence. RDTI predicts the same ordering but hugging or crossing below chance (0.49 -> 0.18 -> 0.09), because its graded redundancy release is essentially confined to D = 1 (relax = psi*exp(-(D-1)/kappa_d) is already <= 0.03 at D = 2) while the count-margin discount of 0.45*2 applies in all three cells. Thus C5 is the cleanest magnitude test: CDCC needs >= 0.85, RDTI at nominal parameters sits at chance and only reaches ~0.8 at the extreme corner of its box.

CONTROLS: every configuration appears in an A-favouring and a B-favouring mirror, so side bias cancels exactly; tied cues are coded sometimes 1-1 and sometimes 0-0 so the number of displayed 1s does not track composition or tier; validity rank is scrambled relative to screen position so positional heuristics cannot mimic tiering; no pair is degenerate (every trial requires reading several ratings) and no cell has the two models making the same prediction.

INTERPRETATION: exact chance on C1/C2, above-chance adherence to the count winner in every one of C3-C6 with the coherent cells highest, and a steep sparse-to-dense decline 0.93/0.75/0.65 at fixed margin uniquely supports CDCC. Strong marginal-tier-driven preferences on the tally-tied cells plus sub-chance choice of the star-backed majority in C3/C6 uniquely supports RDTI-MCR.

[8] GOAL: dissociate RDTI-MCR (advocated) from GVIT-CAP-S4 (competing). Both models discount high-validity experts as 'redundant', both release the discount when only one cue discriminates, and both compress evidence by D, so generic conflict trials are useless. They differ on exactly three things, and the panel is engineered so that each difference produces a SIGN reversal or a parameter-free contrast.

PANEL: 6 experts, validities in scrambled screen order [f0=.95, f1=.78, f2=.82, f3=.55, f4=.76, f5=.80]. Descending rank f0>f2>f5>f1>f4>f3. Two structural properties are deliberate. (a) The median split falls INSIDE a tight cluster (.80 vs .78), so RDTI's coarse ordinal tiering (STAR={f0,f2,f5} weight eps_w; MARGINAL={f1,f4,f3} weight 1) cuts sharply between two nearly identical stated validities. (b) There is one lone floor expert (f3=.55) far below a dense upper cluster, so GVIT's strongly saturating inversion f(s)=s^omega (omega .26-.42) gives ALL of f0,f2,f5,f1,f4 nearly the same (and, for lam>1.1, negative) weight while f3 alone keeps weight +1. Hence GVIT ~ 'follow the least accurate expert, mildly opposed by everyone else' whereas RDTI ~ 'follow the bottom-half tier, stars count 0.18'. Validity rank is decoupled from screen position; n=6 keeps RDTI's capacity term exactly 1 and is below GVIT's 7.5 overload threshold, removing nuisance factors. 6 configurations x 2 exact A/B mirrors = 12 unique pairs x 8 repetitions = 96 trials. Tied cues are coded sometimes 1-1 and sometimes 0-0.

CELLS (A-favouring version listed; the following trial is its exact mirror):
K1 (T1/T2) TALLY-TIED, MEDIAN-BOUNDARY: A wins f5(.80, star tier), B wins f1(.78, marginal tier); 4 cues tied.
K2 (T3/T4) TALLY-TIED 2v2, WIDE MISMATCH: A wins f4(.76)+f3(.55) [marginals], B wins f0(.95)+f2(.82) [stars].
K3 (T5/T6) MID-CLUSTER vs FLOOR: A wins f1(.78)+f4(.76), B wins f3(.55); f0,f2,f5 tied (D=3, margin +1).
K4 (T7/T8) DENSE MIXED (D=6, margin +2): A wins f0,f2,f5 (all stars) AND f3 (floor); B wins f1(.78),f4(.76).
K5 (T9/T10) SOLO STAR: only f0(.95) discriminates, A wins it (D=1).
K6 (T11/T12) SOLO FLOOR: only f3(.55) discriminates, A wins it (D=1).

QUANTITATIVE PREDICTIONS, P(choose A) at nominal parameters.
RDTI-MCR (eps_w=.18, beta=2.9, delta=.075, tau=.9, lapse=.115, count_bias=.45, psi=.60, kappa_d=.30): K1=0.14, K2=0.93, K3=0.81, K4=0.16, K5=0.77, K6=0.85.
GVIT-CAP-S4 (lam=1.45, beta=6.7, gamma=.90, kappa_d=.10, psi=.40, chi=.95, eps=.17, omega=.34): K1=0.500, K2=0.500, K3=0.12, K4=0.62, K5=0.914, K6=0.914.

FOUR ORTHOGONAL DIAGNOSTICS:
(1) TALLY-TIED CELLS K1, K2 are parameter-free for GVIT: nA=nB triggers the count gate and returns EXACTLY 0.5 for every lam, beta, gamma, kappa_d, psi, chi, eps, omega. RDTI predicts strong, systematic preference for the marginal-tier-backed option: across its entire parameter box K1 gives P(A)=0.08-0.27 (i.e. 73-92% choice of the f1-backed option) and K2 gives 0.82-0.95. Each subject contributes 32 such trials, so even a 10-point deviation from 0.5 is decisive. K1 additionally localises the tier boundary to the .80/.78 crack: a subject who tiers at all but places the cut elsewhere will be at chance on K1 while still strong on K2.
(2) SIGN REVERSAL K3 (mid cluster vs floor expert). RDTI: all three discriminating cues are in the marginal tier, so E=+1 exactly regardless of eps_w; P(A)=0.71-0.93 over the whole box. GVIT: the saturating inversion makes f1 and f4 (.78/.76) almost as 'redundant' as the stars while f3 keeps full weight, so E<0 for every admissible lam/omega; P(A)=0.10-0.25. An ~0.6-0.8 swing on identical stimuli. Observing that subjects follow the two middling experts over the single worst expert falsifies GVIT's saturating-inversion commitment; following the .55 expert against two .76-.78 experts falsifies RDTI.
(3) SIGN REVERSAL K4 (dense, margin +2 for A). RDTI: E=3*eps_w+1-2 <= 0.02, and the contrarian consensus discount subtracts count_bias*2, giving P(A)=0.10-0.40 (always sub-chance) - the star-backed majority is REJECTED. GVIT: the floor expert f3 favours A with weight +1 while the opposing mids carry negative weight, so E>0 and P(A)=0.50-0.85. This is the cleanest test of RDTI's majority-discount plus fixed two-valued tiering against GVIT's graded inverse-validity weights.
(4) SOLO CELLS K5 vs K6 (both D=1). For GVIT, lam_eff = lam*(D-1)/((D-1)+kappa_D) = 0 exactly at D=1, so ALL weights are 1 and the two cells have IDENTICAL predictions (~0.86-0.99) for every parameter setting - a parameter-free equality. RDTI releases the star discount only partially (eps_eff = eps_w+(1-eps_w)*psi < 1), so it predicts solo-floor > solo-star by 0.05-0.25 (0.85 vs 0.77 nominally) and predicts the solo-star cell to sit well below GVIT's >=0.86 floor. Comparing K5 (star alone, ~0.77) with K4 (same star embedded in a chorus, ~0.16) also isolates RDTI's graded-relaxation claim.

CONTROLS: every configuration appears in an A-favouring and a B-favouring mirror so side bias cancels exactly; the number of displayed 1s does not track composition; validity rank is scrambled relative to screen position; no pair is degenerate and no cell has the two models predicting the same thing.

INTERPRETATION: exact chance on K1/K2 plus following the .55 expert in K3 and the floor-backed option in K4, with equal near-ceiling adherence on both solo cells, uniquely supports GVIT-CAP-S4. Strong marginal-tier-driven choices on the tally-tied cells, adherence to the two middling experts in K3, sub-chance choice of the star-backed majority in K4, and a lower solo-star than solo-floor adherence uniquely support RDTI-MCR.

[9] GOAL: dissociate GVIT-CAP-S4 (advocated) from RDTI-MCR (competing). Both models discount high-validity experts as 'redundant', both release the discount when a single cue discriminates, and both compress evidence by the number of discriminating cues, so ordinary conflict trials are useless. They differ on exactly three points, and the panel is engineered so each difference produces either a parameter-free equality that only one model can honour, or a sign reversal.

PANEL: 7 experts, validities in deliberately scrambled screen order [f0=.74, f1=.95, f2=.55, f3=.78, f4=.70, f5=.52, f6=.90]. Descending rank f1(.95) > f6(.90) > f3(.78) > f0(.74) > f4(.70) > f2(.55) > f5(.52). Two structural properties are deliberate. (a) RDTI's median split (k = 7//2 = 3) makes STAR = {f1,f6,f3} (weight eps_w) and MARGINAL = {f0,f4,f2,f5} (weight 1), and CRUCIALLY the marginal tier contains BOTH a middling pair (.74/.70) and a genuine bottom pair (.55/.52); because RDTI's weights take only two values, these two pairs are interchangeable for it. (b) GVIT's panel-relative saturating inversion f(s)=s^omega (omega .26-.42) with vmin=.52, vmax=.95 assigns s^.34 = 1.00/.959/.843/.796/.744/.405/0 to f1/f6/f3/f0/f4/f2/f5, so the middling experts (.74,.70) are discounted almost as heavily as the stars (weights ~ -0.07 and ~ +0.00 at lam_eff=1.34) while the two bottom voices keep large positive weight (+0.46 and +1.00). Hence GVIT sharply separates a mid-backed majority from a bottom-backed one, whereas RDTI cannot. n=7 is below GVIT's 7.5 capacity threshold (o=.076), so the inversion is intact and only mildly noisy. 6 configurations x 2 exact A/B mirrors = 12 unique pairs x 8 repetitions = 96 trials; tied cues are coded sometimes 1-1 and sometimes 0-0 so the number of displayed 1s does not track composition.

CELLS (A-favouring version listed; the following trial is its exact mirror):
C1 (T1/T2) MID-MAJORITY vs LONE STAR (D=3, margin +1): A wins f0(.74) and f4(.70); B wins f1(.95); f2,f3,f5,f6 tied.
C2 (T3/T4) BOTTOM-MAJORITY vs LONE STAR (D=3, margin +1): A wins f2(.55) and f5(.52); B wins f1(.95); f0,f3,f4,f6 tied. IDENTICAL tier composition to C1 for RDTI.
C3 (T5/T6) TALLY-TIED 2v2, TIER-MISMATCHED: A wins f2(.55), f5(.52); B wins f1(.95), f6(.90); f0,f3,f4 tied.
C4 (T7/T8) MID-PAIR vs FLOOR (D=3, margin +1): A wins f0(.74), f4(.70); B wins f5(.52); f1,f2,f3,f6 tied.
C5 (T9/T10) SOLO STAR (D=1): only f1(.95) discriminates, A wins it.
C6 (T11/T12) SOLO FLOOR (D=1): only f5(.52) discriminates, A wins it.

QUANTITATIVE PREDICTIONS, P(choose A) at nominal parameters.
GVIT-CAP-S4 (lam=1.45, beta=6.7, gamma=.90, kappa_d=.10, psi=.40, chi=.95, eps=.17, omega=.34): C1=0.63, C2=0.90, C3=0.500 (exact), C4=0.15, C5=0.91, C6=0.91.
RDTI-MCR (eps_w=.18, beta=2.9, delta=.075, tau=.9, lapse=.115, count_bias=.45, psi=.60, kappa_d=.30): C1=0.92, C2=0.92 (exactly equal), C3=0.92, C4=0.83, C5=0.75, C6=0.84.

FOUR ORTHOGONAL DIAGNOSTICS:
(1) MID vs BOTTOM MAJORITY (C1 vs C2) is PARAMETER-FREE for RDTI: both cells have two marginal-tier winners, one star-tier loser, D=3, margin +1, same n, so E = 2 - eps_w and the predicted P(A) is IDENTICAL for every eps_w, beta, delta, tau, lapse, count_bias, psi, kappa_d. GVIT's saturating inversion predicts a large gap: E(C1) = 1 - lam_eff*(s0^om + s4^om - 1) ranges 0.06-0.53 over its whole box (and can go slightly negative at omega=.26, high lam), while E(C2) = 1 + lam_eff*(1 - s2^om) ranges 1.5-2.0, giving C1 ~ 0.52-0.82 versus C2 ~ 0.86-0.96, i.e. a C2-minus-C1 gap of about 0.15-0.40 that RDTI must set to exactly 0. Each subject contributes 16 trials per cell, so this within-subject contrast is decisive.
(2) TALLY-TIED TIER-MISMATCHED CELL (C3) is parameter-free for GVIT: nA = nB triggers the count gate and returns EXACTLY 0.5 for every lam, beta, gamma, kappa_d, psi, chi, eps, omega, with no validity tie-break. RDTI predicts a strong, systematic preference for the option backed only by the two WORST experts (E = 2(1-eps_w) >= 1.32, margin = 0 so the contrarian discount is inert): 0.85-0.96 across its entire box. Note the sign is also opposite to WADD/TTB (which would go below 0.5 here), so the cell is self-diagnostic three ways.
(3) SIGN REVERSAL ON C4 (mid pair against the floor expert). RDTI: all three discriminating cues are marginal-tier, so E = +1 EXACTLY regardless of eps_w, and P(A) = 0.72-0.92 over the box. GVIT: E = 1 - lam_eff*(s0^om + s4^om) is negative for every admissible lam and omega (-0.33 to -1.9), so P(A) = 0.10-0.35 - subjects should follow the single .52 expert AGAINST two .70-.74 experts. Comparing C4 with C1, which is the same A-side profile with only the opponent changed from the floor expert to the star, isolates GVIT's claim that a lone .95 expert is heard as counter-indicative while a lone .52 expert is heard as decisive: GVIT predicts a ~0.48 swing (0.63 vs 0.15), RDTI predicts both high and close (0.92 vs 0.83).
(4) SOLO CELLS C5 vs C6 (both D=1). For GVIT lam_eff = lam_n*(D-1)/((D-1)+kappa_d) = 0 exactly at D=1, so all weights equal 1 and the two cells have IDENTICAL predictions (0.865-0.965) for every parameter setting - a parameter-free equality, and a high floor. RDTI releases the star discount only partially (eps_eff = eps_w + (1-eps_w)*psi < 1), so it predicts solo-floor > solo-star (nominally 0.84 vs 0.75) and allows the solo-star cell to fall as low as ~0.5, below GVIT's reachable floor.

CONTROLS: every configuration appears in an A-favouring and a B-favouring mirror so side bias cancels exactly; validity rank is decoupled from screen position so positional heuristics cannot mimic validity-based weighting; tied cues alternate between 1-1 and 0-0 coding; no pair is degenerate (every trial requires reading several ratings) and no cell has the two models predicting the same thing.

INTERPRETATION: exact chance on C3, a large bottom-backed-minus-mid-backed gap (C2 >> C1) at matched composition, sub-chance choice of the mid-backed majority in C4, and equal near-ceiling adherence on the two solo cells uniquely support GVIT-CAP-S4. Equal adherence on C1 and C2, strong marginal-tier-driven choice on the tally-tied C3, above-chance adherence in C4, and solo-floor exceeding solo-star uniquely support RDTI-MCR.

[10] GOAL: dissociate SR-DT-6b (advocated) from GVIT-CAP-S4 (competing). Both models share a hard count gate (tally ties -> exactly 0.5), both invert validity (stars discounted), both release the discount for solo voices, and both compress by the number of discriminating cues, so ordinary conflict trials and tally-tied trials are useless (they agree exactly). They differ on exactly three things, and the panel is engineered so each difference yields a sign reversal or a huge magnitude gap at a FIXED panel size (n=6, which is below both models' capacity/overload terms: SR-DT's size_fac = (6/6)^tau = 1 and GVIT's o ~ 0, removing all panel-size nuisance).

PANEL (screen order deliberately scrambled so validity rank is decoupled from position): [p0=.78, p1=.95, p2=.68, p3=.84, p4=.90, p5=.72]. Two structural choices are crucial.
(a) THE FLOOR IS HIGH (v_min = .68, well above chance). GVIT anchors seniority at the panel minimum, s=(v-v_min)/(v_max-v_min), so the .68 expert gets s=0 and therefore FULL weight 1.0, while SR-DT anchors at the chance baseline, s=(v-.5)/(v_max-.5), so the .68 expert already has s=0.40 and is itself partly discounted (w~0.68).
(b) THE PANEL IS DENSELY PACKED AT THE TOP (.95/.90/.84/.78/.72/.68) so the two models' exponents bite in opposite directions: GVIT's strongly concave f(s)=s^omega (omega .26-.42) makes the MIDDLING experts (.84,.78,.72) nearly as redundant as the star, while SR-DT's superlinear s^omega (omega>1) leaves them with substantial positive news value. Nominal weights (SR-DT lam=1.6, omega=1.7, floor=-.30 / GVIT lam=1.45, omega=.34): .95 -0.30/-0.41, .90 -0.28/-0.32, .84 +0.03/-0.18, .78 +0.30/-0.02, .72 +0.55/+0.26, .68 +0.68/+1.00. The difference vector is large and sign-flipping exactly on the mid band and on the floor voice.
(c) LONE-VOICE RULE. SR-DT boosts any side with exactly ONE discriminating cue to w = phi0 + (phi-phi0)*s > 1 (seniority-graded), because redundancy requires SAME-SIDE corroboration; GVIT's chorus criterion uses the TOTAL number of discriminating cues D, so a lone dissenting star embedded in a D=3 display is still discounted to w ~ -0.38 (counter-indicative).

6 configurations x 2 exact A/B mirrors = 12 unique pairs x 8 repetitions = 96 trials. Tied cues are coded sometimes 1-1 and sometimes 0-0 so displayed 1-counts do not track composition. Every cell has a strict count winner (no tally ties, where both models trivially predict .5).

CELLS (A-favouring version listed; the following trial is its exact mirror). P(choose A) at nominal parameters: SR-DT (lam=1.6, omega=1.7, beta=13.5, gamma_d=.68, gamma_c=.66, kappa=.05, phi=1.40, phi0=1.08, w_floor=-.30, eps=.09) vs GVIT (lam=1.45, beta=6.7, gamma=.90, kappa_d=.10, omega=.34, eps=.17).
A1 (T1/T2) MID-TRIO vs STAR+FLOOR (mA=3, mB=2, D=5): A wins .84,.78,.72; B wins .95 and .68; .90 tied. SR-DT E=+0.49 -> 0.72; GVIT E=-0.53 -> 0.32. SIGN REVERSAL (this profile maximises the two models' weight-difference vector: A holds the three mid experts SR-DT values and GVIT nullifies; B holds the floor voice GVIT values at full strength).
C2 (T3/T4) MID-PAIR vs LONE FLOOR (D=3): A wins .84,.78; B wins .68 alone. SR-DT 0.08; GVIT 0.12. (Deliberate near-agreement: this is the CONTROL for D3.)
D3 (T5/T6) MID-PAIR vs LONE STAR (D=3): identical A side to C2, only the identity of the single dissenter changes to .95. SR-DT 0.06; GVIT 0.62. The KEY CONTRAST is therefore the (lone-star minus lone-floor) difference with the majority held exactly fixed: SR-DT must give D3 <= C2 (its lone voice is boosted above face value either way, more so for the senior one), i.e. a NEGATIVE or null contrast; GVIT must give D3 >> C2 (+0.35 to +0.50 across its whole box), because the lone star carries weight 1-lam_eff < 0 while the lone floor carries +1. The SIGN of this within-subject contrast is the single cleanest diagnostic and each subject contributes 32 trials to it.
E4 (T9? no: T7/T8) TOP-PAIR+FLOOR vs MID-PAIR (mA=3, mB=2, D=5): A wins .95,.90,.68; B wins .78,.72; .84 tied. SR-DT E=-0.78 -> 0.19 (REJECTS the count winner); GVIT E~+0.02 -> 0.51 (chance) across its box (0.50-0.55). This is the mirror-image of A1 in weight space and separates the two anchorings: for GVIT the floor voice's full weight exactly offsets the two negative stars, for SR-DT it does not.
F5 (T9/T10) LOW-TRIO vs TOP-PAIR (mA=3, mB=2): A wins .78,.72,.68; B wins .95,.90. Both models follow A, but SR-DT 0.94 vs GVIT 0.87: an evidence-saturated ceiling cell that anchors the lapse rate (SR-DT eps <= .13 forces >= .935; GVIT eps up to .27 allows .865), so the two models' ceilings are separately identified and cannot be traded off against slope.
G6 (T11/T12) UNOPPOSED MID-TRIO (mA=3, mB=0, D=3): A wins .84,.78,.72, nothing opposes. SR-DT 0.95 (unopposed majorities grow in subjective strength: no dissent term, sub-linear D dilution); GVIT 0.57 (the same three cues are nearly fully discounted, and D^gamma still divides). A ~0.38 gap on a display with ZERO conflict. G6 is also the unopposed control for A1: SR-DT predicts 0.72 -> 0.95 when the two opposing cues are neutralised, GVIT predicts 0.32 -> 0.57, i.e. both rise but only SR-DT ends near ceiling and only GVIT stays near chance.

INTERPRETATION. Following the mid-backed trio in A1 (>0.6), rejecting it in E4 (<0.3), near-ceiling adherence in the unopposed G6 (>0.9), and a NON-POSITIVE lone-star-minus-lone-floor contrast (D3 <= C2, both far below .5) uniquely support SR-DT-6b. Reversing A1 (<0.4), sitting at chance in E4, only weakly above chance in G6 (~0.55-0.65), and a large POSITIVE D3-minus-C2 swing (following two middling experts over one .95 expert while abandoning them against one .68 expert) uniquely support GVIT-CAP-S4. CONTROLS: exact A/B mirroring cancels side bias; validity rank is scrambled relative to screen position; tie coding alternates 1-1 / 0-0; no cell is degenerate and no tally-tied cells are included (both models would trivially predict 0.5 there).

[11] GOAL: dissociate GVIT-CAP-S4 (advocated) from SR-DT-6b (competing). The two models share a hard count gate (tally ties -> exactly 0.5), both invert stated validity, both release the discount for solo voices, and both compress evidence by the number of discriminating cues, so tally-tied cells and generic conflict cells are useless (they agree exactly). They differ on exactly three things: (i) the SHAPE of the redundancy discount (GVIT: strongly concave s^omega, omega~.26-.42, so low-MID experts are discounted almost as hard as stars; SR-DT: superlinear s^omega, omega>1, so low-mid experts keep large positive news value and only the top one or two voices go negative, bounded by w_floor); (ii) the CHORUS criterion (GVIT's lam_eff uses the TOTAL number of discriminating cues D, so a lone dissenting star inside a D=3 display is still counter-indicative, w = 1-lam_eff < 0; SR-DT's redundancy is SAME-SIDE, so any side with exactly one cue is BOOSTED above face value, w = phi0+(phi-phi0)*s >= 1.15 for a lone star); (iii) DISSENT dilution (SR-DT divides by (1+m_lose)^gamma_c so unopposed majorities approach ceiling; GVIT has no dissent term and simply divides by D^gamma).

PANEL (screen order deliberately scrambled so validity rank is decoupled from position): [f0=.72, f1=.95, f2=.58, f3=.88, f4=.52, f5=.68]; descending rank f1 > f3 > f0 > f5 > f2 > f4. Three structural choices are essential. (a) The floor is at .52, essentially chance, so BOTH models' seniority anchors (GVIT: v_min; SR-DT: 0.5) nearly coincide -- the dissociation is then driven purely by the exponent and the chorus rule, not by an anchoring artefact. (b) There is a LOW-MID band (.72, .68, .58) sitting at panel-relative s = .47/.37/.14, exactly where a concave and a convex exponent diverge maximally: GVIT weights them -0.09 / -0.01 / +0.28 (at lam_eff=1.38) while SR-DT weights them +0.54 / +0.67 / +0.92. (c) n = 6 keeps GVIT's overload term o ~ 0 and SR-DT's size_fac = (6/6)^tau = 1, removing all panel-size nuisance. Only the floor voice f4 (.52) gets full weight 1.0 in GVIT (s = 0 exactly, parameter-free) -- so GVIT is, in this panel, 'follow the least accurate expert, mildly opposed by everyone else', whereas SR-DT is 'everyone below .80 speaks with near-full voice, and a solitary voice speaks louder still'.

DESIGN: 6 configurations x 2 exact A/B mirrors = 12 unique pairs x 8 repetitions = 96 trials. Every cell has a strict count winner (no tally ties, where both models trivially return 0.5). Tied cues are coded sometimes 1-1 and sometimes 0-0 so the number of displayed 1s does not track composition.

CELLS (A-favouring version listed; the next trial is its exact mirror). Nominal predictions P(choose A): GVIT (lam=1.45, beta=6.7, gamma=.90, kappa_d=.10, omega=.34, eps=.17) vs SR-DT (lam=1.6, omega=1.7, beta=13.5, gamma_d=.68, gamma_c=.66, kappa=.05, phi=1.40, phi0=1.08, w_floor=-.30, eps=.09).
K1 (T1/T2) MID-PAIR vs LONE STAR (mA=2, mB=1, D=3): A wins f0(.72), f5(.68); B wins f1(.95) alone. GVIT E=+0.33 -> 0.67; SR-DT E=-0.17 -> 0.35. SIGN REVERSAL.
K2 (T3/T4) MID-PAIR vs LONE FLOOR (2 vs 1, D=3): identical A side, the single dissenter is now f4(.52). GVIT E=-1.05 -> 0.16; SR-DT E=+0.13 -> 0.62. SIGN REVERSAL, in the OPPOSITE direction from K1.
K3 (T5/T6) MID-TRIO vs LONE STAR (3 vs 1, D=4): A wins f0(.72), f5(.68), f2(.58); B wins f1(.95). GVIT 0.72; SR-DT 0.88.
K4 (T7/T8) MID-TRIO vs LONE FLOOR (3 vs 1, D=4): same A side, dissenter f4(.52). GVIT 0.22; SR-DT 0.92. LARGEST SINGLE-CELL GAP (~0.70) and the most parameter-robust: GVIT's E = 2 - lam_eff*(s72^om+s68^om+s58^om) is negative over almost its entire box (P(A) <= ~0.5), while SR-DT stays >= ~0.72 over its entire box.
K5 (T9/T10) UNOPPOSED MID-PAIR (mA=2, mB=0, D=2): A wins f0(.72), f5(.68); nothing opposes. GVIT 0.53 (near chance on a dominance display!); SR-DT 0.955.
K6 (T11/T12) UNOPPOSED BOTTOM-PAIR (mA=2, mB=0, D=2): A wins f2(.58), f4(.52); nothing opposes. GVIT 0.93; SR-DT 0.955.

FOUR ORTHOGONAL DIAGNOSTICS.
(1) LONE-OPPONENT IDENTITY CONTRAST, majority held exactly fixed (K1-K2 and K3-K4). GVIT's chorus criterion is TOTAL-D based, so the lone star carries negative weight while the lone floor voice carries +1: it predicts star-minus-floor = +0.51 (2-cue majority) and +0.50 (3-cue majority) -- i.e. people follow two middling experts against one .95 expert but abandon them against one .52 expert. SR-DT's same-side redundancy boosts BOTH lone voices above face value, more so the senior one, so it must predict a NON-POSITIVE contrast (-0.27 and -0.04). The SIGN of this within-subject contrast, replicated at two majority sizes with 32 trials each, is the single cleanest diagnostic.
(2) MAJORITY-SIZE GROWTH AGAINST THE FLOOR VOICE (K2 -> K4). SR-DT: 0.62 -> 0.92 (adding a third low-mid supporter converts a marginal display into a confident one). GVIT: 0.16 -> 0.22, i.e. still firmly sub-chance, because each added mid contributes ~0 while the floor voice keeps weight 1. A 3-versus-1 count majority being REJECTED is impossible for SR-DT anywhere in its box.
(3) UNOPPOSED-DOMINANCE GAP (K5 vs K6). Two structurally identical zero-conflict displays (2 wins, 0 losses, 4 ties). SR-DT has no dissent term and all sub-.80 experts retain large news value, so it must put BOTH at ceiling (0.955 / 0.955, gap ~0.00, and its eps <= .13 forces >= .935). GVIT's saturating inversion nullifies the mid pair but leaves the floor voice at full weight, predicting a ~0.40 gap (0.53 vs 0.93) -- near-chance responding on a display where one product is weakly dominant. K5 also serves as the unopposed control for K1: GVIT predicts P(A) goes UP (0.53 -> 0.67) when an opposing .95 expert is ADDED, a non-monotonicity SR-DT cannot produce (0.955 -> 0.35).
(4) CEILING/LAPSE ANCHOR (K6). Both models put this cell high (0.93 vs 0.955), which separately identifies each model's lapse rate so that slope and ceiling cannot be traded off when fitting the diagnostic cells.

CONTROLS: every configuration appears in an A-favouring and a B-favouring mirror, so side bias cancels exactly; validity rank is scrambled relative to screen position; tie coding alternates 1-1 / 0-0; no cell is degenerate and no cell is tally-tied (where both models trivially predict 0.5).

INTERPRETATION: a large POSITIVE lone-star-minus-lone-floor contrast at fixed majority (following two/three middling experts over one .95 expert while abandoning them against one .52 expert), sub-chance adherence in K2/K4, and near-chance responding on the unopposed mid-pair K5 with near-ceiling on the unopposed bottom-pair K6 uniquely support GVIT-CAP-S4. A NEGATIVE or null opponent-identity contrast, above-chance adherence to the count winner in all of K1-K4 with K4 near 0.9, and equal ceiling performance on K5 and K6 uniquely support SR-DT-6b.

[12] GOAL: dissociate CG-VI-2 (advocated) from GVIT-CAP-S4 (competing). The two models are nearly identical in spirit (hard count gate -> exact 0.5 on tally ties, inverse-validity 'news value' weights with a strongly saturating exponent s^omega, dilution by D^gamma, capacity term above 7 experts), so tally-tied cells and generic conflict cells are useless \u2014 they agree exactly. They differ on exactly TWO structural commitments, and every cell here is built to turn one of them into a parameter-free equality or a sign reversal.

DIFFERENCE 1 \u2014 THE ZERO FLOOR. CG-VI-2 bounds news value at zero: w = max(0, 1 - lam_eff*s^omega); a redundant expert is IGNORED. GVIT lets w go NEGATIVE, so a high-validity expert actively counts for the opposite product. Consequence: on displays in which every discriminating voice sits appreciably above the panel floor, CG-VI-2's evidence is EXACTLY ZERO (all weights clipped to 0) and it must return EXACTLY 0.500 for every lam, omega, q, kappa, beta, gamma, epsilon, whereas GVIT accumulates negative weights and REJECTS the count winner.

DIFFERENCE 2 \u2014 THE CHORUS CRITERION. CG-VI-2 reads the chorus off the bloc structure, CH = max(mA,mB) + |mA-mB| - 2, with a steep saturating gate CH^q/(CH^q+kappa), q~4. A 2-vs-1 display has CH = 1, so lam_eff = lam/(1+kappa) ~ 0.65 \u2014 only a WEAK discount. GVIT's chorus is the raw number of discriminating cues, lam_eff = lam*(D-1)/((D-1)+kappa_d) with kappa_d <= 0.45, so the same 2-vs-1 display (D=3) gets lam_eff ~ 1.38 \u2014 a NEAR-MAXIMAL discount. Hence 2-vs-1 displays are the sign-reversal engine.

PANEL: 6 experts, validities in deliberately scrambled screen order [f0=.88, f1=.55, f2=.72, f3=.95, f4=.63, f5=.80]; descending rank f3 > f0 > f5 > f2 > f4 > f1. Panel-relative seniority s = (v-.55)/.40 = f3 1.00, f0 .825, f5 .625, f2 .425, f4 .200, f1 .000. One lone floor voice (.55, s=0, full weight 1 in BOTH models), a dense upper cluster (.95/.88/.80/.72) all of whose s^omega exceed the clipping threshold for CG-VI-2 whenever the chorus is >= 2, and one buffer expert (.63). n = 6 keeps both models below their capacity terms, removing panel-size nuisance. 8 configurations x 2 exact A/B mirrors = 16 unique pairs x K = 96//16 = 6 repetitions = 96 trials. Tied cues are coded sometimes 1-1 and sometimes 0-0 so the number of displayed 1s never tracks composition; every cell has a strict count winner (no tally ties, where both models trivially predict .5).

CELLS (A-favouring version listed; the next trial is its exact mirror). Nominal predictions P(choose A): CG-VI-2 (lam=1.75, omega=.34, q=3.95, kappa=1.70, beta=11, gamma=.90, eps=.12) vs GVIT (lam=1.45, omega=.34, beta=6.7, gamma=.90, kappa_d=.10, eps=.17).
A1 (T1/T2) UNOPPOSED STAR-PAIR (mA=2, mB=0, D=2, CH=2): A wins f3(.95), f0(.88); four cues tied. CG-VI-2: both weights clip to 0 -> E = 0 -> EXACTLY 0.500 for every parameter setting (verified at the corners lam=1.6/q=3.4/kappa=2.0/omega=.40, where lam_eff=1.345 and the clipping threshold is s>=.48 < .825). GVIT: w = -0.318 and -0.238 -> P(A) = 0.185, i.e. 80%+ choice of the WEAKLY DOMINATED product. Flagship cell.
A2 (T3/T4) UNOPPOSED BOTTOM-PAIR (2,0, CH=2): A wins f4(.63), f1(.55). Matched composition to A1, opposite identity. CG 0.94 (range .91-.96), GVIT 0.91 (range .79-.96). Lapse/slope anchor, and the A1-vs-A2 within-subject gap at IDENTICAL (nA,nB,nT) is pure identity-driven: CG predicts a drop to exactly chance, GVIT a drop to well below chance.
A3 (T5/T6) STAR-PAIR vs LONE MID (2 v 1, D=3, CH=1): A wins f3(.95), f0(.88); B wins f2(.72). CG: weak discount (lam_eff=.648) leaves w = .352/.392 vs .513 -> E=+0.23 -> 0.694 (range .57-.78 over its whole box: ALWAYS above chance). GVIT: strong discount (lam_eff=1.381) makes the stars counter-indicative -> E=-0.64 -> 0.225 (range .23-.46: never above chance). SIGN REVERSAL, ~0.47 at nominal parameters.
A4 (T7/T8) MID-PAIR vs LONE FLOOR (2 v 1, D=3, CH=1): A wins f5(.80), f2(.72); B wins f1(.55). CG: E = .449+.513-1 = -0.04 -> 0.484, essentially CHANCE (its stated signature for 2-vs-1 mid-versus-floor). GVIT: E = -1.21 -> 0.123, decisive rejection. Gap ~0.36 on a display with a clear count winner.
A5 (T9/T10) MID-TRIO vs LONE FLOOR (3 v 1, D=4, CH=3): A wins f5, f2, f4; B wins f1(.55). CG: the chorus gate now saturates (lam_eff=1.71), the three supporters clip to ~0, the floor voice keeps weight 1 -> 0.097 (range .06-.15). GVIT: lam_eff barely changes from A4 -> 0.181. The DIAGNOSTIC IS THE A4-MINUS-A5 CONTRAST: CG predicts a large POSITIVE swing (+0.39: adding a third supporter makes adherence COLLAPSE from chance to 0.10, because bloc size, not cue count, drives the redundancy construal), GVIT predicts a flat or slightly NEGATIVE swing (-0.06). Non-monotonicity in majority size is the unique CG-VI-2 chorus signature.
A6 (T11/T12) ALL-UPPER 3 v 1 (D=4, CH=3): A wins f3(.95), f0(.88), f5(.80); B wins f2(.72); no floor voice discriminates. CG: every discriminating weight clips to 0 (threshold s>=.655^(1/omega) <= .43 at the worst corner, and s2=.425 with s2^.40=.710 already exceeds 1/lam_eff) -> E = 0 -> EXACTLY 0.500 despite a +2 count margin. GVIT: E = -0.86 -> 0.219. Second parameter-free cell, and the cleanest possible demonstration that 'redundant' means MUTE (CG) rather than COUNTER-INDICATIVE (GVIT).
A7 (T13/T14) MAVERICK-IN-MAJORITY (3 v 2, D=5, CH=2): A wins f3(.95), f0(.88) AND the floor voice f1(.55); B wins f5(.80), f2(.72). CG: all four upper voices are mute, the maverick speaks for the majority -> E=+1 -> 0.858 (near-ceiling following of a majority that contains the floor voice). GVIT: the two stars on A's side subtract while the two mids on B's side add back -> 0.662. Gap ~0.20; combined with A6 it separates 'stars are mute' from 'stars vote against their own side'.
A8 (T15/T16) MID-PAIR vs LONE STAR (2 v 1, D=3, CH=1): A wins f5(.80), f2(.72); B wins f3(.95). CG 0.873, GVIT 0.586. Same majority as A4 with only the identity of the lone dissenter changed: the OPPONENT-IDENTITY contrast A8-minus-A4 is +0.39 for CG and +0.46 for GVIT, but the absolute levels differ by ~0.29-0.36 in both cells, which pins each model's lam_eff at CH=1 / D=3 \u2014 exactly the quantity on which the two chorus rules disagree.

SUMMARY OF PREDICTED PROFILES (A1..A8): CG-VI-2 = .500, .94, .69, .48, .10, .500, .86, .87 ; GVIT = .19, .91, .23, .12, .18, .22, .66, .59. The models agree only on the anchor A2, and disagree in SIGN relative to chance on A1, A3, A4 and A6.

INTERPRETATION. Exact chance on A1 and A6 (two cells with strict count winners, one with a +2 margin), above-chance following of the star pair in A3, chance in A4 collapsing to ~0.10 in A5 when a third mid supporter is added, and near-ceiling in A7 uniquely support CG-VI-2. Systematic BELOW-chance responding in A1, A3, A4 and A6 (i.e. choosing the weakly dominated product in the unopposed star display) with a flat A4-A5 contrast uniquely supports GVIT-CAP-S4.

CONTROLS: every configuration appears in an A-favouring and a B-favouring mirror so side bias cancels exactly; validity rank is decoupled from screen position so positional heuristics cannot mimic validity-based weighting; tie coding alternates 1-1 / 0-0; no cell is tally-tied or otherwise degenerate, and no cell has the two models making the same prediction except the deliberate lapse anchor A2.

[13] GOAL: dissociate GVIT-CAP-S4 (advocated) from CG-VI-2 (competing). The two models share a hard count gate (tally ties -> exactly 0.5), an inverse-validity 'news value' weighting with a strongly saturating exponent s^omega, dilution by D^gamma and a capacity term above 7 experts. They differ on exactly TWO structural commitments, and every cell below turns one of them into a sign reversal or a parameter-free equality. (1) THE ZERO FLOOR: CG clips w = max(0, 1 - lam_eff*s^omega), so a redundant expert is MUTE; GVIT lets w go NEGATIVE, so a high-validity expert votes against his own product. (2) THE CHORUS CRITERION: CG reads the chorus off bloc structure, CH = max(mA,mB)+|mA-mB|-2 with a steep gate CH^q/(CH^q+kappa) (q~4), so a 2-vs-1 display (CH=1) is only WEAKLY discounted (lam_eff ~ 0.53-0.79); GVIT uses the raw number of discriminating cues, lam_eff = lam*(D-1)/((D-1)+kappa_d), so the same 2-vs-1 display (D=3) is NEAR-MAXIMALLY discounted (lam_eff ~ 0.90-1.79). Hence 2-vs-1 displays are the sign-reversal engine and unopposed/all-upper displays are the zero-floor engine.

PANEL: 6 experts, validities in scrambled screen order [f0=.92, f1=.60, f2=.84, f3=.53, f4=.76, f5=.68]; descending rank f0 > f2 > f4 > f5 > f1 > f3. v_min=.53, v_max=.92, span=.39, so panel-relative seniority s = 1.00/.795/.590/.385/.179/0 for f0/f2/f4/f5/f1/f3, and s^.34 = 1.00/.925/.836/.723/.558/0. One lone floor voice (f3=.53, full weight 1 in BOTH models), a dense upper cluster (.92/.84) whose weights clip to zero for CG whenever CH>=2 but go clearly negative for GVIT, a mid band (.76/.68) sitting exactly where GVIT's concave discount nullifies it, and a low-but-not-floor voice (.60). n=6 keeps both models below their capacity terms, removing panel-size nuisance. 8 configurations x 2 exact A/B mirrors = 16 unique pairs x K = 96//16 = 6 repetitions = 96 trials. Tied cues are coded sometimes 1-1 and sometimes 0-0 so the displayed number of 1s never tracks composition; no cell is tally-tied (both models would trivially return 0.5).

CELLS (A-favouring version listed; the next trial is its exact mirror). Nominal P(choose A): GVIT (lam=1.45, omega=.34, beta=6.7, gamma=.90, kappa_d=.10, eps=.17) vs CG-VI-2 (lam=1.75, omega=.34, q=3.95, kappa=1.70, beta=11, gamma=.90, eps=.12).
C1 (T1/T2) UNOPPOSED UPPER PAIR (mA=2, mB=0, D=2, CH=2): A wins f0(.92), f2(.84); four ties. CG: both weights clip to 0 -> E = 0 -> EXACTLY 0.500 at every corner of its box (worst corner lam=1.6,q=3.4,kappa=2.0 gives lam_eff=1.345 and even omega=.40 leaves s_f2^om=.912 > 1/1.345). GVIT: w = -0.318, -0.219 -> P(A) = 0.19, i.e. 80% choice of the weakly DOMINATED product. FLAGSHIP: chance vs strong reversal on a zero-conflict display.
C2 (T3/T4) UNOPPOSED BOTTOM PAIR (2,0, CH=2): A wins f1(.60), f3(.53). GVIT 0.91, CG 0.94. Matched composition to C1 but opposite identity: the C1-minus-C2 within-subject drop is 0.44 for CG (ceiling -> exact chance) and 0.72 for GVIT (ceiling -> far below chance). Also the lapse/slope anchor.
C3 (T5/T6) UPPER-PAIR vs LONE MID (2v1, D=3, CH=1): A wins f0, f2; B wins f5(.68). E = 1 - lam_eff*1.2024 in BOTH models, so the cell isolates lam_eff at 2-vs-1: CG's weak chorus gate keeps E > 0 across its entire box (+0.05 to +0.36) -> P(A) = 0.69 (>0.5 always); GVIT's D-based gate makes E <= 0 across its box -> P(A) = 0.22. SIGN REVERSAL.
C4 (T7/T8) UPPER-PAIR vs LONE LOW (2v1): same A side, dissenter is f1(.60). GVIT 0.17, CG 0.60. Replication of the sign reversal with a different lone opponent; the C3/C4 pair also traces how far down the panel the 'lone dissenter beats the star pair' effect extends.
C5 (T9/T10) MID-PAIR vs LONE STAR (2v1): A wins f4(.76), f5(.68); B wins f0(.92). E = 1 - 0.5582*lam_eff. GVIT: the mids are discounted almost as hard as the star, so the majority is followed only WEAKLY (0.62; 0.50-0.70 over its box) - its commitment-3(i) signature. CG: lam_eff at CH=1 is small, all weights stay large and positive, so the majority is followed strongly (0.88).
C6 (T11/T12) BOTTOM-PAIR vs LONE STAR (2v1): A wins f1(.60), f3(.53); B wins f0(.92). GVIT 0.90, CG 0.94. KEY GRADED CONTRAST C6 - C5 (identical composition, only the seniority band of the majority changes): GVIT predicts a LARGE gap (~0.28, because mid support is nearly mute while bottom support keeps full weight), CG predicts a SMALL gap (~0.06, because at CH=1 nothing is heavily discounted). 32 trials per subject on this contrast.
C7 (T13/T14) ALL-UPPER 3v1 (mA=3, mB=1, D=4, CH=3): A wins f0, f2, f4; B wins f5(.68). CG: the chorus gate saturates, every discriminating weight clips to 0 -> E = 0 -> EXACTLY 0.500 despite a +2 count margin (robust at all box corners). GVIT: E = -0.86 -> 0.22. Second parameter-free cell for CG and the cleanest demonstration that 'redundant' means COUNTER-INDICATIVE (GVIT) rather than MUTE (CG).
C8 (T15/T16) MAVERICK-IN-MAJORITY 3v2 (D=5, CH=2): A wins f0(.92), f2(.84) AND the floor voice f3(.53); B wins f4(.76), f5(.68). CG: all four upper voices are mute, the maverick speaks for the majority -> near-ceiling 0.88. GVIT: A's two stars subtract while B's two mids add back, leaving only a modest positive -> 0.65. Gap ~0.23; combined with C7 it separates 'stars are silent' from 'stars vote against their own side'.

PREDICTED PROFILE (C1..C8): GVIT = .19, .91, .22, .17, .62, .90, .22, .65 ; CG-VI-2 = .500, .94, .69, .60, .88, .94, .500, .88. The models agree only on the anchors C2 and C6, and disagree in SIGN relative to chance on C1, C3, C4 and C7.

INTERPRETATION: systematic BELOW-chance responding on C1 and C7 (choosing the product that no expert prefers / that loses the 3-vs-1 count), below-chance on the 2-vs-1 star-pair cells C3/C4, only weakly-above-chance following of the mid-backed majority in C5 with a large C6-minus-C5 gap, uniquely supports GVIT-CAP-S4. Exact chance on C1 and C7, above-chance following of the star pair in C3/C4, strong adherence in C5 with a flat C6-C5 contrast, and near-ceiling in C8 uniquely supports CG-VI-2.

CONTROLS: every configuration appears in an A-favouring and a B-favouring mirror so side bias cancels exactly; validity rank is decoupled from screen position so positional heuristics cannot mimic validity-based weighting; tie coding alternates 1-1 / 0-0; no cell is tally-tied or otherwise degenerate; every trial requires reading several ratings.

[14] GOAL: dissociate CG-VI-2 (advocated) from MBM-PG7 (competing). The two models share almost everything: a parameter-free count gate (tally ties -> exactly 0.5), panel-relative seniority s=(v-vmin)/(vmax-vmin), inverse-validity 'news value', dilution by D^gamma, a lapse, and a capacity term past 7 voices. They also AGREE on the classic maverick reversals (a lone floor voice opposed by a 3-cue bloc reverses it in both models), so those cells are wasted. They differ on exactly three things, and every cell below turns one of them into a parameter-free equality, a large magnitude gap, or a sign flip.

DIFFERENCE 1 - WHAT HAPPENS WHEN NO CHORUS EXISTS (D=1). CG's chorus argument is CH = max(mA,mB)+|mA-mB|-2, so a SOLO discriminating cue has CH=0 -> lam_eff=0 -> EVERY weight is exactly 1 -> E=1, den=1, z=beta in [9.5,12.5] -> P = 1-eps/2 in [0.91,0.96], IDENTICALLY for a solo .95 expert, a solo .75 expert and a solo .55 expert (a three-way parameter-free equality at near ceiling). MBM has no such release: a solo star is still outside the mute band, so its whole contribution is the thin residual r=eta (0.18-0.28), giving z=beta*eta in [0.13,0.25] -> P in [0.53,0.56]; a solo mid contributes eta+c/2 -> P ~ 0.65-0.77; a solo floor voice contributes 1+eta+c -> P ~ 0.86-0.93. So MBM must produce a steep monotone inverse-validity LADDER across the three solo cells (~0.54 / 0.71 / 0.89) whereas CG must produce a FLAT near-ceiling line (~0.94 / 0.94 / 0.94). C1 (solo star) alone is decisive: the two models' whole parameter boxes are disjoint (CG >= 0.91 vs MBM <= 0.56).

DIFFERENCE 2 - MUTE MEANS ZERO (CG) VS MUTE MEANS A GRADED RESIDUAL (MBM). CG clips w = max(0, 1-lam_eff*s^omega): once the chorus is >= 2 every voice appreciably above the panel floor is EXACTLY mute, so a display in which no floor voice discriminates has E = 0 and returns EXACTLY 0.500 for every lam, omega, q, kappa, beta, gamma, epsilon - even when one product is weakly DOMINANT or wins the tally 3-to-1. MBM's muted experts keep r = eta + c(1-s), an inverse-seniority gradient, so MBM's prediction on such displays tracks the seniority composition. Cells C4 (unopposed .79+.75 pair, mA=2/mB=0, CH=2), C5 (low trio .79/.75/.71 beats lone .95, CH=3) and C6 (senior trio .95/.92/.89 beats lone .71, CH=3) are all built so that every discriminating voice clips at EVERY corner of CG's box (at CH=2 clipping needs s>=0.48 at the worst corner - the cues used are s=0.50 and 0.60; at CH=3 it needs s>=0.35 - the lowest cue used is s=0.40). CG therefore predicts exactly 0.500 in C4, C5 and C6, including a +2 tally margin in C5/C6 and a zero-conflict dominance display in C4. MBM predicts ~0.70 (C4), ~0.66 (C5) and ~0.49 (C6). C5 vs C6 is the cleanest form of this test: IDENTICAL composition (3 vs 1, D=4, CH=3), only the seniority band of the majority is swapped; CG must give an exact null difference at exactly chance, MBM must give a ~0.17 swing that straddles chance.

DIFFERENCE 3 - THE 2-vs-1 REGIME AND OVERALL DECISIVENESS. CG's chorus gate is steep in bloc structure: a 2-vs-1 display has CH=1, so lam_eff = lam/(1+kappa) in [0.53,0.79] - only a WEAK discount, all weights stay positive, and with beta in [9.5,12.5] the weighted tally is followed decisively. MBM has no bloc-size gate for out-of-band voices and a small beta (0.7-0.9), so its 2-vs-1 predictions are always mid-range. C7 (A wins .75 and .71; B wins .95): CG in [0.82,0.93] vs MBM in [0.59,0.72] - nearly disjoint magnitude prediction at the same sign. C8 (A wins .95 and .92; B wins .71): CG in [~0.50,0.78] (its weighted tally still favours the pair) vs MBM in [0.42,0.47] (the single low voice's residual outweighs two star residuals) - a sign contrast around chance. C8 also pairs with C6: CG predicts a NON-MONOTONICITY in majority size (senior pair over lone low = ~0.66, but ADDING a third senior supporter collapses it to exactly 0.500, because bloc size drives the redundancy construal), whereas MBM predicts both cells flat and slightly below chance (~0.45 / ~0.49).

PANEL: 7 experts, validities in deliberately scrambled screen order [f0=.75, f1=.95, f2=.71, f3=.89, f4=.55, f5=.79, f6=.92]; vmin=.55, vmax=.95, span=.40, so s = .50/1.00/.40/.85/.00/.60/.925. Exactly ONE voice sits at the panel floor (f4=.55, s=0: maverick for BOTH models, weight 1 in CG and inside MBM's band for every theta), and every other voice sits at s>=0.40 - far outside MBM's widest possible band (theta<=0.20) and inside CG's clipping region whenever CH>=2. This removes the band-width nuisance entirely and makes the dissociation rest only on the three structural commitments above. n=7 keeps both models at/below their capacity thresholds (over = 0), so no panel-size nuisance.

DESIGN: 8 configurations x 2 exact A/B mirrors = 16 unique pairs x K = 96//16 = 6 repetitions = 96 trials. No cell is tally-tied (both models trivially return 0.5 there) and no cell is fully degenerate.

CELLS (A-favouring version listed first; the next trial is its exact mirror). Nominal predictions P(choose A): CG-VI-2 (lam=1.75, omega=.34, q=3.95, kappa=1.70, beta=11, gamma=.90, eps=.12) vs MBM-PG7 (theta=.16, rho=2.1, eta=.23, c=2.0, beta=.80, gamma=.95, alpha=52, kappa=205, p=6.4, lam_own=.36, eps=.09).
C1 (T1/T2) SOLO STAR: only f1(.95) discriminates. CG 0.94 [0.91,0.96] vs MBM 0.54 [0.53,0.56]. FLAGSHIP - disjoint boxes.
C2 (T3/T4) SOLO MID: only f0(.75) discriminates. CG 0.94 vs MBM 0.71.
C3 (T5/T6) SOLO FLOOR: only f4(.55) discriminates. CG 0.94 vs MBM 0.89. Deliberate near-agreement: the lapse/ceiling anchor that identifies each model's epsilon, and the third rung of the solo ladder.
C4 (T7/T8) UNOPPOSED LOW-UPPER PAIR (2,0,CH=2): A wins f5(.79), f0(.75). CG EXACTLY 0.500 (both clip at every box corner) vs MBM 0.70. Chance responding on a weakly dominant display is CG's signature.
C5 (T9/T10) ALL-UPPER 3v1, LOW TRIO vs LONE STAR: A wins f5,f0,f2; B wins f1(.95). CG EXACTLY 0.500 despite a +2 margin; MBM 0.66.
C6 (T11/T12) ALL-UPPER 3v1, SENIOR TRIO vs LONE LOW: A wins f1,f6,f3; B wins f2(.71). CG EXACTLY 0.500 (identical to C5, parameter-free equality); MBM 0.49.
C7 (T13/T14) 2v1 LOW PAIR vs LONE STAR (CH=1): A wins f0(.75), f2(.71); B wins f1(.95). CG 0.88 [0.82,0.93] vs MBM 0.65 [0.59,0.72].
C8 (T15/T16) 2v1 SENIOR PAIR vs LONE LOW (CH=1): A wins f1(.95), f6(.92); B wins f2(.71). CG 0.66 (>=~0.5 over essentially its whole box) vs MBM 0.45 [0.42,0.47].

PREDICTED PROFILE (C1..C8): CG-VI-2 = .94, .94, .94, .500, .500, .500, .88, .66 ; MBM-PG7 = .54, .71, .89, .70, .66, .49, .65, .45. They agree only on the deliberate anchor C3.

INTERPRETATION: flat near-ceiling adherence across all three solo cells, exact chance on C4/C5/C6 (a dominance display and two +2-margin displays), decisive following of the low pair in C7 and above-chance following of the senior pair in C8 uniquely support CG-VI-2. A steep solo ladder (.54/.71/.89), above-chance following of the tally in C4/C5 with a systematic C5-minus-C6 gap, mid-range (never extreme) adherence in C7, and sub-chance C8 uniquely support MBM-PG7. If instead subjects simply follow validity (WADD/TTB), C6 and C8 will be HIGH and C5/C7 LOW, which falsifies both models - so the design is self-diagnostic.

CONTROLS: every configuration appears in an A-favouring and a B-favouring mirror, so side bias cancels exactly; validity rank is decoupled from screen position so positional heuristics cannot mimic validity-based weighting; tied cues are coded sometimes 1-1 and sometimes 0-0 within each cell so the number of displayed 1s never tracks composition or seniority; every trial has a strict count winner and requires reading several ratings.

[15] GOAL: dissociate MBM-PG7 (advocated) from CG-VI-2 (competing). The two models share a parameter-free count gate (tally ties -> exactly 0.5), panel-relative seniority s=(v-vmin)/(vmax-vmin), an inverse-validity 'news value' logic, dilution by D^gamma, a lapse, and a capacity term past 7 voices. They also AGREE on the classic maverick reversals, so those cells are wasted except as anchors. They differ on three structural points, and every cell below converts one of them into a parameter-free equality, a large magnitude gap, or a sign flip.

DIFFERENCE 1 - SOLO CUES (D=1). CG's chorus argument CH = max(mA,mB)+|mA-mB|-2 is 0 for a solo discriminating cue, so lam_eff=0, EVERY weight is exactly 1, E=1, den=1, z=beta in [9.5,12.5] -> P = 1-eps/2 in [0.91,0.96] IDENTICALLY for a solo .95, a solo .78 and a solo .53 expert (a three-way parameter-free equality at near ceiling). MBM has no such release: a solo star is outside the mute band and contributes only the thin residual eta -> z = beta*eta in [0.13,0.25] -> P in [0.53,0.56]; a solo low-upper (s=.595) contributes eta+0.405c -> P ~ 0.63-0.72; a solo floor voice contributes 1+eta+c -> P ~ 0.83-0.93. So MBM must produce a steep monotone inverse-validity LADDER (~.54/.68/.89) where CG must produce a FLAT line (~.94/.94/.94). Cell C1 alone has DISJOINT parameter boxes (CG >= .91 vs MBM <= .56).

DIFFERENCE 2 - MUTE MEANS EXACTLY ZERO (CG) VS A GRADED RESIDUAL GRADIENT (MBM). CG clips w = max(0, 1-lam_eff*s^omega): once the bloc chorus is CH>=2 every voice appreciably above the panel floor is EXACTLY mute, so any display in which no floor voice discriminates has E=0 and returns EXACTLY 0.500 for every lam, omega, q, kappa, beta, gamma, epsilon - even a zero-conflict dominance display (C4, C5) or a +2 tally margin (C6). MBM's muted experts keep r = eta + c(1-s), so its prediction tracks the seniority composition of the display: above chance and graded. Clipping was verified at every corner of CG's box (at CH=2 clipping needs s >= .476 at the worst corner; the cues used have s = .595/.690/.905/1.00; at CH=3 it needs s >= .345 and the lowest cue used is s = .452).

DIFFERENCE 3 - THE 2-vs-1 REGIME. CG's chorus gate is steep in BLOC structure: a 2-vs-1 display has CH=1, so lam_eff = lam/(1+kappa) in [0.53,0.79] - only a WEAK discount, all weights stay positive, and with beta ~ 11 the weighted tally is followed decisively. MBM has no bloc-size release for out-of-band voices, only the residual gradient and a small beta (0.7-0.9), so two senior voices (residual .23+.42) lose to one low-upper voice (residual 1.04): C7 is a SIGN FLIP with disjoint boxes (CG in [0.59,0.80] vs MBM in [0.455,0.49]).

PANEL: 7 experts, validities in deliberately scrambled screen order [f0=.82, f1=.53, f2=.95, f3=.78, f4=.91, f5=.72, f6=.87]; vmin=.53, vmax=.95, span=.42, so s = .690/.000/1.000/.595/.905/.452/.810. EXACTLY ONE voice sits at the panel floor (f1=.53, s=0: a maverick for both models, weight 1 in CG and inside MBM's band for every theta), and every other voice has s >= .452, far outside MBM's widest possible band (theta <= .20) and inside CG's clipping region whenever CH >= 2. This removes the band-width nuisance completely and makes the dissociation rest only on the three commitments above. n=7 keeps both models at their capacity threshold (over = 0), removing panel-size nuisance.

DESIGN: 8 configurations x 2 exact A/B mirrors = 16 unique pairs x K = 96//16 = 6 repetitions = 96 trials. No cell is tally-tied (both models would trivially return 0.5) and every trial requires reading several ratings.

CELLS (A-favouring version listed first; the next trial is its exact mirror). Nominal P(choose A): MBM-PG7 (theta=.16, rho=2.1, eta=.23, c=2.0, beta=.80, gamma=.95, alpha=52, kappa=205, p=6.4, lam_own=.36, eps=.09) vs CG-VI-2 (lam=1.75, omega=.34, q=3.95, kappa=1.70, beta=11, gamma=.90, eps=.12).
C1 (T1/T2) SOLO STAR: only f2(.95) discriminates. MBM 0.54 [.53,.56] vs CG 0.94 [.91,.96]. FLAGSHIP - disjoint boxes.
C2 (T3/T4) SOLO LOW-UPPER: only f3(.78) discriminates. MBM 0.68 vs CG 0.94.
C3 (T5/T6) SOLO FLOOR: only f1(.53) discriminates. MBM 0.89 vs CG 0.94. Deliberate near-agreement: lapse/ceiling anchor and third rung of the ladder. CG must make C1=C2=C3 exactly; MBM must make them a steep monotone ladder.
C4 (T7/T8) UNOPPOSED LOW PAIR (mA=2, mB=0, CH=2): A wins f3(.78) and f0(.82), five ties. CG EXACTLY 0.500 at every box corner (both weights clip) vs MBM 0.67 [.62,.73]. Chance responding on a weakly dominant display is CG's signature.
C5 (T9/T10) UNOPPOSED SENIOR PAIR (2,0, CH=2): A wins f4(.91) and f2(.95). CG EXACTLY 0.500 (a parameter-free equality with C4) vs MBM 0.56. The C4-minus-C5 contrast at IDENTICAL composition is pure seniority: CG must give exactly 0 (at exactly chance), MBM ~ +0.11 with both above chance.
C6 (T11/T12) ALL-UPPER 3v1: A wins f5(.72), f3(.78), f0(.82); B wins f2(.95). CG EXACTLY 0.500 despite a +2 count margin; MBM 0.64.
C7 (T13/T14) 2v1 SENIOR PAIR vs LONE LOW-UPPER (CH=1): A wins f2(.95), f4(.91); B wins f3(.78). CG 0.72 [.59,.80] vs MBM 0.475 [.455,.49]. SIGN FLIP with disjoint boxes: CG follows the weighted majority decisively, MBM lets the single low-upper's fat residual cancel two thin senior residuals.
C8 (T15/T16) MAVERICK vs SENIOR TRIO: A wins f1(.53) alone; B wins f2(.95), f4(.91), f6(.87). MBM 0.93 vs CG 0.90. Deliberate agreement anchor AND self-diagnostic: WADD/TTB predict ~0.05 here, so systematic choice of B falsifies both models.

PREDICTED PROFILE (C1..C8): MBM-PG7 = .54, .68, .89, .67, .56, .64, .475, .93 ; CG-VI-2 = .94, .94, .94, .500, .500, .500, .72, .90. They agree only on the deliberate anchors C3 and C8.

INTERPRETATION: a steep solo ladder (~.54/.68/.89), systematically ABOVE-chance responding on the all-upper cells C4/C5/C6 with the low-seniority display exceeding the senior display, and at-or-below-chance responding on C7 uniquely support MBM-PG7. Flat near-ceiling adherence across all three solo cells, exact chance on C4, C5 and C6 (a dominance display, a matched dominance display and a +2-margin display), and decisive above-chance following of the senior pair in C7 uniquely support CG-VI-2.

CONTROLS: every configuration appears in an A-favouring and a B-favouring mirror so side bias cancels exactly; validity rank is decoupled from screen position so positional heuristics cannot mimic validity-based weighting; tied cues are coded sometimes 1-1 and sometimes 0-0 within and across cells so the number of displayed 1s never tracks composition or seniority; every trial has a strict count winner.

[16] GOAL: dissociate CTN-BG (advocated) from MBM-PG7 (competing). The two models are near-twins: parameter-free count gate (tally ties -> exactly 0.5), panel-relative seniority s=(v-vmin)/(vmax-vmin), a near-categorical bottom-voice 'maverick/novelty' weight, chorus-gated amplification of that voice, dilution by D^gamma, a lapse, and a capacity term past 7 voices. With one lone floor expert and everything else far above the band, their novelty/mute functions AGREE (nov=1 vs mav=1 at s=0; both ~0 for every s>=0.45, for every theta<=0.20 and every kappa in [14,16.5]). So band-shape cells are wasted. They differ on exactly ONE structural commitment that this design attacks from four directions, plus a magnitude difference in the residual gradient.

THE CRITICAL DIFFERENCE. CTN-BG measures the chorus in MID-EXPERT EQUIVALENTS: Q_j = sum_opp u_i + lam_own*sum_own u_i with u=(1+tau*s)/(1+tau/2) (mid=1.00, star~1.33-1.43, floor~0.6), gated by a narrow logistic at q0~2.45 units. MBM-PG7 measures it in RAW COUNTS: base = (m_opp-1) + lam_own*(m_own-1), gated by a late steep knee (p~6.4, kappa~205) that needs base >~2.3. Hence: TWO STARS are a chorus for CTN (Q=2.63-2.86 > q0) but not for MBM (base=1 -> G=1.20-1.32); THREE MIDS are a chorus for both (Q=3.0; base=2 -> G=11.6-22.2); an UNOPPOSED 3-cue bloc containing the floor voice is a chorus for MBM (base=2) but not for CTN (own-side echo lam_own*(u_star+u_star)=1.32-2.00 < q0).

PANEL (screen order deliberately scrambled so validity rank is decoupled from position): [f0=.74, f1=.94, f2=.54, f3=.76, f4=.92, f5=.72]. vmin=.54, vmax=.94, span=.40 -> s = .50/1.00/.00/.55/.95/.45. Exactly ONE floor voice (f2=.54, s=0: nov=1 in CTN, inside MBM's band for every theta), TWO stars (s=1.00, .95: u=1.34-1.43 each), THREE mids straddling s=0.5 (.45/.50/.55, so any two of them sum to exactly 2.00 mid-equivalents and all three to exactly 3.00). n=6 keeps both models below their capacity terms, removing panel-size nuisance. 8 configurations x 2 exact A/B mirrors = 16 unique pairs x K=96//16=6 repetitions = 96 trials. Tied cues are coded sometimes 1-1 and sometimes 0-0 so the number of displayed 1s never tracks condition. No cell is tally-tied (both models would trivially return 0.5).

CELLS (A-favouring version listed; the next trial is its exact mirror). Nominal P(choose A): CTN-BG (eta=.04, c_res=.12, nu=1.88, kappa=15, tau=1.2, lam_own=.60, gmax=1.9, q0=2.45, wid=.14, c_n=2.9, beta=41, gamma=.96, eps=.12) vs MBM-PG7 (theta=.16, rho=2.1, eta=.23, c=2.0, beta=.80, gamma=.95, alpha=52, kappa=205, p=6.4, lam_own=.36, eps=.09).
C1 (T1/T2) FLOOR vs MID-PAIR (1 v 2, D=3; Q=2.00 units): A wins f2; B wins f5,f3. CTN 0.58, MBM 0.56. Low rung of the chorus ladder; both near chance.
C2 (T3/T4) FLOOR vs STAR-PAIR (1 v 2, D=3; Q=2.71 units, SAME COUNT as C1): A wins f2; B wins f1,f4. CTN 0.93, box [0.86,0.96]; MBM 0.68, box [0.62,0.75]. FLAGSHIP: the parameter boxes are DISJOINT, and the within-subject contrast C2-C1 is +0.25..+0.40 for CTN but only +0.04..+0.21 for MBM (whose only lever here is the thin star residual).
C3 (T5/T6) FLOOR vs MID-TRIO (1 v 3, D=4; Q=3.00 units, only ~0.3 units more than C2 but one more COUNT): A wins f2; B wins f5,f0,f3. CTN 0.90, MBM 0.92. SECOND KEY CONTRAST: C3-C2 is ~0.00 (range -0.05..+0.03) for CTN, because 2 stars and 3 mids are the same chorus in mid-equivalents, but +0.08..+0.33 (nominal +0.24) for MBM, whose knee sits between base=1 and base=2. C2 and C3 together are the crossed dissociation: seniority-driven (CTN) versus count-driven (MBM) amplification.
C4 (T7/T8) FLOOR+STAR-PAIR vs LONE MID (3 v 1, D=4): A wins f2,f1,f4; B wins f0. CTN 0.90 (box [0.66,0.93]); MBM 0.62 (box [0.58,0.65]). CTN's own-side echo of two stars (0.6*2.71=1.63) plus the opposing mid crosses q0; MBM's own-side echo is 0.36*2=0.72 counts, far below its knee.
C5 (T9/T10) FLOOR+MID-PAIR vs LONE STAR (3 v 1, D=4): A wins f2,f5,f3; B wins f1. CTN 0.90 (equal to C4, since 1.375+0.6*2.00 ~ 1.00+0.6*2.71), MBM 0.75. CTN predicts C4=C5; MBM predicts C5-C4 ~ +0.13 (its thick residual gradient c~2.0 makes mid companions worth far more than star companions).
C6 (T11/T12) UNANIMOUS FLOOR+STAR (2 v 0, D=2): A wins f2,f1; four ties. CTN 0.76, MBM 0.79. Deliberate near-agreement anchor that identifies each model's lapse/slope.
C7 (T13/T14) UNANIMOUS FLOOR+STAR+STAR (3 v 0, D=3): A wins f2,f1,f4; three ties. CTN 0.72 (box [0.63,0.89]) vs MBM 0.95 (box [0.93,0.98]). THIRD KEY CONTRAST, and it runs in the OPPOSITE DIRECTION from C2/C4 (so no generic 'one model is more decisive' confound can explain the data): adding a second star to an unopposed bloc containing the floor voice leaves CTN flat or slightly lower (C7-C6 in [-0.06,+0.03], D-dilution offsets the sub-threshold echo) but pushes MBM from base=1 to base=2, i.e. across its knee (C7-C6 = +0.08..+0.26).
C8 (T15/T16) ALL-UPPER MID-TRIO vs LONE STAR (3 v 1, D=4, no floor voice discriminates): A wins f5,f0,f3; B wins f1. CTN 0.63, MBM 0.665. Deliberate agreement cell that is SELF-DIAGNOSTIC: TTB/one-reason would give <0.2 here and near-1.0 in C2/C7, so systematic star-following falsifies BOTH models; it also verifies that the residual gradient alone yields only moderate adherence on displays with no bottom voice.

PREDICTED PROFILE (C1..C8): CTN-BG = .58, .93, .90, .90, .90, .76, .72, .63 ; MBM-PG7 = .56, .92-ambiguous? no: .56, .68, .92, .62, .75, .79, .95, .665. They agree only on the deliberate anchors C1, C3, C6 and C8.

INTERPRETATION. A large C2-C1 jump (~+0.35) with C2 at or above 0.86 and C3 no higher than C2, C4 = C5 near 0.90, and a FLAT C6->C7 (unanimous bloc growth does nothing) uniquely support CTN-BG's mid-expert-equivalent chorus. A flat C2-C1 (~+0.1) with C2 capped near 0.70, a large C3-C2 jump (~+0.24), C5 clearly above C4 with both near 0.6-0.75, and a big C6->C7 jump to ~0.95 uniquely support MBM-PG7's count-based knee.

CONTROLS: every configuration appears in an A-favouring and a B-favouring mirror, so side bias cancels exactly; validity rank is scrambled relative to screen position; tie coding alternates 1-1/0-0 within and across cells; every trial has a strict count winner and requires reading several ratings; n=6 removes all capacity-term nuisance.

[17] GOAL: dissociate MBM-PG7 (advocated) from CTN-BG (competing). The two models are near-twins 
oun the structural level (parameter-free count gate -> exactly 0.5 on tally ties; panel-relative seniority s=(v-vmin)/(vmax-vmin); a near-categorical bottom-voice 'maverick/novelty' weight; chorus-gated amplification of that voice; dilution by D^gamma; a lapse; a capacity term past 7 voices). With exactly ONE floor expert and every other expert at s>=0.31, their band/novelty functions agree (mav=nov=1 at s=0; both ~0 for every s>=0.31 at any theta<=0.20 or kappa>=14), so band-shape cells are wasted. They differ on exactly TWO things, and this design converts both into parametric, within-subject signatures rather than isolated cells.

DIFFERENCE 1 - HOW A CHORUS IS COUNTED. MBM measures the chorus in RAW COUNTS: base = (m_opp-1) + lam_own*(m_own-1), gated by a LATE, STEEP knee (p~6.4, kappa~205) that needs base >~2, i.e. it needs THREE opposing voices; the identity of the opposers is irrelevant to the gain and enters only through the THICK residual gradient r = eta + c(1-s) with c~2.0. CTN measures it in MID-EXPERT EQUIVALENTS: Q = sum_opp u_i + lam_own*sum_own u_i with u=(1+tau*s)/(1+tau/2), gated by a NARROW logistic at q0~2.45 with wid~0.14; hence TWO STARS (u=1.29+1.38=2.66) already constitute a full chorus while two low-upper voices (u=0.86+0.93=1.79) do not, and the identity of the opposers is everything.

THE NOVEL CORE: A FOUR-POINT SENIORITY TITRATION AT FIXED COUNT (C1-C4). A lone floor voice (f1=.52) is opposed by a PAIR whose mid-equivalent sum is titrated in four steps straddling CTN's threshold: Q = 1.79 (C1: .65+.69), 2.14 (C2: .75+.79), 2.39 (C3: .79+.89), 2.66 (C4: .89+.94). For MBM the gain is CONSTANT across all four (base = 1 -> G = 1.25 always); only the opposers' residuals shrink, producing a SHALLOW, nearly linear rise. For CTN the gate sweeps its logistic, producing a STEEP sigmoid. Nominal predictions P(choose the floor-backed option): MBM = .53, .59, .63, .67 (total rise +0.14); CTN = .49, .71, .90, .94 (total rise +0.45). C4 alone has nearly disjoint boxes (MBM [.62,.75] vs CTN [.86,.96]).

THE CROSSED CONTRAST (C4 vs C5). C5 opposes the same floor voice with a THREE-cue LOW bloc (.65/.69/.75; Q=2.82, base=2). MBM crosses its count knee (G jumps 1.25 -> 16.2) and predicts .92; CTN is already saturated at C4 and predicts .92 as well, i.e. C5-C4 = +0.25 for MBM but -0.02 for CTN. Note the two cells are matched in mid-equivalents (2.66 vs 2.82) but differ in COUNT: a seniority-driven chorus (CTN) must be flat here, a count-driven knee (MBM) must jump. C1 vs C5 (+0.39 for both) is the shared control showing the manipulation works.

DIFFERENCE 2 - OWN-SIDE ECHO (C7 vs C6). MBM's own-side term is a COUNT discounted by lam_own~0.36, so an UNOPPOSED bloc of three that contains the floor voice has base = m_own-1 = 2 and crosses the knee, whereas CTN's own-side echo of two stars is 0.6*2.66 = 1.60 mid-equivalents, far BELOW q0, so no amplification ever occurs on unanimous displays. Predictions: C7 (floor + one star, 2v0) MBM .81 / CTN .76 (deliberate near-agreement anchor); C6 (floor + two stars, 3v0) MBM .955 / CTN .72. The within-subject bloc-growth contrast C6-C7 is +0.15 for MBM and -0.04 for CTN (D-dilution wins), and it runs in the OPPOSITE direction from the C1-C4 titration, so no generic 'one model is simply more decisive' account can explain both patterns.

ANCHORS / SELF-DIAGNOSTICS. C1 (both ~chance) fixes the low rung. C8 (all-upper low trio .65/.69/.75 beats a lone .94 star; no floor voice discriminates) is a deliberate agreement cell (MBM .69 / CTN .67) that anchors slope/lapse and is self-diagnostic: TTB/WADD would give <0.2 here and near 1.0 in C4, so systematic star-following falsifies BOTH models.

PANEL: 7 experts, validities in deliberately scrambled screen order [f0=.79, f1=.52, f2=.94, f3=.65, f4=.89, f5=.75, f6=.69]; vmin=.52, vmax=.94, span=.42 -> s = .643/.000/1.000/.310/.881/.548/.405. Exactly one floor voice (f1, s=0: full maverick/novelty in both models); every other voice sits at s>=0.31, outside MBM's widest band (theta<=0.20) and with CTN novelty <=0.004, so the dissociation rests purely on the chorus rule and the residual gradient. n=7 keeps BOTH models exactly at their capacity thresholds (over=0), removing all panel-size nuisance.

DESIGN: 8 configurations x 2 exact A/B mirrors = 16 unique pairs x K = 96//16 = 6 repetitions = 96 trials. No cell is tally-tied (both models trivially return 0.5 there); every trial has a strict count winner and requires reading several ratings.

PREDICTED PROFILE (C1..C8, P(choose the focal/floor-backed option)): MBM-PG7 = .53, .59, .63, .67, .92, .955, .81, .69 ; CTN-BG = .49, .71, .90, .94, .92, .72, .76, .67.

INTERPRETATION: a SHALLOW seniority titration (<=0.15 total across C1-C4, with C4 capped near 0.70), a large count-driven jump from C4 to C5, a large jump from C7 to C6 on unanimous bloc growth, and near-ceiling C6 uniquely support MBM-PG7's raw-count knee with a thick residual gradient. A STEEP sigmoid across C1-C4 (with C3/C4 >= 0.86), a flat or negative C4->C5 step, and a flat-to-negative C7->C6 step uniquely support CTN-BG's mid-expert-equivalent chorus.

CONTROLS: every configuration appears in an A-favouring and a B-favouring mirror, so side bias cancels exactly; validity rank is scrambled relative to screen position so positional heuristics cannot mimic validity-based weighting; tied cues are coded sometimes 1-1 and sometimes 0-0 within and across cells so the number of displayed 1s never tracks condition or seniority.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "GOAL: dissociate CTN-BG (advocated) from FM-ISN-CC5 (competing). The two models are near-twins in surface behaviour: panel-relative seniority s=(v-vmin)/(vmax-vmin), a near-categorical inverse-seniority weight with kappa~15-17 (so only the panel floor voice speaks loudly and everything at s>=0.45 is effectively mute), Luce-like saturating integration with a constant of indecision, count dilution by D^gamma, and a lapse. Generic maverick cells (lone floor voice beating an upper bloc) are therefore WASTED - both predict reversal. They differ on exactly TWO structural commitments, and every cell below turns one of them into either a parameter-free prediction or a parameter-free ZERO contrast.\n\nDIFFERENCE 1 - WHAT A TALLY TIE DOES. CTN-BG's count comparator is a parameter-free GATE on direction: when the two options win equally many attributes the display is non-diagnostic and the model returns EXACTLY 0.500 for every setting of its 15 parameters, with no validity tie-break. FM-ISN-CC5's count comparator gates only CONFIDENCE (gain rho_tie in [0.60,0.78]); the weighted inverse-seniority reading still runs, so a tie in which the floor voice backs one option and a star backs the other must come out strongly ANTI-validity. Cells C1 (1v1: .54 vs .94) and C2 (2v2: floor+mid vs the two stars) implement this. Across FM's ENTIRE box (beta 8.3-9.2, c_n 1.2-1.5, gamma .34-.46, rho_tie .60-.78, eps .05-.10) C1 lies in [0.78,0.88] and C2 in [0.71,0.84]; CTN is pinned at 0.500. 24 trials per subject on these two cells make even a 10-point deviation decisive.\n\nDIFFERENCE 2 - DOES THE FLOOR VOICE'S LOUDNESS DEPEND ON THE CHORUS IT FACES? This is the core claim of CTN-BG (chorus measured in mid-expert equivalents, Q, with a narrow logistic gate at q0~2.45) and the explicitly denied claim of FM-ISN (context-free weights: 'a floor voice opposed by a pair and a floor voice opposed by a trio are read with exactly the same loudness'). The panel is engineered so that the SENIORITY of an opposing PAIR can be titrated while holding the count composition (1 vs 2, D=3) exactly fixed:\n  - MID PAIR = f6(.72, s=.45) + f3(.76, s=.55). Because the two seniorities are symmetric about 0.5, their mid-equivalent sum is EXACTLY 2.00 for every tau in [1.0,1.5] - guaranteed below CTN's threshold q0 in [2.38,2.52] at every wid in [.11,.17] (gate <= 0.20).\n  - STAR PAIR = f1(.94, s=1.00) + f4(.92, s=.95). Their mid-equivalent sum is 2.63-2.81 for every tau - guaranteed above q0 at every corner (gate >= 1.1).\nFor FM-ISN both opposing pairs have weight eta + (0.55)^kappa vs eta + 0^kappa, i.e. they differ by < 1e-4: its predicted C4-minus-C3 difference is < 0.001 - a PARAMETER-FREE ZERO. CTN predicts +0.17 to +0.40 (nominal +0.35). The same titration is REPLICATED one rung up the panel with the second-from-bottom voice f2(.58, s=.10), whose novelty is 0.17-0.23 in CTN and whose FM weight is a context-free 0.198: cells C5 (L vs mid pair) and C6 (L vs star pair) again give FM an exact zero contrast and CTN roughly +0.29. Two independent, oppositely-anchored replications of the same sign-of-contrast test rule out any generic 'one model is more decisive' account.\n\nANCHORS. C7 (all-upper mid trio beats a lone star, no floor voice discriminating) checks CTN's 'near-chance on all-upper displays at every margin' against FM's 'chance, solo >= multi' (CTN .63 vs FM .52) and is SELF-DIAGNOSTIC: a TTB/WADD subject would go far BELOW 0.5 here and far below 0.5 in C1-C6, falsifying both models. C8 (unanimous 2v0 bloc containing the floor voice) anchors the ceiling/lapse: CTN .76 (its D-dilution and thin residuals cap it well below ceiling) vs FM .91.\n\nPANEL: 7 experts, screen order deliberately scrambled so validity rank is decoupled from position: [f0=.74, f1=.94, f2=.58, f3=.76, f4=.92, f5=.54, f6=.72]; vmin=.54, vmax=.94, span=.40 -> s = .50/1.00/.10/.55/.95/.00/.45. Exactly ONE floor voice (f5, s=0: novelty/weight 1 in both models), exactly one second-from-bottom voice (f2, s=.10), three mids straddling s=0.5 and two stars. n=7 keeps BOTH models exactly at their capacity thresholds (over = 0), removing all panel-size nuisance.\n\nDESIGN: 8 configurations x 2 exact A/B mirrors = 16 unique pairs x K = 96//16 = 6 repetitions = 96 trials. Tied cues are coded sometimes 1-1 and sometimes 0-0 within and across cells so the number of displayed 1s never tracks condition or seniority; every trial requires reading several ratings.\n\nCELLS (A-favouring version first; the next trial is its exact mirror). Nominal P(choose the focal option) CTN-BG (eta=.04, c_res=.12, nu=1.88, kappa=15, tau=1.2, lam_own=.60, gmax=1.9, q0=2.45, wid=.14, c_n=2.9, beta=41, gamma=.96, eps=.12) vs FM-ISN-CC5 (eta=.013, kappa=16, beta=8.75, c_n=1.35, gamma=.40, rho_tie=.69, eps=.075):\nC1 (T1/T2) TALLY-TIE 1v1, FLOOR vs STAR: A wins f5(.54), B wins f1(.94). CTN 0.500 (exact) | FM 0.845 [.78,.88].\nC2 (T3/T4) TALLY-TIE 2v2: A wins f5(.54)+f0(.74), B wins f1(.94)+f4(.92). CTN 0.500 (exact) | FM 0.786 [.71,.84].\nC3 (T5/T6) FLOOR vs MID-PAIR (1v2, D=3, Q=2.00): A wins f5; B wins f6(.72), f3(.76). CTN 0.583 | FM 0.880.\nC4 (T7/T8) FLOOR vs STAR-PAIR (1v2, D=3, Q=2.71): A wins f5; B wins f1(.94), f4(.92). CTN 0.935 | FM 0.880.\nC5 (T9/T10) SECOND-FROM-BOTTOM vs MID-PAIR: A wins f2(.58); B wins f6, f3. CTN 0.508 | FM 0.638.\nC6 (T11/T12) SECOND-FROM-BOTTOM vs STAR-PAIR: A wins f2(.58); B wins f1, f4. CTN 0.797 | FM 0.638.\nC7 (T13/T14) ALL-UPPER MID-TRIO vs LONE STAR (3v1): A wins f0,f3,f6; B wins f1. CTN 0.631 | FM 0.522.\nC8 (T15/T16) UNANIMOUS FLOOR+STAR (2v0): A wins f5, f1. CTN 0.760 | FM 0.913.\n\nPREDICTED PROFILE (C1..C8): CTN-BG = .50, .50, .58, .94, .51, .80, .63, .76 ; FM-ISN-CC5 = .85, .79, .88, .88, .64, .64, .52, .91.\n\nINTERPRETATION: exact chance on the two tally-tied cells C1/C2 together with STEEP seniority titrations (C4-C3 ~ +0.3 and C6-C5 ~ +0.3, i.e. a floor voice opposed by two MID experts is a dead heat but the same voice opposed by two STARS is followed ~90% of the time) uniquely supports CTN-BG's chorus amplification. Strong anti-validity choice on the tally-tied cells plus FLAT titrations (C4 = C3 and C6 = C5 to within noise, all four near 0.88 / 0.64) uniquely supports FM-ISN-CC5's context-free inverse-seniority weights. Systematic BELOW-chance responding in C1-C6 (following the star) would falsify both models in favour of WADD/TTB, so the design is self-diagnostic.\n\nCONTROLS: every configuration appears in an A-favouring and a B-favouring mirror so side bias cancels exactly; validity rank is scrambled relative to screen position; tie coding alternates 1-1 / 0-0; no configuration is degenerate and no configuration has the two models predicting the same value.",
  "validities": [
    0.74,
    0.94,
    0.58,
    0.76,
    0.92,
    0.54,
    0.72
  ],
  "trial_a_ratings": [
    [
      1,
      0,
      1,
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
      1,
      0,
      0
    ],
    [
      1,
      0,
      0,
      1,
      0,
      1,
      1
    ],
    [
      0,
      1,
      0,
      1,
      1,
      0,
      1
    ],
    [
      0,
      1,
      1,
      0,
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
      0,
      1
    ],
    [
      1,
      0,
      0,
      1,
      0,
      1,
      0
    ],
    [
      1,
      1,
      0,
      1,
      1,
      0,
      0
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
      0,
      0,
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
      0,
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
      0,
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
      1,
      0,
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
      0,
      0
    ]
  ],
  "trial_b_ratings": [
    [
      1,
      1,
      1,
      0,
      1,
      0,
      0
    ],
    [
      1,
      0,
      1,
      0,
      1,
      1,
      0
    ],
    [
      0,
      1,
      0,
      1,
      1,
      0,
      1
    ],
    [
      1,
      0,
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
      0,
      0,
      1
    ],
    [
      0,
      1,
      1,
      0,
      0,
      1,
      0
    ],
    [
      1,
      1,
      0,
      1,
      1,
      0,
      0
    ],
    [
      1,
      0,
      0,
      1,
      0,
      1,
      0
    ],
    [
      1,
      0,
      0,
      1,
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
      0,
      1,
      0,
      1,
      1,
      1,
      1
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
      0,
      1,
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
      0,
      0,
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
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 56081,
  "output_tokens": 21649
}
```
