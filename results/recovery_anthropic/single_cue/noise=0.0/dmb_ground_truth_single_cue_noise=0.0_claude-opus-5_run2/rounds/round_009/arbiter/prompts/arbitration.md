# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_10" and "pi_11") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_10" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_11" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_10
**Chorus-Threshold Novelty Weighting, Balanced-Gradient variant (CTN-BG).**

People told the accuracies of a panel of binary expert ratings do not weight a rating by its accuracy; they weight it by how much *news* it carries, and news is loud only when it stands against a genuine CHORUS. Six commitments, all smooth and continuous (no mute band, no solo release, no discrete count switch).

**(1) The count comparator is a parameter-free gate.** Before any expert identity is consulted, an unweighted feature-wise comparison asks: does one product win on strictly more attributes? If not — equal win counts, including wholly non-discriminating displays — the display is non-diagnostic and the subject flips a mental coin: exactly 0.5, with NO validity tie-break and NO take-the-best fallback, at every panel composition and size.

**(2) Seniority is panel-relative; novelty is a smooth convex, monotonically DECREASING function of it, and the descent is STEEP.** s_j = (v_j − v_min)/(v_max − v_min); nov_j = (1 − s_j)^kappa with kappa ≈ 14–16.5. Nothing is muted by fiat and nothing is released by fiat, but the curve is sharp: only the panel's genuinely idiosyncratic BOTTOM voice speaks with nov ≈ 1; the second-from-bottom voice retains a real but small share; a mid voice essentially nothing; a star nothing at all. Novelty is never negative — a star is ignored, never counted against its own side.

**(3) THE CORE CLAIM — a novel voice is loud only in proportion to the CHORUS it speaks into, and a chorus is measured in *mid-expert equivalents*.** Each other discriminating voice contributes a subjective unit u_i = (1 + tau·s_i)/(1 + tau/2): a mid-panel voice counts as exactly one unit, a star ≈ 1.4, a bottom voice ≈ 0.6. The chorus a cue j faces is Q_j = Σ_{opposing} u_i + lam_own·Σ_{own side, i≠j} u_i — voices it contradicts counted in full, voices it merely stands beside echoed at a discount. Loudness is a smooth logistic of that chorus, g_j = Gmax·σ((Q_j − q0)/wid), with a narrow but continuous threshold at roughly two-and-a-half mid-equivalents. BOTH how many voices and how senior they are matter, in one scalar: two mid voices are not yet a chorus, two stars just barely are, three upper voices fully are.

**(4) The ungated panel gradient is THIN and only MILDLY convex — a bare ordinal pull that does not, by itself, give the panel's bottom voice an edge over a pair of mid voices.** Every discriminating cue contributes r_j = eta + c·(1 − s_j)^nu with small eta, small c and a moderate exponent nu ≈ 1.7–2.05. The scale is small enough that all-upper displays drift near chance at any count margin, and the exponent is flat enough that a single floor cue's residual is worth about the same as two mid cues' residuals — so a floor voice opposed by a mid PAIR is a dead heat (it needs the chorus gate, not the gradient, to win), while a floor voice opposed by a three-cue upper bloc is carried past chance by the gate.

**(5) Integration is Luce-like and saturating.** w_j = r_j + nov_j·g_j; E = Σ dir_j w_j; z = beta·E/((c_n + Σ_j w_j)·D^gamma). Evidence is judged relative to the total weight of evidence on the table plus a constant of indecision, and is diluted by the number of mutually discriminating attributes with gamma ≈ 1, so a lone cue and an unopposed pair of the same tier feel about equally compelling. An attention lapse epsilon mixes the result with a coin flip.

**(6) Capacity is a near-total collapse of the chorus construal past ~7 voices, with only mild extra noise.** For n > 7 the amplification essentially vanishes (Gmax × (1 − psi·(n−7))) while integration noise grows only slightly, so eight-expert star-backed majorities sit near chance rather than being reversed, and eight-expert plain tally adherence survives on the residual gradient.

**Cross-design signature:** exact chance on tally ties at any composition; near-chance on displays whose every discriminating voice is upper-band, at every margin; moderate (not ceiling) adherence to majorities containing no bottom voice; a bottom voice facing two MID voices lands at chance, facing two STARS reverses substantially, facing a three-cue upper bloc reverses to ≈.12–.18; near-ceiling following of majorities that contain the bottom voice; attenuated integration on eight-expert panels.

`predict(parameters, state, history) -> np.ndarray`:
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

`policy(probs) -> int`:
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

## THEORY 2 — pi_11
**Flat-Maverick Inverse-Seniority Integration with a Count-Confidence Gate and Mild Count Dilution (FM-ISN-CC5).**

When people are told the accuracies of a panel of binary raters, they do not weight a rating by its accuracy and they do not weight it by the crowd it happens to face. They weight it by one thing only: *how far down the displayed panel's accuracy range that voice sits*. Five commitments.

**(1) CONTEXT-FREE INVERSE-SENIORITY WEIGHT (the decisive claim), NEARLY CATEGORICAL about the true panel floor.** Seniority is panel-relative: s_j = (v_j − v_min)/(v_max − v_min) computed over the panel shown in the instructions. Each discriminating cue gets w_j = eta + (1 − s_j)^kappa with kappa ≈ 15–17 and a very thin flat floor eta ≈ 0.008–0.018. Only the panel's genuinely bottom voice speaks with full loudness; the SECOND-from-bottom voice already retains only about a quarter of it, and mid and star voices collapse onto the near-zero flat residual and are effectively interchangeable and mute. Crucially the weight is **completely independent of context**: it does not depend on how many voices oppose the cue, how senior those opponents are, whether the cue stands alone, or whether it sits inside a unanimous bloc. There is NO chorus amplification of any kind. A floor voice opposed by a pair, a floor voice opposed by a trio, and a floor voice inside a unanimous bloc are all read with exactly the same loudness — their observed rates differ only through the ordinary arithmetic of the weighted balance.

**(2) NORMALISED (LUCE-LIKE), SATURATING INTEGRATION WITH MILD COUNT DILUTION.** Evidence is the signed weighted balance E = Σ_j dir_j w_j, and confidence is that balance measured against the total subjective weight actually on the table plus a constant of indecision, further diluted by the number of mutually discriminating attributes: z = beta·E/((c_n + Σ_j w_j)·D^gamma) with gamma ≈ 0.34–0.46. Because a display made entirely of upper-band voices puts almost no subjective weight on the table, such displays sit near chance *at every count margin*, and the mild dilution makes a lone thin voice at least as decisive as an unopposed thin pair. A display containing the panel's floor voice immediately supplies enough weight to drive the choice near ceiling in the floor's direction.

**(3) THE COUNT COMPARATOR GATES CONFIDENCE, NOT DIRECTION.** An unweighted feature-wise count is computed first. It never supplies a direction of its own and never acts as a tie-break by validity rank. What it does is license confidence: when one option wins on strictly more attributes the weighted reading is taken at full gain; when the counts are equal (including wholly non-discriminating displays) the very same weighted reading is taken at a moderately reduced gain rho_tie ≈ 0.60–0.78, so tie displays hover close to chance — slightly *anti*-validity, never pro-validity.

**(4) CAPACITY IS AT MOST A WHISPER.** Holding a whole validity ordering in mind gets marginally harder past about seven voices; the integration slope is divided by 1 + chi·(n − 7) with chi ≈ 0–0.10, i.e. essentially absent. Eight-expert panels retain their tally adherence, and star-backed dense majorities stay at chance for the independent reason that their weights are all near eta.

**(5) LAPSE.** A small attention lapse epsilon ≈ 0.05–0.10 mixes the result with a coin flip.

**Cross-design signature:** near-chance (slightly anti-validity) on tally ties at any composition; chance on displays whose every discriminating voice is upper-band, at any margin, with solo ≥ multi-cue upper; a null on the two contrasts that falsify chorus theories — an opposed floor voice and a floor voice inside a unanimous bloc come out at the same rate, and titrating the seniority of an opposing pair produces a flat line; hard reversal of upper-backed majorities by a lone floor voice; near-ceiling following of majorities containing the floor voice; and only MODERATE anti-validity when the weak side is carried merely by the second-from-bottom voice rather than by the true floor.

`predict(parameters, state, history) -> np.ndarray`:
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

`policy(probs) -> int`:
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

## EXPERIMENT 1 (proposed by pi_10)

### DESIGN
**Validities (n_features=7):** [0.74, 0.94, 0.58, 0.76, 0.92, 0.54, 0.72]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 1, 0, 1, 1, 0]  B=[1, 1, 1, 0, 1, 0, 0]
  trial 2: A=[1, 1, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1, 0]
  trial 3: A=[1, 0, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 1, 0, 1]
  trial 4: A=[0, 1, 0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0, 1, 1]
  trial 5: A=[0, 1, 1, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0, 0, 1]
  trial 6: A=[0, 1, 1, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0, 1, 0]
  trial 7: A=[1, 0, 0, 1, 0, 1, 0]  B=[1, 1, 0, 1, 1, 0, 0]
  trial 8: A=[1, 1, 0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0, 1, 0]
  trial 9: A=[1, 0, 1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1, 0, 1]
  trial 10: A=[1, 0, 0, 1, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 0]
  trial 11: A=[0, 0, 1, 1, 0, 1, 1]  B=[0, 1, 0, 1, 1, 1, 1]
  trial 12: A=[0, 1, 0, 1, 1, 1, 1]  B=[0, 0, 1, 1, 0, 1, 1]
  trial 13: A=[1, 0, 0, 1, 1, 0, 1]  B=[0, 1, 0, 0, 1, 0, 0]
  trial 14: A=[0, 1, 0, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1, 0, 1]
  trial 15: A=[0, 1, 1, 1, 0, 1, 0]  B=[0, 0, 1, 1, 0, 0, 0]
  trial 16: A=[0, 0, 1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0, 1, 0]

**Rationale:** GOAL: dissociate CTN-BG (advocated) from FM-ISN-CC5 (competing). The two models are near-twins in surface behaviour: panel-relative seniority s=(v-vmin)/(vmax-vmin), a near-categorical inverse-seniority weight with kappa~15-17 (so only the panel floor voice speaks loudly and everything at s>=0.45 is effectively mute), Luce-like saturating integration with a constant of indecision, count dilution by D^gamma, and a lapse. Generic maverick cells (lone floor voice beating an upper bloc) are therefore WASTED - both predict reversal. They differ on exactly TWO structural commitments, and every cell below turns one of them into either a parameter-free prediction or a parameter-free ZERO contrast.

DIFFERENCE 1 - WHAT A TALLY TIE DOES. CTN-BG's count comparator is a parameter-free GATE on direction: when the two options win equally many attributes the display is non-diagnostic and the model returns EXACTLY 0.500 for every setting of its 15 parameters, with no validity tie-break. FM-ISN-CC5's count comparator gates only CONFIDENCE (gain rho_tie in [0.60,0.78]); the weighted inverse-seniority reading still runs, so a tie in which the floor voice backs one option and a star backs the other must come out strongly ANTI-validity. Cells C1 (1v1: .54 vs .94) and C2 (2v2: floor+mid vs the two stars) implement this. Across FM's ENTIRE box (beta 8.3-9.2, c_n 1.2-1.5, gamma .34-.46, rho_tie .60-.78, eps .05-.10) C1 lies in [0.78,0.88] and C2 in [0.71,0.84]; CTN is pinned at 0.500. 24 trials per subject on these two cells make even a 10-point deviation decisive.

DIFFERENCE 2 - DOES THE FLOOR VOICE'S LOUDNESS DEPEND ON THE CHORUS IT FACES? This is the core claim of CTN-BG (chorus measured in mid-expert equivalents, Q, with a narrow logistic gate at q0~2.45) and the explicitly denied claim of FM-ISN (context-free weights: 'a floor voice opposed by a pair and a floor voice opposed by a trio are read with exactly the same loudness'). The panel is engineered so that the SENIORITY of an opposing PAIR can be titrated while holding the count composition (1 vs 2, D=3) exactly fixed:
  - MID PAIR = f6(.72, s=.45) + f3(.76, s=.55). Because the two seniorities are symmetric about 0.5, their mid-equivalent sum is EXACTLY 2.00 for every tau in [1.0,1.5] - guaranteed below CTN's threshold q0 in [2.38,2.52] at every wid in [.11,.17] (gate <= 0.20).
  - STAR PAIR = f1(.94, s=1.00) + f4(.92, s=.95). Their mid-equivalent sum is 2.63-2.81 for every tau - guaranteed above q0 at every corner (gate >= 1.1).
For FM-ISN both opposing pairs have weight eta + (0.55)^kappa vs eta + 0^kappa, i.e. they differ by < 1e-4: its predicted C4-minus-C3 difference is < 0.001 - a PARAMETER-FREE ZERO. CTN predicts +0.17 to +0.40 (nominal +0.35). The same titration is REPLICATED one rung up the panel with the second-from-bottom voice f2(.58, s=.10), whose novelty is 0.17-0.23 in CTN and whose FM weight is a context-free 0.198: cells C5 (L vs mid pair) and C6 (L vs star pair) again give FM an exact zero contrast and CTN roughly +0.29. Two independent, oppositely-anchored replications of the same sign-of-contrast test rule out any generic 'one model is more decisive' account.

ANCHORS. C7 (all-upper mid trio beats a lone star, no floor voice discriminating) checks CTN's 'near-chance on all-upper displays at every margin' against FM's 'chance, solo >= multi' (CTN .63 vs FM .52) and is SELF-DIAGNOSTIC: a TTB/WADD subject would go far BELOW 0.5 here and far below 0.5 in C1-C6, falsifying both models. C8 (unanimous 2v0 bloc containing the floor voice) anchors the ceiling/lapse: CTN .76 (its D-dilution and thin residuals cap it well below ceiling) vs FM .91.

PANEL: 7 experts, screen order deliberately scrambled so validity rank is decoupled from position: [f0=.74, f1=.94, f2=.58, f3=.76, f4=.92, f5=.54, f6=.72]; vmin=.54, vmax=.94, span=.40 -> s = .50/1.00/.10/.55/.95/.00/.45. Exactly ONE floor voice (f5, s=0: novelty/weight 1 in both models), exactly one second-from-bottom voice (f2, s=.10), three mids straddling s=0.5 and two stars. n=7 keeps BOTH models exactly at their capacity thresholds (over = 0), removing all panel-size nuisance.

DESIGN: 8 configurations x 2 exact A/B mirrors = 16 unique pairs x K = 96//16 = 6 repetitions = 96 trials. Tied cues are coded sometimes 1-1 and sometimes 0-0 within and across cells so the number of displayed 1s never tracks condition or seniority; every trial requires reading several ratings.

CELLS (A-favouring version first; the next trial is its exact mirror). Nominal P(choose the focal option) CTN-BG (eta=.04, c_res=.12, nu=1.88, kappa=15, tau=1.2, lam_own=.60, gmax=1.9, q0=2.45, wid=.14, c_n=2.9, beta=41, gamma=.96, eps=.12) vs FM-ISN-CC5 (eta=.013, kappa=16, beta=8.75, c_n=1.35, gamma=.40, rho_tie=.69, eps=.075):
C1 (T1/T2) TALLY-TIE 1v1, FLOOR vs STAR: A wins f5(.54), B wins f1(.94). CTN 0.500 (exact) | FM 0.845 [.78,.88].
C2 (T3/T4) TALLY-TIE 2v2: A wins f5(.54)+f0(.74), B wins f1(.94)+f4(.92). CTN 0.500 (exact) | FM 0.786 [.71,.84].
C3 (T5/T6) FLOOR vs MID-PAIR (1v2, D=3, Q=2.00): A wins f5; B wins f6(.72), f3(.76). CTN 0.583 | FM 0.880.
C4 (T7/T8) FLOOR vs STAR-PAIR (1v2, D=3, Q=2.71): A wins f5; B wins f1(.94), f4(.92). CTN 0.935 | FM 0.880.
C5 (T9/T10) SECOND-FROM-BOTTOM vs MID-PAIR: A wins f2(.58); B wins f6, f3. CTN 0.508 | FM 0.638.
C6 (T11/T12) SECOND-FROM-BOTTOM vs STAR-PAIR: A wins f2(.58); B wins f1, f4. CTN 0.797 | FM 0.638.
C7 (T13/T14) ALL-UPPER MID-TRIO vs LONE STAR (3v1): A wins f0,f3,f6; B wins f1. CTN 0.631 | FM 0.522.
C8 (T15/T16) UNANIMOUS FLOOR+STAR (2v0): A wins f5, f1. CTN 0.760 | FM 0.913.

PREDICTED PROFILE (C1..C8): CTN-BG = .50, .50, .58, .94, .51, .80, .63, .76 ; FM-ISN-CC5 = .85, .79, .88, .88, .64, .64, .52, .91.

INTERPRETATION: exact chance on the two tally-tied cells C1/C2 together with STEEP seniority titrations (C4-C3 ~ +0.3 and C6-C5 ~ +0.3, i.e. a floor voice opposed by two MID experts is a dead heat but the same voice opposed by two STARS is followed ~90% of the time) uniquely supports CTN-BG's chorus amplification. Strong anti-validity choice on the tally-tied cells plus FLAT titrations (C4 = C3 and C6 = C5 to within noise, all four near 0.88 / 0.64) uniquely supports FM-ISN-CC5's context-free inverse-seniority weights. Systematic BELOW-chance responding in C1-C6 (following the star) would falsify both models in favour of WADD/TTB, so the design is self-diagnostic.

CONTROLS: every configuration appears in an A-favouring and a B-favouring mirror so side bias cancels exactly; validity rank is scrambled relative to screen position; tie coding alternates 1-1 / 0-0; no configuration is degenerate and no configuration has the two models predicting the same value.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
The two models are near-twins except on two structural commitments, and this metric puts both of them into one scalar with the SAME sign for the competitor and OPPOSITE signs for me, so the gap adds rather than cancels.

Component 1 (24 trials/subject): rate of choosing the option backed by the panel FLOOR voice (f5, v=.54) on displays where the two options win equally many attributes (cells C1 1v1 and C2 2v2). CTN-BG's count comparator gates DIRECTION, so a tally tie is non-diagnostic and returns exactly 0.500 for every parameter setting (~0.50 observed). FM-ISN-CC5's comparator gates only CONFIDENCE (rho_tie), so the inverse-seniority reading still runs and yields a strongly anti-validity ~0.78-0.85.

Component 2 (48 trials/subject): the average seniority-titration contrast, [P(follow floor | opposed by STAR pair) - P(follow floor | opposed by MID pair)] and its replication with the second-from-bottom voice (f2). Count composition (1 vs 2, D=3) is held fixed; only the seniority of the opposing pair changes. FM's weights are context-free, so both opposing pairs differ by <1e-4 in weight: a parameter-free ZERO (~0.00). CTN's chorus gate (mid pair Q=2.00 below q0, star pair Q=2.71 above) predicts ~+0.32.

Metric = tie_rate - mean(titration diffs). Predicted: CTN-BG ~ 0.50 - 0.32 = 0.18; FM-ISN-CC5 ~ 0.82 - 0.00 = 0.82. Gap ~0.64 with per-subject SD ~0.17-0.19 (72 of 96 trials contribute), giving a very large Welch t at N=25. It is not trivially true for my theory: if humans show anti-validity choice on tally ties AND flat titrations, the metric lands near the competitor's value and falsifies CTN-BG; systematic star-following (WADD/TTB) would push tie_rate well below 0.5 and falsify both.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    # Design-time panel (hard-coded from the experimental design):
    # validities = [0.74, 0.94, 0.58, 0.76, 0.92, 0.54, 0.72]
    floor_i = 5      # v = 0.54  -> panel floor voice
    second_i = 2     # v = 0.58  -> second-from-bottom voice
    star_set = frozenset([1, 4])       # 0.94, 0.92
    mid_set = frozenset([0, 3, 6])     # 0.74, 0.76, 0.72

    tie = [0, 0]
    cells = {'C3': [0, 0], 'C4': [0, 0], 'C5': [0, 0], 'C6': [0, 0]}

    for a_r, b_r, resp in zip(data['option_a_ratings'],
                              data['option_b_ratings'],
                              data['response']):
        try:
            a = [int(x) for x in a_r]
            b = [int(x) for x in b_r]
            r = int(resp)
        except Exception:
            continue
        if len(a) != 7 or len(b) != 7:
            continue
        winA = [i for i in range(7) if a[i] > b[i]]
        winB = [i for i in range(7) if b[i] > a[i]]
        if len(winA) == 0 and len(winB) == 0:
            continue

        # ---- tally-tie cells (C1: 1v1 floor vs star, C2: 2v2 with floor) ----
        if len(winA) == len(winB):
            if floor_i in winA:
                focal_is_A = True
            elif floor_i in winB:
                focal_is_A = False
            else:
                continue
            tie[1] += 1
            if (r == 0) == focal_is_A:
                tie[0] += 1
            continue

        # ---- 1-vs-2 seniority-titration cells ----
        if len(winA) == 1 and len(winB) == 2:
            solo = winA[0]
            opp = frozenset(winB)
            focal_is_A = True
        elif len(winB) == 1 and len(winA) == 2:
            solo = winB[0]
            opp = frozenset(winA)
            focal_is_A = False
        else:
            continue

        if opp == star_set:
            tag = 'star'
        elif opp <= mid_set:
            tag = 'mid'
        else:
            continue

        if solo == floor_i:
            key = 'C4' if tag == 'star' else 'C3'
        elif solo == second_i:
            key = 'C6' if tag == 'star' else 'C5'
        else:
            continue

        cells[key][1] += 1
        if (r == 0) == focal_is_A:
            cells[key][0] += 1

    def rate(pair):
        return (pair[0] / float(pair[1])) if pair[1] > 0 else None

    tr = rate(tie)
    if tr is None:
        tr = 0.5

    diffs = []
    for hi, lo in (('C4', 'C3'), ('C6', 'C5')):
        h = rate(cells[hi])
        l = rate(cells[lo])
        if (h is not None) and (l is not None):
            diffs.append(h - l)
    d = float(np.mean(diffs)) if len(diffs) > 0 else 0.0

    return float(tr - d)


### RESULTS
- Predicted under pi_10 (simulated): 0.1808 (var=0.0243)
- Predicted under pi_11 (simulated): 0.8033 (var=0.0147)
- Observed on real data: 0.8417 (var=0.0217)

## EXPERIMENT 2 (proposed by pi_11)

### DESIGN
**Validities (n_features=7):** [0.78, 0.95, 0.55, 0.71, 0.88, 0.62, 0.83]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0, 1]
  trial 2: A=[1, 1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1]
  trial 3: A=[0, 0, 1, 1, 0, 1, 0]  B=[0, 1, 0, 1, 1, 0, 0]
  trial 4: A=[0, 1, 0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 1, 0]
  trial 5: A=[0, 1, 1, 0, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1, 0]
  trial 6: A=[1, 1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 0, 1, 0]
  trial 7: A=[0, 0, 1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1, 0, 1]
  trial 8: A=[1, 0, 0, 1, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0, 0]
  trial 9: A=[1, 0, 1, 0, 0, 1, 0]  B=[1, 1, 0, 0, 1, 1, 0]
  trial 10: A=[1, 1, 0, 0, 1, 1, 0]  B=[1, 0, 1, 0, 0, 1, 0]
  trial 11: A=[0, 1, 1, 1, 1, 0, 1]  B=[0, 0, 0, 1, 0, 0, 1]
  trial 12: A=[0, 0, 0, 1, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0, 1]
  trial 13: A=[1, 1, 0, 1, 1, 0, 1]  B=[0, 1, 0, 0, 1, 0, 1]
  trial 14: A=[0, 1, 0, 0, 1, 0, 1]  B=[1, 1, 0, 1, 1, 0, 1]
  trial 15: A=[1, 0, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0]
  trial 16: A=[1, 0, 0, 1, 0, 1, 0]  B=[1, 0, 1, 1, 0, 1, 0]

**Rationale:** GOAL: dissociate FM-ISN-CC5 (advocated) from CTN-BG (competing). The two models are near-twins in surface behaviour 
taboth use panel-relative seniority s=(v-vmin)/(vmax-vmin), a near-categorical inverse-seniority weight with kappa~15-17 (only the panel floor voice speaks loudly; everything at s>=0.40 is effectively mute), Luce-like saturating integration with a constant of indecision, dilution by D^gamma, a lapse, and a capacity term past 7 voices. Generic maverick cells (lone floor voice beating an upper bloc) are therefore WASTED. They differ on exactly TWO structural commitments, and every cell here turns one of them into a parameter-free prediction or a parameter-free ZERO contrast.

DIFFERENCE 1 
taWHAT A TALLY TIE DOES. CTN-BG's count comparator is a parameter-free DIRECTION gate: when both options win equally many attributes it returns EXACTLY 0.500 for every setting of its 15 parameters, no validity tie-break. FM-ISN-CC5 gates only CONFIDENCE (rho_tie in [0.60,0.78]); the weighted inverse-seniority reading still runs, so a tie in which the floor voice backs one option and a star backs the other must come out strongly ANTI-validity. C1 (1v1: .55 vs .95) and C2 (2v2: floor+second-from-bottom vs the two stars) implement this: across FM's entire box C1 lies ~[.78,.88] and C2 ~[.71,.84]; CTN is pinned at .500. 24 trials/subject on these two cells make a 10-point deviation decisive.

DIFFERENCE 2 
taDOES THE FLOOR VOICE'S LOUDNESS DEPEND ON THE CHORUS IT FACES? FM's decisive claim is that it does NOT (context-free weights): a floor voice opposed by a pair, by a trio, or sitting inside a unanimous bloc is read with exactly the same loudness, so observed rates differ only via the D^gamma dilution and the total weight on the table. CTN's core claim is that loudness is a steep logistic of the chorus measured in mid-expert equivalents Q (q0~2.45, wid~.14). The panel is engineered so that THREE different manipulations of the opposing bloc cross CTN's threshold while leaving FM's weights literally unchanged (all opposers have (1-s)^kappa < 5e-4, i.e. weight = eta to 4 decimals):
  (a) COUNT TITRATION at fixed seniority band: floor vs MID-PAIR f0(.78)+f3(.71) gives Q=1.98 (below q0 at every tau in [1.0,1.5]); floor vs MID-TRIO f0+f3+f6(.83) gives Q=3.12-3.15 (above q0 at every corner). FM: .880 -> .856 (flat, slightly NEGATIVE because of D dilution). CTN: .570 -> .923 (+0.35).
  (b) SENIORITY TITRATION at fixed count: floor vs MID-PAIR (Q=1.98) versus floor vs STAR-PAIR f1(.95)+f4(.88) (Q=2.63-2.81, above q0 at every corner). FM: .880 vs .880 
taa parameter-free ZERO. CTN: .570 vs .932 (+0.36).
  (c) OPPOSITION vs UNANIMITY at fixed seniority composition: the same floor voice and the same two stars, now all on one side (3v0). CTN's own-side echo is lam_own*2.63-2.81 = 1.32-1.97 mid-equivalents, BELOW q0 at every corner, so the amplification switches off: .932 -> .724 (-0.21). FM's context-free weight is unchanged: .880 -> .889 (+0.01).
The three contrasts run in different directions for CTN (+.35, +.36, -.21) but are all ~0 for FM, so no generic 'one model is more/less decisive' account can mimic the pattern.

ANCHORS / SELF-DIAGNOSTICS. C7 (unopposed MID pair f0+f3, a weakly dominant all-upper display with zero conflict) tests FM's 'chance on displays whose every discriminating voice is upper-band, at any margin': FM .529 vs CTN .709 
taCTN's thicker residual gradient (eta=.04, c_res=.12) plus beta=41 lifts it well above chance. C8 (SOLO FLOOR, D=1) is the ceiling/lapse anchor: FM .941 vs CTN .848. Both C7 and C8 are self-diagnostic: a WADD/TTB subject would be near 1.0 in C7 and near 0.5-1.0 in C8 while going far BELOW 0.5 in C1-C6, falsifying both models.

PANEL: 7 experts, screen order deliberately scrambled so validity rank is decoupled from position: [f0=.78, f1=.95, f2=.55, f3=.71, f4=.88, f5=.62, f6=.83]; vmin=.55, vmax=.95, span=.40 -> s = .575/1.000/.000/.400/.825/.175/.700. Exactly ONE floor voice (f2, s=0: weight/novelty 1 in both models), exactly one second-from-bottom voice (f5, s=.175, FM weight .046-.074, CTN novelty .038-.056), two mids straddling s=0.5 whose mid-equivalent sum is ~1.98 for every tau, one upper-mid (f6) that pushes the trio over q0, and two stars whose pair sum is >=2.63 for every tau. n=7 keeps BOTH models exactly at their capacity thresholds (over = 0), removing panel-size nuisance.

DESIGN: 8 configurations x 2 exact A/B mirrors = 16 unique pairs x K = 96//16 = 6 repetitions = 96 trials. Tied cues are coded sometimes 1-1 and sometimes 0-0 within and across cells so the number of displayed 1s never tracks condition or seniority; every trial requires reading several ratings.

CELLS (A-favouring version first; the next trial is its exact mirror). Nominal P(choose the focal option) FM-ISN-CC5 (eta=.013, kappa=16, beta=8.75, c_n=1.35, gamma=.40, rho_tie=.69, eps=.075) vs CTN-BG (eta=.04, c_res=.12, nu=1.88, kappa=15, tau=1.2, lam_own=.60, gmax=1.9, q0=2.45, wid=.14, c_n=2.9, beta=41, gamma=.96, eps=.12):
C1 (T1/T2) TALLY-TIE 1v1: A wins f2(.55), B wins f1(.95). FM .845 | CTN .500 (exact).
C2 (T3/T4) TALLY-TIE 2v2: A wins f2(.55)+f5(.62), B wins f1(.95)+f4(.88). FM .792 | CTN .500 (exact).
C3 (T5/T6) FLOOR vs MID-PAIR (1v2, D=3, Q=1.98): A wins f2; B wins f0(.78), f3(.71). FM .880 | CTN .570.
C4 (T7/T8) FLOOR vs MID-TRIO (1v3, D=4, Q=3.13): A wins f2; B wins f0, f3, f6(.83). FM .856 | CTN .923.
C5 (T9/T10) FLOOR vs STAR-PAIR (1v2, D=3, Q=2.72): A wins f2; B wins f1(.95), f4(.88). FM .880 | CTN .932.
C6 (T11/T12) UNANIMOUS FLOOR+STAR-PAIR (3v0, D=3): A wins f2, f1, f4. FM .889 | CTN .724.
C7 (T13/T14) UNOPPOSED MID-PAIR (2v0, D=2, no floor voice): A wins f0(.78), f3(.71). FM .529 | CTN .709.
C8 (T15/T16) SOLO FLOOR (1v0, D=1): A wins f2 only. FM .941 | CTN .848.

PREDICTED PROFILE (C1..C8): FM-ISN-CC5 = .85, .79, .88, .86, .88, .89, .53, .94 ; CTN-BG = .50, .50, .57, .92, .93, .72, .71, .85.

INTERPRETATION: strong anti-validity choice on the two tally-tied cells (C1/C2 well above .70) together with FLAT titrations (C4 ~ C3 ~ C5 ~ C6, all near .88, i.e. a floor voice is followed equally whether opposed by two mids, three mids, two stars, or flanked by the two stars on its own side) and near-chance responding on the unopposed upper pair C7 uniquely supports FM-ISN-CC5's context-free inverse-seniority weights with a confidence-only count gate. Exact chance on C1/C2 together with a collapse to ~.57 in C3 and jumps to >=.92 in C4/C5 and a drop to ~.72 in C6 uniquely supports CTN-BG's chorus amplification. Systematic BELOW-chance responding in C1-C6 (following the star) would falsify both models in favour of WADD/TTB, so the design is self-diagnostic.

CONTROLS: every configuration appears in an A-favouring and a B-favouring mirror so side bias cancels exactly; validity rank is scrambled relative to screen position; tie coding alternates 1-1 / 0-0; no configuration is degenerate and no configuration has the two models predicting the same value.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
The two theories make a parameter-free structural split on TALLY-TIED displays (cells C1 and C2: 1-vs-1 with .55 against .95, and 2-vs-2 with .55+.62 against .95+.88 \u2014 24 trials per subject). CTN-BG's count comparator is a hard, parameter-free gate: equal win counts => exactly P=0.5 for every parameter setting, no validity tie-break, so its per-subject rate of choosing the low-validity-backed option is pinned at chance with purely binomial noise (var ~ .25/24 = .010). FM-ISN-CC5 gates only confidence (rho_tie in [.60,.78]) and still reads the weighted inverse-seniority balance, so the floor voice wins these ties strongly anti-validity (~.79-.85 across the whole parameter box), with small between-subject spread. The metric measures exactly the proportion of tally-tied trials on which the subject chose the option backed by the lowest-validity discriminating expert (computed generically via an inverse-seniority weighting, so it is not hard-coded to specific trials). Expected ~0.82 under FM vs ~0.50 under CTN, with tight per-subject variance under both, yielding a very large Welch t at N=25. It is also diagnostic for humans: a WADD/TTB responder would fall far BELOW 0.5, distinguishing that outcome from both theories.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    V = np.array([0.78, 0.95, 0.55, 0.71, 0.88, 0.62, 0.83], dtype=float)

    hits = 0.0
    total = 0.0
    for _, row in data.iterrows():
        try:
            a = np.asarray(list(row['option_a_ratings']), dtype=float).ravel()
            b = np.asarray(list(row['option_b_ratings']), dtype=float).ravel()
        except Exception:
            continue
        if a.size == 0 or a.size != b.size:
            continue
        n = int(a.size)
        if n == V.size:
            v = V
        else:
            v = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.75])

        d = np.sign(a - b)
        iA = np.where(d > 0)[0]
        iB = np.where(d < 0)[0]
        mA = int(iA.size)
        mB = int(iB.size)
        D = mA + mB
        # only TALLY-TIED, discriminating displays
        if D == 0 or mA != mB:
            continue

        vmin = float(np.min(v))
        vmax = float(np.max(v))
        span = vmax - vmin
        if not np.isfinite(span) or span <= 1e-9:
            continue
        s = np.clip((v - vmin) / span, 0.0, 1.0)
        w = np.power(np.clip(1.0 - s, 0.0, 1.0), 16.0)

        E = float(np.sum(w[iA])) - float(np.sum(w[iB]))
        if not np.isfinite(E) or abs(E) < 1e-9:
            continue
        # favored side under inverse-seniority (low-validity / "floor voice") reading
        favored = 0 if E > 0 else 1
        try:
            resp = int(row['response'])
        except Exception:
            continue
        total += 1.0
        if resp == favored:
            hits += 1.0

    if total == 0.0:
        return 0.5
    return float(hits / total)

### RESULTS
- Predicted under pi_10 (simulated): 0.4950 (var=0.0087)
- Predicted under pi_11 (simulated): 0.8325 (var=0.0064)
- Observed on real data: 0.8700 (var=0.0105)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    validities = [0.95, 0.85, 0.75, 0.62, 0.55]
    hits = []
    for a_raw, b_raw, r in zip(data['option_a_ratings'],
                               data['option_b_ratings'],
                               data['response']):
        a = np.asarray(list(a_raw), dtype=float)
        b = np.asarray(list(b_raw), dtype=float)
        n = a.shape[0]
        if b.shape[0] != n or n == 0:
            continue
        if n == len(validities):
            v = np.asarray(validities, dtype=float)
        else:
            # fall back: assume features already ordered by descending validity
            v = np.arange(n, 0, -1).astype(float)
        order = np.argsort(-v, kind='stable')

        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        if a_wins != b_wins:
            continue  # only tally-tie trials

        winner = None
        for j in order:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
        if winner is None:
            continue  # no discriminating cue at all

        try:
            resp = int(r)
        except Exception:
            continue
        hits.append(1.0 if resp == winner else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.4738 (var=0.0073)
**Predicted under pi_10:** 0.5112 (var=0.0071)
**Predicted under pi_11:** 0.4900 (var=0.0081)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 0, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 1, 1, 0, 1, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    val = np.array([0.93, 0.57, 0.79, 0.86, 0.62, 0.71], dtype=float)
    hits = []
    for a_raw, b_raw, r in zip(data['option_a_ratings'],
                               data['option_b_ratings'],
                               data['response']):
        a = np.asarray(list(a_raw), dtype=float)
        b = np.asarray(list(b_raw), dtype=float)
        n = a.shape[0]
        if b.shape[0] != n or n == 0:
            continue
        if val.shape[0] == n:
            order = np.argsort(-val, kind='stable')
        else:
            order = np.arange(n)
        # tally margin (A wins minus B wins)
        margin = int(np.sum(a > b) - np.sum(b > a))
        if abs(margin) < 2:
            continue
        tally_winner = 0 if margin > 0 else 1
        # take-the-best winner
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
            continue  # keep only conflict trials
        try:
            resp = int(r)
        except Exception:
            continue
        hits.append(1.0 if resp == tally_winner else 0.0)
    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.6538 (var=0.0092)
**Predicted under pi_10:** 0.7531 (var=0.0055)
**Predicted under pi_11:** 0.7231 (var=0.0047)

### Experiment 5
**Design**
  A=[1, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 1, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    hits = []
    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()
    for a, b, r in zip(a_col, b_col, r_col):
        av = np.asarray(a, dtype=float).ravel()
        bv = np.asarray(b, dtype=float).ravel()
        if av.size == 0 or av.size != bv.size:
            continue
        nA = float(np.sum(av > bv))
        nB = float(np.sum(bv > av))
        if nA == nB:
            continue
        winner = 0 if nA > nB else 1
        try:
            resp = int(r)
        except Exception:
            continue
        hits.append(1.0 if resp == winner else 0.0)
    if len(hits) == 0:
        return float('nan')
    return float(np.mean(hits))

```

**Observed (real) value:** 0.7937 (var=0.0071)
**Predicted under pi_10:** 0.8108 (var=0.0026)
**Predicted under pi_11:** 0.8098 (var=0.0017)

### Experiment 6
**Design**
  A=[1, 1, 0, 1, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 0, 1, 0]  B=[1, 1, 0, 1, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 1, 0, 1]  B=[0, 0, 1, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 1, 1, 0, 1]
  A=[1, 1, 1, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 1, 0, 0]  B=[1, 1, 1, 1, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 1, 1, 0, 1]  B=[0, 0, 0, 1, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 0, 1, 0]  B=[1, 1, 1, 0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Adherence to the full-profile tally winner.

    For every trial we recompute the global feature-wise win counts
    nA = #(A > B) and nB = #(B > A) over ALL features, identify the
    option favoured by the complete tally, and score whether the subject
    chose it. The returned scalar is the proportion of such trials on
    which the subject followed the complete-tally winner.

    Works identically on the pooled DataFrame and on a single subject's
    slice.
    """
    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()

    hits = []
    for a_raw, b_raw, r in zip(a_col, b_col, r_col):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        if a.shape[0] != b.shape[0] or a.shape[0] == 0:
            continue
        nA = float(np.sum(a > b))
        nB = float(np.sum(b > a))
        if nA == nB:
            continue  # undiagnostic (no complete-tally winner)
        winner = 0 if nA > nB else 1
        try:
            resp = int(r)
        except (TypeError, ValueError):
            continue
        if resp not in (0, 1):
            continue
        hits.append(1.0 if resp == winner else 0.0)

    if len(hits) == 0:
        return float('nan')
    return float(np.mean(hits))
```

**Observed (real) value:** 0.7546 (var=0.0064)
**Predicted under pi_10:** 0.6992 (var=0.0030)
**Predicted under pi_11:** 0.7412 (var=0.0014)

### Experiment 7
**Design**
  A=[1, 0, 1, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 1]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    validities = np.array([0.94, 0.57, 0.86, 0.62, 0.70, 0.78], dtype=float)

    coh_hit = 0.0
    coh_n = 0.0
    con_hit = 0.0
    con_n = 0.0

    for a_raw, b_raw, resp in zip(data['option_a_ratings'],
                                  data['option_b_ratings'],
                                  data['response']):
        a = np.asarray(list(a_raw), dtype=float)
        b = np.asarray(list(b_raw), dtype=float)
        n = a.shape[0]
        if n == 0:
            continue
        if validities.shape[0] == n:
            v = validities
        else:
            v = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.75])

        a_win = a > b
        b_win = b > a
        nA = int(a_win.sum())
        nB = int(b_win.sum())
        if nA == nB:
            continue  # no count winner; excluded
        winner = 0 if nA > nB else 1

        disc = a_win | b_win
        if not disc.any():
            continue
        # highest-validity discriminating cue
        idx = int(np.argmax(np.where(disc, v, -np.inf)))
        top_owner = 0 if a_win[idx] else 1
        coherent = (top_owner == winner)

        try:
            r = int(resp)
        except Exception:
            continue
        hit = 1.0 if r == winner else 0.0

        if coherent:
            coh_hit += hit
            coh_n += 1.0
        else:
            con_hit += hit
            con_n += 1.0

    if coh_n == 0.0 or con_n == 0.0:
        return 0.0
    return float(coh_hit / coh_n - con_hit / con_n)

```

**Observed (real) value:** -0.7633 (var=0.0200)
**Predicted under pi_10:** -0.7558 (var=0.0048)
**Predicted under pi_11:** -0.7333 (var=0.0048)

### Experiment 8
**Design**
  A=[1, 1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0, 1]
  A=[0, 1, 0, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 1, 1]  B=[0, 1, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 0, 1]  B=[1, 0, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0, 0, 1]
  A=[1, 0, 0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 1, 0, 1]
  A=[1, 1, 1, 1, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    V = np.array([0.94, 0.58, 0.86, 0.63, 0.72, 0.80, 0.55], dtype=float)

    coh_hit = 0.0; coh_n = 0
    con_hit = 0.0; con_n = 0
    us_hit = 0.0; us_n = 0
    ud_hit = 0.0; ud_n = 0

    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()

    for a_raw, b_raw, resp in zip(a_col, b_col, r_col):
        try:
            a = np.asarray(a_raw, dtype=float).ravel()
            b = np.asarray(b_raw, dtype=float).ravel()
        except Exception:
            continue
        n = int(a.size)
        if n == 0 or int(b.size) != n:
            continue
        if n == int(V.size):
            w = V - 0.5
        elif n > 1:
            w = np.linspace(0.95, 0.55, n) - 0.5
        else:
            w = np.array([0.25])

        aw = a > b
        bw = b > a
        nA = int(np.sum(aw))
        nB = int(np.sum(bw))
        if nA == nB:
            continue
        winner = 0 if nA > nB else 1
        try:
            r = int(resp)
        except Exception:
            continue
        if r not in (0, 1):
            continue
        hit = 1.0 if r == winner else 0.0
        D = nA + nB
        if winner == 0:
            Ww = float(np.sum(w[aw])); Wl = float(np.sum(w[bw])); n_lose = nB
        else:
            Ww = float(np.sum(w[bw])); Wl = float(np.sum(w[aw])); n_lose = nA

        if n_lose == 0:
            # unanimous profiles: sparse (few discriminating cues) vs dense
            if D <= 2:
                us_hit += hit; us_n += 1
            else:
                ud_hit += hit; ud_n += 1
        else:
            if Ww > Wl:
                coh_hit += hit; coh_n += 1
            elif Ww < Wl:
                con_hit += hit; con_n += 1

    if coh_n > 0 and con_n > 0:
        d1 = coh_hit / coh_n - con_hit / con_n
    else:
        d1 = 0.0
    if us_n > 0 and ud_n > 0:
        d2 = us_hit / us_n - ud_hit / ud_n
    else:
        d2 = 0.0

    return float(d1 + d2)

```

**Observed (real) value:** -0.9375 (var=0.0693)
**Predicted under pi_10:** -0.7575 (var=0.0424)
**Predicted under pi_11:** -1.0337 (var=0.0305)

### Experiment 9
**Design**
  A=[1, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 1]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Design validities (fixed by the experiment): [0.95, 0.57, 0.88, 0.60, 0.66, 0.83]
    # Median split -> STAR tier (high validity)  = features 0, 2, 5
    #              -> MARGINAL tier (low validity) = features 1, 3, 4
    validities = np.array([0.95, 0.57, 0.88, 0.60, 0.66, 0.83], dtype=float)

    hits = 0.0
    n = 0.0

    for _, row in data.iterrows():
        try:
            a = np.asarray(list(row['option_a_ratings']), dtype=float).ravel()
            b = np.asarray(list(row['option_b_ratings']), dtype=float).ravel()
        except Exception:
            continue
        if a.size != b.size or a.size == 0:
            continue
        m = a.size
        if validities.size == m:
            v = validities
        else:
            v = np.linspace(0.95, 0.55, m)
        order = np.argsort(-v, kind='stable')
        k = m // 2
        star_idx = order[:k]
        marg_idx = order[k:]

        d = np.sign(a - b)  # +1 A wins the cue, -1 B wins the cue, 0 tie
        star_score = float(np.sum(d[star_idx]))
        marg_score = float(np.sum(d[marg_idx]))

        # keep only trials where BOTH tiers discriminate and they DISAGREE
        if star_score == 0.0 or marg_score == 0.0:
            continue
        if np.sign(star_score) == np.sign(marg_score):
            continue

        resp = row['response']
        try:
            resp = int(resp)
        except Exception:
            continue
        if resp not in (0, 1):
            continue

        marginal_preferred = 0 if marg_score > 0 else 1  # 0 = A, 1 = B
        hits += 1.0 if resp == marginal_preferred else 0.0
        n += 1.0

    if n == 0.0:
        return 0.5
    return float(hits / n)
```

**Observed (real) value:** 0.6606 (var=0.0038)
**Predicted under pi_10:** 0.6647 (var=0.0029)
**Predicted under pi_11:** 0.7494 (var=0.0033)

### Experiment 10
**Design**
  A=[0, 1, 0, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0, 1, 1, 0]  B=[0, 1, 0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 1, 0, 0]  B=[0, 1, 1, 0, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 1, 0, 1]  B=[0, 1, 0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 1, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 0, 1, 0, 1]  B=[0, 0, 1, 1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 0, 1, 0]  B=[1, 1, 1, 0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Design-time validities (fixed by the experiment), used only to define the
    # median-split 'star' tier structurally.
    v = np.array([0.95, 0.56, 0.85, 0.60, 0.65, 0.90, 0.53, 0.80], dtype=float)
    n = v.shape[0]
    order = np.argsort(-v, kind='stable')
    star = set(int(i) for i in order[: n // 2])

    hits = []
    for a_raw, b_raw, resp in zip(data['option_a_ratings'],
                                  data['option_b_ratings'],
                                  data['response']):
        try:
            a = np.asarray(list(a_raw), dtype=float).ravel()
            b = np.asarray(list(b_raw), dtype=float).ravel()
        except Exception:
            continue
        if a.shape[0] != n or b.shape[0] != n:
            continue
        d = np.sign(a - b)
        nA = int(np.sum(d > 0))
        nB = int(np.sum(d < 0))
        D = nA + nB
        # dense / mid displays only (D >= 4), with a strict count winner
        if D < 4 or nA == nB:
            continue
        winner = 0 if nA > nB else 1
        if winner == 0:
            win_idx = set(int(i) for i in np.where(d > 0)[0])
            lose_idx = set(int(i) for i in np.where(d < 0)[0])
        else:
            win_idx = set(int(i) for i in np.where(d < 0)[0])
            lose_idx = set(int(i) for i in np.where(d > 0)[0])
        # keep only STAR-BACKED majorities: the count winner's wins are all
        # high-validity (star) experts, the count loser's wins are all
        # low-validity (marginal) experts.
        if not win_idx.issubset(star):
            continue
        if len(lose_idx & star) > 0:
            continue
        try:
            r = int(resp)
        except Exception:
            continue
        hits.append(1.0 if r == winner else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.4850 (var=0.0085)
**Predicted under pi_10:** 0.4850 (var=0.0089)
**Predicted under pi_11:** 0.4931 (var=0.0095)

### Experiment 11
**Design**
  A=[1, 0, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[1, 1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 1, 1]  B=[1, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1]  B=[1, 1, 0, 0, 1, 1]
  A=[1, 0, 1, 1, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[1, 0, 1, 1, 0, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 1, 1, 0, 0]
  A=[1, 1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 1]  B=[1, 1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Design validities (fixed by the experiment): [f0..f5]
    val = np.array([0.95, 0.78, 0.82, 0.55, 0.76, 0.80], dtype=float)
    n_feat = val.shape[0]
    order = np.argsort(-val, kind='stable')
    marginal_idx = [int(j) for j in order[n_feat // 2:]]  # bottom-half tier: f1, f4, f3

    hits = 0.0
    tot = 0.0

    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()

    for a_raw, b_raw, r in zip(a_col, b_col, r_col):
        try:
            a = np.asarray(list(a_raw), dtype=float).ravel()
            b = np.asarray(list(b_raw), dtype=float).ravel()
        except Exception:
            continue
        if a.shape[0] != n_feat or b.shape[0] != n_feat:
            continue
        d = np.sign(a - b)
        D = int(np.sum(d != 0))
        if D < 2:
            continue  # drop the solo-cue cells (K5/K6), where the two theories nearly agree
        net_marg = float(np.sum(d[marginal_idx]))
        if net_marg == 0.0:
            continue
        pred = 0 if net_marg > 0 else 1  # 0 = chose A, 1 = chose B
        try:
            rr = int(r)
        except Exception:
            continue
        hits += 1.0 if rr == pred else 0.0
        tot += 1.0

    if tot <= 0:
        return float('nan')
    return float(hits / tot)
```

**Observed (real) value:** 0.4050 (var=0.0022)
**Predicted under pi_10:** 0.3747 (var=0.0028)
**Predicted under pi_11:** 0.4028 (var=0.0021)

### Experiment 12
**Design**
  A=[1, 0, 1, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 0, 1, 0]  B=[1, 0, 1, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 1, 0]
  A=[1, 1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1, 0]  B=[1, 1, 0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 1, 1, 1]  B=[1, 0, 0, 1, 1, 0, 1]
  A=[1, 0, 0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Cell C3 (tally-tied 2v2, tier-mismatched): discriminating cues {f1,f2,f5,f6};
    #   one option is backed by the two BOTTOM experts {f2(.55), f5(.52)},
    #   the other by the two STARS {f1(.95), f6(.90)}.
    # Cell C4 (mid-pair vs floor): discriminating cues {f0,f4,f5};
    #   one option is backed by the mid pair {f0(.74), f4(.70)},
    #   the other only by the floor expert {f5(.52)}.
    C3_ALL = frozenset([1, 2, 5, 6])
    C3_BOTTOM = frozenset([2, 5])
    C4_ALL = frozenset([0, 4, 5])
    C4_FLOOR = frozenset([5])

    c3_hits = 0
    c3_n = 0
    c4_hits = 0
    c4_n = 0

    for _, row in data.iterrows():
        try:
            a = [int(x) for x in list(row['option_a_ratings'])]
            b = [int(x) for x in list(row['option_b_ratings'])]
            resp = int(row['response'])
        except Exception:
            continue
        if len(a) != len(b) or len(a) < 7:
            continue
        pos = frozenset([i for i in range(len(a)) if a[i] > b[i]])
        neg = frozenset([i for i in range(len(a)) if b[i] > a[i]])
        allset = frozenset(pos | neg)

        if allset == C3_ALL:
            # which side is the bottom-expert-backed one?
            if pos == C3_BOTTOM:
                target = 0  # A
            elif neg == C3_BOTTOM:
                target = 1  # B
            else:
                target = None
            if target is not None:
                c3_n += 1
                if resp == target:
                    c3_hits += 1

        elif allset == C4_ALL:
            # which side is backed only by the floor expert f5?
            if pos == C4_FLOOR:
                target = 0
            elif neg == C4_FLOOR:
                target = 1
            else:
                target = None
            if target is not None:
                c4_n += 1
                if resp == target:
                    c4_hits += 1

    if c3_n == 0 or c4_n == 0:
        return float('nan')

    p_c4_floor = c4_hits / float(c4_n)
    p_c3_bottom = c3_hits / float(c3_n)
    return float(p_c4_floor - p_c3_bottom)

```

**Observed (real) value:** -0.0225 (var=0.0100)
**Predicted under pi_10:** 0.0387 (var=0.0312)
**Predicted under pi_11:** 0.0325 (var=0.0149)

### Experiment 13
**Design**
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 1, 0]
  A=[1, 0, 1, 1, 0, 1]  B=[0, 1, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0, 1]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[1, 0, 1, 0, 0, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[1, 1, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    cells = {}
    for _, row in data.iterrows():
        try:
            a = [int(x) for x in row['option_a_ratings']]
            b = [int(x) for x in row['option_b_ratings']]
        except Exception:
            continue
        if len(a) != len(b) or len(a) == 0:
            continue
        ai = frozenset([i for i in range(len(a)) if a[i] > b[i]])
        bi = frozenset([i for i in range(len(a)) if b[i] > a[i]])
        if len(ai) == len(bi):
            continue
        try:
            resp = int(row['response'])
        except Exception:
            continue
        if len(ai) > len(bi):
            maj, mino = ai, bi
            follow = 1.0 if resp == 0 else 0.0
        else:
            maj, mino = bi, ai
            follow = 1.0 if resp == 1 else 0.0
        key = (maj, mino)
        if key not in cells:
            cells[key] = []
        cells[key].append(follow)

    def rate(maj, mino):
        k = (frozenset(maj), frozenset(mino))
        v = cells.get(k)
        if v is None or len(v) == 0:
            return None
        return float(np.mean(v))

    # cells identified by which expert indices each side wins
    # validities = [0.78, 0.95, 0.68, 0.84, 0.90, 0.72]
    d3 = rate({0, 3}, {1})          # mid-pair majority vs LONE STAR (.95)
    c2 = rate({0, 3}, {2})          # same majority vs LONE FLOOR (.68)
    e4 = rate({1, 2, 4}, {0, 5})    # star+star+floor majority vs mid-pair
    a1 = rate({0, 3, 5}, {1, 2})    # mid-trio majority vs star+floor

    terms = []
    if (d3 is not None) and (c2 is not None):
        terms.append(d3 - c2)
    if e4 is not None:
        terms.append(e4)
    if a1 is not None:
        terms.append(1.0 - a1)

    if len(terms) == 0:
        return float('nan')
    return float(np.mean(terms))
```

**Observed (real) value:** 0.6983 (var=0.0122)
**Predicted under pi_10:** 0.6479 (var=0.0051)
**Predicted under pi_11:** 0.6767 (var=0.0057)

### Experiment 14
**Design**
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 1, 0, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 1, 0, 1]
  A=[1, 0, 1, 1, 0, 1]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[1, 0, 1, 1, 0, 1]
  A=[1, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[1, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[1, 0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    hits = 0
    tot = 0
    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()

    for a_raw, b_raw, r in zip(a_col, b_col, r_col):
        try:
            a = np.asarray([float(x) for x in list(a_raw)])
            b = np.asarray([float(x) for x in list(b_raw)])
        except Exception:
            continue
        if a.size != b.size or a.size < 5:
            continue
        d = np.sign(a - b)
        iA = np.where(d > 0)[0]
        iB = np.where(d < 0)[0]
        try:
            resp = int(r)
        except Exception:
            continue
        # K4-type displays: a 3-cue majority opposed by exactly ONE cue,
        # and that lone opposing cue is the panel's floor expert (index 4, v=0.52).
        if iA.size == 3 and iB.size == 1 and int(iB[0]) == 4:
            tot += 1
            if resp == 0:
                hits += 1
        elif iB.size == 3 and iA.size == 1 and int(iA[0]) == 4:
            tot += 1
            if resp == 1:
                hits += 1

    if tot == 0:
        return 0.5
    return float(hits) / float(tot)

```

**Observed (real) value:** 0.1450 (var=0.0099)
**Predicted under pi_10:** 0.1225 (var=0.0080)
**Predicted under pi_11:** 0.1800 (var=0.0078)

### Experiment 15
**Design**
  A=[1, 1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[1, 1, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[0, 1, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 1, 1, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 1, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 1, 0]  B=[1, 0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # stated validities of the six experts in screen order (design-time constant)
    V = np.array([0.88, 0.55, 0.72, 0.95, 0.63, 0.80], dtype=float)
    THR = 0.70  # 'upper panel' voices: 0.72, 0.80, 0.88, 0.95

    hits = 0
    tot = 0

    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()

    for a_raw, b_raw, r_raw in zip(a_col, b_col, r_col):
        try:
            a = np.asarray([float(x) for x in a_raw], dtype=float)
            b = np.asarray([float(x) for x in b_raw], dtype=float)
            r = int(r_raw)
        except Exception:
            continue
        if a.size == 0 or a.size != b.size:
            continue
        if a.size == V.size:
            v = V
        else:
            v = np.full(a.size, 1.0, dtype=float)

        d = np.sign(a - b)
        idx = np.nonzero(d)[0]
        if idx.size == 0:
            continue
        nA = int(np.sum(d > 0))
        nB = int(np.sum(d < 0))
        if nA == nB:
            continue  # no strict count winner: both theories predict chance
        # keep only displays where EVERY discriminating voice is an upper-panel expert
        if np.any(v[idx] < THR):
            continue

        winner = 0 if nA > nB else 1  # 0 == chose A, 1 == chose B
        tot += 1
        if r == winner:
            hits += 1

    if tot == 0:
        return 0.5
    return float(hits) / float(tot)
```

**Observed (real) value:** 0.4892 (var=0.0049)
**Predicted under pi_10:** 0.5713 (var=0.0051)
**Predicted under pi_11:** 0.5038 (var=0.0056)

### Experiment 16
**Design**
  A=[1, 0, 1, 1, 0, 1]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 1]
  A=[1, 1, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[1, 1, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 1]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Target cells: displays where the TALLY-MAJORITY side is carried by the two
    # most valid experts (features 0 = .92 and 2 = .84).
    # C1: majority wins {0,2}, minority wins {}        (unopposed star pair, 2v0)
    # C3: majority wins {0,2}, minority wins {5}       (star pair vs lone mid, 2v1)
    # C4: majority wins {0,2}, minority wins {1}       (star pair vs lone low, 2v1)
    # C7: majority wins {0,2,4}, minority wins {5}     (all-upper 3v1)
    targets = {
        (frozenset([0, 2]), frozenset()),
        (frozenset([0, 2]), frozenset([5])),
        (frozenset([0, 2]), frozenset([1])),
        (frozenset([0, 2, 4]), frozenset([5])),
    }

    hits = []
    for _, row in data.iterrows():
        try:
            a = [int(x) for x in row['option_a_ratings']]
            b = [int(x) for x in row['option_b_ratings']]
        except Exception:
            continue
        if len(a) != len(b) or len(a) == 0:
            continue
        idxA = frozenset([i for i in range(len(a)) if a[i] > b[i]])
        idxB = frozenset([i for i in range(len(a)) if b[i] > a[i]])
        if len(idxA) == len(idxB):
            continue
        if len(idxA) > len(idxB):
            maj_set, min_set, maj_resp = idxA, idxB, 0  # A is count winner
        else:
            maj_set, min_set, maj_resp = idxB, idxA, 1  # B is count winner
        if (maj_set, min_set) not in targets:
            continue
        try:
            r = int(row['response'])
        except Exception:
            continue
        hits.append(1.0 if r == maj_resp else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))

```

**Observed (real) value:** 0.4892 (var=0.0049)
**Predicted under pi_10:** 0.5083 (var=0.0056)
**Predicted under pi_11:** 0.4942 (var=0.0053)

### Experiment 17
**Design**
  A=[1, 1, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1]
  A=[0, 1, 1, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1, 1, 0]
  A=[1, 0, 1, 1, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0, 1]
  A=[0, 0, 1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0, 1, 1]
  A=[1, 0, 1, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 1, 1, 0]  B=[1, 0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 0, 1, 1]  B=[0, 0, 1, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Canonical (order-free) keys for the four diagnostic cells.
    # SOLO cells: exactly one discriminating expert, and it is NOT the panel floor
    #   C1 = solo .95 star, C2 = solo .75 mid
    SOLO = {
        ("1001010", "1101010"),   # C1 (trials 1/2)
        ("0010101", "1010101"),   # C2 (trials 3/4)
    }
    # FLAT cells: CH >= 2 displays whose every discriminating voice is upper-band
    #   C4 = unopposed .79/.75 pair (2 vs 0), C5 = low trio vs lone .95 (3 vs 1)
    FLAT = {
        ("0011001", "1011011"),   # C4 (trials 7/8)
        ("0101000", "1011010"),   # C5 (trials 9/10)
    }

    solo_hits = 0.0
    solo_n = 0.0
    flat_hits = 0.0
    flat_n = 0.0

    for _, row in data.iterrows():
        try:
            a = [int(x) for x in list(row['option_a_ratings'])]
            b = [int(x) for x in list(row['option_b_ratings'])]
        except Exception:
            continue
        if len(a) != len(b) or len(a) == 0:
            continue
        ka = ''.join([str(x) for x in a])
        kb = ''.join([str(x) for x in b])
        key = tuple(sorted((ka, kb)))

        if key in SOLO:
            grp = 0
        elif key in FLAT:
            grp = 1
        else:
            continue

        mA = sum(1 for i in range(len(a)) if a[i] > b[i])
        mB = sum(1 for i in range(len(a)) if b[i] > a[i])
        if mA == mB:
            continue
        winner = 0 if mA > mB else 1

        try:
            resp = int(row['response'])
        except Exception:
            continue
        hit = 1.0 if resp == winner else 0.0

        if grp == 0:
            solo_hits += hit
            solo_n += 1.0
        else:
            flat_hits += hit
            flat_n += 1.0

    if solo_n == 0.0 or flat_n == 0.0:
        return float('nan')

    return float(solo_hits / solo_n - flat_hits / flat_n)
```

**Observed (real) value:** 0.0067 (var=0.0123)
**Predicted under pi_10:** -0.0167 (var=0.0167)
**Predicted under pi_11:** 0.0067 (var=0.0177)

### Experiment 18
**Design**
  A=[1, 0, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 0, 1, 0]  B=[1, 0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1]
  A=[0, 1, 1, 0, 1, 0, 1]  B=[0, 1, 1, 1, 1, 0, 1]
  A=[1, 1, 0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 1, 0]
  A=[1, 0, 1, 1, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 1, 1, 0]  B=[0, 0, 1, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 0, 1]
  A=[1, 1, 0, 1, 0, 0, 1]  B=[1, 1, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 1, 0, 1]  B=[0, 1, 0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    # cell accumulators: hits / total
    hits = {}
    tot = {}

    for _, row in data.iterrows():
        try:
            a = [int(x) for x in row['option_a_ratings']]
            b = [int(x) for x in row['option_b_ratings']]
        except Exception:
            continue
        if len(a) != len(b) or len(a) == 0:
            continue
        winsA = [i for i in range(len(a)) if a[i] > b[i]]
        winsB = [i for i in range(len(a)) if b[i] > a[i]]
        mA, mB = len(winsA), len(winsB)
        if mA == mB:
            continue
        if mA > mB:
            win_idx, lose_idx, winner_is_A = set(winsA), set(winsB), True
        else:
            win_idx, lose_idx, winner_is_A = set(winsB), set(winsA), False
        D = mA + mB

        cell = None
        if D == 1:
            j = list(win_idx)[0]
            if j == 2:
                cell = 'C1'      # solo star (v=.95)
            elif j == 3:
                cell = 'C2'      # solo low-upper (v=.78)
        elif D == 2 and len(lose_idx) == 0:
            if win_idx == {0, 3}:
                cell = 'C4'      # unopposed low pair
            elif win_idx == {2, 4}:
                cell = 'C5'      # unopposed senior pair
        elif D == 4 and win_idx == {0, 3, 5} and lose_idx == {2}:
            cell = 'C6'          # all-upper 3v1

        if cell is None:
            continue

        try:
            resp = int(row['response'])
        except Exception:
            continue
        chose_A = (resp == 0)
        follow = 1.0 if (chose_A == winner_is_A) else 0.0

        hits[cell] = hits.get(cell, 0.0) + follow
        tot[cell] = tot.get(cell, 0.0) + 1.0

    def rate(c):
        if tot.get(c, 0.0) <= 0:
            return None
        return hits[c] / tot[c]

    g1 = [r for r in (rate('C1'), rate('C2')) if r is not None]
    g2 = [r for r in (rate('C4'), rate('C5'), rate('C6')) if r is not None]

    if len(g1) == 0 or len(g2) == 0:
        return 0.0
    return float(np.mean(g1) - np.mean(g2))
```

**Observed (real) value:** 0.0517 (var=0.0159)
**Predicted under pi_10:** 0.0369 (var=0.0127)
**Predicted under pi_11:** -0.0250 (var=0.0228)

### Experiment 19
**Design**
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 0, 1]  B=[0, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[0, 0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 0, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 1]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[0, 1, 1, 1, 0, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 0, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    FLOOR = 2          # validity 0.54, unique panel floor
    # cells in which the floor voice sides with a set F against opposition O
    POS = {
        (frozenset({2}), frozenset({1, 4})),        # C2: floor vs STAR-PAIR
        (frozenset({1, 2, 4}), frozenset({0})),     # C4: floor+2 stars vs lone mid
        (frozenset({2, 3, 5}), frozenset({1})),     # C5: floor+2 mids vs lone star
    }
    NEG = {
        (frozenset({1, 2, 4}), frozenset()),        # C7: unanimous floor+star+star
    }

    pos_hits = 0
    pos_n = 0
    neg_hits = 0
    neg_n = 0

    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()

    for a_raw, b_raw, resp in zip(a_col, b_col, r_col):
        try:
            a = np.asarray(list(a_raw), dtype=float).ravel()
            b = np.asarray(list(b_raw), dtype=float).ravel()
        except Exception:
            continue
        if a.size != b.size or a.size == 0:
            continue
        winA = frozenset(int(i) for i in np.where(a > b)[0])
        winB = frozenset(int(i) for i in np.where(b > a)[0])
        if FLOOR in winA:
            F, O = winA, winB
            follow = 1 if int(resp) == 0 else 0
        elif FLOOR in winB:
            F, O = winB, winA
            follow = 1 if int(resp) == 1 else 0
        else:
            continue
        key = (F, O)
        if key in POS:
            pos_n += 1
            pos_hits += follow
        elif key in NEG:
            neg_n += 1
            neg_hits += follow

    if pos_n == 0 or neg_n == 0:
        return float('nan')

    return float(pos_hits) / pos_n - float(neg_hits) / neg_n

```

**Observed (real) value:** -0.0167 (var=0.0075)
**Predicted under pi_10:** 0.1994 (var=0.0261)
**Predicted under pi_11:** -0.0161 (var=0.0115)

### Experiment 20
**Design**
  A=[1, 1, 0, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1, 0, 1]
  A=[1, 0, 0, 1, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 1, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 1, 0]  B=[1, 0, 1, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 1, 1]
  A=[0, 0, 1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    a_list = data['option_a_ratings'].tolist()
    b_list = data['option_b_ratings'].tolist()
    resp = data['response'].tolist()

    titration = []   # cells C3 ({1} vs {0,4}) and C4 ({1} vs {2,4})
    unanimous = []   # cell C6 ({1,2,4} unanimous bloc)

    target_others = (frozenset({0, 4}), frozenset({2, 4}))
    floor_only = frozenset({1})
    bloc = frozenset({1, 2, 4})

    for a, b, r in zip(a_list, b_list, resp):
        try:
            av = [int(x) for x in a]
            bv = [int(x) for x in b]
        except Exception:
            continue
        n = len(av)
        if n != len(bv) or n != 7:
            continue
        setA = frozenset([i for i in range(n) if av[i] > bv[i]])
        setB = frozenset([i for i in range(n) if bv[i] > av[i]])
        if len(setA) == len(setB):
            continue
        if 1 in setA:
            focal, other, focal_is_a = setA, setB, True
        elif 1 in setB:
            focal, other, focal_is_a = setB, setA, False
        else:
            continue
        try:
            chose_a = (int(r) == 0)
        except Exception:
            continue
        chose_focal = 1.0 if (chose_a == focal_is_a) else 0.0

        if focal == floor_only and other in target_others:
            titration.append(chose_focal)
        elif focal == bloc and len(other) == 0:
            unanimous.append(chose_focal)

    if len(titration) == 0 or len(unanimous) == 0:
        return float('nan')

    return float(np.mean(titration) - np.mean(unanimous))
```

**Observed (real) value:** 0.0150 (var=0.0121)
**Predicted under pi_10:** 0.1575 (var=0.0164)
**Predicted under pi_11:** 0.0158 (var=0.0178)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all 20 experiments the evidence is one-sided on the two structural claims that separate these theories.\n\n(1) TALLY TIES. pi_10 (CTN-BG) makes the parameter-free claim that equal win-counts => exactly 0.500. Experiment 1 (metric 0.842 observed vs pi_10's 0.181, pi_11's 0.803) and Experiment 2 (0.870 observed vs 0.495 / 0.833) both show strong ANTI-validity choice on tally-tied displays containing the panel floor voice. pi_10's direction-gate is decisively falsified; pi_11's confidence-only gate is correct in kind, and if anything slightly under-predicts the size of the effect (.80/.83 vs .84/.87).\n\n(2) CHORUS AMPLIFICATION. pi_10's core claim is that a floor voice is loud only in proportion to the chorus it faces. The two experiments designed as direct tests of this (Exp 19: +0.199 predicted vs -0.0167 observed; Exp 20: +0.158 predicted vs +0.015 observed) come out as exact nulls, which is pi_11's parameter-free prediction (-0.016, +0.016). The titration component built into Experiment 1's metric tells the same story. Chorus amplification does not exist in these data.\n\n(3) EVERYTHING ELSE. pi_11 is at least as good as pi_10 on almost every remaining design: Exp 6 (.755 obs; .741 vs .699), Exp 8 (-.938; -1.034 vs -.758), Exp 11 (.405; .403 vs .375), Exp 13 (.698; .677 vs .648), Exp 15 (.489; .504 vs .571 - pi_10 wrongly lifts all-upper majorities above chance), Exp 16 (.489; .494 vs .508), Exp 17 (.0067; .0067 vs -.017). pi_10 is nominally better only on Exp 9, 14, 18 and marginally on Exp 7, all by small margins.\n\nThe overall empirical picture that emerges is very clean: subjects follow the option backed by the panel's LOWEST-validity discriminating expert, panel-relatively defined (Exp 13, where the panel floor is a fairly high .68, still shows strong floor-following, ruling out any absolute-validity-threshold account), at a remarkably CONSTANT rate (~.84-.87 on ties, ~.85 against pairs, ~.86 against trios, ~.85 in unanimous blocs, ~.855 reversal of 3-cue majorities in Exp 14), and otherwise responding at chance whenever the genuine floor voice does not discriminate (Exp 3 ties ~.47, Exp 10 star-backed dense majorities ~.485, Exp 15/16 all-upper displays ~.489). Unweighted tally per se carries essentially no independent pull.\n\npi_11 captures this pattern well but not perfectly: its Luce normalisation, D^gamma dilution and rho_tie gate make ties and dense displays less extreme than they actually are (under-shooting Exp 1/2), and its continuous panel-relative weight gives the second-from-bottom expert real loudness (weight ~.2-.26), which over-predicts marginal-tier following in Exp 9 (.749 vs .661) and would over-predict Exp 10 (observed exactly chance although .60/.65 experts discriminated on the losing side). pi_10 is degenerate: its two signature commitments are both falsified, and its thicker residual gradient additionally mis-predicts all-upper displays (Exp 15).",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace pi_10 entirely. The replacement should be the strongest genuine rival to FM-ISN-CC5 that the data actually suggest: a NON-COMPENSATORY, SINGLE-CUE 'Take-the-Worst' mixture rather than a weighted-balance integrator.\n\nSketch \u2014 **Floor-Voice Lexicographic Rule (TTW-Mixture)**:\n(1) On each trial the subject looks for ONE expert only: the panel's minimum-validity voice (panel-relative, identified from the instructions; Exp 13 shows the floor is defined relative to the displayed panel, not by an absolute closeness-to-.5 criterion).\n(2) If that floor expert discriminates, the subject chooses its side with a single fixed probability p_f (~0.82-0.90 before lapse). Crucially this rate is COUNT-INVARIANT and CONTEXT-INVARIANT: it does not change with the number of opposing cues, their seniority, the total number of discriminating attributes D, whether the counts are tied, or whether the floor sits inside a unanimous bloc. This is a flat, parameter-free-in-shape prediction and is the sharp contrast with pi_11, whose Luce denominator (c_n + sum w), D^gamma dilution and rho_tie confidence gate all make these cells differ (pi_11 systematically under-shoots ties at .80/.83 where the data give .84/.87, and predicts solo > multi-cue displays).\n(3) If the floor expert does NOT discriminate, the rule fails and the subject essentially guesses: near-exact chance, with at most a whisper of a fallback (a small probability delta of consulting the SECOND-lowest expert, delta <= 0.15, and no independent tally pull). This is the second sharp contrast with pi_11, whose continuous (1-s)^kappa weight hands the second-from-bottom voice ~0.2-0.26 of full loudness and therefore over-predicts Exp 9 (.749 vs .661) and would predict clear anti-validity in Exp 10 where the data are exactly chance (.485) and in Exp 3's ties (.474).\n(4) A small lapse/side-noise term epsilon (~0.03-0.10) mixed with a coin flip accounts for residual scatter; no capacity term is needed (Exp 6 with 8 experts behaves like the 6-7 expert panels).\n\nThe predict() implementation should therefore be: locate argmin(validities); if it discriminates return [p_f, 1-p_f] oriented to its side (identical value for every D, every margin, every tie status, every unanimity condition); else, with weight delta orient to the second-lowest discriminating expert and otherwise return chance; finally mix with epsilon. Parameter ranges: p_f in [0.80, 0.92], delta in [0.0, 0.15], epsilon in [0.02, 0.12]. This theory is clearly distinguishable from pi_11 by: (a) tie-vs-majority and solo-vs-dense equality (pi_11 predicts gradients), (b) second-from-bottom cells (pi_11 predicts moderate anti-validity, TTW predicts chance), and (c) count-margin invariance of floor-following."
}
```

## Usage

```json
{
  "input_tokens": 47436,
  "output_tokens": 13071
}
```
