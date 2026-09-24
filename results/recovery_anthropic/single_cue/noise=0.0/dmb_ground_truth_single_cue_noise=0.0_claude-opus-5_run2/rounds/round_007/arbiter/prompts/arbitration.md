# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_8" and "pi_9") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_8" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_9" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_8
**Consensus-Graded Validity Inversion with a Non-Negative News Floor (CG-VI-2).**

People told the stated accuracies of a panel of binary expert ratings do not weight experts *by* accuracy; they weight them by how much *news* a rating carries. Six commitments.

**(1) The count comparator is a gate, not a weight.** Before any expert identity is consulted, an unweighted feature-wise comparison asks a purely ordinal question: does one product win on strictly more attributes? If not — equal win counts, including wholly non-discriminating displays — the display is construed as non-diagnostic and the subject flips a mental coin: exactly 0.5, with NO validity tie-break and NO take-the-best fallback. Parameter-free at every panel composition (Exp 1, Exp 3's tied cells, Exp 7's tied cells, Exp 10's C3 cell).

**(2) When a count winner exists, weights are inverse in stated validity, measured against the panel's least-accurate voice and strongly SATURATING.** s_j = (v_j - v_min)/(v_max - v_min); news value w_j = 1 - lambda_eff * s_j^omega with omega well below 1 (~0.3-0.4). Because the transform is strongly concave, *any* expert appreciably above the panel floor is already heard as "one of the accurate ones" — merely restating the obvious consensus quality of the product — while the panel's genuinely idiosyncratic bottom voice is heard as private, surprising information and keeps full credit.

**(3) THE NEW COMMITMENT — news value is bounded below at ZERO: a redundant expert is IGNORED, never counted against his own side.** w_j = max(0, 1 - lambda_eff*s_j^omega). Redundancy is a reason to stop listening, not a reason to believe the opposite. This is what generates the signature pattern that no signed-inversion model produces: *the side holding the panel's least-accurate voice wins, and when neither side holds it the plain tally decides*. A majority made of stars plus the maverick is followed near ceiling (the stars are mute, the maverick speaks for the majority), whereas the numerically identical majority made of mid-panel experts is rejected when the maverick sits in the minority. It also stops dense, unanimous, high-validity displays from being actively rejected (they merely lose force and drift toward chance), which preserves plain-tally adherence.

**(4) The redundancy construal is graded by how big and how lopsided the visible consensus is, and it saturates sharply.** The subjective chorus is read off the display as CH = max(m_A, m_B) + |m_A - m_B| - 2, i.e. the size of the larger bloc *plus* the net excess of same-direction voices. lambda_eff = lambda * CH^q/(CH^q + kappa) with q ~ 3.4-4.5 and kappa ~ 1.4-2.0. Hence: a solitary discriminating voice (CH = 0) is heard at full face value; a 2-vs-1 display (CH = 1) is only weakly discounted, so a mid pair opposed by a lone floor voice lands essentially at chance; a 3-vs-2, 2-vs-0 or 3-vs-1 display (CH >= 2) is discounted almost maximally, so the floor voice takes over; and everything denser is saturated.

**(5) Confidence is compressed by the sheer number of discriminating cues (D^gamma) and capped by an attention lapse epsilon.** Evidence spread across many mutually contradicting attributes is subjectively weaker than the same net weight carried by a sparse clean split.

**(6) The inversion needs the whole validity ordering in mind, so it degrades past working-memory span (~7 voices).** For n > 7 the discount shrinks (lambda scaled by 1 - psi*(n-7)) and integration gets noisier (divisor multiplied by 1 + chi*(n-7)), so eight-expert star-backed majorities sit at chance rather than being reversed, while plain tally adherence there is preserved.

**Cross-design signature:** exact chance on tally ties at any validity composition; the panel's least-accurate discriminating voice decides whenever it is present on exactly one side, at every panel size up to seven; near-chance for 2-vs-1 mid-versus-floor displays but decisive rejection of 3-cue majorities opposed by the same floor voice; near-ceiling following of majorities that themselves contain the floor voice; lone voices taken at face value; unanimous all-star displays drifting to chance rather than reversing; and markedly attenuated inversion on eight-expert panels.

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
            arr = np.asarray(state, dtype=float)
            arr = arr.reshape(2, -1)
            a = np.asarray(arr[0], dtype=float).ravel()
            b = np.asarray(arr[1], dtype=float).ravel()
        except Exception:
            return np.array([0.5, 0.5], dtype=float)

    if a is None or b is None or a.size == 0 or a.size != b.size:
        return np.array([0.5, 0.5], dtype=float)

    n = int(a.size)

    # ------------- Commitment 1: parameter-free count gate -------------
    d = np.sign(a - b)
    iA = np.where(d > 0)[0]
    iB = np.where(d < 0)[0]
    mA = int(iA.size)
    mB = int(iB.size)
    if mA == mB:                       # includes wholly non-discriminating displays
        return np.array([0.5, 0.5], dtype=float)

    D = float(mA + mB)
    M = float(abs(mA - mB))            # net consensus margin
    c_side = float(max(mA, mB))        # size of the larger bloc
    CH = c_side + M - 2.0              # visible chorus argument
    if (not np.isfinite(CH)) or CH < 0.0:
        CH = 0.0

    # ---------------- parameters ----------------
    def _g(name, default, lo, hi):
        try:
            x = float(parameters.get(name, default))
        except Exception:
            x = float(default)
        if not np.isfinite(x):
            x = float(default)
        return float(min(max(x, lo), hi))

    lam = _g('lam', 1.75, 0.0, 4.0)
    omega = _g('omega', 0.34, 0.10, 1.50)
    q_m = _g('q_m', 3.95, 0.50, 8.00)
    kappa_m = _g('kappa_m', 1.70, 0.05, 8.00)
    beta = _g('beta', 11.0, 0.0, 40.0)
    gamma = _g('gamma', 0.90, 0.0, 2.5)
    psi = _g('psi', 0.50, 0.0, 1.0)
    chi = _g('chi', 0.85, 0.0, 4.0)
    eps = _g('epsilon', 0.12, 0.0, 0.6)

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

    # ------------- Commitment 2: saturating inverse-validity news value -------------
    v_min = float(np.min(v))
    v_max = float(np.max(v))
    span = v_max - v_min
    if (not np.isfinite(span)) or span <= 1e-9:
        s = np.zeros(n, dtype=float)
    else:
        s = np.clip((v - v_min) / span, 0.0, 1.0)
    f = np.power(np.clip(s, 0.0, 1.0), omega)

    # ------------- Commitment 6: capacity-limited construal (n > 7) -------------
    over = max(0.0, float(n) - 7.0)
    lam_n = lam * max(0.0, 1.0 - psi * over)
    if (not np.isfinite(lam_n)) or lam_n < 0.0:
        lam_n = 0.0

    # ------------- Commitment 4: chorus(size+margin)-graded redundancy -------------
    try:
        CHq = float(CH) ** q_m
    except Exception:
        CHq = float(CH)
    if (not np.isfinite(CHq)) or CHq < 0.0:
        CHq = float(CH)
    lam_eff = lam_n * CHq / (CHq + kappa_m)
    if (not np.isfinite(lam_eff)) or lam_eff < 0.0:
        lam_eff = 0.0

    # ------------- Commitment 3: news value is floored at ZERO -------------
    w = np.maximum(0.0, 1.0 - lam_eff * f)

    E = float(np.sum(w[iA]) - np.sum(w[iB]))
    if not np.isfinite(E):
        E = 0.0

    # ------------- Commitment 5: dilution by cue count, plus overload noise -------------
    den = float(D) ** gamma
    if (not np.isfinite(den)) or den <= 0.0:
        den = 1.0
    den = den * (1.0 + chi * over)
    if (not np.isfinite(den)) or den <= 0.0:
        den = 1.0

    z = beta * E / den
    if not np.isfinite(z):
        z = 0.0
    z = float(min(max(z, -30.0), 30.0))

    p_a = 1.0 / (1.0 + np.exp(-z))

    # attention lapse
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

## THEORY 2 — pi_9
**Mute-Band Maverick Arbitration with a HARD (near-step) Band and a Stronger Capacity Noise Term (MBM-PG7).**

People who are told the accuracies of a panel of binary expert ratings do not weight experts *by* accuracy; they weight them by how much *news* a rating carries, and news is a steeply thresholded, panel-relative quantity. Six commitments.

**(1) The count comparator is a parameter-free gate.** Before any expert identity is consulted, an unweighted feature-wise comparison asks a purely ordinal question: does one product win on strictly more attributes? If not — equal win counts, including wholly non-discriminating displays — the display is construed as non-diagnostic and the subject flips a mental coin: exactly 0.5, with NO validity tie-break and NO take-the-best fallback, at every panel composition and size.

**(2) HARD, NEAR-STEP MUTE BAND, CHORUS-INDEPENDENT.** Seniority is read off the *displayed panel*: s_j = (v_j − v_min)/(v_max − v_min). Only voices lying within a narrow band theta (≈12-20% of the panel span) above the panel floor are heard as carrying private, surprising information, and *within* that band the listening function is now strongly convex: m_j = max(0, 1 − s_j/theta)^rho with rho ≈ 1.7-2.5. The maverick construal is therefore close to categorical — only the panel's genuinely idiosyncratic BOTTOM voice speaks with full force; a voice that merely sits low-ish (a fifth to a half of the way into the band) is already nearly as mute as a star. Muting is unconditional: a lone star cue is as mute as a star inside a bloc.

**(3) GRADED, STRONGLY ASYMMETRIC CHORUS AMPLIFICATION WITH A LATE, STEEP KNEE — and the post-knee amplification is DEEP.** A maverick's voice is loud only in proportion to the redundancy it is heard against, and redundancy is *asymmetric*: the bloc a cue CONTRADICTS is the chorus proper, while the bloc a cue sits INSIDE echoes it only weakly and therefore counts at a discount lam_own. For cue j: C_j = (m_opp − 1) + lam_own·(m_own − 1) when the cue is opposed, and C_j = (m_own − 1) when the display is unanimous. Amplification G_j = 1 + alpha·C_j^p/(C_j^p + kappa) with a LATE, STEEP knee (p ≈ 6.4, kappa ≈ 205) and a LARGE post-knee asymptote (alpha ≈ 52). Because (2) is now near-categorical, this large amplification is applied essentially only to true floor voices: a partial, mid-band voice can no longer borrow a 20-30x chorus gain and masquerade as a maverick.

**(4) THE PANEL GRADIENT: the mute band is quiet, not empty, and the listening floor is THIN.** Muted experts do not vanish; each discriminating cue contributes a residual r_j = eta + c·(1 − s_j) — a thin flat listening floor plus an inverse-seniority gradient. Displays in which no voice is inside the mute band hover near chance at any count margin, while the preserved slope still generates the anti-validity contrasts on conflict displays.

**(5) DILUTION AND LAPSE.** Net evidence E = Σ_j dir_j·(G_j·m_j + r_j) is compressed by the number of mutually contradicting cues, D^gamma, and mixed with an attention lapse epsilon.

**(6) CAPACITY IS BOTH A FLATTENING AND A SUBSTANTIAL NOISE SOURCE.** Holding the whole seniority ordering in mind is possible only up to about seven voices. For n > 7 the amplification collapses toward 1 AND integration becomes markedly noisier (divisor × (1 + chi·(n−7)) with chi now appreciably larger), so eight-expert star-backed majorities sit at chance and eight-expert plain tally adherence is systematically LOWER than the six-expert equivalent even though the tally still governs direction.

**Cross-design signature:** exact chance on tally ties at any composition; chance on displays whose every discriminating voice is upper-band, at ANY count margin; only the panel's bottom voice (not the second- or third-from-bottom) can reverse a majority; a lone floor voice opposed by a mere PAIR lands at chance while the same voice opposed by a three-cue bloc reverses it to ≈.15; near-ceiling following of dense unanimous displays and of majorities containing the floor voice; and visibly attenuated, noisy integration on eight-expert panels.

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

    theta = _g('theta', 0.16, 0.02, 0.90)
    rho = _g('rho', 2.1, 0.20, 4.00)
    eta = _g('eta', 0.23, 0.00, 1.00)
    c_res = _g('c_res', 2.0, 0.00, 6.00)
    beta = _g('beta', 0.80, 0.00, 5.00)
    gamma = _g('gamma', 0.95, 0.00, 2.50)
    alpha = _g('alpha', 52.0, 0.00, 400.0)
    kappa = _g('kappa', 205.0, 0.05, 2000.0)
    p_ch = _g('p_ch', 6.4, 0.50, 12.0)
    lam_own = _g('lam_own', 0.36, 0.00, 1.00)
    psi = _g('psi', 0.22, 0.00, 1.00)
    chi = _g('chi', 0.39, 0.00, 3.00)
    eps = _g('epsilon', 0.09, 0.00, 0.60)

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

    # ---------- Commitment 2: hard, near-step, chorus-independent mute band ----------
    band = np.clip(1.0 - s / max(theta, 1e-6), 0.0, 1.0)
    mav = np.power(band, rho)                      # maverick strength, 0 outside the band

    # ---------- Commitment 3: GRADED chorus amplification (cue-specific) ----------
    def _G(base):
        base = float(max(0.0, base))
        try:
            bp = base ** p_ch
        except Exception:
            bp = base
        if (not np.isfinite(bp)) or bp < 0.0:
            bp = 0.0
        g = 1.0 + alpha * bp / (bp + kappa)
        if (not np.isfinite(g)) or g < 1.0:
            g = 1.0
        return g

    # the bloc a cue OPPOSES counts fully; the bloc it sits INSIDE counts at
    # weight lam_own; an unopposed cue hears only the consensus it belongs to.
    if mB > 0:
        base_A = (float(mB) - 1.0) + lam_own * (float(mA) - 1.0)
    else:
        base_A = (float(mA) - 1.0)
    if mA > 0:
        base_B = (float(mA) - 1.0) + lam_own * (float(mB) - 1.0)
    else:
        base_B = (float(mB) - 1.0)

    G_A = _G(base_A)
    G_B = _G(base_B)

    # ---------- Commitment 6: capacity limit past ~7 voices ----------
    over = max(0.0, float(n) - 7.0)
    if over > 0.0:
        shrink = max(0.0, 1.0 - psi * over)
        G_A = 1.0 + (G_A - 1.0) * shrink
        G_B = 1.0 + (G_B - 1.0) * shrink
    noise = 1.0 + chi * over
    if (not np.isfinite(noise)) or noise <= 0.0:
        noise = 1.0

    # ---------- Commitment 4: the panel gradient (quiet, not empty) ----------
    resid = eta + c_res * (1.0 - s)

    w = np.array(resid, dtype=float, copy=True)
    if iA.size > 0:
        w[iA] = G_A * mav[iA] + resid[iA]
    if iB.size > 0:
        w[iB] = G_B * mav[iB] + resid[iB]

    E = float(np.sum(w[iA]) - np.sum(w[iB]))
    if not np.isfinite(E):
        E = 0.0

    # ---------- Commitment 5: dilution by cue count, then lapse ----------
    den = float(D) ** gamma
    if (not np.isfinite(den)) or den <= 0.0:
        den = 1.0
    den = den * noise

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

## EXPERIMENT 1 (proposed by pi_8)

### DESIGN
**Validities (n_features=7):** [0.75, 0.95, 0.71, 0.89, 0.55, 0.79, 0.92]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0]
  trial 2: A=[1, 0, 0, 1, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1, 0]
  trial 3: A=[1, 0, 1, 0, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0, 1]
  trial 4: A=[0, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1]
  trial 5: A=[0, 1, 1, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1, 0]
  trial 6: A=[0, 1, 1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1, 1, 0]
  trial 7: A=[1, 0, 1, 1, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0, 1]
  trial 8: A=[0, 0, 1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0, 1, 1]
  trial 9: A=[1, 0, 1, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0, 0]
  trial 10: A=[0, 1, 0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0, 1, 0]
  trial 11: A=[0, 1, 0, 1, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0, 0]
  trial 12: A=[0, 0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1, 0, 1]
  trial 13: A=[1, 0, 1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1, 1, 0]
  trial 14: A=[0, 1, 0, 0, 1, 1, 0]  B=[1, 0, 1, 0, 1, 1, 0]
  trial 15: A=[0, 1, 0, 1, 0, 1, 1]  B=[0, 0, 1, 1, 0, 1, 0]
  trial 16: A=[0, 0, 1, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 1]

**Rationale:** GOAL: dissociate CG-VI-2 (advocated) from MBM-PG7 (competing). The two models share almost everything: a parameter-free count gate (tally ties -> exactly 0.5), panel-relative seniority s=(v-vmin)/(vmax-vmin), inverse-validity 'news value', dilution by D^gamma, a lapse, and a capacity term past 7 voices. They also AGREE on the classic maverick reversals (a lone floor voice opposed by a 3-cue bloc reverses it in both models), so those cells are wasted. They differ on exactly three things, and every cell below turns one of them into a parameter-free equality, a large magnitude gap, or a sign flip.

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

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
The two models disagree most sharply on two structurally opposite kinds of display, and in OPPOSITE directions, so a difference score amplifies the contrast while cancelling generic per-subject response noise.

(a) SOLO cells (C1 solo .95 star, C2 solo .75 mid; 24 trials/subject). CG-VI-2's chorus argument is CH = max(m)+|\u0394m|-2, so a lone discriminating voice gives CH=0, lam_eff=0, every weight exactly 1, z = beta (9.5-12.5): near-ceiling adherence ~0.94 regardless of the lone expert's validity. MBM-PG7 mutes any voice outside its narrow band unconditionally, so a lone star contributes only the thin residual eta -> ~0.54, and a lone mid ~0.71; mean ~0.63.

(b) FLAT cells (C4 unopposed .79/.75 pair, C5 low trio vs lone .95; 24 trials/subject). Here CH>=2 and every discriminating voice clips to exactly zero news value at every corner of CG's parameter box, so CG returns EXACTLY 0.500 even on a weakly dominant display and on a +2 tally margin. MBM keeps an inverse-seniority residual for muted voices and therefore follows the tally at ~0.70 / ~0.66.

Hence the composite (SOLO adherence - FLAT adherence) is predicted to be ~ +0.44 under CG-VI-2 but ~ -0.05 under MBM-PG7 (a near-zero/slightly negative value, since MBM actually follows the tally somewhat MORE in the dense cells than in the solo-star cell). Both terms are tight per subject: CG's solo term sits at ceiling (binomial var ~0.002 over 24 trials, parameter box only [0.91,0.96]) and its flat term is a parameter-free 0.5; MBM's terms are mid-range but its parameter box is narrow in both cells. The predicted between-theory gap (~0.5) is an order of magnitude larger than the expected per-subject SD (~0.11-0.14), giving a large Welch t at N=25. It is not trivially true of CG: any model that simply weights by validity, or that follows the plain tally, or that has graded (non-clipped) weights would produce a small or negative value here, since tally-following models score HIGH on the 2-vs-0 and 3-vs-1 cells.

Source:
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

### RESULTS
- Predicted under pi_8 (simulated): 0.4275 (var=0.0123)
- Predicted under pi_9 (simulated): -0.0583 (var=0.0163)
- Observed on real data: 0.0067 (var=0.0123)

## EXPERIMENT 2 (proposed by pi_9)

### DESIGN
**Validities (n_features=7):** [0.82, 0.53, 0.95, 0.78, 0.91, 0.72, 0.87]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0]
  trial 2: A=[1, 0, 0, 1, 0, 1, 0]  B=[1, 0, 1, 1, 0, 1, 0]
  trial 3: A=[0, 1, 1, 1, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1]
  trial 4: A=[0, 1, 1, 0, 1, 0, 1]  B=[0, 1, 1, 1, 1, 0, 1]
  trial 5: A=[1, 1, 0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1, 1, 0]
  trial 6: A=[1, 0, 0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 1, 0]
  trial 7: A=[1, 0, 1, 1, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0, 1]
  trial 8: A=[0, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 1, 0, 1]
  trial 9: A=[0, 1, 1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0, 1, 0]
  trial 10: A=[0, 1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1, 1, 0]
  trial 11: A=[1, 0, 0, 1, 1, 1, 0]  B=[0, 0, 1, 0, 1, 0, 0]
  trial 12: A=[0, 0, 1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1, 1, 0]
  trial 13: A=[1, 1, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 0, 1]
  trial 14: A=[1, 1, 0, 1, 0, 0, 1]  B=[1, 1, 1, 0, 1, 0, 1]
  trial 15: A=[0, 1, 0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 1]
  trial 16: A=[0, 0, 1, 1, 1, 0, 1]  B=[0, 1, 0, 1, 0, 0, 0]

**Rationale:** GOAL: dissociate MBM-PG7 (advocated) from CG-VI-2 (competing). The two models share a parameter-free count gate (tally ties -> exactly 0.5), panel-relative seniority s=(v-vmin)/(vmax-vmin), an inverse-validity 'news value' logic, dilution by D^gamma, a lapse, and a capacity term past 7 voices. They also AGREE on the classic maverick reversals, so those cells are wasted except as anchors. They differ on three structural points, and every cell below converts one of them into a parameter-free equality, a large magnitude gap, or a sign flip.

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

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
The two models make structurally opposite predictions about WHEN the count winner is followed. CG-VI-2 releases all discounting for a solitary discriminating cue (chorus argument CH=0, so every weight = 1, z = beta), forcing near-ceiling adherence (~.94) on the solo-star and solo-low-upper cells; but for CH>=2 displays in which no panel-floor voice discriminates, its news values clip to exactly zero, giving E=0 and EXACTLY chance (.50) on the unopposed low pair (C4), the unopposed senior pair (C5) and the all-upper 3-vs-1 (C6), regardless of parameters. MBM-PG7 has no solo release (a lone star is still mute, contributing only the thin residual, ~.54; a lone low-upper ~.68) but its residual gradient keeps all-upper multi-cue displays clearly above chance (~.56-.67). The contrast [mean follow-rate on solo cells C1,C2] minus [mean follow-rate on all-upper cells C4,C5,C6] is therefore ~+0.44 under CG-VI-2 and ~-0.01 under MBM-PG7 -- a ~0.45 gap. Each cell contributes 12 trials per subject, the cells involved sit in the tightest parameter regions of both models (CG's C4/C5/C6 are parameter-free exact .50), so per-subject variance is dominated by binomial noise (~0.010-0.016), giving a Welch t of roughly 14 at N=25. Both theories' predictions here follow directly from their stated commitments rather than being trivially true of either.

Source:
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

### RESULTS
- Predicted under pi_8 (simulated): 0.4408 (var=0.0102)
- Predicted under pi_9 (simulated): -0.0381 (var=0.0193)
- Observed on real data: 0.0517 (var=0.0159)

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
**Predicted under pi_8:** 0.5075 (var=0.0083)
**Predicted under pi_9:** 0.5081 (var=0.0086)

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
**Predicted under pi_8:** 0.7425 (var=0.0058)
**Predicted under pi_9:** 0.7087 (var=0.0069)

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
**Predicted under pi_8:** 0.8994 (var=0.0010)
**Predicted under pi_9:** 0.7742 (var=0.0020)

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
**Predicted under pi_8:** 0.8350 (var=0.0023)
**Predicted under pi_9:** 0.7492 (var=0.0014)

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
**Predicted under pi_8:** -0.7063 (var=0.0049)
**Predicted under pi_9:** -0.6542 (var=0.0050)

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
**Predicted under pi_8:** -1.0012 (var=0.0482)
**Predicted under pi_9:** -0.9213 (var=0.0220)

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
**Predicted under pi_8:** 0.6572 (var=0.0028)
**Predicted under pi_9:** 0.6750 (var=0.0035)

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
**Predicted under pi_8:** 0.4850 (var=0.0181)
**Predicted under pi_9:** 0.4931 (var=0.0066)

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
**Predicted under pi_8:** 0.3950 (var=0.0035)
**Predicted under pi_9:** 0.3756 (var=0.0029)

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
**Predicted under pi_8:** -0.0162 (var=0.0372)
**Predicted under pi_9:** 0.0663 (var=0.0326)

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
**Predicted under pi_8:** 0.6917 (var=0.0064)
**Predicted under pi_9:** 0.6533 (var=0.0064)

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
**Predicted under pi_8:** 0.1375 (var=0.0089)
**Predicted under pi_9:** 0.0788 (var=0.0054)

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
**Predicted under pi_8:** 0.6446 (var=0.0042)
**Predicted under pi_9:** 0.5542 (var=0.0056)

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
**Predicted under pi_8:** 0.5913 (var=0.0058)
**Predicted under pi_9:** 0.4929 (var=0.0046)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the whole corpus the two theories share most machinery (parameter-free chance at tally ties, panel-relative seniority, inverse-validity 'news', D^gamma dilution, lapse, capacity term), so the informative evidence lies exactly where they diverge \\u2014 and there pi_8 is decisively falsified.\n\nExperiments 1 and 2 were built as head-to-head tests of pi_8's two signature commitments: (i) the 'solo release' (CH = max+margin\\u22122 = 0 for a lone discriminating cue \\u21d2 every weight = 1 \\u21d2 near-ceiling ~.94 adherence irrespective of that expert's validity), and (ii) hard clipping to EXACTLY .50 on any CH\\u22652 display with no floor voice. pi_8 therefore predicted the composite (solo \\u2212 flat) at +0.43 / +0.44; pi_9 predicted \\u22120.058 / \\u22120.038; the data gave +0.007 and +0.052. pi_8 misses by ~0.40 in both experiments (a ~3.5 SD miss given between-subject var \\u2248 0.012\\u20130.016); pi_9 misses by 0.065 and 0.090. The solo-release commitment is simply wrong: a lone star cue is not followed at ceiling, and it is followed at roughly the same rate as dense all-upper displays.\n\nThe same asymmetry shows up in the older 'all-upper' probes. Exp 15 (every discriminating voice above .70): observed 0.489 (chance); pi_8 0.645, pi_9 0.554. Exp 16 (majorities carried by the two top experts): observed 0.489; pi_8 0.591, pi_9 0.493. pi_8 systematically over-predicts tally adherence whenever the display is mid/high-validity, and its exact-.50 clipping does not generalise because its CH=1 (2-vs-1) regime is left almost undiscounted. pi_8 also over-shoots plain adherence in Exp 5 (.899 vs .794), Exp 6 (.835 vs .755) and Exp 4 (.743 vs .654), where pi_9 is nearly exact (.774, .749, .709).\n\npi_8 does win a cluster of floor-voice conflict cells \\u2014 Exp 7 (\\u2212.706 vs \\u2212.654, obs \\u2212.763), Exp 12 (\\u2212.016 vs +.066, obs \\u2212.023), Exp 13 (.692 vs .653, obs .698), Exp 14 (.138 vs .079, obs .145), Exp 11 (.395 vs .376, obs .405) \\u2014 but these are small margins (\\u22640.07) on metrics where both theories are in the right regime, whereas pi_8's losses on the new, purpose-built discriminating experiments are of order 0.4. Summing signed errors, pi_9 is the better task-invariant account; pi_8 is now clearly the weaker member of the pair.\n\nThe data that any surviving theory must reproduce are: (a) exact chance on tally ties at every composition (Exp 3: .474); (b) near-chance on ANY display whose discriminating voices are all appreciably above the panel floor, at any count margin and any cue count, including solo stars and unopposed pairs (Exp 1, 2, 15, 16); (c) the panel's genuinely lowest voice dominating massively when opposed by a three-cue upper bloc (Exp 14: .855 against the majority; Exp 7's \\u2212.76 inversion) but only reaching chance when opposed by a mere mid PAIR (Exp 12 C4 \\u2248 .478); (d) moderate, not ceiling, plain-tally adherence on mixed panels (.65\\u2013.79) with a decrement at n = 8 (Exp 6 .755, Exp 10 .485); (e) graded, sub-maximal 'marginal-tier' preference (Exp 9 .66) that reverses when the 'marginal' tier is not really low (Exp 11 .405).\n\nSince pi_8's defining commitments (solo release; zero-clipped weights with chorus gate CH = max+margin\\u22122) are exactly what the last two experiments refute, regenerating its predict function under the same prose is not viable \\u2014 the prose itself asserts the falsified pattern. It should be replaced by a genuinely different competitor that can match pi_9 on (a)\\u2013(e) with less machinery, so that future experiments can discriminate mechanism rather than parameter tuning.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace pi_8 with a SMOOTH ANTI-VALIDITY WEIGHTING account that has NO solo release, NO hard mute band and NO chorus amplification \\u2014 call it e.g. 'Convex Novelty Weighting (CNW)'. Core commitments:\n\n1. Count gate unchanged and parameter-free: equal feature-wise win counts (including wholly non-discriminating displays) \\u21d2 exactly 0.5, no validity tie-break, no take-the-best fallback. (Matches Exp 3 = .474.)\n\n2. Each discriminating cue carries weight that is a CONVEX, monotonically DECREASING function of panel-relative validity: s_j = (v_j \\u2212 v_min)/(v_max \\u2212 v_min), w_j = (1 \\u2212 s_j)^kappa with kappa \\u2248 1.5\\u20133.0, plus a very small floor-independent baseline b \\u2248 0.02\\u20130.10. This is the whole 'news' story: the panel's bottom voice has weight \\u22481, a mid voice (s \\u2248 0.5) retains only 2^\\u2212kappa \\u2248 0.12\\u20130.35 of it, and a star (s \\u2248 1) contributes essentially nothing. Weights are strictly non-negative \\u2014 a star is ignored, never counted against its own side.\n\n3. Because star weights vanish, ALL-UPPER displays \\u2014 solo star, solo mid-high, unopposed upper pair, upper 3-vs-1 \\u2014 automatically land near chance at ANY count margin (Exp 1 solo\\u2212flat \\u2248 0, Exp 2 \\u2248 +0.05, Exp 15 = .489, Exp 16 = .489), WITHOUT needing pi_8's solo release or pi_9's step band. Note that this predicts a shallow, monotone but compressed solo ladder, not a flat ceiling line: the model-writer should choose kappa and beta so that solo-star \\u2248 .52\\u2013.58, solo-mid \\u2248 .58\\u2013.68, solo-floor \\u2248 .85\\u2013.92.\n\n4. The floor-voice reversals come out of the WEIGHT ARITHMETIC, not a chorus knee: a lone floor cue (w\\u22481) against a mid PAIR (2 \\u00d7 ~0.4\\u20130.5) nets \\u22480 \\u21d2 chance (Exp 12 C4 = .478), while the same floor cue against a three-cue UPPER bloc (3 \\u00d7 ~0.05\\u20130.15) nets strongly positive \\u21d2 .80\\u2013.90 against the majority (Exp 14 = .145 adherence; Exp 7 = \\u2212.76). Calibrate kappa so that these two cells fall on opposite sides of chance simultaneously \\u2014 this is the single tightest quantitative constraint in the corpus and it replaces pi_9's alpha/kappa/p_ch knee with one exponent.\n\n5. Integration: E = \\u03a3 dir_j \\u00b7 w_j, compressed by the number of mutually contradicting cues (E / D^gamma, gamma \\u2248 0.6\\u20131.1), passed through a logistic with moderate beta (chosen so that plain mixed-panel adherence lands at .70\\u2013.80, NOT .85\\u2013.90 \\u2014 pi_8's persistent over-shoot on Exp 4/5/6 must not be repeated), plus an attention lapse epsilon \\u2248 0.08\\u20130.15.\n\n6. Capacity: for n > 7 the anti-validity ordering degrades (kappa shrinks toward 1, i.e. weights flatten) AND integration noise rises (divisor \\u00d7 (1 + chi\\u00b7(n\\u22127))), so eight-expert star-backed majorities sit at chance (Exp 10 = .485) and eight-expert tally adherence (.755) is below the six-expert value (.794).\n\nDistinctive, testable contrast with pi_9: CNW is CONTINUOUS in seniority and CHORUS-FREE, so (i) the second- and third-from-bottom experts retain appreciable, graded influence (pi_9 mutes them near-categorically), and (ii) the strength of a floor voice depends only on the summed seniority of what opposes it, not on the COUNT of opposing cues \\u2014 so a floor voice opposed by two STARS should already reverse under CNW (stars \\u2248 0 weight) while pi_9 needs a three-cue bloc to cross its late knee. Future experiments should target exactly those two cells: (a) a lone second-from-bottom voice opposed by a star pair, and (b) a lone floor voice opposed by exactly two stars versus exactly two mids."
}
```

## Usage

```json
{
  "input_tokens": 40995,
  "output_tokens": 10103
}
```
