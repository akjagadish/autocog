# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_8" and "pi_6") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_8" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_6" (THEORY 2 below). Justify your choice.


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

## THEORY 2 — pi_6
**Capacity-Gated Strongly-Saturating Redundancy Validity-Inversion Tallying (GVIT-CAP-S4).**

Five commitments describe how people combine binary expert ratings whose stated validities they have been told.

**(1) The count comparator is a gate, not a weight.** Before any evaluation of *whose* rating is whose, the subject asks a purely ordinal question: does one product win on more attributes than the other? This is answered with unweighted feature-wise comparisons. If the answer is "no" (equal numbers of winning attributes, including displays where nothing discriminates), the display is construed as *non-diagnostic*, no candidate is generated, and the subject flips a mental coin. Stated validities are **never** used as a tie-break. This is a hard, parameter-free prediction: on tally-tied, side-counterbalanced profiles adherence to the most-valid discriminating cue is exactly chance whatever the validity vector.

**(2) When a majority exists, integration uses inverse-informativeness weights.** The subjective news value of a rating is a smooth, panel-relative *decreasing* function of stated validity: s_j = (v_j - v_min)/(v_max - v_min) within the displayed panel, w_j = 1 - lambda_eff*f(s_j). A near-perfect expert is heard as merely restating the obvious consensus quality of the product, so his rating adds almost nothing (and, when lambda_eff*f > 1, is mildly counter-indicative); a barely-diagnostic expert is surprising and is treated as genuine private information carrying full weight. Evidence is the single scalar E = sum_j w_j*sign(a_j - b_j).

**(3) The redundancy construal saturates VERY STRONGLY just above the panel floor.** Redundancy is not judged on a linear scale of stated accuracy: "being one of the accurate ones" is a nearly categorical construal. An expert only has to be slightly above the panel's least-accurate voice before he is already heard as echoing the consensus. Formally the discount is a strongly concave function of panel-relative validity, f(s) = s^omega with omega well below 1 (~0.26-0.42), so the discount rises steeply just above the panel floor and then flattens. Two consequences follow that neither a linear nor a mildly concave inversion produces. (i) A majority carried by *middling* experts (the 2nd-5th most accurate voices) carries almost no subjective evidence: mid-panel support is discounted nearly as heavily as star support, so a mid-backed majority facing a lone star is followed only weakly, close to chance. (ii) Conversely, when the count loser is backed by the panel's genuinely idiosyncratic bottom voices, the anti-count evidence is *strengthened*, sharpening the reversal. So the model separates three displays that a linear inversion renders nearly identical: majority-by-mid-experts (followed barely above chance), majority-by-bottom-voices (followed near ceiling), and star-backed majorities (reversed toward floor).

**(4) Redundancy is comparative; evidence spread over many contradicting cues is compressed.** An expert is discounted only because other experts are heard as echoing him, so the inversion scales with how many voices actually speak on the trial: lambda_eff = lambda_n*(D-1)/((D-1)+kappa_D), where D is the number of discriminating cues; when exactly one expert differentiates the products there is no chorus and his rating is taken at face value. Confidence in E is then compressed by the number of mutually contradicting cues, D^gamma.

**(5) The inversion is a capacity-limited construal: it collapses when the panel exceeds working-memory span.** Discounting the star requires holding the validity ordering of the whole panel in mind, which is possible only up to roughly seven experts. Beyond capacity the ordering blurs, so the redundancy construal washes out toward equal weighting (lambda_n = lambda*(1 - psi*o)) and integration becomes noisier (slope divided by 1 + chi*o), where o = logistic((n - 7.5)/0.2). Finally an attention lapse epsilon replaces the decision by a coin flip.

Cross-design signature: exact chance on tally-tied profiles; strong systematic reversal of star-backed majorities on panels of <=7 experts; near-chance for the same star-backed profiles on panels of 8+ experts, with preserved tally adherence there for broad-mixture majorities; near-ceiling following of a lone discriminating expert; only weakly above-chance following of mid-panel-backed majorities that oppose a lone star; and markedly attenuated confidence whenever the weighted evidence is spread across many mutually contradicting cues.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np

    # ---------- parse the stimulus ----------
    a = None
    b = None
    try:
        if isinstance(state, dict):
            if 'option_a_ratings' in state and 'option_b_ratings' in state:
                a = np.asarray(list(state['option_a_ratings']), dtype=float).ravel()
                b = np.asarray(list(state['option_b_ratings']), dtype=float).ravel()
    except Exception:
        a = None
        b = None

    if a is None or b is None:
        try:
            stim = np.asarray(state, dtype=float)
            if stim.ndim == 1:
                stim = stim.reshape(2, -1)
            elif stim.ndim > 2:
                stim = stim.reshape(2, -1)
            if stim.shape[0] != 2:
                stim = stim.reshape(2, -1)
            a = np.asarray(stim[0], dtype=float).ravel()
            b = np.asarray(stim[1], dtype=float).ravel()
        except Exception:
            return np.array([0.5, 0.5], dtype=float)

    n = int(a.shape[0])
    if n == 0 or int(b.shape[0]) != n:
        return np.array([0.5, 0.5], dtype=float)

    # ---------- feature-wise comparison ----------
    d = np.sign(a - b)                       # +1 A wins cue, -1 B wins cue, 0 tie
    nA = int(np.sum(d > 0))
    nB = int(np.sum(d < 0))
    D = nA + nB

    # ---- Commitment 1: the count comparator is a gate. No majority -> guess.
    if D == 0 or nA == nB:
        return np.array([0.5, 0.5], dtype=float)

    # ---------- parameters ----------
    def _get(name, default):
        try:
            return float(parameters.get(name, default))
        except Exception:
            return float(default)

    lam = _get('lam', 1.45)
    beta = _get('beta', 6.7)
    gamma = _get('gamma', 0.90)
    kappa_d = _get('kappa_d', 0.10)
    psi = _get('psi', 0.40)
    chi = _get('chi', 0.95)
    eps = _get('epsilon', 0.17)
    omega = _get('omega', 0.34)              # saturating-redundancy exponent

    lam = float(min(max(lam, 0.0), 2.5))
    beta = float(min(max(beta, 0.0), 30.0))
    gamma = float(min(max(gamma, 0.0), 2.5))
    kappa_d = float(min(max(kappa_d, 1e-6), 3.0))
    psi = float(min(max(psi, 0.0), 1.0))
    chi = float(min(max(chi, 0.0), 5.0))
    eps = float(min(max(eps, 0.0), 0.6))
    omega = float(min(max(omega, 0.2), 1.5))

    # ---------- stated validities ----------
    v = None
    v_raw = parameters.get('validities', None) if hasattr(parameters, 'get') else None
    if v_raw is not None:
        try:
            v = np.asarray(v_raw, dtype=float).ravel()
        except Exception:
            v = None
    if v is None or v.shape[0] != n:
        v = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.75], dtype=float)

    v_min = float(np.min(v))
    v_max = float(np.max(v))
    rng = v_max - v_min
    if not np.isfinite(rng) or rng <= 1e-9:
        s = np.zeros(n, dtype=float)          # no validity contrast in the panel
    else:
        s = (v - v_min) / rng                 # panel-relative informativeness, in [0,1]

    # ---- Commitment 3: the redundancy construal SATURATES above the floor ----
    # strongly concave transform: being appreciably above the panel's least-accurate
    # voice is already enough to be heard as "one of the accurate ones".
    s = np.power(np.clip(s, 0.0, 1.0), omega)

    # ---- Commitment 5: capacity overload of the panel (~7 experts) --------
    N_CAP = 7.5
    W_CAP = 0.20
    arg_o = (float(n) - N_CAP) / W_CAP
    arg_o = float(min(max(arg_o, -60.0), 60.0))
    o = 1.0 / (1.0 + np.exp(-arg_o))

    lam_n = lam * (1.0 - psi * o)
    if not np.isfinite(lam_n) or lam_n < 0.0:
        lam_n = 0.0

    # ---- Commitment 4: redundancy is comparative (needs a chorus) -------
    lam_eff = lam_n * (float(D) - 1.0) / ((float(D) - 1.0) + kappa_d)
    if not np.isfinite(lam_eff) or lam_eff < 0.0:
        lam_eff = 0.0

    # ---- Commitment 2: inverse-informativeness weights ------------------
    w = 1.0 - lam_eff * s
    E = float(np.sum(w * d))

    # ---- Commitment 4b: compression by the number of contradicting cues --
    denom = float(D) ** gamma
    if not np.isfinite(denom) or denom <= 0.0:
        denom = 1.0
    overload_noise = 1.0 + chi * o
    if not np.isfinite(overload_noise) or overload_noise <= 0.0:
        overload_noise = 1.0
    denom = denom * overload_noise

    z = beta * E / denom
    if z > 30.0:
        z = 30.0
    elif z < -30.0:
        z = -30.0

    p_a = 1.0 / (1.0 + np.exp(-z))

    # ---- attention lapse ------------------------------------------------
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
    if not np.isfinite(s) or s <= 0:
        probs = np.ones_like(probs) / float(len(probs))
    else:
        probs = probs / s
    return int(np.random.choice(len(probs), p=probs))

## EXPERIMENT 1 (proposed by pi_8)

### DESIGN
**Validities (n_features=6):** [0.88, 0.55, 0.72, 0.95, 0.63, 0.8]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1, 0]
  trial 2: A=[0, 1, 0, 0, 1, 0]  B=[1, 1, 0, 1, 1, 0]
  trial 3: A=[0, 1, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0, 1]
  trial 4: A=[0, 0, 1, 0, 0, 1]  B=[0, 1, 1, 0, 1, 1]
  trial 5: A=[1, 0, 0, 1, 1, 1]  B=[0, 0, 1, 0, 1, 1]
  trial 6: A=[0, 0, 1, 0, 1, 1]  B=[1, 0, 0, 1, 1, 1]
  trial 7: A=[1, 0, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 8: A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 1]
  trial 9: A=[0, 0, 1, 1, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  trial 10: A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 11: A=[1, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 0, 0]
  trial 12: A=[0, 1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 0, 1]
  trial 13: A=[1, 1, 0, 1, 1, 0]  B=[0, 0, 1, 0, 1, 1]
  trial 14: A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 1, 1, 0]
  trial 15: A=[1, 0, 1, 0, 1, 1]  B=[1, 0, 0, 1, 1, 0]
  trial 16: A=[1, 0, 0, 1, 1, 0]  B=[1, 0, 1, 0, 1, 1]

**Rationale:** GOAL: dissociate CG-VI-2 (advocated) from GVIT-CAP-S4 (competing). The two models are nearly identical in spirit (hard count gate -> exact 0.5 on tally ties, inverse-validity 'news value' weights with a strongly saturating exponent s^omega, dilution by D^gamma, capacity term above 7 experts), so tally-tied cells and generic conflict cells are useless \u2014 they agree exactly. They differ on exactly TWO structural commitments, and every cell here is built to turn one of them into a parameter-free equality or a sign reversal.

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

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
The two models differ structurally only in whether a 'redundant' (high-validity) expert is MUTE (CG-VI-2: news value floored at zero) or COUNTER-INDICATIVE (GVIT: weights go negative). That difference can only show up on displays in which the discriminating voices are ALL appreciably above the panel floor, i.e. where no low-validity maverick (0.55 or 0.63) speaks. This metric selects exactly those trials (all discriminating features have validity >= 0.72, and a strict count winner exists) and scores adherence to the plain count winner. In this design those are the A1 (T1/T2, unopposed star pair, 2-0), A3 (T5/T6, star pair vs lone mid, 2-1), A6 (T11/T12, all-upper 3-1) and A8 (T15/T16, mid pair vs lone star) cells \u2014 48 of the 96 trials, so the binomial component of per-subject noise is small.

Predictions: under CG-VI-2 all discriminating weights clip to zero in A1 and A6, giving EXACTLY 0.500 there for every parameter setting (zero between-subject parameter variance), plus ~0.68 in A3 and ~0.87 in A8, so the pooled adherence lands near 0.63-0.65. Under GVIT the same experts carry negative weight, so the count winner is actively rejected in A1 (~0.25), A3 (~0.21) and A6 (~0.19) and only weakly followed in A8 (~0.59), giving pooled adherence near 0.30-0.33. Predicted gap ~0.30 in a quantity bounded in [0,1], with tight per-subject distributions (two of the four cells are parameter-free under CG, and the deliberately excluded floor-voice cells \u2014 A2, A4, A5, A7 \u2014 are the ones that carry most of each model's parameter-driven spread). Crucially it is not trivially true for CG: it would come out near chance or below if humans actually treat high-validity ratings as counter-indicative.

Source:
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

### RESULTS
- Predicted under pi_8 (simulated): 0.6446 (var=0.0042)
- Predicted under pi_6 (simulated): 0.3654 (var=0.0148)
- Observed on real data: 0.4892 (var=0.0049)

## EXPERIMENT 2 (proposed by pi_6)

### DESIGN
**Validities (n_features=6):** [0.92, 0.6, 0.84, 0.53, 0.76, 0.68]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 1, 1, 0, 1]  B=[0, 0, 0, 1, 0, 1]
  trial 2: A=[0, 0, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 1]
  trial 3: A=[1, 1, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1, 0]
  trial 4: A=[1, 0, 0, 0, 1, 0]  B=[1, 1, 0, 1, 1, 0]
  trial 5: A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  trial 6: A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0]
  trial 7: A=[1, 0, 1, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1]
  trial 8: A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 1]
  trial 9: A=[0, 0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 10: A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  trial 11: A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 0, 0, 1, 1]
  trial 12: A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 0, 1, 1, 1]
  trial 13: A=[1, 0, 1, 1, 1, 0]  B=[0, 0, 0, 1, 0, 1]
  trial 14: A=[0, 0, 0, 1, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  trial 15: A=[1, 1, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  trial 16: A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]

**Rationale:** GOAL: dissociate GVIT-CAP-S4 (advocated) from CG-VI-2 (competing). The two models share a hard count gate (tally ties -> exactly 0.5), an inverse-validity 'news value' weighting with a strongly saturating exponent s^omega, dilution by D^gamma and a capacity term above 7 experts. They differ on exactly TWO structural commitments, and every cell below turns one of them into a sign reversal or a parameter-free equality. (1) THE ZERO FLOOR: CG clips w = max(0, 1 - lam_eff*s^omega), so a redundant expert is MUTE; GVIT lets w go NEGATIVE, so a high-validity expert votes against his own product. (2) THE CHORUS CRITERION: CG reads the chorus off bloc structure, CH = max(mA,mB)+|mA-mB|-2 with a steep gate CH^q/(CH^q+kappa) (q~4), so a 2-vs-1 display (CH=1) is only WEAKLY discounted (lam_eff ~ 0.53-0.79); GVIT uses the raw number of discriminating cues, lam_eff = lam*(D-1)/((D-1)+kappa_d), so the same 2-vs-1 display (D=3) is NEAR-MAXIMALLY discounted (lam_eff ~ 0.90-1.79). Hence 2-vs-1 displays are the sign-reversal engine and unopposed/all-upper displays are the zero-floor engine.

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

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
The two theories differ most sharply on displays whose tally-majority is carried by the panel's two star experts (f0=.92, f2=.84). CG-VI-2 floors news value at zero, so those stars are merely MUTE: for the unopposed star pair (C1, 2v0) and the all-upper 3v1 (C7) every discriminating weight clips to zero and P(choose the count winner) is exactly 0.500 by construction, while for the 2v1 star-pair cells (C3, C4) its weak chorus gate leaves positive evidence, so the majority is still followed above chance (~.60-.69). GVIT-CAP-S4 lets the weights go NEGATIVE, so on exactly these cells the star-backed majority is actively rejected (~.17-.22). Pooling only these four cells (48 trials per subject, mirrored A/B so side bias cancels) yields a predicted adherence of ~0.20 under GVIT versus ~0.57 under CG-VI-2 — a large mean gap, while the per-subject binomial noise over 48 trials is small and two of the four cells are parameter-free under CG, keeping between-subject variance tight. The metric is a simple proportion, well defined on both pooled and single-subject slices, and is not trivially true of my theory: it is a plain measure of following the unweighted cue count, which most standard accounts predict well above chance.

Source:
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


### RESULTS
- Predicted under pi_8 (simulated): 0.5913 (var=0.0058)
- Predicted under pi_6 (simulated): 0.2792 (var=0.0159)
- Observed on real data: 0.4892 (var=0.0049)

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
**Predicted under pi_6:** 0.5106 (var=0.0056)

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
**Predicted under pi_6:** 0.7025 (var=0.0100)

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
**Predicted under pi_6:** 0.7831 (var=0.0039)

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
**Predicted under pi_6:** 0.7175 (var=0.0042)

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
**Predicted under pi_6:** -0.7525 (var=0.0071)

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
**Predicted under pi_6:** -0.9162 (var=0.0494)

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
**Predicted under pi_6:** 0.6775 (var=0.0023)

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
**Predicted under pi_6:** 0.4919 (var=0.0121)

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
**Predicted under pi_6:** 0.3791 (var=0.0043)

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
**Predicted under pi_6:** 0.3338 (var=0.0282)

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
**Predicted under pi_6:** 0.5292 (var=0.0060)

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
**Predicted under pi_6:** 0.2562 (var=0.0218)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the whole 14-experiment record pi_8 (CG-VI-2) is the stronger, more task-invariant account. It matches almost exactly on Exp 9 (.657 vs .661), Exp 10 (.485 vs .485), Exp 11 (.395 vs .405), Exp 12 (-.016 vs -.023), Exp 13 (.692 vs .698) and Exp 14 (.138 vs .145) \\u2014 the very cells that separate the two accounts. pi_6 (GVIT-CAP-S4) fails badly exactly there: it predicts +.334 where the data give -.023 (Exp 12), .529 vs .698 (Exp 13) and .256 vs .145 (Exp 14). pi_6's only advantages are small, and they come from generic tally-adherence metrics (Exp 4, 5, 6, 7, 8) where pi_8 merely runs ~.05-.10 hot; those are calibration differences, not structural ones. The mechanism that the data endorse is pi_8's bloc-size ('chorus') gating rather than pi_6's raw-cue-count gating: 2-vs-1 mid-versus-floor displays land essentially at chance (Exp 12: floor-choice minus tied-cell baseline \\u2248 0) whereas 3-vs-1 with the same lone floor voice collapses to .145 (Exp 14). pi_6's D-based gate cannot produce that non-monotonicity, and its negative weights predict reversals that never appear.\n\nThe two newest experiments, however, expose a real shortfall in BOTH. Exp 1 and Exp 2 both pooled adherence on all-upper-band displays at 0.489 \\u2014 flat chance, with tight between-subject variance (.005). pi_6's signed inversion predicted .28-.37 (clear rejection of the count winner): decisively falsified. pi_8's zero-floor correctly rules out rejection but over-predicts adherence (.59-.64) because its chorus gate is weak at CH=1: it still lets a star pair versus a lone mid (.69) and a mid pair versus a lone star (.87) sit well above chance. The data say that whenever every discriminating voice sits appreciably above the panel floor \\u2014 whatever the count margin, whatever the 2-vs-0 / 2-vs-1 / 3-vs-1 structure \\u2014 the display is simply non-informative and people coin-flip. So the empirically correct rule is closer to 'only the panel's genuinely low-accuracy voices carry any evidence at all, and their force grows with the size of the bloc they oppose', with high-validity voices mute unconditionally rather than mute-only-when-a-big-chorus-is-present.\n\nHence: pi_6 is the degenerate one and should be replaced, and the replacement should be built to beat pi_8 on precisely the Exp 1 / Exp 2 shortfall while retaining pi_8's successes on Exp 9-14.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace pi_6 with a 'Mute-Band Maverick Arbitration' theory (working name MBM). Core commitments:\n\n(1) UNWEIGHTED COUNT GATE (kept, parameter-free): equal feature-wise win counts (including wholly non-discriminating displays) \\u2192 exactly 0.5, no validity tie-break, no take-the-best fallback. This is required by Exp 3 (.474) and by the tied cells of Exp 7/10/12.\n\n(2) HARD MUTE BAND, CHORUS-INDEPENDENT (the key departure from BOTH pi_8 and pi_6). Panel-relative seniority s_j = (v_j - v_min)/(v_max - v_min). News weight u(s) = max(0, 1 - s/theta)^rho + eta, with theta ~ 0.25-0.40 and rho ~ 0.7-1.5; eta is a tiny residual (0-0.08) common to every discriminating cue. Any expert more than theta of the panel span above the floor is essentially SILENT \\u2014 not negative (pi_6's error) and not conditionally silent only when a big bloc is visible (pi_8's error). Immediate consequence: displays in which every discriminating voice is upper-band (Exp 1 cells A1/A3/A6/A8, Exp 2 cells C1/C3/C4/C7) give E \\u2248 0 and therefore ~0.5 adherence at ANY count margin \\u2014 which is exactly the observed 0.489 in both new experiments, where pi_8 over-predicts (.59-.64) and pi_6 under-predicts (.28-.37). The small eta lets 2-vs-0 and 3-vs-1 upper-only displays sit a hair above chance without ever reaching pi_8's .69-.87.\n\n(3) CHORUS AMPLIFICATION OF THE MAVERICK (replaces pi_6's D-based discount with a bloc-based one, matching the evidence that favoured pi_8's CH rule). The weight of a below-theta voice is multiplied by an increasing, saturating function of the size of the bloc it opposes: g = 1 + alpha * (m_opp - 1)^p / ((m_opp - 1)^p + kappa), m_opp = size of the larger (count-winning) bloc. Calibrate so that a lone floor voice against a 2-cue majority yields only ~0.4-0.5 adherence (Exp 12 \\u2248 0 contrast; Exp 13 c2 \\u2248 .3-.4) while a lone floor voice against a 3-cue majority drives adherence down to ~0.15 (Exp 14 = .145).\n\n(4) DILUTION AND LAPSE: divide the logit by D^gamma (D = number of discriminating cues) and mix with an attention lapse epsilon, as before.\n\n(5) CAPACITY MODULE for n > 7: shrink theta's effect (weights flatten toward equal) AND inflate noise, tuned so that 8-expert star-backed majorities sit at chance (Exp 10 = .485) while overall tally adherence on 8-expert mixed displays stays ~0.75 (Exp 6).\n\nCalibration targets the new theory must hit simultaneously (use these to set parameter ranges): Exp 1 = .49, Exp 2 = .49, Exp 3 = .47, Exp 4 \\u2248 .65, Exp 5 \\u2248 .79, Exp 6 \\u2248 .75, Exp 7 \\u2248 -.76, Exp 8 \\u2248 -.94, Exp 9 \\u2248 .66, Exp 10 \\u2248 .49, Exp 11 \\u2248 .41, Exp 12 \\u2248 -.02, Exp 13 \\u2248 .70, Exp 14 \\u2248 .15. The decisive future test against pi_8 is any 2-vs-1 or 2-vs-0 display whose discriminating voices are all upper-band: MBM says exactly chance, pi_8 says .69-.88."
}
```

## Usage

```json
{
  "input_tokens": 36815,
  "output_tokens": 12980
}
```
