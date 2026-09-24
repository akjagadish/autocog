# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_7" and "pi_6") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_7" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_6" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_7
**Superlinear Redundancy-Discounted Tallying with a Bounded Anti-Count Floor, Baselined Seniority-Scaled Dissent Salience, Dissent-Graded Evidence Dilution and Two-Regime Panel-Size Precision Loss (SR-DT-6b).**

People told the validities of a panel of binary expert ratings do not weight experts *by* validity; they weight them by how much *news* a rating carries, which is a decreasing function of how senior the expert is inside the displayed panel. Eight commitments.

**(1) The count comparator is a gate, not a weight.** Before any expert is evaluated, an unweighted feature-wise comparison asks: does one product win on strictly more attributes? If not (equal win-counts, including wholly non-discriminating displays), the display is construed as non-diagnostic: choice is exactly 0.5, with NO validity tie-break and NO take-the-best fallback. Parameter-free: on tally-tied, side-counterbalanced profiles adherence to the most-valid discriminating cue is exactly chance whatever the validity vector.

**(2) Seniority is measured against the chance baseline and the panel's best voice, and the discount is SUPERLINEAR in seniority.** s_j = (v_j - 0.5)/(v_max - 0.5), clipped to [0,1]; news value w_j = 1 - lambda * s_j^omega with omega > 1. The discount is negligible across the bottom half of the panel, about half weight for genuine mid-panel experts, and only for the top one or two voices does it drive weight to zero or below. Two mid-panel experts therefore roughly offset one floor-level voice, while a single top expert is worth nothing or less.

**(2b) The anti-count is BOUNDED: news value has a mild negative floor.** w_j = max(w_floor, 1 - lambda * s_j^omega) with a small negative w_floor. Endorsement by the panel's star can subtract a little, never a lot.

**(3) Redundancy requires same-side corroboration.** lambda_side = lambda_n * (m_side - 1)/((m_side - 1) + kappa). A lone voice cannot be redundant with anybody.

**(4) A lone voice is heard at MORE than face value, with a seniority-graded bonus on top of a constant dissent baseline.** w = phi0 + (phi - phi0) * s_j for a solitary discriminating cue.

**(5) Confidence is diluted by DISSENT more than by sheer cue count, but only MILDLY so.** Two functionally distinct things can make a display 'busy': many attributes discriminating at all, and some of those attributes pointing the other way. Only the second is genuinely undermining, but its undermining force is *modest and graded*, not overwhelming. Formally the divisor is D^gamma_d * (1 + m_lose)^gamma_c with a sub-linear cue-count exponent gamma_d and a mild-to-moderate dissent exponent gamma_c that is bounded well below 1.0, where m_lose is the number of cues backing the count loser. Consequences: unopposed majorities grow in subjective strength with their size (dense-unanimous adherence exceeds sparse-unanimous adherence), while internally contradicted displays are pulled somewhat — but never all the way — toward indifference. Crucially the *asymmetry* of this dilution is empirically constrained: displays with a single dissenting voice (lone-star dissent, the cells that generate the large negative coherence contrasts) must retain nearly full evidential force, whereas displays with two or more dissenters are noticeably softened. A strong dissent exponent would flatten the former and is ruled out.

**(6) Precision also falls, in two regimes, with panel size:** divisor multiplied by (n/6)^tau * (1 + eta * max(0, n - 7)).

**(7) Ordering fidelity decays smoothly with panel span:** lambda_n = lambda * exp(-rho * max(0, n - 7)).

**(8) The decision stage is low-lapse but only moderately steep.** The lapse rate epsilon alone fixes the ceiling attainable on evidence-saturated displays; the slope beta alone fixes how far intermediate, internally conflicted displays are driven from indifference. Stimulus-driven evidence strength and stimulus-independent attentional failure are separate sources of stochasticity.

**Cross-design signature:** exact chance on tally-tied displays of any validity composition; strong but bounded reversal of chorus-backed star majorities on six/seven-expert panels; near-chance for the same structure on eight-expert panels with visibly reduced pure-tally adherence there; near-ceiling and *increasing-with-size* following of unopposed majorities; distinctly middling adherence whenever a numeric majority is actively contradicted by several cues; and mid-panel-backed majorities facing a lone floor voice sitting essentially at chance.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np

    # ---------- parse the stimulus ----------
    def _pair(st):
        a = None
        b = None
        if isinstance(st, dict):
            for ka, kb in (('option_a_ratings', 'option_b_ratings'),
                           ('a_ratings', 'b_ratings'),
                           ('A', 'B'), ('a', 'b')):
                if ka in st and kb in st:
                    try:
                        a = np.asarray(list(st[ka]), dtype=float).ravel()
                        b = np.asarray(list(st[kb]), dtype=float).ravel()
                    except Exception:
                        a = None
                        b = None
                    break
        if a is None:
            try:
                arr = np.asarray(st, dtype=float)
                arr = arr.reshape(2, -1)
                a = np.asarray(arr[0], dtype=float).ravel()
                b = np.asarray(arr[1], dtype=float).ravel()
            except Exception:
                return None, None
        return a, b

    a, b = _pair(state)
    if a is None or b is None:
        return np.array([0.5, 0.5], dtype=float)
    if a.size == 0 or a.size != b.size:
        return np.array([0.5, 0.5], dtype=float)

    n = int(a.size)

    # ---------- Commitment 1: unweighted count gate ----------
    d = np.sign(a - b)
    iA = np.where(d > 0)[0]
    iB = np.where(d < 0)[0]
    mA = int(iA.size)
    mB = int(iB.size)
    if mA == mB:                       # includes the wholly non-discriminating case
        return np.array([0.5, 0.5], dtype=float)
    D = float(mA + mB)
    m_lose = float(min(mA, mB))        # cues backing the count LOSER (dissent)

    # ---------- parameters ----------
    def _g(name, default, lo, hi):
        try:
            x = float(parameters.get(name, default))
        except Exception:
            x = float(default)
        if not np.isfinite(x):
            x = float(default)
        return float(min(max(x, lo), hi))

    lam = _g('lam', 1.60, 0.0, 3.0)
    omega = _g('omega', 1.70, 0.5, 3.5)
    beta = _g('beta', 13.5, 0.0, 40.0)
    gamma_d = _g('gamma_d', 0.68, 0.0, 2.5)   # sub-linear dilution by sheer cue count
    gamma_c = _g('gamma_c', 0.66, 0.0, 2.5)   # mild dilution by DISSENT
    kappa = _g('kappa', 0.05, 1e-6, 1.0)
    rho = _g('rho', 0.70, 0.0, 2.0)
    tau = _g('tau', 0.20, 0.0, 3.0)
    eta = _g('eta', 2.10, 0.0, 6.0)
    phi = _g('phi', 1.40, 0.5, 2.5)
    phi0 = _g('phi0', 1.08, 0.5, 2.0)
    eps = _g('epsilon', 0.09, 0.0, 0.6)
    w_floor = _g('w_floor', -0.30, -1.5, 0.0)   # bounded anti-count (Commitment 2b)

    # ---------- stated validities ----------
    v = None
    try:
        vr = parameters.get('validities', None) if hasattr(parameters, 'get') else None
        if vr is not None:
            v = np.asarray(vr, dtype=float).ravel()
    except Exception:
        v = None
    if v is None or v.size != n:
        v = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.75], dtype=float)

    # ---------- Commitment 2: seniority anchored at chance & the panel's best voice ----------
    vmax = float(np.max(v))
    span = vmax - 0.5
    if (not np.isfinite(span)) or span <= 1e-9:
        s = np.zeros(n, dtype=float)
    else:
        s = np.clip((v - 0.5) / span, 0.0, 1.0)
    s_pow = np.power(np.clip(s, 0.0, 1.0), omega)   # superlinear discount profile

    # ---------- Commitment 7: span-graded ordering fidelity ----------
    over = max(0.0, float(n) - 7.0)
    lam_n = lam * float(np.exp(-rho * over))
    if (not np.isfinite(lam_n)) or lam_n < 0.0:
        lam_n = 0.0

    # ---------- Commitment 3: redundancy needs same-side corroboration ----------
    def _side_lam(m):
        mm = float(m)
        if mm <= 1.0:
            return 0.0                       # a lone voice cannot be redundant
        val = lam_n * (mm - 1.0) / ((mm - 1.0) + kappa)
        if (not np.isfinite(val)) or val < 0.0:
            return 0.0
        return float(val)

    lam_A = _side_lam(mA)
    lam_B = _side_lam(mB)

    # ---------- Commitment 2b: news value is bounded below by a mild negative floor ----------
    if mA > 0:
        wA = np.maximum(w_floor, 1.0 - lam_A * s_pow[iA])
    else:
        wA = np.array([], dtype=float)
    if mB > 0:
        wB = np.maximum(w_floor, 1.0 - lam_B * s_pow[iB])
    else:
        wB = np.array([], dtype=float)

    # ---------- Commitment 4: baselined, seniority-scaled lone-dissent salience ----------
    phi_eff = phi if phi > phi0 else phi0
    if mA == 1:
        wA = phi0 + (phi_eff - phi0) * s[iA]
    if mB == 1:
        wB = phi0 + (phi_eff - phi0) * s[iB]

    E = float(np.sum(wA) - np.sum(wB))
    if not np.isfinite(E):
        E = 0.0

    # ---------- Commitment 5: DISSENT-graded dilution (sub-linear in sheer cue count) ----------
    den = (float(D) ** gamma_d) * ((1.0 + m_lose) ** gamma_c)
    if (not np.isfinite(den)) or den <= 0.0:
        den = 1.0

    # ---------- Commitment 6: panel-size precision loss (two regimes) ----------
    size_fac = ((float(n) / 6.0) ** tau) * (1.0 + eta * max(0.0, float(n) - 7.0))
    if (not np.isfinite(size_fac)) or size_fac <= 0.0:
        size_fac = 1.0
    den = den * size_fac

    z = beta * E / den
    if not np.isfinite(z):
        z = 0.0
    z = float(min(max(z, -30.0), 30.0))

    p_a = 1.0 / (1.0 + np.exp(-z))

    # ---------- Commitment 8: attention lapse (ceiling) is separate from slope ----------
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

## EXPERIMENT 1 (proposed by pi_7)

### DESIGN
**Validities (n_features=6):** [0.78, 0.95, 0.68, 0.84, 0.9, 0.72]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 1, 0]
  trial 2: A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 1, 1]
  trial 3: A=[1, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 1, 0]
  trial 4: A=[0, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 1, 0]
  trial 5: A=[1, 0, 1, 1, 0, 1]  B=[0, 1, 1, 0, 0, 1]
  trial 6: A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0, 1]
  trial 7: A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 1, 0, 1]
  trial 8: A=[1, 0, 0, 1, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  trial 9: A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  trial 10: A=[0, 1, 0, 0, 1, 0]  B=[1, 0, 1, 0, 0, 1]
  trial 11: A=[1, 1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 1, 0]
  trial 12: A=[0, 1, 0, 0, 1, 0]  B=[1, 1, 0, 1, 1, 1]

**Rationale:** GOAL: dissociate SR-DT-6b (advocated) from GVIT-CAP-S4 (competing). Both models share a hard count gate (tally ties -> exactly 0.5), both invert validity (stars discounted), both release the discount for solo voices, and both compress by the number of discriminating cues, so ordinary conflict trials and tally-tied trials are useless (they agree exactly). They differ on exactly three things, and the panel is engineered so each difference yields a sign reversal or a huge magnitude gap at a FIXED panel size (n=6, which is below both models' capacity/overload terms: SR-DT's size_fac = (6/6)^tau = 1 and GVIT's o ~ 0, removing all panel-size nuisance).

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

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
Composite of three cell-level adherence contrasts, each of which has an analytically guaranteed sign difference between the two models at n_features=6 (both models' panel-size terms are inert here), averaged to shrink binomial noise.

(1) LONE-VOICE contrast (D3 - C2): the majority (the .84 and .78 experts) is held exactly fixed and only the identity of the single dissenter changes (.95 vs .68). Under SR-DT a lone voice is boosted above face value with a seniority-graded bonus, so the star dissenter is at least as damaging as the floor dissenter: the contrast is necessarily <= 0 (simulated ~-0.02 to -0.05, both cells near the floor). Under GVIT the lone voice's weight is 1 - lam_eff*s^omega with v_min anchoring, so the .68 dissenter carries weight exactly +1 while the .95 dissenter carries 1 - lam_eff < 0: the contrast is necessarily large and POSITIVE (~+0.32 to +0.54 across its whole parameter box).

(2) E4 adherence (majority = .95,.90,.68 vs .78,.72): SR-DT's chance-baselined superlinear discount makes the star-backed majority counter-indicative (~0.15-0.30), whereas GVIT's floor-anchored weights give the .68 voice full weight that offsets the two discounted stars, landing at ~0.45-0.60.

(3) Reversed A1 adherence (1 - follow of the mid-trio majority): SR-DT leaves the middling experts substantial positive news value (~0.72 adherence -> term ~0.28); GVIT's strongly concave s^omega nearly nullifies them while the .68 voice on the other side keeps full weight (~0.32 adherence -> term ~0.68).

All three terms point the same way (GVIT high, SR-DT low), each is computed from 16 trials per subject, and each is tight within its own parameter box, so the pooled score should be ~0.15 for SR-DT vs ~0.55 for GVIT with small between-subject variance. Nothing here is trivially true of SR-DT: it is a measured cross-cell pattern that either model could in principle produce.

Source:
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

### RESULTS
- Predicted under pi_7 (simulated): 0.1512 (var=0.0062)
- Predicted under pi_6 (simulated): 0.5292 (var=0.0060)
- Observed on real data: 0.6983 (var=0.0122)

## EXPERIMENT 2 (proposed by pi_6)

### DESIGN
**Validities (n_features=6):** [0.72, 0.95, 0.58, 0.88, 0.52, 0.68]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 1, 0]
  trial 2: A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 1]
  trial 3: A=[1, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 1, 0]
  trial 4: A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 1, 0, 1]
  trial 5: A=[1, 0, 1, 1, 0, 1]  B=[0, 1, 0, 1, 0, 0]
  trial 6: A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 1, 0, 1]
  trial 7: A=[1, 0, 1, 1, 0, 1]  B=[0, 0, 0, 1, 1, 0]
  trial 8: A=[0, 0, 0, 1, 1, 0]  B=[1, 0, 1, 1, 0, 1]
  trial 9: A=[1, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 0]
  trial 10: A=[0, 1, 0, 1, 0, 0]  B=[1, 1, 0, 1, 0, 1]
  trial 11: A=[1, 0, 1, 1, 1, 0]  B=[1, 0, 0, 1, 0, 0]
  trial 12: A=[1, 0, 0, 1, 0, 0]  B=[1, 0, 1, 1, 1, 0]

**Rationale:** GOAL: dissociate GVIT-CAP-S4 (advocated) from SR-DT-6b (competing). The two models share a hard count gate (tally ties -> exactly 0.5), both invert stated validity, both release the discount for solo voices, and both compress evidence by the number of discriminating cues, so tally-tied cells and generic conflict cells are useless (they agree exactly). They differ on exactly three things: (i) the SHAPE of the redundancy discount (GVIT: strongly concave s^omega, omega~.26-.42, so low-MID experts are discounted almost as hard as stars; SR-DT: superlinear s^omega, omega>1, so low-mid experts keep large positive news value and only the top one or two voices go negative, bounded by w_floor); (ii) the CHORUS criterion (GVIT's lam_eff uses the TOTAL number of discriminating cues D, so a lone dissenting star inside a D=3 display is still counter-indicative, w = 1-lam_eff < 0; SR-DT's redundancy is SAME-SIDE, so any side with exactly one cue is BOOSTED above face value, w = phi0+(phi-phi0)*s >= 1.15 for a lone star); (iii) DISSENT dilution (SR-DT divides by (1+m_lose)^gamma_c so unopposed majorities approach ceiling; GVIT has no dissent term and simply divides by D^gamma).

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

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
This isolates the single most diagnostic cell of the design (K4, trials 7/8 and their exact mirrors): a three-cue majority carried by the low-mid experts (.72, .58, .68) opposed by exactly ONE dissenting cue, and that dissenter is the panel's floor voice (.52). The metric is the proportion of those 16 trials on which the subject follows the 3-cue count majority; because both A-favouring and B-favouring mirrors are pooled, side bias cancels exactly.

GVIT-CAP-S4 and SR-DT-6b make opposite-sided predictions here, for structural reasons rather than parameter fine-tuning. Under GVIT, the redundancy discount is strongly concave (omega<1) and its chorus criterion uses the TOTAL number of discriminating cues, so all three mid-level supporters are discounted almost to zero news value while the lone .52 dissenter (panel-relative s = 0 exactly, weight 1.0, parameter-free) retains full weight: E is negative across essentially the whole parameter box, giving sub-chance adherence (~0.2). Under SR-DT, seniority is anchored at chance, the discount is superlinear (so sub-.80 experts keep most of their voice), redundancy is same-side only, and the lone floor dissenter gets only ~phi0 weight; unopposed-ish majorities are further sharpened by the dissent divisor, yielding near-ceiling adherence (~0.9). A 3-vs-1 count majority being REJECTED is impossible anywhere in SR-DT's box.

Each subject contributes 16 trials to the estimate, so within-subject binomial noise is small (<=0.016 variance) and the predicted between-theory gap (~0.7) is many standard errors wide at N=25. I advocate GVIT and predict a value near 0.2; the competitor must predict near 0.9.

Source:
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


### RESULTS
- Predicted under pi_7 (simulated): 0.9187 (var=0.0063)
- Predicted under pi_6 (simulated): 0.2562 (var=0.0218)
- Observed on real data: 0.1450 (var=0.0099)

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
**Predicted under pi_7:** 0.4819 (var=0.0066)
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
**Predicted under pi_7:** 0.7050 (var=0.0185)
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
**Predicted under pi_7:** 0.7769 (var=0.0024)
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
**Predicted under pi_7:** 0.7642 (var=0.0035)
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
**Predicted under pi_7:** -0.7733 (var=0.0093)
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
**Predicted under pi_7:** -0.8237 (var=0.0280)
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
**Predicted under pi_7:** 0.7197 (var=0.0017)
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
**Predicted under pi_7:** 0.4775 (var=0.0150)
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
**Predicted under pi_7:** 0.3969 (var=0.0046)
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
**Predicted under pi_7:** -0.0475 (var=0.0412)
**Predicted under pi_6:** 0.3338 (var=0.0282)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the full set of twelve experiments the robust regularities are: (a) a hard count gate \u2014 on tally-tied displays adherence to the most-valid discriminating cue is exactly chance (Exp 3: 0.474; Exp 12's C3 cell ~0.5; Exp 10 star-backed majorities 0.485); (b) a very strong INVERSE use of stated validity once a count winner exists (Exp 7 coherence contrast -0.76, Exp 8 -0.94, Exp 9 marginal-tier following 0.66); and (c) overall tally adherence of ~0.75-0.79 on generic mixed designs (Exps 5, 6) with rejection of the tally exactly when the count winner is carried by the panel's accurate voices.\n\nThe two theories agree on the gate and on the sign of the inversion; they differ on the SHAPE of the discount (pi_6: strongly concave/saturating \u2014 anyone appreciably above the panel floor is already heard as echoing the consensus, so mid experts are nearly nullified and the floor voice keeps full weight; pi_7: superlinear \u2014 only the top one or two voices are discounted, mids retain large positive news value, lone voices are boosted above face value on either side) and on the chorus criterion. The two head-to-head experiments were engineered precisely on that difference and both come out decisively against pi_7. Exp 1: pi_7 predicted 0.151, pi_6 predicted 0.529, observed 0.698 \u2014 the observed value is not merely closer to pi_6, it lies BEYOND pi_6 in pi_6's own direction, and it arithmetically forces the lone-star-minus-lone-floor contrast to be POSITIVE (>= +0.1), which SR-DT's same-side lone-voice boost makes structurally impossible. Exp 2: pi_7 predicted 0.919, pi_6 predicted 0.256, observed 0.145 \u2014 a 3-cue low-mid majority is rejected 85% of the time in favour of a single .52 dissenter, an outcome SR-DT declares impossible anywhere in its box, and again the observed value is more extreme than pi_6's in pi_6's direction. On the remaining ten experiments the two theories are nearly indistinguishable (Exps 3,4,5,7,10,11 both within noise), with pi_6 better on Exps 8 and 9 and pi_7 better only on Exp 6 (marginally) and Exp 12.\n\nExp 12 is the single genuine anomaly for pi_6: the C4 cell (mid pair .74/.70 vs a lone .52 floor voice) sits at chance (implied p_floor ~= 0.48) whereas pi_6 predicted ~0.83 (metric +0.334 vs observed -0.023). Note that this is structurally the SAME display type as Exp 2's K2/K4 and Exp 1's c2, where the floor voice dominated. The only systematic differences are the size of the opposing chorus (2 corroborators in Exp 12-C4 vs 3 in Exp 2-K4) and the density of the discriminating set (D=3 vs D=4). That points to a real mechanism neither theory has right: the strength of the redundancy/inversion construal grows gradually with the number of SAME-SIDE corroborators, rather than being switched on at D>=2 (pi_6, kappa_d=0.1) or applied only to the top ranks (pi_7). So pi_6 is clearly the better theory overall and should be retained, while pi_7's central commitments (superlinear discount leaving mid-panel experts full voice; non-positive lone-star-minus-lone-floor contrast; dissent-graded dilution) are directly falsified and cannot be rescued by re-parameterisation.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace pi_7 entirely. Do NOT propose another variant in which mid-panel experts retain large positive news value or in which a lone dissenter is boosted regardless of its identity \u2014 Exps 1, 2, 7, 8, 9 and 10 rule that family out.\n\nProposed replacement (a genuine competitor to pi_6, designed to fit ALL twelve datasets, especially Exp 12 where pi_6 fails): 'Chorus-Graded Validity Inversion with a Chance-Anchored Saturating Discount'.\n\n(1) Keep the parameter-free count gate: equal feature-wise win counts (including wholly non-discriminating displays) -> exactly 0.5, with no validity tie-break and no take-the-best fallback. This is required by Exp 3 (0.474), Exp 10 (0.485) and Exp 12's C3 cell.\n\n(2) When a count winner exists, weights are inverse in stated validity, measured against the CHANCE baseline (s_j = (v_j - 0.5)/(v_max - 0.5)) and strongly SATURATING (f(s) = s^omega with omega ~ 0.3-0.5), so that any expert appreciably above chance is heard as merely restating the obvious and stars become counter-indicative (w < 0). This is the pi_6-like ingredient that Exps 1, 2, 7, 8, 9, 10 demand \u2014 the observed inversions are if anything STRONGER than pi_6's nominal predictions (Exp 1 observed 0.698 vs pi_6 0.529; Exp 2 observed 0.145 vs pi_6 0.256), so allow lambda large enough to push stars clearly negative and to leave the panel's least-accurate discriminating voice at or above full weight.\n\n(3) THE NEW COMMITMENT (what distinguishes it from both pi_6 and pi_7): the redundancy construal is graded by the size of the SAME-SIDE chorus and saturates SLOWLY, e.g. lambda_side = lambda * m_side / (m_side + kappa) with kappa ~ 1.5-3 (not pi_6's kappa_d ~ 0.1, and not pi_7's immediate-saturation same-side rule). Consequences that must fall out: a 2-expert mid majority opposed by a lone floor voice is only weakly discounted, so the display lands near chance (Exp 12-C4, observed ~0.48); a 3-expert mid majority opposed by the same lone floor voice is discounted much harder and is actively REJECTED (Exp 2-K4, observed 0.145); a lone star opposing a mid pair is still counter-indicative because its discount comes from the absolute/chance-anchored validity term rather than from the chorus (Exp 7's coherent cells, adherence ~0.15, and Exp 1's positive d3-minus-c2 contrast).\n\n(4) Confidence compression by the total number of discriminating cues (D^gamma) plus an attention lapse epsilon, so that dense unanimous displays stay high (Exps 5, 6 pooled tally adherence 0.75-0.79) and Exp 8's d2 term stays near zero. No capacity/panel-size collapse term should be included: Exp 8 (n=7) and Exp 10 (n=8) both show full-strength inversion, so the n-dependent washout in both current theories is unsupported.\n\nCross-design signature to state explicitly: exact chance on tally ties at any validity composition; reversal of star-backed majorities at every panel size; the least-accurate discriminating voice beating a majority only when that majority has three or more corroborators, and merely drawing level with a two-member majority; positive lone-star-minus-lone-floor contrasts; and near-ceiling adherence to unopposed low-validity majorities."
}
```

## Usage

```json
{
  "input_tokens": 33934,
  "output_tokens": 22968
}
```
