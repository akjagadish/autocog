# Round 6 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_8` — SURVIVED ✓

**Description:** **Consensus-Graded Validity Inversion with a Non-Negative News Floor (CG-VI-2).**

People told the stated accuracies of a panel of binary expert ratings do not weight experts *by* accuracy; they weight them by how much *news* a rating carries. Six commitments.

**(1) The count comparator is a gate, not a weight.** Before any expert identity is consulted, an unweighted feature-wise comparison asks a purely ordinal question: does one product win on strictly more attributes? If not — equal win counts, including wholly non-discriminating displays — the display is construed as non-diagnostic and the subject flips a mental coin: exactly 0.5, with NO validity tie-break and NO take-the-best fallback. Parameter-free at every panel composition (Exp 1, Exp 3's tied cells, Exp 7's tied cells, Exp 10's C3 cell).

**(2) When a count winner exists, weights are inverse in stated validity, measured against the panel's least-accurate voice and strongly SATURATING.** s_j = (v_j - v_min)/(v_max - v_min); news value w_j = 1 - lambda_eff * s_j^omega with omega well below 1 (~0.3-0.4). Because the transform is strongly concave, *any* expert appreciably above the panel floor is already heard as "one of the accurate ones" — merely restating the obvious consensus quality of the product — while the panel's genuinely idiosyncratic bottom voice is heard as private, surprising information and keeps full credit.

**(3) THE NEW COMMITMENT — news value is bounded below at ZERO: a redundant expert is IGNORED, never counted against his own side.** w_j = max(0, 1 - lambda_eff*s_j^omega). Redundancy is a reason to stop listening, not a reason to believe the opposite. This is what generates the signature pattern that no signed-inversion model produces: *the side holding the panel's least-accurate voice wins, and when neither side holds it the plain tally decides*. A majority made of stars plus the maverick is followed near ceiling (the stars are mute, the maverick speaks for the majority), whereas the numerically identical majority made of mid-panel experts is rejected when the maverick sits in the minority. It also stops dense, unanimous, high-validity displays from being actively rejected (they merely lose force and drift toward chance), which preserves plain-tally adherence.

**(4) The redundancy construal is graded by how big and how lopsided the visible consensus is, and it saturates sharply.** The subjective chorus is read off the display as CH = max(m_A, m_B) + |m_A - m_B| - 2, i.e. the size of the larger bloc *plus* the net excess of same-direction voices. lambda_eff = lambda * CH^q/(CH^q + kappa) with q ~ 3.4-4.5 and kappa ~ 1.4-2.0. Hence: a solitary discriminating voice (CH = 0) is heard at full face value; a 2-vs-1 display (CH = 1) is only weakly discounted, so a mid pair opposed by a lone floor voice lands essentially at chance; a 3-vs-2, 2-vs-0 or 3-vs-1 display (CH >= 2) is discounted almost maximally, so the floor voice takes over; and everything denser is saturated.

**(5) Confidence is compressed by the sheer number of discriminating cues (D^gamma) and capped by an attention lapse epsilon.** Evidence spread across many mutually contradicting attributes is subjectively weaker than the same net weight carried by a sparse clean split.

**(6) The inversion needs the whole validity ordering in mind, so it degrades past working-memory span (~7 voices).** For n > 7 the discount shrinks (lambda scaled by 1 - psi*(n-7)) and integration gets noisier (divisor multiplied by 1 + chi*(n-7)), so eight-expert star-backed majorities sit at chance rather than being reversed, while plain tally adherence there is preserved.

**Cross-design signature:** exact chance on tally ties at any validity composition; the panel's least-accurate discriminating voice decides whenever it is present on exactly one side, at every panel size up to seven; near-chance for 2-vs-1 mid-versus-floor displays but decisive rejection of 3-cue majorities opposed by the same floor voice; near-ceiling following of majorities that themselves contain the floor voice; lone voices taken at face value; unanimous all-star displays drifting to chance rather than reversing; and markedly attenuated inversion on eight-expert panels.

**Rationale:** **Minimal diff: three changed lines plus a retune, everything else (gate, floor-anchored s, omega, D^gamma, lapse, n>7 capacity term) verbatim.**

(1) *Chorus argument blends bloc size with net margin* — exactly the critic's point 1. `M` is replaced by `CH = max(mA,mB) + |mA-mB| - 2`. With the retuned Hill (lam≈1.75, kappa≈1.7, q≈4) this gives lam_eff ≈ 0.65 at 2-vs-1 (CH=1, keeps Exp 10 C4 at chance), ≈1.58 at 3-vs-2 / 2-vs-0 / (CH=2) and ≈1.7+ at 3-vs-1 and denser. A lone cue (CH=0) is still taken at face value. This is what the critic asked for and it separates Exp 10 C4 from Exp 11 e4/a1 and Exp 12 K4, which net margin alone could not.

(2) *News value is floored at zero* (`w = max(0, 1 - lam_eff*f)`) — the one genuinely new line, forced by Exp 11 arithmetic. Exp 11's observed cell pattern is "whichever side holds the floor expert (.68) wins; if neither does, the tally wins": e4 high (majority = star+star+floor), a1 low (floor in the minority), c2 low, d3 high. With SIGNED inversion the stars inside the e4 majority contribute large negative weight and cancel the floor voice, so e4 can never exceed ~0.55 no matter what lambda is — that is precisely why iter 1 scored 0.493 and pi_6 0.529. Muting redundant experts instead of inverting them gives e4 ≈ 0.83, 1-a1 ≈ 0.83, d3-c2 ≈ 0.38 → **Exp 11 ≈ 0.68 vs observed 0.698** (iter 1: 0.493, |err| 0.205 → ~0.02). It costs nothing elsewhere because every reversal in the corpus is driven by asymmetry (loser holds the maverick, winner is all stars), not by literal negative evidence: recomputing by hand gives Exp 5 d1 = -0.76 (obs -0.763), Exp 7 = 0.66 (obs 0.661), Exp 9 = 0.40 (obs 0.405), Exp 12 ≈ 0.10 (obs 0.145), Exp 6 ≈ -0.99 (obs -0.9375, better than iter 1's -0.829 as the critic requested), Exp 10 ≈ 0.00 (obs -0.023), Exp 2 ≈ 0.72-0.74 (obs 0.654; iter 1 was 0.568, so the residual flips sign but shrinks). It also removes iter 1's pathology of actively rejecting unanimous high-validity displays, which should lift Exp 3 toward its observed 0.794.

(3) *Retune per the critic's point 3, with one sign correction.* beta widened upward (9.5-12.5) to sharpen Exp 6's d1; the asymptote is capped so lam_eff never exceeds ~1.8, which keeps the 4-vs-1 lone-star cell of Exp 2 from flipping wholesale. On `psi` I go the other way from the critic (0.42-0.58 rather than 0.25-0.40): with zero-floored weights the n=8 star-backed majorities of Exp 8 are *rejected* at ~0.37 if the discount is left near full strength, so slightly MORE capacity attenuation is what puts Exp 8 at ~0.49 (obs 0.485) — the critic's direction was derived from the old signed-weight arithmetic and reverses under the new floor.

**Parameters:**
  - `lam`: `[1.60, 1.90]`
  - `omega`: `[0.28, 0.40]`
  - `q_m`: `[3.40, 4.50]`
  - `kappa_m`: `[1.40, 2.00]`
  - `beta`: `[9.5, 12.5]`
  - `gamma`: `[0.80, 1.00]`
  - `psi`: `[0.42, 0.58]`
  - `chi`: `[0.60, 1.10]`
  - `epsilon`: `[0.08, 0.18]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
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
```

**`policy(probs)`:**
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


### slot 2 — `pi_6` — KILLED ✗

**Description:** **Capacity-Gated Strongly-Saturating Redundancy Validity-Inversion Tallying (GVIT-CAP-S4).**

Five commitments describe how people combine binary expert ratings whose stated validities they have been told.

**(1) The count comparator is a gate, not a weight.** Before any evaluation of *whose* rating is whose, the subject asks a purely ordinal question: does one product win on more attributes than the other? This is answered with unweighted feature-wise comparisons. If the answer is "no" (equal numbers of winning attributes, including displays where nothing discriminates), the display is construed as *non-diagnostic*, no candidate is generated, and the subject flips a mental coin. Stated validities are **never** used as a tie-break. This is a hard, parameter-free prediction: on tally-tied, side-counterbalanced profiles adherence to the most-valid discriminating cue is exactly chance whatever the validity vector.

**(2) When a majority exists, integration uses inverse-informativeness weights.** The subjective news value of a rating is a smooth, panel-relative *decreasing* function of stated validity: s_j = (v_j - v_min)/(v_max - v_min) within the displayed panel, w_j = 1 - lambda_eff*f(s_j). A near-perfect expert is heard as merely restating the obvious consensus quality of the product, so his rating adds almost nothing (and, when lambda_eff*f > 1, is mildly counter-indicative); a barely-diagnostic expert is surprising and is treated as genuine private information carrying full weight. Evidence is the single scalar E = sum_j w_j*sign(a_j - b_j).

**(3) The redundancy construal saturates VERY STRONGLY just above the panel floor.** Redundancy is not judged on a linear scale of stated accuracy: "being one of the accurate ones" is a nearly categorical construal. An expert only has to be slightly above the panel's least-accurate voice before he is already heard as echoing the consensus. Formally the discount is a strongly concave function of panel-relative validity, f(s) = s^omega with omega well below 1 (~0.26-0.42), so the discount rises steeply just above the panel floor and then flattens. Two consequences follow that neither a linear nor a mildly concave inversion produces. (i) A majority carried by *middling* experts (the 2nd-5th most accurate voices) carries almost no subjective evidence: mid-panel support is discounted nearly as heavily as star support, so a mid-backed majority facing a lone star is followed only weakly, close to chance. (ii) Conversely, when the count loser is backed by the panel's genuinely idiosyncratic bottom voices, the anti-count evidence is *strengthened*, sharpening the reversal. So the model separates three displays that a linear inversion renders nearly identical: majority-by-mid-experts (followed barely above chance), majority-by-bottom-voices (followed near ceiling), and star-backed majorities (reversed toward floor).

**(4) Redundancy is comparative; evidence spread over many contradicting cues is compressed.** An expert is discounted only because other experts are heard as echoing him, so the inversion scales with how many voices actually speak on the trial: lambda_eff = lambda_n*(D-1)/((D-1)+kappa_D), where D is the number of discriminating cues; when exactly one expert differentiates the products there is no chorus and his rating is taken at face value. Confidence in E is then compressed by the number of mutually contradicting cues, D^gamma.

**(5) The inversion is a capacity-limited construal: it collapses when the panel exceeds working-memory span.** Discounting the star requires holding the validity ordering of the whole panel in mind, which is possible only up to roughly seven experts. Beyond capacity the ordering blurs, so the redundancy construal washes out toward equal weighting (lambda_n = lambda*(1 - psi*o)) and integration becomes noisier (slope divided by 1 + chi*o), where o = logistic((n - 7.5)/0.2). Finally an attention lapse epsilon replaces the decision by a coin flip.

Cross-design signature: exact chance on tally-tied profiles; strong systematic reversal of star-backed majorities on panels of <=7 experts; near-chance for the same star-backed profiles on panels of 8+ experts, with preserved tally adherence there for broad-mixture majorities; near-ceiling following of a lone discriminating expert; only weakly above-chance following of mid-panel-backed majorities that oppose a lone star; and markedly attenuated confidence whenever the weighted evidence is spread across many mutually contradicting cues.

**Rationale:** MINIMAL DIFF from the running-best iter-6 base: the `predict`/`policy` source is re-emitted verbatim except for two cosmetic `_get` defaults (psi 0.48 -> 0.40, epsilon unchanged at 0.17) that the sampler always overrides. No mechanism, equation, gate, weight function, or name changed. Still a GVIT instantiation: one smooth panel-relative inverse-informativeness weight w_j = 1 - lam_eff*s_j^omega, one scalar E, D^gamma dilution, capacity modulation, lapse. No count-margin contrarian term, no validity tie-break on tally-tied displays, no graded panel-size term.

I execute EXACTLY the critic's pre-specified fallback, which was triggered by iteration 8's trip conditions (Exp4 0.712 < 0.72 and Exp6 -0.905 > -0.90 in magnitude terms):

1) KEEP THE psi MOVE, DROP THE epsilon CENTRE MOVE. psi [0.40, 0.56] -> [0.32, 0.48] (centre 0.40). This is the one knob with a twice-measured, correctly-signed, near-orthogonal gradient: dExp8/dpsi ~ +0.55/unit vs dExp4/dpsi ~ +0.10/unit, so lowering psi pulls Exp8 back from 0.557 toward ~0.50-0.52 (real 0.485 — Exp8 was the LARGEST residual of the iter-6 base at sq ~0.0052) while barely touching Exp4 (same n=8) and leaving Exps 1,2,3,5,6,7 (all n<=7, where the capacity indicator o~0) mathematically untouched. epsilon's CENTRE returns to the iter-6 value 0.17; the +0.02 centre raise is what killed iteration 8 by compressing spread on Exps 5 and 6 and costing Exp4's mean.

2) FIX VARIANCE WITHOUT MOVING MEANS — now the binding term. Point-estimate residuals are already small; the loss is being paid on between-subject dispersion, worst on Exp5 (sim ~0.010 vs real 0.0200) and Exp6 (sim ~0.044 vs real 0.0693). I widen SYMMETRICALLY about the unchanged iter-6 centres only the two knobs that scale the reversal magnitude |E| and the lapse: epsilon [0.09,0.25] -> [0.07,0.27] (centre still 0.17) and lam [1.18,1.72] -> [1.10,1.80] (centre still 1.45). Both spread the per-subject reversal magnitude across draws while leaving the expected value approximately fixed, which is exactly the mechanism the two difference metrics (Exp5, Exp6) need to reproduce their large between-subject variance. Exp3/Exp4 variances should rise toward 0.004-0.006 as a side benefit; Exp1 is driven only by the parameter-free tie gate and is untouched by both widenings, so its slight over-dispersion is not aggravated.

3) FREEZE EVERYTHING ELSE, per the critic. gamma [0.74,1.06] centre 0.90 and beta [5.8,7.6] centre 6.7 are restored/left at iter-6 values (the coupled gamma/beta rotation was REJECTED at iter 7 and must not be repeated or hard-reversed). omega stays at [0.26,0.42] (pushed four times, already overshot its guard rails, no longer the arbiter). kappa_d, chi and the capacity constants are unchanged; chi is explicitly avoided because it moves Exp8 and Exp4 in the same direction and Exp4 cannot afford it.

Expected envelope: Exp1 ~0.47-0.51, Exp2 ~0.69-0.72, Exp3 ~0.76-0.79, Exp4 ~0.73-0.75, Exp5 ~-0.75 to -0.78 with var >= 0.010, Exp6 ~-0.95 to -0.97 with var >= 0.045, Exp7 ~0.66-0.69, Exp8 ~0.50-0.53. The psi step alone recovers most of the Exp8 residual that dominated the iter-6 base (~0.0052 of squared error), and the two symmetric widenings add variance credit on Exps 3/4/5/6 without shifting any mean, so the combined edit should land strictly below the 0.0519 floor rather than trading a mean gain for a variance loss as iteration 8 did.

**Parameters:**
  - `lam`: `[1.10, 1.80]`
  - `beta`: `[5.8, 7.6]`
  - `gamma`: `[0.74, 1.06]`
  - `kappa_d`: `[0.01, 0.45]`
  - `psi`: `[0.32, 0.48]`
  - `chi`: `[0.75, 1.15]`
  - `epsilon`: `[0.07, 0.27]`
  - `omega`: `[0.26, 0.42]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
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
```

**`policy(probs)`:**
```python
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
```


## Replacement

### `pi_9` → slot 2 (via `new_theory`)

**Description:** **Mute-Band Maverick Arbitration with a HARD (near-step) Band and a Stronger Capacity Noise Term (MBM-PG7).**

People who are told the accuracies of a panel of binary expert ratings do not weight experts *by* accuracy; they weight them by how much *news* a rating carries, and news is a steeply thresholded, panel-relative quantity. Six commitments.

**(1) The count comparator is a parameter-free gate.** Before any expert identity is consulted, an unweighted feature-wise comparison asks a purely ordinal question: does one product win on strictly more attributes? If not — equal win counts, including wholly non-discriminating displays — the display is construed as non-diagnostic and the subject flips a mental coin: exactly 0.5, with NO validity tie-break and NO take-the-best fallback, at every panel composition and size.

**(2) HARD, NEAR-STEP MUTE BAND, CHORUS-INDEPENDENT.** Seniority is read off the *displayed panel*: s_j = (v_j − v_min)/(v_max − v_min). Only voices lying within a narrow band theta (≈12-20% of the panel span) above the panel floor are heard as carrying private, surprising information, and *within* that band the listening function is now strongly convex: m_j = max(0, 1 − s_j/theta)^rho with rho ≈ 1.7-2.5. The maverick construal is therefore close to categorical — only the panel's genuinely idiosyncratic BOTTOM voice speaks with full force; a voice that merely sits low-ish (a fifth to a half of the way into the band) is already nearly as mute as a star. Muting is unconditional: a lone star cue is as mute as a star inside a bloc.

**(3) GRADED, STRONGLY ASYMMETRIC CHORUS AMPLIFICATION WITH A LATE, STEEP KNEE — and the post-knee amplification is DEEP.** A maverick's voice is loud only in proportion to the redundancy it is heard against, and redundancy is *asymmetric*: the bloc a cue CONTRADICTS is the chorus proper, while the bloc a cue sits INSIDE echoes it only weakly and therefore counts at a discount lam_own. For cue j: C_j = (m_opp − 1) + lam_own·(m_own − 1) when the cue is opposed, and C_j = (m_own − 1) when the display is unanimous. Amplification G_j = 1 + alpha·C_j^p/(C_j^p + kappa) with a LATE, STEEP knee (p ≈ 6.4, kappa ≈ 205) and a LARGE post-knee asymptote (alpha ≈ 52). Because (2) is now near-categorical, this large amplification is applied essentially only to true floor voices: a partial, mid-band voice can no longer borrow a 20-30x chorus gain and masquerade as a maverick.

**(4) THE PANEL GRADIENT: the mute band is quiet, not empty, and the listening floor is THIN.** Muted experts do not vanish; each discriminating cue contributes a residual r_j = eta + c·(1 − s_j) — a thin flat listening floor plus an inverse-seniority gradient. Displays in which no voice is inside the mute band hover near chance at any count margin, while the preserved slope still generates the anti-validity contrasts on conflict displays.

**(5) DILUTION AND LAPSE.** Net evidence E = Σ_j dir_j·(G_j·m_j + r_j) is compressed by the number of mutually contradicting cues, D^gamma, and mixed with an attention lapse epsilon.

**(6) CAPACITY IS BOTH A FLATTENING AND A SUBSTANTIAL NOISE SOURCE.** Holding the whole seniority ordering in mind is possible only up to about seven voices. For n > 7 the amplification collapses toward 1 AND integration becomes markedly noisier (divisor × (1 + chi·(n−7)) with chi now appreciably larger), so eight-expert star-backed majorities sit at chance and eight-expert plain tally adherence is systematically LOWER than the six-expert equivalent even though the tally still governs direction.

**Cross-design signature:** exact chance on tally ties at any composition; chance on displays whose every discriminating voice is upper-band, at ANY count margin; only the panel's bottom voice (not the second- or third-from-bottom) can reverse a majority; a lone floor voice opposed by a mere PAIR lands at chance while the same voice opposed by a three-cue bloc reverses it to ≈.15; near-ceiling following of dense unanimous displays and of majorities containing the floor voice; and visibly attenuated, noisy integration on eight-expert panels.

**Rationale:** **Minimal diff: `predict`/`policy` are re-emitted verbatim from the ACCEPTED iter-5 base. Exactly TWO parameter ranges change — `rho` (mute-band decay exponent) [0.9,1.5]→[1.7,2.5] and `chi` (capacity noise, active ONLY for n>7) [0.16,0.42]→[0.26,0.52] — plus the matching `_g` defaults. No equations, no mechanism, no new terms.**

**Why I decline the critic's priority (1) (re-apply the seniority-graded own-echo with lam_own≈[0.38,0.50]).** Two reasons. (a) *It is arithmetically inert at that setting.* The own-echo enters only through the chorus base, and the knee is late and steep (p=6.4, kappa=205), so G is in its flat pre-knee region there: a star-flanked 3-bloc gives base 0.44·2=0.88 → G=1.11, versus the base model's uniform 0.365·2=0.73 → G=1.034. That is a +0.08 change in a cue weight of ≈3.3 — roughly 1/12 of the movement iter 6 produced (lam_own=0.625 → G=2.04). The gate cannot resolve that against simulation noise. (b) *Scaling iter 6's own cell table says the package is still net-negative.* iter 6's deltas were Exp11 +.074 but Exp3 −.039, Exp4 −.013, Exp10 −.030, Exp9 −.022, Exp12 −.012. Exp3 and Exp4 have by far the narrowest cross-theory spreads of any experiment (Exp4: .657–.857, Exp3: .638–.899), so per-experiment normalisation weights them ≈3-5x more than Exp6 or Exp5; at ANY lam_own large enough to recover the Exp11 gain, the Exp3/Exp4 re-inflation eats it. That is precisely the ledger the gate reported.

**Edit A — rho 0.9-1.5 → 1.7-2.5 (harden the mute band).** This is the arbiter's own commitment (2) taken seriously: the band should be *hard*. With rho≈2.1 a true floor voice (s=0) is completely unchanged (m=1), but *partial* mavericks — voices inside the band yet not at the floor — collapse. That is where the loop's remaining structured error lives. Hand-computed cells (I corrected an important bookkeeping point: Exp 7's cells 5 and 7 are 1-v-1 and 2-v-2, i.e. gate ties at exactly .5, so only cells 9 and 11 move — with that correction my midpoint arithmetic reproduces the base sim almost exactly: Exp7 .702 vs sim .699, Exp5 −.680 vs −.670, Exp6 −.943 vs −.929, Exp11 .632 vs .630, Exp9 .376 vs .387):
• **Exp 7** (residual +.038, second-narrowest-spread of the conflict metrics). f3 has s=.079 → m falls .443→.26. Cell 9 .896→.817, cell 11 .913→.902 ⇒ metric .702→.680 against real .661. The whole residual nearly closes.
• **Exp 11** (largest normalised residual, −.069). Its e4 leg is currently dragged down because the MINORITY cue f5 (s=.148, m=.047) borrows the 3-bloc amplification G=29.4 and contributes 3.32 — more than a genuine floor voice facing a pair. At rho=2.1 f5 is silenced (m≈.006), the minority mass drops 4.81→3.52, e4 .737→.77, and the composite goes .632→.643.
• **Exp 2** (+.062): f4 (s=.139) loses its partial maverick in the count-winning bloc ⇒ .693→.691 (small, right direction).
• **Exp 5, 8, 9, 10, 12, 13, 14 are structurally untouched** — every maverick they rely on sits at s=0 (m=1 for any rho) or is already outside the band. Exp 13's all-upper cells and Exp 14's C1/C3/C4/C7 contain no in-band voice at all, so the two metrics that are currently exact (Exp14 .490 vs .489) cannot move; Exp 12's lone floor voice (s=0) keeps m=1, so it will NOT sink further below its .12 rail.
• **Cost, bounded and accepted:** Exp 6's coherent cell 5 leans on f1 (s=.077, m .452→.27), so d1+d2 goes −.943→≈−.90 (real −.9375). Exp 6 has the widest cross-theory spread of any experiment (−1.00 to +.72), so a .04 raw move is the cheapest residual in the set — far cheaper than the Exp 7 and Exp 11 gains it buys.

**Edit B — chi 0.16-0.42 → 0.26-0.52 (capacity noise).** `chi` is multiplied by `over = max(0, n−7)` and is therefore *structurally inert on twelve of the fourteen experiments*; it touches ONLY the two eight-expert designs, and both want the same direction. Exp 4 is +.017 too high (and, because its cross-theory spread is the narrowest in the whole set, it is one of the most heavily weighted residuals) and Exp 8 is −.024, i.e. slightly further from chance than people are. Raising the divisor from 1.29 to ≈1.39 shrinks the eight-expert logit ≈8%: Exp 4 .772→≈.757 (real .755) and Exp 8 .461→≈.466 (real .485). This also buys insurance against Edit A leaking upward into Exp 4 via any weakly-muted cue in that panel. It cannot affect any six- or seven-expert design.

**Untouched, exactly as the critic asked:** the parameter-free tie gate (Exp 1's .026 gap is sampling noise on an exactly-0.5 predictor), theta, the knee (alpha, kappa, p_ch — G(C=1)≈1.25 preserved, which is what keeps Exp 10 c4 and Exp 14 C4 behaved), lam_own, the eta/c_res/beta triple (no oscillation), gamma, psi, epsilon, the residual form (no curvature exponent), and the policy.

**Expected net:** Exp 7 ≈+.022, Exp 11 ≈+.011, Exp 4 ≈+.015, Exp 8 ≈+.005, Exp 2 ≈+.002 against Exp 6 ≈−.04 (widest-spread, cheapest metric) — a reduction of roughly 10-12% in the normalised residual norm relative to the 0.0552 floor.

**Parameters:**
  - `theta`: `[0.12, 0.20]`
  - `rho`: `[1.7, 2.5]`
  - `eta`: `[0.18, 0.28]`
  - `c_res`: `[1.7, 2.3]`
  - `beta`: `[0.70, 0.90]`
  - `gamma`: `[0.88, 1.02]`
  - `alpha`: `[46, 58]`
  - `kappa`: `[180, 230]`
  - `p_ch`: `[6.1, 6.7]`
  - `lam_own`: `[0.28, 0.45]`
  - `psi`: `[0.10, 0.35]`
  - `chi`: `[0.26, 0.52]`
  - `epsilon`: `[0.04, 0.14]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
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
```

**`policy(probs)`:**
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
