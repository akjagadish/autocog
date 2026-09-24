# Round 8 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_10` — SURVIVED ✓

**Description:** **Chorus-Threshold Novelty Weighting, Balanced-Gradient variant (CTN-BG).**

People told the accuracies of a panel of binary expert ratings do not weight a rating by its accuracy; they weight it by how much *news* it carries, and news is loud only when it stands against a genuine CHORUS. Six commitments, all smooth and continuous (no mute band, no solo release, no discrete count switch).

**(1) The count comparator is a parameter-free gate.** Before any expert identity is consulted, an unweighted feature-wise comparison asks: does one product win on strictly more attributes? If not — equal win counts, including wholly non-discriminating displays — the display is non-diagnostic and the subject flips a mental coin: exactly 0.5, with NO validity tie-break and NO take-the-best fallback, at every panel composition and size.

**(2) Seniority is panel-relative; novelty is a smooth convex, monotonically DECREASING function of it, and the descent is STEEP.** s_j = (v_j − v_min)/(v_max − v_min); nov_j = (1 − s_j)^kappa with kappa ≈ 14–16.5. Nothing is muted by fiat and nothing is released by fiat, but the curve is sharp: only the panel's genuinely idiosyncratic BOTTOM voice speaks with nov ≈ 1; the second-from-bottom voice retains a real but small share; a mid voice essentially nothing; a star nothing at all. Novelty is never negative — a star is ignored, never counted against its own side.

**(3) THE CORE CLAIM — a novel voice is loud only in proportion to the CHORUS it speaks into, and a chorus is measured in *mid-expert equivalents*.** Each other discriminating voice contributes a subjective unit u_i = (1 + tau·s_i)/(1 + tau/2): a mid-panel voice counts as exactly one unit, a star ≈ 1.4, a bottom voice ≈ 0.6. The chorus a cue j faces is Q_j = Σ_{opposing} u_i + lam_own·Σ_{own side, i≠j} u_i — voices it contradicts counted in full, voices it merely stands beside echoed at a discount. Loudness is a smooth logistic of that chorus, g_j = Gmax·σ((Q_j − q0)/wid), with a narrow but continuous threshold at roughly two-and-a-half mid-equivalents. BOTH how many voices and how senior they are matter, in one scalar: two mid voices are not yet a chorus, two stars just barely are, three upper voices fully are.

**(4) The ungated panel gradient is THIN and only MILDLY convex — a bare ordinal pull that does not, by itself, give the panel's bottom voice an edge over a pair of mid voices.** Every discriminating cue contributes r_j = eta + c·(1 − s_j)^nu with small eta, small c and a moderate exponent nu ≈ 1.7–2.05. The scale is small enough that all-upper displays drift near chance at any count margin, and the exponent is flat enough that a single floor cue's residual is worth about the same as two mid cues' residuals — so a floor voice opposed by a mid PAIR is a dead heat (it needs the chorus gate, not the gradient, to win), while a floor voice opposed by a three-cue upper bloc is carried past chance by the gate.

**(5) Integration is Luce-like and saturating.** w_j = r_j + nov_j·g_j; E = Σ dir_j w_j; z = beta·E/((c_n + Σ_j w_j)·D^gamma). Evidence is judged relative to the total weight of evidence on the table plus a constant of indecision, and is diluted by the number of mutually discriminating attributes with gamma ≈ 1, so a lone cue and an unopposed pair of the same tier feel about equally compelling. An attention lapse epsilon mixes the result with a coin flip.

**(6) Capacity is a near-total collapse of the chorus construal past ~7 voices, with only mild extra noise.** For n > 7 the amplification essentially vanishes (Gmax × (1 − psi·(n−7))) while integration noise grows only slightly, so eight-expert star-backed majorities sit near chance rather than being reversed, and eight-expert plain tally adherence survives on the residual gradient.

**Cross-design signature:** exact chance on tally ties at any composition; near-chance on displays whose every discriminating voice is upper-band, at every margin; moderate (not ceiling) adherence to majorities containing no bottom voice; a bottom voice facing two MID voices lands at chance, facing two STARS reverses substantially, facing a three-cue upper bloc reverses to ≈.12–.18; near-ceiling following of majorities that contain the bottom voice; attenuated integration on eight-expert panels.

**Rationale:** **Minimal diff: four parameter ranges changed (c_res, nu, kappa, c_n) plus the matching in-code defaults; every equation, name and structural line is byte-identical to the accepted iter-4 base.**

**1. Primary fix — joint (c_res, nu) move exactly as the critic specified: c_res 0.14–0.20 → 0.10–0.14 AND nu 2.0–2.8 → 1.70–2.05.** This is not a third uniform scale cut; the scale comes down while the exponent comes *back down* too, which is the unique combination that shrinks the FLOOR residual without further shrinking MID residuals. Check on the Exp-10 panel (v_floor=.52, mids .70/.74, v_max=.95, so s_floor=0, s≈.419/.512): current (c_res=.17, nu=2.4) gives floor r = .04+.17 = .210 versus mids .04+.17·.581^2.4 = .0866 and .04+.17·.488^2.4 = .0719, sum .159 — a +.051 floor edge that drives C4 to ≈.60. Proposed (c_res=.12, nu=1.88): floor r = .160, mids .04+.12·.581^1.88 = .0838 and .04+.12·.488^1.88 = .0722, sum .156 — edge ≈ +.004, i.e. a dead heat, so C4 should fall to ≈.50–.53 against a real ≈.478 and the Exp-10 metric from +0.104 into the target [−0.06, +0.04]. Because the sub-threshold gate leak on that 1-vs-2 display is small and unchanged (Q for the floor cue ≈ 1.95 mid-equivalents, far below q0≈2.45), no gate knob is touched.

**2. Why this does NOT re-inflate Exp 2 or Exp 13.** On Exp 2's 4-vs-1 conflict trial the majority is made of non-floor voices whose s sits in the .1–.6 band; the flatter exponent hands back to those mid/low-mid cues almost exactly what the smaller scale takes away (e.g. s=.139: .152 → .137; s=.389: .111 → .099; s=.556: .096 → .087), so the B-side residual mass changes by only ≈4%. Exp 13's all-upper displays are likewise near-neutral (their cues sit at s≥.42, where the two curves cross within .003). So Exp 2 should stay ≈.73–.75 and Exp 13 ≈.54–.55 rather than regressing, while Exp 10 gains ≈.13.

**3. Secondary — Exp 12-K4 is over-reversed (.094 vs real .145) and this edit corrects it in the right direction.** The floor cue's absolute weight drops from .210+gate to .160+gate while the three opposing upper residuals barely move, so net evidence for the floor shrinks slightly and K4 should rise toward ≈.11–.15, inside the guard-rail [.11,.18]. I deliberately did NOT raise gmax (the critic made it conditional on Exp 6 *and* on K4 not falling below .10 — K4 is already low, so raising gmax is the wrong sign there).

**4. Compensation — c_n 2.9–3.5 → 2.70–3.05, as instructed.** Cutting the floor residual shaves the floor-containing, under-driven cells (Exp 4 .709 vs .755, Exp 9 .379 vs .405, Exp 11 .630 vs .698, Exp 6 −0.810 vs −0.938). Lowering the constant of indecision raises z most where W_tot is small — exactly the sparse eight-expert tally displays of Exp 4/9 and the low-mass cells of Exp 6/11 — and least where W_tot is already large (Exp 2's dense 4-vs-1, whose residual mass I showed is unchanged), so it is the compensator with the smallest Exp-2 side effect. The move is deliberately modest (≈8% on the denominator, ≈+.02–.03 on Exp 4) so it cannot swamp the Exp-10 gain.

**5. kappa held, per instruction: 12.5–17.0 → 14.0–16.5.** The critic asked me not to push kappa further and to hold it at 14–16; I narrowed the range onto that band rather than raising its top, which keeps the second-from-bottom voice (the cue Exp 6 and Exp 11 depend on) from being suppressed any further while removing the low tail that let some subjects over-weight mid voices in Exp 2.

**6. Untouched, per instruction:** eta (holds the Exp 15/16 solo-vs-flat sign), gamma, epsilon, psi (pins Exp 8 at data), gmax, q0, wid, tau, lam_own, chi, and the entire parameter-free count gate — which continues to deliver exact chance on every tally tie, the one experiment-invariant prediction no competitor family makes as cleanly.

**Parameters:**
  - `eta`: `[0.030, 0.048]`
  - `c_res`: `[0.10, 0.14]`
  - `nu`: `[1.70, 2.05]`
  - `kappa`: `[14.0, 16.5]`
  - `tau`: `[1.0, 1.5]`
  - `lam_own`: `[0.50, 0.70]`
  - `gmax`: `[1.70, 2.10]`
  - `q0`: `[2.38, 2.52]`
  - `wid`: `[0.11, 0.17]`
  - `c_n`: `[2.70, 3.05]`
  - `beta`: `[34, 48]`
  - `gamma`: `[0.88, 1.05]`
  - `epsilon`: `[0.09, 0.15]`
  - `psi`: `[0.80, 0.95]`
  - `chi`: `[0.02, 0.08]`
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


### slot 2 — `pi_9` — KILLED ✗

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


## Replacement

### `pi_11` → slot 2 (via `new_theory`)

**Description:** **Flat-Maverick Inverse-Seniority Integration with a Count-Confidence Gate and Mild Count Dilution (FM-ISN-CC5).**

When people are told the accuracies of a panel of binary raters, they do not weight a rating by its accuracy and they do not weight it by the crowd it happens to face. They weight it by one thing only: *how far down the displayed panel's accuracy range that voice sits*. Five commitments.

**(1) CONTEXT-FREE INVERSE-SENIORITY WEIGHT (the decisive claim), NEARLY CATEGORICAL about the true panel floor.** Seniority is panel-relative: s_j = (v_j − v_min)/(v_max − v_min) computed over the panel shown in the instructions. Each discriminating cue gets w_j = eta + (1 − s_j)^kappa with kappa ≈ 15–17 and a very thin flat floor eta ≈ 0.008–0.018. Only the panel's genuinely bottom voice speaks with full loudness; the SECOND-from-bottom voice already retains only about a quarter of it, and mid and star voices collapse onto the near-zero flat residual and are effectively interchangeable and mute. Crucially the weight is **completely independent of context**: it does not depend on how many voices oppose the cue, how senior those opponents are, whether the cue stands alone, or whether it sits inside a unanimous bloc. There is NO chorus amplification of any kind. A floor voice opposed by a pair, a floor voice opposed by a trio, and a floor voice inside a unanimous bloc are all read with exactly the same loudness — their observed rates differ only through the ordinary arithmetic of the weighted balance.

**(2) NORMALISED (LUCE-LIKE), SATURATING INTEGRATION WITH MILD COUNT DILUTION.** Evidence is the signed weighted balance E = Σ_j dir_j w_j, and confidence is that balance measured against the total subjective weight actually on the table plus a constant of indecision, further diluted by the number of mutually discriminating attributes: z = beta·E/((c_n + Σ_j w_j)·D^gamma) with gamma ≈ 0.34–0.46. Because a display made entirely of upper-band voices puts almost no subjective weight on the table, such displays sit near chance *at every count margin*, and the mild dilution makes a lone thin voice at least as decisive as an unopposed thin pair. A display containing the panel's floor voice immediately supplies enough weight to drive the choice near ceiling in the floor's direction.

**(3) THE COUNT COMPARATOR GATES CONFIDENCE, NOT DIRECTION.** An unweighted feature-wise count is computed first. It never supplies a direction of its own and never acts as a tie-break by validity rank. What it does is license confidence: when one option wins on strictly more attributes the weighted reading is taken at full gain; when the counts are equal (including wholly non-discriminating displays) the very same weighted reading is taken at a moderately reduced gain rho_tie ≈ 0.60–0.78, so tie displays hover close to chance — slightly *anti*-validity, never pro-validity.

**(4) CAPACITY IS AT MOST A WHISPER.** Holding a whole validity ordering in mind gets marginally harder past about seven voices; the integration slope is divided by 1 + chi·(n − 7) with chi ≈ 0–0.10, i.e. essentially absent. Eight-expert panels retain their tally adherence, and star-backed dense majorities stay at chance for the independent reason that their weights are all near eta.

**(5) LAPSE.** A small attention lapse epsilon ≈ 0.05–0.10 mixes the result with a coin flip.

**Cross-design signature:** near-chance (slightly anti-validity) on tally ties at any composition; chance on displays whose every discriminating voice is upper-band, at any margin, with solo ≥ multi-cue upper; a null on the two contrasts that falsify chorus theories — an opposed floor voice and a floor voice inside a unanimous bloc come out at the same rate, and titrating the seniority of an opposing pair produces a flat line; hard reversal of upper-backed majorities by a lone floor voice; near-ceiling following of majorities containing the floor voice; and only MODERATE anti-validity when the weak side is carried merely by the second-from-bottom voice rather than by the true floor.

**Rationale:** Minimal-diff edit on the ACCEPTED iter-4 base (loss 0.0474). No mechanism, no equation, no parameter name changed — only sampling bands (and the matching internal defaults) were re-tuned exactly as the iteration-5 critic prescribed, and the rejected iter-5 rho_tie push is neither repeated nor fully reverted.

(1) PRIMARY, untried global-extremity knob: epsilon [0.08,0.15] -> [0.05,0.10]. The residual signs on the iter-4 base now align for more extremity on five experiments (Exp4 -0.026, Exp5 +0.057, Exp6 +0.030, Exp11 -0.037, Exp12 +0.035) against only three that want less (Exp3, Exp7, Exp2). Shrinking the lapse scales every deviation from .5 by ~1.05x: expected Exp5 -0.707 -> ~-0.74 (obs -0.763), Exp12 0.180 -> ~0.165 (obs 0.145), Exp6 -0.908 -> ~-0.94 (obs -0.938), Exp4 0.729 -> ~0.740 (obs 0.755), Exp11 0.661 -> ~0.668, at a cost of ~+0.01 each on Exp3/Exp7/Exp2 and <0.002 on the near-chance nulls (Exp8/13/14/15/17/18), whose predictions are within a hair of 0.5 and therefore essentially immune to the lapse. Net ledger clearly positive. The step is deliberately modest (floor 0.05) because the Exp5/Exp6/Exp12 gains saturate while the Exp3/Exp7/Exp2 costs keep accruing.

(2) Anti-oscillation on rho_tie: set to [0.60,0.78] (centre ~0.69), strictly between iter-4's [0.55,0.75] and the rejected iter-5's [0.74,0.88]. This retains the small, real Exp1 (0.487 -> ~0.478, obs 0.474) and Exp9 (0.397 -> ~0.402, obs 0.405) gains while halving the collateral cost that the full push imposed on Exp7's two tied cells and on Exp10's neighbours. Exp10 (~+0.05) is accepted as intrinsic: its C4-minus-C3 gap is driven by the tie gain and the D-dilution asymmetry, neither of which can be closed without wrecking Exp1 or Exp12.

(3) SECONDARY, near-free: chi [0.05,0.25] -> [0.00,0.10]. Only Exp4 and Exp8 have n=8; Exp4 wants a steeper slope (0.729 vs 0.755) and Exp8's weights are all ~eta so it moves <0.005 and stays at chance. This also brings the model closer to the arbiter's point (5) that capacity should be at most a small slope reduction or absent entirely.

(4) VARIANCE CONTROL: beta -> [8.3,9.2], c_n -> [1.2,1.5], kappa -> [15,17], eta -> [0.008,0.018], gamma -> [0.34,0.46], all narrowed around their current centres. Structurally untouched metrics (Exp12/13/5/4) drifted 0.02-0.04 between iterations 4 and 5, i.e. parameter jitter is now comparable to the residuals being chased and was plausibly what decided the last gate outcome. Narrowing changes no central prediction but should make this comparison against 0.0474 informative rather than noise-dominated.

(5) Explicitly NOT done, per the critic: no further kappa or beta push (that programme is terminal — the second-from-bottom residual is gone), no gamma increase to chase Exp16 (its cells all sit within a few points of chance, its observed +0.052 is inside its between-subject noise, and extra D-dilution pushes Exp12 back toward chance), and no re-opening of the graded residual lam*(1-s), which re-inflates the mid tier in Exp7/Exp2. Every change remains context-free, so the family-diagnostic nulls that are this candidate's principal asset — no chorus term, no validity-rank tie-break, direction never supplied by the count comparator — survive verbatim (Exp17, Exp18, Exp15, Exp14, Exp8).

**Parameters:**
  - `eta`: `[0.008, 0.018]`
  - `kappa`: `[15.0, 17.0]`
  - `beta`: `[8.3, 9.2]`
  - `c_n`: `[1.2, 1.5]`
  - `gamma`: `[0.34, 0.46]`
  - `rho_tie`: `[0.60, 0.78]`
  - `epsilon`: `[0.05, 0.10]`
  - `chi`: `[0.00, 0.10]`
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
