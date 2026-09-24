# Round 6 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_7` — SURVIVED ✓

**Description:** PRIMACY-GATED DISTRUST TALLYING (PGDT), floored-gate edition with a rebalanced distrust share. Subjects scan the expert ratings in presentation order and attention is deployed through a steep convex primacy gate with a substantial nonzero floor: g_j = delta + (1-delta)*((n-j)/(n-1))^eta, so the first one or two experts carry the dominant share of attention but late experts retain a meaningful attention residue. Independently, the communicated validities are distrust-inverted exactly as in the shared kernel: subjects act as if each stated validity were the expert's error rate, inv_j = ((1-v_j) - c*min_k(1-v_k))^gamma / Z. The two kernels combine MULTIPLICATIVELY: w_j proportional to g_j * [(1-tau)/n + tau*inv_j], normalized. The refinement this iteration is a rebalanced distrust share: tau is lowered to [0.82, 0.92] so the flat component (1-tau)/n of the multiplicative bracket grows from ~1.5% to ~4-5% per cue. This directly feeds trusted high-validity LATE experts (whose bracket value is structurally pinned at the flat share regardless of gate shape), softens the distrust-driven allegiance contrasts that overshoot their targets, and pulls the effective weights off pure anti-validity where that overshoots. A modestly raised gate floor delta in [0.20, 0.45] compensates on decreasing-validity layouts by further concentrating the product weights onto the back-loaded low-validity experts, protecting the anti-validity regression. Evidence is integrated additively over all discriminating cues, d = sum_j w_j * sign(A_j - B_j), and choice is logistic in beta*d with a small uniform lapse epsilon.

**Rationale:** Minimal-diff edit per the critic's iter-3 directive: two parameter-range changes on the already-accepted floored-gate PGDT base, mechanism and code path identical. PRIMARY KNOB: tau lowered from [0.90, 0.97] to [0.82, 0.92] (inside the arbiter's prescribed [0.80, 0.97]). The iter-3 diagnosis identified a structural impasse the delta knob cannot break: for a high-validity late cue the multiplicative bracket is pinned at (1-tau)/n ~= 0.01-0.02 when tau >= 0.90, so no amount of gate flattening can rescue it -- this is exactly why Exp8 (0.278 vs 0.396), Exp6 (0.303 vs 0.374) and Exp1 (0.141 vs 0.208) barely moved across the iter-2 to iter-3 delta raise despite hand-checks predicting movement. Lowering tau grows the flat share to ~4-5%, directly feeding Exp8's final 0.90-validity expert, softening the over-committed distrust allegiances (Exp2 +0.162, Exp7 +0.083, Exp4 +0.056 -- the single largest remaining miss cluster), and pulling Exp1 back up off pure anti-validity. SECONDARY (compensating) KNOB: delta ceiling raised from 0.40 to 0.45 with floor 0.20, because the tau drop weakens the distrust kernel's dominance on decreasing-validity layouts and Exp3 (still +0.107 short of -0.576) needs the back-loading strengthened; the iter-2 to iter-3 trajectory verified delta's strong leverage there (-0.318 to -0.469). Cautions from the critique are honored: (i) Exp11 (0.234 vs 0.315) may be squeezed by the delta bump -- if it falls below ~0.20 the trade is accepted, since the combined Exp2+Exp8 misses (0.28) outweigh Exp11's (0.08) and every family theory except pi_1 undershoots Exp11 anyway; eta's ceiling stays at 6 so steep-gate subjects remain available to protect it. (ii) Exp12 is now matched (0.546 vs 0.540); if lowering tau pushes its slope above ~0.60, eta should be pulled toward 4 rather than reverting tau. (iii) Exp2 is treated as a stretch target: banking anything in the 0.20-0.28 range is a win given pi_3's 0.166 is the family best. gamma, c, beta, epsilon are frozen exactly as directed -- the distrust kernel's sharpness (gamma) is what the Exp3/Exp5/Exp7 regime depends on, and the accept gate protects against any regression. Both previous single-knob directives produced accepted candidates (loss 0.3072 -> 0.1413 -> 0.1175); this two-knob edit follows the same verified playbook, targeting the two coherent residual clusters (over-committed allegiances, starved trusted late cues) that the iter-3 diagnosis showed share a single structural cause in tau's dominance share.

**Parameters:**
  - `tau`: `[0.82, 0.92]`
  - `eta`: `[4.0, 6.0]`
  - `delta`: `[0.20, 0.45]`
  - `gamma`: `[2.8, 3.5]`
  - `c`: `[0.10, 0.25]`
  - `beta`: `[2.5, 6.0]`
  - `epsilon`: `[0.06, 0.15]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # PGDT: Primacy-Gated Distrust Tallying, FLOORED-GATE edition,
    # REBALANCED-DISTRUST-SHARE variant.
    # iter-4 minimal-diff edit per critic's iter-3 directive:
    #   (1) PRIMARY KNOB: LOWER TAU. Range moved from [0.90, 0.97] to
    #       [0.82, 0.92] (still inside the arbiter's prescribed
    #       [0.80, 0.97]). This grows the flat share (1-tau)/n of the
    #       multiplicative bracket from ~1.5% to ~4-5% per cue, which is
    #       the ONLY lever that can move Exp8's final 0.90-validity
    #       expert (its weight is structurally pinned at the flat share
    #       regardless of gate shape), and simultaneously softens the
    #       over-committed distrust allegiances on Exp2/Exp4/Exp7.
    #   (2) COMPENSATING KNOB: RAISE DELTA CEILING. Range moved from
    #       [0.15, 0.40] to [0.20, 0.45] so the iter-2->3 trajectory's
    #       verified leverage on Exp3's back-loading (stronger
    #       anti-validity as the early-gate flattens) compensates the
    #       Exp3 weakening that the tau drop would otherwise cause.
    #   (3) Everything else frozen: eta, gamma, c, beta, epsilon.
    # Mechanism structure unchanged from the accepted iter-3 base.
    # Stimulus: array of shape (2, n_features); row 0 = option A, row 1 = B.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"PGDT expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n = int(stim.shape[1])

    v = np.asarray(parameters["validities"], dtype=float).ravel()
    if v.shape[0] != n:
        # Graceful fallback: strictly decreasing validities.
        v = np.linspace(0.95, 0.55, n)
    v = np.clip(v, 0.0, 1.0)

    tau = float(parameters["tau"])      # trust in the distrust kernel
    eta = float(parameters["eta"])       # convexity of the primacy gate
    delta = float(parameters["delta"])  # floor of the primacy gate
    gamma = float(parameters["gamma"])   # steepness of distrust inversion
    c = float(parameters["c"])           # partial centering of error rates
    beta = float(parameters["beta"])     # choice consistency
    eps = float(parameters["epsilon"])  # uniform lapse

    # (1) DISTRUST INVERSION with partial centering: treat the stated
    # validity as an error rate; subtract a fraction c of the smallest
    # error rate before raising to the gamma power, then normalize.
    err = 1.0 - v
    centered = np.maximum(err - c * float(err.min()), 0.0)
    inv = np.power(centered, gamma)
    Z_v = float(inv.sum())
    if Z_v > 1e-12:
        w_inv = inv / Z_v
    else:
        # Degenerate design (all validities tied at the extreme): the
        # distrust kernel is flat and the gate alone carries position.
        w_inv = np.full(n, 1.0 / n)

    # (2) STEEP CONVEX PRIMACY GATE WITH RAISED FLOOR over presentation
    # positions. Cue j (0-indexed) gets gate mass
    #     g_j = delta + (1-delta) * ((n - j)/(n - 1))^eta
    # with delta now in [0.20, 0.45]: the FIRST position still carries
    # the dominant share, but every late position retains a substantial
    # attention residue, so the multiplicative product never approaches
    # a pure primacy gradient and late low-validity (maximum-distrust)
    # cues keep enough weight to carry the anti-validity signature.
    # The raised ceiling further back-loads the product on decreasing-
    # validity layouts, compensating the tau drop for Exp3/Exp5.
    if n > 1:
        pos = np.arange(n, dtype=float)
        g = delta + (1.0 - delta) * np.power((n - pos) / (n - 1.0), eta)
    else:
        g = np.ones(1)

    # (3) MULTIPLICATIVE combination: the gate scales the distrust
    # bracket cue-by-cue. Where the bracket is flat (tied validities)
    # this reduces to the (floored) primacy gradient; where the bracket
    # nearly vanishes (trusted high-validity early experts) the gate
    # multiplies almost nothing and the distrust kernel decides where
    # the remaining weight mass sits. With tau lowered, the flat share
    # (1-tau)/n is large enough that trusted high-validity LATE experts
    # retain meaningful weight through the gate's floor residue -- the
    # structural fix for the under-weighted late-cue cluster (Exp6/8)
    # that no amount of gate flattening could rescue at tau >= 0.90.
    bracket = (1.0 - tau) / n + tau * w_inv
    raw = g * bracket
    Z_t = float(raw.sum())
    if Z_t > 1e-12:
        w = raw / Z_t
    else:
        w = np.full(n, 1.0 / n)

    # (4) Deterministic attention: every discriminating cue is always
    # attended; no capacity interference, beta is design-independent.
    s = np.sign(a - b)

    # (5) Additive integration of all discriminating cues.
    d = float(np.dot(w, s))

    # Numerically stable logistic choice rule.
    p_a = 0.5 * (1.0 + np.tanh(0.5 * beta * d))
    probs = np.array([p_a, 1.0 - p_a], dtype=float)

    # (6) Small uniform lapse; kept below IVC's range because the gate
    # itself supplies response dilution on late-cue trials.
    probs = (1.0 - eps) * probs + eps * 0.5
    return probs
```

**`policy(probs)`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # guard against float drift
    return int(np.random.choice(len(probs), p=probs))
```


### slot 2 — `pi_6` — KILLED ✗

**Description:** PRIMACY-WEIGHTED INVERTED-VALIDITY COUNTING (PW-IVC). Subjects read the communicated validities but systematically distrust-invert them: they act as if each stated validity were the expert's error rate, so highly validated experts are discounted most (the anti-validity signature). Independently and additively, subjective attention is tilted toward EARLY-presented cues by a steep convex primacy gradient concentrated on the first one or two positions, at a small-to-moderate share of the subjective-weight mixture so the distrust kernel retains dominance wherever validity information varies. Each cue j carries weight w_j = (1-tau)/n + tau * [ (1-rho) * ((1-v_j) - c*min_k(1-v_k))^gamma / Z_v + rho * ((n-j)/(n-1))^eta / Z_p ]; evidence d = sum_j w_j * sign(A_j - B_j) is integrated additively over all discriminating cues; choice is logistic in beta*d with a small uniform lapse epsilon. On tied-validity designs the distrust kernel is exactly flat and the primacy kernel alone supplies the early-vs-late contrast, producing below-chance recency-following; on decreasing-validity layouts the distrust kernel dominates and reproduces the anti-validity regression signature.

**Rationale:** Minimal-diff edit exactly per the critic's iter-3 directive; the PW-IVC mechanism and all code are unchanged from the accepted iter-3 base, and only three parameter-box knobs move. (1) rho: [0.16, 0.34] -> [0.16, 0.30]. The iter-3 ceiling raise is what regressed Exp3 from -0.434 to -0.372 (squared error ~0.042, now the dominant loss term): in the strictly-decreasing-validity layout, primacy mass (pro-early = pro-high-validity) directly cancels the distrust kernel, and the fitter was buying Exp9/Exp10 gains with primacy mass at Exp3's expense. The modest ceiling trim — placed between iter-2's 0.28 and iter-3's 0.34, not back at either, to avoid the oscillation the critic warns about — restores distrust dominance in decreasing-validity layouts. (2) beta: [2.2, 6.5] -> [2.2, 8.0], and epsilon floor: 0.08 -> 0.06. These are the commitment knobs that COMPENSATE the tied-validity experiments for the rho trim: on Exp9's count-neutral trials the anti-recency deviation from 0.5 scales roughly with beta*rho at small evidence, so a ~12% rho cut is offset by the ~23% beta headroom, while on Exps 3/5/7/10 the distrust evidence already points the right way and sharper commitment pushes each under-signed metric toward its target (Exp3 more negative toward -0.576, Exp5 toward 0.231, Exp7 up toward 0.788, Exp10 toward -0.173). The lower epsilon floor additionally de-attenuates the Exp3 OLS coefficient, which the lapse shrinks toward zero by roughly (1-eps). (3) Everything else frozen: tau [0.88, 0.97], eta [3.5, 6.5], gamma [2.9, 3.8], c [0.0, 0.2] — all worked in both prior accepted iterations. Guardrails per the critic: Exp4's anchor is lapse-capped (adherence ~ 1 - eps/2 at saturation, so the lowered epsilon floor could push it up — the ceiling 0.22 keeps the mean near 0.84, and if Exp4 exceeds 0.88 the epsilon floor should be restored to 0.10 rather than capping beta); Exp8 is accepted as the family residual (all distrust-flavored theories undershoot the real 0.396) and is monitored but not purchased at the cost of the signature experiments; Exp2 (currently +0.055) is monitored — higher beta sharpens its tally-vs-TTB contrast in the wrong direction, but the signature experiments (3, 5, 7, 9, 10) take priority, exactly as directed. The iter-1 -> iter-3 trajectory (Exp3 error +0.51 -> +0.14 -> +0.20, aggregate 0.151 -> 0.101 -> 0.099) confirms the structure is right; this final coordinated nudge on commitment level and the rho ceiling targets a loss below the 0.0989 floor.

**Parameters:**
  - `tau`: `[0.88, 0.97]`
  - `rho`: `[0.16, 0.30]`
  - `eta`: `[3.5, 6.5]`
  - `gamma`: `[2.9, 3.8]`
  - `c`: `[0.0, 0.2]`
  - `beta`: `[2.2, 8.0]`
  - `epsilon`: `[0.06, 0.22]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # PW-IVC: Primacy-Weighted Inverted-Validity Counting.
    # iter-4 minimal-diff edit per critic's iter-3 directive:
    #   (1) rho ceiling trimmed 0.34 -> 0.30 (restores distrust dominance in
    #       strictly-decreasing-validity layouts; the iter-3 ceiling raise is
    #       what regressed Exp3 from -0.434 to -0.372);
    #   (2) beta ceiling raised 6.5 -> 8.0 and epsilon floor lowered
    #       0.08 -> 0.06 (the commitment knobs that compensate the
    #       tied-validity experiments for the rho trim and de-attenuate the
    #       Exp3 OLS coefficient);
    #   (3) everything else frozen (tau, eta, gamma, c).
    # Mechanism structure unchanged from the accepted iter-3 base.
    # Stimulus: array of shape (2, n_features); row 0 = option A, row 1 = B.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"PW-IVC expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n = int(stim.shape[1])

    v = np.asarray(parameters["validities"], dtype=float).ravel()
    if v.shape[0] != n:
        # Graceful fallback: strictly decreasing validities.
        v = np.linspace(0.95, 0.55, n)
    v = np.clip(v, 0.0, 1.0)

    tau = float(parameters["tau"])      # trust in the non-flat kernel mixture
    rho = float(parameters["rho"])        # primacy share within the mixture
    eta = float(parameters["eta"])       # convexity of the primacy gradient
    gamma = float(parameters["gamma"])   # steepness of distrust inversion
    c = float(parameters["c"])           # partial centering of error rates
    beta = float(parameters["beta"])
    eps = float(parameters["epsilon"])

    # (1) DISTRUST INVERSION with partial centering: treat the stated
    # validity as an error rate; subtract a fraction c of the smallest
    # error rate before raising to the gamma power, then normalize.
    #     inv_j = ((1 - v_j) - c * min_k(1 - v_k))^gamma / Z_v
    # When all validities are tied this kernel is flat (or degenerate at
    # c -> 1, where we fall back to flat), so tied-validity designs are
    # carried entirely by the primacy kernel below.
    err = 1.0 - v
    centered = np.maximum(err - c * float(err.min()), 0.0)
    inv = np.power(centered, gamma)
    Z_v = float(inv.sum())
    if Z_v > 1e-12:
        w_inv = inv / Z_v
    else:
        w_inv = np.full(n, 1.0 / n)

    # (2) STEEP CONVEX PRIMACY GRADIENT over presentation positions: cue j
    # gets raw primacy mass ((n - j)/(n - 1))^eta with eta >= 3.5, so the
    # gradient is concentrated on the FIRST one or two positions. This
    # maximizes the early-vs-late weight CONTRAST per unit of mixture
    # share rho (what the anti-recency metrics score), while the small-to-
    # moderate rho keeps the primacy kernel from fighting the distrust
    # kernel on decreasing-validity layouts (Exps 3, 5, 8).
    if n > 1:
        pos = np.arange(n, dtype=float)
        prim = np.power((n - pos) / (n - 1.0), eta)
    else:
        prim = np.ones(1)
    Z_p = float(prim.sum())
    w_prim = prim / Z_p

    # (3) Subjective weights: mixture of flat counting, the partially
    # centered inverted-validity kernel, and the primacy kernel.
    # tau ~ 0.9+ keeps the non-flat kernels dominant; rho <= 0.30 keeps
    # the distrust kernel the dominant carrier of validity variance.
    w = (1.0 - tau) / n + tau * ((1.0 - rho) * w_inv + rho * w_prim)

    # (4) Deterministic attention: every discriminating cue is always
    # attended; no capacity interference, beta is design-independent.
    s = np.sign(a - b)

    # (5) Additive integration of all discriminating cues.
    d = float(np.dot(w, s))

    # Numerically stable logistic choice rule.
    p_a = 0.5 * (1.0 + np.tanh(0.5 * beta * d))
    probs = np.array([p_a, 1.0 - p_a], dtype=float)

    # (6) Small independent uniform lapse, capping anchor adherence.
    probs = (1.0 - eps) * probs + eps * 0.5
    return probs
```

**`policy(probs)`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # guard against float drift
    return int(np.random.choice(len(probs), p=probs))
```


## Replacement

### `pi_8` → slot 2 (via `new_theory`)

**Description:** POSITION-RECOVERED DISTRUST TALLYING (PRDT), flattened-inversion edition. Subjects scan the expert ratings in presentation order through a steep floored convex primacy gate g_j = delta + (1-delta)*((n-j)/(n-1))^eta, so the first one or two experts carry the dominant share of attention while late experts retain a floor residue. The communicated validities are distrust-inverted — subjects act as if each stated validity were the expert's error rate — but the inversion kernel is now comparatively FLAT: inv_j = ((1-v_j) - c*min_k(1-v_k))^gamma / Z with gamma in [1.6, 2.2], so low-validity experts dominate only moderately rather than monopolizing the weight mass. Trust itself recovers with presentation position: t_j = t0 + (t1-t0)*(j/(n-1))^kappa with t0 pinned at exactly zero — total skepticism toward early-communicated high-validity claims — ramping to a strong late-position residue with low convexity so credibility returns across the mid-late positions. Weights combine multiplicatively: w_j proportional to g_j * [t_j*(1-tau)/n + tau*inv_j]; evidence d = sum_j w_j*sign(A_j - B_j) is integrated additively over discriminating cues; choice is logistic in beta*d with a uniform lapse epsilon. The theoretical claim: the anti-validity signature in human data is a MODERATE distrust gradient, not a steep one — the steep primacy gate supplies the position structure while a flattened distrust kernel lets early high-validity cues retain enough weight to produce the mild TTB-following, weak tally-allegiance, and moderate validity-adherence observed across the pooled experiments.

**Rationale:** This is the flattened-inversion calibration pass the iter-5 critique explicitly requested, applied as a minimal diff on the accepted iter-4 base (loss 0.1497). No kernel, combination rule, or integration rule is altered — only parameter ranges move, and every move follows the gate-verified evidence rather than the falsified levers. (1) TAU REVERTED to [0.90, 0.93]: the iter-5 experiment empirically falsified my iter-4 hypothesis that tau was the over-distrust dial — lowering it moved Exps 1, 2, 3, 5, 7, 10 ALL away from target, because in decreasing-validity designs the position-growing trust ramp channels weight into exactly the late low-validity cues the distrust kernel already favors, so the tau trade was a wash-to-negative. (2) MAIN LEVER — GAMMA LOWERED from [2.0, 3.5] to [1.6, 2.2]. Gamma is the only knob in the prescribed architecture that reduces error-rate-cue dominance WITHOUT shifting weight into the position-ramping trust channel: flattening inv_j toward uniform lets the steep primacy gate express itself as moderate high-validity following — pulling Exp 1 up toward 0.15-0.20, Exp 2 down toward 0.20-0.25, Exp 3 up toward -0.62, Exp 5 up toward 0.18, Exp 7 down toward 0.80, Exp 10 up toward -0.20 — while simultaneously deflating the late 0.50-validity helpers' distrust dominance on Exp 14 (down from ~0.30 toward 0.20-0.25) and RAISING the final 0.90-validity expert's distrust weight on Exp 8 (up from ~0.26 toward 0.30-0.35), the one channel that can break the structural t1*(1-tau)/n cap that tau-lowering failed to break. (3) GATE CALIBRATION KEPT from iter-5's validated residuals despite its rejection: eta [5.0, 6.0], delta [0.25, 0.30] — these produced the near-exact Exp 13 (1.796 vs 1.836, the best value any theory in the pool has produced), the best-ever Exp 14, and improved Exps 8 and 11; the iter-5 rejection is attributable to the tau change, which is now reverted. (4) t0 PINNED AT EXACTLY 0.0 (degenerate interval): Exp 14 is still 2.2x the observed rescue rate and the position-1 trusted expert's bracket must vanish completely. t1 raised to [0.85, 1.00] as the directed final nudge for Exp 8 — safe for Exp 14 because the rescue option's late helpers are 0.50-validity experts whose brackets are distrust-dominated (tau*inv_j >> t_j*(1-tau)/n <= ~0.017) and the trusted expert in that design sits at position 1 where t = t0 = 0. (5) EPSILON RAISED to [0.09, 0.13] per the Exp 4 tripwire (over-determinism 0.911 vs 0.843 persists); beta held at [4.0, 5.5] — neither cut (which collapsed the primacy slopes in iter-2) nor raised (which re-blew up the over-commitment cluster in iter-3). kappa [1.5, 3.0] and c [0.0, 0.3] unchanged — the kappa decoupling validated in iter-4 remains the mechanism carrying the recency family. Tripwires honored: if the gamma drop costs Exp 13 (needs >= ~1.6) or Exp 12 (needs ~0.54), the steepened gate (eta up to 6.0, delta at 0.25) is the compensating lever already in the box — not beta, not tau, both of which the gate has now falsified. Expected net: the over-distrust cluster (the one residual cluster that has persisted through all five iterations) finally moves toward target at modest cost, landing below the 0.1497 floor.

**Parameters:**
  - `tau`: `[0.90, 0.93]`
  - `eta`: `[5.0, 6.0]`
  - `delta`: `[0.25, 0.30]`
  - `gamma`: `[1.6, 2.2]`
  - `c`: `[0.0, 0.3]`
  - `t0`: `[0.0, 0.0]`
  - `t1`: `[0.85, 1.00]`
  - `kappa`: `[1.5, 3.0]`
  - `beta`: `[4.0, 5.5]`
  - `epsilon`: `[0.09, 0.13]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # PRDT: Position-Recovered Distrust Tallying.
    # iter-6 FLATTENED-INVERSION pass per the critic's iter-5 directive.
    # Architecture UNCHANGED from the accepted iter-4 base (distrust
    # inversion + position-growing trust residue + floored primacy gate,
    # multiplicative combination, additive integration, logistic choice).
    # The iter-5 experiment falsified the tau lever empirically: lowering
    # tau moved the ENTIRE over-distrust cluster (Exps 1,2,3,5,7,10) AWAY
    # from target, because in decreasing-validity designs the trust ramp
    # favors the same late low-validity cues the distrust kernel favors.
    # The one knob that reduces error-rate-cue dominance WITHOUT shifting
    # weight into the position-ramping trust channel is GAMMA:
    #   (1) TAU REVERTED to the validated iter-4 range [0.90, 0.93].
    #   (2) MAIN LEVER -- GAMMA LOWERED to [1.6, 2.2]: flattening inv_j
    #       toward uniform lets the (now steep) primacy gate express
    #       itself as moderate high-validity following on Exps 1/2/3/5/7,
    #       deflates the late 0.50-validity helpers' distrust dominance
    #       on Exp 14, and RAISES the final 0.90-expert's distrust weight
    #       on Exp 8 -- the one channel that can break the structural
    #       t1*(1-tau)/n cap.
    #   (3) KEEP the iter-5 gate calibration the residuals validated
    #       despite the rejection: eta [5.0, 6.0], delta [0.25, 0.30]
    #       (produced the near-exact Exp 13 = 1.796 vs 1.836, best-ever
    #       Exp 14, improved Exps 8 and 11).
    #   (4) t0 PINNED AT EXACTLY 0.0 (position-1 trusted bracket must
    #       vanish); t1 raised to [0.85, 1.00] as the directed final
    #       nudge for Exp 8 (safe for Exp 14: the rescue option's late
    #       helpers are 0.50-validity, distrust-dominated, and the
    #       trusted expert there sits at position 1 where t0 ~ 0).
    #   (5) epsilon raised to [0.09, 0.13] per the Exp 4 tripwire
    #       (over-determinism 0.911 vs 0.843 persists); beta held at
    #       [4.0, 5.5] -- NOT cut, NOT raised.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"PRDT expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n = int(stim.shape[1])

    v = np.asarray(parameters["validities"], dtype=float).ravel()
    if v.shape[0] != n:
        # Graceful fallback: strictly decreasing validities.
        v = np.linspace(0.95, 0.55, n)
    v = np.clip(v, 0.0, 1.0)

    tau = float(parameters["tau"])        # distrust share of the bracket
    eta = float(parameters["eta"])         # convexity of the primacy gate
    delta = float(parameters["delta"])    # floor of the primacy gate
    gamma = float(parameters["gamma"])    # steepness of distrust inversion (LOW)
    c = float(parameters["c"])             # partial centering of error rates
    t0 = float(parameters["t0"])           # trust residue at position 1 (= 0)
    t1 = float(parameters["t1"])           # trust residue at the last position
    kappa = float(parameters["kappa"])    # convexity of the trust ramp (LOW)
    beta = float(parameters["beta"])      # choice consistency
    eps = float(parameters["epsilon"])   # uniform lapse

    # (1) DISTRUST INVERSION with partial centering: treat the stated
    # validity as an error rate; subtract a fraction c of the smallest
    # error rate before raising to the gamma power, then normalize.
    # gamma now in [1.6, 2.2]: the kernel is comparatively FLAT, so
    # low-validity cues dominate only moderately -- the moderate
    # anti-validity signature the pooled data demand.
    err = 1.0 - v
    centered = np.maximum(err - c * float(err.min()), 0.0)
    inv = np.power(centered, gamma)
    Z_v = float(inv.sum())
    if Z_v > 1e-12:
        w_inv = inv / Z_v
    else:
        # Degenerate design (all validities tied at the extreme): the
        # distrust kernel is flat and position alone carries structure.
        w_inv = np.full(n, 1.0 / n)

    if n > 1:
        pos = np.arange(n, dtype=float)
        # (2) POSITION-GROWING TRUST RESIDUE: total skepticism toward
        # early-communicated trusted claims (t0 = 0 exactly), progressive
        # credibility concession as trusted experts appear later, with
        # LOW convexity kappa so trust returns across mid-late positions.
        t = t0 + (t1 - t0) * np.power(pos / (n - 1.0), kappa)
        # (3) STEEP CONVEX PRIMACY GATE WITH FLOOR over presentation order
        # (the iter-5 calibration validated by the residuals: eta in
        # [5.0, 6.0], delta in [0.25, 0.30]).
        g = delta + (1.0 - delta) * np.power((n - pos) / (n - 1.0), eta)
    else:
        t = np.array([1.0])
        g = np.array([1.0])

    # (4) MULTIPLICATIVE combination: the gate scales the bracket cue-by-cue.
    # The bracket's trusted channel is modulated by the position-growing
    # trust residue, so an early trusted expert's bracket vanishes (t0 = 0)
    # while a late trusted expert regains up to the full flat share (t1).
    bracket = t * (1.0 - tau) / n + tau * w_inv
    raw = g * bracket
    Z_t = float(raw.sum())
    if Z_t > 1e-12:
        w = raw / Z_t
    else:
        w = np.full(n, 1.0 / n)

    # (5) Deterministic attention: every discriminating cue is attended;
    # no capacity interference, beta is design-independent.
    s = np.sign(a - b)

    # Additive integration of all discriminating cues.
    d = float(np.dot(w, s))

    # Numerically stable logistic choice rule.
    p_a = 0.5 * (1.0 + np.tanh(0.5 * beta * d))
    probs = np.array([p_a, 1.0 - p_a], dtype=float)

    # Uniform lapse.
    probs = (1.0 - eps) * probs + eps * 0.5
    return probs
```

**`policy(probs)`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # guard against float drift
    return int(np.random.choice(len(probs), p=probs))
```
