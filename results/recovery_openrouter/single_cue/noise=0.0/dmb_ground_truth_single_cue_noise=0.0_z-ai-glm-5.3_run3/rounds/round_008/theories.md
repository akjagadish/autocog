# Round 8 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_7` — KILLED ✗

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


### slot 2 — `pi_9` — SURVIVED ✓

**Description:** POSITION-BRIDGED DISTRUST TALLYING (PBDT), un-bundled-gate edition. Subjects scan expert ratings in presentation order through a steep floored convex primacy gate g_j = delta + (1-delta)*((n-j)/(n-1))^eta, with the gate restored to the steep end of its box (eta high, delta low) so late low-validity helper cues are crushed relative to early ones — the structural lever separating primacy-rescue trials from validity-position-confounded layouts. Stated validities are distrust-inverted (treated as error rates) via inv_j = ((1-v_j) - c*min_k(1-v_k))^gamma / Z, kept at the steep bridged range [2.6, 2.9] with a high distrust share tau, so the anti-validity gradient in confounded layouts is carried by the distrust kernel alone rather than by the gate. Trust in high-validity experts ramps with position from a small-but-nonzero floor t_j = t0 + (t1-t0)*(j/(n-1))^kappa, with a slightly concave-leaning ramp (kappa below 1) so late trusted experts recover nearly the full flat share without raising the early floor. Weights combine multiplicatively, w_j proportional to g_j * [t_j*(1-tau)/n + tau*inv_j]; evidence d = sum_j w_j*sign(A_j - B_j) is integrated additively over discriminating cues; choice is logistic in beta*d with a uniform lapse epsilon, with noise set between the two previous iterations so slope/contrast metrics that were over-attenuated recover while anchor allegiances keep their margin.

**Rationale:** This is a minimal-diff parameter re-balancing of the ACCEPTED iter-2 PBDT base that implements the critic's three un-bundled directives exactly, with no architectural change (the multiplicative gate-times-bracket kernel, additive integration, and logistic-lapse choice rule are preserved verbatim). The iter-2 diagnosis was that directive (b) of the previous round had bundled two moves whose effects opposed each other: the distrust strengthening (gamma/tau) delivered the Exp 3 win (-0.265 -> -0.479 vs real -0.576, variance now matching the data), while the eta-down/delta-up gate flattening was counterproductive, pushing Exp 14's trusted-early following UP (0.312 -> 0.350 vs real 0.132) by un-crushing the late 50%-validity helpers on the trusted-early side, and dropping Exp 13 (1.58 -> 1.35 vs real 1.84). (A) therefore reverts the gate to the steep end (eta [6.5, 8.0], delta [0.15, 0.25]) while freezing gamma [2.6, 2.9] and tau [0.91, 0.93] exactly at their iter-2 values: the steep gate is the only lever that structurally separates Exp 14 from Exp 3, and iter-1's steep-gate configuration already demonstrated Exp 13 values near 1.58 that the flattened iter-2 gate lost. (B) partially rolls back the global noise to between the two prior calibrations (beta [3.6, 4.4], epsilon [0.11, 0.14]): iter-2's over-attenuation cost Exps 1 (0.156 vs 0.208), 11 (0.206 vs 0.315), 12 (0.404 vs 0.540) and 15 (0.911 vs 1.036), all of which should recover, at only ~+0.01-0.02 on Exps 4/7 which currently sit with margin (0.879 vs 0.843, 0.820 vs 0.788). Exp 2 is explicitly NOT chased -- every distrust-family theory sits near 0.30 there and the critic classified it as noise-insensitive and structural, so it is held as a monitored constraint. (C) widens only the ramp's convexity downward (kappa [0.8, 1.5], slightly below the arbiter's original floor of 1.0) so the late trusted expert recovers more of the flat share earlier in the tail -- the one remaining lever for Exp 8's FEDI (0.294 -> toward 0.396) that does not conflict with (A) or (B), since it touches neither t0 (which Exp 14 needs low), tau, nor the gate. Watch-points honored: Exp 16 (0.350 vs 0.322) should not move more than ~0.05 since the steep gate acts on the same late tail that Exp 16's discriminant weights, and Exp 3's coefficient is protected by the preserved gamma/tau channel rather than by any gate re-flattening.

**Parameters:**
  - `tau`: `[0.91, 0.93]`
  - `eta`: `[6.5, 8.0]`
  - `delta`: `[0.15, 0.25]`
  - `gamma`: `[2.6, 2.9]`
  - `c`: `[0.05, 0.25]`
  - `t0`: `[0.25, 0.35]`
  - `t1`: `[0.95, 1.00]`
  - `kappa`: `[0.8, 1.5]`
  - `beta`: `[3.6, 4.4]`
  - `epsilon`: `[0.11, 0.14]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # PBDT: Position-Bridged Distrust Tallying, UN-BUNDLED-GATE edition
    # (iter-3). Minimal-diff edit of the ACCEPTED iter-2 base per the
    # critic's iter-2 directive, which diagnosed that the iter-2 move (b)
    # had bundled an efficacious half (distrust strengthening: the Exp 3
    # win) with a counterproductive half (gate flattening: the Exp 14
    # and Exp 13 regressions). This iteration UN-BUNDLES them:
    #   (A) REVERT THE GATE FLATTENING, KEEP THE DISTRUST STRENGTHENING:
    #       eta [5.5,6.0] -> [6.5, 8.0] (steep end), delta [0.25,0.30]
    #       -> [0.15, 0.25] (low floor), while gamma stays [2.6, 2.9]
    #       and tau stays [0.91, 0.93] exactly as in iter-2. The steep
    #       gate crushes the late 50%-validity helpers on Exp 14's
    #       trusted-early side (0.350 -> toward 0.132) and restores the
    #       M/C position families on Exp 13 (1.35 -> toward 1.84);
    #       the anti-validity gradient in confounded layouts (Exp 3)
    #       is carried by the distrust kernel, which iter 2 verified.
    #   (B) PARTIALLY ROLL BACK THE NOISE: beta [3.2,4.0] -> [3.6, 4.4]
    #       and epsilon [0.13,0.16] -> [0.11, 0.14] -- between the
    #       iter-1 and iter-2 settings. Lifts the over-attenuated slope
    #       metrics (Exps 1, 11, 12, 13, 15) at a small cost on Exps
    #       4/7, which currently have margin. Exp 2 is held as a
    #       monitored constraint (structural, noise-insensitive).
    #   (C) WIDEN THE RAMP CONVEXITY DOWNWARD: kappa [1.0,1.5] ->
    #       [0.8, 1.5], a concave-leaning ramp so the late trusted
    #       expert recovers more of the flat share (Exp 8 FEDI
    #       0.294 -> toward 0.396) without touching t0, tau, or the
    #       gate. t0/t1 unchanged from iter-2.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"PBDT expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n = int(stim.shape[1])

    v = np.asarray(parameters["validities"], dtype=float).ravel()
    if v.shape[0] != n:
        # Graceful fallback: strictly decreasing validities.
        v = np.linspace(0.95, 0.55, n)
    v = np.clip(v, 0.0, 1.0)

    tau = float(parameters["tau"])        # distrust share of the bracket (HIGH, kept)
    eta = float(parameters["eta"])        # convexity of the primacy gate (STEEP, reverted)
    delta = float(parameters["delta"])    # floor of the primacy gate (LOW, reverted)
    gamma = float(parameters["gamma"])    # steepness of distrust inversion (kept)
    c = float(parameters["c"])           # partial centering of error rates
    t0 = float(parameters["t0"])          # trust floor at position 1 (LOW, kept)
    t1 = float(parameters["t1"])          # trust residue at the last position (HIGH, kept)
    kappa = float(parameters["kappa"])   # convexity of the trust ramp (WIDENED DOWN)
    beta = float(parameters["beta"])     # choice consistency (PARTIALLY RESTORED)
    eps = float(parameters["epsilon"])   # uniform lapse (PARTIALLY RESTORED)

    # (1) DISTRUST INVERSION with partial centering: treat the stated
    # validity as an error rate; subtract a fraction c of the smallest
    # error rate before raising to the gamma power, then normalize.
    # gamma in [2.6, 2.9] and tau in [0.91, 0.93] are preserved verbatim
    # from iter-2 -- the verified carriers of the anti-validity gradient
    # in validity-position-confounded layouts (Exp 3).
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
        # (2) TRUST RAMP with widened-down convexity: low floor t0,
        # high t1, and kappa now allowed below 1.0 so the ramp is
        # concave-leaning -- the late trusted expert recovers more of
        # the flat share earlier in the tail (the Exp 8 FEDI lever)
        # without raising the early floor that Exp 14 needs kept low.
        t = t0 + (t1 - t0) * np.power(pos / (n - 1.0), kappa)
        # (3) PRIMACY GATE REVERTED TO THE STEEP END: eta in [6.5, 8.0]
        # with delta in [0.15, 0.25]. The steep, low-floored gate
        # concentrates attention on the first one or two experts and
        # crushes the late tail -- the structural fix for Exp 14's
        # late-helper over-weighting and Exp 13's M/C position
        # families. The anti-validity regression in confounded layouts
        # is carried by the (preserved) steep distrust kernel instead.
        g = delta + (1.0 - delta) * np.power((n - pos) / (n - 1.0), eta)
    else:
        t = np.array([1.0])
        g = np.array([1.0])

    # (4) MULTIPLICATIVE combination: the gate scales the bracket
    # cue-by-cue. tau in [0.91, 0.93] keeps the distrust channel
    # dominant; the trusted channel is carried by the ramp (small
    # early, near-full and now earlier-recovering late).
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

    # Numerically stable logistic choice rule -- beta in [3.6, 4.4],
    # between the iter-1 and iter-2 calibrations, so the slope metrics
    # over-attenuated by iter-2's noise (Exps 1, 11, 12, 13, 15)
    # recover while Exps 4/7 keep their remaining margin.
    p_a = 0.5 * (1.0 + np.tanh(0.5 * beta * d))
    probs = np.array([p_a, 1.0 - p_a], dtype=float)

    # (6) Uniform lapse, epsilon in [0.11, 0.14]: partially rolled back
    # from iter-2's [0.13, 0.16] per the critic's split-the-difference
    # directive. Exp 2 is NOT chased with further noise -- it is held
    # as a monitored structural constraint of the distrust family.
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

### `pi_10` → slot 1 (via `new_theory`)

**Description:** RECIPROCAL-TRUST DISTRUST TALLYING (RTDT), gate-steepened retune. Subjects scan expert ratings in presentation order through a steep floored convex primacy gate g_j = delta + (1-delta)*((n-j)/(n-1))^eta, with the gate now at the steeper/low-floored corner (eta in [5.5, 7.5], delta in [0.12, 0.25]) so the first one or two experts carry the dominant share of attention while the late tail retains only a small floor residue. Stated validities are distrust-inverted (treated as error rates) via a moderately steep kernel inv_j = ((1-v_j) - c*min_k(1-v_k))^gamma / Z with gamma backed off to [2.7, 3.3] and tau in [0.91, 0.95] — the calibration the iter-2 experiment verified moves the anti-validity slope, validity adherence, and recency composites toward their observed values without overshooting. The reciprocal trust ramp is retained at its moderate band: t_j = t0*(t1/t0)^((j/(n-1))^kappa) with t0 in [0.03, 0.12] (early trusted claims heavily discounted but not annihilated), super-unity t1 in [1.0, 1.6] amplifying late high-validity experts above the flat share, and kappa in [0.6, 1.0] so trust recovers early in the tail. Weights combine multiplicatively, w_j proportional to g_j*[t_j*(1-tau)/n + tau*inv_j]; evidence d = sum_j w_j*sign(A_j - B_j) is integrated additively over discriminating cues; choice is logistic in beta*d with a uniform lapse epsilon, with the softened noise box (beta [2.5, 4.0], epsilon [0.12, 0.20]) that iter-2 verified improves the tally-allegiance contrast while the anchors retain headroom.

**Rationale:** This is a minimal-diff retune of the ACCEPTED iter-1 RTDT base that applies exactly the three components of the critic's iter-2 directive, each of which is now evidence-graded by the gate's accept/reject trajectory. (1) KEEP THE KERNEL BACKOFF VERBATIM (gamma [2.7, 3.3], tau [0.91, 0.95]): the iter-2 experiment, though rejected overall, showed this single change moved every intended target toward observed — Exp3 slope -0.6225 -> -0.5569 (obs -0.5756, now within noise), Exp5 adherence 0.196 -> 0.209 (obs 0.231), Exp9 0.328 -> 0.343 (obs 0.341, essentially perfect), Exp10 -0.228 -> -0.190 (obs -0.173), Exp2 0.304 -> 0.289 (obs 0.158). This is the winning half of iter-2 and is preserved without modification. (2) REVERT THE TRUST-RAMP EXTREMES: the iter-1 directive to crush t0 to [0, 0.05] and the iter-2 widening of t1 to 2.0 with kappa 0.5 were both refuted — Exp14 rescue went UP not down (t0 is not the rescue lever), Exp8 FEDI was untouched (0.308 -> 0.310), and Exp18 exploded from a near-perfect -0.967 to -3.80 because the extreme ramp ratio over-expressed the crushed-early-trusted pattern far past the observed -0.93. The ramp returns to a moderate band between the two flanking iterations: t0 [0.03, 0.12], t1 [1.0, 1.6], kappa [0.6, 1.0]. Exp8's residual (~1.3 SD) is tolerated rather than chased with ramp extremes that cost 60-SD on Exp18. (3) NEW LEVER — STEEPEN THE PRIMACY GATE (eta [4.5, 6.5] -> [5.5, 7.5], delta [0.15, 0.30] -> [0.12, 0.25]): this addresses the two coherent residual clusters the iter-1 base left open. First, Exp14's primacy-rescue (0.331 vs obs 0.132, >2 SD): the rescue trials pivot on the early 50%-validity expert in the rescue option vs late 50%-validity helpers in the opposing option (the 95% experts carry almost no distrust weight), so a steeper gate directly pushes the rescue rate toward the lapse floor ~0.13 — the structural fix t0 could not deliver. Second, the primacy-composite undershoot cluster (Exp11 0.236 vs 0.315, Exp12 0.394 vs 0.540, Exp13 1.428 vs 1.836, Exp15 0.936 vs 1.036): all four currently undershoot, and steepening the gate while reverting the ramp act in the same direction on the g_j*t_j product, restoring the effective primacy signal the extreme ramp had flattened. The softened noise box (beta [2.5, 4.0], epsilon [0.12, 0.20]) is kept as verified in iter-2: Exp2 improved to ~1.8 SD while Exp4 anchors retain headroom (0.851 vs 0.843). Falsifiable checks for this iteration: Exp18 back in [-1.3, -0.6] (the hardest constraint — if it leaves this band the ramp is still too extreme), Exp14 rescue in [0.10, 0.22], Exp13 >= 1.5, Exp15 in [0.9, 1.2], Exp3 stays in [-0.60, -0.54], Exp4 >= 0.82, Exp17 in [-10, -8]. The architecture, mechanism family, and all function structure are unchanged from the accepted base — only five parameter ranges (gamma, tau, eta, delta, t0/t1/kappa, beta/epsilon) are retuned.

**Parameters:**
  - `tau`: `[0.91, 0.95]`
  - `eta`: `[5.5, 7.5]`
  - `delta`: `[0.12, 0.25]`
  - `gamma`: `[2.7, 3.3]`
  - `c`: `[0.05, 0.25]`
  - `t0`: `[0.03, 0.12]`
  - `t1`: `[1.0, 1.6]`
  - `kappa`: `[0.6, 1.0]`
  - `beta`: `[2.5, 4.0]`
  - `epsilon`: `[0.12, 0.20]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # RTDT: Reciprocal-Trust Distrust Tallying, GATE-STEEPENED retune.
    # Minimal-diff edit of the ACCEPTED iter-1 base per the critic's
    # iter-2 directive. The iter-2 experiment decomposed cleanly:
    #   WINNER (kept verbatim): the kernel backoff -- gamma [3.0,4.2] ->
    #     [2.7, 3.3], tau [0.92,0.96] -> [0.91, 0.95]. This moved
    #     Exp3 (-0.62 -> -0.557, obs -0.576), Exp5 (0.196 -> 0.209),
    #     Exp9 (0.328 -> 0.343, obs 0.341), Exp10 (-0.228 -> -0.190)
    #     and Exp2 (0.304 -> 0.289) all toward observed.
    #   LOSER (reverted): the trust-ramp extremes (t0 crushed to
    #     [0,0.05], t1 widened to [1.0,2.0], kappa lowered to [0.5,0.9])
    #     broke Exp18 (-0.97 -> -3.80) and flattened the g*t primacy
    #     product on Exps 11-15 without moving their intended targets
    #     (Exp14 rescue went UP; Exp8 FEDI untouched). Reverted to the
    #     moderate band: t0 [0.03, 0.12], t1 [1.0, 1.6], kappa [0.6, 1.0].
    #   NEW LEVER: the primacy gate steepened -- eta [4.5,6.5] ->
    #     [5.5, 7.5], delta [0.15,0.30] -> [0.12, 0.25]. The Exp14
    #     rescue trials pivot on the early 50%-validity expert vs late
    #     50% helpers, so a steeper gate pushes rescue toward the lapse
    #     floor (~0.13 observed); simultaneously Exps 11/12/13/15 all
    #     UNDERSHOOT their primacy composites, and steepening the gate
    #     and reverting the ramp act in the same direction on g*t.
    #   Noise box kept softened: beta [2.5, 4.0], epsilon [0.12, 0.20]
    #     (iter-2 verified Exp2 improved to ~1.8 SD with anchor headroom).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"RTDT expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n = int(stim.shape[1])

    v = np.asarray(parameters["validities"], dtype=float).ravel()
    if v.shape[0] != n:
        # Graceful fallback: strictly decreasing validities.
        v = np.linspace(0.95, 0.55, n)
    v = np.clip(v, 0.0, 1.0)

    tau = float(parameters["tau"])        # distrust share (BACKED OFF)
    eta = float(parameters["eta"])        # primacy-gate convexity (STEEPENED)
    delta = float(parameters["delta"])    # primacy-gate floor (LOW)
    gamma = float(parameters["gamma"])    # distrust-inversion steepness (BACKED OFF)
    c = float(parameters["c"])            # partial centering of error rates
    t0 = float(parameters["t0"])          # early-trust floor (MODERATE, reverted)
    t1 = float(parameters["t1"])          # late-trust residue (SUPER-UNITY, reverted)
    kappa = float(parameters["kappa"])    # reciprocal-ramp exponent (reverted)
    beta = float(parameters["beta"])      # choice consistency (SOFTENED)
    eps = float(parameters["epsilon"])    # uniform lapse (SOFTENED)

    # (1) DISTRUST INVERSION with partial centering: treat the stated
    # validity as an error rate; subtract a fraction c of the smallest
    # error rate before raising to the gamma power, then normalize.
    # gamma in [2.7, 3.3] with tau in [0.91, 0.95] is the verified
    # calibration: steep enough to carry the anti-validity gradient in
    # validity-position-confounded layouts, flat enough not to
    # over-anti-follow high-validity cues (Exp3/5/8/10 all verified).
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
        # (2) RECIPROCAL TRUST RAMP at the moderate band, computed in
        # log space for numerical stability:
        #   log t_j = log t0 + (j/(n-1))^kappa * (log t1 - log t0).
        # t0 in [0.03, 0.12] discounts early trusted claims without
        # annihilating them (the iter-2 t0~0 corner exploded Exp18);
        # t1 in [1.0, 1.6] with kappa in [0.6, 1.0] amplifies late
        # trusted experts above the flat share moderately.
        t0_eff = max(t0, 1e-4)  # guard the geometric form against t0 = 0
        t1_eff = max(t1, 1e-4)
        frac = np.power(pos / (n - 1.0), kappa)  # in [0, 1]
        t = np.exp(np.log(t0_eff) + frac * (np.log(t1_eff) - np.log(t0_eff)))
        # (3) STEEP FLOORED CONVEX PRIMACY GATE, steepened per the
        # critic's iter-2 directive: eta in [5.5, 7.5], delta in
        # [0.12, 0.25]. The steeper gate pushes Exp14's rescue rate
        # toward the lapse floor and strengthens the primacy composites
        # on Exps 11/12/13/15, which all currently undershoot.
        g = delta + (1.0 - delta) * np.power((n - pos) / (n - 1.0), eta)
    else:
        t = np.array([1.0])
        g = np.array([1.0])

    # (4) MULTIPLICATIVE combination: the gate scales the bracket
    # cue-by-cue. The trusted channel is modulated by the reciprocal
    # ramp -- heavily discounted early, moderately amplified late --
    # while the distrust kernel (tau >= 0.91) dominates the bracket
    # elsewhere.
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

    # (6) Numerically stable logistic choice rule -- beta in [2.5, 4.0]
    # with a uniform lapse epsilon in [0.12, 0.20]: the softened noise
    # box the iter-2 experiment verified (Exp2 improved toward 0.158
    # while Exp4 anchors retain headroom above 0.82).
    p_a = 0.5 * (1.0 + np.tanh(0.5 * beta * d))
    probs = np.array([p_a, 1.0 - p_a], dtype=float)
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
