# Round 9 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_10` — SURVIVED ✓

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


### slot 2 — `pi_9` — KILLED ✗

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

### `pi_11` → slot 2 (via `new_theory`)

**Description:** ANCHORED-FLOOR DISTRUST TALLYING (AFDT), tau-release retune. Subjects scan expert ratings in presentation order through a floored convex primacy gate g_j = delta + (1-delta)*((n-j)/(n-1))^eta (moderate steepness, upper-end floor), so the first one or two experts dominate attention while the late tail retains a meaningful residue. The communicated validities are DISTRUST-INVERTED — subjects act as if each stated validity were the expert's error rate — via inv_j = ((1-v_j) - c*min_k(1-v_k))^gamma / Z at the steep corner (gamma 3.2-3.4, minimal centering), but the distrust SHARE tau is now released to [0.91, 0.93]: the trusted (flat-share) channel carries a larger slice of the bracket, so the position-ramped trust channel — anchored low at t0 in [0.15, 0.20] and capped strictly below unity at t1 in [0.90, 1.00] with a concave-leaning kappa in [0.8, 1.0] — has roughly double the leverage on early-vs-late trusted duels. Because the early-vs-late trusted margin scales with (1-tau)/n*(t0 - delta*t1), the released tau strengthens the late-trusted preference that the Exp 17/18 discriminants demand, shrinks the trusted cue's flat-share bracket (pushing the Exp 14 rescue rate down toward the observed lapse floor), and dilutes pure distrust dominance in the right direction on Exps 5/7; the concave kappa raises mid-position trust, pulling the over-negative Exp 18 contrast back toward zero while leaving the ends-duels (gate positions 0 and n-1, where g is eta-insensitive) untouched. Evidence d = sum_j w_j*sign(A_j - B_j) is integrated additively over all discriminating cues with w_j proportional to g_j*[t_j*(1-tau)/n + tau*inv_j]; choice is logistic in beta*d with a uniform lapse epsilon.

**Rationale:** This is a minimal-diff, in-family retune of the ACCEPTED iter-2 AFDT base (loss 0.0751), applying the critic's iter-3 directive exactly: change ONLY the two knobs that have never been moved since iter 1, and keep everything else verbatim. (A) PRIMARY — RELEASE TAU from [0.94, 0.95] down to [0.91, 0.93] (still inside the arbiter's [0.91, 0.95] box). The iter-3 experiment empirically refuted the eta lever (the gate rejected that candidate; Exp 14 worsened 0.304 -> 0.311, Exps 11/12/15 regressed, and Exp 17 drifted the wrong way toward zero). The critic's algebra identifies tau as the strongest remaining in-box lever on the dominant error: on the Exp 17 ends-duels the early-vs-late trusted margin contains the term (1-tau)/n*(t0 - delta*t1), so cutting tau from 0.94 to 0.91 roughly doubles the late-trusted margin — the direction that should deepen Exp 17 from -3.52 toward the observed -9.12. The same release shrinks the trusted cue's flat-share bracket, pushing Exp 14's rescue rate down from 0.304 toward the observed 0.132 (the family-wide failure), and dilutes pure distrust dominance in the correct direction on Exps 5 (0.205 -> toward 0.231) and 7 (0.814 -> toward 0.788). The known risk corner is Exp 3's anti-validity slope: it is carried by gamma 3.2-3.4 in the position-confounded layout and should stay <= -0.52, but the low end of the tau range is where it could flatten — hence tau is floored at 0.91 rather than opened wider. (B) SECONDARY — NARROW KAPPA to the concave low end [0.8, 1.0]. Exp 18's over-negative contrast (-3.24 vs obs -0.93) is governed by mid-position trust, not the ends-duels: a concave ramp raises t_1..t_3, pulling Exp 18 back toward zero while leaving the eta-insensitive ends (g_0 = 1, g_5 = delta) that carry Exp 17 untouched, and plausibly lifting the undershot Exps 11/12/15 primacy composites via stronger mid-position trusted weights. (C) KEPT VERBATIM: eta [6.0, 7.0], c [0.05, 0.10], gamma [3.2, 3.4], the (t0, delta, t1) corner with delta*t1 > t0, and the noise box (beta [2.8, 4.0], epsilon [0.12, 0.18]) that anchors Exps 4/5/19 — the iter-3 record shows eta moves are net-negative and the noise box is already dead-on where it matters. (D) NOT chased: Exp 2's family-wide +0.14 overshoot, Exp 8's family-wide -0.09, the residual Exp 17 gap imposed by the arbiter's t1 <= 1.0 ceiling, and Exp 20's knife-edge binary flip (0.0005-wide theory gap, near-chance between-subject variance) — all documented as irreducible-for-now. Every edit is a parameter-range re-anchoring inside the arbiter-prescribed AFDT architecture; the multiplicative gate-x-bracket combination, additive cue integration, and logistic-lapse choice rule are untouched, so the iter-2 gains (Exp 3 dead-on at -0.578, Exp 17 on the correct negative side, Exp 15 at 0.95, Exps 1/4/5/6/7/9/10/11/12/16/19 within ~1 sd) are preserved while the two untouched knobs are finally exercised in the direction the residual geometry demands.

**Parameters:**
  - `tau`: `[0.91, 0.93]`
  - `eta`: `[6.0, 7.0]`
  - `delta`: `[0.20, 0.25]`
  - `gamma`: `[3.2, 3.4]`
  - `c`: `[0.05, 0.10]`
  - `t0`: `[0.15, 0.20]`
  - `t1`: `[0.90, 1.00]`
  - `kappa`: `[0.8, 1.0]`
  - `beta`: `[2.8, 4.0]`
  - `epsilon`: `[0.12, 0.18]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # AFDT: ANCHORED-FLOOR DISTRUST TALLYING, TAU-RELEASE retune.
    # Minimal-diff edit of the ACCEPTED iter-2 base per the critic's
    # iter-3 directive, which diagnosed that the eta lever is
    # EMPIRICALLY EXHAUSTED (the iter-3 eta push was rejected: Exp 14
    # worsened, Exps 11/12/15 regressed, Exp 17 drifted toward zero).
    # The two knobs that have NEVER been moved since iter 1 are the
    # trust-channel SHARE (tau) and the trust-ramp SHAPE (kappa):
    #   (A) PRIMARY -- TAU RELEASED from [0.94, 0.95] to [0.91, 0.93]
    #       (inside the arbiter's [0.91, 0.95] box). On the Exp 17
    #       ends-duels the early-vs-late trusted margin is
    #       (1-delta)*tau*inv_trusted + (1-tau)/n*(t0 - delta*t1):
    #       the late-favoring term scales with (1-tau), so lowering
    #       tau from 0.94 to 0.91 roughly DOUBLES the late-trusted
    #       margin -- the strongest available in-box lever on the
    #       dominant Exp 17 error (-3.52 vs obs -9.12). It also
    #       shrinks the trusted cue's flat-share bracket, pushing
    #       Exp 14's rescue rate down from 0.304 toward the observed
    #       0.132, and dilutes distrust dominance in the right
    #       direction on Exps 5 (0.205 -> toward 0.231) and 7
    #       (0.814 -> toward 0.788). Exp 3's anti-validity slope is
    #       carried by gamma (3.2-3.4) in the position-confounded
    #       layout and should stay <= -0.52; the low end of the tau
    #       range is the risk corner.
    #   (B) SECONDARY -- KAPPA NARROWED to the concave low end
    #       [0.8, 1.0]. Exp 18's over-negative contrast (-3.24 vs obs
    #       -0.93) is governed by MID-POSITION trust: a concave ramp
    #       raises t_1..t_3, pulling Exp 18 toward zero without
    #       touching the ends-duels that carry Exp 17, and may lift
    #       the undershot Exps 11/12/15 primacy composites via
    #       stronger mid-position trusted weights.
    #   KEPT VERBATIM from the accepted base: eta [6.0, 7.0], c
    #     [0.05, 0.10], gamma [3.2, 3.4], the (t0, delta, t1) corner
    #     (t0 [0.15, 0.20], delta [0.20, 0.25], t1 [0.90, 1.00],
    #     delta*t1 > t0 robustly), and the noise box (beta [2.8,
    #     4.0], epsilon [0.12, 0.18]) that anchors Exps 4/5/19.
    #   NOT chased (family-wide irreducible or arbiter-barred):
    #     Exp 2 (+0.14), Exp 8 (-0.09), the residual Exp 17 gap from
    #     the t1 <= 1.0 ceiling, and Exp 20's knife-edge binary flip.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"AFDT expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n = int(stim.shape[1])

    v = np.asarray(parameters["validities"], dtype=float).ravel()
    if v.shape[0] != n:
        # Graceful fallback: strictly decreasing validities.
        v = np.linspace(0.95, 0.55, n)
    v = np.clip(v, 0.0, 1.0)

    tau = float(parameters["tau"])        # distrust share (RELEASED DOWN)
    eta = float(parameters["eta"])        # primacy-gate convexity (KEPT)
    delta = float(parameters["delta"])    # primacy-gate floor (KEPT, upper end)
    gamma = float(parameters["gamma"])    # distrust-inversion steepness (KEPT)
    c = float(parameters["c"])            # partial centering (KEPT, low)
    t0 = float(parameters["t0"])          # trust floor (KEPT, anchored low)
    t1 = float(parameters["t1"])          # trust cap (KEPT, never super-unity)
    kappa = float(parameters["kappa"])    # ramp exponent (NARROWED CONCAVE)
    beta = float(parameters["beta"])      # choice consistency (KEPT)
    eps = float(parameters["epsilon"])    # uniform lapse (KEPT)

    # (1) DISTRUST INVERSION with partial centering: treat the stated
    # validity as an error rate; subtract a fraction c of the smallest
    # error rate before raising to the gamma power, then normalize.
    # gamma in [3.2, 3.4] with c in [0.05, 0.10] is the steep corner
    # that carries the full anti-validity gradient in
    # validity-position-confounded layouts (Exps 3/5/9/10); tau now in
    # [0.91, 0.93] leaves the trusted channel a larger flat share.
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
        # (2) CAPPED POWER-LAW TRUST RAMP with the floor anchored at the
        # LOWER end and the exponent narrowed to the concave low end:
        # t0 in [0.15, 0.20] keeps only 15-20% of the flat share on the
        # early trusted expert; t1 in [0.90, 1.00] caps the late residue
        # strictly below unity (no late amplification); kappa in
        # [0.8, 1.0] raises trust at the MID positions (the Exp 18
        # lever) while leaving the ends contrast intact.
        t = t0 + (t1 - t0) * np.power(pos / (n - 1.0), kappa)
        # (3) FLOORED CONVEX PRIMACY GATE with the floor at the UPPER
        # end and moderate steepness (KEPT from the accepted base):
        # eta in [6.0, 7.0], delta in [0.20, 0.25]. The ordering
        # delta*t1 > t0 holds robustly, so early-vs-late trusted duels
        # resolve in favor of the late trusted expert.
        g = delta + (1.0 - delta) * np.power((n - pos) / (n - 1.0), eta)
    else:
        t = np.array([1.0])
        g = np.array([1.0])

    # (4) MULTIPLICATIVE combination: the gate scales the bracket
    # cue-by-cue. The trusted channel is modulated by the capped ramp
    # -- anchored-low early, near-full-but-never-amplified late --
    # while the distrust kernel (tau >= 0.91) still dominates the
    # bracket, now with a larger trusted flat share that doubles the
    # ramp's leverage on the early-vs-late trusted duels.
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

    # (6) Numerically stable logistic choice rule -- beta in [2.8, 4.0]
    # with a uniform lapse epsilon in [0.12, 0.18], kept verbatim from
    # the accepted calibration (anchors Exps 4/5/19).
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
