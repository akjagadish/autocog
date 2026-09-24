# Round 8 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_9` — SURVIVED ✓

**Description:** Heavy-Tailed Graded Dual-Route Distrust (HT-GDRD). Subjects evaluate expert endorsements through two concurrent routes. ROUTE 1 (analytic): every endorsement is a charge on its option — for the distrustful majority a LIABILITY whose magnitude grows with the endorser's raw claimed validity, d_j = c0 + c1*v_j^kappa; for a small 'trust-leaning' minority an ASSET proportional to the (capped) log-odds of validity, alpha*logit(v_j), that dominates the liability for high-validity endorsers. ROUTE 2 (graded counting): a continuous preference for the fewer-endorsement option, sigmoid(gamma*(cnt_B-cnt_A)). The population is an explicitly over-dispersed THREE-COMPONENT mixture driven by one uniform quantile u_pop: (i) a skeptic majority (~78%) with a wide, heavy-tailed lapse distribution (eps in [0.13, 0.27], Beta-like t^0.8 shaping) and a loose route weight w in [0.82, 0.96]; (ii) a counter minority (~16%) with near-balanced routes (w in [0.46, 0.70]), low-moderate lapse, and a steep count slope gamma >= 2.5; and (iii) a small trust-leaning mode (~6%) with a dominant positive micro-trust term (alpha in [0.50, 0.68]) and an elevated lapse. The single update this iteration concerns the analytic route's steepness: the iter-1 residual pattern — soft gradient assays (Exps 5-7, 9, 10) overshooting toward 0.5 WHILE ceiling-compliance assays (Exps 8, 11, 12) undershoot their ceilings — is the signature of an under-steep sigmoid(beta*D), not of excess lapse (three gate rejections established the lapse dimension is exhausted and sits near its local optimum). Raising the beta box from [1.3, 2.4] to [1.8, 3.0] simultaneously deepens the validity-gradient contrasts, raises saturated skeptic compliance toward the observed ceilings, and pulls the soft assays away from chance, without touching the lapse bands that the ceiling-prevalence and dispersion assays (Exps 13, 14, 16) are knife-edge calibrated to.

**Rationale:** This is the minimal-diff edit the iter-4 critique prescribes: the iter-1 accepted base (loss 0.0607) re-emitted verbatim with exactly ONE knob changed -- the analytic-route slope beta, raised from [1.3, 2.4] to [1.8, 3.0]. The decision logic: (1) Three consecutive gate rejections (iters 2, 3, 4) all involved lapse-band edits, and the Exp-13 trajectory across those attempts (0.22 -> 0.40 -> 0.52 -> 0.42 against an observed 0.28) shows the lapse dimension is a knife-edge whose iter-1 setting is already near its local optimum; the critique explicitly forbids touching it again, so all lapse bands, mixture masses, w-ranges, gamma bands, c1, kappa, and w_spread are byte-identical to the accepted base (including the counter w ceiling restored to 0.70, whose iter-4 cut to 0.64 was diagnosed as the cause of the Exp 11/12 drag). (2) The iter-1 residual pattern is diagnostic of under-steepness rather than over-lapse: lapse flattens symmetrically (softening gradient assays AND lowering ceiling compliance via the eps/2 floor), whereas the observed residuals -- Exps 5/6/7/9/10 overshooting toward 0.5 and Exps 8/11/12 undershooting their ceilings -- are exactly what a too-shallow sigmoid(beta*D) produces, since raising beta simultaneously deepens the validity-gradient contrasts (Exps 3, 6, 9), raises saturated skeptic compliance toward the observed 0.78/0.79/0.83 ceilings (Exps 8, 11, 12), and pulls the soft assays away from chance (Exps 5, 7). (3) The beta-UP direction is genuinely untested: iter-2 trimmed beta down and was rejected; iter-3 restored it and was the least-bad of the rejected attempts on the gradient assays. Raising the box shifts the sampled mean beta from ~1.85 to ~2.4, a substantial but not radical steepening. Guardrails honored: no dispersion-widening (w_spread stays +/-0.02; the gate rejected that direction twice), no lapse edits, no c1/kappa changes -- the critique designates a small kappa trim only as a contingency if Exp 1 overshoots beyond +0.05, which cannot be pre-observed here, so the single-knob discipline is maintained. Expected outcome: Exps 8, 11, 12 rise toward their ceilings without any lapse reduction; Exps 3, 6, 9 gradients strengthen; Exps 5, 7 move down off 0.5; Exps 13, 14, 16 stay near their iter-1 values (which were 0.22, 0.1051, 0.1649 vs observed 0.28, 0.1152, 0.1433) since neither lapse nor within-mode spread was touched -- with the caveat that sharper skeptic saturation slightly raises Exp-13 ceiling prevalence, which is the direction its residual (-0.06) demands.

**Parameters:**
  - `c0`: `[0.06, 0.13]`
  - `c1`: `[1.00, 1.35]`
  - `kappa`: `[1.6, 2.1]`
  - `beta`: `[1.8, 3.0]`
  - `alpha`: `[0.0, 0.05]`
  - `u_pop`: `[0, 1]`
  - `w_spread`: `[-0.02, 0.02]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # Heavy-Tailed Graded Dual-Route Distrust (HT-GDRD), iter-5:
    # the UNCHANGED iter-1 accepted base with exactly ONE knob moved,
    # per the iter-4 critique -- the analytic-route slope beta is raised
    # from [1.3, 2.4] to [1.8, 3.0]. Three consecutive gate rejections
    # of lapse-band edits (iters 2-4) established that the lapse
    # dimension cannot resolve the Exps 5-10 vs 13/14/15 tension and
    # that iter-1's lapse bands sit near their local optimum. The
    # iter-1 residual pattern (gradient assays too soft toward 0.5 AND
    # ceiling assays below their ceilings) is instead the signature of
    # an under-steep analytic sigmoid: raising beta pushes BOTH groups
    # toward the data simultaneously. The beta-UP direction has never
    # been tested (iter-2 trimmed beta DOWN and was rejected; iter-3
    # restored it and was neutral-to-positive). Everything else --
    # lapse bands, counter w ceiling 0.70, c1 [1.00, 1.35], kappa
    # [1.6, 2.1], w_spread +/-0.02, gamma bands, mixture masses
    # (16% counter / 6% trust) -- is byte-identical to iter 1.
    #
    # Stimulus: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    #
    # ROUTE 1 (weight w): per-endorsement charge
    #   d_j = c0 + c1 * v_j^kappa                     (distrust charge)
    #   D = pen(B) - pen(A) + alpha * sum_j clip(logit(v_j), 0, 3) * (a_j - b_j)
    #   P_route1(A) = sigmoid(beta * D)
    # For skeptics and counters alpha is a tiny residual micro-trust edge
    # (alpha in [0, 0.05]). For the trust-leaning mode alpha is DOMINANT
    # (0.50-0.68): high-validity endorsers become net assets.
    #
    # ROUTE 2 (weight 1-w): graded fewer-endorsement preference
    #   P_route2(A) = sigmoid(gamma * (cnt_B - cnt_A))
    #
    # THREE-COMPONENT HEAVY-TAILED POPULATION, driven by one uniform
    # quantile u_pop in [0, 1]:
    #   u < 0.16        -> COUNTER (16%):  w in [0.46, 0.70],
    #                      eps in [0.13, 0.23], gamma in [2.5, 4.0]
    #   0.16 <= u < 0.94 -> SKEPTIC (78%): eps = 0.13 + 0.14*t^0.8
    #                      (heavy-tailed over [0.13, 0.27]),
    #                      w in [0.82, 0.96] (loose, plus w_spread),
    #                      gamma in [1.5, 3.5]
    #   u >= 0.94       -> TRUST (6%):     alpha in [0.50, 0.68],
    #                      w in [0.44, 0.56] (count-leaning),
    #                      eps in [0.26, 0.34], gamma in [1.5, 3.0]
    #
    # History is ignored: validities are communicated in the
    # instructions, so all charges are fixed for the whole block.
    if isinstance(state, dict):
        a_vec = np.asarray(state["option_a_ratings"], dtype=float).ravel()
        b_vec = np.asarray(state["option_b_ratings"], dtype=float).ravel()
        stim = np.vstack([a_vec, b_vec])
    else:
        stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"HT-GDRD expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n_features = stim.shape[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.shape[0] != n_features:
        raise ValueError(
            f"validities length {v.shape[0]} != n_features {n_features}."
        )
    v_pow = np.clip(v, 0.5, 1.0)          # power route: v=1 harmless
    v_tr = np.clip(v, 0.5, 1.0 - 1e-6)     # guard the logit

    kappa = float(parameters["kappa"])
    beta = float(parameters["beta"])
    c0 = float(parameters["c0"])
    c1 = float(parameters["c1"])
    alpha_base = float(parameters["alpha"])
    u_pop = float(parameters["u_pop"])
    w_spread = float(parameters["w_spread"])

    # --- Three-component heavy-tailed population mixture ---
    f_counter = 0.16
    f_trust_lo = 0.94
    u = min(max(u_pop, 0.0), 1.0)
    if u < f_counter:
        # Counter minority: near-balanced routes, low-moderate lapse,
        # decisive graded count slope (gamma >= 2.5).
        t = u / f_counter
        w_mix = 0.46 + 0.24 * t          # [0.46, 0.70]
        epsilon = 0.13 + 0.10 * t        # [0.13, 0.23]
        gamma = 2.5 + 1.5 * t            # [2.5, 4.0]
        alpha_mode = alpha_base
    elif u < f_trust_lo:
        # Skeptic majority: analytic distrust dominant, with a genuinely
        # wide (heavy-tailed, Beta-like) lapse spread and a LOOSE route
        # weight (no tight neutrality coupling -- the loose version is
        # what reproduces the observed Exp-14 excess dispersion).
        t = (u - f_counter) / (f_trust_lo - f_counter)
        epsilon = 0.13 + 0.14 * (t ** 0.8)   # [0.13, 0.27], mean ~0.21
        w_mix = 0.86 + 0.08 * t + w_spread     # ~[0.82, 0.96]
        w_mix = float(min(max(w_mix, 0.82), 0.96))
        gamma = 1.5 + 2.0 * t            # [1.5, 3.5]
        alpha_mode = alpha_base
    else:
        # Trust-leaning residual mode: the positive micro-trust term is
        # DOMINANT, so near-perfect / high-validity endorsers act as net
        # assets; the route weight is count-leaning and the lapse is
        # elevated (a validity-following but inattentive subject).
        t = (u - f_trust_lo) / (1.0 - f_trust_lo)
        alpha_mode = 0.50 + 0.18 * t     # [0.50, 0.68]
        w_mix = 0.44 + 0.12 * t          # [0.44, 0.56]
        epsilon = 0.26 + 0.08 * t        # [0.26, 0.34]
        gamma = 1.5 + 1.5 * t            # [1.5, 3.0]
    epsilon = float(min(max(epsilon, 0.02), 0.45))

    # --- Route 1: validity-proportional charge plus micro-trust edge ---
    d = c0 + c1 * np.power(v_pow, kappa)
    pen_a = float(np.dot(d, a))
    pen_b = float(np.dot(d, b))

    # Micro-trust: logit(0.5) = 0 (coin-flip experts contribute nothing);
    # capped at 3 so perfect experts exert a bounded pull.
    lg = np.clip(np.log(v_tr / (1.0 - v_tr)), 0.0, 3.0)
    trust_edge = float(np.dot(lg, a - b))

    # D > 0 favors A.
    D = (pen_b - pen_a) + alpha_mode * trust_edge
    p_route1_a = 0.5 * (1.0 + np.tanh(0.5 * beta * D))  # stable sigmoid

    # --- Route 2: graded endorsement-count preference ---
    cnt_a = float(np.sum(a))
    cnt_b = float(np.sum(b))
    p_route2_a = 0.5 * (1.0 + np.tanh(0.5 * gamma * (cnt_b - cnt_a)))

    # --- Dual-route mixture ---
    p_a = w_mix * p_route1_a + (1.0 - w_mix) * p_route2_a
    p_a = float(min(max(p_a, 0.0), 1.0))
    p_core = np.array([p_a, 1.0 - p_a], dtype=float)

    # Independent lapse: with probability epsilon pick uniformly.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_8_1` — KILLED ✗

**Description:** Graded Dual-Route Distrust with a Bimodal Strategy Population (GDRD-B). Subjects evaluate expert endorsements through two concurrent routes. ROUTE 1 (analytic distrust): every endorsement is a liability whose magnitude grows monotonically in the endorser's RAW claimed validity, d_j = c0 + c1*v_j^kappa; accumulated suspicion P(X) = sum_j x_j*d_j is compared across options through sigmoid(beta*(P(B)-P(A))), plus a small residual micro-trust edge alpha*sum_j clip(logit(v_j),0,3)*(a_j-b_j) (capped so perfect experts exert bounded, not unbounded, trust pull). ROUTE 2 (graded counting): a continuous preference for the fewer-endorsement option, sigmoid(gamma*(cnt_B-cnt_A)). The population is BIMODAL in the route weight w: a large majority (~88%) of 'skeptics' (w in [0.85, 0.95], moderate lapse) coexists with a minority (~12%) of 'counters' (w in [0.50, 0.62], low lapse, decisive count slope gamma >= 2). The mixture is implemented as a quantile transform of a single uniform per-subject population variable u_pop, which holds the population-MEAN route weight at its base value while exploding the between-subject behavioral variance — the empirically observed 3-12x super-binomial variance on the saturation assays (Exps 11, 12) that no uniform range-widening could reproduce.

**Rationale:** This is the minimal-diff tuning pass the iter-4 critique explicitly requested, applied on top of the accepted iter-4 base. Exactly two functional lines change; every other mechanism, parameter range, and the jitter-free neutrality-coupled population structure are re-emitted verbatim.

(1) SKEPTIC EPSILON LOWERED from [0.18, 0.27] to [0.15, 0.23] (mean ~0.19), with the neutrality pairing w = 1 - (0.25 - eps)/(1 - eps) automatically re-solved over the new range (eps = 0.15 -> w ~ 0.88; eps = 0.23 -> w ~ 0.97, clipped at 0.95; the clip only makes skeptic v slightly positive, the safe direction for Exp 14). This attacks the two largest residuals simultaneously: (a) Exp 3 (-0.4558 vs obs -0.5467) is a difference of two choice probabilities compressed by the factor (1-eps); lowering the mean epsilon from ~0.225 to ~0.19 decompresses the contrast back toward the -0.50/-0.52 band that iter 1/iter 3 delivered at lower lapse. (b) Exp 13 ceiling prevalence (0.2200 vs obs 0.2800) is a threshold statistic driven by the low-eps tail: per-subject saturation accuracy 1 - eps/2 rises from [0.865, 0.91] to [0.885, 0.925], and P(12/12 correct) rises from ~0.18-0.32 to ~0.23-0.39 across the skeptic band, moving the pooled prevalence toward ~0.26-0.30 against the observed 0.28. Critically, the coupling makes the saturated-conflict compliance (1-eps)*w + eps/2 = 0.75 + eps/2 approximately eps-invariant along the curve (~0.83-0.87 over the whole range, mean ~0.845 vs the iter-4 ~0.86), so the excellent Exp 11 (0.7812 vs 0.7875) and Exp 12 (0.8090 vs 0.8320) means are structurally protected -- this invariance is exactly what makes the eps reduction low-risk.

(2) COUNTER w RANGE RESTORED to the iter-2-validated [0.42, 0.56] (from [0.45, 0.62]), keeping f_counter = 0.13, counter eps [0.12, 0.22], and counter gamma >= 2. The iter-4 drift of the counter w range was the only substantive change from iter 2 on the Exp 6 assay, and it coincided with the regression from 0.2087 to 0.2475 (obs 0.1925): higher-w counters are less count-decisive on the matched-count gap/coalition cells. At 13% population mass the counters set no other mean, so this restoration is essentially risk-free elsewhere.

(3) NOTHING ELSE CHANGES: beta [1.3, 2.4], (c0, c1, kappa, alpha) ranges, the +/-0.02 w_spread, and the jitter-free population are all empirically validated across iterations -- the iter-2/iter-3 history shows that touching the skeptic w floor, the jitters, or beta destroys more than it gains. Per the critic's explicit instruction I do not chase the stable Exp 9/10 residuals (+0.04) or the between-subject variance shortfall (two dedicated attempts were gate-rejected; the gap is shared by every reference theory).

Projected movement versus the accepted base: Exp 3 error +0.091 -> ~+0.05, Exp 13 error -0.060 -> ~+0.00 to +0.02, Exp 6 error +0.055 -> ~+0.02, with Exps 1, 2, 4, 5, 7, 8, 9, 10, 11, 12 essentially unchanged (the compliance invariance protects 11/12; Exp 8 may tick up marginally as E[w] falls slightly toward ~0.91, which is the correct direction given its -0.035 residual). That is a clear net reduction in pooled point-estimate error at no variance cost, which should land below the 0.0545 accept-gate floor.

**Parameters:**
  - `c0`: `[0.06, 0.13]`
  - `c1`: `[1.00, 1.35]`
  - `kappa`: `[1.6, 2.1]`
  - `beta`: `[1.3, 2.4]`
  - `alpha`: `[0.0, 0.05]`
  - `u_pop`: `[0, 1]`
  - `w_spread`: `[-0.02, 0.02]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # Graded Dual-Route Distrust with a Bimodal Strategy Population
    # (GDRD-B) -- neutrality-coupled skeptic-mode variant, final tuning
    # pass per the validated iter-4 critique: skeptic epsilon lowered
    # to [0.15, 0.23] (decompresses the Exp-3 contrast, raises the
    # Exp-13 ceiling prevalence toward 0.28) with the neutrality
    # pairing re-solved over the new range, and the counter w range
    # restored to the iter-2-validated [0.42, 0.56] (pulls Exp 6 back
    # toward 0.21). Everything else is unchanged.
    #
    # Stimulus: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    #
    # ROUTE 1 (analytic distrust, weight w): every endorsement is a
    # LIABILITY whose magnitude grows with the endorser's RAW claimed
    # validity:
    #   d_j = c0 + c1 * v_j^kappa          (kappa >= 1, no dampening
    #                                        near v = 0.5)
    #   accumulated suspicion: P(X) = sum_j x_j * d_j
    #   D = (P(B) - P(A)) + alpha * sum_j clip(logit(v_j), 0, 3)*(a_j - b_j)
    #     (micro-trust term; the logit is CAPPED at 3 so perfect
    #      experts exert bounded, not unbounded, trust pull)
    #   P_skeptic(A) = sigmoid(beta * D)
    # ROUTE 2 (graded counting, weight 1-w): a CONTINUOUS preference
    #   for the option with fewer total endorsements:
    #   P_count(A) = sigmoid(gamma * (cnt_B - cnt_A))
    #
    # BIMODAL POPULATION: a single uniform population quantile u_pop
    # in [0, 1] maps onto a two-component mixture:
    #   u_pop < 0.13  -> 'counter' minority:  w in [0.42, 0.56],
    #                    epsilon in [0.12, 0.22], gamma in [2.0, 3.5]
    #   otherwise     -> 'skeptic' majority:   epsilon in [0.15, 0.23],
    #                    w COUPLED to epsilon along the neutrality curve
    #                        w = 1 - (0.25 - eps)/(1 - eps)
    #                    (clipped to [0.85, 0.95]) plus a tiny independent
    #                    per-subject spread w_spread in [-0.02, 0.02],
    #                    gamma in [1.5, 3.5].
    #
    # NEUTRALITY COUPLING: a skeptic's Exp-14 excess-dispersion
    # contribution is v = (x-0.21)^2 - (r-0.96)^2 with
    # x = (1-eps)(1-w) + eps/2 and r = 1 - eps/2. Setting x - 0.21 =
    # (r - 0.96) gives exactly (1-eps)(1-w) = 0.25 - eps, i.e.
    # w = 1 - (0.25 - eps)/(1 - eps). Tying w to eps along this curve
    # makes every skeptic's v approximately ZERO at ANY lapse level.
    # Re-solving over the new eps range [0.15, 0.23]: eps = 0.15 ->
    # w ~ 0.88, eps = 0.23 -> w ~ 0.97 (clipped at 0.95 -- the clip
    # only makes skeptic v slightly positive, the safe direction for
    # Exp 14). A key structural property: along the curve the
    # saturated-conflict compliance (1-eps)*w + eps/2 = 0.75 + eps/2
    # is approximately eps-invariant (~0.83-0.87 over the whole
    # range), so the excellent Exp 11/12 means are protected while
    # the epsilon reduction decompresses the lapse-sensitive assays
    # (Exp 3) and raises the Exp-13 ceiling prevalence.
    #
    # History is ignored: validities are communicated in the
    # instructions, so the distrust weights are fixed for the whole
    # block. beta is sampled per subject from [1.3, 2.4] (the range
    # validated by the iter-3 beta revert, which repaired Exps 3/10).
    if isinstance(state, dict):
        a_vec = np.asarray(state["option_a_ratings"], dtype=float).ravel()
        b_vec = np.asarray(state["option_b_ratings"], dtype=float).ravel()
        stim = np.vstack([a_vec, b_vec])
    else:
        stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"GDRD expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n_features = stim.shape[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.shape[0] != n_features:
        raise ValueError(
            f"validities length {v.shape[0]} != n_features {n_features}."
        )
    # Power route: v = 1 is harmless (no logs). Trust route: guard
    # the logit against v = 1 (perfect experts) and v < 0.5.
    v_pow = np.clip(v, 0.5, 1.0)
    v_tr = np.clip(v, 0.5, 1.0 - 1e-6)

    kappa = float(parameters["kappa"])
    beta = float(parameters["beta"])
    c0 = float(parameters["c0"])
    c1 = float(parameters["c1"])
    alpha = float(parameters["alpha"])
    u_pop = float(parameters["u_pop"])
    w_spread = float(parameters["w_spread"])

    # --- Bimodal population mixture over (w, epsilon, gamma) ---
    # Quantile transform of one uniform variable. No within-mode
    # jitters (they failed twice: contaminating Exp 6 and Exp 13
    # while moving no variance metric). Only a tiny +/-0.02 w spread
    # around the skeptic neutrality curve remains.
    f_counter = 0.13
    u = min(max(u_pop, 0.0), 1.0)
    if u < f_counter:
        # Counter minority: near-balanced routes, low lapse, decisive
        # graded count slope (gamma >= 2, per the theory's prose).
        # w restored to the iter-2-validated [0.42, 0.56]: the higher-w
        # counters of the iter-4 range were less count-decisive on
        # Exp 6's gap/coalition cells, elevating that assay.
        t = u / f_counter
        w_mix = 0.42 + 0.14 * t          # [0.42, 0.56]
        epsilon = 0.12 + 0.10 * t        # [0.12, 0.22]
        gamma = 2.0 + 1.5 * t            # [2.0, 3.5]
    else:
        # Skeptic majority: analytic-distrust dominant. The route
        # weight is COUPLED to the lapse along the Exp-14 neutrality
        # curve w = 1 - (0.25 - eps)/(1 - eps). The epsilon range is
        # LOWERED from [0.18, 0.27] to [0.15, 0.23] (mean ~0.19):
        # this decompresses the lapse-sensitive Exp-3 contrast back
        # toward -0.50/-0.52 and raises the Exp-13 ceiling prevalence
        # toward the observed 0.28, while the coupling keeps every
        # skeptic's Exp-14 v near zero and holds the saturated
        # compliance ~0.75 + eps/2 (eps-invariant along the curve),
        # protecting the Exp 11/12 means. Clipped to the theory's
        # skeptic band [0.85, 0.95].
        t = (u - f_counter) / (1.0 - f_counter)
        epsilon = 0.15 + 0.08 * t        # [0.15, 0.23]
        w_neutral = 1.0 - (0.25 - epsilon) / (1.0 - epsilon)
        w_mix = float(min(max(w_neutral + w_spread, 0.85), 0.95))
        gamma = 1.5 + 2.0 * t            # [1.5, 3.5]
    epsilon = float(min(max(epsilon, 0.02), 0.35))

    # --- Route 1: raw validity-proportional distrust charge ---
    d = c0 + c1 * np.power(v_pow, kappa)
    pen_a = float(np.dot(d, a))
    pen_b = float(np.dot(d, b))

    # Residual micro-trust: a small validity-weighted endorsement
    # edge. logit(0.5) = 0 (coin-flip experts contribute nothing) and
    # the logit is CAPPED at 3 so perfect experts (v = 1.0) exert a
    # bounded pull.
    lg = np.clip(np.log(v_tr / (1.0 - v_tr)), 0.0, 3.0)
    trust = float(np.dot(lg, a - b))

    # D > 0 favors A (B carries more accumulated suspicion, net of
    # the small trust edge).
    D = (pen_b - pen_a) + alpha * trust
    p_skeptic_a = 0.5 * (1.0 + np.tanh(0.5 * beta * D))  # stable sigmoid

    # --- Route 2: graded endorsement-count preference ---
    cnt_a = float(np.sum(a))
    cnt_b = float(np.sum(b))
    p_count_a = 0.5 * (1.0 + np.tanh(0.5 * gamma * (cnt_b - cnt_a)))

    # --- Dual-route mixture ---
    p_a = w_mix * p_skeptic_a + (1.0 - w_mix) * p_count_a
    p_a = float(min(max(p_a, 0.0), 1.0))
    p_core = np.array([p_a, 1.0 - p_a], dtype=float)

    # Independent lapse: with probability epsilon pick uniformly.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```


## Replacement

### `pi_10` → slot 2 (via `new_theory`)

**Description:** GDRD-T3 (Trimodal Calibrated-Tail Graded Dual-Route Distrust), variance-reduced minority modes. Subjects evaluate expert endorsements through two concurrent, history-free routes. ROUTE 1 (analytic distrust): every endorsement is a charge on its option whose magnitude grows with the endorser's raw claimed validity, d_j = c0 + c1*v_j^kappa; D = pen(B) - pen(A) + alpha_mode*sum_j clip(logit(v_j),0,3)*(a_j-b_j); P1(A) = sigmoid(beta*D). ROUTE 2 (graded counting): P2(A) = sigmoid(gamma*(cnt_B-cnt_A)). Mixture p = w*P1 + (1-w)*P2 plus independent lapse eps. The population is TRIMODAL via one uniform u_pop: (a) skeptic majority ~83% with heavy-tailed lapse (eps = 0.14+0.10*t^0.8 over [0.14,0.24]) and a route weight drawn from an INDEPENDENT second uniform quantile u_w (w in [0.83,0.95]); (b) counter minority ~13% with parameter-HOMOGENEOUS bands (w in [0.46,0.54], eps in [0.14,0.18], same means as before); (c) a mild trust-leaning mode ~4%, also parameter-homogeneous (alpha in [0.47,0.57], eps in [0.18,0.22], w in [0.47,0.53], gamma in [1.9,2.6]). The novelty this iteration: the minority modes are claimed to be behaviorally near-homogeneous within mode — the between-mode structure, not within-mode parameter scatter, carries the population dispersion. This is a falsifiable fifth register: if the minority modes were truly heterogeneous, the small-N population statistics (cross-subject slopes, population maxima) would show run-to-run swings far larger than binomial noise; homogenized modes predict draw-stable statistics centered between the two extreme draws already observed.

**Rationale:** This is the minimal-diff application of the iter-5 critique on the iter-3 accepted base, with exactly one new class of knob: WITHIN-MODE SPREAD COMPRESSION of the minority modes (variance reduction at fixed means/masses). The iter-5 event was decisive: a byte-identical restoration of iter 3 scored 0.0964 versus 0.0794 on a fresh draw, proving the run-to-run SD of the aggregate loss (~0.017) now exceeds every calibration margin, and the swing decomposed onto exactly the small-N population statistics (Exp 17: -0.32 vs -0.80; Exp 18: 2.30 vs 3.51; Exp 13: 0.26 vs 0.42; Exp 15: 0.95 vs 1.00) that are leveraged by the 0-2 trust subjects and 2-6 counter subjects a Binomial(25, 0.04)/Binomial(25, 0.13) draw delivers. Re-running an identical candidate is a coin flip at the noise floor, so the only defensible move is to shrink the sampling distribution of those statistics without moving their centers. Edits: (1) trust mode compressed to alpha [0.47,0.57], eps [0.18,0.22], w [0.47,0.53], gamma [1.9,2.6] — identical means to iter 3, half the spread; (2) counter mode compressed to w [0.46,0.54], eps [0.14,0.18] — identical means, same 13% mass; (3) skeptic branch, beta, c0/c1/kappa, and all mode masses untouched (skeptic eps was falsified in both narrowing directions in iter 4; the trust means were flanked to exhaustion in iters 1-3). Mechanistically, homogenized minority subjects sit at their cluster centers rather than at band edges, so whichever minority subjects a draw delivers produce near-modal lever arms: Exp 17's slope concentrates between the two observed draws (-0.32, -0.80) rather than at either, closer to the observed -0.599; Exp 18's slope+mean(y) concentrates near its ~2.3-2.5 center rather than swinging to 3.5; Exp 15's max stabilizes just below saturation as the arbiter's calibration logic requires. Favorable expected side effects: Exp 16's dispersion (0.182-0.195 vs obs 0.143) falls since within-mode spread feeds it, and Exp 14 (0.122-0.161 vs obs 0.115) is mode-separation-driven and unaffected. The pooled-rate registers (Exps 2,3,4,5,7,8,10,11) depend on mode means and masses, both unchanged, so they should hold. The new fifth falsifiable register — minority modes are parameter-homogeneous — is directly testable: if it is wrong, the small-N statistics will keep swinging on re-runs. Per the pre-registered stop condition, if this candidate also fails to beat 0.0794, iter 3 must be shipped as-is: the noise floor will have been definitively reached, and iter 3 already dominates pi_9 on Exps 1/17/18 and pi_8_1 on nearly every register.

**Parameters:**
  - `c0`: `[0.05, 0.14]`
  - `c1`: `[1.0, 1.3]`
  - `kappa`: `[1.6, 2.0]`
  - `beta`: `[1.5, 2.6]`
  - `alpha`: `[0.0, 0.05]`
  - `u_pop`: `[0, 1]`
  - `u_w`: `[0, 1]`
  - `w_spread`: `[-0.02, 0.02]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # GDRD-T3, iter-6: VARIANCE-REDUCED minority modes on the iter-3
    # accepted base, per the iter-5 critique. The iter-5 event proved
    # the run-to-run SD of the aggregate loss (~0.017) exceeds every
    # tuning margin: byte-identical code scored 0.0794 and 0.0964 on
    # two draws, with the swings concentrated in the small-N
    # population statistics (Exps 17/15/18/13) that are leveraged by
    # the 0-2 trust subjects and 2-6 counter subjects per sample.
    # The single untried in-family lever is WITHIN-MODE SPREAD
    # COMPRESSION of the minority modes, holding means and masses:
    #   (1) TRUST MODE: alpha [0.43,0.61] -> [0.47,0.57] (same ~0.52
    #       mean), eps [0.16,0.24] -> [0.18,0.22] (same 0.20 mean),
    #       w [0.44,0.56] -> [0.47,0.53], gamma [1.5,3.0] -> [1.9,2.6].
    #       4% mass EXACTLY (the calibration logic for Exps 1/17/18
    #       depends on it).
    #   (2) COUNTER MODE: w [0.42,0.58] -> [0.46,0.54], eps
    #       [0.12,0.20] -> [0.14,0.18], same means, same 13% mass.
    #   (3) Skeptic branch, beta, c0/c1/kappa boxes, and all mode
    #       masses: UNTOUCHED (skeptic eps narrowing was falsified in
    #       iter 4 in both directions; the trust branch means were
    #       flanked to exhaustion in iters 1-3).
    if isinstance(state, dict):
        a_vec = np.asarray(state["option_a_ratings"], dtype=float).ravel()
        b_vec = np.asarray(state["option_b_ratings"], dtype=float).ravel()
        stim = np.vstack([a_vec, b_vec])
    else:
        stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"GDRD-T3 expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n_features = stim.shape[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.shape[0] != n_features:
        raise ValueError(
            f"validities length {v.shape[0]} != n_features {n_features}."
        )
    v_pow = np.clip(v, 0.5, 1.0)          # power route: v=1 harmless
    v_tr = np.clip(v, 0.5, 1.0 - 1e-6)     # guard the logit

    kappa = float(parameters["kappa"])
    beta = float(parameters["beta"])
    c0 = float(parameters["c0"])
    c1 = float(parameters["c1"])
    alpha_base = float(parameters["alpha"])
    u_pop = float(parameters["u_pop"])
    u_w = float(parameters["u_w"])
    w_spread = float(parameters["w_spread"])

    # --- Trimodal population mixture (variance-reduced minorities) ---
    f_counter = 0.13
    f_trust_lo = 0.96
    u = min(max(u_pop, 0.0), 1.0)
    if u < f_counter:
        # Counter minority (13%), COMPRESSED: same means as the
        # validated band, tighter within-mode scatter. The saturated
        # coin-side registers (Exps 11/12/14) depend on the counter
        # MEAN, not its spread; the compression only stabilizes the
        # small-N slope statistics (Exps 17/18) against draw luck.
        t = u / f_counter
        w_mix = 0.46 + 0.08 * t          # [0.46, 0.54]
        epsilon = 0.14 + 0.04 * t        # [0.14, 0.18]
        gamma = 2.0 + 1.5 * t            # [2.0, 3.5]
        alpha_mode = alpha_base
    elif u < f_trust_lo:
        # Skeptic majority (83%): analytic distrust dominant. The
        # lapse is heavy-tailed (Beta-like t^0.8 shaping) over
        # [0.14, 0.24]. The route weight is drawn from an INDEPENDENT
        # uniform quantile u_w -- w and eps are statistically
        # independent across skeptics (banked since iter 2).
        t = (u - f_counter) / (f_trust_lo - f_counter)
        epsilon = 0.14 + 0.10 * (t ** 0.8)   # [0.14, 0.24], mean ~0.19
        tw = min(max(u_w, 0.0), 1.0)
        w_mix = 0.83 + 0.12 * tw + w_spread    # ~[0.83, 0.95], indep of eps
        w_mix = float(min(max(w_mix, 0.83), 0.95))
        gamma = 1.5 + 2.0 * t            # [1.5, 3.5]
        alpha_mode = alpha_base
    else:
        # Trust-leaning residual mode (4%, MILD and now HOMOGENIZED):
        # alpha [0.47, 0.57] and eps [0.18, 0.22] hold the iter-3
        # means (~0.52, ~0.20) with half the spread, so whichever 0-2
        # trust subjects land in a sample sit near the cluster center
        # rather than at the band edges. w and gamma similarly
        # compressed. This targets the demonstrated draw-to-draw
        # swings in Exps 17/15/18 without moving any mode mean.
        t = (u - f_trust_lo) / (1.0 - f_trust_lo)
        alpha_mode = 0.47 + 0.10 * t     # [0.47, 0.57]
        w_mix = 0.47 + 0.06 * t          # [0.47, 0.53]
        epsilon = 0.18 + 0.04 * t        # [0.18, 0.22]
        gamma = 1.9 + 0.7 * t            # [1.9, 2.6]
    epsilon = float(min(max(epsilon, 0.02), 0.45))

    # --- Route 1: validity-proportional charge plus micro-trust edge ---
    d = c0 + c1 * np.power(v_pow, kappa)
    pen_a = float(np.dot(d, a))
    pen_b = float(np.dot(d, b))

    # Micro-trust: logit(0.5) = 0 (coin-flip experts contribute nothing);
    # capped at 3 so perfect experts exert a bounded pull.
    lg = np.clip(np.log(v_tr / (1.0 - v_tr)), 0.0, 3.0)
    trust_edge = float(np.dot(lg, a - b))

    # D > 0 favors A.
    D = (pen_b - pen_a) + alpha_mode * trust_edge
    p_route1_a = 0.5 * (1.0 + np.tanh(0.5 * beta * D))  # stable sigmoid

    # --- Route 2: graded endorsement-count preference ---
    cnt_a = float(np.sum(a))
    cnt_b = float(np.sum(b))
    p_route2_a = 0.5 * (1.0 + np.tanh(0.5 * gamma * (cnt_b - cnt_a)))

    # --- Dual-route mixture ---
    p_a = w_mix * p_route1_a + (1.0 - w_mix) * p_route2_a
    p_a = float(min(max(p_a, 0.0), 1.0))
    p_core = np.array([p_a, 1.0 - p_a], dtype=float)

    # Independent lapse: with probability epsilon pick uniformly.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```
