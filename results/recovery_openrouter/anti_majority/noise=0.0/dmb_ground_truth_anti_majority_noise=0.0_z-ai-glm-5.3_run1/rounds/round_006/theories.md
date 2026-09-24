# Round 6 — Theories

**Verdict:** `new_model` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_7` — SURVIVED ✓

**Description:** Softened Validity-Distrust with Count Mixture (SVD). Every expert endorsement is a liability whose magnitude increases monotonically in the endorser's RAW claimed validity: d_j = c0 + c1*v_j^kappa, with no dampening near v=0.5 and no trust refund. Decision making is a dual-process mixture at the trial level: with probability w the subject is a 'skeptic' who integrates the accumulated-suspicion difference D = P(B) - P(A) through sigmoid(beta*D); with probability (1-w) the subject is a 'counter' who simply chooses the option with FEWER total endorsements (indifferent at matched counts). Two heterogeneity mechanisms soften the pooled extremes and capture between-subject variability: (i) the sigmoid slope beta is sampled from a WIDE per-subject range, so the population is a mixture of shallow and steep sigmoids — sensitive near D=0 (matched-count validity-composition trials) yet compressed at large |D| (unequal-count trials); (ii) the uniform lapse epsilon is also sampled from a wide per-subject range, which pulls every saturated large-D assay toward 0.5 while adding genuine between-subject dispersion in overall choice rates.

**Rationale:** This is a parameter-box-only minimal-diff edit of the ACCEPTED iter-5 base (loss 0.0481): predict/policy are re-emitted verbatim and only two ranges move, exactly per the critic's iter-5 diagnosis. (1) PRIMARY KNOB — epsilon raised and widened from [0.10, 0.16] to [0.12, 0.24] (mean ~0.18, inside the arbiter's 0.10-0.25 sketch). The iter-5 decomposition showed every remaining residual lives on the saturated large-D / unequal-count assays (Exp 1 +0.077, Exp 10 +0.031, Exp 4 -0.027, Exp 2 -0.026), where beta heterogeneity has no leverage because both mixture branches are pinned (the counter branch votes deterministically for the TTB winner on Exp 1's conflict trials and the skeptic's sigmoid is saturated for most of the beta range). The lapse is the only in-family knob that pulls saturated rates toward 0.5: it moves Exp 1 from 0.789 toward ~0.74-0.75, Exp 10 toward ~0.52, Exp 4 toward -0.20, Exp 2 toward -0.26, and even Exp 5 (rate 0.095, target 0.114) in the correct direction, at a small ~0.01-0.02 cost on the well-calibrated matched-count assays (6, 7, 8, 9). The wide per-subject spread also directly attacks the systematic 3-10x underprediction of between-subject variance on every experiment, which a narrow epsilon box cannot. (2) MICRO-KNOB — w ceiling raised from 0.90 to 0.92 only. The critic explicitly corrected the iter-4 guardrail's sign error: lowering w would shift weight onto the counter branch's deterministic 1.0 on Exp 1's conflict trials and RAISE Exp 1 further; raising w slightly shifts weight onto the skeptic branch (which sits below 1.0 there), mildly lowering Exp 1 and Exp 10 while sharpening the matched-count assays (Exp 8 up toward 0.78, Exp 9 down toward 0.37), partially offsetting the epsilon costs. The move is capped at +0.02 per the critic's instruction. (3) FROZEN — beta stays at the wide [1.2, 2.6] box that just closed Exps 7 and 8 (re-narrowing it would undo the loop's biggest win); c0, c1, kappa stay at the validated iter-3/5 charge geometry that produces correct signs and near-correct magnitudes on every validity-composition assay. Guardrails respected: if Exp 8 falls below ~0.75 or Exp 6 rises above ~0.24, the retreat is the epsilon ceiling (to ~0.19), not beta or w; if Exp 2 drifts below -0.30, back w off rather than epsilon. Architecture, mechanism family, and all function code are untouched.

**Parameters:**
  - `c0`: `[0.05, 0.12]`
  - `c1`: `[0.95, 1.3]`
  - `kappa`: `[1.5, 1.8]`
  - `beta`: `[1.2, 2.6]`
  - `w`: `[0.80, 0.92]`
  - `epsilon`: `[0.12, 0.24]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # Softened Validity-Distrust with Count Mixture (SVD).
    # Stimulus: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    #
    # Branch 1 (skeptic, prob w): each endorsement is a LIABILITY whose
    # magnitude grows with the endorser's RAW claimed validity:
    #   d_j = c0 + c1 * v_j^kappa   (kappa >= 1.5; NO dampening near
    #                                v = 0.5, NO trust refund)
    #   accumulated suspicion: P(X) = sum_j x_j * d_j
    #   D = P(B) - P(A)  (positive favors A)
    #   P_skeptic(A) = sigmoid(beta * D)
    # Branch 2 (counter, prob 1-w): pure tallying on endorsement COUNTS;
    #   choose the option with FEWER total endorsements; exactly
    #   indifferent (0.5) at matched counts.
    # Mixture: P_core(A) = w * P_skeptic(A) + (1-w) * P_count(A),
    # mixed with a uniform lapse epsilon.
    # History is ignored: validities are in the instructions, so the
    # distrust weights are fixed for the whole block. beta AND epsilon
    # are sampled per subject from WIDE ranges: between-subject
    # heterogeneity in the sigmoid slope and in the lapse rate are
    # explicit parts of the theory.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"SVD expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n_features = stim.shape[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.shape[0] != n_features:
        raise ValueError(
            f"validities length {v.shape[0]} != n_features {n_features}."
        )
    v = np.clip(v, 0.5, 1.0)  # defensive; no logs taken, v=1 harmless.

    kappa = float(parameters["kappa"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    c0 = float(parameters["c0"])
    c1 = float(parameters["c1"])
    w_mix = float(parameters["w"])

    # --- Skeptic branch: raw validity-proportional distrust charge ---
    d = c0 + c1 * np.power(v, kappa)
    pen_a = float(np.dot(d, a))
    pen_b = float(np.dot(d, b))

    # D > 0 favors A (B carries more accumulated suspicion).
    D = pen_b - pen_a
    x = beta * D
    p_skeptic_a = 0.5 * (1.0 + np.tanh(0.5 * x))  # stable sigmoid

    # --- Counter branch: pure endorsement-count tallying ---
    cnt_a = float(np.sum(a))
    cnt_b = float(np.sum(b))
    if cnt_a < cnt_b:
        p_count_a = 1.0      # A has fewer endorsements -> prefer A
    elif cnt_a > cnt_b:
        p_count_a = 0.0
    else:
        p_count_a = 0.5      # matched counts -> indifferent

    # --- Dual-process mixture ---
    p_a = w_mix * p_skeptic_a + (1.0 - w_mix) * p_count_a
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


### slot 2 — `pi_8` — KILLED ✗

**Description:** Graded Dual-Route Distrust with a Bimodal Strategy Population (GDRD-B). Subjects evaluate expert endorsements through two concurrent routes. ROUTE 1 (analytic distrust): every endorsement is a liability whose magnitude grows monotonically in the endorser's RAW claimed validity, d_j = c0 + c1*v_j^kappa; accumulated suspicion P(X) = sum_j x_j*d_j is compared across options through sigmoid(beta*(P(B)-P(A))), plus a small residual micro-trust edge alpha*sum_j clip(logit(v_j),0,3)*(a_j-b_j) (capped so perfect experts exert bounded, not unbounded, trust pull). ROUTE 2 (graded counting): a continuous preference for the fewer-endorsement option, sigmoid(gamma*(cnt_B-cnt_A)). The population is BIMODAL in the route weight w: a large majority (~88%) of 'skeptics' (w in [0.85, 0.95], moderate lapse) coexists with a minority (~12%) of 'counters' (w in [0.50, 0.62], low lapse, decisive count slope gamma >= 2). The mixture is implemented as a quantile transform of a single uniform per-subject population variable u_pop, which holds the population-MEAN route weight at its base value while exploding the between-subject behavioral variance — the empirically observed 3-12x super-binomial variance on the saturation assays (Exps 11, 12) that no uniform range-widening could reproduce.

**Rationale:** This edit follows the iter-4 critic's prescription directly, on top of the unchanged iter-1 GDRD base. (1) The logit cap clip(logit(v), 0, 3) in the micro-trust term is retained — it was validated twice (iter-2: Exp 12 = 0.8300; iter-4: 0.8275 vs real 0.8320) and is provably inert for all v <= 0.95, so it can only help Exp 12. (2) alpha, c1, kappa, c0, beta are kept EXACTLY at the iter-1 base values — three consecutive rejections have fully mapped the trade-offs of touching them, and the iter-4 postmortem showed those rejections were driven by evaluation noise swamping sub-noise edits, not by the cap. (3) THE SUBSTANTIVE CHANGE: the uniform w in [0.80, 0.92] (with independent epsilon, gamma) is replaced by a BIMODAL two-component population implemented as a quantile transform of a single uniform parameter u_pop in [0, 1]: a ~12% 'counter' minority at w in [0.50, 0.62] with low lapse [0.05, 0.10] and decisive gamma in [2.0, 3.0], and an ~88% 'skeptic' majority at w in [0.85, 0.95] with epsilon in [0.08, 0.18] and gamma in [1.5, 3.0]. The key structural property (and the reason iters 2-3's uniform widening failed while this should succeed) is that the population-MEAN route weight is held at the calibrated base value E[w] ~ 0.86, so the well-fit point estimates on Exps 2-5, 7, 9 are essentially undamaged, while the between-subject variance explodes on exactly the assays where the real variances exceed the binomial floor by 4-12x: Exp 11 (counters sit near 0.5-0.6 on saturated-conflict cells vs skeptics near 0.82, reproducing both the real mean ~0.7875 and pulling var from 0.012 toward the real 0.0537), Exp 12 (var from 0.005 toward the real 0.0438), Exp 10 (the counter pole pulls the mean from 0.55 toward the real 0.5096 AND adds spread), Exp 1 (counters follow the TTB winner via the count route, adding ~0.004 super-binomial variance at only ~+0.01 mean cost), Exp 8 (counters sit near chance on matched-count cells — variance gain, small mean cost). The counter pole was chosen over a high-lapse or high-alpha pole because it is directionally CONSISTENT with the distrust majority on Exps 5 and 6 (the count route also avoids the endorsement-majority/top-cue option), so those near-fit assays are undamaged. (4) Calibration choices per the critic: minority fraction f = 0.12; the majority's epsilon is lowered to [0.08, 0.18] (mean 0.13, above the ~0.08 floor below which Exp 5's mean would drop under the real 0.1143) to keep the Exp 11/12 skeptic-pole means at target. (5) The between-subject variance was the only remaining discrepancy an order of magnitude larger than the ~±0.005 evaluation noise floor; every point-estimate error left in the portfolio is <= 0.05 and three rejections have shown the point portfolio sits at its in-family Pareto frontier. This mixture is the one in-family mechanism that attacks the variance structure without moving the means, and it is MORE faithful to the arbiter's original GDRD sketch (which explicitly prescribed w down to 0.5 and wide per-subject heterogeneity) than the narrowed iter-1 box.

**Parameters:**
  - `c0`: `[0.06, 0.13]`
  - `c1`: `[1.00, 1.35]`
  - `kappa`: `[1.6, 2.1]`
  - `beta`: `[1.3, 2.5]`
  - `alpha`: `[0.0, 0.05]`
  - `u_pop`: `[0, 1]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # Graded Dual-Route Distrust, bimodal-population variant (GDRD-B).
    #
    # Stimulus: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    #
    # ROUTE 1 (analytic distrust, weight w): each endorsement is a
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
    # for the option with fewer total endorsements:
    #   P_count(A) = sigmoid(gamma * (cnt_B - cnt_A))
    #
    # BIMODAL POPULATION: instead of sampling w, gamma, epsilon from
    # independent uniform ranges (which shifts the population MEANS
    # whenever the ranges widen), a single uniform population quantile
    # u_pop in [0, 1] is deterministically mapped onto a two-component
    # mixture:
    #   u_pop < 0.12  -> 'counter' minority:  w in [0.50, 0.62],
    #                    epsilon in [0.05, 0.10], gamma in [2.0, 3.0]
    #   otherwise     -> 'skeptic' majority:   w in [0.85, 0.95],
    #                    epsilon in [0.08, 0.18], gamma in [1.5, 3.0]
    # This holds E[w] at the calibrated base value (~0.86) while
    # producing genuine between-subject bimodality — the mechanism the
    # real super-binomial variances on Exps 11/12 demand.
    #
    # History is ignored: validities are communicated in the
    # instructions, so the distrust weights are fixed for the whole
    # block. beta is sampled per subject from a wide range.
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

    # --- Bimodal population mixture over (w, epsilon, gamma) ---
    # Quantile transform of one uniform variable: the population mean
    # route weight stays at the calibrated base (~0.86) while the
    # between-subject distribution becomes bimodal.
    f_counter = 0.12
    u = min(max(u_pop, 0.0), 1.0)
    if u < f_counter:
        # Counter minority: near-balanced routes, low lapse, decisive
        # graded count slope.
        t = u / f_counter
        w_mix = 0.50 + 0.12 * t          # [0.50, 0.62]
        epsilon = 0.05 + 0.05 * t        # [0.05, 0.10]
        gamma = 2.0 + 1.0 * t            # [2.0, 3.0]
    else:
        # Skeptic majority: analytic-distrust dominant, moderate lapse.
        t = (u - f_counter) / (1.0 - f_counter)
        w_mix = 0.85 + 0.10 * t          # [0.85, 0.95]
        epsilon = 0.08 + 0.10 * t        # [0.08, 0.18]
        gamma = 1.5 + 1.5 * t            # [1.5, 3.0]
    w_mix = float(min(max(w_mix, 0.5), 0.95))

    # --- Route 1: raw validity-proportional distrust charge ---
    d = c0 + c1 * np.power(v_pow, kappa)
    pen_a = float(np.dot(d, a))
    pen_b = float(np.dot(d, b))

    # Residual micro-trust: a small validity-weighted endorsement
    # edge. logit(0.5) = 0 (coin-flip experts contribute nothing) and
    # the logit is CAPPED at 3 so perfect experts (v = 1.0) exert a
    # bounded pull (validated twice on the Exp-12 saturation assay).
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

### `pi_8_1` → slot 2 (via `new_model`)

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
