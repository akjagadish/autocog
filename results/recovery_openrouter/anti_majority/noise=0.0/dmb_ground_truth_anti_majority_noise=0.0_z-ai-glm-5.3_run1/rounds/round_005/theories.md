# Round 5 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

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


### slot 2 — `pi_5` — KILLED ✗

**Description:** Validity-Proportional Skepticism (VPS), a.k.a. Confidence-Distrust Defect Integration. Subjects treat every expert endorsement not as an asset but as a liability whose magnitude scales with the endorser's claimed validity: a highly valid expert's praise is a strong claim, and strong claims attract the most suspicion. On each trial the decision maker computes a penalty for each option, P(X) = sum_j x_j * (c0 + c1 * v_j^kappa), where c0 is a flat defect-count term (c1 = 0 recovers pure unweighted defect counting) and c1 * v_j^kappa is the confidence-distrust term, with penalty INCREASING in the endorser's validity and kappa controlling how steeply. The subject chooses the lower-penalty option via P(A) = sigmoid(beta * (P(B) - P(A))) mixed with a uniform lapse epsilon. History is ignored because validities are communicated in the instructions, so skepticism weights are fixed for the whole block. VPS reproduces the sign of every observed metric while fixing the one structural blind spot of pure defect counting (SDI): on matched-count trials with a large validity gap (e.g., a single 0.90 endorser vs a single 0.60 endorser), SDI is pinned at 0.5 while VPS predicts a clear majority for the LOWER-validity endorser — exactly what the Experiment 6 composite (observed 0.193) requires.

**Rationale:** The arbiter diagnosed VWEI's failures (endorsements treated as assets, insensitivity to 50%-expert endorsements, positive evidence slopes) and prescribed a brand-new competitor to SDI: Validity-Proportional Skepticism. I implement that mechanism faithfully. (1) Core mechanism: penalty P(X) = sum_j x_j * (c0 + c1 * v_j^kappa), choice = lower penalty via sigmoid(beta*(P(B)-P(A))) with lapse epsilon. The c1*v^kappa term is the key innovation over SDI's (1-v)^delta: because the penalty INCREASES with validity, a highly valid expert's endorsement is the most suspicious claim of all. (2) Hand-verified behavior at the range center (kappa=2, beta=2, epsilon=0.1, c0=0.1, c1=1): Exp 1 — the TTB winner carries fewer endorsements, so its penalty is lower; typical penalty gaps of ~0.3-0.9 give P(TTB winner) ~0.65-0.70 against the observed 0.7117, slightly better than SDI's overshoot (0.7792). Exp 2 — the tally winner's larger endorsing coalition drowns in accumulated penalty, so P(choose tally winner | conflict) sits well below 0.5, matching the observed -0.2562. Exp 3 — on HIGH trials the 0.90-expert endorsement carries the single largest penalty, so the TTB winner is AVOIDED; on LOW trials the opposing multi-endorsement coalition accumulates more penalty, so the TTB winner is FOLLOWED; the contrast is strongly negative (~-0.5), matching the observed -0.5467 where SDI undershot (-0.415). Exp 4 — more validity-weighted evidence behind the TTB winner means higher-validity endorsements means higher penalty, producing the negative signed-evidence slope (observed -0.1938). Exp 5 — the top-cue option is always the endorsement-majority option and its 0.95 endorsement carries the largest single penalty, so it is strongly avoided (~0.11-0.15), matching the observed 0.1143. Exp 6 — the crucial improvement over SDI: on matched-count single-endorser trials the 0.90 endorser is penalized more than the 0.60 endorser (c0 + c1*0.81 vs c0 + c1*0.36 at kappa=2), pulling the gap component below 0.5 (~0.35-0.40) where SDI is structurally pinned at ~0.5; the coalition component stays low (~0.05-0.10), so the composite lands near ~0.20-0.23 versus the observed 0.1925 (SDI: 0.304). (3) Parameter ranges are centered on the arbiter's hand-verified sweet spot but kept wide enough to express genuine between-subject heterogeneity; every parameter appears in predict and every declared parameter is used. (4) Falsifiability: VPS separates from SDI on a sharp axis — matched endorsement counts with graded validity gaps (VPS: monotone preference for weaker endorsers; SDI: flat ~0.5) — and from VWEI everywhere VWEI failed, since endorsements are liabilities, 50%-expert endorsements still carry penalty c0 + c1*0.5^kappa > 0, and validity-weighted evidence behind an option REDUCES the probability it is chosen.

**Parameters:**
  - `kappa`: `[1.6, 2.4]`
  - `beta`: `[1.6, 2.4]`
  - `epsilon`: `[0.07, 0.13]`
  - `c0`: `[0.05, 0.15]`
  - `c1`: `[0.7, 1.3]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # Validity-Proportional Skepticism (VPS) /
    # Confidence-Distrust Defect Integration.
    #
    # Stimulus: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    #
    # Each endorsement is a LIABILITY whose magnitude scales with the
    # endorser's claimed validity (a strong claim attracts suspicion):
    #   penalty for option X:  P(X) = sum_j x_j * (c0 + c1 * v_j^kappa)
    #     c0        -> flat defect-count term (c1 = 0 recovers pure
    #                  unweighted endorsement counting)
    #     c1*v^kappa -> confidence-distrust term, INCREASING in validity
    # Decision variable:  D = P(B) - P(A)   (positive favors A)
    # P(A) = sigmoid(beta * D), mixed with a uniform lapse epsilon.
    # History is ignored: validities are in the instructions, so the
    # skepticism weights are fixed for the whole block.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"VPS expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n_features = stim.shape[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.shape[0] != n_features:
        raise ValueError(
            f"validities length {v.shape[0]} != n_features {n_features}."
        )
    # Validities live in [0.5, 1.0]; clip defensively (no logarithms
    # are taken, so v = 1 is harmless here).
    v = np.clip(v, 0.5, 1.0)

    kappa = float(parameters["kappa"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    c0 = float(parameters["c0"])
    c1 = float(parameters["c1"])

    # Per-endorser liability: flat suspicion plus a
    # validity-proportional confidence-distrust charge.
    d = c0 + c1 * (v ** kappa)

    # Total accumulated suspicion for each option.
    pen_a = float(np.dot(d, a))
    pen_b = float(np.dot(d, b))

    # D > 0 favors A (B carries more accumulated suspicion than A).
    D = pen_b - pen_a

    # Numerically stable sigmoid: 0.5 * (1 + tanh(x/2)).
    # D == 0 (exact tie in penalties) -> exactly 0.5.
    x = beta * D
    p_a = 0.5 * (1.0 + np.tanh(0.5 * x))
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

### `pi_8` → slot 2 (via `new_theory`)

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
