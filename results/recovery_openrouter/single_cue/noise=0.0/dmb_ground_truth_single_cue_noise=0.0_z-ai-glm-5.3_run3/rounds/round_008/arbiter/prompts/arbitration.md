# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_7" and "pi_9") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_7" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_9" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_7
PRIMACY-GATED DISTRUST TALLYING (PGDT), floored-gate edition with a rebalanced distrust share. Subjects scan the expert ratings in presentation order and attention is deployed through a steep convex primacy gate with a substantial nonzero floor: g_j = delta + (1-delta)*((n-j)/(n-1))^eta, so the first one or two experts carry the dominant share of attention but late experts retain a meaningful attention residue. Independently, the communicated validities are distrust-inverted exactly as in the shared kernel: subjects act as if each stated validity were the expert's error rate, inv_j = ((1-v_j) - c*min_k(1-v_k))^gamma / Z. The two kernels combine MULTIPLICATIVELY: w_j proportional to g_j * [(1-tau)/n + tau*inv_j], normalized. The refinement this iteration is a rebalanced distrust share: tau is lowered to [0.82, 0.92] so the flat component (1-tau)/n of the multiplicative bracket grows from ~1.5% to ~4-5% per cue. This directly feeds trusted high-validity LATE experts (whose bracket value is structurally pinned at the flat share regardless of gate shape), softens the distrust-driven allegiance contrasts that overshoot their targets, and pulls the effective weights off pure anti-validity where that overshoots. A modestly raised gate floor delta in [0.20, 0.45] compensates on decreasing-validity layouts by further concentrating the product weights onto the back-loaded low-validity experts, protecting the anti-validity regression. Evidence is integrated additively over all discriminating cues, d = sum_j w_j * sign(A_j - B_j), and choice is logistic in beta*d with a small uniform lapse epsilon.

`predict(parameters, state, history) -> np.ndarray`:
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

`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # guard against float drift
    return int(np.random.choice(len(probs), p=probs))

## THEORY 2 — pi_9
POSITION-BRIDGED DISTRUST TALLYING (PBDT), un-bundled-gate edition. Subjects scan expert ratings in presentation order through a steep floored convex primacy gate g_j = delta + (1-delta)*((n-j)/(n-1))^eta, with the gate restored to the steep end of its box (eta high, delta low) so late low-validity helper cues are crushed relative to early ones — the structural lever separating primacy-rescue trials from validity-position-confounded layouts. Stated validities are distrust-inverted (treated as error rates) via inv_j = ((1-v_j) - c*min_k(1-v_k))^gamma / Z, kept at the steep bridged range [2.6, 2.9] with a high distrust share tau, so the anti-validity gradient in confounded layouts is carried by the distrust kernel alone rather than by the gate. Trust in high-validity experts ramps with position from a small-but-nonzero floor t_j = t0 + (t1-t0)*(j/(n-1))^kappa, with a slightly concave-leaning ramp (kappa below 1) so late trusted experts recover nearly the full flat share without raising the early floor. Weights combine multiplicatively, w_j proportional to g_j * [t_j*(1-tau)/n + tau*inv_j]; evidence d = sum_j w_j*sign(A_j - B_j) is integrated additively over discriminating cues; choice is logistic in beta*d with a uniform lapse epsilon, with noise set between the two previous iterations so slope/contrast metrics that were over-attenuated recover while anchor allegiances keep their margin.

`predict(parameters, state, history) -> np.ndarray`:
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

`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # guard against float drift
    return int(np.random.choice(len(probs), p=probs))

## EXPERIMENT 1 (proposed by pi_7)

### DESIGN
**Validities (n_features=6):** [0.95, 0.85, 0.6, 0.6, 0.85, 0.95]

**Trial pairs (n=24):**
  trial 1: A=[1, 1, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 2: A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  trial 3: A=[1, 1, 1, 1, 0, 1]  B=[1, 0, 1, 1, 1, 1]
  trial 4: A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 1]
  trial 5: A=[1, 1, 1, 0, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  trial 6: A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 1, 1]
  trial 7: A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 8: A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  trial 9: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 10: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 11: A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 0, 1]
  trial 12: A=[0, 1, 1, 1, 0, 1]  B=[1, 0, 1, 1, 1, 1]
  trial 13: A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  trial 14: A=[1, 1, 1, 1, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  trial 15: A=[1, 1, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]
  trial 16: A=[0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 1, 1]
  trial 17: A=[1, 1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1]
  trial 18: A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  trial 19: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 20: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  trial 21: A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  trial 22: A=[1, 1, 1, 1, 1, 0]  B=[1, 0, 1, 1, 1, 1]
  trial 23: A=[0, 1, 1, 1, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  trial 24: A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 1, 1, 0, 1]

**Rationale:** STRUCTURAL LEVER. PGDT and PBDT share the distrust-inversion kernel and a floored convex primacy gate; they differ in exactly two quantitatively dissociable respects. (1) THE FLAT (TRUSTED) SHARE. PGDT's bracket is (1-tau)/n + tau*inv_j with tau in [0.82, 0.92]: a POSITION-INDEPENDENT flat share of 1.3-3.0% per cue (n=6). PBDT's bracket is t_j*(1-tau)/n + tau*inv_j with tau in [0.91, 0.93] and a trust ramp t_j rising from t0 in [0.25, 0.35] to t1 in [0.95, 1.0]. Consequence, verified at the corners of both boxes: the raw weight of an EARLY high-validity (95%) expert is DISJOINT across the theories -- PGDT [0.013, 0.031] vs PBDT [0.004, 0.007], a 2-7x gap -- because PBDT's t0 crushes the flat share exactly where the gate is maximal. For any cue whose bracket is flat-share-relevant (validity 0.85-0.95), PBDT's ramp CANCELS its gate, so PBDT's position effects on the trusted/mid channel collapse toward zero, while PGDT's remain gate-scaled. (2) GATE SHAPE. PGDT's floor is high (delta 0.20-0.45, eta 4-6); PBDT's gate is steeper with a lower floor (delta 0.15-0.25, eta 6.5-8). On tail-position contrasts among kernel cues (positions 3 vs 4), PGDT's gate contrast (1-delta)(0.6^eta-0.4^eta) reaches 2.5x PBDT's maximum. LAYOUT: palindromic v = [0.95, 0.85, 0.60, 0.60, 0.85, 0.95]. The palindrome creates VALIDITY-MATCHED position duels at three validity levels -- 95s at positions 1 vs 6, 85s at 2 vs 5, 60s at 3 vs 4 -- so the distrust kernel cancels exactly within each duel and choice is decided by gate x flat-share architecture alone. All boxes below are mirror-corrected P(A), verified at the corners of both parameter spaces (tau x eta x delta x gamma x c, with beta/epsilon at worst-case ends).

AXIS 1 -- VALIDITY-MATCHED POSITION-TRANSFER BATTERY (T16, M25, L34 + mirrors). T16 (95@1 vs 95@6): PGDT d = b_tr*(1-delta) in [0.0075, 0.025] raw -> P(A) in [0.51, 0.66]; PBDT d = flat'*(t0 - delta*t1) + leakage in [0.0004, 0.0033] -> P(A) in [0.50, 0.52]. PBDT is HARD-CAPPED at ~0.52: observed P(early-95 product) >= 0.55 is impossible for PBDT at any admissible parameter combination. M25 (85@2 vs 85@5): PGDT [0.51, 0.60]; PBDT [0.50, 0.55] (its ramp cancels its gate for the mid channel; the flat-share term straddles zero). L34 (60@3 vs 60@4): PGDT [0.52, 0.70]; PBDT [0.51, 0.62] -- observed >= 0.63 kills PBDT. PGDT requires a robustly nonzero transfer effect at ALL THREE validity levels; PBDT requires near-chance on the 95- and 85-level duels. Flatness at T16/M25 falsifies PGDT; any effect >= 0.55 at T16 or >= 0.57 at M25 falsifies PBDT's entire space.

AXIS 2 -- BLOCK-FLIP STAIRCASE (T16 -> BLK2 -> BLK3 + mirrors). BLK2 (A wins {95@1, 85@2} vs B wins {85@5, 95@6}, 60s tied): PGDT P in [0.51, 0.70]; PBDT [0.50, 0.55] -- raw-d boxes DISJOINT ([0.011, 0.044] vs [0.000, 0.011]). BLK3 (full 3v3 block flip): PGDT [0.53, 0.82]; PBDT [0.52, 0.66]. The within-subject staircase increments are the signature: PGDT's T16->BLK2 increment (adding the 85@2/85@5 pair) is robustly POSITIVE at every corner (raw [0.004, 0.019]) because the flat share x gate gives the early mid expert real weight; PBDT's increment STRADDLES ZERO ([-0.000, 0.008]) because its ramped flat share cancels the gate. Observed mirror-corrected P(BLK2) - P(T16) >= 0.04 is impossible for PBDT and required by PGDT's upper box; a flat or negative increment falsifies PGDT.

AXIS 3 -- EARLY-TRUSTED RESCUE AND POSITION TRANSFER (X1, X3 + mirrors). X1 (A wins {95@1, 85@5} vs B wins {85@2}): the two 85 endorsements cancel in the distrust kernel, so the 95@1 expert's flat-share weight decides. PGDT: d in [+0.002, +0.015] at every corner -> P(A) in [0.51, 0.60] (the early 95 RESCUES product A). PBDT: d in [-0.004, +0.007] -> straddles chance. X3 (A wins {85@5, 95@6} vs B wins {85@2}) is the position-transfer twin: the same 95 endorsement moved from position 1 to position 6. The transfer difference X1 - X3 equals b_tr*(1-delta) under PGDT -- robustly in [0.0075, 0.025] raw, P-difference >= 0.02 and up to 0.20 -- while under PBDT it equals flat'*(t0 - delta*t1) + leakage, confined to [0.0004, 0.0033] raw (P-difference <= 0.01). A reliable X1-vs-X3 difference >= 0.04 falsifies PBDT's entire space; a zero or negative difference falsifies PGDT.

AXIS 4 -- ANCHORS AND SCREENS. DINV (the product endorsed by all four 95/85 experts vs the product endorsed by the two 60 experts): BOTH theories confidently prefer the 60-product (PGDT d in [-0.39, -0.09]; PBDT [-0.24, -0.09]) -- verifies that subjects read and distrust-invert the stated validities, and excludes normative weighting, TTB (follows expert 1), and tallying (4 cues beat 2). VETO (95@1 vs 60@3): both prefer the 60-product ([-0.23, -0.07] / [-0.13, -0.06]) -- the single BEST expert's product loses to a mid-list distrusted expert's product. TOT: comprehension anchor (both ~0.95). R1 (95@6 vs 85@2) and X2 are agreement/fitting cells where both theories land on the same side (B, and straddle, respectively), constraining gate/ramp/beta jointly.

AGGREGATE SIGNATURE. Mirror-corrected P(A) regressed on PGDT reference evidence (g_j*[(1-tau)/n + tau*inv_j] at tau=0.87, eta=5, delta=0.3) versus PBDT reference evidence (g_j*[t_j*(1-tau)/n + tau*inv_j] at t0=0.30, kappa=1.0) loads with opposite profiles: the two evidence axes are anti-correlated on the X1/X3/BLK family by construction (position-1 trusted weight is maximal under the first, ~zero under the second). Single-number readouts: (i) family mean over {T16, M25, BLK2, X1} is hard-capped at 0.55 by PBDT (its maximum over all corners) while PGDT's central mass is 0.55-0.70; (ii) the transfer battery slope (effect size vs validity level: 95-duel, 85-duel, 60-duel) is strictly positive at every PGDT corner and confined to [~0, 0.05] under PBDT for the top two rungs; (iii) the X1-X3 transfer difference. Every cell type appears in A/B-mirrored form to absorb side and position bias. 24 unique pairs, K = 96 // 24 = 4 repetitions in independently randomized order = exactly 96 trials per subject.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



### METRIC
Rationale:
WHY THE PREVIOUS ATTEMPT FAILED. The rejected metric was a NORMALIZED (mean-type) weighted contrast. Hand-analysis of the two predict functions shows that with 96 near-chance binary trials the per-subject binomial noise floor (~0.10-0.13 SD) is large relative to the box-averaged choice-probability gap (~0.08), so any per-subject-normalized statistic caps at t~2.2-2.3 at N=25 -- below the 0.01-level threshold. The information is there, but a mean-type summary throws away the accumulation structure of evidence.

NEW METRIC: TOTAL BOX-MARGINAL LOG-LIKELIHOOD RATIO (log Bayes factor). The metric embeds exact replicas of both predict functions, integrates each over its FULL admissible parameter box (all 2^k corners + box centre + 192 seeded uniform interior draws; 321 points for PGDT, 1217 for PBDT), obtaining each theory's MARGINAL choice probability for each of the 24 unique pairs. For every trial in the DataFrame it adds the Bernoulli log-likelihood ratio log P_PBDT/PGDT(chose A) or log((1-P_PGDT)/(1-P_PBDT)) for a B choice, and returns the TOTAL. This is the Neyman-Pearson/Bayes-optimal discriminator of the two composite hypotheses: it is literally the log Bayes factor accumulated over all observed choices.

KEY DESIGN PROPERTIES. (1) ADDITIVITY: log-likelihoods add over trials and subjects. On a single-subject slice the metric is that subject's personal log-BF (a well-defined, bounded, binomial-limited quantity -- small between-subject spread); on the pooled DataFrame it is the TOTAL evidence, which accumulates linearly in the number of subjects while the between-subject spread stays fixed. This is exactly the accumulation property a likelihood-ratio statistic is supposed to have, and it is what the previous normalized version destroyed. (2) AUTOMATIC WEIGHTING: trials where the theories agree (DINV, VETO, and -- after box-averaging -- TOT, whose marginal probabilities nearly coincide) receive near-zero log-likelihood weights; the discrimination is carried by the cells where the theories make OPPOSITE-signed predictions at every corner of both boxes: the early-trusted rescue pair X1/X2 (PGDT's position-independent flat share (1-tau)/n gives the early 95% expert real weight, so the early-95 product is rescued, m in [+0.01,+0.09]; PBDT's trust ramp t0 in [0.25,0.35] crushes the early flat share, m in [-0.06,-0.04]; X2 is the sign-mirror), plus the position-transfer cells M25/R1/X3, the gate-shape cells L34/BLK3, and T16. (3) ROBUSTNESS: weights are box-marginals, not point estimates, so the metric does not depend on where in its box each theory is simulated; mirror trials are handled automatically (each pair's own probabilities are used, absorbing side/position bias).

EXPECTED BEHAVIOUR. Under PGDT-simulated data the per-trial expected increment is +KL(PGDT||PBDT) (positive, ~0.2-0.4 nats per 96-trial subject); under PBDT-simulated data it is -KL(PBDT||PGDT) (negative, similar magnitude). Hence the pooled point estimate is decisively positive for PGDT data and decisively negative for PBDT data, while per-subject values (each an independent log-BF with SD ~0.7-1.0) remain stable -- the two simulated theories separate by a large multiple of the between-subject spread at N=25. Sign convention: positive = PGDT (flat-share early-trusted weighting), negative = PBDT (ramped early trust).

Source:
import numpy as np
import pandas as pd

_CACHE = {}

_TRIALS = [
    ((1, 1, 1, 1, 1, 0), (0, 1, 1, 1, 1, 1)),
    ((0, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 0)),
    ((1, 1, 1, 1, 0, 1), (1, 0, 1, 1, 1, 1)),
    ((1, 0, 1, 1, 1, 1), (1, 1, 1, 1, 0, 1)),
    ((1, 1, 1, 0, 1, 1), (1, 1, 0, 1, 1, 1)),
    ((1, 1, 0, 1, 1, 1), (1, 1, 1, 0, 1, 1)),
    ((1, 1, 1, 1, 0, 0), (0, 0, 1, 1, 1, 1)),
    ((0, 0, 1, 1, 1, 1), (1, 1, 1, 1, 0, 0)),
    ((1, 1, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1)),
    ((0, 0, 0, 1, 1, 1), (1, 1, 1, 0, 0, 0)),
    ((1, 0, 1, 1, 1, 1), (0, 1, 1, 1, 0, 1)),
    ((0, 1, 1, 1, 0, 1), (1, 0, 1, 1, 1, 1)),
    ((1, 0, 1, 1, 1, 1), (1, 1, 1, 1, 0, 0)),
    ((1, 1, 1, 1, 0, 0), (1, 0, 1, 1, 1, 1)),
    ((1, 1, 0, 0, 1, 1), (0, 0, 1, 1, 0, 0)),
    ((0, 0, 1, 1, 0, 0), (1, 1, 0, 0, 1, 1)),
    ((1, 1, 0, 1, 1, 1), (0, 1, 1, 1, 1, 1)),
    ((0, 1, 1, 1, 1, 1), (1, 1, 0, 1, 1, 1)),
    ((1, 1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0)),
    ((0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1, 1)),
    ((1, 0, 1, 1, 1, 1), (1, 1, 1, 1, 1, 0)),
    ((1, 1, 1, 1, 1, 0), (1, 0, 1, 1, 1, 1)),
    ((0, 1, 1, 1, 0, 1), (1, 0, 1, 1, 1, 0)),
    ((1, 0, 1, 1, 1, 0), (0, 1, 1, 1, 0, 1)),
]

_V = np.array([0.95, 0.85, 0.60, 0.60, 0.85, 0.95])
_A = np.array([t[0] for t in _TRIALS], dtype=float)
_B = np.array([t[1] for t in _TRIALS], dtype=float)
_SIGN = np.sign(_A - _B)

_PG_LO = np.array([0.82, 4.0, 0.20, 2.8, 0.10, 2.5, 0.06])
_PG_HI = np.array([0.92, 6.0, 0.45, 3.5, 0.25, 6.0, 0.15])
_PB_LO = np.array([0.91, 6.5, 0.15, 2.6, 0.05, 0.25, 0.95, 0.80, 3.6, 0.11])
_PB_HI = np.array([0.93, 8.0, 0.25, 2.9, 0.25, 0.35, 1.00, 1.50, 4.4, 0.14])


def _pgdt_pvec(par):
    tau, eta, delta, gamma, c, beta, eps = par
    tau = float(tau); eta = float(eta); delta = float(delta)
    gamma = float(gamma); c = float(c); beta = float(beta); eps = float(eps)
    n = 6
    err = 1.0 - _V
    centered = np.maximum(err - c * float(err.min()), 0.0)
    inv = np.power(centered, gamma)
    Zv = float(inv.sum())
    if Zv > 1e-12:
        w_inv = inv / Zv
    else:
        w_inv = np.full(n, 1.0 / n)
    pos = np.arange(n, dtype=float)
    g = delta + (1.0 - delta) * np.power((n - pos) / (n - 1.0), eta)
    bracket = (1.0 - tau) / n + tau * w_inv
    raw = g * bracket
    w = raw / float(raw.sum())
    d = _SIGN.dot(w)
    p = 0.5 * (1.0 + np.tanh(0.5 * beta * d))
    return (1.0 - eps) * p + eps * 0.5


def _pbdt_pvec(par):
    tau, eta, delta, gamma, c, t0, t1, kappa, beta, eps = par
    tau = float(tau); eta = float(eta); delta = float(delta)
    gamma = float(gamma); c = float(c); t0 = float(t0); t1 = float(t1)
    kappa = float(kappa); beta = float(beta); eps = float(eps)
    n = 6
    err = 1.0 - _V
    centered = np.maximum(err - c * float(err.min()), 0.0)
    inv = np.power(centered, gamma)
    Zv = float(inv.sum())
    if Zv > 1e-12:
        w_inv = inv / Zv
    else:
        w_inv = np.full(n, 1.0 / n)
    pos = np.arange(n, dtype=float)
    t = t0 + (t1 - t0) * np.power(pos / (n - 1.0), kappa)
    g = delta + (1.0 - delta) * np.power((n - pos) / (n - 1.0), eta)
    bracket = t * (1.0 - tau) / n + tau * w_inv
    raw = g * bracket
    w = raw / float(raw.sum())
    d = _SIGN.dot(w)
    p = 0.5 * (1.0 + np.tanh(0.5 * beta * d))
    return (1.0 - eps) * p + eps * 0.5


def _make_grid(lo, hi, n_random=192, seed=1234567):
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    k = len(lo)
    m = 2 ** k
    idx = np.arange(m)
    bits = ((idx[:, None] >> np.arange(k)[None, :]) & 1).astype(float)
    corners = lo[None, :] * (1.0 - bits) + hi[None, :] * bits
    center = ((lo + hi) / 2.0)[None, :]
    rng = np.random.default_rng(seed)
    rnd = rng.uniform(lo, hi, size=(n_random, k))
    return np.vstack([corners, center, rnd])


def _weights():
    if 'wmap' in _CACHE:
        return _CACHE['wmap']
    pg_grid = _make_grid(_PG_LO, _PG_HI)
    pb_grid = _make_grid(_PB_LO, _PB_HI)
    nt = len(_TRIALS)
    p_pg = np.zeros(nt)
    for par in pg_grid:
        p_pg += _pgdt_pvec(par)
    p_pg /= float(len(pg_grid))
    p_pb = np.zeros(nt)
    for par in pb_grid:
        p_pb += _pbdt_pvec(par)
    p_pb /= float(len(pb_grid))
    p_pg = np.clip(p_pg, 0.02, 0.98)
    p_pb = np.clip(p_pb, 0.02, 0.98)
    lA = np.log(p_pg / p_pb)
    lB = np.log((1.0 - p_pg) / (1.0 - p_pb))
    wmap = {}
    for i in range(nt):
        wmap[_TRIALS[i]] = (float(lA[i]), float(lB[i]))
    _CACHE['wmap'] = wmap
    return wmap


def _to_tuple(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return tuple(int(round(float(v))) for v in x)
    s = str(x).replace("'", "").replace('"', '')
    s = s.strip().strip('[](){} ').replace(',', ' ')
    return tuple(int(round(float(p))) for p in s.split())


def metric(data: pd.DataFrame) -> float:
    wmap = _weights()
    if len(data) == 0:
        return 0.0
    ra = data['option_a_ratings'].apply(_to_tuple)
    rb = data['option_b_ratings'].apply(_to_tuple)
    resp = pd.to_numeric(data['response']).to_numpy()
    total = 0.0
    for (a, b), r in zip(zip(ra, rb), resp):
        pr = wmap.get((a, b))
        if pr is None:
            continue
        total += pr[0] if int(r) == 0 else pr[1]
    return float(total)

### RESULTS
- Predicted under pi_7 (simulated): 12.6306 (var=0.4648)
- Predicted under pi_9 (simulated): -8.3125 (var=0.4421)
- Observed on real data: -9.1163 (var=0.3521)

## EXPERIMENT 2 (proposed by pi_9)

### DESIGN
**Validities (n_features=6):** [0.95, 0.95, 0.75, 0.75, 0.6, 0.95]

**Trial pairs (n=32):**
  trial 1: A=[0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0]
  trial 2: A=[1, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 3: A=[0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  trial 4: A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 5: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  trial 6: A=[0, 0, 0, 1, 0, 1]  B=[0, 0, 1, 0, 0, 0]
  trial 7: A=[0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 1]
  trial 8: A=[1, 0, 0, 1, 0, 1]  B=[0, 0, 1, 0, 0, 0]
  trial 9: A=[0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1]
  trial 10: A=[0, 1, 0, 1, 0, 1]  B=[0, 0, 1, 0, 0, 0]
  trial 11: A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 12: A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 1, 0, 0, 0]
  trial 13: A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 14: A=[0, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 15: A=[0, 0, 1, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0]
  trial 16: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 1]
  trial 17: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 18: A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 19: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 20: A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  trial 21: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 22: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 23: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 24: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 25: A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 26: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 27: A=[1, 1, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  trial 28: A=[0, 0, 0, 0, 1, 0]  B=[1, 1, 0, 0, 0, 1]
  trial 29: A=[1, 1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 0]
  trial 30: A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1]
  trial 31: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 32: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Rationale:** STRUCTURAL LEVER. PBDT and PGDT share the distrust-inversion kernel and a floored convex primacy gate; they differ in exactly one quantitatively decisive respect: the TRUSTED (flat-share) channel. PGDT's bracket is (1-tau)/n + tau*inv_j -- POSITION-INDEPENDENT -- so an early high-validity expert's raw weight is g_1*flat: the full amplified position-1 gate (n/(n-1))^eta times the FULL flat share. PBDT multiplies the flat share by a trust ramp t_j rising from t0 in [0.25, 0.35] at position 1 to t1 in [0.95, 1.0] at position n, so its position-1 trusted weight is g_1*t0*flat -- crushed by exactly the factor t0, a PARAMETER-FREE 2.9-4.0x reduction relative to PGDT (1/t0 at every corner), while late trusted experts (t1~1) are left nearly intact. PGDT's trusted profile is g_j (monotone, position-1 dominant); PBDT's is g_j*t_j. Prior design [16] probed this with a palindromic layout and pure trusted-expert duels. This design embeds the lever in a KERNEL-ACTIVE architecture and measures the early-trusted rescue behaviorally against a finely balanced mid-validity pair whose gate gap is calibrated to sit between the two theories' early-trusted weights.

LAYOUT (n=6): v = [0.95, 0.95, 0.75, 0.75, 0.60, 0.95]. Trusted 95% experts sit at the three gate anchors: position 1 (amplified gate (6/5)^eta in [2.1, 4.3] over both boxes), position 2 (gate EXACTLY 1.0 in both theories), position 6 (gate floor delta). The two 75% experts at positions 3-4 form a finely balanced kernel pair whose weight gap (g_3-g_4)*bracket_75 is, at every corner of both boxes, SMALLER than PGDT's minimal position-1 trusted weight but comparable to PBDT's maximal one -- this calibration is what makes the rescue cells sign-diagnostic. The single 60% expert at position 5 is the dominant kernel cue anchoring the anti-validity screens. All boxes below are mirror-corrected P(A), verified at the corners and adversarial interior points of both parameter boxes (gate/ramp/kernel corners with beta and lapse at their worst-case ends).

AXIS 1 -- EARLY-TRUSTED RESCUE (the headline). Base rung M34: A wins the 75% expert at position 3 vs B the 75% expert at position 4 -- both theories prefer A (PBDT [0.54, 0.60], PGDT [0.49, 0.63]). R1 adds the 95%@1 endorsement to B's side (A{75@3} vs B{75@4, 95@1}): PGDT's early-trusted weight (raw 0.028-0.057, always exceeding the 0.75-pair gap 0.011-0.028) flips choice to B at EVERY corner: P(A) in [0.39, 0.47]. PBDT's early-trusted weight (raw 0.011-0.021) usually does NOT close the gap: PBDT straddles chance, P(A) in [0.46, 0.55]. R1p (A{75@3} vs B{75@4, 95@1, 95@6}): PGDT robustly B, P(A) in [0.38, 0.44]; PBDT at chance, [0.48, 0.54] -- DISJOINT BOXES. Observed mirror-corrected P(A) >= 0.46 on R1p (or >= 0.50 on R1) is impossible for PGDT at any admissible parameter combination. Conversely R2 (A{75@3} vs B{75@4, 95@2}): PBDT's position-2 trusted weight (t2*flat, raw 0.005-0.009) never exceeds the 0.75-pair gap, so PBDT stays with A at EVERY corner: P(A) in [0.51, 0.64]; PGDT's position-2 trusted weight (the full flat share, raw 0.013-0.031) sometimes closes it: PGDT at chance, [0.47, 0.51]. Observed P(A) <= 0.50 on R2 kills PBDT's entire space. R2p (helper pair {95@2, 95@6}): PBDT [0.50, 0.61] vs PGDT [0.44, 0.50]. R3 (helper 95@6, the LATE trusted expert whose weight is nearly theory-invariant): BOTH theories stay with A (PBDT [0.54, 0.61], PGDT [0.51, 0.60]) -- the null rung proving the rescue effect is specific to the EARLY trusted channel, not a generic any-95%-helper effect. Note R1/R2 are 1-vs-2 count trials on which tallying picks B (2 cues beat 1) and TTB picks B (the top cue, 95%@1 or 95%@2, is on B's side): PBDT's A-leaning prediction on R2 is unique among all theories considered. The joint readout {R1p <= 0.45 AND R2 >= 0.55} is uniquely PBDT; {R1p <= 0.45 AND R2 <= 0.50} is uniquely PGDT.

AXIS 2 -- TRUSTED-SWING BATTERY (quantitative core). The S-cells (S1: A{75@3, 95@1} vs B{75@4}, etc.) paired with the R-cells give within-subject double differences equal to twice the behavioral weight of a 95% endorsement at positions 1, 2, 6. PGDT's profile is w1:w2:w6 = g_1:1:delta; PBDT's is g_1*t0 : t2 : delta*t1. Boxes: S1 PBDT [0.61, 0.70] vs PGDT [0.57, 0.79]; S2 PBDT [0.58, 0.65] vs PGDT [0.55, 0.74]; S6 PBDT [0.57, 0.61] vs PGDT [0.53, 0.64]. Parametric signatures: (i) the raw position-1 trusted weight ratio PGDT/PBDT = 1/t0 in [2.9, 4.0] -- parameter-free; (ii) the estimated ratio w(95@2)/w(95@6) = 1/delta >= 2.2 at every PGDT corner but can fall to 1.25 under PBDT (t2/(delta*t1)) -- an observed swing ratio <= 2.0 excludes PGDT's entire box, while a ratio >= 3.8 excludes PBDT's; (iii) PGDT's swing@1/swing@6 = g_1/delta >= 4.1 everywhere, reachable by PBDT only above 2.9.

AXIS 3 -- TRUSTED POSITION DUELS. T16 (95@1 vs 95@6): PBDT [0.52, 0.54] vs PGDT [0.52, 0.65] -- PGDT's box extends far higher because its early trusted expert carries the full flat share under the amplified gate. T26 (95@2 vs 95@6): PBDT [0.51, 0.55] vs PGDT [0.51, 0.64]. T12 (95@1 vs 95@2): PBDT [0.53, 0.57] vs PGDT [0.52, 0.60]. At matched (central) corners PGDT's trusted-position gradient is 2-3x PBDT's across the whole battery.

AXIS 4 -- SCREENS AND ANCHORS. D60 (the single 60% expert at position 5 vs the 75% expert at position 3): BOTH theories prefer the 60%-endorsed product (PBDT [0.67, 0.80], PGDT [0.59, 0.87]) -- verifies that subjects read and distrust-invert the communicated validities. VETO (the three 95% experts vs the single 60% expert): both theories prefer the 60%-endorsed product (PBDT [0.32, 0.40], PGDT [0.36, 0.41]) -- the product endorsed by ALL THREE best experts loses to the worst expert's product; excludes normative weighting, TTB, and tallying (3 cues beat 1). DINV (the three 95s vs the 75/75/60 block): both strongly prefer the low-validity block (PBDT [0.11, 0.19], PGDT [0.03, 0.30]). TOT: comprehension anchor, both >= 0.85.

AGGREGATE SIGNATURE. Mirror-corrected P(A) regressed on PGDT reference evidence (w_j proportional to g_j*[(1-tau)/n + tau*inv_j] at tau=0.87, eta=5, delta=0.3, gamma=3.15) versus PBDT reference evidence (w_j proportional to g_j*[t_j*(1-tau)/n + tau*inv_j] at t0=0.30, kappa=1.0, eta=7, delta=0.2, tau=0.92, gamma=2.75) loads with opposite profiles: the two evidence axes differ almost exclusively on the R1/R1p/S1/T16 family, where PGDT's early-trusted term is large and PBDT's is crushed by t0. Single-number readouts: (i) R1p sign (PGDT requires P(A) <= 0.45; PBDT allows >= 0.48 -- disjoint); (ii) R2 sign (PBDT requires P(A) >= 0.51; PGDT <= 0.51); (iii) the fitted trusted-channel position profile (w1, w2, w6) from the swing battery, whose w2/w6 ratio separates the theories at 2.2 (PGDT floor) vs 1.25 (PBDT floor). Every cell type appears in A/B-mirrored form to absorb side and position bias. 32 unique pairs, K = 96 // 32 = 3 repetitions in independently randomized order = exactly 96 trials per subject.

**Computed schedule:** 32 unique pairs × 3 reps = 96 trials per subject.



### METRIC
Rationale:
The two theories share the distrust kernel and a floored primacy gate; they differ almost exclusively in the TRUSTED (flat-share) channel. PGDT's bracket is position-independent, so a 95% expert at position 1 carries the full amplified gate weight; PBDT multiplies the flat share by a trust ramp with floor t0 in [0.25, 0.35], crushing the early trusted weight by a parameter-free factor 1/t0 in [2.9, 4.0] while leaving late trusted experts intact. The design was built to make this the only quantitatively decisive axis: R1/R1p (early-trusted helper) cells flip toward B under PGDT but stay at chance under PBDT; S1/S2/S6 and T16/T26/T12 measure the same trusted-channel position profile; R2/R2p/M34 and the D60/VETO/DINV/TOT anchors contribute smaller, consistently-signed contrast. The metric implements BOTH theories' exact weight computations (distrust inversion, gates, trust ramp, logistic lapse) at five parameter settings spanning each theory's box, averages the predicted P(choose A) per trial, and takes the per-trial contrast c_k = p_PGDT - p_PBDT as a fixed, data-independent weighting template. Observed mirror-corrected choices are scored against this template and standardized (z = sum c_k y_k / sqrt(sum c_k^2)), which is the optimal linear discriminator of the two theories on this design: every diagnostic cell receives weight proportional to its expected between-theory gap, mirror pairs contribute with automatically opposite signs (absorbing side bias), and the non-diagnostic anchors receive near-zero weight. Under PBDT-simulated data the standardized score is small (choices do not follow the strong early-trusted profile); under PGDT-simulated data it is large and positive (choices track the early-trusted endorsements on R1/R1p/S1/T16 etc.). Because each subject contributes only 6 trials per cell and the per-trial gaps are ~0.03-0.04 in probability, a single subject's raw score is binomial-noise-dominated (sd ~1 in z units); the metric therefore applies empirical-Bayes shrinkage toward 0 with a fixed pseudo-observation count (K=2000), which stabilizes the per-subject values (variance ~0.002, i.e., tight across subjects) while leaving the pooled, all-subject estimate — based on ~2400 trials — essentially unregularized and hence sharply different between the two theories. This directly targets the acceptance criterion: a large pooled between-theory gap with low between-subject variance at N=25. The metric is defined identically on any data slice (pooled or single subject), uses only the schema columns, avoids hashing sequence cells (keys are projected to int tuples), and is deterministic.

Source:
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    """
    Early-trusted-channel contrast score (PBDT vs PGDT diagnostic).

    The design's decisive lever is the TRUSTED (flat-share) channel: PGDT's
    bracket (1-tau)/n + tau*inv_j is position-independent, so an early
    high-validity expert carries the full amplified position-1 gate weight;
    PBDT multiplies the flat share by a trust ramp rising from a low floor
    t0, crushing the early trusted weight by ~1/t0 while leaving late trusted
    experts nearly intact.  Behaviorally this shows up as a specific pattern
    of mirror-corrected choice probabilities across the R1/R1p/R2/R2p/S1/S2/
    S6/T16/T26/T12 cells (and weakly on the anchors).

    This metric computes, for every one of the 32 fixed trial pairs, the
    predicted P(choose A) under PGDT and under PBDT (both engines
    implemented below, each averaged over five parameter settings spanning
    its box), takes the per-trial contrast c_k = p_PGDT - p_PBDT, and scores
    the observed choices against that contrast:

        S = sum_rows c_k * y,   y = +1 if A chosen, -1 if B chosen
        z = S / sqrt(sum_rows c_k^2)

    z is the standardized behavioral expression of the early-trusted
    channel (positive = choices follow PGDT's strong early-trusted
    weighting; near zero / negative = choices follow PBDT's crushed
    early-trusted channel).  Because a single subject contributes only 6
    trials per cell and the per-trial contrast is small, the single-subject
    z is dominated by binomial noise; we therefore report an empirical-Bayes
    shrunken version

        metric = (n / (n + K)) * z,   K = 2000 pseudo-observations,

    which pulls sparse single-subject estimates strongly toward 0 (chance)
    while leaving the full pooled dataset (n ~ 2400) essentially
    unregularized.  The statistic is well defined on any slice of the data.
    """
    V = (0.95, 0.95, 0.75, 0.75, 0.60, 0.95)
    TRIALS = [
        ((0, 0, 1, 0, 0, 0), (1, 0, 0, 1, 0, 0)),
        ((1, 0, 0, 1, 0, 0), (0, 0, 1, 0, 0, 0)),
        ((0, 0, 1, 0, 0, 0), (0, 1, 0, 1, 0, 0)),
        ((0, 1, 0, 1, 0, 0), (0, 0, 1, 0, 0, 0)),
        ((0, 0, 1, 0, 0, 0), (0, 0, 0, 1, 0, 1)),
        ((0, 0, 0, 1, 0, 1), (0, 0, 1, 0, 0, 0)),
        ((0, 0, 1, 0, 0, 0), (1, 0, 0, 1, 0, 1)),
        ((1, 0, 0, 1, 0, 1), (0, 0, 1, 0, 0, 0)),
        ((0, 0, 1, 0, 0, 0), (0, 1, 0, 1, 0, 1)),
        ((0, 1, 0, 1, 0, 1), (0, 0, 1, 0, 0, 0)),
        ((1, 0, 1, 0, 0, 0), (0, 0, 0, 1, 0, 0)),
        ((0, 0, 0, 1, 0, 0), (1, 0, 1, 0, 0, 0)),
        ((0, 1, 1, 0, 0, 0), (0, 0, 0, 1, 0, 0)),
        ((0, 0, 0, 1, 0, 0), (0, 1, 1, 0, 0, 0)),
        ((0, 0, 1, 0, 0, 1), (0, 0, 0, 1, 0, 0)),
        ((0, 0, 0, 1, 0, 0), (0, 0, 1, 0, 0, 1)),
        ((1, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1)),
        ((0, 0, 0, 0, 0, 1), (1, 0, 0, 0, 0, 0)),
        ((0, 1, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1)),
        ((0, 0, 0, 0, 0, 1), (0, 1, 0, 0, 0, 0)),
        ((1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0)),
        ((0, 1, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0)),
        ((0, 0, 1, 0, 0, 0), (0, 0, 0, 1, 0, 0)),
        ((0, 0, 0, 1, 0, 0), (0, 0, 1, 0, 0, 0)),
        ((0, 0, 0, 0, 1, 0), (0, 0, 1, 0, 0, 0)),
        ((0, 0, 1, 0, 0, 0), (0, 0, 0, 0, 1, 0)),
        ((1, 1, 0, 0, 0, 1), (0, 0, 0, 0, 1, 0)),
        ((0, 0, 0, 0, 1, 0), (1, 1, 0, 0, 0, 1)),
        ((1, 1, 0, 0, 0, 1), (0, 0, 1, 1, 1, 0)),
        ((0, 0, 1, 1, 1, 0), (1, 1, 0, 0, 0, 1)),
        ((1, 1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0)),
        ((0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1, 1)),
    ]

    v = np.asarray(V, dtype=float)
    err = 1.0 - v
    pos = np.arange(6, dtype=float)

    def pgdt_w(tau, eta, delta, gamma, c):
        # PGDT weights: floored convex primacy gate x position-independent
        # bracket (flat share + distrust kernel).
        cen = np.maximum(err - c * float(err.min()), 0.0)
        inv = cen ** gamma
        w_inv = inv / inv.sum()
        g = delta + (1.0 - delta) * ((6.0 - pos) / 5.0) ** eta
        br = (1.0 - tau) / 6.0 + tau * w_inv
        raw = g * br
        return raw / raw.sum()

    def pbdt_w(tau, eta, delta, gamma, c, t0, t1, kappa):
        # PBDT weights: steep low-floored gate x (trust-ramped flat share +
        # distrust kernel).
        cen = np.maximum(err - c * float(err.min()), 0.0)
        inv = cen ** gamma
        w_inv = inv / inv.sum()
        t = t0 + (t1 - t0) * (pos / 5.0) ** kappa
        g = delta + (1.0 - delta) * ((6.0 - pos) / 5.0) ** eta
        br = t * (1.0 - tau) / 6.0 + tau * w_inv
        raw = g * br
        return raw / raw.sum()

    def p_a(w, A, B, beta, eps):
        d = float(np.dot(w, np.sign(np.asarray(A, dtype=float) - np.asarray(B, dtype=float))))
        p = 0.5 * (1.0 + np.tanh(0.5 * beta * d))
        return (1.0 - eps) * p + 0.5 * eps

    # Parameter settings spanning each theory's box (midpoint, key-dim
    # variants, weak corner, strong corner).
    PG = [
        (0.87, 5.00, 0.325, 3.15, 0.175, 4.25, 0.105),
        (0.82, 5.00, 0.325, 3.15, 0.175, 4.25, 0.105),
        (0.92, 5.00, 0.325, 3.15, 0.175, 4.25, 0.105),
        (0.92, 4.00, 0.200, 2.80, 0.100, 2.50, 0.150),
        (0.82, 6.00, 0.450, 3.50, 0.250, 6.00, 0.060),
    ]
    PB = [
        (0.92, 7.25, 0.20, 2.75, 0.15, 0.30, 0.975, 1.15, 4.00, 0.125),
        (0.91, 7.25, 0.20, 2.75, 0.15, 0.25, 0.975, 1.15, 4.00, 0.125),
        (0.93, 7.25, 0.20, 2.75, 0.15, 0.35, 0.975, 1.15, 4.00, 0.125),
        (0.93, 6.50, 0.25, 2.60, 0.05, 0.35, 0.950, 1.50, 3.60, 0.140),
        (0.91, 8.00, 0.15, 2.90, 0.25, 0.25, 1.000, 0.80, 4.40, 0.110),
    ]

    pg_ws = [pgdt_w(s[0], s[1], s[2], s[3], s[4]) for s in PG]
    pb_ws = [pbdt_w(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]) for s in PB]

    contrast = {}
    for A, B in TRIALS:
        pG = float(np.mean([p_a(w, A, B, s[5], s[6]) for (w, s) in zip(pg_ws, PG)]))
        pB = float(np.mean([p_a(w, A, B, s[8], s[9]) for (w, s) in zip(pb_ws, PB)]))
        contrast[(A, B)] = pG - pB

    def as_key(x):
        return tuple(int(round(float(t))) for t in x)

    n = int(len(data))
    if n == 0:
        return 0.0

    S = 0.0
    Q = 0.0
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'],
                        data['response']):
        key = (as_key(a), as_key(b))
        c = contrast.get(key)
        if c is None:
            continue
        if abs(c) < 1e-12:
            continue
        y = 1.0 if int(round(float(r))) == 0 else -1.0  # +1 = chose A
        S += c * y
        Q += c * c

    if Q <= 1e-24:
        return 0.0

    z = S / np.sqrt(Q)
    # Empirical-Bayes shrinkage toward 0 (chance-level expression of the
    # contrast) with a fixed pseudo-observation count: single-subject
    # slices (n=96) are shrunk hard; the full pooled dataset (n~2400) is
    # dominated by the data.
    lam = float(n) / (float(n) + 2000.0)
    return float(lam * z)

### RESULTS
- Predicted under pi_7 (simulated): 1.8638 (var=0.0018)
- Predicted under pi_9 (simulated): -1.5890 (var=0.0021)
- Observed on real data: -0.9319 (var=0.0016)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    # Design validities (fixed by experiment): strictly decreasing,
    # so the TTB cue order is simply feature 0,1,2,3,4.
    validities = np.array([0.95, 0.85, 0.75, 0.65, 0.5])
    cue_order = np.argsort(-validities, kind="stable")

    def _to_vec(x):
        if isinstance(x, str):
            return np.array(
                [int(v) for v in x.strip("[]()").replace(",", " ").split()]
            )
        return np.asarray(x, dtype=float)

    hits = []
    a_col = data["option_a_ratings"].tolist()
    b_col = data["option_b_ratings"].tolist()
    r_col = data["response"].tolist()

    for a_cell, b_cell, resp in zip(a_col, b_col, r_col):
        a = _to_vec(a_cell)
        b = _to_vec(b_cell)
        if a.size == 0 or a.shape != b.shape:
            continue

        # --- TTB winner: first discriminating cue in validity order ---
        ttb_winner = None
        for j in cue_order:
            if j >= a.size:
                continue
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break
        if ttb_winner is None:
            continue  # no discriminating cue: TTB guesses, uninformative

        # --- Tallying winner: strict feature-wise win counts ---
        a_wins = int(np.sum(a > b))
        b_wins = int(np.sum(b > a))
        if a_wins == b_wins:
            continue  # tally ties -> Tallying guesses at 0.5, dilutes contrast
        tally_winner = 0 if a_wins > b_wins else 1

        if tally_winner == ttb_winner:
            continue  # heuristics agree: uninformative

        # Pure conflict trial: TTB's option vs the tally majority's option.
        hits.append(1.0 if int(resp) == ttb_winner else 0.0)

    if not hits:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.2080 (var=0.0060)
**Predicted under pi_7:** 0.1637 (var=0.0053)
**Predicted under pi_9:** 0.1423 (var=0.0016)

### Experiment 4
**Design**
  A=[0, 1, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Tally-vs-TTB choice-allegiance contrast.

    For every trial we determine (i) the TALLYING winner (option with more
    strict feature-wise wins) and (ii) the TAKE-THE-BEST winner (winner on
    the highest-validity discriminating cue, validities fixed by design as
    [0.95, 0.9, 0.85, 0.65, 0.6, 0.5]).  The metric is

        mean( response == tally winner ) - mean( response == TTB winner )

    On the 4 trial types where both heuristics pick the same option
    (trials 1, 3, 5, 7 and mirrors) the two indicators are identical and
    cancel.  On the 12 conflict trial types they are exact complements, so
    the metric reduces to 0.75 * (2*P(tally winner | conflict) - 1): strongly
    positive under Tallying, strongly negative under TTB.
    """
    validities = np.array([0.95, 0.9, 0.85, 0.65, 0.6, 0.5])
    cue_order = np.argsort(-validities, kind="stable")

    a_col = data["option_a_ratings"].tolist()
    b_col = data["option_b_ratings"].tolist()
    r_col = data["response"].tolist()

    tally_match = []
    ttb_match = []

    for ca, cb, cr in zip(a_col, b_col, r_col):
        a = np.asarray(ca, dtype=float).ravel()
        b = np.asarray(cb, dtype=float).ravel()
        r = int(cr)

        # Tallying winner: more strict feature-wise wins (ties on a feature
        # contribute to neither side).
        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        if a_wins > b_wins:
            tally_winner = 0
        elif b_wins > a_wins:
            tally_winner = 1
        else:
            tally_winner = -1  # undiscriminating tally -> excluded

        # TTB winner: first discriminating cue in descending-validity order.
        ttb_winner = -1
        for j in cue_order:
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break

        if tally_winner >= 0:
            tally_match.append(1.0 if r == tally_winner else 0.0)
        if ttb_winner >= 0:
            ttb_match.append(1.0 if r == ttb_winner else 0.0)

    if len(tally_match) == 0 or len(ttb_match) == 0:
        return 0.0

    return float(np.mean(tally_match) - np.mean(ttb_match))
```

**Observed (real) value:** 0.1583 (var=0.0053)
**Predicted under pi_7:** 0.2996 (var=0.0066)
**Predicted under pi_9:** 0.3146 (var=0.0051)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd


def _to_vec(x):
    """Robustly convert a cell (list/tuple/ndarray/str) to a float 1-D array."""
    if isinstance(x, np.ndarray):
        return x.astype(float)
    if isinstance(x, (list, tuple)):
        return np.asarray(list(x), dtype=float)
    s = str(x).strip().replace("[", " ").replace("]", " ").replace(",", " ")
    return np.asarray(s.split(), dtype=float)


def metric(data: pd.DataFrame) -> float:
    df = data.reset_index(drop=True)
    n_rows = len(df)
    if n_rows == 0:
        return 0.0

    A = np.vstack([_to_vec(x) for x in df["option_a_ratings"]])
    B = np.vstack([_to_vec(x) for x in df["option_b_ratings"]])
    n_feat = A.shape[1]

    # Design validities [0.95, 0.85, 0.75, 0.65, 0.60]; fall back gracefully.
    v = np.array([0.95, 0.85, 0.75, 0.65, 0.60], dtype=float)
    if v.shape[0] != n_feat:
        v = np.linspace(0.95, 0.60, n_feat)

    # Chose-A indicator (response: 0 = A, 1 = B).
    y = (np.asarray(df["response"], dtype=float) == 0.0).astype(float)

    a_win = A > B
    b_win = B > A

    # Signed tally margin (Tallying's ONLY explanatory variable).
    margin = a_win.sum(axis=1).astype(float) - b_win.sum(axis=1).astype(float)

    # Validity-weighted evidence difference (NVW's driving quantity).
    d = (a_win * v).sum(axis=1) - (b_win * v).sum(axis=1)

    # OLS: y ~ 1 + margin + d. Return the partial coefficient on d.
    X = np.column_stack([np.ones(n_rows), margin, d])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[2])
```

**Observed (real) value:** -0.5756 (var=0.0186)
**Predicted under pi_7:** -0.4381 (var=0.0264)
**Predicted under pi_9:** -0.4178 (var=0.0203)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 1, 1, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0, 0, 1, 1]  B=[1, 0, 0, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Anchor-allegiance: proportion of trials with a large tally margin
    (|a_wins - b_wins| >= 6) on which the subject chose the option that
    wins on more features (the tally winner).

    Under Tallying, these are the unambiguous anchors (8-0 and 7-1 splits:
    design trials 7, 8, 21, 22, 23, 24 and their 4 repetitions = 24 trials
    per subject), where the softmax over tallies predicts near-deterministic
    choice for any non-trivial beta. Under NVW, the capacity-interference
    factor delta^4 crushes the effective inverse temperature to
    beta_eff <= 11 * 0.30^4 ~ 0.09, so even the maximum validity-weighted
    evidence difference (~6) yields P ~ sigmoid(0.5) <= 0.64; attention
    lapses and the uniform-mixing epsilon only pull this further toward
    0.5. The two theories therefore make sharply separated predictions on
    this proportion, and the A/B-mirrored anchors cancel side bias.
    """
    a_col = data['option_a_ratings']
    b_col = data['option_b_ratings']
    r_col = data['response']

    n_anchor = 0
    n_hit = 0

    for ra, rb, resp in zip(a_col, b_col, r_col):
        va = np.asarray(ra, dtype=float)
        vb = np.asarray(rb, dtype=float)
        if va.shape != vb.shape:
            continue

        a_wins = float(np.sum(va > vb))
        b_wins = float(np.sum(vb > va))
        margin = a_wins - b_wins

        if abs(margin) >= 6:
            n_anchor += 1
            winner_is_a = margin > 0
            chose_a = (int(resp) == 0)
            if chose_a == winner_is_a:
                n_hit += 1

    if n_anchor == 0:
        return 0.5
    return float(n_hit) / float(n_anchor)

```

**Observed (real) value:** 0.8433 (var=0.0152)
**Predicted under pi_7:** 0.8925 (var=0.0035)
**Predicted under pi_9:** 0.8792 (var=0.0043)

### Experiment 7
**Design**
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Validity-adherence rate on VALIDITY-vs-RECENCY conflict trials.
    #
    # For every trial we compute two decision variables from the presented
    # cue patterns:
    #   d_v = sum_j v_j * sign(A_j - B_j)   (validity-weighted evidence for A)
    #   d_r = sum_j (j/(n-1))^3 * sign(A_j - B_j)  (convex cue-order recency
    #                                              evidence for A)
    # A trial is a CONFLICT trial when the two point in opposite directions
    # and both are non-trivial in magnitude. On those trials the metric is
    # the proportion of choices that follow the VALIDITY-implied option.
    #
    # NVW (validity-weighted integration) predicts adherence clearly above
    # chance (~0.6); SWC (recency-dominated subjective weights) predicts
    # adherence well below chance (~0.3), because on these trials its
    # recency gradient favors the opposite option.

    validities = np.array([0.95, 0.90, 0.85, 0.62, 0.50, 0.50], dtype=float)
    n = int(validities.shape[0])
    # Convex recency profile over presentation positions (eta ~ 3).
    recency = (np.arange(n, dtype=float) / float(n - 1)) ** 3.0

    a_col = data["option_a_ratings"]
    b_col = data["option_b_ratings"]
    r_col = data["response"]

    hits = []
    for a_raw, b_raw, resp in zip(a_col, b_col, r_col):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        if a.shape[0] != n or b.shape[0] != n:
            continue
        s = np.sign(a - b)
        if not np.any(s != 0):
            continue

        d_v = float(np.dot(validities, s))
        d_r = float(np.dot(recency, s))

        # Need a real conflict with non-trivial evidence on both axes.
        if abs(d_v) < 0.30 or abs(d_r) < 0.50:
            continue
        if np.sign(d_v) == np.sign(d_r):
            continue

        validity_winner_is_a = (d_v > 0.0)
        chose_a = (int(resp) == 0)
        hits.append(1.0 if (chose_a == validity_winner_is_a) else 0.0)

    if len(hits) == 0:
        return 0.0
    return float(np.mean(hits))
```

**Observed (real) value:** 0.2313 (var=0.0046)
**Predicted under pi_7:** 0.2334 (var=0.0055)
**Predicted under pi_9:** 0.1894 (var=0.0018)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Recency-consistency rate: fraction of choices aligned with a convex
    # cue-order recency gradient, restricted to trials where that gradient
    # is strong and its direction is unambiguous across SWC's eta range.
    #
    # SWC (advocated): late cues dominate via convex recency -> rate well
    #   above 0.5 (predicted ~0.65-0.85 across its parameter box).
    # NVW (competing): validity-weighted integration with capacity
    #   interference (beta_eff = beta*delta^3 <= ~0.30 at n=7) -> choices
    #   near chance on nearly every trial, and anti-recency or neutral on
    #   the conflict trials -> rate ~0.50.
    df = data.reset_index(drop=True)
    if len(df) == 0:
        return 0.5

    a_arrs = [np.asarray(list(x), dtype=float) for x in df['option_a_ratings']]
    b_arrs = [np.asarray(list(x), dtype=float) for x in df['option_b_ratings']]
    resp = pd.to_numeric(df['response'], errors='coerce').to_numpy()

    if len(a_arrs) == 0 or a_arrs[0].ndim != 1:
        return 0.5
    n = int(a_arrs[0].shape[0])
    if n < 2:
        return 0.5

    pos = np.arange(n, dtype=float) / (n - 1.0)
    w_lo = pos ** 2.5   # SWC's eta lower corner
    w_mid = pos ** 3.5  # SWC's eta midpoint
    w_hi = pos ** 4.5   # SWC's eta upper corner

    consistent = []
    for a, b, r in zip(a_arrs, b_arrs, resp):
        if a.shape[0] != n or b.shape[0] != n:
            continue
        if not np.isfinite(r):
            continue
        s = np.sign(a - b)
        if not np.any(s != 0):
            continue
        e_lo = float(np.dot(w_lo, s))
        e_mid = float(np.dot(w_mid, s))
        e_hi = float(np.dot(w_hi, s))
        # Require the recency gradient's direction to be invariant across
        # SWC's whole eta range (drops ambiguous trials, e.g. 1 late cue vs
        # 4 mid cues, where the sign flips with eta).
        if e_lo == 0.0 or e_hi == 0.0 or np.sign(e_lo) != np.sign(e_hi):
            continue
        # Require non-trivial gradient magnitude (drops near-degenerate
        # trials where even SWC predicts ~0.52).
        if abs(e_mid) < 0.25:
            continue
        fav_a = e_mid > 0.0
        chose_a = (int(r) == 0)  # response: 0 = chose A, 1 = chose B
        consistent.append(1.0 if (chose_a == fav_a) else 0.0)

    if len(consistent) == 0:
        return 0.5
    return float(np.mean(consistent))
```

**Observed (real) value:** 0.3739 (var=0.0026)
**Predicted under pi_7:** 0.3028 (var=0.0052)
**Predicted under pi_9:** 0.3208 (var=0.0017)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Anti-validity vs. recency disagreement score.

    On trials where IVC's distrust-inverted weights and SWC's recency-dominated
    weights point to OPPOSITE options, report the proportion of choices that
    follow the IVC-preferred option. IVC-simulated data -> well above 0.5;
    SWC-simulated data -> well below 0.5.
    """
    data = data.reset_index(drop=True)
    if len(data) == 0:
        return 0.5

    # Build stimulus matrices without hashing any list-valued cells.
    a = np.vstack([np.asarray(r, dtype=float) for r in data["option_a_ratings"]])
    b = np.vstack([np.asarray(r, dtype=float) for r in data["option_b_ratings"]])
    n = a.shape[1]

    # Design validities (fallback mirrors both theories' graceful degradation).
    v = np.array([0.5, 0.9, 0.6, 0.95, 0.5, 0.6], dtype=float)
    if v.shape[0] != n:
        v = np.linspace(0.95, 0.55, n)

    # --- IVC weights at mid-range parameters (tau=0.925, kappa=2.7) ---
    tau, kappa = 0.925, 2.7
    inv = np.power(1.0 - v, kappa)
    Z = float(inv.sum())
    w_inv = inv / Z if Z > 1e-12 else np.full(n, 1.0 / n)
    w_ivc = (1.0 - tau) / n + tau * w_inv

    # --- SWC weights at mid-range parameters (lam=0.9, rho=0.9, eta=3.5) ---
    lam, rho, eta = 0.9, 0.9, 3.5
    vspan = float(v.max() - v.min())
    if vspan > 1e-12:
        vnorm = (v - v.min()) / vspan
    else:
        vnorm = np.full(n, 0.5)
    if n > 1:
        pos = (np.arange(n, dtype=float) / (n - 1.0)) ** eta
    else:
        pos = np.full(n, 0.5)
    g = (1.0 - rho) * vnorm + rho * pos
    w_swc = (1.0 - lam) / n + lam * g

    # Signed discriminating-cue vector per trial.
    s = np.sign(a - b)
    d_ivc = s @ w_ivc
    d_swc = s @ w_swc

    # Keep only trials where the two theories STRICTLY disagree in sign.
    # (d_ivc == 0, e.g. the validity-matched EQ trials, is excluded.)
    mask = (d_ivc * d_swc) < 0.0
    if not np.any(mask):
        return 0.5

    resp = np.asarray(data["response"].to_numpy())
    r = resp[mask]
    ivc_prefers_a = d_ivc[mask] > 0.0
    chose_ivc_preferred = np.where(ivc_prefers_a, r == 0, r == 1)
    return float(chose_ivc_preferred.mean())
```

**Observed (real) value:** 0.7878 (var=0.0086)
**Predicted under pi_7:** 0.8439 (var=0.0041)
**Predicted under pi_9:** 0.8572 (var=0.0032)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    """Final-Expert Dominance Index (FEDI).

    Among trials where the LAST-presented expert (feature position n-1) is the
    ONLY cue favoring one option (i.e., that option's rating vector is all
    zeros except a 1 in the final position), compute the proportion of
    choices that follow the final expert's recommendation.

    A convex-recency weighting (SWC) predicts adherence well above 0.5
    (the last cue carries the largest subjective weight at every corner of
    its parameter box). A distrust-inverted validity weighting (IVC)
    predicts adherence below 0.5, because the final expert here has a high
    stated validity (0.90) and is therefore heavily discounted.
    """
    df = data.reset_index(drop=True)

    A = np.array(
        [list(map(int, x)) for x in df['option_a_ratings']], dtype=int
    )
    B = np.array(
        [list(map(int, x)) for x in df['option_b_ratings']], dtype=int
    )
    r = df['response'].to_numpy(dtype=int)

    if A.ndim != 2 or A.shape[0] == 0:
        return float('nan')

    n_feat = A.shape[1]
    last = n_feat - 1

    a_last = A[:, last]
    b_last = B[:, last]

    # Last expert discriminates between the options.
    disc = a_last != b_last

    # The option endorsed by the final expert wins ONLY that final cue
    # (its rating vector sums to 1). This selects the single-cue and
    # multi-cue 'ladder' duels against the last expert (trials 1-14 and
    # their mirrors) and excludes trials where the last-cue option also
    # wins other cues.
    a_sel = disc & (a_last == 1) & (A.sum(axis=1) == 1)
    b_sel = disc & (b_last == 1) & (B.sum(axis=1) == 1)
    sel = a_sel | b_sel

    if not np.any(sel):
        return float('nan')

    # response: 0 = chose A, 1 = chose B.
    # Did the subject follow the final expert's endorsed option?
    followed = np.where(a_sel[sel], r[sel] == 0, r[sel] == 1)

    return float(np.mean(followed))
```

**Observed (real) value:** 0.3957 (var=0.0045)
**Predicted under pi_7:** 0.2832 (var=0.0037)
**Predicted under pi_9:** 0.2789 (var=0.0034)

### Experiment 11
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Recency-following rate on count-neutral trials.
    #
    # With all six validities equal (0.95), IVC's weight vector is EXACTLY
    # flat for every admissible parameter combination (the distrust kernel
    # is constant across features, so w_j = 1/n identically). Hence on any
    # trial where the two options win an equal number of discriminating
    # cues (count difference = 0), IVC's evidence d = 0 and P(choose A) =
    # 0.50 exactly -- the lapse preserves 0.5. This is parameter-free.
    #
    # SWC, in contrast, has strictly increasing weights in cue position
    # (lambda >= 0.8, rho >= 0.8, convex recency), and on count-neutral
    # trials its evidence reduces EXACTLY to d = lambda*rho*sum_j pos_j*s_j,
    # so it must prefer the option supported by the LATER cues, with
    # subject-level choice rates roughly in [0.60, 0.79].
    #
    # The metric: among count-neutral trials, the fraction of choices that
    # go to the option favored by the later (more recent) discriminating
    # cues. IVC -> 0.50 for every subject; SWC -> well above 0.50.

    def _mat(col):
        return np.array(
            [np.asarray(v, dtype=float).ravel() for v in list(col)],
            dtype=float,
        )

    A = _mat(data['option_a_ratings'])
    B = _mat(data['option_b_ratings'])
    resp = np.asarray(pd.Series(data['response']).to_numpy(), dtype=float)

    if A.size == 0 or A.ndim != 2 or A.shape[0] == 0 or B.shape[0] != A.shape[0]:
        return 0.5
    n = int(A.shape[1])
    if n < 2:
        return 0.5

    # Convex recency profile over cue positions (any strictly increasing
    # profile gives the same favored direction on count-neutral trials;
    # 3.5 matches SWC's mid-gradient). Direction is what matters here.
    pos = (np.arange(n, dtype=float) / (n - 1.0)) ** 3.5

    S = np.sign(A - B)            # +1: cue favors A, -1: favors B, 0: tie
    count = S.sum(axis=1)         # IVC's evidence direction (flat weights)
    rec = S @ pos                 # recency evidence for A

    neutral = (np.abs(count) < 1e-9) & (np.abs(rec) > 1e-9)
    if not np.any(neutral):
        return 0.5

    chose_a = resp[neutral] == 0
    fav_a = rec[neutral] > 0      # option supported by the later cues
    followed = np.where(fav_a, chose_a, ~chose_a)
    return float(np.mean(followed))
```

**Observed (real) value:** 0.3414 (var=0.0042)
**Predicted under pi_7:** 0.2989 (var=0.0035)
**Predicted under pi_9:** 0.2954 (var=0.0028)

### Experiment 12
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Recency-gradient choice score on theory-divergent trials.
    #
    # Design validities: experts at positions 1 and 4 (0-indexed 0 and 3)
    # have validity 0.95; the other four experts have validity 0.75.
    #
    # Per trial:
    #   s_j = sign(A_j - B_j)   which option expert j favors
    #   x   = sum_j p_j * s_j   recency evidence for A, with convex
    #                           position weights p_j = (j/5)^3.5
    #   y   = +1 if A chosen, -1 if B chosen
    #
    # Scored trial families (48 of the 96 trials per subject):
    # (1) VALIDITY-MATCHED trials (types 3/4, 9/10, 13/14, 17/18, 25/26
    #     plus mirrors): the two options' discriminating experts carry
    #     identical validity multisets. Any validity-only weighting (IVC)
    #     yields exactly zero evidence, so E[x*y] = 0 for EVERY parameter
    #     combination (the uniform lapse preserves 0.5). SWC's convex
    #     recency gradient makes E[x*y] strongly positive.
    # (2) VALIDITY-CONFOUNDED trials (types 5/6, 11/12, 23/24 plus
    #     mirrors): the option whose experts sit LATER also carries
    #     strictly more 0.95 experts. IVC's distrust inversion favors
    #     the EARLY option, so E[x*y] < 0; SWC's recency gradient still
    #     favors the LATE option, so E[x*y] > 0.
    #
    # Metric = mean of x*y over these trials.
    # SWC: strongly positive (~ +0.28 across its box, range ~ +0.17..+0.41).
    # IVC: zero on the matched family for all parameters, slightly
    # negative overall (~ -0.03).

    n = 6
    V = np.array([0.95, 0.75, 0.75, 0.95, 0.75, 0.75])
    P = (np.arange(n, dtype=float) / float(n - 1)) ** 3.5
    hi = V >= 0.9

    def to_arr(cell):
        if isinstance(cell, np.ndarray):
            return np.asarray(cell, dtype=float).ravel()
        if isinstance(cell, (list, tuple)):
            return np.asarray(list(cell), dtype=float).ravel()
        txt = str(cell).strip().strip('[]()')
        parts = txt.replace(',', ' ').split()
        return np.asarray([float(t) for t in parts], dtype=float).ravel()

    a_cells = data['option_a_ratings'].tolist()
    b_cells = data['option_b_ratings'].tolist()
    r_cells = data['response'].tolist()

    scores = []
    for i in range(len(a_cells)):
        try:
            a = to_arr(a_cells[i])
            b = to_arr(b_cells[i])
        except Exception:
            continue
        if a.shape[0] != n or b.shape[0] != n:
            continue
        try:
            resp = int(r_cells[i])
        except Exception:
            continue
        if resp != 0 and resp != 1:
            continue

        s = np.sign(a - b)
        ia = np.where(s > 0)[0]
        ib = np.where(s < 0)[0]
        if ia.size == 0 or ib.size == 0:
            continue

        va = sorted(V[ia].tolist())
        vb = sorted(V[ib].tolist())
        matched = (va == vb)

        x = float(np.dot(P, s))
        if not matched:
            if x == 0:
                continue
            late_a = x > 0
            late_hi = int(hi[ia].sum()) if late_a else int(hi[ib].sum())
            early_hi = int(hi[ib].sum()) if late_a else int(hi[ia].sum())
            if late_hi <= early_hi:
                continue

        y = 1.0 if resp == 0 else -1.0
        scores.append(x * y)

    if len(scores) == 0:
        return 0.0
    return float(np.mean(scores))
```

**Observed (real) value:** -0.1729 (var=0.0118)
**Predicted under pi_7:** -0.2325 (var=0.0136)
**Predicted under pi_9:** -0.2701 (var=0.0071)

### Experiment 13
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    # Within-allegiance-cell OLS slope of choice on reference primacy evidence.
    #
    # Identification: group every trial by (c1, c2) where
    #   c1 = number of 0.60-expert wins for A minus for B (positions 2,3,5),
    #   c2 = total signed cue count.
    # Any position-INVARIANT validity-weighted model (IVC at ANY parameter
    # combo, plain tallying, any lapse/transform) makes P(choose A) a function
    # of (c1, c2) ONLY. Demeaning the primacy regressor within these cells
    # therefore makes the population OLS slope EXACTLY 0 under IVC,
    # parameter-free, for every subject. PW-IVC's primacy gradient breaks the
    # cell-conditional independence, producing a strictly positive slope.

    ratings_a = data['option_a_ratings']
    ratings_b = data['option_b_ratings']
    A = np.array([np.asarray(r, dtype=float) for r in ratings_a], dtype=float)
    B = np.array([np.asarray(r, dtype=float) for r in ratings_b], dtype=float)
    if A.ndim != 2 or A.shape[0] == 0:
        return 0.0
    n = A.shape[1]

    # Reference primacy kernel, eta = 5 (middle of PW-IVC's eta range [3.5, 6.5]).
    if n > 1:
        pos = np.arange(n, dtype=float)
        wp = ((n - pos) / (n - 1.0)) ** 5.0
        wp = wp / wp.sum()
    else:
        wp = np.ones(1)

    S = np.sign(A - B)
    y = 1.0 - np.asarray(data['response'], dtype=float)  # 1 = chose A

    x1 = S.dot(wp)  # primacy evidence favoring A

    # Allegiance cell for this design: validities [0.95, 0.6, 0.6, 0.95, 0.6, 0.95]
    # -> 0.60 experts at 0-indexed positions 1, 2, 4.
    if n == 6:
        c1 = S[:, 1] + S[:, 2] + S[:, 4]
    else:
        c1 = np.zeros(A.shape[0])
    c2 = S.sum(axis=1)

    g = c1.astype(int).astype(str) + "_" + c2.astype(int).astype(str)
    df = pd.DataFrame({'x1': x1, 'y': y, 'g': g})

    # Demean primacy evidence within allegiance cell: removes ALL validity-based
    # mean structure (distrust kernel at any gamma/c/tau, flat counting, lapse).
    df['x1c'] = df['x1'] - df.groupby('g')['x1'].transform('mean')
    ssx = float((df['x1c'] ** 2).sum())
    if ssx <= 1e-12:
        return 0.0
    ybar = float(df['y'].mean())
    slope = float((df['x1c'] * (df['y'] - ybar)).sum() / ssx)
    return slope
```

**Observed (real) value:** 0.3148 (var=0.0095)
**Predicted under pi_7:** 0.2316 (var=0.0094)
**Predicted under pi_9:** 0.2873 (var=0.0067)

### Experiment 14
**Design**
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """
    Primacy-slope metric on validity-matched cells (with X2-family
    orientation absorbed by a dummy).

    Logic: on every trial where the two options' winning-cue validity
    multisets are IDENTICAL (the 20 matched cells: D0,Z1,D1,W1,W2,D2,
    D2b,D3,D4,D5 + mirrors), the distrust kernel and the flat counting
    kernel cancel EXACTLY, so IVC predicts P(A)=0.500 parameter-free
    (the lapse preserves 0.5). PW-IVC predicts a systematic deviation
    graded by the primacy contrast C = w_prim . sign(A-B), with a
    steep convex reference gradient (eta=5).

    We also include the six X2-family cells (2 wins per side, unequal
    validity multisets). Under IVC all X2 cells share ONE probability
    that depends only on orientation (A-won-the-0.9 vs mirror), so a
    single +-1 orientation dummy Z absorbs IVC's entire mean structure
    there; the within-family position spread then loads on C.

    Metric = OLS coefficient of C in y ~ 1 + C + Z, with y = +1 (chose
    A) / -1 (chose B). Under IVC this coefficient is EXACTLY 0 in
    expectation for every admissible parameter. Under PW-IVC it is
    strictly positive (roughly 2*(1-eps)*beta*tau*rho scale), typically
    0.15-0.8.
    """
    import numpy as np
    import pandas as pd

    n = 6
    v = np.array([0.9, 0.6, 0.6, 0.6, 0.6, 0.9], dtype=float)

    # Reference primacy kernel (eta = 5): convex, front-loaded.
    pos = np.arange(n, dtype=float)
    prim = ((n - pos) / (n - 1.0)) ** 5.0
    w_prim = prim / prim.sum()

    # Reference inverted-validity kernel (gamma = 3, c = 0.1).
    # Only the SIGN of its dot product with s is used (orientation).
    err = 1.0 - v
    cent = np.maximum(err - 0.1 * float(err.min()), 0.0)
    u = cent ** 3.0
    u = u / u.sum()

    a = np.array([[float(t) for t in r] for r in data['option_a_ratings']], dtype=float)
    b = np.array([[float(t) for t in r] for r in data['option_b_ratings']], dtype=float)
    y = 1.0 - 2.0 * np.asarray(data['response'], dtype=float).ravel()  # +1 = A

    m = a.shape[0]
    if m == 0:
        return 0.0

    s = np.sign(a - b)
    aWin = a > b
    bWin = b > a

    C = np.zeros(m)
    Z = np.zeros(m)
    keep = np.zeros(m, dtype=bool)

    for i in range(m):
        ka = tuple(sorted(v[aWin[i]].tolist()))
        kb = tuple(sorted(v[bWin[i]].tolist()))
        nA = int(aWin[i].sum())
        nB = int(bWin[i].sum())
        if ka == kb and nA > 0:
            # Matched-validity cell: distrust+flat kernels cancel exactly.
            keep[i] = True
            C[i] = float(np.dot(w_prim, s[i]))
        elif nA == 2 and nB == 2 and ka != kb:
            # X2-family cell: orientation dummy absorbs IVC's constant
            # allegiance effect; position spread loads on C.
            keep[i] = True
            C[i] = float(np.dot(w_prim, s[i]))
            Z[i] = float(np.sign(np.dot(u, s[i])))

    if int(keep.sum()) < 12:
        return 0.0

    Ck = C[keep]
    Zk = Z[keep]
    yk = y[keep]

    X = np.column_stack([np.ones_like(Ck), Ck, Zk])
    try:
        coef, _, _, _ = np.linalg.lstsq(X, yk, rcond=None)
        return float(coef[1])
    except Exception:
        return 0.0
```

**Observed (real) value:** 0.5396 (var=0.0423)
**Predicted under pi_7:** 0.4678 (var=0.0344)
**Predicted under pi_9:** 0.5986 (var=0.0372)

### Experiment 15
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Trial specification keyed by (A ratings, B ratings) -> (family, aligned response).
    # Families (validities = [0.95, 0.5, 0.95, 0.5, 0.95, 0.5]; experts 1,3,5 are 95%, experts 2,4,6 are 50%):
    #   "H" : high-validity (95%) duels (trials 1-4); aligned = product winning the LATER expert (anti-primacy)
    #   "L" : low-validity (50%) duels (trials 7-10); aligned = product winning the EARLIER expert (primacy)
    #   "M" : distrust-neutralized position cells (trials 13-18); aligned = product winning expert 2
    #   "C" : count/validity-balanced coalition cells (trials 21-24); aligned = product winning experts {2,3}
    spec = {
        ((1, 0, 0, 0, 0, 0), (0, 0, 0, 0, 1, 0)): ("H", 1),
        ((0, 0, 0, 0, 1, 0), (1, 0, 0, 0, 0, 0)): ("H", 0),
        ((1, 0, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0)): ("H", 1),
        ((0, 0, 1, 0, 0, 0), (1, 0, 0, 0, 0, 0)): ("H", 0),
        ((0, 1, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1)): ("L", 0),
        ((0, 0, 0, 0, 0, 1), (0, 1, 0, 0, 0, 0)): ("L", 1),
        ((0, 1, 0, 0, 0, 0), (0, 0, 0, 1, 0, 0)): ("L", 0),
        ((0, 0, 0, 1, 0, 0), (0, 1, 0, 0, 0, 0)): ("L", 1),
        ((0, 1, 0, 0, 0, 0), (1, 0, 0, 0, 0, 1)): ("M", 0),
        ((1, 0, 0, 0, 0, 1), (0, 1, 0, 0, 0, 0)): ("M", 1),
        ((0, 1, 0, 0, 0, 0), (1, 0, 0, 1, 0, 0)): ("M", 0),
        ((1, 0, 0, 1, 0, 0), (0, 1, 0, 0, 0, 0)): ("M", 1),
        ((0, 1, 1, 0, 0, 0), (1, 0, 0, 0, 0, 1)): ("M", 0),
        ((1, 0, 0, 0, 0, 1), (0, 1, 1, 0, 0, 0)): ("M", 1),
        ((0, 1, 1, 0, 0, 0), (0, 0, 0, 1, 1, 0)): ("C", 0),
        ((0, 0, 0, 1, 1, 0), (0, 1, 1, 0, 0, 0)): ("C", 1),
        ((0, 1, 1, 0, 0, 0), (0, 0, 0, 0, 1, 1)): ("C", 0),
        ((0, 0, 0, 0, 1, 1), (0, 1, 1, 0, 0, 0)): ("C", 1),
    }

    hits = {"H": 0, "L": 0, "M": 0, "C": 0}
    tots = {"H": 0, "L": 0, "M": 0, "C": 0}

    a_cells = data["option_a_ratings"].tolist()
    b_cells = data["option_b_ratings"].tolist()
    r_vals = data["response"].tolist()

    for a_cell, b_cell, r in zip(a_cells, b_cells, r_vals):
        try:
            ka = tuple([int(x) for x in a_cell])
            kb = tuple([int(x) for x in b_cell])
            ri = int(r)
        except Exception:
            continue
        entry = spec.get((ka, kb))
        if entry is None:
            continue
        fam, al = entry
        tots[fam] += 1
        if ri == al:
            hits[fam] += 1

    def prop(fam):
        if tots[fam] == 0:
            return 0.5
        return hits[fam] / float(tots[fam])

    p_l = prop("L")  # P(earlier-expert product | low-validity duel)
    p_h = prop("H")  # P(later-expert product  | high-validity duel)
    p_m = prop("M")  # P(expert-2 product | distrust-neutralized cells)
    p_c = prop("C")  # P({2,3}-coalition product | balanced coalition cells)

    # (1) Validity-scaled position crossover: C = P(early|low duel) - P(early|high duel)
    #     = p_l + p_h - 1. Strictly positive under PGDT, strictly negative under PW-IVC
    #     at every corner of both parameter boxes (position effects are validity-
    #     AMPLIFIED under PGDT's multiplicative kernel, validity-INVARIANT under
    #     PW-IVC's additive kernel, where the position-1 expert always dominates).
    # (2) M-family: distrust terms cancel exactly in BOTH theories, leaving opposite-
    #     signed position architecture (PGDT: d>0 at every corner; PW-IVC: d<0 at every
    #     corner). Double-weighted because this is the only sign-DISJOINT family.
    # (3) Coalition cells: counting and distrust both cancel; only position varies.
    #     PGDT separates the coalition strongly (d in [0.08, 0.39]); PW-IVC caps at
    #     d <= 0.11. Double-weighted to use the 12 trials efficiently.
    crossover = p_l + p_h - 1.0
    m_term = 2.0 * (p_m - 0.5)
    c_term = 2.0 * (p_c - 0.5)

    return float(crossover + m_term + c_term)
```

**Observed (real) value:** 1.8356 (var=0.2112)
**Predicted under pi_7:** 1.1900 (var=0.2144)
**Predicted under pi_9:** 1.6694 (var=0.1653)

### Experiment 16
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 0, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Critical "primacy-rescue" trial family: exactly one option carries the
    # position-1 trusted (0.95) expert, the opposing option carries only
    # early low-validity (0.50) experts. Six f1-side patterns (option that
    # holds the position-1 trusted expert) x two counterpart patterns.
    P = {
        (1, 0, 0, 0, 0, 1),  # {95%@1, 50%@6}          (L1 rung)
        (1, 0, 0, 0, 1, 1),  # {95%@1, 50%@5, 50%@6}   (L2 rung)
        (1, 0, 1, 0, 1, 1),  # {95%@1, 50%@3,5,6}     (L3 rung)
        (1, 0, 1, 0, 0, 0),  # {95%@1, 50%@3}          (L1e early helper)
        (1, 0, 1, 1, 0, 0),  # {95%@1, 50%@3, 95%@4}   (TB+ cell)
        (1, 0, 0, 0, 1, 0),  # {95%@1, 50%@5}          (FL cell)
    }
    Q = {
        (0, 1, 0, 0, 0, 0),  # {50%@2}
        (0, 1, 0, 1, 0, 0),  # {50%@2, 95%@4}
    }

    def to_key(x):
        return tuple(int(v) for v in x)

    hits = []
    a_col = data["option_a_ratings"]
    b_col = data["option_b_ratings"]
    r_col = data["response"]
    for a_raw, b_raw, r_raw in zip(a_col, b_col, r_col):
        a = to_key(a_raw)
        b = to_key(b_raw)
        r = int(r_raw)
        if a in P and b in Q:
            # trusted-early product is option A
            hits.append(1.0 if r == 0 else 0.0)
        elif b in P and a in Q:
            # trusted-early product is option B
            hits.append(1.0 if r == 1 else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.1322 (var=0.0089)
**Predicted under pi_7:** 0.4311 (var=0.0074)
**Predicted under pi_9:** 0.2639 (var=0.0052)

### Experiment 17
**Design**
  A=[1, 1, 1, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0, 1]  B=[0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 0]  B=[1, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0, 1]  B=[1, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 0, 1, 1, 1]  B=[1, 1, 0, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 0, 1, 1]  B=[1, 1, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 1, 1, 1]  B=[1, 1, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 0, 0]  B=[1, 1, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 1, 1, 1]  B=[0, 0, 0, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 1, 1]  B=[1, 1, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 1, 0]
  A=[1, 1, 1, 1, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    """Gate-vs-ramp discrimination score (theory-informed linear contrast).

    Each of the 32 design cells gets a fixed coefficient c equal (in sign and
    rough magnitude) to the predicted difference P(A | PGDT) - P(A | PRDT)
    for that cell.  The score is the mean of c * (+1 if chose A, -1 if chose B)
    over the session, x100.  Under PGDT (position-independent trusted bracket
    ordered by the primacy gate) the score is strongly positive; under PRDT
    (t0 = 0 trust ramp that kills early trusted experts) it is much smaller.
    Mirrored cells carry opposite coefficients, so side/position bias cancels.
    """
    if data is None or len(data) == 0:
        return 0.0
    for col in ("option_a_ratings", "option_b_ratings", "response"):
        if col not in data.columns:
            return 0.0

    # (A_ratings, B_ratings) -> contrast coefficient
    _TRIALS = [
        # --- trusted-position axis (sign-robust at every parameter corner) ---
        ((1, 1, 1, 1, 1, 1, 0), (0, 1, 1, 1, 1, 1, 1),  0.050),  # D17
        ((0, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 0), -0.050),
        ((1, 1, 1, 1, 1, 0, 1), (0, 1, 1, 1, 1, 1, 1),  0.045),  # D16
        ((0, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 0, 1), -0.045),
        ((1, 1, 1, 1, 1, 1, 0), (1, 0, 1, 1, 1, 1, 1),  0.022),  # D27-type
        ((1, 0, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 0), -0.022),
        ((1, 1, 1, 1, 1, 0, 1), (1, 0, 1, 1, 1, 1, 1),  0.020),  # D26-type
        ((1, 0, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 0, 1), -0.020),
        ((1, 0, 1, 1, 1, 1, 1), (0, 1, 1, 1, 1, 1, 1),  0.012),  # pos1 vs pos0
        ((0, 1, 1, 1, 1, 1, 1), (1, 0, 1, 1, 1, 1, 1), -0.012),
        ((1, 1, 1, 1, 1, 0, 0), (0, 0, 1, 1, 1, 1, 1),  0.060),  # BLK
        ((0, 0, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 0, 0), -0.060),
        ((1, 1, 1, 1, 1, 0, 0), (0, 1, 1, 1, 1, 1, 1),  0.040),  # BLKa
        ((0, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 0, 0), -0.040),
        ((1, 1, 1, 1, 1, 1, 0), (0, 0, 1, 1, 1, 1, 1),  0.070),  # T21
        ((0, 0, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 0), -0.070),
        # --- kernel-steepness / anti-validity axis ---
        ((1, 1, 1, 0, 1, 1, 1), (1, 1, 0, 1, 0, 1, 1), -0.080),  # ML1
        ((1, 1, 0, 1, 0, 1, 1), (1, 1, 1, 0, 1, 1, 1),  0.080),
        ((1, 1, 1, 0, 1, 1, 1), (1, 1, 0, 1, 1, 1, 1), -0.050),  # ML2
        ((1, 1, 0, 1, 1, 1, 1), (1, 1, 1, 0, 1, 1, 1),  0.050),
        ((0, 0, 0, 1, 0, 0, 0), (1, 1, 1, 0, 1, 1, 1),  0.050),  # ANCH
        ((1, 1, 1, 0, 1, 1, 1), (0, 0, 0, 1, 0, 0, 0), -0.050),
        ((1, 1, 1, 0, 1, 1, 1), (0, 1, 1, 1, 1, 1, 1), -0.015),  # TL
        ((0, 1, 1, 1, 1, 1, 1), (1, 1, 1, 0, 1, 1, 1),  0.015),
        ((1, 1, 1, 1, 0, 1, 1), (1, 1, 0, 1, 1, 1, 1),  0.012),  # MM
        ((1, 1, 0, 1, 1, 1, 1), (1, 1, 1, 1, 0, 1, 1), -0.012),
        ((1, 1, 1, 1, 0, 1, 0), (0, 1, 0, 1, 1, 1, 1),  0.045),  # mixed
        ((0, 1, 0, 1, 1, 1, 1), (1, 1, 1, 1, 0, 1, 0), -0.045),
        ((1, 1, 1, 1, 1, 0, 0), (0, 1, 1, 0, 1, 1, 1),  0.005),  # MX
        ((0, 1, 1, 0, 1, 1, 1), (1, 1, 1, 1, 1, 0, 0), -0.005),
        # TOT anchors: no theory contrast (lapse-only difference) -> weight 0
        ((1, 1, 1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0, 0),  0.000),
        ((0, 0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1, 1, 1),  0.000),
    ]
    W = {(a, b): c for a, b, c in _TRIALS}

    def _key(cell):
        # robustly convert a list/tuple/ndarray/string of binary ratings
        # into a hashable tuple of ints
        if isinstance(cell, str):
            s = cell.replace("[", " ").replace("]", " ").replace(",", " ")
            try:
                return tuple(int(round(float(t))) for t in s.split())
            except ValueError:
                return None
        try:
            arr = np.asarray(cell, dtype=float).ravel()
            return tuple(int(round(float(v))) for v in arr)
        except Exception:
            return None

    total = 0.0
    n = 0
    for a, b, r in zip(data["option_a_ratings"].tolist(),
                       data["option_b_ratings"].tolist(),
                       data["response"].tolist()):
        ka = _key(a)
        kb = _key(b)
        if ka is None or kb is None:
            continue
        c = W.get((ka, kb), 0.0)
        try:
            rv = int(round(float(r)))
        except (TypeError, ValueError):
            continue
        y = 1.0 if rv == 0 else -1.0  # +1 = chose A, -1 = chose B
        total += c * y
        n += 1

    if n == 0:
        return 0.0
    return 100.0 * total / n
```

**Observed (real) value:** 1.0358 (var=0.1926)
**Predicted under pi_7:** 1.2013 (var=0.2191)
**Predicted under pi_9:** 1.0273 (var=0.1192)

### Experiment 18
**Design**
  A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 1, 1, 1]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 1, 1, 1]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 1, 1, 1]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 1, 1, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    # Two-stage, layout-adaptive theory discriminant.
    # Stage 1 infers which validity layout generated the data (the design
    # header lists [0.99, 0.99, 0.99, 0.75, 0.75, 0.75, 0.75, 0.75] while
    # the rationale/trial list is built around a 0.55 expert at position
    # 5). The inference uses only the clean two-cue duels involving
    # position 5: under the 0.55 layout BOTH theories prefer the
    # position-5-endorsed product with P ~ 0.7-0.9; under the all-0.75
    # layout that probability is ~ 0.5. This is a theory-agnostic,
    # mirror-balanced statistic, so it is safe to branch on.
    # Stage 2 computes a gap-weighted choice discriminant: for every unique
    # (A, B) rating pattern it evaluates the box-averaged predicted P(A)
    # of PRDT (distrust inversion with gamma in [1.6, 2.2], floored primacy
    # gate, position-growing trust ramp with t0 pinned at 0) and of PGDT
    # (gamma in [2.8, 3.5], position-independent flat share (1-tau)/8),
    # using each theory's own mechanics verbatim, and weights the observed
    # choose-A indicator by the signed predicted gap. Mirrored cells
    # automatically reinforce. The score is linear in the trial outcomes,
    # so the pooled value equals the mean of the per-subject values and
    # the metric works on single-subject slices.

    def _vec(x):
        try:
            arr = np.asarray(list(x), dtype=float)
            if arr.size == 0:
                raise ValueError("empty cell")
            return arr
        except Exception:
            s = str(x).strip()
            for ch in "[](),":
                s = s.replace(ch, " ")
            return np.asarray([float(t) for t in s.split() if t != ""], dtype=float)

    if data is None or len(data) == 0:
        return 0.0

    df = data.reset_index(drop=True)
    a_list = [_vec(x) for x in df["option_a_ratings"].tolist()]
    b_list = [_vec(x) for x in df["option_b_ratings"].tolist()]
    resp = df["response"].to_numpy(dtype=float)
    n = int(a_list[0].size)
    chose_a = 1.0 - resp

    if n != 8:
        return float(np.mean(chose_a))

    # ---------- Stage 1: layout inference ----------
    y5 = []
    for av, bv, ca in zip(a_list, b_list, chose_a):
        s = np.sign(av - bv)
        nz = np.where(np.abs(s) > 0.5)[0]
        if len(nz) == 2 and 4 in nz:
            endorsed_a = bool(s[4] > 0)
            picked_a = bool(ca > 0.5)
            y5.append(1.0 if (endorsed_a == picked_a) else 0.0)
    p5 = float(np.mean(y5)) if len(y5) > 0 else 0.5

    if p5 > 0.60:
        v = np.array([0.99, 0.99, 0.99, 0.75, 0.55, 0.75, 0.75, 0.75], dtype=float)
    else:
        v = np.array([0.99, 0.99, 0.99, 0.75, 0.75, 0.75, 0.75, 0.75], dtype=float)

    # ---------- Stage 2: reference theory predictions ----------
    def _prdt_w(tau, eta, delta, gamma, c, t1, kappa):
        err = 1.0 - v
        centered = np.maximum(err - c * float(err.min()), 0.0)
        inv = np.power(centered, gamma)
        Z = float(inv.sum())
        if Z > 1e-12:
            w_inv = inv / Z
        else:
            w_inv = np.full(8, 0.125)
        pos = np.arange(8, dtype=float)
        t = t1 * np.power(pos / 7.0, kappa)  # t0 pinned at exactly 0
        g = delta + (1.0 - delta) * np.power((8.0 - pos) / 7.0, eta)
        bracket = t * (1.0 - tau) / 8.0 + tau * w_inv
        raw = g * bracket
        Zt = float(raw.sum())
        if Zt > 1e-12:
            return raw / Zt
        return np.full(8, 0.125)

    def _pgdt_w(tau, eta, delta, gamma, c):
        err = 1.0 - v
        centered = np.maximum(err - c * float(err.min()), 0.0)
        inv = np.power(centered, gamma)
        Z = float(inv.sum())
        if Z > 1e-12:
            w_inv = inv / Z
        else:
            w_inv = np.full(8, 0.125)
        pos = np.arange(8, dtype=float)
        g = delta + (1.0 - delta) * np.power((8.0 - pos) / 7.0, eta)
        bracket = (1.0 - tau) / 8.0 + tau * w_inv
        raw = g * bracket
        Zt = float(raw.sum())
        if Zt > 1e-12:
            return raw / Zt
        return np.full(8, 0.125)

    def _pa(w, s, beta, eps):
        d = float(np.dot(w, s))
        p = 0.5 * (1.0 + np.tanh(0.5 * beta * d))
        return (1.0 - eps) * p + 0.5 * eps

    prdt_base = dict(tau=0.915, eta=5.5, delta=0.275, gamma=1.9, c=0.15,
                     t1=0.925, kappa=2.25, beta=4.75, eps=0.11)
    prdt_draws = [dict(prdt_base)]
    for gm in (1.6, 2.2):
        for ta in (0.90, 0.93):
            for be in (4.0, 5.5):
                for t1v in (0.85, 1.0):
                    dr = dict(prdt_base)
                    dr["gamma"] = gm
                    dr["tau"] = ta
                    dr["beta"] = be
                    dr["t1"] = t1v
                    prdt_draws.append(dr)
    prdt_edges = {"eta": (5.0, 6.0), "delta": (0.25, 0.30),
                  "kappa": (1.5, 3.0), "c": (0.0, 0.3), "eps": (0.09, 0.13)}
    for key in prdt_edges:
        for val in prdt_edges[key]:
            dr = dict(prdt_base)
            dr[key] = val
            prdt_draws.append(dr)

    pgdt_base = dict(tau=0.87, eta=5.0, delta=0.325, gamma=3.15, c=0.175,
                     beta=4.25, eps=0.105)
    pgdt_draws = [dict(pgdt_base)]
    for gm in (2.8, 3.5):
        for ta in (0.82, 0.92):
            for dl in (0.20, 0.45):
                for be in (2.5, 6.0):
                    dr = dict(pgdt_base)
                    dr["gamma"] = gm
                    dr["tau"] = ta
                    dr["delta"] = dl
                    dr["beta"] = be
                    pgdt_draws.append(dr)
    pgdt_edges = {"eta": (4.0, 6.0), "c": (0.10, 0.25), "eps": (0.06, 0.15)}
    for key in pgdt_edges:
        for val in pgdt_edges[key]:
            dr = dict(pgdt_base)
            dr[key] = val
            pgdt_draws.append(dr)

    # ---------- Stage 3: gap-weighted choice discriminant ----------
    info = {}
    for av, bv, ca in zip(a_list, b_list, chose_a):
        key = (tuple(int(round(x)) for x in av), tuple(int(round(x)) for x in bv))
        if key not in info:
            s = np.sign(av - bv)
            pp = []
            for dr in prdt_draws:
                w = _prdt_w(dr["tau"], dr["eta"], dr["delta"], dr["gamma"],
                            dr["c"], dr["t1"], dr["kappa"])
                pp.append(_pa(w, s, dr["beta"], dr["eps"]))
            pg = []
            for dr in pgdt_draws:
                w = _pgdt_w(dr["tau"], dr["eta"], dr["delta"], dr["gamma"], dr["c"])
                pg.append(_pa(w, s, dr["beta"], dr["eps"]))
            info[key] = [0, 0.0, float(np.mean(pp)) - float(np.mean(pg))]
        info[key][0] += 1
        info[key][1] += float(ca)

    num = 0.0
    den = 0.0
    for key in info:
        cnt, na, gap = info[key]
        if abs(gap) < 1e-9:
            continue
        y = na / float(cnt)
        num += gap * (2.0 * y - 1.0) * cnt
        den += abs(gap) * cnt
    if den <= 1e-12:
        return 0.0
    return float(num / den)
```

**Observed (real) value:** 0.3217 (var=0.0143)
**Predicted under pi_7:** 0.2921 (var=0.0253)
**Predicted under pi_9:** 0.3709 (var=0.0137)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the full experiment set, PBDT (pi_9) is the better of the two theories, and the decisive evidence comes from the experiments specifically built as optimal discriminators of the two. Experiment 1's metric is a box-marginal log Bayes factor between the two composite hypotheses: PGDT-simulated data yields +12.63, PBDT-simulated data yields -8.31, and the observed value is -9.12 \u2014 almost exactly on PBDT's prediction and decisively in PBDT's sign region. Experiment 2's early-trusted-channel contrast shows the same: observed -0.93 vs PGDT +1.86 / PBDT -1.59. Experiment 16 (primacy-rescue) is the third strong discriminator: observed 0.132, with PBDT (0.264) far closer than PGDT (0.431). Experiments 15 and 17 also favor PBDT (1.669 vs observed 1.836; 1.027 vs observed 1.036). PGDT does win on more experiments in raw count (Exps 3, 4, 5, 7, 9, 11, 12, 14, 18), but most of those wins are marginal (differences of 0.01-0.06 in predicted values), whereas PBDT's wins on the discriminant experiments are large and often an order of magnitude larger than the between-subject variance. The core structural conclusion is clear: subjects' trust in high-validity experts is position-dependent with a crushed early floor (PBDT's ramp), not position-independent (PGDT's flat share). However, PBDT is not adequate as a final answer: it systematically over-predicts primacy-rescue (Exp16: 0.264 vs 0.132 observed \u2014 the early trusted expert is even more crushed than PBDT allows), over-predicts tally/count allegiance (Exp4: 0.315 vs 0.158), under-predicts late-trusted-expert following (Exp10: 0.279 vs 0.396), and under-shoots the anti-validity slope (Exp5: -0.418 vs -0.576) and the position-crossover composite (Exp15: 1.669 vs 1.836). These are consistent, structural residuals, not noise. Per the arbitration logic, since one theory is clearly superior but still fails systematically on a subset of experiments, the right move is to replace the degenerate loser (PGDT, pi_7) with a new theory that is a stronger competitor to PBDT \u2014 one that keeps PBDT's verified architecture (distrust inversion + trust ramp with crushed early floor) while repairing the shared failure modes.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose 'RECIPROCAL-TRUST DISTRUST TALLYING (RTDT)' as the replacement for pi_7. Architecture: (1) Keep the distrust-inversion kernel verbatim (inv_j = ((1-v_j) - c*min(1-v))^gamma / Z) but steepen it: gamma in [3.0, 4.2], tau in [0.92, 0.96]. This directly repairs the under-predicted anti-validity slope (Exp5 observed -0.576 vs -0.418) and the under-shot position-crossover composite (Exp15), and the stronger kernel dominance automatically weakens count/tally allegiance toward Exp4's observed 0.158. (2) Replace the trust ramp with a RECIPROCAL one: t_j = t0 * (t1/t0)^(j/(n-1))^kappa with t0 in [0.00, 0.12] \u2014 an even lower early-trust floor than PBDT's [0.25, 0.35], fixing Exp16's severe over-prediction of primacy-rescue (observed 0.132 vs PBDT 0.264) \u2014 and t1 in [1.0, 1.6] with kappa in [0.6, 1.2], so LATE high-validity experts are not merely restored but AMPLIFIED above the flat share. This late-trusted amplification is the key novel lever: it fixes Exp10 (final 0.90-validity expert followed at 0.396 while both current theories predict ~0.28) without raising the early floor. (3) Keep the floored convex primacy gate but at the moderate range (eta [4.5, 6.5], delta [0.15, 0.30]) \u2014 the steep-gate/low-floor corner that PBDT verified. (4) Attenuate the choice rule: beta in [2.8, 4.0], epsilon in [0.12, 0.18], since both current theories systematically over-commit on allegiance metrics (Exps 4, 6, 9). (5) Evidence integration stays additive over discriminating cues, d = sum_j w_j*sign(A_j - B_j), logistic in beta*d with uniform lapse. The theory differs from PBDT in exactly three quantifiable respects (near-zero t0, super-unity t1, steeper gamma) each targeted at a documented residual, and differs from PGDT in retaining the position-dependent trusted channel that the log-Bayes-factor evidence decisively supports. It should be falsifiable on: Exp16 (predict rescue rate near 0.10-0.20), Exp10 (FEDI near 0.38-0.45), Exp4 (allegiance near 0.15-0.22), while matching PBDT on Exps 1, 2, 15, 17 where PBDT already succeeds."
}
```

## Usage

```json
{
  "input_tokens": 54067,
  "output_tokens": 3106,
  "reasoning_tokens": 1838,
  "provider": "Sail Research"
}
```
