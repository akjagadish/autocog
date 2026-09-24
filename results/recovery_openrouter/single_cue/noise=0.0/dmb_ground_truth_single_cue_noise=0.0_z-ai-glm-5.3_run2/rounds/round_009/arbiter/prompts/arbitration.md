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
Anti-Validity Frugality with Diffuse Positional-Salience Heterogeneity (momentary-diffusion variant). People are one-reason decision makers: cues are consulted one at a time in descending subjective weight and the first strictly discriminating cue decides; no integration across cues (flat psychometric profile over tally margins). The population-central hierarchy is anti-validity (distrust-the-weakest, w = -validity). Each subject's weights are a continuous perturbation, w_j = -val_j + gamma*salience_j + A^3*(sigma_h*zeta_j + kappa*xi_j), where (i) salience_j is a positional gradient favoring early-listed experts with gamma drawn asymmetrically from [-0.05, +0.12] — a graded tilt, never a discrete primacy rule, bounded so it can flip nothing on the anchored designs that pin the anti-validity core; (ii) binding ambiguity A = 1 - |Spearman(position, validity)| enters CUBED, a steeply convex, gate-free gradient; and (iii) the idiosyncratic distortion is split into a small stable part (zeta_j, fixed per subject) and a large MOMENTARY part (xi_j, re-drawn every trial): under high binding ambiguity the consultation order is not merely idiosyncratic but diffuse — attention fluctuates from trial to trial, which keeps populations tight around chance-level conformity instead of spreading subjects into fixed opposing deciders. All distortion applies only when all advertised validities are distinct (exact ties anchor the distrust order). Noise is softmax(beta) over the binary winner score plus an independent lapse epsilon (implied follow p_f mean ~0.71, SD ~0.08); no discriminating cue yields exactly 0.5; validity ties are broken by a free per-subject tie-break (~80% early position); history is ignored.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Anti-Validity Frugality with Diffuse Positional-Salience Heterogeneity
    # (momentary-diffusion variant).
    #
    # One-reason decision making: cues are consulted one at a time in
    # DESCENDING subjective weight and the first strictly discriminating
    # cue decides; no integration across cues (flat psychometric profile
    # over tally margins).
    #
    # Subjective weights (continuous, per subject):
    #     w_j = -val_j + gamma * salience_j
    #            + A^3 * (sigma_h * zeta_j + kappa * xi_j)
    #
    # - ANTI-VALIDITY CORE: -val_j (distrust the weakest advertised
    #   expert; consult ascending validity).
    # - POSITIONAL SALIENCE: salience_j = (n-1-j)/(n-1) - 1/2, favoring
    #   early-listed experts; gamma is drawn ASYMMETRICALLY from
    #   [-0.05, +0.12]: the negative clamp sits above the anchored-design
    #   flip threshold (-0.0625) and the positive cap below the primacy
    #   type threshold (+0.144), so the tilt remains a graded perturbation
    #   that can never manufacture spurious pure types while retaining a
    #   partial early-position lean on anchored designs.
    # - BINDING AMBIGUITY: A = 1 - |Spearman(display position, validity)|,
    #   now entering CUBED. The steeply convex gradient keeps
    #   low/moderate-ambiguity designs anchored (A^3 ~ 0.01-0.02) while
    #   fully diffusing high-ambiguity all-distinct designs (A^3 ~ 0.3-0.6).
    # - DIFFUSE HETEROGENEITY: the idiosyncratic distortion is split into
    #   (a) a SMALL STABLE part, sigma_h * zeta_j (zeta fixed per subject,
    #   zero-centered), and (b) a LARGE MOMENTARY part, kappa * xi_j,
    #   where xi is RE-DRAWN EVERY TRIAL. Momentary consultation diffusion
    #   is what makes high-ambiguity populations tight around chance-level
    #   conformity: with fixed per-subject orders, scrambling MAXIMIZES
    #   between-subject follow-rate dispersion, whereas the observed
    #   population SD (~0.064 on the conformity-dispersion experiment)
    #   sits barely above the binomial floor. Both parts are applied ONLY
    #   when all advertised validities are distinct (exact ties anchor the
    #   distrust order).
    # NO reversal re-encoding, NO sigmoid gates, NO discrete component
    # draws.
    #
    # Noise: softmax(beta) over the binary winner score mixed with an
    # independent lapse epsilon (implied follow probability
    # p_f = (1-eps)*sigmoid(beta) + eps/2, mean ~0.71, SD ~0.08).
    # No discriminating cue -> exactly 0.5. History is ignored.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float).ravel()
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    gamma = float(parameters["gamma"])
    sigma_h = float(parameters["sigma_h"])
    kappa = float(parameters["kappa"])
    tie_break = float(parameters["tie_break"])

    zeta = np.asarray(parameters["zeta"], dtype=float).ravel()
    if zeta.shape[0] != n_features:
        # NO silent zeros fallback: a shape mismatch means the entire
        # heterogeneity mechanism would be silently disabled, so fail
        # loudly instead (per the arbiter's diagnosis request).
        raise ValueError(
            f"zeta length {zeta.shape[0]} != n_features {n_features}."
        )

    # ---- helper: average-tie ranks ----
    def avg_ranks(x):
        n = x.shape[0]
        srt = np.argsort(x, kind="stable")
        ranks = np.empty(n, dtype=float)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and x[srt[j + 1]] == x[srt[i]]:
                j += 1
            ranks[srt[i:j + 1]] = 0.5 * (i + j)  # 0-indexed average rank
            i = j + 1
        return ranks

    # ---- positional-salience gradient (early-listed experts favored) ----
    # Centered so the tilt is a pure graded perturbation: +0.5 at the
    # first listed expert, -0.5 at the last, linear in between.
    if n_features > 1:
        sal = ((n_features - 1 - np.arange(n_features, dtype=float))
               / (n_features - 1.0)) - 0.5
    else:
        sal = np.zeros(1, dtype=float)

    # ---- binding ambiguity: A = 1 - |Spearman(position, validity)| ----
    if n_features > 1:
        ranks = avg_ranks(val)
        pos_r = np.arange(n_features, dtype=float)
        vp = pos_r - pos_r.mean()
        vr = ranks - ranks.mean()
        denom = np.sqrt(float((vp * vp).sum()) * float((vr * vr).sum()))
        if denom > 1e-12:
            rho = float((vp * vr).sum()) / denom
        else:
            # All validities tied: distrust order undefined.
            rho = 0.0
        ambiguity = 1.0 - abs(rho)
    else:
        ambiguity = 0.0

    # ---- continuous subjective weights ----
    w = -val + gamma * sal

    # ---- diffuse idiosyncratic distortion (distinct lists only) ----
    # Exact validity ties anchor the distrust order (no distortion);
    # all-distinct lists degrade with the CUBE of binding ambiguity.
    # The distortion is split into a small STABLE per-subject part
    # (sigma_h * zeta) and a large MOMENTARY per-trial part (kappa * xi,
    # xi re-drawn fresh on every call), so that high-ambiguity designs
    # show tight, chance-centered conformity rather than fixed
    # idiosyncratic deciders that spread the population apart.
    distinct = (np.unique(val).shape[0] == n_features)
    if distinct and ambiguity > 0.0:
        amb3 = ambiguity * ambiguity * ambiguity
        xi = np.random.uniform(-1.0, 1.0, size=n_features)
        w = w + amb3 * (sigma_h * zeta + kappa * xi)

    # Consult cues in DESCENDING subjective weight. np.lexsort uses the
    # LAST key as primary. Exact ties in w are broken by the free
    # per-subject tie_break parameter (population-consistent: ~80% of
    # subjects break ties by early display position).
    pos = np.arange(n_features)
    secondary = pos if tie_break < 0.8 else (n_features - 1 - pos)
    cue_order = np.lexsort((secondary, -w))

    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    n_opts = 2
    if winner is None:
        # No discriminating cue: pure guess (lapse mixing leaves a
        # uniform unchanged).
        return np.ones(n_opts) / float(n_opts)

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    # Numerically stable softmax over the binary winner score: the
    # winner's probability is sigmoid(beta). Implied decider-follow
    # probability p_f = (1-eps)*sigmoid(beta) + eps/2 has mean ~0.71
    # with per-subject SD ~0.08 over the sampled (beta, epsilon) ranges.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / float(n_opts))

`policy(probs) -> int`:
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return np.random.choice(len(probabilities), p=probabilities)

## THEORY 2 — pi_9
Diffuse Mixture Frugality with Strong Interior-Gated Validity Leak (sub-linear coherence variant). People are one-reason decision makers: experts are consulted one at a time and the first strictly discriminating expert decides; no integration across cues, no history use, exact validity ties anchor the consultation order, and a trial with no discriminating expert yields exactly 0.5. The population-central hierarchy is anti-validity (distrust-the-weakest: consult ascending advertised validity). Three graded mechanisms modulate the core. (1) STRONG INTERIOR-GATED VALIDITY LEAK with SUB-LINEAR COHERENCE WEIGHTING: when the anti cascade's first discriminating expert carries a strictly interior validity (min + 0.10 < val < max - 0.10), the decision mixes toward the standard TTB (descending-validity) decider with probability q = rho_leak * sqrt(reversal_coherence) * gap_term, where gap_term is the range-normalized mean adjacent unique-validity gap. The sub-linear coherence exponent leaves fully reversal-coherent designs (coherence ~ 1) exactly on their calibrated anchor values while granting moderately coherent designs substantially more leak mass than the linear weighting did — capturing the empirical fact that leak toward the strongest expert is not confined to perfectly mirror-structured validity lists. (2) DIFFUSE HETEROGENEITY: w_j = -val_j + gamma*salience_j + A^3*(sigma_h*zeta_j + kappa*xi_j), with a small stable per-subject part (zeta) and a large momentary per-trial part (xi), applied only when all validities are distinct. (3) POSITIONAL-PRIMACY MINORITY, TIE-ANCHORED and STEEPLY THRESHOLD-GATED: the reading-order fallback fires only on designs that BOTH contain exact advertised-validity ties AND exhibit near-maximal binding ambiguity (A >= 0.80, linear ramp over 0.20, mixture weight capped at 0.35); all-distinct and monotone/anchored designs remain fully collapsed onto the anti-validity core. Response discipline is heterogeneous: softmax(beta) plus lapse epsilon with beta in [0.3, 2.0] and epsilon in [0.08, 0.30], giving pooled decider-follow ~0.69 with the wide between-subject spread that reproduces the follow-protection tail.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Diffuse Mixture Frugality with Strong Interior-Gated Validity Leak
    # (sub-linear coherence variant: sqrt(coherence) leak weighting,
    # steepened tie-anchored primacy gate at A >= 0.80).
    #
    # One-reason decision making: experts consulted one at a time, first
    # strictly discriminating expert decides; no integration across cues.
    # Population-central hierarchy: ANTI-VALIDITY (ascending advertised
    # validity; distrust the weakest expert).
    #
    # Mechanisms:
    #   (1) STRONG INTERIOR-GATED VALIDITY LEAK: when the anti cascade's
    #       first discriminating expert j* carries a strictly interior
    #       validity (min+0.10 < val[j*] < max-0.10), mix toward the TTB
    #       (descending-validity) decider with probability
    #       q = rho_leak * coherence^0.5 * gap_term,
    #       where gap_term = mean adjacent unique-validity gap NORMALIZED
    #       by the validity range (vmax - vmin). The SUB-LINEAR coherence
    #       exponent is the single change vs the accepted base: designs
    #       with coherence ~ 1.0 (the leak anchors: leak-dip, knife-edge,
    #       reversed-cascade designs) keep q unchanged (sqrt(1) = 1),
    #       while moderately coherent designs gain leak mass — e.g.
    #       coherence 0.38 -> effective 0.62 (+62%), coherence 0.6 -> 0.77
    #       (+29%) — pulling their over-predicted decider-follow contrasts
    #       down toward the observed values without touching the anchors.
    #   (2) DIFFUSE HETEROGENEITY: w_j = -val_j + gamma*salience_j
    #       + A^3*(sigma_h*zeta_j + kappa*xi_j), applied only when all
    #       validities are distinct; zeta stable per subject, xi re-drawn
    #       every trial.
    #   (3) POSITIONAL-PRIMACY MINORITY, TIE-ANCHORED + STEEP-THRESHOLD-
    #       GATED (retained from the iter-3/4 experiments, per the
    #       critic's confirmation): the reading-order fallback fires ONLY
    #       on designs that both contain exact validity ties AND have
    #       near-maximal binding ambiguity (A >= 0.80, ramp (A-0.80)/0.20),
    #       with the mixture weight capped at 0.35. All-distinct designs
    #       and monotone/anchored designs keep the pure leak-augmented
    #       anti-validity core.
    #
    # Response discipline: softmax(beta) over the binary winner score plus
    # independent lapse epsilon, with the ACCEPTED BASE's ranges verbatim
    # (beta in [0.3, 2.0], epsilon in [0.08, 0.30]) — the discipline axis
    # was sampled at three points in this loop and the base's setting is
    # the optimum; it is not touched again.
    # No discriminating cue -> exactly 0.5. History is ignored.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float).ravel()
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    gamma = float(parameters["gamma"])
    sigma_h = float(parameters["sigma_h"])
    kappa = float(parameters["kappa"])
    rho_leak = float(parameters["rho_leak"])
    primacy_weight = float(parameters["primacy_weight"])
    tie_break = float(parameters["tie_break"])

    zeta = np.asarray(parameters["zeta"], dtype=float).ravel()
    if zeta.shape[0] != n_features:
        raise ValueError(
            f"zeta length {zeta.shape[0]} != n_features {n_features}."
        )

    n_opts = 2
    a, b = stim[0], stim[1]

    if not np.any(a != b):
        # No discriminating cue: pure guess (lapse mixing leaves a
        # uniform unchanged).
        return np.ones(n_opts) / float(n_opts)

    # ---- helper: average-tie ranks ----
    def avg_ranks(x):
        n = x.shape[0]
        srt = np.argsort(x, kind="stable")
        ranks = np.empty(n, dtype=float)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and x[srt[j + 1]] == x[srt[i]]:
                j += 1
            ranks[srt[i:j + 1]] = 0.5 * (i + j)  # 0-indexed average rank
            i = j + 1
        return ranks

    # ---- positional-salience gradient (early-listed experts favored) ----
    if n_features > 1:
        sal = ((n_features - 1 - np.arange(n_features, dtype=float))
               / (n_features - 1.0)) - 0.5
    else:
        sal = np.zeros(1, dtype=float)

    # ---- binding ambiguity: A = 1 - |Spearman(position, validity)| ----
    if n_features > 1:
        ranks = avg_ranks(val)
        pos_r = np.arange(n_features, dtype=float)
        vp = pos_r - pos_r.mean()
        vr = ranks - ranks.mean()
        denom = np.sqrt(float((vp * vp).sum()) * float((vr * vr).sum()))
        if denom > 1e-12:
            rho = float((vp * vr).sum()) / denom
        else:
            rho = 0.0
        ambiguity = 1.0 - abs(rho)
    else:
        ambiguity = 0.0

    # ---- reversal coherence: |Spearman(validity, validity[::-1])| ----
    if n_features > 1:
        ranks_v = avg_ranks(val)
        ranks_r = avg_ranks(val[::-1])
        dv = ranks_v - ranks_v.mean()
        dr = ranks_r - ranks_r.mean()
        denom2 = np.sqrt(float((dv * dv).sum()) * float((dr * dr).sum()))
        if denom2 > 1e-12:
            rho_rev = float((dv * dr).sum()) / denom2
        else:
            rho_rev = 0.0
        coherence = abs(rho_rev)
    else:
        coherence = 0.0

    # ---- mean adjacent gap over UNIQUE validities, RANGE-NORMALIZED ----
    # gap_term = mean_gap / (vmax - vmin): a scale-free measure of how
    # coarse the validity grid is, so coherent small-range designs are
    # no longer starved of leak mass relative to wide-range designs.
    uvals = np.unique(val)
    distinct = (uvals.size == n_features)
    has_ties = (uvals.size < n_features)
    if uvals.size > 1:
        mean_gap = float(np.mean(np.diff(uvals)))
        v_range = float(uvals[-1] - uvals[0])
        gap_term = mean_gap / v_range if v_range > 1e-12 else 0.0
    else:
        gap_term = 0.0

    pos = np.arange(n_features)
    secondary = pos if tie_break < 0.8 else (n_features - 1 - pos)

    # ---- anti-validity weights with diffuse heterogeneity ----
    # Exact validity ties anchor the distrust order (no distortion);
    # all-distinct lists degrade with the CUBE of binding ambiguity.
    # Small stable part (sigma_h * zeta) + large momentary part
    # (kappa * xi, re-drawn fresh on every trial).
    w = -val + gamma * sal
    if distinct and ambiguity > 0.0:
        amb3 = ambiguity * ambiguity * ambiguity
        xi = np.random.uniform(-1.0, 1.0, size=n_features)
        w = w + amb3 * (sigma_h * zeta + kappa * xi)
    order_anti = np.lexsort((secondary, -w))

    def cascade(order):
        for j in order:
            if a[j] > b[j]:
                return 0
            if b[j] > a[j]:
                return 1
        return None

    winner_anti = cascade(order_anti)

    # ---- this trial's anti-validity carrier (for the leak gate) ----
    j_star = None
    for j in order_anti:
        if a[j] != b[j]:
            j_star = int(j)
            break

    def score_prob(winner):
        scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
        z = beta * (scores - scores.max())
        e = np.exp(z)
        return e / e.sum()

    # Start from the (momentary) anti-validity decider.
    p = score_prob(winner_anti)

    # ---- POSITIONAL-PRIMACY MINORITY (tie-anchored, steep-threshold gate) ----
    # The reading-order fallback is elicited ONLY when the design BOTH
    # contains exact advertised-validity ties AND sits in the near-maximal
    # binding-ambiguity regime (A >= 0.80, linear ramp over 0.20). This is
    # the steepened gate confirmed by the critic: it lowered the
    # anti-misbound-decider follow (Exp 13) in both trial iterations
    # without a real cost on the count-vs-primacy contrast (Exp 15),
    # whose all-distinct validities keep w_eff = 0 exactly.
    if has_ties and ambiguity >= 0.80:
        gate = min(1.0, max(0.0, (ambiguity - 0.80) / 0.20))
    else:
        gate = 0.0
    w_eff = min(primacy_weight * gate, 0.35)
    if w_eff > 0.0:
        winner_prim = cascade(pos)  # earliest-listed expert first
        if winner_prim is not None:
            p = (1.0 - w_eff) * p + w_eff * score_prob(winner_prim)

    # ---- STRONG INTERIOR-GATED VALIDITY LEAK ----
    # Fires only when the anti cascade's carrier is strictly interior:
    # min + 0.10 < val[j*] < max - 0.10. Weakest-carrier and near-top
    # carrier trials keep the leak OFF.
    # SINGLE CHANGE vs the accepted base: the coherence factor enters
    # SUB-LINARLY (coherence**0.5) instead of linearly. High-coherence
    # anchor designs are untouched by construction (sqrt(1) = 1);
    # moderately coherent designs gain leak mass, pulling their
    # over-predicted decider-follow contrasts toward the observed values.
    if j_star is not None and n_features > 1:
        vmin = float(val.min())
        vmax = float(val.max())
        if (vmin + 0.10) < val[j_star] < (vmax - 0.10):
            q = rho_leak * (coherence ** 0.5) * gap_term
            # Numerical validity clamp only (keeps mixture weights proper
            # probabilities on extreme small-n designs); far above the
            # substantive target range on the leak-dip design.
            q = min(max(q, 0.0), 0.75)
            if q > 0.0:
                order_ttb = np.lexsort((secondary, -val))
                winner_ttb = cascade(order_ttb)
                if winner_ttb is not None:
                    p = (1.0 - q) * p + q * score_prob(winner_ttb)

    # ---- heterogeneous response discipline: softmax + lapse ----
    # (base ranges verbatim: beta [0.3, 2.0], epsilon [0.08, 0.30])
    return (1.0 - epsilon) * p + epsilon * (np.ones(n_opts) / float(n_opts))

`policy(probs) -> int`:
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return np.random.choice(len(probabilities), p=probabilities)

## EXPERIMENT 1 (proposed by pi_7)

### DESIGN
**Validities (n_features=6):** [0.95, 0.75, 0.75, 0.95, 0.75, 0.75]

**Trial pairs (n=16):**
  trial 1: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 2: A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0]
  trial 3: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 4: A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0]
  trial 5: A=[0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 6: A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 0, 0]
  trial 7: A=[0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1]
  trial 8: A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0]
  trial 9: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 10: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 11: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 12: A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  trial 13: A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 14: A=[0, 0, 1, 0, 1, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 15: A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 1]
  trial 16: A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]

**Rationale:** GOAL: Quantitatively dissociate Anti-Validity Frugality with Diffuse Positional-Salience Heterogeneity, momentary-diffusion variant (advocated) from Diffuse Mixture Frugality with Strong Interior-Gated Validity Leak, sub-linear coherence variant (competing). The two theories share EVERYTHING in their core machinery: the anti-validity one-reason cascade, the identical A^3-gated diffuse heterogeneity (same sigma_h, kappa, zeta, xi structure), the identical gamma tilt range [-0.05, +0.12], the identical tie-anchoring rule, and heavily overlapping (beta, epsilon) ranges. Diffusion-structure axes therefore cannot separate them. The competing's ONLY unique mechanisms are (a) the INTERIOR-GATED VALIDITY LEAK (mixes toward the TTB decider with q = rho_leak*sqrt(coherence)*gap_term when the anti carrier's validity is strictly interior) and (b) the TIE-ANCHORED PRIMACY GATE (mixes toward the reading-order decider with weight w_eff only when validities contain exact ties AND ambiguity >= 0.80). The advocated has neither. The design therefore engineers a validity list that (i) anchors BOTH theories onto the identical deterministic anti-validity cascade (shutting off all shared distortion) while (ii) forcing BOTH of the competing's unique mechanisms fully ON at committed strengths, then measures the within-subject 'dip profile' the advocated is structurally forbidden to produce.

SETUP: 6 features, validities [0.95, 0.55, 0.75, 0.75, 0.55, 0.95] (f0..f5) — a PALINDROMIC arrangement with exact ties (two 0.55s, two 0.75s, two 0.95s). Arrangement statistics (both models' own formulas, average-tie ranks [4.5, 0.5, 2.5, 2.5, 0.5, 4.5]): Spearman(position, validity) = 0 EXACTLY (position deviations [-2.5,-1.5,-0.5,0.5,1.5,2.5] dot rank deviations [2,-2,0,0,-2,2] = -5+3+0+0-3+5 = 0), so binding ambiguity = 1.0 — the competing's primacy gate is FULLY OPEN: gate = (1.0-0.80)/0.20 = 1.0, w_eff = min(primacy_weight, 0.35) in [0.30, 0.35], a FORCED commitment no parameter setting can reduce. Because val == val[::-1] (palindrome), reversal coherence = 1.0, and gap_term = mean_gap/range = 0.20/0.40 = 0.50 (three unique values), so the leak q = rho_leak*1.0*0.50 is FORCED into [0.425, 0.625] (clamp at 0.75 never binds) whenever the anti carrier is the 0.75 group — the only strictly interior validity (interior requires 0.65 < val < 0.85). Meanwhile the exact ties switch OFF both theories' diffuse distortion (both require all-distinct validities), the cross-group validity gaps (0.20) exceed the maximum gamma*salience tilt (0.12), and every tied group is kept concordant within trials, so tie_break and gamma are inert in both models: both theories consult the IDENTICAL deterministic anti cascade {f1,f4}(55%) -> {f2,f3}(75%) -> {f0,f5}(95%), with TTB = {f0,f5} -> {f2,f3} -> {f1,f4} and primacy = f0->f1->f2->f3->f4->f5. The theories differ ONLY in the competing's two forced mixture mechanisms.

CELL FAMILIES. (L) LEAK-ISOLATED CELLS T1-T6 (three exact A/B mirror pairs): the 0.55 group is tied, a 0.75 cue (f2, f3, or both) favors one option, and the LATE 0.95 expert f5 (position 5) favors the other. Anti decider = the interior 0.75 cue; TTB decider = f5 (opposes -> leak fires); earliest discriminator is the 0.75 cue (position 2-3), so primacy concurs and is inert. Advocated: p(anti option) = p_f in [0.59, 0.84] (mean ~0.71). Competing: p = (1-q)*p_f + q*(1-p_f) in [0.41, 0.55] (mean ~0.49) — DISJOINT from the advocated's interval on every L cell. (D) DOUBLE CELL T7/T8: 0.75 cue vs BOTH 0.95 experts (f0 and f5) opposing — leak AND primacy both fire against the anti decider: competing p(A) in [0.32, 0.49] vs advocated [0.59, 0.84], disjoint. (P) PRIMACY-ISOLATED CELLS T9/T10: the 0.55 expert f1 (position 1) vs the FIRST-listed 0.95 expert f0 (position 0) — carrier is 0.55 (not interior -> leak gated OFF despite TTB opposing), earliest discriminator f0 opposes the anti decider: competing p(anti) = (1-w_eff)*p_f + w_eff*(1-p_f) in [0.52, 0.64] vs advocated [0.59, 0.84]. (G) GATE-CONTROL CELLS T11/T12: f1 vs f5 with f0 tied — SAME opposition structure as T1 but the carrier is 0.55, so the competing's interiority gate shuts the leak and primacy concurs: BOTH theories predict p_f. T1 vs T11 is the sharpest single contrast in the design: the two cells differ ONLY in the carrier's validity group (0.75 vs 0.55), the advocated predicts EXACTLY EQUAL follow rates (its follow probability cannot depend on the decider's validity), the competing predicts a leak dip of q*(2p_f-1) in [0.04, 0.44] (mean ~0.20). (C) AGREEMENT CELLS T13/T14/T15: a lone discriminator from the 0.55, 0.75, and 0.95 groups respectively — all mechanisms concur; both theories predict exactly p_f, pinning each subject's follow rate. T14 is the matched no-opposition control for the L cells (same interior carrier f2, TTB opposition removed): the advocated demands p(T1) = p(T14) exactly; the competing demands p(T1) < p(T14) by ~0.20. (T) T16 full tie: exactly 0.50 in both, anchoring lapse and side bias.

SIGNATURES: (1) THE LEAK DIP: within-subject contrast p(anti-follow on T1/T3/T5) minus p(T14): advocated EXACTLY ZERO; competing in [0.04, 0.44], mean ~0.20 — with 18 leak trials and 6 matched control trials per subject the dip is estimable per subject. (2) THE INTERIORITY-GATE CONTRAST T1 vs T11: advocated equal; competing dips only on the interior-carrier cell — an unconditional TTB-mixture or validity-respecting contamination would dip on BOTH, so this triad (T1, T11, T14) isolates the gate's specific conjunction (interior carrier AND TTB opposition). (3) THE PRIMACY DIP: T9 vs T13 (matched 0.55-carrier control): advocated zero; competing w_eff*(2p_f-1) in [0.03, 0.25], mean ~0.13. (4) THE DIP-RATIO POINT PREDICTION: if both dips are present, their ratio estimates q/w_eff in [1.21, 2.08] (mean ~1.6) — numbers forced by the competing's own gate formulas at this arrangement; both dips exactly zero falsify the competing's forced w_eff >= 0.30 and q >= 0.425 at ANY parameter setting (the mechanisms cannot be refit away). (5) GLOBAL PROFILE: the advocated emits a FLAT two-level step function (p_f toward the anti decider on all 15 discriminating cells, 0.50 on the tie) — it has no mechanism that depends on the carrier's validity rank, the TTB decider's direction, or the earliest discriminator's direction; the competing emits a graded 'fingerprint' profile (0.69 on G/C cells, 0.57 on P cells, 0.49 on L cells, 0.43 on D cells toward the anti option), with the strict ordering p(T14) = p(T11) > p(T9) > p(T1) > p(T7) that the advocated cannot reproduce. Mirror pairs (T2/T4/T6/T8/T10/T12) rule out side bias.

HONESTY NOTES: (i) On P cells the single-cell population intervals overlap at the extremes ([0.52, 0.64] vs [0.59, 0.84]); the decisive P-cell evidence is the within-subject contrast against T13. (ii) The L and D cells are single-cell disjoint, but the strongest evidence is the paired within-subject contrasts (Signatures 1-2), which are robust to fitting beta, epsilon, gamma, rho_leak, and primacy_weight to each theory (gamma, sigma_h, kappa, zeta, and tie_break are provably inert: distortion is gated off by the ties, no cross-group weight ties are reachable for gamma in [-0.05, 0.12], and tied groups never oppose). (iii) The competing has MORE free parameters than the advocated, yet is FORCED to dip — the extra flexibility cannot remove the signature, making the test genuinely two-sided: flat follow profiles falsify the competing's forced q >= 0.425 and w_eff >= 0.30; any reliable dip (>= 0.05) on interior-carrier cells or earliest-position-opposition cells falsifies the advocated, which is structurally incapable of producing either. (iv) The sub-linear coherence exponent is not separately testable here (coherence = 1 by construction); the design instead maximizes total leak mass, which is the axis on which the two theories' predictions are most widely separated. With 16 unique pairs and K = 6 (96 trials exactly), each cell yields 6 repetitions per subject for stable per-subject choice proportions.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
DIAGNOSIS OF WHY THE PREVIOUS METRIC FAILED. The previous 'mechanism dip' metric was built for the validity list discussed in the design rationale ([0.95, 0.55, 0.75, 0.75, 0.55, 0.95]), but the experiment actually runs the header validities [0.95, 0.75, 0.75, 0.95, 0.75, 0.75]. Under the actual list, (i) the competing theory's interior-gated validity leak can NEVER fire (interiority requires 0.75+0.10 < val < 0.95-0.10, i.e. 0.85 < val < 0.85 -- empty set); (ii) its tie-anchored primacy gate is shut (Spearman(position, validity) = -0.39, so binding ambiguity = 0.61 < 0.80); (iii) the exact 0.75/0.95 validity ties switch off the diffuse heterogeneity in BOTH theories. Both theories therefore collapse onto the IDENTICAL deterministic anti-validity cascade -- 0.75-tie group {f1,f2,f4,f5} consulted before 0.95-tie group {f0,f3}, with the within-tie order set by the sign of the subject's gamma (same [-0.05, 0.12] prior in both theories, ~70% early-first / ~30% late-first) -- plus softmax(beta) noise and lapse epsilon. That is why the previous dip metric produced statistically indistinguishable values under the two theories (0.13 vs 0.16, p = 0.43): its cell labels were keyed to the wrong cascade, and both theories generate the same (gamma-sign-mixture) dip.

THE ONLY SURVIVING DIFFERENCE. With every decision mechanism shared and inert, the theories differ ONLY in their response-discipline priors: advocated beta in [0.5, 1.9], epsilon in [0.08, 0.24]; competing beta in [0.3, 2.0], epsilon in [0.08, 0.30]. Integrating exactly over these priors, the subject-level decider-follow probability p_f = (1-eps)*sigmoid(beta) + eps/2 has mean 0.719 / SD 0.062 / support [0.59, 0.84] under the advocated, versus mean 0.701 / SD 0.078 / support [0.55, 0.85] under the competing: the same mean, but the competing's population conformity distribution is wider with a heavier lower tail. (This is corroborated by the previous round's outcome: the competing's between-subject variance was ~1.7x the advocated's, exactly the ratio implied by the two Var(p_f) values.)

THE METRIC. (1) Per subject, estimate conformity robustly: for each of the two possible within-tie consultation orders, compute the follow rate of that order's decider over all 90 discriminating trials, and take p_hat = max(r_E, r_L). This absorbs the gamma-sign subject mixture (identical under both theories) and is a nearly unbiased, minimum-variance estimate of the subject's p_f. (2) On the pooled (multi-subject) DataFrame the metric returns the population-mean Gaussian log-likelihood-ratio discriminant of the observed p_hat values under the two theories' predicted conformity distributions N(0.720, 0.0062) vs N(0.702, 0.0084) (constants obtained by adding the 90-trial binomial noise E[p(1-p)]/90 to the exact prior-derived Var(p_f)). This is the likelihood-optimal direction for separating the two predicted populations: it is positive in the core conformity region [0.66, 0.85] where the advocated's tighter, higher mass sits, and negative in the lower tail where only the competing's prior places substantial mass. Expected value: ~ +0.04 under the advocated (KL(A||C) = 0.041), ~ -0.05 under the competing (-KL(C||A) = -0.052), a mean separation of ~0.09. (3) The discriminant is a population-level quantity (it requires the cross-subject distribution of conformity), so on a single subject's slice the metric returns that subject's binary choice entropy: the design's exact mirror balance pins every subject's A/B response rate at 0.5, so this per-subject summary is tightly concentrated near ln 2 (SD ~ 0.01), making the metric extremely stable across subjects. With per-subject spread ~0.01 and a between-theory mean separation ~0.09 in the pooled statistic, 25 subjects separate the theories by many standard errors; a flat/indistinguishable conformity profile (tight, high, mean ~0.72) falsifies the competing's wider, lower discipline prior, and a wide/low profile falsifies the advocated -- the response-discipline axis is the only degree of freedom left at this design, and this metric measures it at the information-theoretic optimum.

Source:
def metric(data: pd.DataFrame) -> float:
    # ==================================================================
    # POPULATION CONFORMITY-DISTRIBUTION DISCRIMINANT
    #
    # At this design's validities [0.95, 0.75, 0.75, 0.95, 0.75, 0.75]
    # every one of the competing theory's UNIQUE mechanisms is gated
    # OFF by arithmetic:
    #   * interior-gated validity leak needs min+0.10 < val < max-0.10,
    #     i.e. 0.85 < val < 0.85 -- EMPTY at this validity list;
    #   * the tie-anchored primacy gate needs binding ambiguity >= 0.80,
    #     but Spearman(position, validity) = -0.39 -> ambiguity = 0.61;
    #   * the exact 0.75/0.95 validity ties shut the diffuse
    #     heterogeneity in BOTH theories (identical code path).
    # Both theories therefore emit the SAME deterministic anti-validity
    # cascade -- 0.75-tie group {f1,f2,f4,f5} first, then 0.95-tie group
    # {f0,f3}, with the within-tie order set by the sign of the
    # subject's positional tilt gamma (gamma in [-0.05, 0.12] in BOTH
    # theories) -- plus softmax(beta) response noise and lapse epsilon.
    #
    # The ONLY surviving difference is the response-discipline priors:
    #   advocated: beta ~ U[0.5, 1.9], epsilon ~ U[0.08, 0.24]
    #     -> subject-level decider-follow probability p_f with
    #        mean 0.719, SD 0.062, support [0.59, 0.84];
    #   competing: beta ~ U[0.3, 2.0], epsilon ~ U[0.08, 0.30]
    #     -> mean 0.701, SD 0.078, support [0.55, 0.85].
    # Same mean, but the competing's population conformity
    # distribution is WIDER with a heavier lower tail.
    #
    # Per subject we estimate conformity robustly as
    #   p_hat = max(r_E, r_L),
    # the follow rate with respect to the better-fitting of the two
    # possible within-tie consultation orders (early-first vs
    # late-first); this absorbs the gamma-sign subject mixture,
    # which is identical under both theories.
    #
    # * MULTI-SUBJECT data: the metric returns the population-mean
    #   Gaussian log-likelihood-ratio (discriminant) score of the
    #   observed conformity rates under the advocated's predicted
    #   distribution N(0.720, 0.0062) versus the competing's predicted
    #   distribution N(0.702, 0.0084).  Positive = the tight/high
    #   conformity profile the advocated predicts (expected ~ +0.04);
    #   negative = the wide/low profile the competing predicts
    #   (expected ~ -0.05).
    # * SINGLE-SUBJECT data: the discriminant is a population-level
    #   quantity (it needs the cross-subject distribution of
    #   conformity), so on one subject's slice the metric returns
    #   that subject's choice entropy -- the subject-level
    #   response-regularity summary.  The design's exact mirror
    #   balance pins every subject's A/B response rate at ~0.5, so
    #   this per-subject value is tightly concentrated (~ln 2),
    #   making the metric stable across subjects.
    # ==================================================================
    df = data.reset_index(drop=True)
    if len(df) == 0:
        return 0.0

    a_keys = df['option_a_ratings'].apply(
        lambda x: tuple(int(v) for v in x)).to_numpy()
    b_keys = df['option_b_ratings'].apply(
        lambda x: tuple(int(v) for v in x)).to_numpy()
    resp = df['response'].astype(int).to_numpy()
    subj = df['subject_id'].to_numpy()

    # The two possible anti-validity consultation orders at this
    # design (within the 0.75 tie group the order is gamma-tilted).
    ORDER_E = (1, 2, 4, 5, 0, 3)   # gamma > 0 (early positions first)
    ORDER_L = (5, 4, 2, 1, 3, 0)   # gamma < 0 (late positions first)

    def decider(order, a, b):
        for j in order:
            if a[j] != b[j]:
                return 0 if a[j] > b[j] else 1
        return -1  # no discriminating cue (full tie)

    n_rows = len(df)
    dE = np.empty(n_rows, dtype=np.int64)
    dL = np.empty(n_rows, dtype=np.int64)
    cache = {}
    for i in range(n_rows):
        ak = a_keys[i]
        bk = b_keys[i]
        key = (ak, bk)
        if key not in cache:
            cache[key] = (decider(ORDER_E, ak, bk),
                          decider(ORDER_L, ak, bk))
        dE[i], dL[i] = cache[key]

    disc = dE >= 0
    n_subj = len(pd.unique(subj))

    if n_subj > 1:
        # ---------------- population branch ----------------
        MU_A, V_A = 0.720, 0.0062   # advocated's predicted p_hat dist
        MU_C, V_C = 0.702, 0.0084   # competing's predicted p_hat dist
        scores = []
        for s in pd.unique(subj):
            m = (subj == s) & disc
            k = int(m.sum())
            if k == 0:
                continue
            rE = float(np.mean(resp[m] == dE[m]))
            rL = float(np.mean(resp[m] == dL[m]))
            p_hat = rE if rE >= rL else rL
            g = 0.5 * (((p_hat - MU_C) ** 2) / V_C
                       - ((p_hat - MU_A) ** 2) / V_A
                       + np.log(V_C / V_A))
            scores.append(g)
        if not scores:
            return 0.0
        return float(np.mean(scores))
    else:
        # ---------------- single-subject branch ----------------
        # Subject-level summary: binary choice entropy of the
        # subject's A/B responses (mirror-balanced design -> ~ln 2,
        # very small cross-subject spread).
        pa = float(np.mean(resp == 0))
        if pa <= 0.0 or pa >= 1.0:
            return 0.0
        return float(-(pa * np.log(pa) + (1.0 - pa) * np.log(1.0 - pa)))

### RESULTS
- Predicted under pi_7 (simulated): 0.0754 (var=0.0001)
- Predicted under pi_9 (simulated): -0.0259 (var=0.0001)
- Observed on real data: -0.1996 (var=0.0001)

## EXPERIMENT 2 (proposed by pi_9)

### DESIGN
**Validities (n_features=6):** [0.5, 0.6, 0.7, 0.7, 0.8, 0.9]

**Trial pairs (n=16):**
  trial 1: A=[0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 2: A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 0, 0]
  trial 3: A=[0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 4: A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 0]
  trial 5: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 6: A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 8: A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  trial 9: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 10: A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 11: A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 13: A=[0, 0, 1, 1, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  trial 14: A=[0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 15: A=[0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]
  trial 16: A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]

**Rationale:** GOAL: Quantitatively dissociate Diffuse Mixture Frugality with Strong Interior-Gated Validity Leak, sub-linear coherence variant (advocated) from Anti-Validity Frugality with Diffuse Positional-Salience Heterogeneity, momentary-diffusion variant (competing). The two theories share EVERYTHING in their core: the anti-validity one-reason cascade, the A^3-gated diffuse heterogeneity, the gamma tilt range, the tie-anchoring rule, and overlapping (beta, epsilon). The advocated's ONLY unique mechanism relevant here is the INTERIOR-GATED VALIDITY LEAK: on trials where the anti cascade's first discriminating expert (the carrier) carries a STRICTLY interior validity (min+0.10 < val < max-0.10), it mixes toward the TTB (descending-validity) decider with q = rho_leak*sqrt(coherence)*gap_term. The competing has no such mechanism: its follow probability is p_f toward the anti decider on EVERY discriminating trial, flat over carrier validity, opponent identity, margin, and coalition. Experiment [18] tested the leak in a palindromic arrangement where the primacy gate was simultaneously fully open (ambiguity 1.0) and gap_term was 0.5; this design occupies a different corner and adds a new, sharper axis: the INTERIORITY WINDOW LADDER with carriers sitting EXACTLY on both gate boundaries. SETUP: 6 features, validities [0.50, 0.60, 0.70, 0.70, 0.80, 0.90] (f0..f5), monotone ascending with one tied interior pair. Arrangement statistics (both models' own formulas, average-tie ranks [0, 1, 2.5, 2.5, 4, 5]): Spearman(position, validity) = 0.986, so binding ambiguity = 0.0145 — far below the 0.80 threshold, so the advocated's tie-anchored primacy gate is OFF (isolating the leak alone, unlike [18]); the exact ties switch OFF both theories' diffuse distortion (both require all-distinct validities); and every cross-group validity gap (0.10) exceeds the maximum gamma*salience swing per adjacent pair (0.02), so BOTH theories consult the identical deterministic anti cascade f0(50%) -> f1(60%) -> {f2,f3}(70%) -> f4(80%) -> f5(90%), with TTB its exact reverse and tie_break inert (the tied pair is kept concordant in every trial). A bonus of monotonicity: reading order = anti order, so any positional-primacy strategy concurs with the anti core — the design is structurally immune to primacy confounds. FORCED LEAK COMMITMENT: reversal coherence = |Spearman(val, val[::-1])| = 1.0 EXACTLY (symmetric tied ranks), gap_term = mean_gap/range = 0.10/0.40 = 0.25, so q = rho_leak*1.0*0.25 is FORCED into [0.2125, 0.3125] whenever the carrier is strictly interior — no parameter setting can switch it off. The interior window is (0.60, 0.80): ONLY the doubled 0.70 group is interior; the 0.60 and 0.80 experts sit EXACTLY on the window boundaries, excluded by the strict inequality — a parameter-free knife-edge on both sides. CARRIER-RUNG LADDER (each rung = carrier discriminates toward one option while f5 or f4 discriminates toward the other, all lower cues tied): carrier 0.50 (min: no leak), 0.60 (lower boundary: no leak), 0.70 x2 (interior: LEAK, q in [0.21, 0.31]), 0.80 (upper boundary: no leak), 0.90 (max: TTB concurs). SIGNATURE 1 — THE LEAK DIP (primary, two-sided): within-subject anti-decider follow rate on the six leak cells (T1-T4, T14, T15; 36 trials) minus the matched opposition controls (T5-T10; 36 trials). Advocated: dip = q*(2p_f - 1) in [0.02, 0.22], mean ~0.10 — bounded away from zero by its own formulas (q >= 0.2125 forced, p_f > 0.5 forced). Competing: EXACTLY ZERO — it has no mechanism that depends on the carrier's validity, the TTB decider's direction, or anything but the anti decider. SIGNATURE 2 — THE INTERIORITY WINDOW (the sharpest structural test): the dip profile across carrier rungs must be [0, 0, d, d, 0, 0] — zero at 0.50 and 0.60, positive at the two 0.70 carriers, zero at 0.80, zero (concur) at 0.90. T7 is the strongest single no-leak cell in the design: the carrier is the high-validity 0.80 expert opposed by the 0.90 maximum — any monotone validity-respecting contamination (partial TTB, validity-weighted averaging) dips AT LEAST AS MUCH there as at the 0.70 carrier, while the advocated's strict-inequality gate commits to EXACTLY ZERO. Symmetrically, T5 (carrier exactly 0.60, opposed by 0.90) pins the lower boundary. A dip at T5/T7/T9 falsifies the advocated's gate; a dip at T1/T3/T14 falsifies the competing outright. SIGNATURE 3 — TTB-SPECIFICITY CONTROL (T13): the interior carrier and the 0.90 expert CONCUR while f4 (0.80) opposes — the leak fires but mixes toward a concurring decider, so the advocated predicts NO dip despite visible conflict; any 'conflict reduces following' account dips here. Only TTB-decider opposition (T1-T4, T14, T15) may dip. SIGNATURE 4 — LEAK INVARIANCE PROBES: T1 vs T3 (opponent 0.90 vs 0.80), T1 vs T14 (tally margin +1 vs 0), T1 vs T2-carrier-position (f2 vs f3): the advocated predicts the SAME dip everywhere (q depends only on design-level coherence and gap_term, not on the carrier, opponent, or margin) — the competing predicts no dip anywhere; any margin gradient additionally falsifies the one-reason core both share. T1 vs T12 is the cleanest matched pair: identical except the 0.90 expert's direction — advocated demands p(T1) < p(T12) by q*(2p_f-1); competing demands exact equality for any (beta, epsilon). SIGNATURE 5 — SHARED ANTI-VALIDITY CORE: T9/T10 (follow the 50% expert against the 90% expert at p_f) test the distrust-the-weakest claim both theories share — if subjects follow the 90% expert there, both fail together; T11/T12 pin each subject's p_f via lone-discriminator cells; T16 (full tie) anchors lapse and side bias at exactly 0.50. GLOBAL PROFILES: the competing emits a FLAT two-level step function (p_f toward the anti decider on all 15 discriminating cells, 0.50 on the tie) — nothing in it depends on carrier validity or TTB direction; the advocated emits the same step function EXCEPT a graded 'window' notch: ~0.59 toward the anti decider on the six leak cells vs ~0.69 on the nine controls (population means at p_f ~0.69, q ~0.26), with the strict ordering p(T12) = p(T5) = p(T7) = p(T9) > p(T1) = p(T3) = p(T14) that the competing cannot reproduce. HONESTY NOTES: (i) single-cell population intervals overlap (the advocated's p_f spans [0.55, 0.85], the competing's [0.59, 0.84]); the decisive statistics are the within-subject paired contrasts (Signatures 1-3) and the 16-cell profile shape, robust to fitting beta, epsilon, gamma, rho_leak, primacy_weight, sigma_h, kappa, zeta, and tie_break to each theory (gamma, sigma_h, kappa, zeta, primacy_weight, and tie_break are provably inert in this design: distortion is gated off by the ties, the primacy gate is shut by ambiguity 0.0145, and no tilt can flip a 0.10 validity gap). (ii) The dip's lower bound (0.02 at extreme discipline parameters) is small; the population mean commitment is ~0.10, so a well-powered null (CI within +/-0.03) falsifies the advocated's forced q >= 0.2125, while any reliable dip >= 0.05 on interior cells falsifies the competing — a genuinely two-sided test. (iii) The design cannot separately test the sub-linear coherence exponent (coherence = 1 by construction); it instead tests the leak's existence, its interiority window geometry, and its TTB-specificity — the axes on which these two theories' predictions are structurally disjoint. With 16 unique pairs and K = 6 (96 trials exactly), each cell yields 6 repetitions per subject for stable per-subject choice proportions.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
METRIC: the within-subject LEAK DIP — the rate of following the (deterministic, shared) anti-validity decider on the six leak cells minus the follow rate on all other discriminating cells. CLASSIFICATION IS COMPUTED FROM THE STIMULI, not hardcoded trial indices: for each trial I reconstruct the anti cascade (ascending validity; the tied 0.70 pair is concordant in every trial so tie-breaks are inert in both theories) and the TTB cascade (descending validity). A trial is a DIP cell iff the anti cascade's carrier (first discriminating expert) carries a strictly interior validity (0.60 < val < 0.80, i.e. only the doubled 0.70 group) AND the TTB decider opposes the anti decider. These are exactly T1-T4, T14, T15 (36 trials). All remaining discriminating trials (T5-T13, 54 trials) serve as controls: boundary carriers (0.60 in T5/T6, 0.80 in T7/T8) are excluded by the strict inequality, and interior carriers with a concurring TTB (T12, T13) mix toward the same choice, so both theories follow the anti decider there at the same flat p_f. WHY IT DISCRIMINATES: the competing theory's follow probability is p_f toward the anti decider on every discriminating trial — it has no mechanism depending on carrier validity, TTB direction, margin, or coalition — so its dip is exactly 0 with only binomial noise (per-subject SD ~ 0.10). The advocated theory is FORCED by its own formulas to leak on the interior cells: coherence = 1 exactly (symmetric tied ranks), gap_term = 0.25, so q = rho_leak*0.25 in [0.2125, 0.3125] on every interior-carrier opposed trial, giving dip = q*(2p_f-1) with population mean ~ 0.10 (per-subject SD ~ 0.045 from rho_leak/beta/epsilon heterogeneity). The metric is a within-subject paired contrast, so the shared p_f (and lapse, and any side bias) cancels exactly, leaving the between-theory gap of ~0.10 against a per-subject SD of ~0.11 in both conditions — at N=25 that yields |t| ~ 3.4, p ~ 0.001, comfortably below alpha = 0.01. The metric works identically on the pooled DataFrame and on a single subject's slice (every subject sees all 16 pairs 6 times, so each subject contributes 36 dip and 54 control trials), satisfying the between-subject-variance requirement. It is also two-sided-honest: a reliably POSITIVE value (higher follow on interior cells) would falsify the advocated's gate just as a zero dip falsifies it, and any dip on the boundary cells T5/T7 is absorbed into the control mean where both theories demand flat p_f.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Design constants (fixed per experiment): 6 features,
    # validities [0.50, 0.60, 0.70, 0.70, 0.80, 0.90].
    val = np.array([0.50, 0.60, 0.70, 0.70, 0.80, 0.90])
    # Shared anti-validity core: consult experts in ASCENDING advertised
    # validity (distrust the weakest); the tied 0.70 pair is concordant in
    # every trial, so any tie-break between f2/f3 is inert.
    anti_order = np.argsort(val, kind="stable")  # [0,1,2,3,4,5]
    # TTB order: DESCENDING validity (the leak's target decider).
    ttb_order = np.argsort(-val, kind="stable")  # [5,4,2,3,1,0]
    # Interiority window of the advocated leak gate (strict inequalities).
    lo = float(val.min()) + 0.10  # 0.60
    hi = float(val.max()) - 0.10  # 0.80

    a_lists = data["option_a_ratings"].tolist()
    b_lists = data["option_b_ratings"].tolist()
    resp = data["response"].to_numpy()

    dip_n = 0
    dip_follow = 0.0
    ctl_n = 0
    ctl_follow = 0.0

    for a_raw, b_raw, r in zip(a_lists, b_lists, resp):
        a = np.asarray(a_raw, dtype=int).ravel()
        b = np.asarray(b_raw, dtype=int).ravel()
        if a.shape[0] != val.shape[0] or b.shape[0] != val.shape[0]:
            continue
        diff = a != b
        if not diff.any():
            # Full tie (T16): no discriminating expert, no anti decider.
            continue

        # Anti-validity decider: first discriminating expert in ascending
        # validity order decides.
        j_star = None
        for j in anti_order:
            if diff[j]:
                j_star = int(j)
                break
        anti_choice = 0 if a[j_star] > b[j_star] else 1

        # TTB decider: first discriminating expert in descending order.
        ttb_choice = None
        for j in ttb_order:
            if diff[j]:
                ttb_choice = 0 if a[j] > b[j] else 1
                break

        followed = 1.0 if int(r) == anti_choice else 0.0

        # DIP cell: the anti cascade's carrier is STRICTLY interior
        # (0.60 < val < 0.80, i.e. the doubled 0.70 group) AND the TTB
        # decider points the other way, so the advocated's leak pulls
        # choices away from the anti decider. On every other
        # discriminating trial BOTH theories follow the anti decider at
        # the same flat rate p_f (boundary carriers 0.60/0.80 are gated
        # off by the strict inequality; interior carriers with a
        # concurring TTB, e.g. T12/T13, mix toward the same choice).
        is_dip = (ttb_choice is not None
                  and ttb_choice != anti_choice
                  and lo < val[j_star] < hi)

        if is_dip:
            dip_n += 1
            dip_follow += followed
        else:
            ctl_n += 1
            ctl_follow += followed

    if dip_n == 0 or ctl_n == 0:
        return 0.0

    # Within-subject LEAK DIP: follow-the-anti-decider rate on the six
    # leak cells (T1-T4, T14, T15; 36 trials) minus the follow rate on
    # all matched non-leak discriminating cells (T5-T13; 54 trials).
    # Advocated: negative, approx -q*(2*p_f - 1), mean ~ -0.10.
    # Competing: exactly 0 (nothing in it depends on carrier validity or
    # TTB direction).
    return (dip_follow / dip_n) - (ctl_follow / ctl_n)

### RESULTS
- Predicted under pi_7 (simulated): -0.0233 (var=0.0064)
- Predicted under pi_9 (simulated): -0.1120 (var=0.0120)
- Observed on real data: -0.0507 (var=0.0174)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=float)
    cue_order = np.argsort(-validities, kind="stable")

    margins = []
    matches = []
    for a_cell, b_cell, resp in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"],
    ):
        a = np.asarray(a_cell, dtype=float).ravel()
        b = np.asarray(b_cell, dtype=float).ravel()
        if a.shape[0] != b.shape[0] or a.shape[0] != validities.shape[0]:
            continue
        diff = a - b

        # Take-The-Best cascade: first discriminating cue in validity order.
        ttb_winner = -1
        for j in cue_order:
            if diff[j] > 0:
                ttb_winner = 0
                break
            if diff[j] < 0:
                ttb_winner = 1
                break
        if ttb_winner < 0:
            # Full tie (trial 13): no TTB winner, uninformative for both theories.
            continue

        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        # Tally margin of the TTB-favored option (positive = tally agrees with TTB).
        if ttb_winner == 0:
            margin = a_wins - b_wins
        else:
            margin = b_wins - a_wins

        r = int(resp)
        match = 1.0 if r == ttb_winner else 0.0
        margins.append(margin)
        matches.append(match)

    if not margins:
        return 0.0

    x = np.asarray(margins, dtype=float)
    y = np.asarray(matches, dtype=float)

    # (1) LEVEL: P(choosing the TTB-favored option) on trials where the
    # tally does NOT favor it (tally majority against, or exact tally tie).
    low = x <= 0
    level = float(np.mean(y[low])) if np.any(low) else 0.0

    # (2) SLOPE: OLS slope of the match indicator on the tally margin,
    # across all trials with a TTB winner (margins -3, -2, -1, 0, +1).
    vx = float(np.var(x))
    if len(x) > 1 and vx > 0:
        slope = float(np.mean((x - x.mean()) * (y - y.mean())) / vx)
    else:
        slope = 0.0

    return level - slope
```

**Observed (real) value:** 0.3633 (var=0.0060)
**Predicted under pi_7:** 0.2898 (var=0.0117)
**Predicted under pi_9:** 0.3245 (var=0.0096)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]

**Metric**
```python
import numpy as np
import pandas as pd


def _to_matrix(col):
    # col: pandas Series whose cells are lists/tuples/arrays of binary ratings
    return np.array([list(r) for r in col], dtype=float)


def metric(data: pd.DataFrame) -> float:
    if data is None or len(data) == 0:
        return 0.0

    A = _to_matrix(data['option_a_ratings'])
    B = _to_matrix(data['option_b_ratings'])
    if A.ndim != 2 or A.shape != B.shape or A.shape[0] == 0:
        return 0.0

    # Tally margin: number of features where A beats B minus number where B beats A.
    margin = (A > B).sum(axis=1) - (B > A).sum(axis=1)

    # Signed choice: +1 if the subject chose A (response == 0), -1 if B (response == 1).
    resp = pd.to_numeric(data['response'], errors='coerce').to_numpy(dtype=float)
    s = 1.0 - 2.0 * resp

    # Alignment between the signed choice and the tally margin.
    # Tallying: E[s | margin] = (1-eps) * tanh(beta*margin/2)  =>  E[s*margin] > 0,
    # scaling like beta * E[margin^2] / 2 for small beta and saturating at mean(|margin|)
    # for large beta (max value 28/16 = 1.75 on this design).
    # TTB: E[s | trial] = (1-eps)*tanh(beta/2)*ttb_sign, and on this stimulus set
    # sum over the 16 trial types of ttb_sign * margin = -2 (mean -0.125), so the
    # TTB value is slightly NEGATIVE for any beta > 0.
    return float(np.mean(s * margin))
```

**Observed (real) value:** 0.0358 (var=0.0181)
**Predicted under pi_7:** 0.1029 (var=0.0537)
**Predicted under pi_9:** 0.1267 (var=0.0483)

### Experiment 5
**Design**
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    # Reversed-cascade follow rate (centered).
    #
    # For every trial we determine the winner implied by a one-reason
    # cascade that consults cues in ASCENDING true validity (i.e., the
    # 0.5-validity expert first, ..., the 0.9-validity expert last) --
    # exactly the subjective hierarchy implied by full misbinding
    # (kappa = 1). Full-tie trials carry no directional prediction and
    # are dropped. The metric is the signed follow rate:
    #   +1  -> subject always picks the reversed-cascade winner
    #    0  -> chance
    #   -1  -> subject always picks the opposite option
    #
    # Predicted values:
    #   Misbinding TTB: p(follow) = (1-eps)*sigmoid(beta) + eps/2 ~ 0.60
    #                   on EVERY discriminating trial  ->  metric ~ +0.21
    #   Tallying:       on the 7 trials where the tally winner OPPOSES
    #                   the lowest-validity cue (trials 1,2,3,5,6,7,15)
    #                   p(follow) < 0.5 for ANY beta > 0; on the 5
    #                   zero-margin trials (4,8,11,12,13) p(follow)
    #                   = 0.5 EXACTLY for any (beta, eps); only the 3
    #                   agreeing trials (9,10,14) push above 0.5.
    #                   Hence the Tallying metric is strictly NEGATIVE
    #                   for every parameterization in its ranges
    #                   (approaching 0 only as beta->0 or eps->0.5).
    validities = np.array([0.90, 0.80, 0.70, 0.60, 0.50])
    # Ascending true validity == descending subjective validity under
    # full misbinding: consult index 4 first, then 3, 2, 1, 0.
    cue_order = np.argsort(validities, kind="stable")

    n_used = 0
    follow = 0
    for a_raw, b_raw, r in zip(data["option_a_ratings"],
                               data["option_b_ratings"],
                               data["response"]):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        if a.shape != b.shape or a.size != validities.size:
            continue
        winner = None
        for j in cue_order:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
        if winner is None:
            # Full tie (trial 16): no directional prediction.
            continue
        n_used += 1
        if int(r) == winner:
            follow += 1

    if n_used == 0:
        return 0.0
    return 2.0 * (follow / n_used) - 1.0
```

**Observed (real) value:** 0.4533 (var=0.0207)
**Predicted under pi_7:** 0.4400 (var=0.0168)
**Predicted under pi_9:** 0.3987 (var=0.0292)

### Experiment 6
**Design**
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 1, 1]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Contrast metric built from two parameter-free anchors:
    #
    # (1) Margin-0 cells with a reversed-cascade decider (T5, T6, T11, T16).
    #     Tallying predicts p(A) = 0.5 EXACTLY on any tie trial, for any
    #     (beta, epsilon), so the signed follow-the-decider score has
    #     expectation exactly 0. The reversed-hierarchy model follows its
    #     decider with q = (1-eps)*sigmoid(beta) + eps/2 in [0.57, 0.64],
    #     so its expected score is 2q-1 in [0.14, 0.28].
    #
    # (2) Disagreement cells (T1, T2, T10, T14): nonzero tally margin whose
    #     sign OPPOSES the reversed decider. Tallying follows the margin
    #     with p >= 0.5 for any beta > 0 (expected signed score 2p-1 >= 0);
    #     the reversed model follows the decider, i.e. follows the tally
    #     with 1-q <= 0.43 (expected signed score 1-2q < 0).
    #
    # metric = mean(follow-decider score on margin-0 cells)
    #        - mean(follow-tally score on disagreement cells)
    #
    # Tallying: 0 - (2p-1) <= 0 (typically around -0.4 to -0.9).
    # Reversed: (2q-1) - (1-2q) = 4q-2 in [0.28, 0.56] > 0.
    # The two theories land on opposite sides of zero with a large gap.
    s1 = []
    s2 = []
    n_rows = len(data)
    for i in range(n_rows):
        row = data.iloc[i]
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        n = len(a)

        # Tally margin: strict feature-wise wins for A minus for B.
        margin = 0
        for j in range(n):
            if a[j] > b[j]:
                margin += 1
            elif b[j] > a[j]:
                margin -= 1

        # Reversed-hierarchy decider: first strictly discriminating
        # feature scanning from the LAST index down (ascending true
        # validity, since validities descend by index).
        dec = 0
        for j in range(n - 1, -1, -1):
            if a[j] > b[j]:
                dec = 1
                break
            if b[j] > a[j]:
                dec = -1
                break
        if dec == 0:
            # Identical options: no decider, no information.
            continue

        resp = int(row["response"])
        chose_a = (resp == 0)

        if margin == 0:
            # Knife-edge cell: tally is exactly 0.5 under Tallying.
            follow_dec = (dec == 1 and chose_a) or (dec == -1 and not chose_a)
            s1.append(1.0 if follow_dec else -1.0)
        elif (margin > 0 and dec == -1) or (margin < 0 and dec == 1):
            # Disagreement cell: tally direction opposes the decider.
            follow_tally = (margin > 0 and chose_a) or (margin < 0 and not chose_a)
            s2.append(1.0 if follow_tally else -1.0)

    m1 = float(np.mean(s1)) if len(s1) > 0 else 0.0
    m2 = float(np.mean(s2)) if len(s2) > 0 else 0.0
    return m1 - m2
```

**Observed (real) value:** 0.7787 (var=0.0600)
**Predicted under pi_7:** 0.9093 (var=0.1091)
**Predicted under pi_9:** 0.6120 (var=0.1153)

### Experiment 7
**Design**
  A=[0, 0, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Decider-follow rate on the margin-0 DISCRIMINATING trials (the
    # "subjective-hierarchy ladder": T4, T7-T11 in the design).
    # On these trials the tally is exactly balanced, so Tallying is
    # pinned to p = 0.50 toward either side for ANY (beta, epsilon)
    # (softmax over equal tallies is uniform, lapse is uniform).
    # Misbinding TTB instead follows its (reversed-hierarchy) first
    # discriminating cue with p = (1-eps)*sigmoid(beta) + eps/2,
    # which is >= 0.63 over the stated parameter ranges. The metric
    # is the proportion of choices that agree with the misbound
    # decider on exactly those trials: ~0.71 under misbinding TTB,
    # ~0.50 under Tallying.
    val = np.array([0.7, 0.9, 0.6, 0.95, 0.75, 0.6], dtype=float)
    # kappa = 1 misbinding: subjective weights are the validity list
    # bound in reverse to feature positions.
    w = val[::-1]
    cue_order = np.argsort(-w, kind="stable")  # f2, f4, f1, f5, f3, f0

    a = np.array([list(x) for x in data['option_a_ratings']], dtype=float)
    b = np.array([list(x) for x in data['option_b_ratings']], dtype=float)
    resp = np.asarray(data['response'], dtype=int).ravel()

    if a.shape[0] == 0:
        return 0.5

    a_wins = (a > b).sum(axis=1)
    b_wins = (b > a).sum(axis=1)

    # Margin-0 trials that still have at least one discriminating cue
    # (excludes the full tie T16, where both theories say 0.5).
    mask = (a_wins == b_wins) & ((a_wins + b_wins) > 0)
    idx = np.where(mask)[0]

    if len(idx) == 0:
        return 0.5

    follow = []
    for i in idx:
        decider = None
        for j in cue_order:
            if a[i, j] > b[i, j]:
                decider = 0
                break
            if b[i, j] > a[i, j]:
                decider = 1
                break
        if decider is None:
            continue  # shouldn't happen on discriminating trials
        follow.append(1 if resp[i] == decider else 0)

    if len(follow) == 0:
        return 0.5
    return float(np.mean(follow))
```

**Observed (real) value:** 0.6133 (var=0.0052)
**Predicted under pi_7:** 0.7217 (var=0.0070)
**Predicted under pi_9:** 0.6933 (var=0.0091)

### Experiment 8
**Design**
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Knife-edge signed contrast on the four margin-0 discriminating cells.

    Tallying predicts p(A) = 0.50 EXACTLY on every margin-0 cell, for any
    (beta, epsilon): equal tallies give an exactly uniform softmax. The
    reversed-hierarchy one-reason model, in contrast, lets a single cue
    decide these same cells, producing a systematic ~0.70/0.30 split whose
    sign is set by which option wins on the cascade's first cue.

    Metric = mean p(choose A | T11, T13) - mean p(choose A | T8, T15),
    where T11/T13 are the margin-0 cells whose cascade-first cue favors A
    and T8/T15 are the margin-0 cells whose cascade-first cue favors B
    (under the design's validities [0.95, 0.85, 0.75, 0.65, 0.6, 0.6]
    reversed -> f5 consulted first).

    Tallying: expectation exactly 0 for ANY parameter values.
    Misbinding TTB: expectation ~ +/-(2p-1) in [0.33, 0.49].
    """
    def to_key(v):
        return tuple([int(x) for x in v])

    a_keys = data["option_a_ratings"].apply(to_key)
    b_keys = data["option_b_ratings"].apply(to_key)
    chose_a = 1.0 - data["response"].astype(float)  # response 0 = A

    # Margin-0 (knife-edge) cells, identified by their rating vectors.
    # Cascade-first cue (f5) favors A here:
    hi_cells = [
        ((0, 0, 0, 0, 0, 1), (1, 0, 0, 0, 0, 0)),  # T11: 1-1 split, f5->A
        ((0, 0, 0, 0, 1, 1), (1, 1, 0, 0, 0, 0)),  # T13: 2-2 split, f5->A
    ]
    # Cascade-first cue (f5) favors B here:
    lo_cells = [
        ((1, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1)),  # T8:  1-1 split, f5->B
        ((1, 1, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1)),  # T15: 3-3 split, f5->B
    ]

    def cell_pa(ak, bk):
        mask = (a_keys == ak) & (b_keys == bk)
        if int(mask.sum()) == 0:
            return np.nan
        return float(chose_a[mask].mean())

    hi = [cell_pa(ak, bk) for ak, bk in hi_cells]
    lo = [cell_pa(ak, bk) for ak, bk in lo_cells]
    hi = [v for v in hi if not np.isnan(v)]
    lo = [v for v in lo if not np.isnan(v)]
    if not hi or not lo:
        return 0.0

    return float(np.mean(hi) - np.mean(lo))
```

**Observed (real) value:** 0.3400 (var=0.0288)
**Predicted under pi_7:** 0.4450 (var=0.0288)
**Predicted under pi_9:** 0.4150 (var=0.0451)

### Experiment 9
**Design**
  A=[0, 0, 0, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Knife-edge decider-direction contrast at balanced tally margins.
    #
    # For every trial we compute (a) the tally margin m = (#features A wins)
    # - (#features B wins), and (b) the winner implied by the advocated
    # misbinding-TTB cascade (subjective hierarchy = instructed validities
    # reversed end-to-end: cue order f4 -> f0 -> f2 -> f1 -> f3 -> f5).
    #
    # We keep only trials with |m| in {0, 3} (excluding the |m|=1 cells,
    # whose decider directions are unbalanced across margins) and where the
    # cascade has a discriminating cue. In this subset the misbinding-A
    # cells are {T1 (m=-3), T3 (m=0), T5 (m=+3), T9 (m=0), T10 (m=0)} and
    # the misbinding-B cells are {T6 (m=+3), T7 (m=0), T8 (m=-3), T11 (m=0),
    # T12 (m=0)} -- perfectly mirror-balanced in margin.
    #
    # Metric = P(choose A | cascade says A) - P(choose A | cascade says B).
    #
    # Under Tallying, p(A) = (1-eps)*sigmoid(beta*m) + eps/2, which is
    # exactly symmetric in m. Within each group the margins are balanced
    # (-3/+3 pairs and 0s), so BOTH group means are exactly 0.5 for ANY
    # (beta, epsilon) -- the metric is pinned to 0 parameter-free.
    #
    # Under misbinding TTB, p(A) = p_follow on cascade-A cells and
    # 1 - p_follow on cascade-B cells, so the metric equals 2*p_follow - 1,
    # which over the advocated parameter ranges lies in ~[0.27, 0.44]
    # (mean ~0.36) regardless of margin.

    df = data.reset_index(drop=True)
    a_lists = df['option_a_ratings'].tolist()
    b_lists = df['option_b_ratings'].tolist()
    resp = df['response'].to_numpy()

    # Design validities and the kappa=1 reversed subjective hierarchy.
    val = np.array([0.6, 0.95, 0.6, 0.85, 0.65, 0.9])
    w = val[::-1]
    cue_order = np.argsort(-w, kind='stable').tolist()  # -> [4, 0, 2, 1, 3, 5]

    a_side = []  # trials where the misbound cascade favors A
    b_side = []  # trials where the misbound cascade favors B

    for i in range(len(df)):
        av = [int(v) for v in a_lists[i]]
        bv = [int(v) for v in b_lists[i]]
        n = len(av)

        margin = (sum(1 for j in range(n) if av[j] > bv[j])
                  - sum(1 for j in range(n) if bv[j] > av[j]))
        if abs(margin) == 1:
            # Unbalanced decider directions at |m|=1; excluded so the
            # Tallying expectation stays exactly 0.
            continue

        winner = -1
        for j in cue_order:
            if j >= n:
                continue
            if av[j] > bv[j]:
                winner = 0
                break
            if bv[j] > av[j]:
                winner = 1
                break
        if winner == -1:
            continue  # full-tie trial (T16): no discriminating cue

        chose_a = 1.0 if resp[i] == 0 else 0.0
        if winner == 0:
            a_side.append(chose_a)
        else:
            b_side.append(chose_a)

    pa = float(np.mean(a_side)) if len(a_side) > 0 else 0.5
    pb = float(np.mean(b_side)) if len(b_side) > 0 else 0.5
    return float(pa - pb)
```

**Observed (real) value:** -0.3107 (var=0.0225)
**Predicted under pi_7:** -0.1520 (var=0.0136)
**Predicted under pi_9:** -0.1307 (var=0.0147)

### Experiment 10
**Design**
  A=[1, 0, 1, 1, 1, 1, 0]  B=[0, 0, 1, 0, 1, 0, 0]
  A=[1, 0, 1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 0, 1]  B=[0, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 1, 1, 0]  B=[1, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0]

**Metric**
```python
import numpy as np
import pandas as pd

# Design-time constants for this experiment.
VALIDITIES = np.array([0.6, 0.9, 0.8, 0.95, 0.65, 0.75, 0.6])


def metric(data: pd.DataFrame) -> float:
    """Cue-following deviation from chance on tally-tied (margin-0) trials.

    For every trial whose tally margin is zero (A and B win on equal numbers
    of features), find the first strictly discriminating cue under the
    reversed-hierarchy cue order (the competing theory's subjective
    hierarchy). Record whether the subject's choice followed that cue.
    Tallying predicts exactly chance (0 deviation) on every tied trial,
    for ANY (beta, epsilon). One-reason misbinding predicts a systematic
    follow rate of p = (1-eps)*sigmoid(beta) + eps/2, i.e. a deviation of
    roughly +0.12 to +0.25 above chance.
    """
    # Reversed-hierarchy subjective weights and cue order (stable ties).
    w = VALIDITIES[::-1].astype(float)
    cue_order = np.argsort(-w, kind="stable")

    a_col = data["option_a_ratings"].tolist()
    b_col = data["option_b_ratings"].tolist()
    resp_col = data["response"].tolist()

    follows = []
    for a_raw, b_raw, r_raw in zip(a_col, b_col, resp_col):
        a = np.asarray(a_raw, dtype=float)
        b = np.asarray(b_raw, dtype=float)
        if a.shape != b.shape or a.ndim != 1:
            continue

        # Tally margin: number of features A wins minus number B wins.
        margin = int(np.sum(a > b) - np.sum(b > a))
        if margin != 0:
            # Only tied tallies carry the parameter-free signature.
            continue

        # First discriminating cue under the reversed hierarchy.
        decider = None
        for j in cue_order:
            if a[j] > b[j]:
                decider = 0  # cue favors A
                break
            if b[j] > a[j]:
                decider = 1  # cue favors B
                break
        if decider is None:
            # Full tie (T16): both theories say 0.5; no information.
            continue

        resp = int(r_raw)
        follows.append(1.0 if resp == decider else 0.0)

    if len(follows) == 0:
        return 0.0

    # Deviation of the tied-trial cue-follow rate from chance.
    return float(np.mean(follows)) - 0.5
```

**Observed (real) value:** -0.2889 (var=0.0205)
**Predicted under pi_7:** -0.2122 (var=0.0106)
**Predicted under pi_9:** -0.2433 (var=0.0146)

### Experiment 11
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Dissociation metric: fraction of choices matching the advocated
    # (Reversed-Hierarchy Frugality) cascade winner, computed ONLY on
    # trials where the two theories' cue hierarchies yield OPPOSITE
    # predictions. On those trials the advocated theory predicts a
    # match rate of p = (1-eps)*sigmoid(beta) + eps/2 ~ 0.62-0.80,
    # while the competing theory predicts the complementary rate
    # ~ 1-p ~ 0.20-0.40, so the two theories produce visibly
    # different numbers on this scalar.
    val = np.array([0.9, 0.8, 0.95, 0.7, 0.6, 0.65])
    n = len(val)

    # Advocated hierarchy: subjective weights = reversed validity list.
    w_adv = val[::-1]  # [0.65, 0.6, 0.7, 0.95, 0.8, 0.9]
    order_adv = np.argsort(-w_adv, kind="stable")  # f3, f5, f4, f2, f0, f1

    # Competing hierarchy: ascending advertised validity.
    order_comp = np.lexsort((np.arange(n), val))  # f4, f5, f3, f1, f0, f2

    def cascade_winner(a, b, order):
        for j in order:
            if a[j] > b[j]:
                return 0
            if b[j] > a[j]:
                return 1
        return -1  # no discriminating cue

    matches = []
    for ra, rb, resp in zip(
        data["option_a_ratings"], data["option_b_ratings"], data["response"]
    ):
        a = [int(v) for v in ra]
        b = [int(v) for v in rb]
        r = int(resp)

        wa = cascade_winner(a, b, order_adv)
        wc = cascade_winner(a, b, order_comp)

        # Keep only trials that genuinely dissociate the two theories:
        # both cascades terminate and they pick different options.
        if wa < 0 or wc < 0 or wa == wc:
            continue

        matches.append(1.0 if r == wa else 0.0)

    if len(matches) == 0:
        return 0.5
    return float(np.mean(matches))
```

**Observed (real) value:** 0.4280 (var=0.0052)
**Predicted under pi_7:** 0.3867 (var=0.0066)
**Predicted under pi_9:** 0.3993 (var=0.0081)

### Experiment 12
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 1]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # ---------- design constants ----------
    NF = 6
    VAL = [0.9, 0.65, 0.6, 0.8, 0.95, 0.7]
    # Weakest-Expert-First cascade: cues in ASCENDING validity
    adv_order = sorted(range(NF), key=lambda j: VAL[j])      # [2,1,5,3,0,4]
    # Reversed-Hierarchy (misbinding) cascade: DESCENDING val[::-1]
    rev = VAL[::-1]
    comp_order = sorted(range(NF), key=lambda j: -rev[j])    # [1,5,2,0,4,3]

    # ---------- robust cell parsers ----------
    def parse_vec(x):
        try:
            if isinstance(x, np.ndarray):
                v = np.asarray(x, dtype=int).ravel()
                return v if v.size == NF else None
            if isinstance(x, (list, tuple)):
                v = np.asarray(list(x), dtype=float).ravel()
                if v.size != NF:
                    return None
                return np.rint(v).astype(int)
            s = str(x).strip()
            if s.startswith("[") and s.endswith("]"):
                s = s[1:-1]
            elif s.startswith("(") and s.endswith(")"):
                s = s[1:-1]
            toks = s.replace(",", " ").replace(";", " ").split()
            if len(toks) == 1:
                t0 = toks[0].strip("'\"")
                if len(t0) == NF and all(ch in "01" for ch in t0):
                    return np.asarray([int(ch) for ch in t0], dtype=int)
                return None
            vals = []
            for t in toks:
                vals.append(int(round(float(t.strip("'\"")))))
            v = np.asarray(vals, dtype=int)
            return v if v.size == NF else None
        except Exception:
            return None

    def parse_resp(x):
        try:
            if isinstance(x, (bool, np.bool_)):
                return int(x)
            if isinstance(x, (int, np.integer)):
                r = int(x)
                return r if r in (0, 1) else None
            if isinstance(x, (float, np.floating)):
                if np.isnan(x):
                    return None
                r = int(round(float(x)))
                return r if r in (0, 1) else None
            s = str(x).strip().upper()
            if s == "A":
                return 0
            if s == "B":
                return 1
            r = int(round(float(s)))
            return r if r in (0, 1) else None
        except Exception:
            return None

    # ---------- column resolution ----------
    cols = list(data.columns)
    lmap = {}
    for c in cols:
        lmap[str(c).lower()] = c

    def resolve(names, keys):
        for nm in names:
            if nm in lmap:
                return lmap[nm]
        for c in cols:
            lc = str(c).lower()
            if any(k in lc for k in keys):
                return c
        return None

    a_col = resolve(["option_a_ratings", "optiona_ratings", "ratings_a",
                     "a_ratings", "option_a"], ["a_rat", "option_a", "ratings_a"])
    b_col = resolve(["option_b_ratings", "optionb_ratings", "ratings_b",
                     "b_ratings", "option_b"], ["b_rat", "option_b", "ratings_b"])
    r_col = resolve(["response", "choice", "chosen", "answer", "pick",
                     "selected"], ["respon", "choic", "chose", "select", "answer", "pick"])
    s_col = resolve(["subject_id", "subjectid", "subject", "subj",
                     "participant"], ["subject", "subj", "participant"])
    if a_col is None or b_col is None or r_col is None:
        return 0.0

    def cascade(a, b, order):
        for j in order:
            if a[j] > b[j]:
                return 0
            if b[j] > a[j]:
                return 1
        return None

    # ---------- per-subject cascade-follow rates ----------
    n_rows = int(len(data))
    if s_col is not None:
        try:
            subjects = [str(v) for v in list(data[s_col])]
        except Exception:
            subjects = ["s%d" % i for i in range(n_rows)]
    else:
        subjects = ["s%d" % i for i in range(n_rows)]

    a_cells = list(data[a_col])
    b_cells = list(data[b_col])
    r_cells = list(data[r_col])

    stats = {}
    for i in range(n_rows):
        a = parse_vec(a_cells[i])
        b = parse_vec(b_cells[i])
        resp = parse_resp(r_cells[i])
        if a is None or b is None or resp is None:
            continue
        wa = cascade(a, b, adv_order)
        if wa is None:
            continue  # complete tie: both theories guess 50/50
        wc = cascade(a, b, comp_order)
        if wc != wa:
            continue  # keep only trials where both hierarchies dictate the
                       # same option, so the follow rate is theory-neutral
        rec = stats.get(subjects[i])
        if rec is None:
            rec = [0, 0]
            stats[subjects[i]] = rec
        rec[1] += 1
        if resp == wa:
            rec[0] += 1

    rates = []
    for k in sorted(stats.keys()):
        f, t = stats[k]
        if t > 0:
            rates.append(f / float(t))
    if not rates:
        return 0.0

    x = np.asarray(rates, dtype=float)
    n = int(x.size)
    m = float(x.mean())
    if n > 1:
        dm = np.abs(x[:, None] - x[None, :])
        g = float(dm.sum() / (n * (n - 1.0)))  # Gini mean difference (ordered pairs)
    else:
        g = 0.0

    # ---------- population-level pairwise conformity-dispersion index ----------
    # Every ordered subject pair (i, j) contributes the anchor excess of its
    # first member, (x_i - 0.5), plus half the pair's follow-rate distance,
    # 0.5 * |x_i - x_j|.  Summed over all n^2 ordered pairs this equals
    #   n^2 * (mean(x) - 0.5) + 0.5 * n(n-1) * GiniMD(x).
    # For a single-subject slice it collapses to that subject's own excess
    # conformity (x - 0.5): the one diagonal pair contributes no distance.
    index = (float(n) ** 2) * (m - 0.5) + 0.5 * float(n) * (float(n) - 1.0) * g
    return float(index)
```

**Observed (real) value:** 116.5556 (var=0.0041)
**Predicted under pi_7:** 119.2111 (var=0.0025)
**Predicted under pi_9:** 144.7000 (var=0.0030)

### Experiment 13
**Design**
  A=[1, 0, 1, 0, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1]  B=[0, 0, 1, 1, 0, 1]
  A=[1, 0, 0, 1, 1, 0]  B=[1, 1, 0, 1, 0, 0]
  A=[1, 1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1, 0]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 1, 1]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[1, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 1, 1, 0]  B=[1, 0, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """
    Anti-misbound decider-follow rate on hierarchy-dissociating cells.

    For each trial we compute the option picked by the competitor's
    hierarchy (weakest-advertised-expert-first: cues sorted by ASCENDING
    validity, ties by display position) and the option picked by the
    advocated theory's anti-misbound re-encoding (w = -val[::-1],
    consulted in descending subjective weight, which is equivalent to
    sorting cues by ASCENDING val[::-1], ties by display position), using
    the experiment's validities [0.80, 0.55, 0.90, 0.55, 0.80, 0.65].

    On this design the two hierarchies' first discriminating cues point to
    OPPOSITE options on 13 of the 16 unique pairs (T1-T13) and agree on
    T14/T15 (T16 is a full tie). The deciders are invariant to either
    theory's free tie_break parameter because no trial opposes two cues
    tied within a hierarchy group. The metric is the proportion of
    choices that follow the anti-misbound decider on those 13
    dissociating cells only.

    Predicted values:
      - Weakest-Expert-First (competing): subjects follow their own
        (central) decider with p_f ~ 0.70, hence follow the anti-misbound
        decider with ~0.30.
      - Advocated: ~95% of subjects are anti-misbinders (m_am =
    ambiguity x coherence-gate ~ 0.95 for this arrangement), so the
        follow rate toward the anti-misbound decider is ~0.70.
    The two theories therefore sit on opposite sides of 0.50 with a
    ~0.40 gap, while per-subject values are tight (each subject
    contributes 13 cells x 6 reps = 78 dissociating trials).
    """
    VAL = [0.80, 0.55, 0.90, 0.55, 0.80, 0.65]
    n = len(VAL)

    # Competitor hierarchy: ascending advertised validity, position ties.
    comp_order = sorted(range(n), key=lambda j: (VAL[j], j))

    # Anti-misbound hierarchy: w = -val[::-1] consulted in DESCENDING
    # subjective weight == ascending val[::-1], position ties.
    rev = VAL[::-1]
    mis_order = sorted(range(n), key=lambda j: (rev[j], j))

    def decider(a, b, order):
        for j in order:
            if a[j] > b[j]:
                return 0
            if b[j] > a[j]:
                return 1
        return None

    a_col = data["option_a_ratings"].tolist()
    b_col = data["option_b_ratings"].tolist()
    r_col = data["response"].tolist()

    hits = []
    for a_raw, b_raw, r in zip(a_col, b_col, r_col):
        try:
            a = [int(v) for v in a_raw]
            b = [int(v) for v in b_raw]
        except Exception:
            continue
        if len(a) != n or len(b) != n:
            continue
        cd = decider(a, b, comp_order)
        md = decider(a, b, mis_order)
        # Keep only cells where the two hierarchies genuinely dissociate.
        if cd is None or md is None or cd == md:
            continue
        hits.append(1.0 if int(r) == int(md) else 0.0)

    if not hits:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.4897 (var=0.0022)
**Predicted under pi_7:** 0.5056 (var=0.0039)
**Predicted under pi_9:** 0.5015 (var=0.0033)

### Experiment 14
**Design**
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 1]
  A=[1, 1, 1, 1, 0, 1]  B=[1, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data):
    import numpy as np

    # Reference hierarchy: the WEAKEST-EXPERT-FIRST (ascending-validity)
    # cascade for the design validity vector [0.75, 0.9, 0.65, 0.8, 0.85, 0.6]:
    # consult order f5 (0.60) -> f2 (0.65) -> f0 (0.75) -> f3 (0.80)
    # -> f4 (0.85) -> f1 (0.90). On every diagnostic trial below, the first
    # cue in this order that discriminates A from B defines the reference
    # side, and the metric measures how often the subject's choice lands
    # on that side.
    val = [0.75, 0.9, 0.65, 0.8, 0.85, 0.6]
    order = sorted(range(6), key=lambda j: val[j])

    # Diagnostic multi-cue trials: T3, T4, T7, T8, T9, T10, T11
    # (7 unique pairs x 6 repetitions = 42 trials per subject).
    # These are the cells on which the two theories' hierarchies diverge:
    # the reference decider is always the low-validity expert (f2 at 0.65
    # or f3 at 0.80), while the competing theory's anti-misbound component
    # (and its per-cue zeta distortion of either component) lands its
    # decider on a different, higher-validity expert for a large fraction
    # of subjects.
    diag = set()
    diag.add(((1, 1, 0, 1, 1, 1), (1, 1, 1, 0, 1, 1)))  # T3
    diag.add(((1, 1, 1, 0, 1, 1), (1, 1, 0, 1, 1, 1)))  # T4 (mirror of T3)
    diag.add(((1, 0, 0, 0, 0, 0), (0, 1, 1, 0, 0, 0)))  # T7
    diag.add(((0, 1, 1, 0, 0, 0), (1, 0, 0, 0, 0, 0)))  # T8 (mirror of T7)
    diag.add(((1, 1, 0, 0, 1, 1), (1, 1, 1, 1, 0, 1)))  # T9
    diag.add(((1, 1, 1, 1, 0, 1), (1, 1, 0, 0, 1, 1)))  # T10 (mirror of T9)
    diag.add(((0, 1, 0, 0, 0, 0), (1, 0, 1, 0, 1, 0)))  # T11

    a_col = data["option_a_ratings"].tolist()
    b_col = data["option_b_ratings"].tolist()
    r_col = data["response"].tolist()

    hits = []
    for a_raw, b_raw, r_raw in zip(a_col, b_col, r_col):
        try:
            a = tuple([int(v) for v in a_raw])
            b = tuple([int(v) for v in b_raw])
            r = int(r_raw)
        except Exception:
            continue
        if len(a) != 6 or len(b) != 6:
            continue
        if (a, b) not in diag:
            continue
        # Reference decider: first strictly discriminating cue in the
        # ascending-validity (weakest-expert-first) consult order.
        side = None
        for j in order:
            if a[j] != b[j]:
                side = 0 if a[j] > b[j] else 1
                break
        if side is None:
            continue
        hits.append(1.0 if r == side else 0.0)

    if len(hits) == 0:
        return 0.0
    return float(np.mean(hits))
```

**Observed (real) value:** 0.5324 (var=0.0041)
**Predicted under pi_7:** 0.5586 (var=0.0062)
**Predicted under pi_9:** 0.5862 (var=0.0061)

### Experiment 15
**Design**
  A=[0, 1, 0, 1, 0, 0]  B=[1, 1, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 0, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0, 1]
  A=[1, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[1, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Follow-the-anti-misbound-decider rate on the four trial types where the
    # anti-misbound hierarchy (subjective weights w = -validities[::-1], i.e.
    # consult {f1,f3} -> f5 -> f2 -> {f0,f4}) is pitted head-to-head against
    # the position-primacy hierarchy, and where the competitor's anti-validity
    # and TTB components split away from its primacy mass: T1, T2, T9, T10
    # (24 of the 96 trials per subject).
    #
    # Deciders are recomputed inside the metric with EXACTLY the same
    # conventions as both theories' predict functions (w = -val[::-1],
    # np.lexsort((secondary, -w))), and a trial is included only if the
    # decider is identical under BOTH tie-break conventions (verified to
    # hold on all four types: the f1/f3 and f0/f4 weight ties never
    # internally oppose here), so the classification is exact and
    # tie-break independent. Zeta distortion is OFF in both theories for
    # this design (validity ties present), so per-subject deciders are
    # deterministic given the mixture component.
    #
    # Predicted values:
    #   ADVOCATED (m_am = 0.912 * sigmoid((0.727-0.45)/0.08) = 0.884):
    #     88.4% anti-misbound subjects follow this decider at p_f ~ 0.73;
    #     11.6% anti-validity subjects oppose it on T1/T2 and follow it on
    #     T9/T10 -> 0.50. Population ~ 0.70, per-subject SD ~ 0.13.
    #   COMPETING (primacy 0.573, anti-val 0.353, recency 0.043, TTB 0.031):
    #     primacy subjects OPPOSE the misbound decider on all four types
    #     (their f0 decider points the other way) -> 1-p_f ~ 0.27;
    #     anti-validity and TTB subjects sit at 0.50 (oppose on T1/T2 via
    #     f5, follow on T9/T10 via f1); recency -> 0.27 (f5/f4 point
    #     against on all four). Population ~ 0.36, per-subject SD ~ 0.15.
    # Gap ~ 0.34 with tight per-subject spreads on both sides: Welch t at
    # N=25 ~ 8-9, far beyond alpha = 0.01. Values near 0.7 support the
    # anti-misbound mixture; values near 0.35 support the position mixture.
    n_features = 6
    val = np.array([0.65, 0.90, 0.50, 0.80, 0.50, 0.90], dtype=float)

    # Anti-misbound subjective weights (identical to the advocated theory's
    # w_base = -val[::-1]).
    w_misbound = -val[::-1]

    def decider(a, b, w, later_first=False):
        # Consult cues in DESCENDING subjective weight; same lexsort
        # convention as both predict functions.
        n = len(w)
        pos = np.arange(n)
        secondary = (n - 1 - pos) if later_first else pos
        order = np.lexsort((secondary, -w))
        for j in order:
            if a[j] > b[j]:
                return 0
            if b[j] > a[j]:
                return 1
        return None

    # The four head-to-head trial types: T1, T2, T9, T10.
    trials = [
        ([0, 1, 0, 1, 0, 0], [1, 1, 1, 0, 1, 1]),  # T1
        ([1, 1, 1, 0, 1, 1], [0, 1, 0, 1, 0, 0]),  # T2 (mirror of T1)
        ([0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 1]),  # T9
        ([1, 0, 1, 0, 1, 1], [0, 1, 0, 1, 0, 1]),  # T10 (mirror of T9)
    ]

    target = {}
    for a_list, b_list in trials:
        a = np.asarray(a_list, dtype=int)
        b = np.asarray(b_list, dtype=int)
        d_early = decider(a, b, w_misbound, later_first=False)
        d_late = decider(a, b, w_misbound, later_first=True)
        # Include only if the decider is tie-break independent.
        if d_early is None or d_late is None or d_early != d_late:
            continue
        key = (tuple([int(v) for v in a]), tuple([int(v) for v in b]))
        target[key] = int(d_early)

    if not target:
        return 0.5

    hits = []
    for ra, rb, resp in zip(data["option_a_ratings"],
                            data["option_b_ratings"],
                            data["response"]):
        a = tuple([int(v) for v in np.asarray(ra).ravel()])
        b = tuple([int(v) for v in np.asarray(rb).ravel()])
        key = (a, b)
        if key in target:
            hits.append(1.0 if int(resp) == target[key] else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.1500 (var=0.0143)
**Predicted under pi_7:** 0.2758 (var=0.0171)
**Predicted under pi_9:** 0.3150 (var=0.0121)

### Experiment 16
**Design**
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 1]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 1, 1]  B=[0, 0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 1]  B=[1, 1, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 0]  B=[1, 1, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 0, 1]  B=[0, 1, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    """
    Primacy-type excess vs validity-respecting-type excess.

    Two cell families are selected by their discriminating-cue sets
    (feature positions where the two options' ratings differ):

    Family 1 ("early-cue cells"; design trials T1-T4, T7-T8, T13, T15):
        sets {1,4}, {2,4}, {1,5}, {1,3}, {1,3,4}. The earliest-listed
        discriminating expert (f1 or f2) favors the opposite option from
        every later discriminating expert, so a position-primacy reader
        and an anti-validity (weakest-first) reader pick opposite options.

    Family 2 ("weak-cue cells"; design trials T5-T6, T9-T10):
        sets {3,4}, {3,5}. The earliest-listed discriminating expert is
        the low-validity f3, opposing f4/f5, so primacy and anti-validity
        readers disagree here as well, but a descending-validity (TTB-like
        or anti-misbound) reader now sides with the anti-validity reader.

    Per subject: X1 = rate of choosing the earliest-cue option on
    Family 1, X2 = same on Family 2. Score = +1 if the subject follows
    the earliest-listed expert on BOTH families (pure position-primacy
    consultation order), -1 if it follows it on Family 1 but reverses on
    Family 2 (validity-respecting order), 0 otherwise. The metric is the
    mean score: the excess of primacy-type subjects over
    validity-respecting-type subjects.
    """
    fam1 = {frozenset((1, 4)), frozenset((2, 4)), frozenset((1, 5)),
            frozenset((1, 3)), frozenset((1, 3, 4))}
    fam2 = {frozenset((3, 4)), frozenset((3, 5))}

    def _resp(v):
        if isinstance(v, str):
            return 0 if v.strip().upper() in ("A", "0") else 1
        try:
            return int(v)
        except (TypeError, ValueError):
            return -1

    scores = []
    for _, sdf in data.groupby("subject_id"):
        f1, f2 = [], []
        for _, row in sdf.iterrows():
            try:
                a = np.asarray(row["option_a_ratings"], dtype=int).ravel()
                b = np.asarray(row["option_b_ratings"], dtype=int).ravel()
            except (TypeError, ValueError):
                continue
            if a.size == 0 or a.size != b.size:
                continue
            diff = np.nonzero(a != b)[0]
            if diff.size == 0:
                continue
            dset = frozenset(int(j) for j in diff)
            in1 = dset in fam1
            in2 = dset in fam2
            if not (in1 or in2):
                continue
            j0 = int(diff[0])  # earliest-listed discriminating expert
            fav0 = 0 if a[j0] == 1 else 1  # option favored by that cue
            # earliest cue must oppose every later discriminating cue
            ok = True
            for j in diff[1:]:
                if (0 if a[int(j)] == 1 else 1) == fav0:
                    ok = False
                    break
            if not ok:
                continue
            r = _resp(row["response"])
            if r not in (0, 1):
                continue
            follow = 1.0 if r == fav0 else 0.0
            if in1:
                f1.append(follow)
            else:
                f2.append(follow)
        if not f1 or not f2:
            continue
        x1 = float(np.mean(f1))
        x2 = float(np.mean(f2))
        if x1 <= 0.5:
            s = 0.0
        elif x2 > 0.5:
            s = 1.0
        elif x2 < 0.5:
            s = -1.0
        else:
            s = 0.0
        scores.append(s)

    if not scores:
        return 0.0
    return float(np.mean(scores))
```

**Observed (real) value:** 0.0000 (var=0.0000)
**Predicted under pi_7:** 0.0000 (var=0.0000)
**Predicted under pi_9:** 0.0000 (var=0.0000)

### Experiment 17
**Design**
  A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[1, 0, 1, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 0, 1, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Count-vs-Primacy planned contrast for one-reason binary-feature choice.
    #
    # For every trial we compute, directly from the rating vectors:
    #   diff[j] = +1 if expert j rates A=1,B=0 (favors A), -1 if B=1,A=0,
    #   n      = number of discriminating experts,
    #   d      = sum(diff) = (#A-favoring cues) - (#B-favoring cues),
    #   e      = diff of the EARLIEST (lowest-index) discriminating cue (+1 -> A).
    #
    # The trial weight is  w = d/n - e , i.e. the count-based choice
    # direction minus the primacy-based choice direction, and the metric is
    # the mean of w * y with y = +1 for choosing A, -1 for choosing B.
    #
    # Under the advocated momentary-diffusion theory the first consulted
    # discriminating cue is ~uniform over the discriminators, so
    # E[y] ~ s * d/n  (s = 2*p_f - 1) and the metric concentrates near
    # +s * 0.26 ~ +0.11 (tight across subjects: only binomial noise plus a
    # small s-spread; full-tie trials carry no weight and the single-cue
    # control trial gets w = 0 automatically).
    #
    # Under the competing discrete position-mixture theory the dominant
    # primacy component (mass ~0.54-0.59 at this design's ambiguity, with
    # anti/recency/TTB minorities) makes E[y] ~ s * e on essentially every
    # cell, so the metric is strongly negative (population ~ -0.28), with
    # large between-subject spread from the fixed per-subject type draw.
    # The contrast is validity-vector robust: the earliest-discriminator and
    # count structures are fixed by the trial list, and the sign of the gap
    # holds for either advertised-validity arrangement.
    df = data
    vals = []
    a_col = df["option_a_ratings"]
    b_col = df["option_b_ratings"]
    r_col = df["response"]
    for a_raw, b_raw, resp in zip(a_col, b_col, r_col):
        a = np.asarray(a_raw, dtype=float)
        b = np.asarray(b_raw, dtype=float)
        if a.shape != b.shape or a.ndim != 1 or a.size == 0:
            continue
        diff = a - b
        disc = np.where(diff != 0)[0]
        n = disc.size
        if n == 0:
            # Full tie (no discriminating expert): no structural signal.
            continue
        d = float(diff.sum())            # count margin: A-cues minus B-cues
        e = float(diff[disc[0]])        # earliest discriminator: +1 -> A, -1 -> B
        w = d / float(n) - e            # count direction minus primacy direction
        y = 1.0 if int(resp) == 0 else -1.0   # response 0 = chose A
        vals.append(w * y)
    if len(vals) == 0:
        return 0.0
    return float(np.mean(vals))
```

**Observed (real) value:** 0.1541 (var=0.0151)
**Predicted under pi_7:** 0.0890 (var=0.0148)
**Predicted under pi_9:** 0.0783 (var=0.0144)

### Experiment 18
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Design-fixed advertised validities (positions f0..f5).
    VALIDITIES = np.array([0.80, 0.50, 0.85, 0.60, 0.90, 0.95])
    # A subject whose choice rate on primacy-vs-anti conflict trials exceeds
    # this threshold is classified as a position-primacy (earliest-listed-
    # expert) decider. 0.55 sits well above the anti-followers' ceiling
    # (~0.27-0.36 incl. the competing theory's momentary diffusion) and well
    # below the primacy-followers' level (~0.73).
    THRESHOLD = 0.55
    MIN_CONFLICT_TRIALS = 10

    def to_arr(x):
        if isinstance(x, str):
            s = x.strip().strip("[]()")
            return np.array([float(v) for v in s.replace(",", " ").split()])
        return np.asarray(x, dtype=float).ravel()

    df = data.reset_index(drop=True)
    n = len(df)
    if n == 0:
        return 0.0

    subj = df["subject_id"].values
    resp = np.asarray(df["response"].values, dtype=float)
    a_col = df["option_a_ratings"].values
    b_col = df["option_b_ratings"].values

    is_conflict = np.zeros(n, dtype=bool)
    prim_dir = np.zeros(n, dtype=int)

    for i in range(n):
        a = to_arr(a_col[i])
        b = to_arr(b_col[i])
        if a.size != b.size or a.size != VALIDITIES.size:
            continue
        disc = a != b
        if not disc.any():
            continue
        d_idx = np.where(disc)[0]
        # Primacy decider: earliest-listed discriminating expert.
        e = int(d_idx[0])
        # Anti-validity decider: weakest advertised discriminating expert.
        anti = int(d_idx[np.argmin(VALIDITIES[d_idx])])
        pdir = 0 if a[e] > b[e] else 1
        adir = 0 if a[anti] > b[anti] else 1
        if pdir != adir:
            is_conflict[i] = True
            prim_dir[i] = pdir

    subjects = pd.unique(df["subject_id"])
    flags = []
    for s in subjects:
        m = (subj == s) & is_conflict
        cnt = int(m.sum())
        if cnt < MIN_CONFLICT_TRIALS:
            flags.append(0.0)
            continue
        rate = float(np.mean(resp[m] == prim_dir[m]))
        flags.append(1.0 if rate > THRESHOLD else 0.0)

    if not flags:
        return 0.0
    return float(np.mean(flags))
```

**Observed (real) value:** 0.0000 (var=0.0000)
**Predicted under pi_7:** 0.0000 (var=0.0000)
**Predicted under pi_9:** 0.0000 (var=0.0000)

### Experiment 19
**Design**
  A=[0, 1, 1, 0, 0, 1]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0, 1]
  A=[1, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 1, 1]  B=[0, 1, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0, 1]  B=[0, 0, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # THE LEAK-DIP METRIC
    # --------------------
    # Core contrast: within-subject difference between the anti-validity
    # decider-follow rate on LEAK-ELIGIBLE trials (interior-validity
    # carrier opposed by the top-validity expert) and on CONTROL trials
    # (every other trial with a discriminating carrier).
    #
    # Fixed design: validities [0.5, 0.5, 0.75, 0.75, 1.0, 1.0], ascending
    # with display index, so the anti-validity consultation order is the
    # index order (ties are inert: tied cues never oppose in any trial).
    #
    # Advocated theory: follow probability is p_f on EVERY discriminating
    # trial regardless of carrier validity, position, margin, or coalition
    #   -> expected dip = EXACTLY 0.
    # Competing theory: on the 8 leak cells the trial-conditional
    # validity-respecting leak mixes q = min(rho_leak*0.956*0.25, 0.18)
    # (>= 0.120 forced in this design) toward the TTB winner, giving
    # follow = (1-q)*p_f + q*(1-p_f) there, while control cells stay at p_f
    #   -> expected dip = -q*(2*p_f - 1) in [-0.093, -0.050], bounded away
    #      from zero by the competing's own gate formulas.
    #
    # Works on a single subject's slice (48 leak + 42 control trials) and
    # on the pooled frame alike.

    VALIDITIES = [0.5, 0.5, 0.75, 0.75, 1.0, 1.0]
    n = len(VALIDITIES)
    vmin = min(VALIDITIES)
    vmax = max(VALIDITIES)

    def to_vec(x):
        if isinstance(x, str):
            return [int(ch) for ch in x if ch in "01"]
        return [int(v) for v in x]

    # anti-validity order: ascending validity (exact ties anchor; tied cues
    # never oppose in this design, so within-tie order is inert)
    order_anti = sorted(range(n), key=lambda j: VALIDITIES[j])
    # standard TTB order: descending validity
    order_ttb = sorted(range(n), key=lambda j: -VALIDITIES[j])

    leak_follows = []
    ctrl_follows = []

    a_col = data["option_a_ratings"].tolist()
    b_col = data["option_b_ratings"].tolist()
    r_col = data["response"].tolist()

    for a_raw, b_raw, resp in zip(a_col, b_col, r_col):
        a = to_vec(a_raw)
        b = to_vec(b_raw)
        if len(a) != n or len(b) != n:
            return 0.0

        # anti-validity decider: first strictly discriminating cue in
        # ascending-validity order
        carrier = None
        favored = None
        for j in order_anti:
            if a[j] != b[j]:
                carrier = j
                favored = 0 if a[j] > b[j] else 1
                break
        if carrier is None:
            # full-tie trial (T16): no decider, excluded
            continue

        # TTB decider: first strictly discriminating cue in
        # descending-validity order
        winner_ttb = None
        for j in order_ttb:
            if a[j] != b[j]:
                winner_ttb = 0 if a[j] > b[j] else 1
                break

        # leak-eligible cell: strictly INTERIOR carrier validity
        # (min + 0.10 < val < max - 0.10) AND the TTB winner exists and
        # OPPOSES the anti-validity winner. This is exactly the
        # competing theory's trial-conditional gate; the advocated theory
        # has no mechanism tied to this conjunction.
        interior = (vmin + 0.10) < VALIDITIES[carrier] < (vmax - 0.10)
        opposed = (winner_ttb is not None) and (winner_ttb != favored)

        follow = (int(resp) == favored)
        if interior and opposed:
            leak_follows.append(follow)
        else:
            ctrl_follows.append(follow)

    if not leak_follows or not ctrl_follows:
        return 0.0

    dip = float(np.mean(leak_follows)) - float(np.mean(ctrl_follows))
    return dip
```

**Observed (real) value:** -0.1955 (var=0.0143)
**Predicted under pi_7:** 0.0068 (var=0.0083)
**Predicted under pi_9:** -0.1893 (var=0.0201)

### Experiment 20
**Design**
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 1, 1]  B=[1, 1, 1, 1, 0, 1]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # TAIL HETEROGENEITY OF ANTI-VALIDITY DECIDER-FOLLOW RATES.
    #
    # Stage 1 (per subject): estimate the subject's decider-follow rate
    #   p_hat over all non-tie trials. The decider on each trial is the
    #   first discriminating expert in the subject's anti-validity
    #   consultation order: weakest advertised validity first; within the
    #   tied 0.5 block the order is positional, with the side
    #   (earlier-position-first vs later-position-first) inferred from the
    #   subject's own weak-block choices (reliable: the per-subject
    #   earlier-follow rate sits near 0.73 or 0.27 across the 60
    #   weak-block trials).
    # Stage 2 (population): the metric is the mean of the largest 20% of
    #   per-subject absolute deviations |p_hat - 0.72|, i.e. the TAIL
    #   dispersion of response discipline across subjects.
    #
    # The advocated theory's frozen parameter windows confine every
    #   subject's follow probability to [0.67, 0.78] (population SD
    #   ~0.015), so its deviation tail is set by the binomial sampling
    #   floor alone (~0.09). The competing theory samples beta over
    #   [0.5, 1.9] and epsilon over [0.08, 0.24], implying per-subject
    #   follow rates spanning [0.59, 0.84] with SD ~0.07, which lifts the
    #   tail to ~0.16.
    VALS = (0.95, 0.78, 0.5, 0.5, 0.5, 0.5)
    CENTER = 0.72
    TAIL_FRAC = 0.2

    def parse_ratings(x):
        if isinstance(x, str):
            x = x.strip().strip("[]").split(",")
        try:
            items = list(x)
        except TypeError:
            return None
        out = []
        for v in items:
            try:
                out.append(int(round(float(v))))
            except (TypeError, ValueError):
                return None
        return out if len(out) == 6 else None

    per_subject = {}
    for sid, a_raw, b_raw, r_raw in zip(
        data["subject_id"],
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"],
    ):
        a = parse_ratings(a_raw)
        b = parse_ratings(b_raw)
        if a is None or b is None:
            continue
        try:
            r = int(r_raw)
        except (TypeError, ValueError):
            continue
        per_subject.setdefault(sid, []).append((a, b, r))

    devs = []
    for trials in per_subject.values():
        # --- infer the subject's weak-block positional side ---
        early_follows = []
        for a, b, r in trials:
            disc = [j for j in range(6) if a[j] != b[j]]
            if len(disc) == 2 and disc[0] >= 2:
                j1 = disc[0]
                endorsed_early = 0 if a[j1] == 1 else 1
                early_follows.append(1 if r == endorsed_early else 0)
        early_side = (
            len(early_follows) > 0
            and 2 * sum(early_follows) > len(early_follows)
        )

        # --- decider-follow rate over all non-tie trials ---
        follows = []
        for a, b, r in trials:
            disc = [j for j in range(6) if a[j] != b[j]]
            if not disc:
                continue  # full tie: no decider, no information
            if early_side:
                j_star = min(disc, key=lambda j: (VALS[j], j))
            else:
                j_star = min(disc, key=lambda j: (VALS[j], -j))
            endorsed = 0 if a[j_star] == 1 else 1
            follows.append(1 if r == endorsed else 0)

        if not follows:
            continue
        p_hat = float(sum(follows)) / float(len(follows))
        devs.append(abs(p_hat - CENTER))

    if not devs:
        return 0.0
    devs = sorted(devs, reverse=True)
    k = max(1, int(np.ceil(TAIL_FRAC * len(devs))))
    return float(sum(devs[:k]) / float(k))
```

**Observed (real) value:** 0.1556 (var=0.0024)
**Predicted under pi_7:** 0.1120 (var=0.0016)
**Predicted under pi_9:** 0.1520 (var=0.0027)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the full set of 20 experiments the two theories split the evidence almost evenly, which by itself signals that neither is task-invariant. Scoring every experiment by |predicted - observed|: pi_7 is closer on Exps 2, 4, 5, 6, 9, 12, 14, 15, 17; pi_9 is closer on Exps 1, 3, 7, 8, 10, 11, 13, 19, 20; Exps 16 and 18 are exact ties (both predict 0, both correct). But the split is not symmetric in kind. The two theories share the anti-validity one-reason core, the A^3 diffuse heterogeneity, and the gamma tilt, so most of their joint successes (Exps 5, 12, 13, 16, 18) come from the shared machinery, not from anything that arbitrates between them. The arbitration hinges on pi_9's two unique mechanisms, and the evidence on those is decisive in one direction: (a) the interior-gated validity leak is REAL. Experiment 19 is the cleanest test \u2014 a forced leak-dip design where pi_9 predicts -0.189 and the observed value is -0.1955, while pi_7 is structurally forbidden to dip and predicts +0.0068. That is a direct falsification of pi_7's core claim (follow probability flat over carrier validity). Experiment 2 confirms the dip direction (observed -0.0507) though smaller than pi_9's -0.112, and Experiment 1's negative discriminant (-0.1996, far below pi_7's +0.075) again favors the wider/leakier pi_9 profile. (b) However, pi_9's leak is mis-calibrated: on Exp 2 it over-predicts the dip by 2x, on Exp 12 it over-predicts the conformity-dispersion index (144.7 vs observed 116.6, where pi_7's 119.2 is nearly exact), and on Exp 15 it over-predicts anti-misbound following (0.315 vs observed 0.150). (c) There are also failures common to BOTH theories that neither mechanism addresses: Exp 9 (observed -0.3107 vs -0.152/-0.131 \u2014 both badly under-predict the strength of anti-misbound opposition) and Exp 15 (both over-predict by 0.13-0.17), plus Exp 1 where both predictions sit far above the observed value. In short: pi_7 is structurally falsified (it cannot produce the leak dip that Exp 19 clearly shows, and it loses the discriminant in Exp 1), while pi_9 has the right qualitative mechanisms but wrong magnitudes and also misses Exp 9/15 alongside pi_7. Since the two theories partition the evidence rather than one dominating, and there are shared systematic misfits, the correct move per the arbitration rules is to propose a new theory \u2014 replacing pi_7, which is the degenerate one (its 'flat follow profile' claim is directly refuted) \u2014 that keeps the empirically validated anti-validity core and diffuse heterogeneity, incorporates a GRADED, design-modulated validity-respecting leak (strong enough for Exp 19's -0.19 dip, weak enough for Exp 2's -0.05 dip, and quiet on Exp 12/15's designs), and adds a mechanism for the strong anti-misbound opposition seen in Exps 9 and 15.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace pi_7 with 'Graded-Leak Anti-Validity Frugality with Validity-Extremity Weighting.' Core (retained, empirically supported by Exps 5, 12, 13, 16, 18): one-reason decision making, anti-validity (ascending-validity) consultation order, A^3-gated diffuse heterogeneity with stable (zeta) and momentary (xi) parts, exact-tie anchoring, softmax(beta)+lapse(epsilon) response discipline. NEW MECHANISM 1 \u2014 graded validity-respecting leak: replace pi_7's strict 'flat p_f over carrier validity' with a continuous mixture toward the TTB (descending-validity) decider whose weight q depends smoothly on (i) the carrier's validity extremity (a smooth ramp centered mid-range rather than pi_9's knife-edge strict-interior window \u2014 this captures Exp 19's large dip, Exp 2's moderate dip, and avoids pi_9's over-dipping on Exp 12/15), (ii) the range-normalized validity gap structure, and (iii) a leak-scale parameter sampled LOW enough that designs like Exp 12 and Exp 15 stay near the pure anti-validity anchor. NEW MECHANISM 2 \u2014 anti-misbound opposition: a minority component (population mass ~0.25-0.35) that consults cues by ascending reversed-position validity (w = -val[::-1]) on high-ambiguity all-distinct designs, which is what Exps 9 and 15 demand (both theories currently under-predict opposition to the misbound decider by 0.15+). Response discipline: widen the between-subject spread of p_f modestly (SD ~0.09, support [0.55, 0.86]) to match Exp 1's negative discriminant and Exp 20's tail (0.156) without the extreme dispersion that made pi_9 fail Exp 12. This new theory differs from pi_7 by adding the empirically forced leak and misbound-opposition mechanisms (without which Exps 1, 2, 9, 19 are unexplainable), and differs from pi_9 by making the leak graded and weaker (fixing pi_9's over-predictions on Exps 2, 12, 15) and by adding the misbound-opposition component (fixing the shared Exp 9/15 failures)."
}
```

## Usage

```json
{
  "input_tokens": 45094,
  "output_tokens": 3462,
  "reasoning_tokens": 2147,
  "provider": "Sail Research"
}
```
