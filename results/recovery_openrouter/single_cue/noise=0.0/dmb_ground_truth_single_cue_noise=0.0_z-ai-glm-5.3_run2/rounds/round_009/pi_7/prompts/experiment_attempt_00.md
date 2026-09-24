# experiment_attempt_00

## System Prompt

You are a renowned cognitive scientist designing an experiment in the Decision Making (Binary Features) domain.

Your goal is to be an adversarial collaborator: propose a design whose outcomes would be predicted by your advocated theory but NOT by the competing theory. Both are provided below.

A useful proposal targets a *quantitative* dissociation between the two theories — how they respond differently to specific stimuli in addition to differences in overall performance.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

A multi-attribute decision-making experiment. On each trial the subject sees two options (A, B), each described by `n_features` integer expert ratings (`n_features` is set by the length of `validities` you propose). Choose `validities` — one per feature, each in [0.5, 1.0], order free — to fix each expert's advertised accuracy; subjects are told these values up front. Then choose `trial_a_ratings/trial_b_ratings` (each rating value in [0, 1]) so that the intended decision-making theories make distinguishable predictions: avoid degenerate pairs where every theory agrees, and prefer pairs that maximally dissociate the candidate theories. Validities and the trial ratings together define the design; they are fixed across all trials in this experiment. No trial-by-trial correctness feedback. The total number of trials per subject is held at roughly 96: each unique pair is repeated K = max(1, 96 // n_unique_pairs) times in an independently-randomized order per subject.

Subjects see the following instructions:
In this experiment you will repeatedly choose between two fictitious products, A and B. On every trial you will see `n_features` expert ratings for each product (the number of experts is fixed across all trials and is set by the length of `validities`).

Each rating is an integer in [0, 1]. The ratings are displayed as a horizontal filled bar with the numeric value (e.g. "0/1") shown next to it. Higher = more positive.

The same experts (in the same order) provide ratings for both products on every trial. Each expert's accuracy (their validity expressed as a percentage, e.g. "Expert 1 (80%)") is shown next to their rating on every trial AND is also listed up front in an "Expert accuracies" panel.

On each trial, decide which product is of higher quality and press A for product A or B for product B. There is no time limit and no feedback. Note that for the first ~`min_rt_ms` of each trial the answer prompt is hidden and the keys are locked, so subjects first see the full ratings and can answer once the A / B prompt appears — design pairs that actually require comparing the ratings.

Total trials per subject is roughly `MAX_TRIALS`: each unique pair you propose is repeated `K = max(1, MAX_TRIALS // n_unique_pairs)` times in an independently-randomized order per subject.

## ADVOCATED THEORY
**Description:** Anti-Validity Frugality with Diffuse Positional-Salience Heterogeneity (momentary-diffusion variant). People are one-reason decision makers: cues are consulted one at a time in descending subjective weight and the first strictly discriminating cue decides; no integration across cues (flat psychometric profile over tally margins). The population-central hierarchy is anti-validity (distrust-the-weakest, w = -validity). Each subject's weights are a continuous perturbation, w_j = -val_j + gamma*salience_j + A^3*(sigma_h*zeta_j + kappa*xi_j), where (i) salience_j is a positional gradient favoring early-listed experts with gamma drawn asymmetrically from [-0.05, +0.12] — a graded tilt, never a discrete primacy rule, bounded so it can flip nothing on the anchored designs that pin the anti-validity core; (ii) binding ambiguity A = 1 - |Spearman(position, validity)| enters CUBED, a steeply convex, gate-free gradient; and (iii) the idiosyncratic distortion is split into a small stable part (zeta_j, fixed per subject) and a large MOMENTARY part (xi_j, re-drawn every trial): under high binding ambiguity the consultation order is not merely idiosyncratic but diffuse — attention fluctuates from trial to trial, which keeps populations tight around chance-level conformity instead of spreading subjects into fixed opposing deciders. All distortion applies only when all advertised validities are distinct (exact ties anchor the distrust order). Noise is softmax(beta) over the binary winner score plus an independent lapse epsilon (implied follow p_f mean ~0.71, SD ~0.08); no discriminating cue yields exactly 0.5; validity ties are broken by a free per-subject tie-break (~80% early position); history is ignored.

**Parameters:**
- beta: [0.5, 1.9]
- epsilon: [0.08, 0.24]
- gamma: [-0.05, 0.12]
- sigma_h: [0.6, 1.2]
- kappa: [8, 12]
- zeta: [(-1, 1)] * n_features
- tie_break: [0, 1]
- validities: validities

**`predict source code`:**
```python
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
```

**`policy source code`:**
```python
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Diffuse Mixture Frugality with Strong Interior-Gated Validity Leak (sub-linear coherence variant). People are one-reason decision makers: experts are consulted one at a time and the first strictly discriminating expert decides; no integration across cues, no history use, exact validity ties anchor the consultation order, and a trial with no discriminating expert yields exactly 0.5. The population-central hierarchy is anti-validity (distrust-the-weakest: consult ascending advertised validity). Three graded mechanisms modulate the core. (1) STRONG INTERIOR-GATED VALIDITY LEAK with SUB-LINEAR COHERENCE WEIGHTING: when the anti cascade's first discriminating expert carries a strictly interior validity (min + 0.10 < val < max - 0.10), the decision mixes toward the standard TTB (descending-validity) decider with probability q = rho_leak * sqrt(reversal_coherence) * gap_term, where gap_term is the range-normalized mean adjacent unique-validity gap. The sub-linear coherence exponent leaves fully reversal-coherent designs (coherence ~ 1) exactly on their calibrated anchor values while granting moderately coherent designs substantially more leak mass than the linear weighting did — capturing the empirical fact that leak toward the strongest expert is not confined to perfectly mirror-structured validity lists. (2) DIFFUSE HETEROGENEITY: w_j = -val_j + gamma*salience_j + A^3*(sigma_h*zeta_j + kappa*xi_j), with a small stable per-subject part (zeta) and a large momentary per-trial part (xi), applied only when all validities are distinct. (3) POSITIONAL-PRIMACY MINORITY, TIE-ANCHORED and STEEPLY THRESHOLD-GATED: the reading-order fallback fires only on designs that BOTH contain exact advertised-validity ties AND exhibit near-maximal binding ambiguity (A >= 0.80, linear ramp over 0.20, mixture weight capped at 0.35); all-distinct and monotone/anchored designs remain fully collapsed onto the anti-validity core. Response discipline is heterogeneous: softmax(beta) plus lapse epsilon with beta in [0.3, 2.0] and epsilon in [0.08, 0.30], giving pooled decider-follow ~0.69 with the wide between-subject spread that reproduces the follow-protection tail.

**Parameters:**
- rho_leak: [0.85, 1.25]
- primacy_weight: [0.3, 0.45]
- gamma: [-0.05, 0.12]
- sigma_h: [0.6, 1.2]
- kappa: [8, 12]
- beta: [0.3, 2.0]
- epsilon: [0.08, 0.30]
- tie_break: [0, 1]
- zeta: [(-1, 1)] * n_features
- validities: validities

**`predict source code`:**
```python
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
```

**`policy source code`:**
```python
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return np.random.choice(len(probabilities), p=probabilities)
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] The design exploits the core structural difference between Take-The-Best (TTB) and Tallying: TTB's choice is determined solely by the single highest-validity discriminating cue, whereas Tallying integrates the sign of every feature comparison. I use 5 features with a clear validity hierarchy [0.90, 0.80, 0.70, 0.60, 0.55] so the cue cascade is unambiguous. The trial set is built to produce a QUANTITATIVE dissociation, not just a qualitative one. (1) Conflict trials T1-T3: option A wins only on the top cue (validity 0.90) while B wins on progressively fewer lower cues (tally margins -3, -2, -1 for A). TTB predicts p(A) is IDENTICAL across T1, T2, T3 (a constant (1-eps)*sigmoid(beta)+eps/2, since the TTB score is always [1,0]); Tallying predicts a strictly graded p(A) that increases with the tally margin (sigmoid(-3*beta), sigmoid(-2*beta), sigmoid(-beta)). A flat vs. graded profile across these three trial types is the central quantitative signature. (2) Mirror conflict trials T4-T6 test the same dissociation in the opposite direction (TTB->B, Tallying->A), ruling out side-bias explanations. (3) Tally-tie trials T7-T8: the tally is exactly 1-1, so Tallying is forced to 50/50 regardless of beta, while TTB is near-deterministic (top cue decides). These trials alone can falsify Tallying at the aggregate level: any consistent preference is TTB-only. (4) Cascade-depth trials T11-T12: the top cue ties and the SECOND cue (0.80) discriminates against the tally majority — TTB follows cue 2, Tallying follows the 3-vs-2 count. This confirms subjects respect the validity ordering, not just 'any high cue wins'. (5) Agreement trials T9-T10 (both theories favor the same option, margin +1) and a full-tie trial T13 (both predict exactly 50/50) serve as controls: they anchor the lapse rate and verify that neither theory is needed where both agree. Crucially, TTB predicts a single constant choice probability across all five TTB->A trial types (T1, T2, T3, T7, T9), whereas Tallying predicts five distinct probabilities spanning from well below 0.5 to well above 0.5 — model comparison on the response profile across these cells cleanly separates the theories even after fitting beta and epsilon to each. With 13 unique pairs and K=7 repetitions (91 trials), each cell is sampled enough to estimate choice proportions per subject.

[1] This design dissociates Tallying from Take-The-Best (TTB) along two orthogonal quantitative dimensions simultaneously: (i) the tally MARGIN (a_wins - b_wins), which is the ONLY trial-dependent quantity in Tallying's softmax (p(A) = (1-eps)*sigmoid(beta*margin) + eps/2), and (ii) the identity/validity of the first discriminating cue, which is the ONLY trial-dependent quantity in TTB (p(A) = (1-eps)*sigmoid(beta) + eps/2 toward the top-cue winner, constant across ALL discriminating trials). I use 6 features with an unambiguous validity hierarchy [0.95, 0.85, 0.75, 0.65, 0.60, 0.55]. CENTRAL SIGNATURE 1 — the full crossing psychometric function (T1-T5 and mirror T6-T10): the top cue (0.95) always favors one option while the tally margin sweeps across its full range (-4, -2, 0, +2, +4). TTB predicts a FLAT p(A) = (1-eps)*sigmoid(beta)+eps/2 across all five trials in each direction; Tallying predicts a strictly monotone graded curve that passes through EXACTLY 0.5 at margin 0 (T3/T8) — a knife-edge prediction independent of beta that any consistent preference falsifies — and reverses sign at negative margins (T1/T2: TTB says A with high confidence, Tallying says B; T6/T7: the mirror sign reversal, ruling out side bias). T1 vs T10 is especially clean: identical tally margins (-4) but opposite TTB winners, so Tallying predicts identical p(A) on both while TTB predicts opposite extremes. CENTRAL SIGNATURE 2 — cascade-depth invariance (T11-T15): the top cue TIES and the TTB decision comes from cue 2 (0.85), cue 3 (0.75), or even cue 5 (0.60, T14). TTB predicts the SAME constant p(A) on T1, T2, T3, T11, T12, T13, T14 (all TTB->A regardless of which cue decided or how deep the cascade ran); Tallying predicts six distinct values spanning sigmoid(-2beta) to sigmoid(+1beta), including exact 0.5 on T3/T13/T14. T15 is a depth-2 sign reversal (cue 2 favors B, tally margin +2 favors A). T14 probes the bottom of the hierarchy: when cues 1-4 all tie, TTB commits to the single 0.60-validity cue while Tallying sits at exactly 50/50 — if subjects treat a lone low-validity cue as uninformative, TTB fails where Tallying succeeds. T16 (full tie) is the shared 50/50 control anchoring lapse rate and side bias. With 16 unique pairs and K=6 (96 trials exactly), each cell yields 6 repetitions per subject for stable choice-proportion estimates. Model comparison on the full 16-cell response profile — flat vs. graded across margins, constant vs. varied across cascade depths, and exact-0.5 vs. deterministic on margin-0 trials — cleanly separates the theories even after fitting beta and epsilon to each.

[2] This design dissociates Reversed-Hierarchy Frugality (misbinding TTB) from Tallying by exploiting their two orthogonal trial-dependent quantities: (i) Tallying's choice depends ONLY on the tally margin m = a_wins - b_wins, giving p(A) = (1-eps)*sigmoid(beta*m) + eps/2, a strictly graded psychometric curve; (ii) misbinding TTB's choice depends ONLY on which cue first discriminates when the cascade runs in ASCENDING true validity (kappa=1 binds the highest subjective validity to the LAST feature), giving a CONSTANT p(winner) = (1-eps)*sigmoid(beta) + eps/2 ~ 0.60 on every discriminating trial. I use 5 features with validities [0.90, 0.80, 0.70, 0.60, 0.55], so the reversed cascade consults the 0.55 expert first, then 0.60, 0.70, 0.80, 0.90. CENTRAL SIGNATURE 1 - margin sweep with the lowest-validity cue as fixed decider (trials 1-4 and 5-8): the 0.55 expert always favors one option while the tally margin sweeps across -3, -2, -1, 0 (and mirrored +3, +2, +1, 0). Misbinding TTB predicts a FLAT p ~ 0.60 toward the lowest-cue winner on all eight trials; Tallying predicts a strictly monotone graded curve that passes through EXACTLY 0.5 at margin 0 (trials 4, 8) - a knife-edge prediction independent of beta - and reverses sign at negative margins (trial 1: Tallying says B strongly, misbinding TTB says A). CENTRAL SIGNATURE 2 - matched-margin opposite-decider pairs, a parameter-free falsification for Tallying: pairs (2, 10), (4, 8), (6, 9), (7, 14) have IDENTICAL tally margins (-2, 0, +2, +1) but opposite lowest-cue winners. Tallying predicts IDENTICAL p(A) within each pair for ANY (beta, eps); misbinding TTB predicts a ~0.60 vs ~0.40 split within each pair. Trial 14 is especially diagnostic: it is the deepest cascade (only the 0.90 expert discriminates, all lower cues tie), and misbinding TTB predicts the SAME constant p there as on trial 1 (where the 0.55 expert decides) - cascade-depth invariance - while Tallying treats trials 7 and 14 identically (both margin +1) though misbinding TTB gives opposite choices. CENTRAL SIGNATURE 3 - reversed-hierarchy depth probes at margin 0 (trials 11, 12, 13): the decider is the 0.60, 0.70, or 0.80 expert while the tally is exactly 1-1, so Tallying is pinned to exactly 0.50 regardless of parameters while misbinding TTB stays at ~0.60/~0.40. Trial 15 is a depth-3 conflict (0.70 expert favors A, tally margin -1 favors B). Trial 5 additionally separates misbinding TTB from STANDARD TTB: the top-validity 0.90 expert and the tally both favor A, but the reversed cascade follows the 0.55 expert to B - if subjects choose A here, the misbinding claim fails. Trial 16 is a full-tie control anchoring lapse and side bias (both theories: exactly 0.5). The overall profile is non-monotone in margin (margin +3 -> p(A) ~ 0.40; margin -3 -> p(A) ~ 0.60), which no monotone psychometric function of margin - i.e., no parameterization of Tallying - can reproduce, whereas misbinding TTB predicts one constant across all 15 discriminating trials with variation only in direction. 16 unique pairs, K = 6 (96 trials exactly), give 6 repetitions per cell for stable per-subject choice proportions.

[3] This design dissociates Tallying from Reversed-Hierarchy Frugality (misbinding TTB, kappa=1) along three novel quantitative axes that the previous designs did not exploit. SETUP: 6 features with validities [0.90, 0.80, 0.70, 0.65, 0.60, 0.55]. Because validities descend across positions, the misbound hierarchy (positional reversal) equals an ascending-validity cascade: the reversed TTB model consults f5 (0.55) first, then f4, f3, f2, f1, f0 (0.90) last. AXIS 1 — THE CONFIDENCE CEILING (T3, T4, T12): the competing theory's beta is locked in [0.45, 0.55] with epsilon in [0.11, 0.15], so its predicted choice probability on ANY discriminating trial is capped at ~(0.87)*sigmoid(0.55)+0.075 ≈ 0.62. T3/T4 are unanimous trials (margin ±6) and T12 is a sparse trial where only the two highest-validity experts discriminate, both favoring A. Every sensible strategy — including Tallying with a moderate beta (p(A) = (1-eps)*sigmoid(2*beta)+eps/2, up to ~0.93) — predicts near-ceiling consistency here, but the misbinding model is mathematically FORBIDDEN from exceeding ~0.62. Human consistency above ~65% on these cells falsifies the competing theory at its parameter boundary while Tallying accommodates it. AXIS 2 — EXTREME-MARGIN CONFLICT WITH ORDINAL INVERSION (T1, T2, T14): on T1, A wins the four highest-validity experts (margin +4) but the lowest-validity expert (f5) favors B. Tallying predicts p(A) = (1-eps)*sigmoid(4*beta)+eps/2 ≥ 0.59 (well above 0.5 for any beta > 0.1); reversed TTB follows f5 to B, predicting p(A) ≈ 0.39. Crucially, compare T1 to T5 (margin 0, decider f5 favors A): Tallying predicts p(T1) >> p(T5) = exactly 0.50, while reversed TTB predicts p(T1) ≈ 0.39 < p(T5) ≈ 0.61 — a complete ORDINAL INVERSION of the two-cell response profile that no parameter setting of either model can reconcile with the other's prediction. T2 and T14 mirror this inversion, ruling out side bias. AXIS 3 — KNIFE-EDGE AND ISO-MARGIN FAMILIES (T5–T11, T13, T14, T16): five margin-0 trials (T5–T8, T16) pin Tallying to EXACTLY 0.50 regardless of (beta, epsilon) — a parameter-free prediction — while reversed TTB predicts 0.61/0.39 splits on every one, at shallow (f5), middle (f3), and deep (f1) cascade depths, i.e., depth-invariant. The iso-margin family {T9, T10, T11, T12} all have margin +2: Tallying predicts IDENTICAL p(A) on all four for any parameters; reversed TTB predicts 0.61, 0.39, 0.39, 0.61 (decider flips between f5, f3, f1). The mirror family {T13, T14} does the same at margin -2, where Tallying prefers B (~0.30 for A) but reversed TTB prefers A on T14 (0.61) — opposite sides of 0.5. T15 (full tie) is the shared 50/50 control anchoring lapse and side bias. GLOBALLY: reversed TTB's entire 16-cell profile is a two-level step function (≈0.61 toward the lowest-validity discriminator, ≈0.39 against, 0.50 on the full tie), whereas Tallying predicts a monotone psychometric function of margin with exact 0.5 at m=0 and identical values within margin levels — two cross-structured profiles that model comparison on the full response profile separates cleanly even after fitting beta and epsilon to each. With 16 unique pairs and K = 6 (96 trials exactly), each cell yields 6 repetitions per subject.

[4] GOAL: Quantitatively dissociate Reversed-Hierarchy Frugality (misbinding TTB, kappa=1) from Tallying, while adding a novel axis absent from prior designs: a SCRAMBLED (non-monotone) validity arrangement that makes the misbound hierarchy a non-trivial permutation, enabling a 'subjective-hierarchy ladder' and a 'decider-validity invariance' test. SETUP: 6 features, validities [0.70, 0.90, 0.60, 0.95, 0.75, 0.55] (positions f0..f5). The misbound subjective weights are w = val reversed = [0.55, 0.75, 0.95, 0.60, 0.90, 0.70], so the misbinding cascade consults f2 -> f4 -> f1 -> f5 -> f3 -> f0. Crucially, the misbound model's FIRST cue is Expert 3 (advertised only 60%), while the truly best expert (95%, f3) is consulted 5th. Tallying ignores validities entirely, so its predictions depend only on the tally margin m = a_wins - b_wins: p(A) = (1-eps)*sigmoid(beta*m) + eps/2. The misbinding model's predictions depend only on the decider: constant p = (1-eps)*sigmoid(beta)+eps/2 in [0.63, 0.78] toward the first cue in the order f2,f4,f1,f5,f3,f0 that discriminates. SIGNATURE 1 - margin sweep with fixed decider (T1-T6): Expert 3 (f2) always favors A while the margin sweeps -4, -2, 0, +1, +3 (T1, T3, T4, T5, T6), plus the A/B mirror at margin +4 with decider->B (T2). Misbinding TTB predicts a FLAT p(A) in [0.63, 0.78] on all six; Tallying predicts a strictly graded psychometric curve that passes through EXACTLY 0.5 at T4 (margin 0, parameter-free knife-edge) and stays below 0.5 at negative margins (T1: Tallying <= 0.45 for ANY beta in [0.1, 20], eps in [0, 0.5], while misbinding TTB >= 0.63 - non-overlapping prediction intervals on this single cell). SIGNATURE 2 - extreme-margin matched pair (T2 vs T15): both have margin +4, but the decider is f2->B on T2 and f2->A on T15. Tallying predicts IDENTICAL p(A) on the two cells for ANY (beta, eps) - a parameter-free prediction - while misbinding TTB predicts ~0.22-0.37 on T2 vs ~0.63-0.78 on T15, opposite sides of 0.5. SIGNATURE 3 - the subjective-hierarchy ladder at margin 0 (T4, T7-T11): six knife-edge cells, each with exactly one cue favoring each option, so Tallying is pinned to EXACTLY 0.50 on every rung; any consistent preference on any rung falsifies Tallying regardless of parameters. The rungs pit adjacent cues of the MISBOUND order against each other (f2 vs f4; f2 vs f1; f4 vs f1; f5 vs f3 and its mirror; f3 vs f0), so subjects' choices trace out their subjective priority order. Because the arrangement is scrambled, five of six rungs also place standard validity-respecting TTB on the OPPOSITE side (e.g., T9/T10: the 55% expert f5 vs the 95% expert f3 - misbinding follows the 55% expert, standard TTB the 95% expert), making the ladder a simultaneous three-way separation: misbinding TTB ~0.7 toward its decider, Tallying exactly 0.5, standard TTB often the reverse. SIGNATURE 4 - decider-validity invariance at fixed margin (T5, T12, T13, T14): four margin-+1 cells whose misbound deciders are advertised at 60%, 75%, 90%, and 55% respectively. Misbinding TTB predicts the SAME p(A) on all four (the follow rate cannot depend on the decider's advertised accuracy); any monotone effect of decider validity falsifies the kappa=1 misbinding claim. These are agreement cells for Tallying (identical margin), but they pin the flat level as decider-independent, and the within-decider contrast T4 vs T5 (margin 0 vs +1, same decider) tests Tallying's strict margin effect where misbinding TTB predicts none. SIGNATURE 5 - consistency ceiling (T15): a margin-+4 agreement cell where only the 95% expert dissents. Misbinding TTB is capped at ~0.78 by its parameter ranges (beta <= 1.2, eps >= 0.08); human consistency above ~0.80 here falsifies the advocated theory at its boundary while Tallying accommodates it. CONTROLS: T16 is a full tie (both theories: exactly 0.5, anchoring lapse and side bias); T1/T2 are exact A/B mirrors; T10 and T8 provide decider->B cells at margin 0 to rule out side bias in the ladder. GLOBAL PREDICTED PROFILES: misbinding TTB produces a two-level step function over the 16 cells (~0.7 toward the misbound decider on all 15 discriminating trials, exactly 0.5 on T16), invariant to margin, cascade depth, and decider validity; Tallying produces a monotone psychometric function of margin with exact 0.5 on the six margin-0 cells and identical values within margin levels (T2 = T15; T5 = T12 = T13 = T14). The profiles are structurally disjoint (e.g., misbinding says p(T1) = p(T6); Tallying says p(T1) < 0.45 < p(T6)), so model comparison on the full 16-cell response profile separates the theories even after fitting beta and epsilon to each. With 16 unique pairs and K = 6 (96 trials exactly), each cell yields 6 repetitions per subject for stable choice proportions.

[5] NOVEL AXIS - THE CONCORDANCE LADDER: every prior design dissociated Tallying from TTB-family models using CONFLICT trials (some cues favor each side). This design's centerpiece is a family of zero-conflict trials where every discriminating cue favors the same option and only the NUMBER of supporting cues varies (ties fill the rest). Tallying's output depends only on the margin m = a_wins - b_wins, so it predicts a strictly GRADED psychometric curve p(A) = (1-eps)*sigmoid(beta*m) + eps/2 even when no cue disagrees — climbing from ~0.62 at m=+1 to ~0.97 at m=+6 for moderate beta. Reversed-Hierarchy Frugality is a one-reason model: on every concordance trial the first discriminating cue in its misbound cascade decides, so it predicts ONE CONSTANT p(A) = (1-eps)*sigmoid(beta)+eps/2 across the entire ladder — and, critically, its fixed noise regime (beta in [0.8,1.2], eps in [0.08,0.14]) mathematically CAPS its follow probability at ~0.75. Human accuracy above ~0.80 on high-rung concordance cells falsifies the competing theory at its parameter boundary while Tallying accommodates it. SETUP: 6 features with ASCENDING validities [0.55, 0.60, 0.65, 0.75, 0.85, 0.95] (f0..f5) — an arrangement no prior design used. The misbound hierarchy w = val reversed = [0.95, ..., 0.55] makes the reversed cascade consult f0 first: the FIRST-listed, LOWEST-validity (55%) expert is the misbinding model's most trusted cue, and the 95% expert is consulted last. SIGNATURE 1 - CONCORDANCE LADDER (T1-T5): A weakly dominates B, margins +1, +2, +4, +5, +6; the misbound decider sweeps across f5 (95%), f4 (85%), f2 (65%), f1 (60%), f0 (55%). Misbinding TTB: flat ~0.70 on all five rungs (jointly a cascade-depth- AND decider-validity-invariance claim); Tallying: strictly increasing, exceeding the misbinding cap on the top rungs. SIGNATURE 2 - SINGLE-CELL NON-OVERLAP CONFLICT SWEEP (T6-T8, T9-T11): the 55% expert f0 always decides for A (T6: m=-4, T7: m=-2, T8: m=0) or always for B (T9: m=+4, T10: m=+2, T11: m=0). On T6/T7 Tallying is mathematically BELOW 0.50 for ANY parameters (negative margin), while misbinding is at or above 0.663 — the prediction intervals are disjoint, so each cell alone falsifies one theory. On T9/T10 the inversion is complete (Tallying > 0.50, misbinding <= 0.337). T6 vs T9 are exact A/B mirrors ruling out side bias. SIGNATURE 3 - KNIFE-EDGE FAMILY (T8, T11, T13, T15): four margin-0 cells (1-1, 1-1, 2-2, 3-3) where Tallying is pinned to EXACTLY 0.50 independent of (beta, epsilon) — any consistent preference on any of these cells falsifies Tallying — while misbinding predicts 0.70/0.30 splits (decider f0) at every tally composition, i.e., invariance to how many cues sit on each side. SIGNATURE 4 - ISO-MARGIN OPPOSITE-DECIDER TRIPLETS (parameter-free for Tallying): {T2, T10, T12} all have m=+2 — Tallying predicts IDENTICAL p(A) on all three for any parameters; misbinding predicts 0.70 (T2, decider f4->A), 0.70 (T12, decider f0->A), 0.30 (T10, decider f0->B). {T3, T9} both m=+4: Tallying identical; misbinding 0.70 vs 0.30. {T1, T14} both m=+1 with deciders at opposite ends of the subjective hierarchy (f5, 95% vs f0, 55%): BOTH theories predict identical p(A) here — a shared control that pins misbinding's flat level and tests its claim that the decider's advertised accuracy is irrelevant. T16 is the full-tie control (both theories: exactly 0.50, anchoring lapse and side bias). GLOBAL PROFILES: misbinding TTB emits a two-level step function over the 16 cells (~0.70 on the eleven decider->A cells including ALL ladder rungs and ALL negative-margin conflicts; ~0.30 on the four decider->B cells; 0.50 on the tie), invariant to margin, tally composition, cascade depth, and decider validity. Tallying emits a monotone psychometric function of margin with exact 0.50 on the four margin-0 cells, exact equality within margin levels (T2=T10=T12; T3=T9; T1=T14), and values above the misbinding ceiling on T4/T5. The profiles are structurally disjoint — e.g., misbinding says p(T1)=p(T5)=p(T6); Tallying says p(T6)<0.50<p(T1)<p(T5) — so model comparison on the full 16-cell response profile separates the theories even after fitting beta and epsilon to each. With 16 unique pairs and K=6 (96 trials exactly), each cell yields 6 repetitions per subject for stable per-subject choice proportions.

[6] GOAL: Quantitatively dissociate Reversed-Hierarchy Frugality (misbinding TTB, kappa=1) from Tallying via a validity arrangement whose MISBOUND hierarchy consults the three WEAKEST experts before any of the three strongest — a 'weak-expert dictatorship' schedule no prior design used. SETUP: 6 features, validities [0.60, 0.95, 0.55, 0.85, 0.65, 0.90] (f0..f5). The misbound weights w = val reversed = [0.90, 0.65, 0.85, 0.55, 0.95, 0.60], so the misbinding cascade consults f4 (advertised 65%) -> f0 (60%) -> f2 (55%) -> f1 (95%) -> f5 (90%) -> f3 (85%): the three lowest-validity experts occupy the top three rungs of the subjective hierarchy, and the 95% expert is not consulted until depth 4. Tallying ignores validities entirely: p(A) = (1-eps)*sigmoid(beta*m) + eps/2, depending only on the tally margin m. The misbinding model emits a constant p(decider winner) = (1-eps)*sigmoid(beta)+eps/2, which over its parameter ranges (beta in [0.6,1.2], eps in [0.08,0.18]) lies in [0.62, 0.80] — a hard ceiling. SIGNATURE 1 — WEAK-EXPERT DICTATORSHIP SWEEP (T1-T5): the misbound top cue f4 (a 65% expert) always decides for A while the margin sweeps -3, -1, 0, +1, +3. Misbinding TTB predicts a FLAT p(A) in [0.62, 0.80] on all five; Tallying predicts a strictly graded monotone curve passing through EXACTLY 0.5 at T3 (m=0, parameter-free knife-edge). Single-cell non-overlap at T1 (m=-3): Tallying is mathematically <= 0.463 for ANY (beta, eps) in its ranges (max at beta=0.1, eps=0.5), while misbinding is >= 0.62 — disjoint prediction intervals. T6-T8 are the A/B mirrors (decider f4 -> B at m=+3, 0, -3): at T6 the four STRONGEST experts (95%, 90%, 85%, 60%) all favor A with margin +3, yet misbinding follows the lone 65% expert to B (p(A) in [0.20, 0.38]) while Tallying is >= 0.537 — again disjoint. T5 provides the ceiling test: unanimous-ish margin +3 where Tallying can reach ~0.97 but misbinding is capped at 0.80; human consistency above ~0.82 falsifies the advocated theory at its parameter boundary. SIGNATURE 2 — ISO-MARGIN OPPOSITE-DECIDER PAIRS (parameter-free for Tallying): (T4, T12) at m=+1, (T3, T7) at m=0, (T5, T6) at m=+3 — identical margins, opposite f4 deciders. Tallying predicts IDENTICAL p(A) within each pair for ANY parameters; misbinding predicts a ~0.70 vs ~0.30 split within each pair, straddling 0.5. SIGNATURE 3 — KNIFE-EDGE LADDER AT m=0 WITH VARYING CASCADE DEPTH (T3, T7, T9, T10, T11): five margin-0 cells pinning Tallying to EXACTLY 0.50 regardless of (beta, epsilon) — any consistent preference on any rung falsifies Tallying — while the misbound decider sits at depth 1 (f4, T3/T7), depth 2 (f0, T9), depth 3 (f2, T10), and depth 4 (f1, T11), in both directions. Misbinding predicts the same ~0.70/0.30 split at every depth: depth invariance. SIGNATURE 4 — DECIDER-VALIDITY INVARIANCE AT m=+1 (T4, T13, T14, T15): four margin-+1 cells whose misbound deciders are advertised at 65% (f4), 60% (f0), 55% (f2), and 95% (f1). Misbinding predicts the SAME follow rate on all four — the follow probability cannot depend on the decider's advertised accuracy; any monotone effect of decider validity (e.g., subjects following the 95% expert in T15 more than the 55% expert in T14) falsifies the kappa=1 misbinding claim. These are agreement cells for Tallying (identical margin), so they isolate the misbinding model's internal structure. GLOBAL PROFILES: misbinding TTB emits a two-level step function over the 16 cells (~[0.62,0.80] toward the misbound decider on all 15 discriminating trials, exactly 0.5 on the full tie), invariant to margin, cascade depth, and decider validity; Tallying emits a monotone psychometric function of margin with exact 0.5 on all five margin-0 cells and exact equality within margin levels (T4=T12=T13=T14=T15; T5=T6; T1=T8). The profiles are structurally disjoint — e.g., misbinding says p(T1)=p(T5) (both decider f4->A) while Tallying says p(T1) <= 0.463 < 0.537 <= p(T5); misbinding says p(T3)=p(T4) while Tallying says p(T3)=0.5 < p(T4) — so model comparison on the full 16-cell response profile separates the theories even after fitting beta and epsilon to each. T16 (full tie) anchors lapse rate and side bias. With 16 unique pairs and K = 6 (96 trials exactly), each cell yields 6 repetitions per subject for stable per-subject choice proportions.

[7] GOAL: Quantitatively dissociate Tallying (advocated) from Reversed-Hierarchy Frugality (misbinding TTB, kappa=1) via a design organized around a NOVEL structural centerpiece: a balanced, orthogonal 2x5 FACTORIAL GRID crossing tally margin with misbound-decider direction, plus a 'sandwich-scrambled' validity arrangement whose misbound hierarchy is non-monotone in advertised accuracy. SETUP: 7 features (no prior design used more than 6, giving a wider margin range up to +/-5) with validities [0.60, 0.90, 0.55, 0.80, 0.65, 0.75, 0.95] (f0..f6). The misbound weights w = val reversed = [0.95, 0.75, 0.65, 0.80, 0.55, 0.90, 0.60], so the misbinding cascade consults f0 -> f5 -> f3 -> f1 -> f2 -> f6 -> f4, i.e., advertised accuracies along the subjective hierarchy run 60% -> 75% -> 80% -> 90% -> 55% -> 95% -> 65%: a NON-MONOTONE sandwich in which the first-listed 60% expert is trusted most, the 95% expert is consulted SIXTH (next to last), and the 55% expert sits above it at depth 5. Tallying ignores validities entirely: p(A) = (1-eps)*sigmoid(beta*m) + eps/2, depending only on margin m. Misbinding TTB emits a constant follow probability p = (1-eps)*sigmoid(beta)+eps/2, which over its parameter ranges (beta in [0.6,1.2], eps in [0.08,0.18]) lies in [0.62, 0.75] toward the first discriminating cue in the order f0,f5,f3,f1,f2,f6,f4. SIGNATURE 1 - THE ORTHOGONAL FACTORIAL GRID (T1-T10): the misbound TOP cue f0 (advertised only 60%) always decides, toward A in T1-T5 and toward B in T6-T10 (exact A/B mirrors), while the margin sweeps -3, -1, 0, +1, +3. This is a fully balanced 2 (decider direction) x 5 (margin) design, and the two theories predict TRANSPOSED main-effect structures in the marginal means: Tallying predicts a NULL decider-row effect (both row means exactly 0.5, since margins are symmetric within each row) with a strictly graded margin-column effect passing through EXACTLY 0.5 at m=0; misbinding TTB predicts a strong decider-row effect (~0.70 vs ~0.30) with a NULL margin-column effect (every column mean exactly 0.5). A 2x5 ANOVA or model comparison on the grid cleanly awards the data to one theory: any significant margin effect with null decider effect falsifies misbinding; any significant decider effect with null margin effect falsifies Tallying. SIGNATURE 2 - SINGLE-CELL DISJOINT PREDICTION INTERVALS AT EXTREME MARGINS (T14, T15): on T14, ALL six other experts - including the 95% expert - favor A (margin +5) and only the first-listed 60% expert favors B. Misbinding follows the 60% expert to B, predicting p(A) <= 0.38; Tallying predicts p(A) = (1-eps)*sigmoid(5*beta)+eps/2 >= 0.56 even at its WORST parameters (beta=0.1, eps=0.5) - the intervals are disjoint, so this one cell alone falsifies one theory. T15 is the exact mirror (margin -5, lone 60% expert favors A): Tallying <= 0.44, misbinding >= 0.62. These cells also separate misbinding from STANDARD TTB (which follows the 95% expert to A on T14), making the design adversarial to the whole TTB family. SIGNATURE 3 - PARAMETER-FREE KNIFE EDGES WITH COMPOSITION CONTROL (T3, T8, T11): three margin-0 cells - a 2-2 tally with decider f0->A (T3), its mirror with decider f0->B (T8), and a sparse 1-1 tally with decider f0->A (T11). Tallying is pinned to EXACTLY 0.50 on all three regardless of (beta, eps) and regardless of tally composition; misbinding predicts ~0.70/~0.30 splits on all three, invariant to whether 2 or 6 cues sit on each side. SIGNATURE 4 - DECIDER-VALIDITY AND DEPTH INVARIANCE AT FIXED MARGIN (T2, T12, T13): three margin-+1 cells whose misbound deciders are the 60% first-listed expert at depth 1 (T2), the 90% expert at depth 4 (T12), and the 95% expert at depth 6 (T13, where it is the ONLY discriminating cue). Misbinding predicts the SAME follow rate on all three - the follow probability cannot depend on the decider's advertised accuracy or cascade depth; any monotone validity gradient (e.g., following the lone 95% expert more than the 60% expert) falsifies the kappa=1 misbinding claim. Tallying also predicts identical values here (same margin), so these cells isolate the competitor's internal structure. GLOBAL PROFILES: misbinding TTB emits a two-level step function over the 16 cells (~[0.62,0.75] toward the misbound decider on all 15 discriminating trials, exactly 0.5 on the full tie), invariant to margin, tally composition, cascade depth, and decider validity; Tallying emits a monotone psychometric function of margin with exact 0.5 on the three margin-0 cells, exact equality within margin levels (T1=T10-margin... specifically T1=T10? No: T1 (m=+3, dec->A) = T10 (m=+3, dec->B) in p(A); T2=T9=T12=T13; T5=T6; T14 highest, T15 lowest), and values far outside the misbinding cap on T14/T15. The profiles are structurally disjoint - e.g., misbinding says p(T15)=p(T1) (both decider->A) while Tallying says p(T15) <= 0.44 < 0.56 <= p(T1); misbinding says p(T3)=p(T2) while Tallying says p(T3)=0.5 < p(T2) - so model comparison on the full 16-cell response profile separates the theories even after fitting beta and epsilon to each. T16 (full tie) anchors lapse rate and side bias (both theories: exactly 0.5). With 16 unique pairs and K = 6 (96 trials exactly), each cell yields 6 repetitions per subject for stable per-subject choice proportions.

[8] GOAL: Quantitatively dissociate Reversed-Hierarchy Frugality (advocated; positional misbinding, w = val[::-1]) from Weakest-Expert-First Frugality (competing; value-based anti-validity sort, ascending validity). These two one-reason models are behaviorally IDENTICAL whenever the validity list is monotone in position (both then consult cues in ascending validity), so the design's foundation is a SCRAMBLED validity arrangement that makes the two hierarchies genuinely different permutations — the one structural axis that separates them. SETUP: 6 features with all-distinct validities [0.90, 0.55, 0.70, 0.60, 0.80, 0.95] (f0..f5). Distinctness is deliberate: it makes the competing model's free tie_break parameter irrelevant, stripping it of extra flexibility. The two hierarchies are: ADVOCATED (w = val reversed = [0.95, 0.80, 0.60, 0.70, 0.55, 0.90], descending): f0 -> f5 -> f1 -> f3 -> f2 -> f4, i.e., the misbinding model's most trusted cue is the FIRST-listed 90% expert and its second is the LAST-listed 95% expert. COMPETING (ascending validity): f1 (55%) -> f3 (60%) -> f2 (70%) -> f4 (80%) -> f0 (90%) -> f5 (95%), i.e., the weakest expert first and the two strongest experts consulted LAST. The hierarchies disagree on 12 of 15 cue pairs; crucially, whenever f0 or f5 discriminates against any of f1..f4, the advocated model follows the 90%/95% (STRONG) expert while the competing model follows the WEAKER discriminator — a head-to-head 'strong-expert vs weak-expert dictatorship' test of the two anti-validity claims. SIGNATURE 1 — STRONG-VS-WEAK DISSOCIATION LADDER (T1-T13): on T1-T8 the first-listed 90% expert (f0, the advocated model's top cue, consulted 5th by the competitor) favors one option while a single weaker expert (55%, 70%, 60%, or 80%) favors the other; on T9-T13 the last-listed 95% expert (f5, the advocated model's second cue, consulted LAST by the competitor) plays the same role. Advocated predicts p(A) = (1-eps)*sigmoid(beta)+eps/2 in [0.62, 0.75] toward the strong expert on every rung; competing predicts the same quantity toward the weak expert, i.e., p(A) in [0.20, 0.40]. THE PREDICTION INTERVALS ARE DISJOINT ON EVERY ONE OF THE 13 DISSOCIATING CELLS: any cell with p(A) > 0.55 falsifies the weak-expert-first theory; any cell with p(A) < 0.45 falsifies the misbinding theory. T1-T12 are exact A/B mirror pairs, ruling out side bias. SIGNATURE 2 — DECIDER-VALIDITY GRADIENT (parameter-free direction test): across T1, T3, T5, T7, T9, T11, T13 the advocated decider is always the highest-validity discriminator (90% or 95%) while the competing decider sweeps 55-80%. If subjects' choices track the STRONG expert's direction, the competing theory's core claim (subjective weights = -validity) is falsified regardless of (beta, eps, tie_break); if they track the WEAK expert's direction, the kappa=1 positional-misbinding claim is falsified. Both models additionally predict follow-rate INVARIANCE across which weak/strong expert decides (flat follow rate across rungs) — a shared internal check. SIGNATURE 3 — COALITION CONFLICT (T14): the 90% expert alone favors A while the 55%, 60%, and 70% experts favor B (tally margin -2). Advocated follows f0 -> A; competing follows f1 -> B; Tallying also says B. This cell simultaneously dissociates the advocated model from the competing theory AND from any margin-integrating strategy, and tests the shared one-reason claim p(T14) = p(T1) (same deciders, different margin and cascade context). SIGNATURE 4 — CONTROLS: T15 (only the 60% expert discriminates) is an agreement cell where both theories follow the same decider at different cascade depths (advocated depth 4, competing depth 2) — it anchors the follow rate and tests both models' depth-invariance claim; any difference between p(T15) and the dissociation-cell follow rates falsifies the flat-follow assumption of whichever model mispredicts it. T16 (full tie) pins both theories to exactly 0.50, anchoring lapse rate and side bias. GLOBAL PROFILES: the advocated model emits p(A) ≈ [0.62, 0.75] on T1,T3,T5,T7,T9,T11,T13,T14,T15 and ≈ [0.25, 0.38] on T2,T4,T6,T8,T10,T12, with 0.50 on T16; the competing model emits the EXACT TRANSPOSE (near-perfect anti-correlation on 14 of 16 cells, agreement only on T15 and T16). Because the intervals are disjoint cell-by-cell, model comparison on the full 16-cell response profile separates the theories even after fitting beta, epsilon, and tie_break to each — and even the AGGREGATE choice rate across the 12 mirror cells (advocated: strong-expert direction ~70%; competing: weak-expert direction ~70%) is diagnostic. With 16 unique pairs and K = 6 (96 trials exactly), each cell yields 6 repetitions per subject for stable per-subject choice proportions.

[9] GOAL: Quantitatively dissociate Weakest-Expert-First Frugality (advocated; value-based ascending-validity cascade) from Reversed-Hierarchy Frugality (competing; positional misbinding, w = val[::-1]). The two theories are behaviorally IDENTICAL whenever the validity list is monotone in position, so the design rests on a scrambled arrangement that turns them into two genuinely different cue permutations — but with a novel twist relative to design [8]: here BOTH hierarchies are 'weak-expert dictatorships', so the design tests the MECHANISM of the anti-validity claim (value-sort vs positional misbinding) without the confound that subjects simply follow strong experts. SETUP: 6 features, all-distinct validities [0.55, 0.90, 0.65, 0.95, 0.60, 0.80] (f0..f5). Distinctness strips the advocated model's free tie_break parameter of any role. ADVOCATED hierarchy (ascending validity): f0 (55%) -> f4 (60%) -> f2 (65%) -> f5 (80%) -> f1 (90%) -> f3 (95%). COMPETING hierarchy (w = val[::-1] = [0.80, 0.60, 0.95, 0.65, 0.90, 0.55], descending): f2 (65%) -> f4 (60%) -> f0 (55%) -> f3 (95%) -> f1 (90%) -> f5 (80%). Note the competing model's TOP cue is the 65% expert f2 and the advocated model's top cue is the 55% expert f0 — both models trust weak experts first, but DIFFERENT weak experts, and both consult the 95% expert late. The two hierarchies disagree on exactly the cue pairs (f0,f2), (f0,f4), (f2,f4), (f1,f3), (f1,f5), (f3,f5); f4 and f1 are 'fixed points' sitting at the SAME cascade depth in both hierarchies (depth 2 and depth 5 respectively). SIGNATURE 1 — THE PURE PERMUTATION DISSOCIATION GRID AT THE TALLY KNIFE-EDGE (T1-T12): twelve cells, one per dissociating cue pair per direction, each with EXACTLY two discriminating cues in opposition and all others tied (tally margin exactly 0). On every one of these cells the advocated model follows its earlier cue and the competing model follows the OPPOSITE cue, so their predicted intervals are DISJOINT CELL-BY-CELL: advocated p(A) in [0.54, 0.80] toward its decider vs competing p(A) in [0.20, 0.43] — a direction flip on all twelve cells (T1-T12 come in six exact A/B mirror pairs, ruling out side bias). Because every cell sits at margin 0, ANY margin-integrating strategy (e.g. Tallying) is pinned to EXACTLY 0.50 on all twelve — a parameter-free knife-edge — so any consistent preference on any cell simultaneously falsifies the entire integration family while the two one-reason theories are the only candidates predicting systematic deviation: a three-way separation no prior design combined with the permutation axis. SIGNATURE 2 — WEAK-VS-WEAK DICTATORSHIP (novel axis vs design [8], which pitted strong against weak experts): the advocated decider sweeps 55% (f0: T1-T4), 60% (f4: T5-T6), 80% (f5: T9-T12), 90% (f1: T7-T8); the competing decider sweeps 65% (f2: T1,T2,T5,T6), 95% (f3: T7,T8,T11,T12), 90% (f1: T9,T10). If subjects follow the 95% expert on T7/T11 or the 90% expert on T9, BOTH anti-validity theories fail — a shared falsification probe that design [8] could not mount, since there the advocated model itself followed the strong experts. SIGNATURE 3 — CASCADE-DEPTH INVARIANCE WITHIN EACH MODEL: the deciding rung moves from depth 1 (T1-T4 for advocated; T1,T2,T5,T6 for competing) through depths 4-5 (T7-T12), and each model predicts the SAME follow probability p = (1-eps)*sigmoid(beta)+eps/2 regardless of whether its decider is consulted first or fifth — testable flatness across the grid, with the mirror structure balancing depth across options. SIGNATURE 4 — COALITION CELLS WITH ROLE REVERSAL (T13, T14): both at tally margin +1 for A. On T13 the advocated top cue f0 leads a 3-vs-2 majority for A while the competing top cue f2 dissents to B; on T14 the roles reverse (f0 dissents to B against a 3-2 majority, f2 leads the majority to A). Each model gets exactly one majority-siding and one minority-siding coalition cell, so a 'follow-the-majority' heuristic cannot mimic either theory's profile; both models also predict p(T13) = p(T1) = p(T3) (same decider f0) and p(T14) = p(T5) (same decider f2) — context invariance across margin and coalition size. SIGNATURE 5 — FIXED-POINT AND TIE CONTROLS: T15 has only f4 discriminating, a cue at depth 2 in BOTH hierarchies — an agreement cell where the two theories concur in choice AND in cascade depth, anchoring the follow rate and testing both models' depth-invariance claims where they cannot differ; any deviation of p(T15) from the dissociation-cell follow rates falsifies the flat-follow assumption of whichever model mispredicts it. T16 is a full tie (both theories: exactly 0.50), anchoring lapse rate and side bias. GLOBAL PROFILES: the advocated model emits p(A) ≈ [0.54, 0.80] on T1, T3, T6, T7, T10, T11, T13, T15 and ≈ [0.20, 0.46] on T2, T4, T5, T8, T9, T12, T14, with 0.50 on T16; the competing model emits the EXACT TRANSPOSE — a near-perfect sign-flipped anti-correlation on 14 of 16 cells (agreement only on T15 and T16). Because the intervals are disjoint cell-by-cell on all twelve permutation cells, model comparison on the full 16-cell response profile separates the theories even after fitting beta, epsilon, and tie_break to each — and even the aggregate direction rate across the twelve mirror cells (advocated: its decider's direction ~70%; competing: the opposite direction ~70%) is diagnostic. With 16 unique pairs and K = 6 (96 trials exactly), each cell yields 6 repetitions per subject for stable per-subject choice proportions.

[10] GOAL: Quantitatively dissociate Distrust-Weighted Frugality with Reversal-Coherent Mixture (advocated) from Weakest-Expert-First Frugality (competing). The two theories are behaviorally IDENTICAL whenever the mixture mass m_am = ambiguity x gate(coherence) is ~0, because the advocated model's central component (w = -val) IS the competitor. The entire dissociation therefore rests on engineering the validity arrangement so that the advocated theory's own gate formulas open the anti-misbound channel: the subject population should be ~91% positional-reversal misbinders and only ~9% value-based anti-validity deciders, while the competitor asserts 100% value-based anti-validity deciders. SETUP: 6 features with validities [0.80, 0.55, 0.90, 0.55, 0.80, 0.65] — an anti-symmetric ('alternating') arrangement chosen so that (i) Spearman(position, validity) = -0.088, giving binding ambiguity = 0.912 (the validity order is essentially uncorrelated with display position, so the binding of accuracy to expert is maximally ambiguous), and (ii) |Spearman(validity, validity-reversed)| = 0.909, so the reversal-coherence gate sigmoid((0.909-0.45)/0.08) ~= 0.997 is fully OPEN. The advocated theory therefore commits, as a POINT PREDICTION with no free parameter, to m_am = 0.912 x 0.997 = 0.909: 90.9% of subjects adopt the anti-misbound hierarchy w = -val[::-1] and 9.1% the central w = -val. Crucially, the validities contain exact TIES (two 0.55s, two 0.80s): per the advocated theory's own tie-anchoring rule, per-cue zeta distortion is switched OFF (zeta_scale = 0), so its predictions are sharp rather than diffuse; and the trial set contains NO oppositions within any tied group ({f1,f3}, {f0,f4} for the value sort; {f2,f4}, {f1,f5} for the reversal), so the free tie_break parameter is inert in BOTH models — neither gains flexibility, and the theories differ only in hierarchy direction plus (beta, epsilon). THE TWO HIERARCHIES: competitor (ascending validity): {f1,f3} (55%) -> f5 (65%) -> {f0,f4} (80%) -> f2 (90%). Anti-misbound (w = -val[::-1]): {f2,f4} -> f0 -> {f1,f5} -> f3 — i.e., the misbinding subject consults the 90% and 80% experts FIRST and the 55% experts LAST, the exact transpose of the competitor's claim. Note this alignment of the reversal-coherent misbound hierarchy with descending validity is not an accident but a consequence the advocated theory itself forces: with high ambiguity, the gate can only open on a NEGATIVELY coherent (anti-symmetric) arrangement, under which trusting by mirror-position weakness equals trusting by own strength. The design therefore pits the competitor's core anti-validity claim (follow the WEAKEST-advertised discriminator) directly against the advocated claim (follow the misbound decider, which here is the STRONGEST-advertised discriminator). SIGNATURE 1 — DISJOINT PREDICTION INTERVALS ON 13 OF 16 CELLS: on every dissociating cell (T1-T13) the competitor's first discriminating cue and the anti-misbound first discriminating cue point to OPPOSITE options. Population-level, with follow rate p_f = (1-eps)*sigmoid(beta)+eps/2 in [0.59, 0.80] for both models: competitor predicts p(anti-misbound decider's option) = 1 - p_f in [0.20, 0.40]; advocated predicts 0.909*p_f + 0.091*(1-p_f) in [0.57, 0.74]. The intervals are DISJOINT, so any single dissociating cell with p > 0.45 toward the anti-misbound decider falsifies the competitor at any parameter setting, and p < 0.55 falsifies the advocated. T1-T12 are six exact A/B mirror pairs, ruling out side bias. SIGNATURE 2 — THE MIXTURE PROPORTION IS A PARAMETER-FREE POINT PREDICTION: agreement cells T14 (only the 55% expert f1 discriminates; both hierarchies follow it, at competitor depth 1 vs anti-misbound depth 4) and T15 (only the 65% expert f5 discriminates; depths 3 vs 4) pin the follow rate p_f; the dissociating cells then estimate the anti-misbound mass m via p_dissoc = m*p_f + (1-m)*(1-p_f). The advocated theory uniquely predicts m_hat ~= 0.91 FOR THIS ARRANGEMENT (a number derived entirely from its ambiguity/coherence gate formulas); the competitor predicts m_hat = 0. Observed m_hat > 0.45 falsifies the competitor; m_hat far below ~0.85 falsifies the advocated theory's gate calibration — a genuinely two-sided quantitative test. SIGNATURE 3 — PER-SUBJECT CONSISTENCY AND CLASSIFICATION: each subject holds ONE hierarchy, so choices across the 13 dissociating cells x 6 repetitions (78 trials) should be near-perfectly consistent with one of the two orders; the advocated predicts ~91%/9% split of subject types, the competitor 0%/100%. Per-trial likelihood ratio for an anti-misbound subject is (0.72/0.28)^78 — overwhelming model comparison even after fitting beta, epsilon, and tie_break to each model. SIGNATURE 4 — DECIDER-VALIDITY AND CASCADE-DEPTH INVARIANCE: across T1, T3, T5, T7, T9, T11, T13 the anti-misbound decider sweeps f2 (90%), f4 (80%), f0 (80%), f5 (65%) at cascade depths 1-3, and the competitor's decider sweeps f1/f3 (55%), f5 (65%), f0 (80%) at depths 1-4. Both one-reason theories predict a FLAT follow rate toward their own decider across all of these cells — any validity gradient or depth effect falsifies the one-reason claim of whichever model is otherwise supported. T3/T4 (90% vs 80% expert in conflict) is the strong-strong cell that removes any weak-vs-strong confound in interpreting the tilt; T9/T10 (65% vs 55%) is the weak-weak counterpart. SIGNATURE 5 — COALITION CELL (T13): the 90% expert alone favors A while the 55%, 55%, and 65% experts favor B (tally margin -2). The competitor follows a lone 55% expert against the 90% expert AND a 3-vs-1 majority; the advocated follows the 90% expert; any margin-integrating strategy says B — a three-way separation on one cell, with the one-reason models additionally predicting p(T13) = p(T1) (same deciders, different context). CONTROLS: T16 is a full tie (both theories: exactly 0.50) anchoring lapse rate and side bias. GLOBAL PROFILES: the competitor emits p(A) ~= 0.72 on T2, T4, T6, T8, T10, T12, T14 and ~= 0.28 on T1, T3, T5, T7, T9, T11, T13, T15, with 0.50 on T16; the advocated emits the near-TRANSPOSE on the 13 dissociating cells (~0.68 toward the misbound decider, slightly diluted below p_f by the 9% central mass — itself a quantitative signature: p_dissoc = 0.818*p_f + 0.091) while agreeing on T14/T15/T16. Because the advocated model's central component is exactly the competitor, the advocated nests the competitor; the experiment is therefore a clean one-sided falsification of the competitor (if the anti-misbound mass is real) plus a point test of the advocated gate (if it is not). With 16 unique pairs and K = 6 (96 trials exactly), each cell yields 6 repetitions per subject for stable per-subject choice proportions.

[11] GOAL: Quantitatively dissociate Weakest-Expert-First Frugality (advocated; value-based ascending-validity cascade) from Distrust-Weighted Frugality with Reversal-Coherent Mixture Heterogeneity (competing). The competitor NESTS my theory as its central component (w = -val), so a dissociation is only possible by engineering the validity arrangement so that the competitor's own gate formulas commit it to a large anti-misbound mass — then its population prediction differs from mine on every cell where the two hierarchies' first discriminating cues oppose. The design must also be novel relative to experiment [10] (which used a near-fully-coherent anti-symmetric arrangement where the anti-misbound hierarchy equaled descending validity, i.e. 'strongest-first'). Here the anti-misbound hierarchy is a genuinely SCRAMBLED, non-monotone-in-validity order, which (a) decouples the competitor's prediction from any strength heuristic, and (b) yields a different, precisely-computed gate commitment. SETUP: 6 features, validities [0.75, 0.55, 0.90, 0.65, 0.80, 0.55]. The exact ties (two 0.55s) deliberately switch the competitor's per-cue zeta distortion OFF (its own tie-anchoring rule: zeta_scale = 0 when validities are not all distinct), so its predictions are sharp, not diffuse; and the trial set contains NO oppositions within any tied group ({f1,f5} for the value sort, {f0,f4} for the positional reversal), so the free tie_break parameter is inert in BOTH models — neither gains flexibility, and the theories differ only in hierarchy direction plus (beta, epsilon). ARRANGEMENT STATISTICS (computed with the competitor's own formulas): Spearman(position, validity) = -0.145 -> binding ambiguity = 0.855; Spearman(validity, validity-reversed) = -0.618 -> coherence = 0.618; softened gate sigmoid((0.618-0.45)/0.08) = 0.890; hence m_am = 0.855 x 0.890 = 0.761. THE COMPETITOR'S POINT PREDICTION (no free parameter): 76.1% of subjects adopt the anti-misbound hierarchy w = -val[::-1], 23.9% the central w = -val. MY THEORY: 100% value-based anti-validity (weakest-advertised expert first). THE TWO HIERARCHIES: central (mine): {f1, f5} (55%) -> f3 (65%) -> f0 (75%) -> f4 (80%) -> f2 (90%). Anti-misbound: {f0, f4} -> f2 -> f5 -> f1 -> f3, i.e. in OWN advertised validity the misbinding cascade runs 75% -> 80% -> 90% -> 55% -> 55% -> 65%: a NON-MONOTONE schedule that is neither 'strongest-first' nor 'weakest-first' — the competitor's majority-subject behavior cannot be redescribed by any validity heuristic. SIGNATURE 1 — DIRECTION-FLIP DISSOCIATION GRID WITH DISJOINT PREDICTION INTERVALS (T1-T6, T11): six mirror-pair cells where the central decider (55% or 65% expert) opposes the anti-misbound decider (75%, 80%, or 90% expert), plus the coalition cell T11 (the lone 55% expert f1 for A against f0, f2, f4 for B; margin -2). My theory: p(central decider's option) = p_f = (1-eps)*sigmoid(beta)+eps/2 in [0.60, 0.80] over my parameter ranges. Competitor: p(central side) = 0.239*p_f + 0.761*(1-p_f) in [0.35, 0.42] over ITS ranges (p_f in [0.66, 0.79]; ~0.39 at its tightened-regime mean 0.72). THE INTERVALS ARE DISJOINT (gap >= 0.18), so any single dissociating cell at the population level falsifies one theory at any parameter setting; pooling all 11 dissociation cells (66 trials/subject) makes per-subject classification overwhelming (likelihood ratio ~ (p_f/(1-p_f))^66). SIGNATURE 2 — THE MIXTURE MASS IS A PARAMETER-FREE POINT PREDICTION: agreement cells T12-T15 (lone 55%, 80%, 90%, 65% expert) are followed by BOTH hierarchies at every cascade depth, so they pin p_f without knowing the mixture; the dissociation cells then estimate m_hat = (p_obs - (1-p_f))/(2*p_f - 1). My theory: m_hat = 0 exactly. Competitor: m_hat = 0.761, a number derived entirely from its ambiguity x coherence-gate formulas — a genuinely two-sided test of its softened-gate calibration (center 0.45, width 0.08): m_hat ~ 0 falsifies the gate; m_hat far from 0.76 miscalibrates it. SIGNATURE 3 — COMPETITOR-ISOLATED THREE-WAY CELLS (T7-T10): on T7 (f0: 75% -> A; f1: 55% and f2: 90% -> B) the anti-misbound decider is the 75% expert while the 90% expert, the 55% expert, AND the tally majority (margin -1) all point the other way. My theory, standard validity-respecting TTB, and Tallying ALL say B; ONLY the competitor's 76% majority says A (p(A) ~ 0.61). T8-T10 mirror and vary this (65%-vs-80% decider at T9/T10). If subjects show any validity sensitivity on these cells, the competitor fails where my theory survives; conversely T1-T6/T11 isolate MY anti-validity claim (I follow the 55-65% expert against the 75-90% expert and, on T11, against a 3-vs-1 majority) — the combination yields a full three-way separation (weakest-first vs strongest-first vs misbound-reversal) plus Tallying. SIGNATURE 4 — MARGIN-INVARIANCE EQUALITY BLOCKING INTEGRATION MIMICS: T1 (margin 0), T7 (margin -1), T11 (margin -2) all share decider pair (central f1 vs anti-misbound f0); both one-reason theories predict p(T1) = p(T11) and p(T7) = 1 - p(T1), whereas any margin-integrating strategy (e.g., Tallying) predicts a strictly ordered graded profile across them (0.5 / sigmoid(-beta) / sigmoid(-2beta)) — an internal check that the data belong to the one-reason family before adjudicating between its two members. SIGNATURE 5 — SHARED FLATNESS PROBES: (i) across T12-T15 the lone decider's advertised validity sweeps 55-90% and its cascade depth sweeps 1-5 in both hierarchies; both theories predict a FLAT follow rate — any monotone validity gradient or depth effect falsifies the one-reason core of whichever model is otherwise supported (a deliberate shared-risk probe); (ii) the competitor's tightened regime claims per-subject follow-rate SD ~ 0.02, whereas my parameter ranges span p_f in [0.60, 0.80] — the between-subject SD of follow rates on T12-T15 separately discriminates the two noise regimes. SIGNATURE 6 — PER-SUBJECT BIMODALITY: the competitor predicts a 76/24 split of subject types (anti-misbound vs central), each internally consistent across all 11 dissociation cells; my theory predicts a single central type. T16 (full tie) pins both theories to exactly 0.50, anchoring lapse and side bias; T1/T2, T3/T4, T5/T6, T7/T8, T9/T10 are exact A/B mirrors ruling out side bias. GLOBAL PREDICTED PROFILES: my theory emits p_f (~0.70) toward the WEAKEST-advertised discriminator's option on T1, T3, T5, T8, T10, T11, T12, T14 and (1-p_f) on T2, T4, T6, T7, T9, T13, T15, with 0.50 on T16; the competitor emits ~0.61 toward the anti-misbound decider's option on T2, T4, T6, T7, T9, T11 and ~0.39 on T1, T3, T5, T8, T10 — near-perfect anti-correlation on the 11 dissociation cells with disjoint parameter-robust intervals ([0.20, 0.40] vs [0.58, 0.65] toward the anti-misbound side), agreement on T12-T16. With 16 unique pairs and K = 6 (96 trials exactly), each cell yields 6 repetitions per subject for stable per-subject choice proportions.

[12] GOAL: Quantitatively dissociate Distrust-Weighted Frugality with Reversal-Coherent Mixture Heterogeneity (advocated) from Attenuated Frugality with Position-Validity Mixture Heterogeneity (competing). Both are one-reason models sharing the anti-validity central component, so the dissociation must come from their DIFFERENT non-central heterogeneity channels: the advocated adds ONLY an anti-misbound component (w = -val[::-1]) whose adoption is gated by m_am = ambiguity x sigmoid((coherence-0.45)/0.08); the competitor adds ONLY position-primacy (w = -position) and position-recency (w = +position) components whose weights grow LINEARLY in ambiguity, plus a small TTB residual — with NO misbound component and NO gate. The design engineers a validity arrangement that opens BOTH theories' non-central channels to their committed maxima, then pits their unique components head-to-head.

SETUP: 6 features, validities [0.65, 0.90, 0.55, 0.80, 0.55, 0.90]. Arrangement statistics (computed with both models' own formulas, average-tie ranks): Spearman(position, validity) = +0.088, so binding ambiguity = 0.912 — high enough that the competitor's position components dominate its mixture (normalized masses: primacy 0.573, anti-validity 0.353, recency 0.043, TTB 0.031) AND the advocated's gate product is large. Spearman(validity, validity-reversed) = -0.727, so reversal coherence = 0.727, gate = sigmoid((0.727-0.45)/0.08) = 0.970, and the advocated commits — as a PARAMETER-FREE POINT PREDICTION — to m_am = 0.912 x 0.970 = 0.884: 88.4% of subjects adopt the anti-misbound hierarchy, 11.6% the central anti-validity hierarchy, 0% primacy/recency/TTB. The competitor commits to 57.3% primacy, 35.3% anti-validity, 7.4% recency+TTB, and 0% misbound. The predicted mixture distributions have DISJOINT SUPPORTS on both the misbound and primacy masses. The validities contain exact ties (two 0.55s, two 0.90s), which switches per-cue zeta distortion OFF in BOTH models (each model's own tie-anchoring rule), making predictions sharp; and every trial keeps discriminating cues concordant within each tied group (features {f0,f2,f4} never internally oppose; {f1,f3,f5} never internally oppose), so the free tie_break parameter is inert in both models — neither gains flexibility, and the theories differ only in mixture composition plus (beta, epsilon).

THE FIVE HIERARCHIES under this arrangement: anti-validity: {f2,f4}(55%) -> f0(65%) -> f3(80%) -> {f1,f5}(90%); anti-misbound (w = -val[::-1]): {f1,f3} -> f5 -> f2 -> {f0,f4} (consulted validities 90, 80, 90, 55, 65, 55 — a genuinely scrambled, non-monotone order reducible to no validity heuristic); primacy: f0->f1->f2->f3->f4->f5; recency: f5->f4->f3->f2->f1->f0; TTB: {f1,f5} -> f3 -> f0 -> {f2,f4}. Because the strong experts (0.90, 0.90, 0.80) occupy ODD positions and the weak ones (0.55, 0.55, 0.65) EVEN positions, every trial with both groups discriminating in opposition pits the anti-misbound decider (odd group) against the anti-validity decider (even group), while the primacy and recency deciders are STEERED INDEPENDENTLY by choosing which specific features discriminate (primacy follows the earliest discriminating feature; recency the latest).

SIGNATURE 1 — PRIM-FACTOR FLATNESS (advocated) vs GRADED (competitor): trials T1, T3, T5, T7, T9, T11 all have misbound-decider -> A and anti-validity-decider -> B, but the primacy decider points to B on T1/T3/T5/T9 and to A on T7/T11, and the recency decider flips across T3/T5 (->A), T9 (->B), T11 (->B). The advocated theory has NO position components, so it predicts EXACTLY EQUAL p(A) = m_am*p_f + (1-m_am)*(1-p_f) ~= 0.67 on all six cells (p_f = (1-eps)*sigmoid(beta)+eps/2 in [0.66, 0.72] in its tightened regime). The competitor predicts 0.27 (T1), 0.30 (T3), 0.30 (T5), 0.57 (T7), 0.28 (T9), 0.55 (T11) — a ~0.30 spread driven by its 57% primacy mass. The paired contrasts p(T7)-p(T3) and p(T11)-p(T9) are +0.26 under the competitor and EXACTLY ZERO under the advocated; the mirror contrasts p(T8)-p(T4) and p(T12)-p(T10) are -0.26 vs zero. A flat-vs-graded profile on a factor (position-rule decider direction) that the advocated theory is structurally blind to is the central quantitative dissociation.

SIGNATURE 2 — THE GRAND DISSOCIATION CELL (T1/T2): on T1 the anti-misbound decider (f3, the 80% expert) favors A while ALL FOUR competitor components (anti-validity via f2, primacy via f0, recency via f5, TTB via f5) favor B. Advocated: p(A) ~= 0.67 [0.62, 0.72]; competitor: p(A) = 1-p_f ~= 0.27 [0.18, 0.40] — disjoint intervals on a single cell. T2 is the exact mirror (advocated 0.33, competitor 0.73). T1 additionally separates the misbound hierarchy from validity-respecting TTB (which follows f5 to B): if the majority behavior were standard TTB rather than misbinding, choices on T1 go to B. T1 also has tally margin -3, so any margin-integrating strategy (Tallying) predicts p(A) well below 0.5 there while the advocated predicts 0.67 — the orientation-1 cells sweep margins -3, 0, +1, +2, -1, 0 with a FLAT advocated prediction, simultaneously falsifying integration strategies.

SIGNATURE 3 — POINT-PREDICTED MIXTURE MASSES: agreement cells T13-T15 (all five hierarchies concur; both theories: p_f toward the common decider) pin the follow rate p_f without knowing the mixture; the 12 dissociation cells then estimate the population hierarchy masses. The advocated uniquely predicts mass(misbound) = 0.884 and mass(primacy) = 0 for THIS arrangement — numbers derived entirely from its own ambiguity x coherence-gate formulas; the competitor predicts mass(primacy) = 0.573 and mass(misbound) = 0. Observed misbound mass > 0.2 falsifies the competitor at any parameter setting; observed primacy mass > 0.1 falsifies the advocated — a genuinely two-sided quantitative test of each theory's heterogeneity structure.

SIGNATURE 4 — PER-SUBJECT TYPE CLASSIFICATION: both theories give each subject ONE hierarchy; the 12 dissociation cells x 6 repetitions (72 trials/subject) classify subjects nearly perfectly (per-trial likelihood ratio up to (0.72/0.28)^6 per cell). Advocated: ~88% misbound-consistent, ~12% anti-validity-consistent, 0% primacy-consistent. Competitor: ~57% primacy-consistent, ~35% anti-validity-consistent, ~7% recency/TTB, 0% misbound-consistent. A single primacy-consistent subject is evidence against the advocated; a single misbound-consistent subject (beyond the ~3% TTB confound, itself resolved by T1) is evidence against the competitor.

SIGNATURE 5 — SHARED FLATNESS PROBES: T3 vs T5 (anti-validity decider f2 at depth 1 vs f0 at depth 3; misbound decider f1 at 0.90 vs f3 at 0.80) and T1 vs T3 (misbound decider f3 vs f1): both theories predict EQUAL follow rates — any depth or decider-validity gradient falsifies the one-reason core of whichever model is otherwise supported. SIGNATURE 6 — NOISE REGIME: per-subject follow-rate SD on T13-T15 discriminates the advocated's tightened regime (~0.02) from the competitor's attenuated regime (~0.06). CONTROLS: T2/T4/T6/T8/T10/T12 are exact A/B mirrors of T1/T3/T5/T7/T9/T11, ruling out side bias; T16 is a full tie (both theories: exactly 0.50) anchoring lapse rate. GLOBAL PREDICTED PROFILES: the advocated emits a two-level profile FLAT within each orientation (~0.67 on T1,T3,T5,T7,T9,T11; ~0.33 on mirrors; p_f on T13,T15; 1-p_f on T14; 0.50 on T16); the competitor emits a graded profile within each orientation (0.27-0.57 and 0.43-0.73 respectively) driven by the primacy mass. The profiles are structurally disjoint — model comparison on the 16-cell response profile separates the theories even after fitting beta, epsilon, and tie_break to each. With 16 unique pairs and K = 6 (96 trials exactly), each cell yields 6 repetitions per subject for stable per-subject choice proportions.

[13] GOAL: Quantitatively dissociate Attenuated Frugality with Position-Validity Mixture Heterogeneity (advocated) from Distrust-Weighted Frugality with Reversal-Coherent Mixture Heterogeneity (competing). Both are one-reason models sharing the anti-validity central component, so the dissociation must come from their DIFFERENT non-central heterogeneity channels: the advocated adds position-primacy (w = -position) and position-recency (w = +position) components whose masses grow LINEARLY in binding ambiguity with NO coherence dependence; the competing adds ONLY an anti-misbound component (w = -val[::-1]) whose adoption is gated by m_am = ambiguity x sigmoid((coherence-0.45)/0.08). Prior designs [10]-[12] all operated in the HIGH-coherence regime where the gate is OPEN ([12], the same theory pair with roles swapped, used coherence 0.727 and exact validity ties). This design occupies the opposite, unexplored corner: MAXIMAL ambiguity with NEAR-ZERO reversal coherence. This directly tests the gate-vs-linear structural difference: the competitor claims non-central heterogeneity requires reversal-coherent list structure; the advocated claims position heterogeneity tracks ambiguity ALONE.

SETUP: 6 features, ALL-DISTINCT validities [0.55, 0.86, 0.78, 0.95, 0.62, 0.70] (f0..f5). Arrangement statistics (both models' own formulas, distinct ranks): Spearman(position, validity) = 3/35 = 0.086, so binding ambiguity = 0.914; Spearman(validity, validity-reversed) = 1/35 = 0.029, so the competitor's gate = sigmoid((0.029-0.45)/0.08) = 0.005, essentially SHUT. A methodological novelty: because BOTH models scale their per-cue zeta distortion by ambiguity x (coherence or gate), both zeta scales are negligible here (advocated <= 0.037, competitor <= 0.008, vs. minimum validity/position weight gaps of 0.07/1.0) - predictions are SHARP without needing exact validity ties, and the free tie_break is inert in both models (no weight ties).

PARAMETER-FREE MIXTURE COMMITMENTS: Advocated: w_prim = 0.80x0.914, w_anti = 0.45, w_rec = 0.06x0.914, w_ttb = 0.04 -> normalized masses PRIMACY 0.573, ANTI-VALIDITY 0.353, RECENCY 0.043, TTB 0.031, MISBOUND 0. Competitor: m_am = 0.914 x 0.005 = 0.0047 -> ANTI-VALIDITY 0.995, MISBOUND 0.005, POSITION COMPONENTS 0 (structurally absent). The predicted type distributions have disjoint supports on the primacy mass (0.573 vs 0).

THE FIVE HIERARCHIES: ANTI (ascending validity): f0(.55) -> f4(.62) -> f5(.70) -> f2(.78) -> f1(.86) -> f3(.95). PRIMACY: f0 -> f1 -> f2 -> f3 -> f4 -> f5. RECENCY: f5 -> f4 -> f3 -> f2 -> f1 -> f0. TTB: f3 -> f1 -> f2 -> f5 -> f4 -> f0. MISBOUND: f2 -> f4 -> f3 -> f0 -> f1 -> f5. Crucially, ANTI and PRIMACY share f0 as top cue, so every dissociation trial keeps f0 TIED, forcing consultation to depth >= 2 where the two orders diverge: primacy's second cue is the 2nd-listed 86% expert f1, anti-validity's is the 5th-listed 62% expert f4.

SIGNATURE 1 - DIRECTION-FLIP DISSOCIATION GRID WITH DISJOINT INTERVALS (T1-T10): on T1/T3/T5 the earliest discriminator (f1/f2/f3, positions 1-3, validities 0.78-0.95) favors A while the weakest discriminator f4 (position 5, 62%) favors B; on T7/T9 the weakest discriminator is f5 (position 6, 70%). Deciders on every one of these cells: PRIMACY and TTB -> the early/strong cue; ANTI-VALIDITY, RECENCY, and MISBOUND -> the weak/late cue. Advocated: p(A) = 0.604*p_f + 0.396*(1-p_f) in [0.532, 0.567] (above 0.5 for ANY parameters). Competitor: p(A) = 1 - p_f (+0.005 misbound tilt on T3/T7/T9) in [0.208, 0.344] (below 0.5 for ANY parameters). THE INTERVALS ARE DISJOINT AND ON OPPOSITE SIDES OF 0.5 ON ALL TEN CELLS: the sign of the population preference on any single cell falsifies one theory at any parameter setting. T2/T4/T6/T8/T10 are exact A/B mirrors ruling out side bias. The grid also sweeps primacy-decider depth (2, 3, 4) and validity (0.78-0.95) and anti-decider identity (f4 vs f5): both theories predict a FLAT follow rate across the sweep - a shared one-reason flatness probe.

SIGNATURE 2 - THE PRIMACY-FLIP PAIRED CONTRAST, EXACTLY ZERO VS STRONGLY POSITIVE (T1/T5 vs T11, T2/T6 vs T12): T11 (f4->B, f5->A) has the IDENTICAL anti-validity and misbound deciders as T1/T5 (anti f4->B, mis f4->B), so the competitor predicts p(T11) = p(T1) = p(T5) EXACTLY (both = 1 - p_f) - it is structurally blind to the fact that the primacy, recency, and TTB deciders all flipped. The advocated predicts p(T1) - p(T11) = 0.530*(2*p_f - 1) in [0.165, 0.341]. The paired within-subject contrast is 0 (competitor) vs >= 0.165 (advocated) - disjoint. Mirrors give the symmetric negative contrast. This flat-vs-graded profile on a factor the competitor cannot represent is the central quantitative dissociation.

SIGNATURE 3 - THE ANTI-FLIP SWING MAGNITUDE (T13 vs T1): T13 (f1->A, f3->B) holds the primacy decider fixed (f1->A) while flipping the anti-validity decider (f1->A instead of f4->B). Competitor swing p(T13)-p(T1) = 1.98*p_f - 0.995 in [0.306, 0.577]; advocated swing = 0.644*p_f - 0.322 in [0.100, 0.208]. DISJOINT INTERVALS: the anti-flip swing alone separates the theories. Combined with Signature 2, the two swing magnitudes form a double dissociation: the competitor demands anti-flip swing (>= 0.31) >> primacy-flip swing (= 0); the advocated demands primacy-flip swing (>= 0.165) > anti-flip swing (<= 0.208), reflecting its 57/35 primacy-vs-anti mass split against the competitor's 99.5/0.5.

SIGNATURE 4 - PER-SUBJECT TYPE CLASSIFICATION (the sharpest test): 11 dissociation cells (T1-T10, T15) x 6 repetitions = 66 trials/subject. The competitor predicts ~99.5% of subjects are anti-validity-consistent (always following the weakest-advertised discriminator, f4 or f5); the advocated predicts only ~35% anti-consistent, ~57% PRIMACY-consistent (always following the earliest-listed discriminator), ~7% recency/TTB. A single primacy-consistent subject is strong evidence against the competitor (which assigns position rules zero mass); a near-100% anti-consistent sample falsifies the advocated's point-predicted 0.573 primacy mass - a genuinely two-sided test of each theory's heterogeneity structure at a parameter-free commitment.

SIGNATURE 5 - TTB CONFOUND GUARDS AND CONTROLS: T13/T14 are agreement cells (both theories ~0.69-0.72 toward the early cue f1) where standard validity-respecting TTB follows the 95% expert f3 the OTHER way - if subjects were TTB, p(T13) ~ 0.28, falsifying both target theories while validating the anti-validity claim shared by both. T15 pits PRIMACY ALONE (f1->A) against ANTI + RECENCY + TTB + MISBOUND (f3/f4->B): advocated p(A) = 0.573*p_f + 0.427*(1-p_f) in [0.523, 0.547] vs competitor 1 - p_f in [0.208, 0.344] - disjoint, and simultaneously separating primacy-following from strong-expert-following. T16 (full tie) pins both theories to exactly 0.50, anchoring lapse and side bias. The noise regimes also differ: per-subject follow-rate SD on the agreement cells (T13/T14) is ~0.02 under the competitor's tightened regime vs ~0.06 under the advocated's attenuated regime.

GLOBAL PREDICTED PROFILES (p_f ~ 0.72): Advocated: p(A) ~ 0.546 on T1,T3,T5,T7,T9; ~0.454 on T2,T4,T6,T8,T10; 0.312 on T11/T14; 0.688 on T12/T13; 0.532 on T15; 0.500 on T16. Competitor: ~0.28 on T1,T3,T5,T7,T9,T11,T15; ~0.72 on T2,T4,T6,T8,T10,T12,T13; ~0.28 on T14; 0.500 on T16. The competitor emits a two-level step function constant across ALL cells sharing its (anti, misbound) decider pair - in particular p(T1)=p(T11)=p(T15) exactly - while the advocated emits a graded profile on those same cells driven by its primacy mass. Model comparison on the 16-cell response profile separates the theories even after fitting beta, epsilon, and tie_break to each. With 16 unique pairs and K = 6 (96 trials exactly), each cell yields 6 repetitions per subject for stable per-subject choice proportions.

[14] GOAL: Quantitatively dissociate Anti-Validity Frugality with Diffuse Positional-Salience Heterogeneity — momentary-diffusion variant (advocated) from Attenuated Frugality with Position-Validity Mixture Heterogeneity (competing). Both are one-reason models sharing the anti-validity core, so the dissociation must come from their OPPOSITE heterogeneity mechanics: the advocated adds a LARGE MOMENTARY per-trial diffusion (kappa*xi, re-drawn every trial, gated by A^3) that makes each subject's consultation order an essentially fresh random permutation on every trial; the competing adds a FIXED per-subject discrete draw from {primacy, anti, recency, TTB} with primacy mass 0.80*ambiguity. The design engineers a validity list that (i) fully opens the advocated's diffusion channel, (ii) keeps the competing's per-cue zeta too weak to blur its dominant primacy component, and (iii) makes the EARLIEST-listed expert (f0) ALSO the WEAKEST-advertised (0.55), so that whenever f0 discriminates the competing's two major components (primacy + anti, ~93% mass) CONCUR — while the advocated's diffusion is structurally blind to both position and the decider's identity, responding only to the NUMBER of discriminators on each side.

SETUP: 6 features, all-distinct validities [0.55, 0.95, 0.78, 0.86, 0.62, 0.70] (f0..f5). Arrangement statistics (both models' own formulas, average-tie ranks): Spearman(position, validity) = -0.029, so binding ambiguity = 0.971 and A^3 = 0.916 — the advocated's momentary term (kappa in [8,12] times A^3, i.e. +/-[7.3, 11] per cue per trial) fully dominates its stable part (-val <= 0.40, gamma*sal <= 0.06, sigma_h*zeta <= 1.1), so the consultation order is re-randomized essentially every trial. Reversal coherence = |Spearman(val, val[::-1])| = 0.20, so the competing's zeta scale = sigma_h*ambiguity*coherence <= 0.27: it CANNOT flip the primacy component's position gaps (1.0) or recency (1.0), leaving the competing's dominant 58.6% primacy mass fully sharp (it only mildly scrambles the competing's 34% anti mass, slightly diluting but never reversing its predictions). All-distinct validities make the free tie_break parameter inert in BOTH models (no weight ties), so neither gains flexibility; the theories differ only in heterogeneity structure plus (beta, epsilon). PARAMETER-FREE MIXTURE COMMITMENT of the competing at ambiguity 0.971: primacy 0.586, anti 0.340, recency 0.044, TTB 0.030. The advocated commits to NO discrete types at all: everyone is a momentary diffuser whose per-trial decider is ~uniform over the discriminating cues, tilted only slightly by the stable anti-validity core (tilt ~ gap/(kappa*A^3) ~ 0.03 per validity gap).

THE FIVE HIERARCHIES: ANTI: f0(.55)->f4(.62)->f5(.70)->f2(.78)->f3(.86)->f1(.95). PRIMACY: f0->f1->f2->f3->f4->f5. RECENCY: f5->f4->f3->f2->f1->f0. TTB: f1->f3->f2->f5->f4->f0. ADVOCATED: ~random permutation per trial with a small low-validity lean.

SIGNATURE 1 — CONCORDANCE LADDER, FLAT vs GRADED (T1-T3, mirrors T4-T6): f0 (earliest AND weakest) always favors one option while a growing coalition of later, stronger experts (including the strongest f1 and the latest f5, so the competing's recency and TTB components oppose) favors the other. The competing's decider structure is IDENTICAL on all three rungs (primacy->f0, anti->f0, recency->B-cue, TTB->B-cue), so it predicts EXACTLY FLAT p = 0.926*p_f + 0.074*(1-p_f) ~ 0.67 [0.54, 0.80] across T1, T2, T3 — a parameter-free equality. The advocated's momentary diffusion makes the first discriminating cue ~uniformly random among the k discriminators, so p(A) = P(A-cue first)*p_f + (1-P)*(1-p_f) tracks the DISCRIMINATOR COUNT: ~0.51 (T1, 1-vs-1), ~0.45 (T2, 1-vs-2), ~0.42 (T3, 1-vs-3) — strictly GRADED and below 0.5 where the count opposes f0. The paired contrast p(T1)-p(T3) is exactly 0 under the competing and ~ +0.09 under the advocated; on T2/T3 the population intervals are DISJOINT (advocated <= 0.48 vs competing >= 0.54). Mirrors T4-T6 give the symmetric pattern (advocated 0.49/0.55/0.57; competing flat 0.33), ruling out side bias. This inverts the classic one-reason-vs-tally signature: here the DISCRETE one-reason mixture is flat and the DIFFUSE one-reason model is graded by count.

SIGNATURE 2 — PRIMACY-ALONE CELLS WITH DISJOINT INTERVALS (T7/T8): on T7 the earliest discriminator f2 (78%) favors A while the weakest (f4), latest (f5), and strongest (f3) discriminators all favor B (count 1-vs-3). The competing follows its 58.6% primacy mass to A: p(A) ~ 0.55-0.62, ABOVE 0.5 for any parameters. The advocated follows the count: p(A) ~ 0.38 [0.33, 0.45], BELOW 0.5 — the two theories predict OPPOSITE SIGNS of preference on a single cell. T8 is the exact mirror (advocated 0.62, competing 0.45). These cells also estimate the primacy mass m_hat = (p_A-(1-p_f))/(2p_f-1): the competing commits to m ~ 0.586 for THIS arrangement (a number derived entirely from its own ambiguity formula); the advocated commits to m ~ 0 (its bounded gamma tilt, <= 0.06, can never flip a decider and is swamped by the diffusion anyway).

SIGNATURE 3 — COUNT-BALANCED STRUCTURAL CELLS (T13/T14): a 2-vs-2 cell (f0,f4 -> A vs f1,f5 -> B) where the advocated's count-based diffusion is pinned near its knife-edge (~0.51, since the count is balanced) while the competing's deciders (primacy f0, anti f0 vs recency f5, TTB f1) drive p(A) ~ 0.67. Advocated <= 0.53 vs competing >= 0.54 — disjoint at the population level; the mirror T14 gives the transpose.

SIGNATURE 4 — PER-SUBJECT STRUCTURE: THE SHARPEST TEST (T7-T12): on the opposition cells the earliest discriminator and the weakest discriminator point to OPPOSITE options (T9: f2->A vs f4->B; T11: f3->A vs f5->B; T7: f2->A vs f4->B within a 1-vs-3 count). The competing predicts each subject holds ONE fixed hierarchy and is CONSISTENT across all 6 repetitions and all 6 cells: ~59% of subjects are primacy-consistent (always following the earliest discriminator, per-cell p ~ 0.73), ~34% anti-consistent (always following the weakest, p ~ 0.27) — a BIMODAL population with between-subject SD of cell choice proportions ~ 0.22 and strong cross-cell consistency (a primacy subject picks the earliest discriminator's option on T7, T9, AND T11). The advocated predicts NO structural types: momentary re-drawing keeps every subject's per-cell proportion tight around the count-based value (SD ~ 0.10) and choices INDEPENDENT across cells (cross-cell consistency ~ chance). This is a double dissociation in DISPERSION STRUCTURE: the competing predicts between-subject spread exactly where the advocated predicts tightness — the advocated theory's own headline claim (diffuse attention keeps populations tight around chance-level conformity) is tested head-on. With 6 cells x 6 repetitions = 36 classification trials per subject, the per-subject likelihood ratio between 'fixed primacy type' and 'momentary diffuser' is overwhelming.

SIGNATURE 5 — WITHIN-SUBJECT REPETITION FLIP RATE: on opposition cells the competing predicts flip rate ~2*p_f*(1-p_f) ~ 0.40 between repetitions of the same cell; the advocated predicts ~0.47-0.50 (per-trial p near 0.5). Pooled over cells this is estimable per subject.

CONTROLS: T15 (single discriminator f4->A) is an agreement cell where ALL hierarchies concur — both theories predict exactly p_f toward A, pinning the follow rate and lapse without knowing the mixture; any deviation of the dissociation-cell follow rates from p(T15) falsifies the flat-follow assumption of whichever model is otherwise supported. T16 (full tie) pins both theories to exactly 0.50, anchoring lapse and side bias. T4-T6, T8, T10, T12, T14 are exact A/B mirrors ruling out side bias.

GLOBAL PREDICTED PROFILES (population means): ADVOCATED: T1 0.51, T2 0.45, T3 0.42, T4 0.49, T5 0.55, T6 0.57, T7 0.38, T8 0.62, T9 0.49, T10 0.51, T11 0.49, T12 0.51, T13 0.51, T14 0.49, T15 ~0.71, T16 0.50 — a profile ordered by discriminator COUNT (with a small anti-validity leak toward the weakest discriminator), tight across subjects. COMPETING: T1 = T2 = T3 = T13 ~ 0.67 (EXACTLY equal — parameter-free), T4 = T5 = T6 = T14 ~ 0.33, T7 ~ 0.58, T8 ~ 0.42, T9 = T11 ~ 0.60, T10 = T12 ~ 0.40, T15 ~ 0.73, T16 0.50 — a decider-structured step function, bimodal across subjects. The profiles are structurally disjoint (e.g., competing says p(T1) = p(T3); advocated says p(T1) - p(T3) ~ +0.09; competing says p(T7) > 0.5; advocated says p(T7) < 0.45), so model comparison on the 16-cell response profile — plus the between-subject dispersion and within-subject consistency statistics — separates the theories even after fitting beta, epsilon, sigma_h, zeta, kappa, gamma, comp, and tie_break to each. The test is genuinely two-sided: a flat ladder plus primacy-consistent subject types falsifies the momentary-diffusion claim; a count-graded ladder plus a typeless, tight population falsifies the discrete position-mixture claim. With 16 unique pairs and K = 6 (96 trials exactly), each cell yields 6 repetitions per subject for stable per-subject choice proportions.

[15] GOAL: Quantitatively dissociate Attenuated Frugality with Position-Validity Mixture Heterogeneity (advocated) from Anti-Validity Frugality with Diffuse Positional-Salience Heterogeneity, momentary-diffusion variant (competing). Both are one-reason models sharing the anti-validity core, so the dissociation must come from how their NON-central heterogeneity scales with binding ambiguity: the advocated enters ambiguity LINEARLY (primacy mass = 0.80*A/(0.49+0.86*A)), while the competing enters it CUBED (momentary diffusion amplitude = kappa*A^3) with a bounded gamma tilt that 'can flip nothing on anchored designs'. Experiment [14] tested the same theory pair only in the HIGH-ambiguity corner (A = 0.971, full diffusion vs full primacy). This design occupies the opposite, unexplored corner: MODERATE-LOW ambiguity A = 0.286, chosen as the point of maximal structural divergence — the advocated's linear term already commits 31.1% of subjects to pure position-primacy hierarchies, while the competing's own cubic gradient (A^3 = 0.023, exactly the 'anchored' regime its description cites) commits it to ~0% position-following: everyone is an anchored anti-validity decider with a bounded tilt that mathematically cannot flip any decider in this design.

SETUP: 6 features, all-distinct validities [0.80, 0.50, 0.85, 0.60, 0.90, 0.95] (f0..f5). Arrangement statistics (both models' own formulas): validity ranks by position [2, 0, 3, 1, 4, 5], Spearman(position, validity) = +0.714, so binding ambiguity A = 0.286; reversal coherence = 0.657. ADVOCATED mixture commitment (parameter-free, from its own formulas): anti-validity 0.612, position-primacy 0.311, TTB 0.054, recency 0.023. COMPETING commitment: A^3 = 0.023, so its momentary term kappa*A^3 <= 0.28 and stable term <= 0.028 are BELOW every cross-block validity gap used in the design (f1-f0 = 0.30, f1-f5 = 0.45, f2-f3 = 0.25, f0-f3 = 0.20), and its gamma tilt (max 0.12 * salience-gap, i.e. <= 0.024 for adjacent positions, <= 0.12 for extremes vs the 0.15 minimum strong-block gap) can flip NO decider: the competing is EXACTLY anchored — its decider is always the weakest-advertised discriminator, with flip probability < 1% on every trial. The advocated's own zeta blur (scale <= sigma_h*A*coherence = [0.075, 0.262]) never flips primacy/recency types (position gaps 1.0) and flips anti types on the key f0-vs-f1 cells only ~2-5% at typical sigma_h; all-distinct validities make tie_break inert in both models. DESIGN RULES: the weak block {f1, f3} and the strong block {f0, f2, f4, f5} are never internally opposed (small intra-block gaps would blur both models), except the f0/f5 oppositions in T5-T10 where only the 5.4% TTB mass is mildly blurred (~1.7% population effect); all dissociations pit the blocks against each other across large gaps.

THE FOUR HIERARCHIES: ANTI: f1(.50)->f3(.60)->f0(.80)->f2(.85)->f4(.90)->f5(.95). PRIMACY: f0->f1->f2->f3->f4->f5. RECENCY: f5->f4->f3->f2->f1->f0. TTB: f5->f4->f2->f0->f3->f1. The earliest-listed expert f0 (0.80) and the second-listed weakest expert f1 (0.50) are the core dissociating pair; f2 (pos 2, 0.85) vs f3 (pos 3, 0.60) provides an independent replication pair.

SIGNATURE 1 — THE PRIMACY-MASS KNIFE EDGE (parameter-free equality vs graded swing): T5 (f0->A, f1->B, f5->B) and T6 (f0->B, f1->B, f5->B) have the IDENTICAL competing decider (f1->B) — the competing predicts p(T5) = p(T6) EXACTLY, for any (beta, epsilon, gamma, sigma_h, kappa, zeta): the earliest discriminator's direction is structurally irrelevant to it. The advocated predicts p(T5) - p(T6) = m_prim*(2p_f-1) ~= +0.13 [0.06, 0.24] — on T5 its 31.1% primacy mass follows f0->A against the anti, recency, AND TTB components, against the strongest expert (0.95), and against the tally majority (margin -1). Mirrors: p(T8) - p(T9) = -0.13 (advocated) vs 0 (competing); also p(T1) = p(T6) exactly under the competing (same decider f1->B) vs a +0.16 gap under the advocated. Any reliable deviation from these equalities falsifies the competing's A^3 anchoring claim at any parameter setting; near-zero deviations falsify the advocated's point-predicted 0.311 primacy mass.

SIGNATURE 2 — THE SWING-RATIO POINT PREDICTION: the anti-flip contrast T7 - T5 (identical except f1's direction) measures the anti mass: advocated m_anti*(2p_f-1) ~= 0.26, competing (2p_f-1) ~= 0.42. Combined with Signature 1, the ratio of swings primacy-swing/anti-swing = m_prim/m_anti = 0.8A/0.45 = 0.509 is a PARAMETER-FREE point prediction of the advocated for THIS arrangement (derived entirely from its own mixture formula at A = 0.286), while the competing commits to a ratio of ~0. The two ratios (0.51 vs 0.00) are disjoint — a genuinely two-sided quantitative test of the linear-vs-cubic gradient assumption.

SIGNATURE 3 — GLOBAL PROFILE SHAPE: the competing emits a TWO-LEVEL STEP FUNCTION over the 16 cells: p_f (~0.71) on T2, T4, T7, T8, T9, T11, T13, T15 and 1-p_f (~0.29) on T1, T3, T5, T6, T10, T12, T14, exactly 0.5 on T16 — invariant to margin, coalition size, decider identity, and the earliest/strongest discriminators' directions. The advocated emits a MULTI-LEVEL profile (~0.42 on T5; ~0.44 on T1/T3/T12/T14; ~0.31 on T10; ~0.28 on T6; ~0.58 on T8; ~0.56 on T2/T4/T13/T15; ~0.69 on T7; p_f on T9/T11; 0.5 on T16), pulled toward 0.5 on every conflict cell by its 36.5% primacy+TTB mass. In particular the competing demands p(T5)=p(T1)=p(T6) and p(T8)=p(T9)=p(T7); the advocated demands strict orderings within each set. Model comparison on the 16-cell profile separates the theories even after fitting all free parameters to each.

SIGNATURE 4 — PER-SUBJECT TYPE CLASSIFICATION (sharpest test): 12 conflict cells x 6 repetitions = 72 classification trials per subject. The advocated predicts ~31% PRIMACY-consistent subjects (follow the earliest discriminator: A on T1/T3/T5/T7, B on T8/T10), ~61% anti-consistent (follow the weakest discriminator), ~5% TTB, ~2% recency. The competing predicts ~0% primacy-consistent subjects — its bounded gamma tilt and <1% diffusion cannot produce a subject who reliably follows f0 against f1+f5 on T5. T5/T8/T10 also cleanly separate primacy types from standard validity-respecting TTB strategists (a TTB strategist follows the 0.95 expert: B on T5, A on T8/T10 — the opposite of a primacy type). Observed primacy-consistent fraction ~0.3 supports the advocated's linear-ambiguity mixture; ~0 falsifies its 0.311 point commitment; > 0.1 falsifies the competing outright.

SIGNATURE 5 — SHARED FLATNESS + THREE-WAY SEPARATION FROM INTEGRATION: both theories predict p(T1) = p(T12) = p(T14) (same decider structure, margins 0/-1/0, anti decider at depth 1 vs depth 2) — any margin gradient falsifies the one-reason core of whichever model is otherwise supported, and Tallying (which predicts 0.5 / <0.5 / 0.5 across them) is separately falsified by any consistent preference on the margin-0 conflict cells. T11 (only f1 discriminates) and T9 (unanimous A) pin the follow rate p_f without knowing the mixture; T16 (full tie) anchors lapse and side bias. T2/T4/T8/T10/T13/T15 are exact A/B mirrors ruling out side bias.

HONESTY NOTE: on single cells the population prediction intervals overlap after parameter heterogeneity (both theories' major mass is anti-validity, and both mean p_f ~ 0.71-0.73); the decisive evidence is the paired within-subject contrasts (Signature 1-2), the profile shape (Signature 3), and the between-subject type distribution (Signature 4) — all robust to fitting beta, epsilon, gamma, sigma_h, kappa, zeta, comp, and tie_break to each theory. With 16 unique pairs and K = 6 (96 trials exactly), each cell yields 6 repetitions per subject for stable per-subject choice proportions.

[16] GOAL: Quantitatively dissociate Anti-Validity Frugality with Diffuse Positional-Salience Heterogeneity, momentary-diffusion variant (advocated) from Anchored Anti-Validity Frugality with Validity-Respecting Leak, frozen-discipline variant (competing). The two theories share the anti-validity one-reason core, the same A^3 ambiguity gate, comparable diffusion magnitudes, and the same tie-anchoring rule — so diffusion-structure axes cannot cleanly separate them (at any ambiguity where the advocated diffuses, the competing's gap-scaled noise is of the same order). The ONE structural mechanism with diverging, design-forced commitments is the competing's TRIAL-CONDITIONAL VALIDITY-RESPECTING LEAK: a probability mixture (ceiling 0.18) toward the standard TTB (descending-validity) decider that fires ONLY when the anti-validity cascade's first discriminating expert carries a strictly INTERIOR validity (min+0.10 < val < max-0.10). The advocated has NO such mechanism: in the anchored regime its follow probability is p_f toward the anti-decider on EVERY discriminating trial, flat over decider validity, decider position, margin, and coalition size. The design therefore engineers the anchored regime and maximizes the competing's committed leak strength, then tests a flat-vs-dipped follow profile.

SETUP: 6 features with validities [0.50, 0.50, 0.75, 0.75, 1.00, 1.00] (f0..f5), monotone ascending with exact ties. Consequences (both models' own formulas, average-tie ranks): Spearman(position, validity) = 0.956, so binding ambiguity A = 0.044 and A^3 = 8.4e-5 — both theories' distortion channels are switched off entirely (their own rules: distortion requires all-distinct validities AND ambiguity > 0; ties anchor the distrust order in both). Every within-tied-pair gap (0.25) dwarfs both theories' gamma*sal tilts (advocated <= 0.072, competing <= 0.018), so the consultation order is EXACTLY the anti-validity cascade {f0,f1}(50%) -> {f2,f3}(75%) -> {f4,f5}(100%) in both models, with tie_break inert (tied cues never oppose in any trial). Reversal coherence = 0.956; mean_gap over unique validities = 0.25 — the MAXIMUM achievable while keeping an interior validity (three unique values spanning [0.5, 1.0]). Hence the competing's leak probability q = min(rho_leak * 0.956 * 0.25, 0.18) is forced into [0.120, 0.18] for ANY rho_leak in [0.5, 3.0]: the competing CANNOT switch its leak off in this design. Its p_f = (1-eps)*sigmoid(beta*(1+boost*0.956)) + eps/2 lies in [0.708, 0.754] (distinct-design bonus inert because validities are tied). The advocated's p_f lies in [0.59, 0.84] (beta in [0.5,1.9], eps in [0.08,0.24]).

CELL FAMILIES. (L) LEAK CELLS (T1-T8, four exact A/B mirror pairs): the anti-decider is an interior 0.75 cue and a 100% expert discriminates in OPPOSITION (all 50% cues tied, so the carrier is unambiguous). T1/T2: f2 vs f4; T3/T4: f3 vs f5 (position-balanced replicate); T5/T6: both 0.75s vs both 100%s (tally margin 0); T7/T8: one 0.75 vs both 100%s (margin -1). Advocated: p(carrier's option) = p_f on ALL eight. Competing: p(carrier's option) = (1-q)*p_f + q*(1-p_f) in [0.633, 0.693] — a DIP of q*(2p_f-1) in [0.050, 0.093] relative to its own no-leak follow rate. (C) CONTROL CELLS (T9-T16): T9/T10 — interior 0.75 carrier with the 100% expert CONCURRING (leak eligible but winner_ttb = winner_anti, so no mixture): both theories p_f; this matches decider validity with the leak cells, isolating TTB opposition as the dip's cause. T11/T12 — WEAKEST carrier (50%) opposed by the 100% expert: leak OFF (carrier not interior); both theories p_f toward the 50% expert. T13/T14 — weakest carrier (50%) opposed by the interior 0.75 expert: leak OFF; a generic always-on TTB mixture would dip here too, but the competing's trial-conditional gate does not — this cell separates the leak from any unconditional validity-respecting contamination. T15 — lone interior decider (only f2 discriminates; TTB concurs): p_f. T16 — full tie: exactly 0.50 in both, anchoring lapse and side bias.

SIGNATURE 1 — THE LEAK DIP (central, two-sided, parameter-forced): define each subject's carrier-follow rate (choices matching the anti-validity decider's option). Advocated: EXACTLY EQUAL follow rates on leak and control cells (both are p_f; no mechanism in the advocated depends on the decider's validity rank, its position, or the opposition's validity). Competing: follow rate on the eight leak cells is lower by q*(2p_f-1) in [0.050, 0.093] — bounded AWAY FROM ZERO by its own gate formulas (q >= 0.120 is forced by rho_leak >= 0.5, coherence 0.956, mean_gap 0.25). The prediction intervals on this within-subject paired contrast are DISJOINT: [0, 0] vs [0.050, 0.093]. With 48 leak trials and 42 control trials per subject, a population dip estimate of ~0 (CI within +/-0.03) falsifies the competing at any parameter setting; a dip >= 0.05 falsifies the advocated, which cannot produce it.

SIGNATURE 2 — THE CONDITIONALITY TRIAD (specificity of the dip): the competing predicts the dip ONLY on the conjunction (interior carrier AND top-validity opposition): T1 vs T9/T15 (interior carrier, opposition removed -> no dip), T1 vs T11 (opposition retained, carrier demoted to weakest -> no dip), T1 vs T13 (higher-validity opposition retained, carrier weakest -> no dip). The advocated predicts no dip in any cell. An unconditional partial-TTB mixture (the natural simpler competitor) dips on T11 AND T13 as well — so the triad also protects the dissociation from being absorbed by a less conditional theory.

SIGNATURE 3 — FLAT PROFILE OVER MARGIN AND COALITION (shared one-reason core + integration kill): across T1 (margin 0, 1v1), T5 (margin 0, 2v2), T7 (margin -1) both theories predict IDENTICAL carrier-follow rates — no margin effect, no coalition-size effect. Tallying is pinned to EXACTLY 0.50 on T5/T6 (margin 0) and below 0.5 on T7, while both one-reason theories predict ~p_f toward the 0.75-cue option — any consistent preference on T5/T6 falsifies the entire integration family, and any reliable margin gradient falsifies the one-reason core of whichever target theory is otherwise supported.

SIGNATURE 4 — NOISE-REGIME DISPERSION (secondary): on the no-leak cells the advocated's wide (beta, epsilon) ranges produce a between-subject SD of follow rates ~0.06-0.10, while the competing's frozen discipline ([0.80,1.00]x[0.28,0.42]x[0.12,0.16]) produces ~0.02 true SD (observed ~binomial floor). The between-subject dispersion of T9/T11/T13/T15 follow rates separately tests the two noise regimes.

HONESTY NOTE: single-cell population intervals overlap (the advocated's p_f reaches 0.84; the competing's leak-cell floor is 0.633), so no single cell is decisive; the decisive statistics are the within-subject paired contrasts (Signatures 1-2), the 16-cell profile shape, and the dispersion test — all robust to fitting beta, epsilon, gamma, sigma_h, kappa, zeta, rho_leak, boost, and tie_break to each theory (gamma, sigma_h, kappa, zeta, and tie_break are provably inert in this design: distortion is gated off, and no tilt can flip a 0.25 validity gap). The test is genuinely two-sided: a flat follow profile falsifies the competing's forced q >= 0.120 leak; a dip >= 0.05 falsifies the advocated's leak-free flatness claim. 16 unique pairs, K = 6 (96 trials exactly), give 6 repetitions per cell for stable per-subject choice proportions.

[17] GOAL: Quantitatively dissociate Anchored Anti-Validity Frugality with Validity-Respecting Leak, frozen-discipline variant (advocated) from Anti-Validity Frugality with Diffuse Positional-Salience Heterogeneity, momentary-diffusion variant (competing). Both share the anti-validity one-reason core, the same A^3-gated distortion rule, and the same tie-anchoring rule, and experiment [16] already exploited the one mechanism the advocated uniquely has (the trial-conditional TTB leak) in a tied-validity, ambiguity-0.044 regime. This design occupies a different, unexplored corner: the ZERO-AMBIGUITY, ALL-DISTINCT regime in which every gated channel (momentary diffusion, stable zeta, gap-proportional anchoring, leak on weak-block deciders) is STRUCTURALLY OFF in both theories, leaving exactly one active difference: the UNGATED positional-salience tilt gamma*salience, which both theories apply unconditionally but with committed different magnitudes — the competing draws gamma asymmetrically from [-0.05, +0.12] (a real, population-level early-position lean), while the advocated's gamma is confined to [-0.01, +0.03] (effectively zero). The design converts this tilt-magnitude difference into a measurable VALIDITY-GAP PSYCHOMETRIC.

SETUP: 6 features, validities [0.95, 0.78, 0.534, 0.53, 0.52, 0.50] (f0..f5), strictly DECREASING in position, so Spearman(position, validity) = -1, binding ambiguity = 0 exactly: both theories' `distinct and ambiguity > 0` gates shut all distortion off, and all validities distinct makes tie_break inert in both. The consultation order is therefore deterministic per subject: w_j = -val_j + gamma*sal_j with sal = [+0.5, +0.3, +0.1, -0.1, -0.3, -0.5]. The anti-validity core consults f5 (50%) -> f4 (52%) -> f3 (53%) -> f2 (53.4%) -> f1 (78%) -> f0 (95%). KEY MECHANIC: for any two cues X (earlier position, higher validity) and Y (later, weaker), the subject consults X before Y iff gamma > Delta_v / Delta_sal =: g. So on a trial where ONLY X and Y discriminate in opposition, the choice reveals a threshold crossing of the subject's gamma. The weak block {f2..f5} is engineered with tiny, varied gaps so that different cue pairs realize a LADDER of thresholds g = 0.020, 0.035, 0.050, 0.057, 0.075, 0.100 — spanning the competing's gamma range while sitting (except the lowest rung) entirely ABOVE the advocated's gamma cap of 0.03.

PREDICTED FLIP RATES P(consult the earlier cue first): competing, gamma ~ Uniform[-0.05, 0.12]: P(gamma > g) = (0.12 - g)/0.17 = 0.588, 0.500, 0.412, 0.372, 0.265, 0.118 across the six rungs. Advocated, gamma ~ Uniform[-0.01, 0.03]: 0.25 at g = 0.020 and EXACTLY ZERO at every rung with g >= 0.035 — RANGE-FORCED, since gamma <= 0.03 < 0.035 for any admissible parameter setting. Population p(earlier-cue option), p_E = P_flip*p_f + (1-P_flip)*(1-p_f): competing (p_f mean 0.71) predicts 0.537, 0.500, 0.463, 0.446, 0.401, 0.340; advocated (p_f = (1-eps)*sigmoid(beta*(1+boost)*(1+b_distinct)) + eps/2 in [0.678, 0.832] at ambiguity 0 with the distinct-design bonus) predicts 0.334-0.411 at g = 0.020 and a FLAT 1 - p_f in [0.168, 0.322] on all five higher rungs.

SIGNATURE 1 — THE GAP PSYCHOMETRIC (primary): the competing predicts a strictly GRADED p_E(g), falling ~0.20 from the lowest to the highest rung (0.537 -> 0.340); the advocated predicts a FLAT profile at 1 - p_f across g = 0.035...0.100 (range-forced zero flips), with at most a single small step at g = 0.020. The paired within-subject contrast p_E(g=0.035) - p_E(g=0.100) is ~ +0.16 under the competing and EXACTLY ZERO under the advocated. Mirror pairs (T1/T2, T3/T4, T5/T6, T9/T10) cancel side bias; p_E is pooled over each mirror pair (12 trials/subject per rung).

SIGNATURE 2 — THE MIDPOINT KNIFE EDGE (T3/T4, g = 0.035, the exact midpoint of the competing's gamma interval): the competing predicts p_E = 0.500 EXACTLY, for ANY (beta, epsilon) — with P_flip = 0.5, p_E = 0.5*p_f + 0.5*(1-p_f) = 0.5, a parameter-free point prediction. The advocated is range-forced to p_E = 1 - p_f <= 0.322. The intervals [0.500] and [0.168, 0.322] are DISJOINT: population p_E > 0.42 on this rung falsifies the advocated at any parameter setting; p_E < 0.40 falsifies the competing's asymmetric-tilt prior.

SIGNATURE 3 — PER-SUBJECT STEP FUNCTION AND DISPERSION (sharpest structural test): each competing subject holds ONE gamma, so choices across the six rungs trace a STEP FUNCTION in g (follow the earlier cue on rungs below gamma-hat, the weaker cue above it); ~50% of subjects have gamma-hat inside (0.035, 0.12] and thus flip somewhere on the ladder, and at g = 0.035 the population is BIMODAL (half at p_f ~ 0.71, half at 1 - p_f ~ 0.29; between-subject SD ~ 0.21). The advocated predicts ZERO subjects flip on any g >= 0.035 rung — every subject follows the weaker cue, tight cluster at 1 - p_f (true SD ~ 0.03-0.04). A single subject following the earlier cue consistently (beyond binomial noise, P ~ 0.09 per 6-rep cell at worst-case p_f) on any g >= 0.035 cell falsifies the advocated outright; a near-unimodal low cluster falsifies the competing's committed gamma distribution. The design thus MEASURES the population gamma distribution directly, making the competing's prior-based commitment empirically testable rather than assumable.

SIGNATURE 4 — THE ADVOCATED'S OWN GAMMA CAP (T1/T2, g = 0.020): the advocated predicts a 25% flip rate there (p_E ~ 0.375), the competing 58.8% (p_E ~ 0.537) — disjoint intervals [0.334, 0.411] vs [0.50, 0.57], a genuinely two-sided test of each theory's tilt prior at its left end.

SIGNATURE 5 — SECONDARY LEAK PROBE (C3 vs C4): C3 pits the interior-validity 78% expert (the stable anti-decider, j* interior since 0.60 < 0.78 < 0.85) against the 95% expert in opposition; the advocated's trial-conditional leak fires with q = min(rho_leak * 1.00 * 0.090, 0.18) in [0.045, 0.18] (forced by rho_leak >= 0.5, coherence = 1, mean_gap = 0.090), predicting a within-subject dip q*(2*p_f - 1) in [0.016, 0.120] relative to the matched concordance cell C4 (same decider, TTB opposition removed, no mixture). The competing predicts EXACTLY ZERO dip. This axis is weak here by construction (the tilt design requires tiny gaps, which shrink mean_gap), so it is secondary; [16] tested the leak at full strength. All tilt-cell and control-cell deciders are weak-block or min/max-validity cues, so the leak is provably OFF on T1-T10, C1, C2, C5 — the tilt signatures are uncontaminated.

SIGNATURE 6 — CONTROLS: C1/C5 (lone weakest / lone block decider) are agreement cells pinning p_f for both theories and separating the noise regimes (advocated frozen-narrow p_f in [0.678, 0.832], between-subject SD ~ 0.03; competing p_f in [0.51, 0.92], SD ~ 0.08). C2 (weakest 50% expert vs the 95% expert, in opposition) tests the SHARED anti-validity core: both theories follow the 50% expert at p_f; if subjects instead follow the 95% expert, both target theories fail together. C6 (full tie) pins both to exactly 0.50, anchoring lapse and side bias.

HONESTY NOTES: (i) Within every critical pair the earlier-position cue is also the higher-validity cue — unavoidable, since zero ambiguity (Spearman = -1) forces position-validity alignment; a pure validity-respecting strategist would also produce the competing's tilt-side predictions, so elevated p_E falsifies the advocated without uniquely confirming the competing (C2 partially guards the shared anti-validity core). (ii) The competing's flip commitment is prior-based (its asymmetric gamma draw), not range-forced: a post-hoc refit with gamma ~ 0 for every subject could mimic the flat profile, but only by contradicting the competing's own stated population prior (mean +0.035, 41% of subjects above 0.05, 50% above 0.035) — the ladder measures that distribution directly, so the concession would be visible and quantified. (iii) Single-cell intervals overlap at the high-g rungs (g = 0.075, 0.100); the decisive evidence is the low-g rungs, the psychometric slope, and the per-subject step/dispersion structure — all robust to fitting beta, epsilon, gamma, sigma_h, kappa, zeta, rho_leak, boost, b_distinct, and tie_break to each theory (sigma_h, kappa, zeta, tie_break are provably inert at ambiguity 0 with distinct validities). (iv) The g = 0.035 knife edge assumes the competing's gamma draw is uniform on [-0.05, +0.12]; under any prior with appreciable mass in the upper half of that interval the prediction remains well above the advocated's forced ceiling. With 16 unique pairs and K = 6 (96 trials exactly), each cell yields 6 repetitions per subject for stable choice proportions.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "GOAL: Quantitatively dissociate Anti-Validity Frugality with Diffuse Positional-Salience Heterogeneity, momentary-diffusion variant (advocated) from Diffuse Mixture Frugality with Strong Interior-Gated Validity Leak, sub-linear coherence variant (competing). The two theories share EVERYTHING in their core machinery: the anti-validity one-reason cascade, the identical A^3-gated diffuse heterogeneity (same sigma_h, kappa, zeta, xi structure), the identical gamma tilt range [-0.05, +0.12], the identical tie-anchoring rule, and heavily overlapping (beta, epsilon) ranges. Diffusion-structure axes therefore cannot separate them. The competing's ONLY unique mechanisms are (a) the INTERIOR-GATED VALIDITY LEAK (mixes toward the TTB decider with q = rho_leak*sqrt(coherence)*gap_term when the anti carrier's validity is strictly interior) and (b) the TIE-ANCHORED PRIMACY GATE (mixes toward the reading-order decider with weight w_eff only when validities contain exact ties AND ambiguity >= 0.80). The advocated has neither. The design therefore engineers a validity list that (i) anchors BOTH theories onto the identical deterministic anti-validity cascade (shutting off all shared distortion) while (ii) forcing BOTH of the competing's unique mechanisms fully ON at committed strengths, then measures the within-subject 'dip profile' the advocated is structurally forbidden to produce.\n\nSETUP: 6 features, validities [0.95, 0.55, 0.75, 0.75, 0.55, 0.95] (f0..f5) \u2014 a PALINDROMIC arrangement with exact ties (two 0.55s, two 0.75s, two 0.95s). Arrangement statistics (both models' own formulas, average-tie ranks [4.5, 0.5, 2.5, 2.5, 0.5, 4.5]): Spearman(position, validity) = 0 EXACTLY (position deviations [-2.5,-1.5,-0.5,0.5,1.5,2.5] dot rank deviations [2,-2,0,0,-2,2] = -5+3+0+0-3+5 = 0), so binding ambiguity = 1.0 \u2014 the competing's primacy gate is FULLY OPEN: gate = (1.0-0.80)/0.20 = 1.0, w_eff = min(primacy_weight, 0.35) in [0.30, 0.35], a FORCED commitment no parameter setting can reduce. Because val == val[::-1] (palindrome), reversal coherence = 1.0, and gap_term = mean_gap/range = 0.20/0.40 = 0.50 (three unique values), so the leak q = rho_leak*1.0*0.50 is FORCED into [0.425, 0.625] (clamp at 0.75 never binds) whenever the anti carrier is the 0.75 group \u2014 the only strictly interior validity (interior requires 0.65 < val < 0.85). Meanwhile the exact ties switch OFF both theories' diffuse distortion (both require all-distinct validities), the cross-group validity gaps (0.20) exceed the maximum gamma*salience tilt (0.12), and every tied group is kept concordant within trials, so tie_break and gamma are inert in both models: both theories consult the IDENTICAL deterministic anti cascade {f1,f4}(55%) -> {f2,f3}(75%) -> {f0,f5}(95%), with TTB = {f0,f5} -> {f2,f3} -> {f1,f4} and primacy = f0->f1->f2->f3->f4->f5. The theories differ ONLY in the competing's two forced mixture mechanisms.\n\nCELL FAMILIES. (L) LEAK-ISOLATED CELLS T1-T6 (three exact A/B mirror pairs): the 0.55 group is tied, a 0.75 cue (f2, f3, or both) favors one option, and the LATE 0.95 expert f5 (position 5) favors the other. Anti decider = the interior 0.75 cue; TTB decider = f5 (opposes -> leak fires); earliest discriminator is the 0.75 cue (position 2-3), so primacy concurs and is inert. Advocated: p(anti option) = p_f in [0.59, 0.84] (mean ~0.71). Competing: p = (1-q)*p_f + q*(1-p_f) in [0.41, 0.55] (mean ~0.49) \u2014 DISJOINT from the advocated's interval on every L cell. (D) DOUBLE CELL T7/T8: 0.75 cue vs BOTH 0.95 experts (f0 and f5) opposing \u2014 leak AND primacy both fire against the anti decider: competing p(A) in [0.32, 0.49] vs advocated [0.59, 0.84], disjoint. (P) PRIMACY-ISOLATED CELLS T9/T10: the 0.55 expert f1 (position 1) vs the FIRST-listed 0.95 expert f0 (position 0) \u2014 carrier is 0.55 (not interior -> leak gated OFF despite TTB opposing), earliest discriminator f0 opposes the anti decider: competing p(anti) = (1-w_eff)*p_f + w_eff*(1-p_f) in [0.52, 0.64] vs advocated [0.59, 0.84]. (G) GATE-CONTROL CELLS T11/T12: f1 vs f5 with f0 tied \u2014 SAME opposition structure as T1 but the carrier is 0.55, so the competing's interiority gate shuts the leak and primacy concurs: BOTH theories predict p_f. T1 vs T11 is the sharpest single contrast in the design: the two cells differ ONLY in the carrier's validity group (0.75 vs 0.55), the advocated predicts EXACTLY EQUAL follow rates (its follow probability cannot depend on the decider's validity), the competing predicts a leak dip of q*(2p_f-1) in [0.04, 0.44] (mean ~0.20). (C) AGREEMENT CELLS T13/T14/T15: a lone discriminator from the 0.55, 0.75, and 0.95 groups respectively \u2014 all mechanisms concur; both theories predict exactly p_f, pinning each subject's follow rate. T14 is the matched no-opposition control for the L cells (same interior carrier f2, TTB opposition removed): the advocated demands p(T1) = p(T14) exactly; the competing demands p(T1) < p(T14) by ~0.20. (T) T16 full tie: exactly 0.50 in both, anchoring lapse and side bias.\n\nSIGNATURES: (1) THE LEAK DIP: within-subject contrast p(anti-follow on T1/T3/T5) minus p(T14): advocated EXACTLY ZERO; competing in [0.04, 0.44], mean ~0.20 \u2014 with 18 leak trials and 6 matched control trials per subject the dip is estimable per subject. (2) THE INTERIORITY-GATE CONTRAST T1 vs T11: advocated equal; competing dips only on the interior-carrier cell \u2014 an unconditional TTB-mixture or validity-respecting contamination would dip on BOTH, so this triad (T1, T11, T14) isolates the gate's specific conjunction (interior carrier AND TTB opposition). (3) THE PRIMACY DIP: T9 vs T13 (matched 0.55-carrier control): advocated zero; competing w_eff*(2p_f-1) in [0.03, 0.25], mean ~0.13. (4) THE DIP-RATIO POINT PREDICTION: if both dips are present, their ratio estimates q/w_eff in [1.21, 2.08] (mean ~1.6) \u2014 numbers forced by the competing's own gate formulas at this arrangement; both dips exactly zero falsify the competing's forced w_eff >= 0.30 and q >= 0.425 at ANY parameter setting (the mechanisms cannot be refit away). (5) GLOBAL PROFILE: the advocated emits a FLAT two-level step function (p_f toward the anti decider on all 15 discriminating cells, 0.50 on the tie) \u2014 it has no mechanism that depends on the carrier's validity rank, the TTB decider's direction, or the earliest discriminator's direction; the competing emits a graded 'fingerprint' profile (0.69 on G/C cells, 0.57 on P cells, 0.49 on L cells, 0.43 on D cells toward the anti option), with the strict ordering p(T14) = p(T11) > p(T9) > p(T1) > p(T7) that the advocated cannot reproduce. Mirror pairs (T2/T4/T6/T8/T10/T12) rule out side bias.\n\nHONESTY NOTES: (i) On P cells the single-cell population intervals overlap at the extremes ([0.52, 0.64] vs [0.59, 0.84]); the decisive P-cell evidence is the within-subject contrast against T13. (ii) The L and D cells are single-cell disjoint, but the strongest evidence is the paired within-subject contrasts (Signatures 1-2), which are robust to fitting beta, epsilon, gamma, rho_leak, and primacy_weight to each theory (gamma, sigma_h, kappa, zeta, and tie_break are provably inert: distortion is gated off by the ties, no cross-group weight ties are reachable for gamma in [-0.05, 0.12], and tied groups never oppose). (iii) The competing has MORE free parameters than the advocated, yet is FORCED to dip \u2014 the extra flexibility cannot remove the signature, making the test genuinely two-sided: flat follow profiles falsify the competing's forced q >= 0.425 and w_eff >= 0.30; any reliable dip (>= 0.05) on interior-carrier cells or earliest-position-opposition cells falsifies the advocated, which is structurally incapable of producing either. (iv) The sub-linear coherence exponent is not separately testable here (coherence = 1 by construction); the design instead maximizes total leak mass, which is the axis on which the two theories' predictions are most widely separated. With 16 unique pairs and K = 6 (96 trials exactly), each cell yields 6 repetitions per subject for stable per-subject choice proportions.",
  "validities": [
    0.95,
    0.75,
    0.75,
    0.95,
    0.75,
    0.75
  ],
  "trial_a_ratings": [
    [
      0,
      0,
      1,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      1
    ],
    [
      0,
      0,
      0,
      1,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      1
    ],
    [
      0,
      0,
      1,
      1,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      1
    ],
    [
      0,
      0,
      1,
      0,
      0,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0,
      1
    ],
    [
      0,
      1,
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      1
    ],
    [
      0,
      1,
      0,
      1,
      0,
      0
    ],
    [
      0,
      0,
      1,
      0,
      1,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0,
      1
    ],
    [
      1,
      0,
      1,
      0,
      1,
      0
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      0,
      0,
      0,
      0,
      1
    ],
    [
      0,
      0,
      1,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      1
    ],
    [
      0,
      0,
      0,
      1,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      1
    ],
    [
      0,
      0,
      1,
      1,
      0,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0,
      1
    ],
    [
      0,
      0,
      1,
      0,
      0,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      1
    ],
    [
      0,
      1,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      1,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      1,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      1
    ],
    [
      1,
      0,
      1,
      0,
      1,
      0
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 36944,
  "output_tokens": 31114,
  "reasoning_tokens": 28228,
  "provider": "Sail Research"
}
```
