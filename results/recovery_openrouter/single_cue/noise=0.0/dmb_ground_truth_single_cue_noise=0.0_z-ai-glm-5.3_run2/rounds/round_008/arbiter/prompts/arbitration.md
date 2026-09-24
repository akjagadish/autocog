# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_7" and "pi_8") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_7" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_8" (THEORY 2 below). Justify your choice.


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

## THEORY 2 — pi_8
Anchored Anti-Validity Frugality with Validity-Respecting Leak (frozen-discipline variant). People are one-reason decision makers: experts are consulted one at a time and the first strictly discriminating expert decides; no integration across cues. The population-central hierarchy is anti-validity (distrust-the-weakest) and the population is typeless — every subject is a continuous perturbation around that core. Three graded, design-scaled mechanisms modulate the core. (1) GAP-PROPORTIONAL ANCHORING: per-trial attention noise on expert j is scaled by the local validity gaps adjacent to j in the subjective order, gated by cubed binding ambiguity, so validity-isolated experts anchor the order while dense clusters diffuse; exact validity ties anchor the order entirely. (2) VALIDITY-RESPECTING LEAK, now TRIAL-CONDITIONAL and implemented as a probability mixture: the leak toward the standard TTB (descending-validity) order fires only when the anti-validity cascade's first discriminating expert on that trial is a strictly INTERIOR validity (min_val + 0.10 < val < max_val - 0.10), with design-level probability q = min(rho_leak * reversal_coherence * mean_gap, 0.18). When the anti-decider's carrier is a weakest or near-top expert, the leak is switched off — protecting the pure anti-validity-follow designs — while genuinely mid-validity carriers let a fraction of trials flip toward the strongest discriminator, generating below-chance reversed-cascade following where the data demand it. (3) DESIGN-SCALED RESPONSE DISCIPLINE with a DISTINCT-DESIGN BONUS: p_f = (1-eps)*sigmoid(beta_eff) + eps/2 with beta_eff = beta*(1 + boost*(1-A))*(1 + b_distinct) when all validities are distinct — distinct-validity designs (which the data show want tighter decider-following) get a bounded bonus, tie designs are untouched. Response-discipline and diffusion parameters are FROZEN at narrow windows reproducing the accepted base's behavior, so only the leak strength and the distinct bonus are free.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Anchored Anti-Validity Frugality with Validity-Respecting Leak
    # (frozen-discipline variant).
    #
    # Minimal-diff edit of the accepted iter-1 base:
    #   (a) The validity-respecting leak is now TRIAL-CONDITIONAL: it is
    #       eligible only when the anti-validity cascade's first
    #       discriminating expert j* on this trial carries a strictly
    #       INTERIOR validity (min + 0.10 < val[j*] < max - 0.10), and
    #       it is implemented as a PROBABILITY MIXTURE (no Bernoulli
    #       draw) with ceiling 0.18.
    #   (b) A DISTINCT-DESIGN BONUS b_distinct multiplies beta_eff when
    #       all validities are distinct (tie designs untouched).
    #   (c) Response-discipline / diffusion parameters are FROZEN at
    #       narrow windows that reproduce the accepted base's behavior;
    #       only rho_leak and b_distinct are genuinely free.
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
    boost = float(parameters["boost"])
    b_distinct = float(parameters["b_distinct"])
    epsilon = float(parameters["epsilon"])
    gamma = float(parameters["gamma"])
    kappa = float(parameters["kappa"])
    sigma_h = float(parameters["sigma_h"])
    rho_leak = float(parameters["rho_leak"])
    tie_break = float(parameters["tie_break"])

    zeta = np.asarray(parameters["zeta"], dtype=float).ravel()
    if zeta.shape[0] != n_features:
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

    # ---- mean adjacent validity gap over UNIQUE validities ----
    uvals = np.unique(val)
    if uvals.size > 1:
        mean_gap = float(np.mean(np.diff(np.sort(uvals))))
    else:
        mean_gap = 0.0

    distinct = (uvals.size == n_features)

    # ---- gap-proportional anchoring ratios (verbatim from base) ----
    ref_gap = 0.10
    gap_ratio = np.ones(n_features, dtype=float)
    if distinct and n_features > 1:
        srt = np.argsort(val, kind="stable")
        sv = val[srt]
        gs = np.ones(n_features, dtype=float)
        for r in range(n_features):
            g = 0.0
            if r > 0:
                g += sv[r] - sv[r - 1]          # gap to next weaker
            if r < n_features - 1:
                g += sv[r + 1] - sv[r]          # gap to next stronger
            gs[srt[r]] = max(g, 0.05)           # numerical floor
        gap_ratio = np.minimum(ref_gap / gs, 2.0)

    amb3 = ambiguity * ambiguity * ambiguity

    pos = np.arange(n_features)
    secondary = pos if tie_break < 0.8 else (n_features - 1 - pos)

    a, b = stim[0], stim[1]

    def cascade_winner(order):
        for j in order:
            if a[j] > b[j]:
                return 0
            if b[j] > a[j]:
                return 1
        return None

    # ---- STABLE anti-validity weights (no momentary noise): used ONLY
    # to locate this trial's anti-validity carrier j* for the leak gate.
    w_stable = -val + gamma * sal
    if distinct and ambiguity > 0.0:
        w_stable = w_stable + amb3 * kappa * gap_ratio * sigma_h * zeta
    order_stable = np.lexsort((secondary, -w_stable))

    j_star = None
    for j in order_stable:
        if a[j] != b[j]:
            j_star = int(j)
            break

    # ---- TRIAL-CONDITIONAL leak eligibility: strictly interior carrier.
    # Weakest-expert carriers (val <= min + 0.10) and near-top carriers
    # (val >= max - 0.10) switch the leak OFF; only genuinely mid-
    # validity carriers let the TTB leak fire. Ceiling 0.18.
    leak_q = 0.0
    if j_star is not None and n_features > 1:
        vmin = float(val.min())
        vmax = float(val.max())
        if (vmin + 0.10) < val[j_star] < (vmax - 0.10):
            leak_q = float(min(rho_leak * coherence * mean_gap, 0.18))

    # ---- per-trial subjective hierarchy (momentary diffusion added) ----
    w = w_stable.copy()
    if distinct and ambiguity > 0.0:
        xi = np.random.normal(0.0, 1.0, size=n_features)
        w = w + amb3 * kappa * gap_ratio * xi
    order_anti = np.lexsort((secondary, -w))
    winner_anti = cascade_winner(order_anti)

    n_opts = 2
    if winner_anti is None:
        # No discriminating cue: pure guess (lapse mixing leaves a
        # uniform unchanged).
        return np.ones(n_opts) / float(n_opts)

    # ---- design-scaled response discipline with DISTINCT-DESIGN BONUS ----
    beta_eff = beta * (1.0 + boost * (1.0 - ambiguity))
    if distinct:
        beta_eff = beta_eff * (1.0 + b_distinct)

    def score_prob(winner):
        scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
        z = beta_eff * (scores - scores.max())
        e = np.exp(z)
        return e / e.sum()

    p_core = score_prob(winner_anti)

    # ---- VALIDITY-RESPECTING LEAK as a probability mixture (no
    # Bernoulli draw -> lower Monte Carlo variance in simulated metrics).
    if leak_q > 0.0:
        order_ttb = np.lexsort((secondary, -val))
        winner_ttb = cascade_winner(order_ttb)
        if winner_ttb is not None and winner_ttb != winner_anti:
            p_core = (1.0 - leak_q) * p_core + leak_q * score_prob(winner_ttb)

    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / float(n_opts))

`policy(probs) -> int`:
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return np.random.choice(len(probabilities), p=probabilities)

## EXPERIMENT 1 (proposed by pi_7)

### DESIGN
**Validities (n_features=6):** [0.5, 0.5, 0.75, 0.75, 1.0, 1.0]

**Trial pairs (n=16):**
  trial 1: A=[0, 1, 1, 0, 0, 1]  B=[0, 1, 0, 0, 1, 1]
  trial 2: A=[0, 1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0, 1]
  trial 3: A=[1, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 1]
  trial 4: A=[1, 0, 0, 0, 0, 1]  B=[1, 0, 0, 1, 0, 0]
  trial 5: A=[0, 1, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  trial 6: A=[0, 1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0]
  trial 7: A=[1, 0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  trial 8: A=[1, 0, 0, 1, 1, 1]  B=[1, 0, 1, 1, 0, 0]
  trial 9: A=[0, 1, 1, 0, 1, 1]  B=[0, 1, 0, 0, 0, 1]
  trial 10: A=[0, 1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 1, 1]
  trial 11: A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  trial 12: A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 1, 0, 0]
  trial 13: A=[1, 0, 0, 1, 0, 1]  B=[0, 0, 1, 1, 0, 1]
  trial 14: A=[0, 0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0, 1]
  trial 15: A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1, 0]
  trial 16: A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]

**Rationale:** GOAL: Quantitatively dissociate Anti-Validity Frugality with Diffuse Positional-Salience Heterogeneity, momentary-diffusion variant (advocated) from Anchored Anti-Validity Frugality with Validity-Respecting Leak, frozen-discipline variant (competing). The two theories share the anti-validity one-reason core, the same A^3 ambiguity gate, comparable diffusion magnitudes, and the same tie-anchoring rule — so diffusion-structure axes cannot cleanly separate them (at any ambiguity where the advocated diffuses, the competing's gap-scaled noise is of the same order). The ONE structural mechanism with diverging, design-forced commitments is the competing's TRIAL-CONDITIONAL VALIDITY-RESPECTING LEAK: a probability mixture (ceiling 0.18) toward the standard TTB (descending-validity) decider that fires ONLY when the anti-validity cascade's first discriminating expert carries a strictly INTERIOR validity (min+0.10 < val < max-0.10). The advocated has NO such mechanism: in the anchored regime its follow probability is p_f toward the anti-decider on EVERY discriminating trial, flat over decider validity, decider position, margin, and coalition size. The design therefore engineers the anchored regime and maximizes the competing's committed leak strength, then tests a flat-vs-dipped follow profile.

SETUP: 6 features with validities [0.50, 0.50, 0.75, 0.75, 1.00, 1.00] (f0..f5), monotone ascending with exact ties. Consequences (both models' own formulas, average-tie ranks): Spearman(position, validity) = 0.956, so binding ambiguity A = 0.044 and A^3 = 8.4e-5 — both theories' distortion channels are switched off entirely (their own rules: distortion requires all-distinct validities AND ambiguity > 0; ties anchor the distrust order in both). Every within-tied-pair gap (0.25) dwarfs both theories' gamma*sal tilts (advocated <= 0.072, competing <= 0.018), so the consultation order is EXACTLY the anti-validity cascade {f0,f1}(50%) -> {f2,f3}(75%) -> {f4,f5}(100%) in both models, with tie_break inert (tied cues never oppose in any trial). Reversal coherence = 0.956; mean_gap over unique validities = 0.25 — the MAXIMUM achievable while keeping an interior validity (three unique values spanning [0.5, 1.0]). Hence the competing's leak probability q = min(rho_leak * 0.956 * 0.25, 0.18) is forced into [0.120, 0.18] for ANY rho_leak in [0.5, 3.0]: the competing CANNOT switch its leak off in this design. Its p_f = (1-eps)*sigmoid(beta*(1+boost*0.956)) + eps/2 lies in [0.708, 0.754] (distinct-design bonus inert because validities are tied). The advocated's p_f lies in [0.59, 0.84] (beta in [0.5,1.9], eps in [0.08,0.24]).

CELL FAMILIES. (L) LEAK CELLS (T1-T8, four exact A/B mirror pairs): the anti-decider is an interior 0.75 cue and a 100% expert discriminates in OPPOSITION (all 50% cues tied, so the carrier is unambiguous). T1/T2: f2 vs f4; T3/T4: f3 vs f5 (position-balanced replicate); T5/T6: both 0.75s vs both 100%s (tally margin 0); T7/T8: one 0.75 vs both 100%s (margin -1). Advocated: p(carrier's option) = p_f on ALL eight. Competing: p(carrier's option) = (1-q)*p_f + q*(1-p_f) in [0.633, 0.693] — a DIP of q*(2p_f-1) in [0.050, 0.093] relative to its own no-leak follow rate. (C) CONTROL CELLS (T9-T16): T9/T10 — interior 0.75 carrier with the 100% expert CONCURRING (leak eligible but winner_ttb = winner_anti, so no mixture): both theories p_f; this matches decider validity with the leak cells, isolating TTB opposition as the dip's cause. T11/T12 — WEAKEST carrier (50%) opposed by the 100% expert: leak OFF (carrier not interior); both theories p_f toward the 50% expert. T13/T14 — weakest carrier (50%) opposed by the interior 0.75 expert: leak OFF; a generic always-on TTB mixture would dip here too, but the competing's trial-conditional gate does not — this cell separates the leak from any unconditional validity-respecting contamination. T15 — lone interior decider (only f2 discriminates; TTB concurs): p_f. T16 — full tie: exactly 0.50 in both, anchoring lapse and side bias.

SIGNATURE 1 — THE LEAK DIP (central, two-sided, parameter-forced): define each subject's carrier-follow rate (choices matching the anti-validity decider's option). Advocated: EXACTLY EQUAL follow rates on leak and control cells (both are p_f; no mechanism in the advocated depends on the decider's validity rank, its position, or the opposition's validity). Competing: follow rate on the eight leak cells is lower by q*(2p_f-1) in [0.050, 0.093] — bounded AWAY FROM ZERO by its own gate formulas (q >= 0.120 is forced by rho_leak >= 0.5, coherence 0.956, mean_gap 0.25). The prediction intervals on this within-subject paired contrast are DISJOINT: [0, 0] vs [0.050, 0.093]. With 48 leak trials and 42 control trials per subject, a population dip estimate of ~0 (CI within +/-0.03) falsifies the competing at any parameter setting; a dip >= 0.05 falsifies the advocated, which cannot produce it.

SIGNATURE 2 — THE CONDITIONALITY TRIAD (specificity of the dip): the competing predicts the dip ONLY on the conjunction (interior carrier AND top-validity opposition): T1 vs T9/T15 (interior carrier, opposition removed -> no dip), T1 vs T11 (opposition retained, carrier demoted to weakest -> no dip), T1 vs T13 (higher-validity opposition retained, carrier weakest -> no dip). The advocated predicts no dip in any cell. An unconditional partial-TTB mixture (the natural simpler competitor) dips on T11 AND T13 as well — so the triad also protects the dissociation from being absorbed by a less conditional theory.

SIGNATURE 3 — FLAT PROFILE OVER MARGIN AND COALITION (shared one-reason core + integration kill): across T1 (margin 0, 1v1), T5 (margin 0, 2v2), T7 (margin -1) both theories predict IDENTICAL carrier-follow rates — no margin effect, no coalition-size effect. Tallying is pinned to EXACTLY 0.50 on T5/T6 (margin 0) and below 0.5 on T7, while both one-reason theories predict ~p_f toward the 0.75-cue option — any consistent preference on T5/T6 falsifies the entire integration family, and any reliable margin gradient falsifies the one-reason core of whichever target theory is otherwise supported.

SIGNATURE 4 — NOISE-REGIME DISPERSION (secondary): on the no-leak cells the advocated's wide (beta, epsilon) ranges produce a between-subject SD of follow rates ~0.06-0.10, while the competing's frozen discipline ([0.80,1.00]x[0.28,0.42]x[0.12,0.16]) produces ~0.02 true SD (observed ~binomial floor). The between-subject dispersion of T9/T11/T13/T15 follow rates separately tests the two noise regimes.

HONESTY NOTE: single-cell population intervals overlap (the advocated's p_f reaches 0.84; the competing's leak-cell floor is 0.633), so no single cell is decisive; the decisive statistics are the within-subject paired contrasts (Signatures 1-2), the 16-cell profile shape, and the dispersion test — all robust to fitting beta, epsilon, gamma, sigma_h, kappa, zeta, rho_leak, boost, and tie_break to each theory (gamma, sigma_h, kappa, zeta, and tie_break are provably inert in this design: distortion is gated off, and no tilt can flip a 0.25 validity gap). The test is genuinely two-sided: a flat follow profile falsifies the competing's forced q >= 0.120 leak; a dip >= 0.05 falsifies the advocated's leak-free flatness claim. 16 unique pairs, K = 6 (96 trials exactly), give 6 repetitions per cell for stable per-subject choice proportions.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
THE METRIC IS THE WITHIN-SUBJECT 'LEAK DIP'. For every trial I reconstruct the anti-validity decider (first strictly discriminating cue in ascending-validity order; validities [0.5,0.5,0.75,0.75,1,1] ascend with index and tied cues never oppose, so the reconstruction is exact and tie_break/gamma inert). A trial is LEAK-ELIGIBLE iff the carrier is strictly interior (0.75) AND the standard TTB decider (descending validity) picks the opposite option — exactly the competing theory's trial-conditional gate (trials 1-8, 48 trials/subject). All other discriminating trials are CONTROLS (trials 9-15, 42 trials/subject); trial 16 (full tie) is excluded. The metric is P(follow anti-decider | leak cells) - P(follow anti-decider | control cells), computed within subject, so each subject's own p_f cancels. PREDICTIONS: the advocated theory's follow probability is flat p_f on every discriminating trial (no mechanism depends on carrier validity, position, margin, or coalition), so its dip is exactly 0 in expectation, with only binomial noise (per-subject SD ~ sqrt(p_f(1-p_f)(1/48+1/42)) ~ 0.095, giving between-subject variance ~0.009). The competing theory is FORCED by its own gate formulas to leak q = min(rho_leak*0.956*0.25, 0.18) >= 0.120 on the leak cells (its follow there is (1-q)p_f + q(1-p_f)), producing a dip of -q(2p_f-1) in [-0.093, -0.050] (mean ~ -0.08 since q saturates at 0.18 for ~90% of rho_leak's range and p_f in [0.71, 0.75]). The control cells are matched on decider validity (T9/T10/T15 are interior-carrier cells where TTB concurs) and on opposition presence (T11-T14 keep the higher-validity opposition but demote the carrier to 0.5, where the competing's gate switches the leak OFF), so both theories predict p_f there — the contrast isolates the trial-conditional leak as the sole source of any dip. Expected Welch t at N=25: gap ~0.08 against per-subject SD ~0.095 in both conditions (binomial floor dominates both theories, so between-subject variances are nearly equal at ~0.009), giving t ~ 3.0, p ~ 0.004 < alpha = 0.01. The metric is well-defined on a single subject's slice (48 leak + 42 control trials) and on the pooled frame, and it is two-sided-falsifiable: a population dip reliably below -0.05 refutes the advocated's flatness claim; a dip indistinguishable from 0 refutes the competing's forced q >= 0.120 leak.

Source:
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

### RESULTS
- Predicted under pi_7 (simulated): 0.0068 (var=0.0083)
- Predicted under pi_8 (simulated): -0.0798 (var=0.0081)
- Observed on real data: -0.1955 (var=0.0143)

## EXPERIMENT 2 (proposed by pi_8)

### DESIGN
**Validities (n_features=6):** [0.95, 0.78, 0.5, 0.5, 0.5, 0.5]

**Trial pairs (n=16):**
  trial 1: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 2: A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1]
  trial 3: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 4: A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0, 1]
  trial 5: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 6: A=[1, 1, 1, 0, 1, 1]  B=[1, 1, 1, 1, 0, 1]
  trial 7: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 8: A=[1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 1, 1]
  trial 9: A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 10: A=[1, 1, 1, 1, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  trial 11: A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 13: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 14: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 15: A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1]
  trial 16: A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]

**Rationale:** GOAL: Quantitatively dissociate Anchored Anti-Validity Frugality with Validity-Respecting Leak, frozen-discipline variant (advocated) from Anti-Validity Frugality with Diffuse Positional-Salience Heterogeneity, momentary-diffusion variant (competing). Both share the anti-validity one-reason core, the same A^3-gated distortion rule, and the same tie-anchoring rule, and experiment [16] already exploited the one mechanism the advocated uniquely has (the trial-conditional TTB leak) in a tied-validity, ambiguity-0.044 regime. This design occupies a different, unexplored corner: the ZERO-AMBIGUITY, ALL-DISTINCT regime in which every gated channel (momentary diffusion, stable zeta, gap-proportional anchoring, leak on weak-block deciders) is STRUCTURALLY OFF in both theories, leaving exactly one active difference: the UNGATED positional-salience tilt gamma*salience, which both theories apply unconditionally but with committed different magnitudes — the competing draws gamma asymmetrically from [-0.05, +0.12] (a real, population-level early-position lean), while the advocated's gamma is confined to [-0.01, +0.03] (effectively zero). The design converts this tilt-magnitude difference into a measurable VALIDITY-GAP PSYCHOMETRIC.

SETUP: 6 features, validities [0.95, 0.78, 0.534, 0.53, 0.52, 0.50] (f0..f5), strictly DECREASING in position, so Spearman(position, validity) = -1, binding ambiguity = 0 exactly: both theories' `distinct and ambiguity > 0` gates shut all distortion off, and all validities distinct makes tie_break inert in both. The consultation order is therefore deterministic per subject: w_j = -val_j + gamma*sal_j with sal = [+0.5, +0.3, +0.1, -0.1, -0.3, -0.5]. The anti-validity core consults f5 (50%) -> f4 (52%) -> f3 (53%) -> f2 (53.4%) -> f1 (78%) -> f0 (95%). KEY MECHANIC: for any two cues X (earlier position, higher validity) and Y (later, weaker), the subject consults X before Y iff gamma > Delta_v / Delta_sal =: g. So on a trial where ONLY X and Y discriminate in opposition, the choice reveals a threshold crossing of the subject's gamma. The weak block {f2..f5} is engineered with tiny, varied gaps so that different cue pairs realize a LADDER of thresholds g = 0.020, 0.035, 0.050, 0.057, 0.075, 0.100 — spanning the competing's gamma range while sitting (except the lowest rung) entirely ABOVE the advocated's gamma cap of 0.03.

PREDICTED FLIP RATES P(consult the earlier cue first): competing, gamma ~ Uniform[-0.05, 0.12]: P(gamma > g) = (0.12 - g)/0.17 = 0.588, 0.500, 0.412, 0.372, 0.265, 0.118 across the six rungs. Advocated, gamma ~ Uniform[-0.01, 0.03]: 0.25 at g = 0.020 and EXACTLY ZERO at every rung with g >= 0.035 — RANGE-FORCED, since gamma <= 0.03 < 0.035 for any admissible parameter setting. Population p(earlier-cue option), p_E = P_flip*p_f + (1-P_flip)*(1-p_f): competing (p_f mean 0.71) predicts 0.537, 0.500, 0.463, 0.446, 0.401, 0.340; advocated (p_f = (1-eps)*sigmoid(beta*(1+boost)*(1+b_distinct)) + eps/2 in [0.678, 0.832] at ambiguity 0 with the distinct-design bonus) predicts 0.334-0.411 at g = 0.020 and a FLAT 1 - p_f in [0.168, 0.322] on all five higher rungs.

SIGNATURE 1 — THE GAP PSYCHOMETRIC (primary): the competing predicts a strictly GRADED p_E(g), falling ~0.20 from the lowest to the highest rung (0.537 -> 0.340); the advocated predicts a FLAT profile at 1 - p_f across g = 0.035...0.100 (range-forced zero flips), with at most a single small step at g = 0.020. The paired within-subject contrast p_E(g=0.035) - p_E(g=0.100) is ~ +0.16 under the competing and EXACTLY ZERO under the advocated. Mirror pairs (T1/T2, T3/T4, T5/T6, T9/T10) cancel side bias; p_E is pooled over each mirror pair (12 trials/subject per rung).

SIGNATURE 2 — THE MIDPOINT KNIFE EDGE (T3/T4, g = 0.035, the exact midpoint of the competing's gamma interval): the competing predicts p_E = 0.500 EXACTLY, for ANY (beta, epsilon) — with P_flip = 0.5, p_E = 0.5*p_f + 0.5*(1-p_f) = 0.5, a parameter-free point prediction. The advocated is range-forced to p_E = 1 - p_f <= 0.322. The intervals [0.500] and [0.168, 0.322] are DISJOINT: population p_E > 0.42 on this rung falsifies the advocated at any parameter setting; p_E < 0.40 falsifies the competing's asymmetric-tilt prior.

SIGNATURE 3 — PER-SUBJECT STEP FUNCTION AND DISPERSION (sharpest structural test): each competing subject holds ONE gamma, so choices across the six rungs trace a STEP FUNCTION in g (follow the earlier cue on rungs below gamma-hat, the weaker cue above it); ~50% of subjects have gamma-hat inside (0.035, 0.12] and thus flip somewhere on the ladder, and at g = 0.035 the population is BIMODAL (half at p_f ~ 0.71, half at 1 - p_f ~ 0.29; between-subject SD ~ 0.21). The advocated predicts ZERO subjects flip on any g >= 0.035 rung — every subject follows the weaker cue, tight cluster at 1 - p_f (true SD ~ 0.03-0.04). A single subject following the earlier cue consistently (beyond binomial noise, P ~ 0.09 per 6-rep cell at worst-case p_f) on any g >= 0.035 cell falsifies the advocated outright; a near-unimodal low cluster falsifies the competing's committed gamma distribution. The design thus MEASURES the population gamma distribution directly, making the competing's prior-based commitment empirically testable rather than assumable.

SIGNATURE 4 — THE ADVOCATED'S OWN GAMMA CAP (T1/T2, g = 0.020): the advocated predicts a 25% flip rate there (p_E ~ 0.375), the competing 58.8% (p_E ~ 0.537) — disjoint intervals [0.334, 0.411] vs [0.50, 0.57], a genuinely two-sided test of each theory's tilt prior at its left end.

SIGNATURE 5 — SECONDARY LEAK PROBE (C3 vs C4): C3 pits the interior-validity 78% expert (the stable anti-decider, j* interior since 0.60 < 0.78 < 0.85) against the 95% expert in opposition; the advocated's trial-conditional leak fires with q = min(rho_leak * 1.00 * 0.090, 0.18) in [0.045, 0.18] (forced by rho_leak >= 0.5, coherence = 1, mean_gap = 0.090), predicting a within-subject dip q*(2*p_f - 1) in [0.016, 0.120] relative to the matched concordance cell C4 (same decider, TTB opposition removed, no mixture). The competing predicts EXACTLY ZERO dip. This axis is weak here by construction (the tilt design requires tiny gaps, which shrink mean_gap), so it is secondary; [16] tested the leak at full strength. All tilt-cell and control-cell deciders are weak-block or min/max-validity cues, so the leak is provably OFF on T1-T10, C1, C2, C5 — the tilt signatures are uncontaminated.

SIGNATURE 6 — CONTROLS: C1/C5 (lone weakest / lone block decider) are agreement cells pinning p_f for both theories and separating the noise regimes (advocated frozen-narrow p_f in [0.678, 0.832], between-subject SD ~ 0.03; competing p_f in [0.51, 0.92], SD ~ 0.08). C2 (weakest 50% expert vs the 95% expert, in opposition) tests the SHARED anti-validity core: both theories follow the 50% expert at p_f; if subjects instead follow the 95% expert, both target theories fail together. C6 (full tie) pins both to exactly 0.50, anchoring lapse and side bias.

HONESTY NOTES: (i) Within every critical pair the earlier-position cue is also the higher-validity cue — unavoidable, since zero ambiguity (Spearman = -1) forces position-validity alignment; a pure validity-respecting strategist would also produce the competing's tilt-side predictions, so elevated p_E falsifies the advocated without uniquely confirming the competing (C2 partially guards the shared anti-validity core). (ii) The competing's flip commitment is prior-based (its asymmetric gamma draw), not range-forced: a post-hoc refit with gamma ~ 0 for every subject could mimic the flat profile, but only by contradicting the competing's own stated population prior (mean +0.035, 41% of subjects above 0.05, 50% above 0.035) — the ladder measures that distribution directly, so the concession would be visible and quantified. (iii) Single-cell intervals overlap at the high-g rungs (g = 0.075, 0.100); the decisive evidence is the low-g rungs, the psychometric slope, and the per-subject step/dispersion structure — all robust to fitting beta, epsilon, gamma, sigma_h, kappa, zeta, rho_leak, boost, b_distinct, and tie_break to each theory (sigma_h, kappa, zeta, tie_break are provably inert at ambiguity 0 with distinct validities). (iv) The g = 0.035 knife edge assumes the competing's gamma draw is uniform on [-0.05, +0.12]; under any prior with appreciable mass in the upper half of that interval the prediction remains well above the advocated's forced ceiling. With 16 unique pairs and K = 6 (96 trials exactly), each cell yields 6 repetitions per subject for stable choice proportions.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
WHY THE PREVIOUS METRIC FAILED AND WHAT SURVIVES. With this experiment's actual validities [0.95, 0.78, 0.5, 0.5, 0.5, 0.5], the weak block is EXACTLY TIED, so `distinct` is False in both theories' predict functions: every gated distortion channel (momentary diffusion, stable zeta, gap-proportional anchoring, distinct-design bonus) is structurally OFF in BOTH theories. Both reduce to the same deterministic anti-validity cascade whose weak-block consultation order is set purely by the SIGN of the positional tilt gamma, plus a subject-level decider-follow probability p_f = (1-eps)*sigmoid(beta_eff) + eps/2. The failed tilt metric showed the gamma-sign weights are nearly identical across theories (advocated 0.75, competing ~0.8), so the tilt axis is dead. The one large population-level difference that survives is RESPONSE-DISCIPLINE DISPERSION — the very 'frozen-discipline vs diffuse-heterogeneity' contrast defining the two theories. Advocated: beta in [0.8,1.0], boost in [0.28,0.42], epsilon in [0.12,0.16] give beta_eff in [0.99,1.35] and p_f confined to [0.67,0.78] with population SD ~0.015. Competing: beta in [0.5,1.9] and epsilon in [0.08,0.24] give p_f spanning [0.59,0.84] with SD ~0.07 — a 4-5x dispersion ratio.

METRIC CONSTRUCTION. (1) Per subject, infer the weak-block positional side behaviorally (follow-earlier rate over the 60 weak-block trials is ~0.73 or ~0.27, so classification is essentially error-free and even handles gamma=0 tie-break subjects correctly). (2) Label every non-tie trial's decider as the weakest-validity discriminating expert (weak-block ties resolved by the inferred side) and compute the subject's decider-follow rate p_hat over the 90 informative trials. (3) The metric is the mean of the largest 20% of per-subject |p_hat - 0.72| values — the tail of the discipline distribution. The tail is used because the theories differ most there: the advocated HARD-BOUNDS every subject's p_f into [0.67,0.78] (its tail is pure binomial noise), while the competing regularly manufactures extreme followers (p_f near 0.6 or 0.84); a plain mean of deviations would be diluted by the shared binomial floor. Averaging the top quintile (rather than taking a maximum) keeps the statistic stable against single-subject binomial flukes.

PREDICTIONS. Advocated: p_hat ~ N(0.726, 0.049), top-quintile mean deviation ~0.09 (range 0.08-0.11). Competing: p_hat ~ N(0.73, 0.085), top-quintile mean ~0.16 (range 0.13-0.19). The per-subject statistic |p_hat-0.72| is tight (SD ~0.03 advocated, ~0.05 competing), so the between-subject variance is small while the pooled gap is ~0.07 — a Welch t well above threshold at N=25. The contrast is robust: it persists even if the competing's effective p_f SD is as low as 0.04, it is insensitive to the tilt-sign similarity that killed the previous metric, it uses all 90 informative trials per subject, mirror pairs cancel side bias, and the formula is well-defined on any subset (a single-subject slice returns that subject's own deviation), so the per-subject reapplication needed for the between-subject variance is valid. It also cleanly separates the shared anti-validity core (used to identify deciders) from the discipline dispersion being measured, so it is not trivially satisfied by either theory's cascade structure.

Source:
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

### RESULTS
- Predicted under pi_7 (simulated): 0.1120 (var=0.0016)
- Predicted under pi_8 (simulated): 0.0753 (var=0.0006)
- Observed on real data: 0.1556 (var=0.0024)

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
**Predicted under pi_8:** 0.2622 (var=0.0083)

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
**Predicted under pi_8:** 0.1058 (var=0.0349)

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
**Predicted under pi_8:** 0.4920 (var=0.0059)

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
**Predicted under pi_8:** 0.9560 (var=0.0463)

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
**Predicted under pi_8:** 0.6900 (var=0.0042)

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
**Predicted under pi_8:** 0.4700 (var=0.0333)

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
**Predicted under pi_8:** -0.1673 (var=0.0158)

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
**Predicted under pi_8:** -0.1967 (var=0.0089)

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
**Predicted under pi_8:** 0.4053 (var=0.0082)

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
**Predicted under pi_8:** 107.0444 (var=0.0025)

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
**Predicted under pi_8:** 0.5026 (var=0.0029)

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
**Predicted under pi_8:** 0.5729 (var=0.0032)

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
**Predicted under pi_8:** 0.3358 (var=0.0067)

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
**Predicted under pi_8:** 0.0000 (var=0.0000)

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
**Predicted under pi_8:** 0.1434 (var=0.0178)

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
**Predicted under pi_8:** 0.0000 (var=0.0000)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all 18 experiments, pi_7 is the modestly better theory, but neither theory is task-invariant, and each fails structurally on at least one axis that its own description forbids it from fixing.\n\nPoint-estimate scorecard (closer theory per experiment): pi_7 wins Experiments 2, 3, 4, 5, 6, 8, 10, 12, 14, 15 (10 wins); pi_8 wins Experiments 1 (direction only), 7, 9, 11, 13, 17 (6 wins); Experiments 16 and 18 are exact ties (all zeros). So pi_7 is the keeper on aggregate.\n\nBut the failures are diagnostic rather than incremental:\n\n(1) Experiment 1 (leak dip) falsifies pi_7's core flatness claim. pi_7 predicts a dip of exactly 0 (observed 0.0068) and the real data show -0.1955 \u2014 a 29-sigma-style structural miss. pi_8's trial-conditional leak gets the sign right but its hard ceiling q <= 0.18 caps the dip at ~-0.08 (simulated -0.0798); reproducing -0.196 with p_f ~ 0.72 requires q ~ 0.44. So the validity-respecting leak is real, interior-carrier-gated (Exp 1's triad structure holds: no dip on weakest-carrier cells), but 2.5x stronger than pi_8 allows.\n\n(2) Experiment 2 (discipline dispersion) falsifies pi_8's frozen-discipline commitment. Observed tail dispersion 0.1556 vs pi_8's 0.0753 \u2014 the frozen narrow (beta, boost, epsilon) windows are too tight by half. pi_7's diffuse variant (0.1120) is closer but still undershoots; the population is even more heterogeneous than pi_7's ranges produce.\n\n(3) Both theories systematically underpredict the magnitude of validity-respecting behavior: Exp 3 (TTB follow level 0.3633 vs 0.29/0.26), Exp 9 (-0.3107 vs -0.15/-0.17), Exp 10 (-0.2889 vs -0.21/-0.20) \u2014 all in the direction of more weight on the standard descending-validity hierarchy than either anti-validity theory allocates. Conversely both overpredict determinism: Exp 6 (0.78 vs 0.91/0.96) and Exp 15 (anti-misbound follow 0.15 vs 0.28/0.34 \u2014 subjects follow the anti-misbound decider far less, implying a non-trivial positional-primacy / other-hierarchy minority).\n\n(4) The shared anti-validity one-reason core is well supported (Exp 5: 0.4533 vs 0.44/0.49; Exp 12, 13, 14 all close), so the core should be retained.\n\nVerdict logic: pi_7's description explicitly forbids any mechanism that depends on carrier validity (its Exp-1 dip is exactly zero by construction), so no reparameterization of pi_7 can capture the -0.196 dip. pi_8's description hard-caps the leak at 0.18 and freezes the discipline windows, so it cannot capture either the -0.196 dip or the 0.156 tail dispersion. Since pi_7 is the better theory overall, the correct move is to replace the weaker theory (pi_8) with a new theory that synthesizes the surviving components of both: pi_7's diffuse momentary heterogeneity (which won Exp 2's direction) fused with a much stronger, uncapped interior-gated validity-respecting leak (pi_8's one good idea, at 2.5x strength) plus a positional-primacy minority to absorb Exp 15/17.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace pi_8 with a 'Diffuse Mixture Frugality with Strong Interior-Gated Validity Leak' theory, constructed as a synthesis of the surviving parts of both competitors:\n\n1. CORE (retained from both): one-reason decision making; population-central hierarchy anti-validity (consult ascending subjective weight, first strictly discriminating cue decides); exact validity ties anchor the order; history ignored; no discriminating cue gives exactly 0.5.\n\n2. STRONG INTERIOR-GATED LEAK (pi_8's mechanism, uncapped): when the anti-validity cascade's first discriminating expert on a trial carries a strictly interior validity (min + 0.10 < val < max - 0.10), mix toward the standard TTB (descending-validity) decider with probability q = rho_leak * reversal_coherence * mean_gap, with rho_leak ranging up to ~3.0 and NO 0.18 ceiling \u2014 target q in [0.35, 0.50] on high-coherence, large-gap designs (this reproduces Exp 1's observed dip of -0.196 at p_f ~ 0.72, and amplifies the negatives in Exp 9 toward -0.31 and Exp 10 toward -0.29). Weakest-carrier and near-top-carrier trials keep the leak OFF, preserving Exp 1's conditionality triad and Exp 5's strong reversed-cascade following (0.727).\n\n3. DIFFUSE HETEROGENEITY (pi_7's machinery, widened): per-subject stable distortion sigma_h*zeta plus per-trial momentary kappa*xi, both gated by cubed binding ambiguity A^3 and applied only when all validities are distinct; positional-salience tilt gamma drawn asymmetrically (roughly [-0.05, +0.12]). Critically, WIDEN the response-discipline ranges beyond even pi_7's: beta in [0.4, 2.2], epsilon in [0.06, 0.28], so the between-subject follow-protection tail reaches ~0.16 (Exp 2 observed 0.1556; pi_7's 0.112 and pi_8's 0.075 both undershoot) and the pooled determinism drops enough to fix Exp 6's overprediction (0.78 observed vs 0.91/0.96).\n\n4. POSITIONAL-PRIMACY MINORITY (new): a population mixture weight of roughly 0.15-0.25 on a pure earliest-listed-expert-first consultation order (with the remainder on the leak-augmented anti-validity core), which lowers the anti-misbound decider follow toward Exp 15's observed 0.15 (both current theories overpredict at 0.28/0.34) while keeping Exp 17's count-vs-primacy contrast near its observed +0.15 (the mixture's average first-cue distribution must be checked against this constraint during fitting).\n\n5. Drop pi_8's 'frozen discipline' and 'distinct-design bonus' machinery entirely (falsified by Exp 2), and drop pi_7's flatness claim (falsified by Exp 1). All other conventions (lexsort tie-breaking with free per-subject tie_break ~80% early position, softmax(beta) noise plus lapse epsilon, no integration across cues) carry over unchanged. The new theory's free parameters are rho_leak, the primacy mixture weight, gamma, sigma_h, kappa, beta, epsilon, tie_break, and per-subject zeta."
}
```

## Usage

```json
{
  "input_tokens": 40296,
  "output_tokens": 6135,
  "reasoning_tokens": 3895,
  "provider": "AkashML"
}
```
