# metric_exp00_attempt_00

## System Prompt

You are a psychology researcher proposing a metric in the Decision Making (Binary Features) domain.

Your goal is adversarial: propose a metric that DISCRIMINATES the two theories — i.e., its value, computed on data simulated under your advocated theory, should be as far as possible from its value computed on data simulated under the competing theory. The direction of the gap does not matter; what matters is that the two theories produce visibly different numbers on this metric. The metric is computed on the data collected from the experimental design provided in the prompt. Produce a metric where you're prediction will be much more accurate than the competing theory's prediction on human data.

Your metric is a Python function

    metric(data: pd.DataFrame) -> float

Available imports inside `metric`:
- numpy as np
- pandas as pd

The system evaluates your metric in two ways and reports the pair as `point_estimate (var=between_subject_variance)` everywhere downstream:
- `point_estimate` is `metric(data)` applied to the FULL pooled DataFrame (all subjects together) — the canonical scalar;
- `between_subject_variance` is the population variance (`ddof=0`) of `metric(subj_df)` re-applied per `subject_id`, summarising how stable the metric is across subjects. If your metric only makes sense on multi-subject data this will fall back to `n/a` and the metric is rejected (the acceptance test below cannot run without it). Prefer metrics that work both on the pooled DataFrame and on a single subject's slice.

Acceptance rule: the system simulates each theory and runs Welch's two-sample t-test on `(point_estimate_self, between_subject_variance_self, N)` vs. `(point_estimate_adv, between_subject_variance_adv, N)`, where N is the number of HUMAN subjects the experiment will actually be run with (a fixed small number, currently 25). Your metric is admitted iff the two-sided p-value is below the significance level (currently alpha=0.01). Implication: a large between-theory gap is NOT enough — if either theory's metric is also highly variable across subjects, N humans won't reliably distinguish them and the metric will be rejected. Aim for contrasts that are both large in mean AND tight per subject.

Do NOT propose metrics that are trivially true for your theory.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

Each subject completes ~96 trials in a single block, with order randomized independently per subject. On every trial the subject sees two options A and B, each described by `n_features` binary expert ratings (each 0 or 1). The per-feature validities and n_features are fixed per experiment (design-time choices). Validities are communicated to the subject in the instructions. Both `n_features` and `validities` are exposed to your `predict` via the `parameters` dict. The subject chooses A or B; no correctness feedback is provided after the choice.

## CHOSEN EXPERIMENTAL DESIGN
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



## ADVOCATED THEORY
**Description:** Anchored Anti-Validity Frugality with Validity-Respecting Leak (frozen-discipline variant). People are one-reason decision makers: experts are consulted one at a time and the first strictly discriminating expert decides; no integration across cues. The population-central hierarchy is anti-validity (distrust-the-weakest) and the population is typeless — every subject is a continuous perturbation around that core. Three graded, design-scaled mechanisms modulate the core. (1) GAP-PROPORTIONAL ANCHORING: per-trial attention noise on expert j is scaled by the local validity gaps adjacent to j in the subjective order, gated by cubed binding ambiguity, so validity-isolated experts anchor the order while dense clusters diffuse; exact validity ties anchor the order entirely. (2) VALIDITY-RESPECTING LEAK, now TRIAL-CONDITIONAL and implemented as a probability mixture: the leak toward the standard TTB (descending-validity) order fires only when the anti-validity cascade's first discriminating expert on that trial is a strictly INTERIOR validity (min_val + 0.10 < val < max_val - 0.10), with design-level probability q = min(rho_leak * reversal_coherence * mean_gap, 0.18). When the anti-decider's carrier is a weakest or near-top expert, the leak is switched off — protecting the pure anti-validity-follow designs — while genuinely mid-validity carriers let a fraction of trials flip toward the strongest discriminator, generating below-chance reversed-cascade following where the data demand it. (3) DESIGN-SCALED RESPONSE DISCIPLINE with a DISTINCT-DESIGN BONUS: p_f = (1-eps)*sigmoid(beta_eff) + eps/2 with beta_eff = beta*(1 + boost*(1-A))*(1 + b_distinct) when all validities are distinct — distinct-validity designs (which the data show want tighter decider-following) get a bounded bonus, tie designs are untouched. Response-discipline and diffusion parameters are FROZEN at narrow windows reproducing the accepted base's behavior, so only the leak strength and the distinct bonus are free.

**Parameters:**
- beta: [0.80, 1.00]
- boost: [0.28, 0.42]
- b_distinct: [0.0, 0.25]
- epsilon: [0.12, 0.16]
- gamma: [-0.01, 0.03]
- kappa: [7.5, 9.5]
- sigma_h: [0.18, 0.28]
- rho_leak: [0.5, 3.0]
- zeta: [(-1, 1)] * n_features
- tie_break: [0, 1]
- validities: validities

**`predict source code`:**
```python
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
```

**`policy source code`:**
```python
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
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


## DATA SCHEMA
Your metric receives a tidy per-trial pandas DataFrame stacking all subjects (rows grouped by `subject_id`, in trial order). Columns:
- subject_id: Subject identifier (one row per trial per subject).
- option_a_ratings: List of n_features binary expert ratings (each 0 or 1) for option A on this trial.
- option_b_ratings: List of n_features binary expert ratings (each 0 or 1) for option B on this trial.
- response: 0 if subject chose A, 1 if subject chose B.

## IMPLEMENTATION GUARDRAILS
Any column in the schema above whose description names a list / tuple / np.ndarray (i.e. a per-trial sequence of values) holds non-scalar cells. Those cells are NOT hashable, so operations that hash row values fail with `TypeError: unhashable type: 'list'`. Treating `<seq_col>` as a placeholder for any such sequence-valued column:
- Avoid: `data.groupby('<seq_col>')`, `data['<seq_col>'].value_counts()`,     `data['<seq_col>'].nunique()`, `data['<seq_col>'].unique()` (returns     an object array but downstream `set()` / `in dict` will crash),     `set(data['<seq_col>'])`, `data['<seq_col>'].isin([...])` against list     values, or using a list cell as a dict key.
- If you need a hashable surrogate, project to one first, e.g.:
    - `data['<seq_col>_key'] = data['<seq_col>'].apply(tuple)` then group by `<seq_col>_key`
    - `data['<seq_col>_str'] = data['<seq_col>'].apply(lambda x: ''.join(map(str, x)))`
    Scalar columns (ints, floats, strings like `subject_id`, integer     responses, etc.) hash fine and can be used directly.
- Generator expressions inside function calls like `map()` or `join()` MUST be     parenthesized. For example:
    - WRONG: `map(str, int(v) for v in x)` → SyntaxError
    - RIGHT: `map(str, (int(v) for v in x))` or use a list comp: `[str(int(v)) for v in x]`
- Always verify your code is syntactically valid Python before returning it.

## METRICS YOU ALREADY TRIED AND FAILED ON
Each entry below is a metric you previously proposed in this round that did NOT discriminate the two theories at the human sample size — either it errored, its between-subject variance was unavailable, or Welch's t-test on `(self mean, self var, N)` vs. `(adv mean, adv var, N)` returned p ≥ alpha. The `outcome` line is the simulation result (means, between-subject variances, t-statistic and p-value at the human N) on the same `data_self` / `data_adv` your next metric will be evaluated on. Use the numbers to see where your hypothesised contrast collapsed — small mean gap, large per-subject variance, or both — and propose something qualitatively different. Don't repeat the same idea with cosmetic tweaks.
(none yet)

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # TILT PSYCHOMETRIC, HIGH RUNGS (primary signature of the design).\n    #\n    # On the eight mirror \"tilt\" trials T3-T10, exactly two experts\n    # discriminate, both from the weak block (positions 2-5), and they\n    # endorse OPPOSITE options. The consultation order between an\n    # earlier-position expert X and a later-position expert Y is decided by\n    # the subject's positional-salience tilt gamma: X is consulted before Y\n    # iff gamma > (val_X - val_Y) / (sal_X - sal_Y). The included trial\n    # pairs realize thresholds g = 0.035 (T3/T4: pair {2,4}), 0.050\n    # (T5/T6: pair {3,4}), 0.057 (T7: pair {2,5}), 0.075 (T8: pair {3,5}),\n    # and 0.100 (T9/T10: pair {4,5}).\n    #\n    # The advocated theory has gamma confined to [-0.01, +0.03], which is\n    # BELOW every included threshold: it is RANGE-FORCED to always consult\n    # the later (weaker) expert first, so it picks the option endorsed by\n    # the EARLIER expert only via response noise, at rate 1 - p_f\n    # (~0.25, tight across subjects).\n    #\n    # The competing theory draws gamma ~ Uniform[-0.05, +0.12]; a large\n    # fraction of subjects sit above the rung thresholds, so a graded\n    # fraction of these trials follows the earlier expert's endorsement\n    # (~0.43 pooled, with genuine between-subject spread).\n    #\n    # The metric = P(choose the option endorsed by the earlier-listed of\n    # the two discriminating weak-block experts), pooled over the 48\n    # qualifying trials per subject. Mirror pairs cancel side bias; the\n    # lowest rung (pair {2,3}, g=0.020, where the advocated still flips\n    # 25% of the time) is EXCLUDED to keep the advocated prediction\n    # range-forced and its between-subject variance minimal.\n    a_col = data[\"option_a_ratings\"]\n    b_col = data[\"option_b_ratings\"]\n    r_col = data[\"response\"]\n\n    vals = []\n    for a_raw, b_raw, resp in zip(a_col, b_col, r_col):\n        try:\n            a = [int(round(float(x))) for x in list(a_raw)]\n            b = [int(round(float(x))) for x in list(b_raw)]\n        except (TypeError, ValueError):\n            continue\n        if len(a) != 6 or len(b) != 6:\n            continue\n        # positions where the two options differ\n        disc = [j for j in range(6) if a[j] != b[j]]\n        if len(disc) != 2:\n            continue\n        j1, j2 = disc  # j1 < j2 (disc is built in ascending position)\n        # both discriminating experts must be weak-block experts\n        # (positions 2..5); trials where a strong expert (0 or 1)\n        # discriminates are controls, not tilt trials\n        if j1 < 2:\n            continue\n        # exclude the lowest rung {2,3} (g = 0.020, T1): the advocated\n        # flips there 25% of the time, which would blur the contrast\n        if (j1, j2) == (2, 3):\n            continue\n        # option endorsed by the EARLIER-listed discriminating expert\n        early_fav = 0 if a[j1] == 1 else 1\n        vals.append(1.0 if int(resp) == early_fav else 0.0)\n\n    if len(vals) == 0:\n        return 0.5\n    return float(np.mean(vals))",
  "rationale": "The design's decisive axis is the consultation-order tilt gamma. With strictly decreasing, all-distinct validities, binding ambiguity is exactly 0, so every diffusion/anchoring/leak channel is structurally OFF in both theories (the leak is additionally off on all weak-block deciders), leaving gamma as the only active difference. On the eight mirror tilt trials T3-T10, exactly two weak-block experts discriminate in opposition, and the earlier-listed one is consulted first iff gamma exceeds a known rung threshold (0.035, 0.050, 0.057, 0.075, 0.100). The metric is the pooled probability of choosing the option endorsed by the earlier-listed discriminating expert over these 48 trials/subject. Advocated: gamma <= 0.03 < 0.035 = lowest included rung, so it is range-forced to consult the later/weaker expert first on EVERY included trial; the metric equals 1 - p_f ~ 0.25 (range ~[0.17, 0.32]) with tiny between-subject spread (SD ~0.07, dominated by binomial noise over 48 trials). Competing: gamma ~ U[-0.05, +0.12] puts ~50% of subjects above the lowest rung, so subjects follow the earlier expert on a graded fraction of trials; the pooled metric is ~0.43 with SD ~0.18 (genuine step-function heterogeneity). Expected Welch t at N=25: gap ~0.19, SE ~0.04, t ~4.5-4.8, p < 0.001 \u2014 well below alpha = 0.01. Design safeguards baked in: mirror pairs (T3/T4, T5/T6, T9/T10) cancel side bias; the lowest rung {2,3} (g=0.020, where the advocated still flips with probability 0.25) is excluded so the advocated prediction stays range-forced with minimal variance; strong-expert trials (T12, T13, T14), single-cue trials (T2, T11, T15) and the full tie (T16) are excluded, keeping the metric a pure tilt psychometric; the metric is a simple proportion, so it is stable both pooled and per subject."
}
```

## Usage

```json
{
  "input_tokens": 10054,
  "output_tokens": 46586,
  "reasoning_tokens": 45154,
  "provider": "Sail Research"
}
```
