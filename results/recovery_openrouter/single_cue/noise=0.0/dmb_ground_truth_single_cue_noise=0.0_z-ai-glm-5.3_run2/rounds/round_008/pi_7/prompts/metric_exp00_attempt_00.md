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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # THE LEAK-DIP METRIC\n    # --------------------\n    # Core contrast: within-subject difference between the anti-validity\n    # decider-follow rate on LEAK-ELIGIBLE trials (interior-validity\n    # carrier opposed by the top-validity expert) and on CONTROL trials\n    # (every other trial with a discriminating carrier).\n    #\n    # Fixed design: validities [0.5, 0.5, 0.75, 0.75, 1.0, 1.0], ascending\n    # with display index, so the anti-validity consultation order is the\n    # index order (ties are inert: tied cues never oppose in any trial).\n    #\n    # Advocated theory: follow probability is p_f on EVERY discriminating\n    # trial regardless of carrier validity, position, margin, or coalition\n    #   -> expected dip = EXACTLY 0.\n    # Competing theory: on the 8 leak cells the trial-conditional\n    # validity-respecting leak mixes q = min(rho_leak*0.956*0.25, 0.18)\n    # (>= 0.120 forced in this design) toward the TTB winner, giving\n    # follow = (1-q)*p_f + q*(1-p_f) there, while control cells stay at p_f\n    #   -> expected dip = -q*(2*p_f - 1) in [-0.093, -0.050], bounded away\n    #      from zero by the competing's own gate formulas.\n    #\n    # Works on a single subject's slice (48 leak + 42 control trials) and\n    # on the pooled frame alike.\n\n    VALIDITIES = [0.5, 0.5, 0.75, 0.75, 1.0, 1.0]\n    n = len(VALIDITIES)\n    vmin = min(VALIDITIES)\n    vmax = max(VALIDITIES)\n\n    def to_vec(x):\n        if isinstance(x, str):\n            return [int(ch) for ch in x if ch in \"01\"]\n        return [int(v) for v in x]\n\n    # anti-validity order: ascending validity (exact ties anchor; tied cues\n    # never oppose in this design, so within-tie order is inert)\n    order_anti = sorted(range(n), key=lambda j: VALIDITIES[j])\n    # standard TTB order: descending validity\n    order_ttb = sorted(range(n), key=lambda j: -VALIDITIES[j])\n\n    leak_follows = []\n    ctrl_follows = []\n\n    a_col = data[\"option_a_ratings\"].tolist()\n    b_col = data[\"option_b_ratings\"].tolist()\n    r_col = data[\"response\"].tolist()\n\n    for a_raw, b_raw, resp in zip(a_col, b_col, r_col):\n        a = to_vec(a_raw)\n        b = to_vec(b_raw)\n        if len(a) != n or len(b) != n:\n            return 0.0\n\n        # anti-validity decider: first strictly discriminating cue in\n        # ascending-validity order\n        carrier = None\n        favored = None\n        for j in order_anti:\n            if a[j] != b[j]:\n                carrier = j\n                favored = 0 if a[j] > b[j] else 1\n                break\n        if carrier is None:\n            # full-tie trial (T16): no decider, excluded\n            continue\n\n        # TTB decider: first strictly discriminating cue in\n        # descending-validity order\n        winner_ttb = None\n        for j in order_ttb:\n            if a[j] != b[j]:\n                winner_ttb = 0 if a[j] > b[j] else 1\n                break\n\n        # leak-eligible cell: strictly INTERIOR carrier validity\n        # (min + 0.10 < val < max - 0.10) AND the TTB winner exists and\n        # OPPOSES the anti-validity winner. This is exactly the\n        # competing theory's trial-conditional gate; the advocated theory\n        # has no mechanism tied to this conjunction.\n        interior = (vmin + 0.10) < VALIDITIES[carrier] < (vmax - 0.10)\n        opposed = (winner_ttb is not None) and (winner_ttb != favored)\n\n        follow = (int(resp) == favored)\n        if interior and opposed:\n            leak_follows.append(follow)\n        else:\n            ctrl_follows.append(follow)\n\n    if not leak_follows or not ctrl_follows:\n        return 0.0\n\n    dip = float(np.mean(leak_follows)) - float(np.mean(ctrl_follows))\n    return dip",
  "rationale": "THE METRIC IS THE WITHIN-SUBJECT 'LEAK DIP'. For every trial I reconstruct the anti-validity decider (first strictly discriminating cue in ascending-validity order; validities [0.5,0.5,0.75,0.75,1,1] ascend with index and tied cues never oppose, so the reconstruction is exact and tie_break/gamma inert). A trial is LEAK-ELIGIBLE iff the carrier is strictly interior (0.75) AND the standard TTB decider (descending validity) picks the opposite option \u2014 exactly the competing theory's trial-conditional gate (trials 1-8, 48 trials/subject). All other discriminating trials are CONTROLS (trials 9-15, 42 trials/subject); trial 16 (full tie) is excluded. The metric is P(follow anti-decider | leak cells) - P(follow anti-decider | control cells), computed within subject, so each subject's own p_f cancels. PREDICTIONS: the advocated theory's follow probability is flat p_f on every discriminating trial (no mechanism depends on carrier validity, position, margin, or coalition), so its dip is exactly 0 in expectation, with only binomial noise (per-subject SD ~ sqrt(p_f(1-p_f)(1/48+1/42)) ~ 0.095, giving between-subject variance ~0.009). The competing theory is FORCED by its own gate formulas to leak q = min(rho_leak*0.956*0.25, 0.18) >= 0.120 on the leak cells (its follow there is (1-q)p_f + q(1-p_f)), producing a dip of -q(2p_f-1) in [-0.093, -0.050] (mean ~ -0.08 since q saturates at 0.18 for ~90% of rho_leak's range and p_f in [0.71, 0.75]). The control cells are matched on decider validity (T9/T10/T15 are interior-carrier cells where TTB concurs) and on opposition presence (T11-T14 keep the higher-validity opposition but demote the carrier to 0.5, where the competing's gate switches the leak OFF), so both theories predict p_f there \u2014 the contrast isolates the trial-conditional leak as the sole source of any dip. Expected Welch t at N=25: gap ~0.08 against per-subject SD ~0.095 in both conditions (binomial floor dominates both theories, so between-subject variances are nearly equal at ~0.009), giving t ~ 3.0, p ~ 0.004 < alpha = 0.01. The metric is well-defined on a single subject's slice (48 leak + 42 control trials) and on the pooled frame, and it is two-sided-falsifiable: a population dip reliably below -0.05 refutes the advocated's flatness claim; a dip indistinguishable from 0 refutes the competing's forced q >= 0.120 leak."
}
```

## Usage

```json
{
  "input_tokens": 9553,
  "output_tokens": 17459,
  "reasoning_tokens": 15679,
  "provider": "Sail Research"
}
```
