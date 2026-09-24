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
**Validities (n_features=9):** [0.99, 0.99, 0.94, 0.75, 0.5, 0.5, 0.5, 0.5, 0.5]

**Trial pairs (n=24):**
  trial 1: A=[1, 1, 1, 1, 0, 0, 1, 1, 1]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1]
  trial 2: A=[0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 1, 1, 1]
  trial 3: A=[1, 1, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 0, 0]
  trial 4: A=[0, 0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 0, 0, 1, 1, 1, 0, 0]
  trial 5: A=[1, 1, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 1, 0]
  trial 6: A=[1, 1, 1, 1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 1, 1, 1, 0, 1]
  trial 7: A=[0, 0, 0, 1, 1, 1, 1, 0, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 1]
  trial 8: A=[1, 1, 1, 1, 0, 0, 1, 0, 1]  B=[0, 0, 0, 0, 1, 0, 1, 0, 1]
  trial 9: A=[0, 0, 0, 0, 1, 0, 1, 0, 1]  B=[1, 1, 1, 1, 0, 0, 1, 0, 1]
  trial 10: A=[1, 0, 0, 1, 1, 0, 1, 0, 1]  B=[0, 0, 1, 1, 1, 0, 1, 0, 1]
  trial 11: A=[1, 0, 0, 1, 1, 0, 1, 0, 1]  B=[0, 0, 1, 1, 0, 0, 1, 0, 1]
  trial 12: A=[1, 0, 0, 1, 1, 1, 0, 1, 0]  B=[0, 0, 1, 1, 0, 0, 0, 1, 0]
  trial 13: A=[1, 1, 0, 0, 1, 1, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0, 0, 1, 0]
  trial 14: A=[1, 0, 0, 1, 1, 1, 1, 1, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0]
  trial 15: A=[0, 1, 1, 0, 0, 0, 0, 1, 0]  B=[1, 1, 0, 0, 1, 1, 1, 1, 0]
  trial 16: A=[1, 1, 0, 1, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 0, 1, 0]
  trial 17: A=[0, 0, 1, 1, 1, 1, 0, 1, 0]  B=[1, 1, 0, 1, 0, 0, 0, 1, 0]
  trial 18: A=[1, 1, 0, 1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 1, 1, 1, 0, 1, 0]
  trial 19: A=[0, 1, 0, 1, 1, 1, 0, 1, 0]  B=[1, 1, 0, 1, 0, 0, 0, 1, 0]
  trial 20: A=[1, 0, 1, 0, 1, 1, 1, 0, 0]  B=[1, 0, 1, 0, 0, 0, 0, 1, 1]
  trial 21: A=[1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1, 0, 1]
  trial 22: A=[0, 0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1, 1]
  trial 23: A=[1, 1, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 1, 1, 1, 0, 0]
  trial 24: A=[1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1]

**Rationale:** CORE LOGIC. CREDT (advocated) and BODT (competing) share the identical architecture (sign votes x panel-relative weights x low-gain softmax x lapse) and agree on the plateau (w_p), on the trust ceiling (~0.981), and on the consensus-gated chance flank (identical g0/s_gap/lambda_ext machinery). They differ structurally in exactly one place: the DEPTH of distrust of an above-ceiling outlier. CREDT's discount is an UNBOUNDED additive term rho*(P + C*sig_gap); BODT's is a BOUNDED saturating shrinkage toward w_floor (~-0.28) plus a composition route that is multiplied by far(gap), a logistic gate its own description commits to being 'essentially closed at gap ~0.05'. The design therefore engineers the single panel configuration that maximally separates these two formulas: a 0.99 claim sitting exactly 0.05 above its next rung (gap = g_star, CREDT's PEAKED spurious-precision bump P ~ 1) on a CHANCE-LADEN panel (c_frac = 5/9 = 0.556 > both theories' c_half), with TWO tied 0.99 experts so the lever is applied twice. Panel: [0.99, 0.99, 0.94, 0.75, 0.50, 0.50, 0.50, 0.50, 0.50]. On this panel CREDT's composition amplifier fires fully at the peak (C ~ 1.24, sig_gap ~ 0.63, no far-gate), giving w_top in [-2.03, -1.36] across its ENTIRE parameter box -- each 99% expert is worth roughly -1.7, i.e., about two trusted cues AGAINST. BODT's bounded shrinkage caps w_top_mod at w_floor ~ -0.28 and its composition route is strangled by far(0.05) in [0.06, 0.19], giving w_top in [-0.40, +0.09] across its ENTIRE box. THE TWO w_top BOXES ARE DISJOINT with a gap of ~1.0-1.6 weight units per 99% expert, while every other weight is common currency: w(0.94) = w(0.75) = w_p exactly in both, and the gated chance weight is 0.657 (CREDT) vs 0.641 (BODT) -- the kappa_lo difference (0.20 vs 0.27) is suppressed to ~0.016 per cue by the ~0.78-open consensus gate, which is itself nearly identical across boxes. The design thus isolates ONE structural disagreement -- how much a 99%-atop-94% claim on a coin-flip-laden panel is distrusted -- and measures it psychometrically. 24 unique pairs x K = 96//24 = 4 = 96 trials.

FAMILY S (flagship: the 0.99 PAIR vs varying counterweights). S1 (A wins both 99%s + 0.94 + 0.75; B wins two coin-flippers): CREDT p(A) = .24 (box [.19, .30]); BODT p(A) = .55 (box [.47, .58]) -- SIGN-DISJOINT. S6 (same A-stack vs ONE coin-flipper): CREDT .29 [.23, .35] vs BODT .57 [.52, .63] -- sign-disjoint. S2 (both 99%s vs 0.94+0.75): CREDT .12 [.09, .19] vs BODT .34 [.26, .35]. S3, S4, K1 extend the counterweight ladder; K1 is the unanimity cell (A wins ALL five real experts incl. both 99%s; B wins four coin-flippers): CREDT .20 [.14, .24] vs BODT .47 [.39, .51] -- even BODT's near-tie here is diagnostic, since its bounded floor cannot push 2*w_top below -0.80, so a subject p(A) <= 0.35 on K1 falsifies BODT outright, while p(A) >= 0.30 falsifies CREDT outright. Mirrors S1m/S2m/S4m/S6m control side bias.

FAMILY R (single-0.99 crossover psychometric -- the key quantitative signature). A wins one 99% expert plus j coin-flippers; B wins the 0.94 expert; j = 0..4 (R0-R4). CREDT centers: .26/.31/.38/.44/.50; BODT centers: .41/.47/.55/.57/.62. The psychometric CROSSES p=0.5 at j* ~ 3.9 for CREDT but j* ~ 1.6 for BODT -- a ~2.3-coin-flip separation in crossover location, equivalent to ~1.3 evidence units, that no parameter setting in either box can close (CREDT's box at R3 is [.38, .47], BODT's is [.53, .62] -- disjoint; at R2 they are edge-disjoint [.32, .41] vs [.48, .55]). R3 is the sharpest single cell: one 99% + three coin-flippers vs one 94% expert -- CREDT says B, BODT says A, boxes disjoint. R3m mirrors.

FAMILY D/V (0.99 vs 0.94 / coin-flippers head-to-head). D2 (both 99%s vs 0.94 + two coin-flippers): CREDT .11 [.07, .12] vs BODT .31 [.22, .34]. V1 (one 99% vs two coin-flippers): CREDT .21 [.19, .25] vs BODT .38 [.33, .41]. Mirrors included. These are the honest ceiling probes: if subjects follow a double-99% endorsement over a 94% + 75% panel at p >= 0.27 (S2) or p(A) >= 0.30 on K1, CREDT is falsified with zero free parameters (its w_top cannot exceed -1.36); conversely if subjects abandon the 99%-endorsed option at p >= 0.47 wherever BODT demands >= 0.50 (S1) or at p <= 0.42 on R2/R3, BODT's bounded-floor claim is falsified outright.

CONTROLS. T1 (the two 99% experts split between A and B): BOTH theories must predict EXACTLY 0.50 for every parameter (tied softmax is uniform, lapse injects uniform) -- a parameter-free side/display-bias assay. Z1 (three coin-flippers vs two, nothing else discriminates): both theories agree (.57 vs .55), anchoring the common gated-chance currency -- a deviation here indicts the SHARED gating machinery, not the disputed distrust depth. G1/G1m (both trusted experts + three coin-flippers vs both 99%s + two coin-flippers): agreement anchors on choice whose CONFIDENCE also dissociates -- CREDT [.90, .95] vs BODT [.68, .76], disjoint boxes even on the anchor, because the 99%s on B's side actively help A under CREDT but are merely discounted under BODT. Tie patterns mix 0/0 and 1/1 throughout so 'count the 1s' is unviable (e.g., S1 has MORE 1s for A while CREDT predicts A loses by 4 evidence units; K1 has 5-vs-4 ones for A while both theories favor B; R0/V1/T1 have exactly tied 1-counts). No all-tie trials; every trial has 3-8 discriminators among 9 experts requiring genuine scanning.

KEY QUANTITATIVE METRICS: (i) crossover location j* in the R-ladder (CREDT ~3.9 vs BODT ~1.6); (ii) p(A) on the sign-flip cells S1/S6/R2/R3 (CREDT .24/.29/.38/.44 vs BODT .55/.57/.55/.57); (iii) the S-family counterweight slope (how much opposition the 99% endorsement needs before subjects abandon it); (iv) anchor confidence on G1 (0.90 vs 0.71 at centers, disjoint boxes); (v) the K1 unanimity cell. PREDICTED PATTERN IF CREDT IS RIGHT (centers): S1 .24, S2 .12, S3 .18, S4 .15, S6 .29, K1 .20, R0-R4 .26/.31/.38/.44/.50, D2 .11, V1 .21, Z1 .57, T1 .50, G1 .92 -- subjects treat each 99%-atop-94% endorsement on this coin-flip-laden panel as ~2 trusted cues AGAINST, crossing over only when ~4 coin-flippers back it. IF BODT IS RIGHT: S1 .55, S2 .34, S3 .43, S4 .39, S6 .57, K1 .47, R0-R4 .41/.47/.55/.57/.62, D2 .31, V1 .38, Z1 .55, T1 .50, G1 .71 -- the same endorsement is worth only ~0.14 cues against (near-zero), crossovers at ~1.6 coin-flippers, and near-tie on the unanimity cell. The w_top boxes are disjoint by construction, so the two predicted patterns are mutually exclusive essentially everywhere. DIFFERENTIATION FROM PRIOR EXPERIMENTS: [10] instantiated the large-edge (gap 0.09) chance-laden regime against a monotone competitor and [11] the gap-0.05 COHERENT regime (c_frac 0.25, amplifier sub-threshold); no prior design instantiated the CONJUNCTION of the peaked bump (gap = g_star) with a composition-amplified chance-laden panel -- CREDT's absolute maximum-distrust configuration, where its additive discount (-1.7 per expert) and BODT's far-gated bounded floor (-0.14 per expert) are separated by more than an entire plateau weight -- and no prior round pitted CREDT against BODT at all. The tied 0.99 PAIR (doubling the lever), the R-ladder crossover psychometric, and the disjoint anchor-confidence boxes are new quantitative signatures. Sides are balanced via mirrors in every family (8 A-tilted / 13 B-tilted / 3 near-neutral under CREDT centers; near-balanced under BODT centers, reflecting that the B-tilt is itself CREDT's signature prediction).

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Consensus-Relative Edge-Distrust Tallying (CREDT), corroborated-precision edition. People integrate ALL binary expert ratings as sign votes (+1/-1/0) into a zero-sum evidence score. The subjective weight of an expert is set by: (1) a trust plateau over stated validity [v_lo, v_hi] with the ceiling just above 0.98; (2) a small negative chance flank (w(0.5) = -kappa_lo); (3) consensus gating of that flank by the panel's distrust signal; and (4) the novel claim: distrust of the panel-max expert is distrust of UNCOPROBORATED PRECISION and is NON-MONOTONE in the edge (gap) the top claim holds over the next rung. A dense ladder (gap ~ 0.01) corroborates near-certainty (the panel itself approaches the claim) -> trust. A small uncorroborated edge (gap ~ 0.05, e.g. 0.99 atop 0.95) reads as spurious precision -- the same underlying competence with inflated confidence -> maximal distrust. A large edge (gap >= ~0.09) on a coherent, competent panel reads as a categorically better instrument and is taken at face value -> trust returns. But a far outlier on an incoherent, chance-laden scale (a 0.99 floating above a panel stuffed with coin-flip experts) is unearned certainty -> collapse again, via a composition-amplified route that fires only on chance-laden panels. Choice is a low-gain softmax plus a uniform lapse.

**Parameters:**
- w_p: [0.88, 0.92]
- v_lo: [0.52, 0.54]
- v_hi: [0.9805, 0.9825]
- kappa_lo: [0.19, 0.21]
- g0: [0.041, 0.045]
- s_gap: [0.012, 0.014]
- rho: [1.30, 1.42]
- d_w: [0.35, 0.45]
- lambda_ext: [0.35, 0.45]
- g_star: [0.046, 0.050]
- s_l: [0.011, 0.013]
- s_r: [0.030, 0.036]
- c_half: [0.34, 0.38]
- s_c: [0.05, 0.07]
- c_amp: [1.22, 1.36]
- beta: [0.41, 0.43]
- epsilon: [0.04, 0.06]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Consensus-Relative Edge-Distrust Tallying (CREDT),
    # corroborated-precision edition.
    # Stimulus: array of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are binary expert ratings (0/1).
    # Each feature j casts a sign vote sign(a_j - b_j), weighted by a
    # subjective trust function of the expert's STATED validity v,
    # evaluated RELATIVE to the panel:
    #   (i) TRUST PLATEAU UP TO NEAR-CERTAINTY: all v in [v_lo, v_hi]
    #       (v_hi ~ 0.981, i.e. a 98% expert is still INSIDE the
    #       plateau) get approximately equal positive weight w_p.
    #  (ii) CHANCE FLANK: v near 0.5 gets a small NEGATIVE weight,
    #       reaching -kappa_lo at v = 0.5, rising linearly to the
    #       plateau at v_lo.
    # (iii) CONSENSUS GATING of the flank: when an above-ceiling
    #       claim is present, gate = lambda_ext + (1-lambda_ext)
    #       * sig_gap -- distrust generalizes to the scale and
    #       chance cues are counted (weight -> w_p) rather than
    #       inverted; at-or-below-ceiling panels keep the full
    #       negative flank. UNCHANGED from the accepted base.
    #  (iv) ABOVE-CEILING DISTRUST OF THE PANEL-MAX EXPERT IS
    #       NON-MONOTONE IN THE RUNG GAP (distrust of uncorroborated
    #       precision): the discount is rho * ( P(gap) + C * sig_gap ),
    #       where P is an asymmetric PEAKED bump (maximal at gap ~
    #       g_star ~ 0.05, near zero for dense ladders gap <= 0.02,
    #       receding for large edges gap >= 0.09 on coherent panels)
    #       and C * sig_gap is a composition-amplified route that
    #       re-imposes full collapse for far outliers on CHANCE-LADEN
    #       panels (C fires only when the fraction of chance-flank
    #       cues exceeds c_half). The extremity term d_w * (v_top -
    #       v_hi) is unchanged.
    # Choice is a low-gain softmax (inverse temperature beta) over
    # the two mirrored scores, plus a uniform lapse (epsilon).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    n_features = stim.shape[1]
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    w_p = float(parameters["w_p"])
    v_lo = float(parameters["v_lo"])
    v_hi = float(parameters["v_hi"])
    kappa_lo = float(parameters["kappa_lo"])
    g0 = float(parameters["g0"])
    s_gap = float(parameters["s_gap"])
    rho = float(parameters["rho"])
    d_w = float(parameters["d_w"])
    lambda_ext = float(parameters["lambda_ext"])
    g_star = float(parameters["g_star"])
    s_l = float(parameters["s_l"])
    s_r = float(parameters["s_r"])
    c_half = float(parameters["c_half"])
    s_c = float(parameters["s_c"])
    c_amp = float(parameters["c_amp"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    a, b = stim[0], stim[1]

    # Feature-wise votes: +1 where A wins the feature, -1 where B
    # wins, 0 on ties (a tied cue is uninformative for the pair).
    votes = np.sign(a - b)

    v = np.clip(val, 0.5, 1.0)

    # ---- Panel structure: top validity and its gap to the
    # next-highest DISTINCT validity in the panel. ----
    v_top = float(v.max())
    below = v[v < v_top - 1e-9]
    # If every expert shares the top validity, the panel offers no
    # corroboration ladder; treat the next rung as chance (0.5).
    v_second = float(below.max()) if below.size > 0 else 0.5
    gap = v_top - v_second

    # Outlier = a claim ABOVE the trust ceiling. With v_hi ~ 0.981
    # a 0.98-topped panel (Exp 8) stays inside the trusted plateau;
    # only 99%+ claims enter the distrust branch.
    outlier = v_top > v_hi
    if outlier:
        x = (gap - g0) / s_gap
        # Numerically stable logistic via the identity
        # sigmoid(x) = 0.5 * (1 + tanh(x / 2)).
        sig_gap = 0.5 * (1.0 + np.tanh(0.5 * x))
    else:
        sig_gap = 0.0

    # ---- (iii): consensus gate for the chance flank. UNCHANGED
    # from the accepted base (same g0 / s_gap, same lambda_ext). ----
    if outlier:
        gate = lambda_ext + (1.0 - lambda_ext) * float(sig_gap)
    else:
        gate = 0.0

    # ---- (i) + (ii) + (iii): plateau with a consensus-gated
    # chance flank. UNCHANGED from the accepted base. ----
    w = np.full(n_features, w_p, dtype=float)
    flank = v < v_lo
    if np.any(flank):
        denom = max(v_lo - 0.5, 1e-6)
        w_flank = -kappa_lo + (w_p + kappa_lo) * np.clip(
            v[flank] - 0.5, 0.0, None
        ) / denom
        # Consensus gating: with an above-ceiling claim in the
        # panel, chance cues are counted (weight -> w_p) rather
        # than inverted; with no above-ceiling claim the full
        # negative flank is retained.
        w[flank] = (1.0 - gate) * w_flank + gate * w_p

    # ---- (iv): above-ceiling discount of the panel-max expert,
    # NON-MONOTONE in the rung gap (the only edited block). ----
    if outlier:
        # (a) Panel composition: fraction of chance-flank cues.
        c_frac = float(np.mean(flank))
        # Composition amplifier: distrust of a far-outlying top
        # claim re-imposes itself only when the panel's validity
        # scale is chance-laden (incoherent), e.g. a 0.99 floating
        # above four coin-flip experts.
        x_c = (c_frac - c_half) / s_c
        C = c_amp * 0.5 * (1.0 + np.tanh(0.5 * x_c))
        # (b) Peaked edge-distrust bump: maximal when the top
        # expert's edge over the next rung is small but
        # uncorroborated (spurious precision, gap ~ g_star ~ 0.05),
        # near zero for dense ladders (corroborated precision) and
        # receding for large edges on coherent panels (a
        # categorically better instrument, taken at face value).
        width = s_l if gap < g_star else s_r
        P = float(np.exp(-0.5 * ((gap - g_star) / width) ** 2))
        w_top = (
            w_p
            - d_w * (v_top - v_hi) / max(1.0 - v_hi, 1e-6)
            - rho * (P + C * float(sig_gap))
        )
        # All experts tied at the panel maximum share the distrust
        # (e.g., a pair of isolated 0.99 super-experts).
        w[v >= v_top - 1e-9] = w_top

    # Weighted evidence score. B's score is the mirror of A's
    # because every vote is zero-sum across the two options.
    s_a = float(np.dot(w, votes))
    scores = np.array([s_a, -s_a], dtype=float)

    # Numerically stable low-gain softmax. When all cues tie
    # (s_a == 0) the softmax is exactly uniform.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    # Independent lapse: with probability epsilon output a uniform
    # pick over the two options.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** Bounded Outlier-Distrust Tallying (BODT). Subjects integrate all binary expert ratings as sign votes (+1/-1/0) into a zero-sum evidence score, with subjective weights computed RELATIVE to the panel rather than as a monotone function of stated validity: (i) a trust plateau over [v_lo, v_hi] (ceiling just above 0.98) gives all mid-band and 98% experts approximately equal weight; (ii) a small negative chance flank at v near 0.5, consensus-gated so chance cues are counted rather than inverted when an above-ceiling outlier destabilizes the scale; (iii) distrust of an above-ceiling outlier is a BOUNDED, SATURATING shrinkage of its weight toward a floor, peaked in the outlier's uncorroborated edge over the next rung (maximal near gap ~0.05: spurious precision; small but nonzero on dense ladders; receding for large edges on coherent panels); (iv) strongly negative distrust reserved for the composition-amplified regime — an outlier floating above a majority-chance panel. Choice is a low-gain softmax plus a uniform lapse. This round is a pure calibration claim: the only two levers whose slopes have been verified against reliably-signed residuals across the loop (the composition-route amplitude c_amp on Exp6, and the chance-flank magnitude kappa_lo on Exp5) are moved to their anti-oscillation midpoints; every structural parameter is held at its accepted value.

**Parameters:**
- w_p: [0.88, 0.92]
- v_lo: [0.52, 0.54]
- v_hi: [0.9805, 0.9825]
- kappa_lo: [0.26, 0.28]
- g0: [0.041, 0.045]
- s_gap: [0.012, 0.014]
- lambda_ext: [0.35, 0.45]
- g_star: [0.046, 0.050]
- s_l: [0.018, 0.024]
- s_r: [0.045, 0.055]
- w_floor: [-0.30, -0.25]
- rho_m: [0.9, 1.2]
- d_w: [0.15, 0.22]
- g_far: [0.10, 0.12]
- s_far: [0.025, 0.035]
- c_half: [0.32, 0.35]
- s_c: [0.05, 0.07]
- c_amp: [2.5, 2.8]
- beta: [0.34, 0.39]
- epsilon: [0.05, 0.07]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Bounded Outlier-Distrust Tallying (BODT), engaged-shoulder edition.
    # Stimulus: array of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are binary expert ratings (0/1).
    # Each feature j casts a sign vote sign(a_j - b_j), weighted by a
    # subjective trust function of the expert's STATED validity v,
    # evaluated RELATIVE to the panel:
    #   (i) TRUST PLATEAU: all v in [v_lo, v_hi] (v_hi just above
    #       0.98, so a 98% expert stays INSIDE the plateau) get
    #       approximately equal positive weight w_p.
    #  (ii) CHANCE FLANK + CONSENSUS GATE: v near 0.5 gets a small
    #       negative weight (-kappa_lo at v=0.5, rising linearly to
    #       the plateau at v_lo). When an above-ceiling claim is
    #       present, the flank is gated: gate = lambda_ext +
    #       (1-lambda_ext)*sig_gap, so chance cues are COUNTED
    #       (weight -> w_p) rather than inverted.
    # (iii) BOUNDED OUTLIER DISTRUST (the core BODT claim): the
    #       discount of the panel-max expert is a SATURATING
    #       SHRINKAGE toward a floor, not an unbounded sign flip:
    #         w_top_mod = w_floor + (w_p - w_floor) * exp(-rho_m * P)
    #       where P is a PEAKED bump in the rung gap (maximal at
    #       gap ~ g_star ~ 0.05 -- spurious precision; a SMALL but
    #       nonzero discount on dense ladders gap ~ 0.01 via the
    #       widened left width s_l; receding for large edges on
    #       coherent panels via the right width s_r, WIDENED so
    #       the shoulder covers the moderate-gap regime gap ~ 0.09).
    #       Because of the exponential saturation, w_top_mod never
    #       leaves [w_floor, w_p].
    #  (iv) COMPOSITION-AMPLIFIED ROUTE: an additive discount
    #         comp = c_amp * sig_gap * far(gap) * gate(c_frac)
    #       where far(gap) is a logistic gate whose center g_far is
    #       LOW enough (~0.10-0.12) to partially engage on
    #       chance-laden moderate-gap panels (gap ~ 0.09 with
    #       c_frac > c_half), while remaining essentially closed at
    #       gap ~ 0.05 and fully open only for far outliers
    #       (gap >= ~0.15) on majority-chance panels.
    # A small bounded extremity term d_w*(v_top - v_hi)/(1 - v_hi)
    # is retained. Choice is a low-gain softmax (inverse temperature
    # beta) over the two mirrored scores, plus a uniform lapse
    # (epsilon).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    n_features = stim.shape[1]
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    w_p = float(parameters["w_p"])
    v_lo = float(parameters["v_lo"])
    v_hi = float(parameters["v_hi"])
    kappa_lo = float(parameters["kappa_lo"])
    g0 = float(parameters["g0"])
    s_gap = float(parameters["s_gap"])
    lambda_ext = float(parameters["lambda_ext"])
    g_star = float(parameters["g_star"])
    s_l = float(parameters["s_l"])
    s_r = float(parameters["s_r"])
    w_floor = float(parameters["w_floor"])
    rho_m = float(parameters["rho_m"])
    d_w = float(parameters["d_w"])
    g_far = float(parameters["g_far"])
    s_far = float(parameters["s_far"])
    c_half = float(parameters["c_half"])
    s_c = float(parameters["s_c"])
    c_amp = float(parameters["c_amp"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    def _sigmoid(x):
        # Numerically stable logistic via the identity
        # sigmoid(x) = 0.5 * (1 + tanh(x / 2)).
        return 0.5 * (1.0 + np.tanh(0.5 * x))

    a, b = stim[0], stim[1]

    # Feature-wise votes: +1 where A wins the feature, -1 where B
    # wins, 0 on ties (a tied cue is uninformative for the pair).
    votes = np.sign(a - b)

    v = np.clip(val, 0.5, 1.0)

    # ---- Panel structure: top validity and its gap to the
    # next-highest DISTINCT validity in the panel. ----
    v_top = float(v.max())
    below = v[v < v_top - 1e-9]
    # If every expert shares the top validity, the panel offers no
    # corroboration ladder; treat the next rung as chance (0.5).
    v_second = float(below.max()) if below.size > 0 else 0.5
    gap = v_top - v_second

    # Outlier = a claim ABOVE the trust ceiling. With v_hi ~ 0.981
    # a 0.98-topped panel stays inside the trusted plateau; only
    # 99%+ claims enter the distrust branch.
    outlier = v_top > v_hi

    if outlier:
        sig_gap = _sigmoid((gap - g0) / s_gap)
    else:
        sig_gap = 0.0

    # ---- (ii): consensus gate for the chance flank. ----
    if outlier:
        gate = lambda_ext + (1.0 - lambda_ext) * float(sig_gap)
    else:
        gate = 0.0

    # ---- (i) + (ii): plateau with a consensus-gated chance flank. ----
    w = np.full(n_features, w_p, dtype=float)
    flank = v < v_lo
    if np.any(flank):
        denom = max(v_lo - 0.5, 1e-6)
        w_flank = -kappa_lo + (w_p + kappa_lo) * np.clip(
            v[flank] - 0.5, 0.0, None
        ) / denom
        # With an above-ceiling claim in the panel, chance cues are
        # counted (weight -> w_p) rather than inverted.
        w[flank] = (1.0 - gate) * w_flank + gate * w_p

    # ---- (iii) + (iv): bounded outlier distrust, with a
    # composition-amplified route that now partially engages on
    # chance-laden MODERATE-gap panels. ----
    if outlier:
        # (a) Peaked bump in the uncorroborated edge: maximal near
        # gap ~ g_star (spurious precision), a SMALL nonzero
        # discount on dense ladders (widened s_l), receding for
        # large edges on coherent panels (widened right shoulder
        # s_r covers the moderate-gap regime gap ~ 0.09).
        width = s_l if gap < g_star else s_r
        P = float(np.exp(-0.5 * ((gap - g_star) / width) ** 2))

        # (b) BOUNDED, SATURATING shrinkage toward the floor:
        # w_top_mod in [w_floor, w_p] for every P; even at peak
        # distrust the outlier is only demoted toward the floor
        # (~ -0.25..-0.30 plateau units), never unboundedly flipped.
        w_top_mod = w_floor + (w_p - w_floor) * float(np.exp(-rho_m * P))

        # (c) Composition-amplified route: a logistic far gate
        # centered at g_far ~ 0.10-0.12, so it PARTIALLY engages on
        # chance-laden moderate-gap panels (gap ~ 0.09, c_frac >
        # c_half) and is fully open only for far outliers
        # (gap >= ~0.15) on majority-chance panels.
        c_frac = float(np.mean(flank))
        comp_gate = _sigmoid((c_frac - c_half) / s_c)
        far = _sigmoid((gap - g_far) / s_far)
        comp = c_amp * float(sig_gap) * float(far) * float(comp_gate)

        # (d) Small bounded extremity term.
        ext = d_w * (v_top - v_hi) / max(1.0 - v_hi, 1e-6)

        w_top = w_top_mod - ext - comp
        # All experts tied at the panel maximum share the distrust
        # (e.g., a pair of isolated 0.99 super-experts).
        w[v >= v_top - 1e-9] = w_top

    # Weighted evidence score. B's score is the mirror of A's
    # because every vote is zero-sum across the two options.
    s_a = float(np.dot(w, votes))
    scores = np.array([s_a, -s_a], dtype=float)

    # Numerically stable low-gain softmax. When all cues tie
    # (s_a == 0) the softmax is exactly uniform.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    # Independent lapse: with probability epsilon output a uniform
    # pick over the two options.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return int(np.random.choice(len(probabilities), p=probabilities))
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # \"Double-99 endorsement avoidance\" metric.\n    #\n    # On every trial, identify whether BOTH top-validity experts (the two\n    # tied 0.99 experts, features 0 and 1 of the fixed panel\n    # [0.99, 0.99, 0.94, 0.75, 0.5, 0.5, 0.5, 0.5, 0.5]) cast the SAME\n    # vote (both favor A, or both favor B). On those trials, record\n    # whether the subject chose the option the 0.99 pair endorsed.\n    # The metric is the proportion of such choices.\n    #\n    # This isolates the single structural disagreement between CREDT and\n    # BODT on this panel: the DEPTH of distrust of a 0.99 claim sitting\n    # gap=0.05 above its next rung on a chance-laden panel.\n    #   CREDT: w_top ~ -1.7 per 99% expert  -> the endorsement is worth\n    #          roughly two trusted cues AGAINST -> p(pick endorsed) ~ 0.16\n    #          (box ~[0.12, 0.19]).\n    #   BODT:  w_top ~ -0.14 per 99% expert -> the endorsement is worth\n    #          roughly nothing -> p(pick endorsed) ~ 0.40 (box ~[0.35, 0.47]).\n    # The two predicted boxes are disjoint by ~0.2 in probability units.\n    # Mirror trials (e.g., trials 1/2, 3/4, 6/7, 8/9, 16/17) put the 99-pair\n    # on opposite sides, so pure A/B side bias cancels.\n    # Works identically on the pooled DataFrame and on a single subject's\n    # slice (each subject has 14 qualifying trial types x 4 reps = 56 rows).\n    validities = [0.99, 0.99, 0.94, 0.75, 0.5, 0.5, 0.5, 0.5, 0.5]\n    val = np.asarray(validities, dtype=float)\n    v_top = float(val.max())\n    top_idx = np.where(val >= v_top - 1e-9)[0]\n\n    picks = []\n    a_col = data['option_a_ratings']\n    b_col = data['option_b_ratings']\n    r_col = data['response']\n\n    for a_cell, b_cell, r in zip(a_col, b_col, r_col):\n        a = np.asarray(a_cell, dtype=float).ravel()\n        b = np.asarray(b_cell, dtype=float).ravel()\n        if a.shape[0] != val.shape[0] or b.shape[0] != val.shape[0]:\n            continue\n        diffs = a[top_idx] - b[top_idx]\n        # Both top experts must vote, and vote together.\n        if np.all(diffs > 0):\n            endorsed = 0  # both 0.99 experts favor A\n        elif np.all(diffs < 0):\n            endorsed = 1  # both 0.99 experts favor B\n        else:\n            # split votes or ties among the top experts -> not a\n            # double-endorsement trial\n            continue\n        resp = int(r)\n        picks.append(1.0 if resp == endorsed else 0.0)\n\n    if len(picks) == 0:\n        return 0.5\n    return float(np.mean(picks))",
  "rationale": "The design's whole point is that CREDT and BODT agree on every weight EXCEPT the weight of the tied 0.99 pair on this gap=0.05, chance-laden panel (CREDT's additive discount gives w_top ~ -1.4 to -2.0 per 99% expert; BODT's bounded floor plus far-gated composition route gives w_top ~ -0.47 to +0.09). So the sharpest possible metric is the direct behavioral readout of that one weight: the proportion of trials on which the subject CHOOSES the option endorsed by both 0.99 experts. I recomputed per-trial predictions from both predict() functions at their box centers: on the 14 double-endorsement trial types (trials 1,2,3,4,5,6,7,8,9,16,17,22,23,24; 4 reps each = 56 trials/subject), CREDT predicts p(pick endorsed) of .24/.24/.12/.12/.10/.15/.15/.29/.29/.11/.11/.10/.10/.20, averaging ~0.16 (and staying in ~[0.12, 0.19] across its ENTIRE parameter box, since even CREDT's least-distrustful corner, w_top ~ -1.42, yields ~0.18). BODT predicts .52/.52/.33/.33/.28/.39/.39/.57/.57/.30/.30/.28/.28/.47, averaging ~0.40 (and staying in ~[0.35, 0.47] across its box; even its most-distrustful corner, w_top ~ -0.47, yields ~0.35). The predicted boxes are disjoint with a ~0.2-0.24 probability gap, while the per-subject sampling noise is small: 56 Bernoulli trials give binomial SD ~ sqrt(0.25/56) ~ 0.067, and parameter-box heterogeneity adds only ~0.02-0.03 SD because the disputed lever moves all 14 trial types in the same direction. At N=25 the Welch t-statistic is on the order of 10-13, far below the alpha=0.01 threshold. The metric is side-balanced (mirror pairs put the 99-pair on opposite sides, so pure A/B response bias cancels), is falsifiable in both directions (human p >= 0.30 kills CREDT; p <= 0.28 kills BODT), works identically per subject and pooled, and avoids any hashing of list-valued columns."
}
```

## Usage

```json
{
  "input_tokens": 11080,
  "output_tokens": 23735,
  "reasoning_tokens": 22319,
  "provider": "Sail Research"
}
```
