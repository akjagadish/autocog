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
**Validities (n_features=10):** [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5]

**Trial pairs (n=24):**
  trial 1: A=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0, 0, 0, 1, 0]
  trial 2: A=[0, 1, 0, 1, 0, 1, 1, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1, 0, 0, 0, 1]
  trial 3: A=[1, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  trial 4: A=[0, 1, 0, 1, 0, 1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0, 1, 1, 1, 0, 1]
  trial 5: A=[1, 0, 0, 0, 1, 0, 1, 0, 1, 0]  B=[1, 1, 0, 0, 1, 0, 0, 0, 1, 0]
  trial 6: A=[0, 0, 1, 0, 0, 1, 1, 1, 0, 1]  B=[0, 1, 1, 0, 0, 1, 0, 0, 0, 1]
  trial 7: A=[1, 0, 0, 1, 0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
  trial 8: A=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0, 1, 1, 1, 0]
  trial 9: A=[1, 0, 1, 0, 1, 0, 0, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0, 0, 1, 0, 1]
  trial 10: A=[0, 0, 1, 0, 1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 0, 1, 0, 1]
  trial 11: A=[1, 0, 0, 1, 0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0, 1, 0, 1, 0]
  trial 12: A=[1, 1, 0, 0, 1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
  trial 13: A=[1, 1, 1, 0, 0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1, 0, 0, 1, 0, 0]
  trial 14: A=[1, 0, 0, 1, 1, 0, 0, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 1, 0, 0]
  trial 15: A=[0, 1, 1, 0, 0, 0, 1, 0, 0, 1]  B=[0, 0, 1, 0, 0, 1, 1, 0, 0, 1]
  trial 16: A=[0, 0, 1, 0, 0, 1, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0, 0, 1, 0, 0, 1]
  trial 17: A=[1, 0, 0, 1, 0, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 0, 0, 0, 1, 0]
  trial 18: A=[0, 1, 0, 0, 1, 1, 1, 1, 0, 1]  B=[0, 1, 0, 0, 1, 0, 0, 0, 0, 1]
  trial 19: A=[0, 1, 0, 0, 1, 0, 1, 1, 0, 1]  B=[0, 1, 0, 0, 1, 1, 0, 0, 0, 1]
  trial 20: A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  trial 21: A=[1, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 22: A=[1, 1, 1, 1, 1, 1, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0, 1, 0, 1, 0]
  trial 23: A=[1, 0, 0, 0, 0, 0, 1, 0, 1, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 1, 0]
  trial 24: A=[0, 1, 0, 0, 1, 0, 1, 0, 0, 1]  B=[0, 1, 0, 0, 1, 0, 0, 1, 0, 1]

**Rationale:** CORE LOGIC. Validities are fixed for the whole experiment, so both theories reduce to a FIXED weight vector over the 10 experts, and the design's job is to make the two weight vectors maximally anti-correlated. The panel [0.99, 0.90, 0.80, 0.70, 0.60, 0.51, 0.50, 0.50, 0.50, 0.50] is engineered to push CREDT into its most distinctive regime, never instantiated in any prior experiment: (a) the 0.99 claim is an above-ceiling outlier with gap = 0.09 over the 0.90 second rung (sig_gap ~ 0.96-0.98, so the consensus gate is ~open); (b) the panel is CHANCE-LADEN: 5 of 10 experts sit below v_lo, so c_frac = 0.5 > c_half for the entire box, firing the composition amplifier (C ~ 1.0-1.3). Result: CREDT's weight on the 0.99 expert is w_top in [-1.93, -0.96] across its ENTIRE box (composition-amplified collapse), while the gate simultaneously pushes the chance flank to POSITIVE near-plateau weight: w(0.50) in [0.86, 0.91], w(0.51) ~ [0.87, 0.91], and the plateau experts (0.60-0.90) all get exactly w_p in [0.88, 0.92]. So on this panel CREDT is approximately 'count everybody's endorsement equally, EXCEPT the 99% expert, whose endorsement counts strongly AGAINST'. FDEI is the exact structural opposite: monotone graded weights with w(0.50) = 0 EXACTLY for every parameter, w(0.51) in [0, 0.73], and w(0.99) the LARGEST weight, in [1.17, 1.36]. Critically, FDEI is exactly invariant to the validities of tied cues and to panel composition, so the entire panel-structure manipulation is invisible to it except through which cues vote -- the structural source of every dissociation below. 24 unique pairs x K = 96//24 = 4 = 96 trials.

FAMILY Z (chance-only ladder; FDEI parameter-free knife). Z1/Z2/Z3: only 50% experts discriminate (A wins 1/2/3 of them); Z2m mirrors. FDEI must predict EXACTLY p = 0.50 on all four for every (delta, tau, beta, epsilon) -- all attending weights are identically zero. CREDT predicts a steep positive ladder: Z1 in [0.66, 0.68], Z2 in [0.79, 0.81], Z3 in [0.87, 0.90] across its whole box. Any reliable deviation from 0.50 on any Z trial falsifies FDEI outright with zero free parameters; a flat 0.50 falsifies CREDT's consensus-gating claim (which here flips chance cues from anti-diagnostic to near-fully trusted -- the OPPOSITE sign from the chance flank tested in [8], whose panel had c_frac = 0.27, below the amplifier threshold).

FAMILY X (90% expert vs k chance cues; FDEI flatness knife + sign flip). X1/X2/X3: B wins the 0.90 expert, A wins 1/2/3 chance cues; X3m mirrors. FDEI must predict X1 = X2 = X3 EXACTLY (chance weight is identically zero), with common value in [0.28, 0.37]. CREDT predicts a monotone ladder crossing 0.5: X1 in [0.49, 0.51], X2 in [0.65, 0.68], X3 in [0.78, 0.82]. X2 and X3 are box-disjoint SIGN FLIPS: 'does a 90% expert beat two coin-flippers?' CREDT: no (the coin-flippers win); FDEI: yes, strongly. Double bind: any X-ladder slope falsifies FDEI parameter-free; any X2 < 0.5 falsifies CREDT.

FAMILY S/V (0.99 inversion). S1 (A wins only the 0.99 expert): CREDT [0.19, 0.32] vs FDEI [0.64, 0.73] -- sign-disjoint. S1m mirrors. S2 (0.99 for A vs 0.90 for B): CREDT [0.11, 0.20] vs FDEI [0.51, 0.54]. V1 (0.99 for A vs one 50% for B): CREDT [0.11, 0.20] vs FDEI [0.64, 0.73] -- the single sharpest sign flip: FDEI says the 99% expert crushes a coin-flipper; CREDT says both point the same way, toward B.

FAMILY K (flagship: four trusted vs four coin-flip). K1: A wins the 0.90/0.80/0.70/0.60 experts, B wins all four 50% experts. CREDT: s = 4*w_p - 4*w(0.5) ~ 0, p in [0.48, 0.56] -- a COIN FLIP, because gated chance cues carry the same weight as plateau experts. FDEI: p in [0.88, 0.95]. A 35-45 point dissociation on a unanimous-expertise trial. K1m mirrors. p(K1) >= 0.60 falsifies CREDT outright; p(K1) <= 0.70 falsifies FDEI outright.

FAMILY P (plateau flatness; CREDT parameter-free knife). P2/P2m: A wins {0.90, 0.80} vs B wins {0.70, 0.60} (and mirror). All four discriminators sit on CREDT's plateau with tied counts, so CREDT must predict EXACTLY 0.50 for every parameter. FDEI's graded weights predict a tilt: P2 in [0.53, 0.58], P2m in [0.42, 0.47]. Any systematic tilt falsifies CREDT's plateau with no parameters; exact 0.50 falsifies FDEI's delta > 0. P4/P4m (0.90 vs 0.51 head-to-head): CREDT [0.49, 0.51] (gated 0.51 ~ plateau weight) vs FDEI [0.55, 0.69] / [0.31, 0.45] -- disjoint: CREDT says a 51% expert is exchangeable with a 90% expert on this panel; FDEI says the 90% expert clearly wins.

FAMILY W (honest knives against CREDT). W1 (0.51 for A vs 0.50 for B): CREDT is capped at 0.513 across its entire box; FDEI allows up to 0.63. If subjects distinguish 51% from 50% at p > 0.52, CREDT's gated-equalization claim is falsified. W2 (0.51 for A vs two 0.50 for B): CREDT [0.32, 0.35] vs FDEI [0.50, 0.63] -- disjoint; W2m mirrors.

ANCHORS/CONTROLS. G3/G3m (five sub-ceity experts vs one coin-flipper) are agreement anchors (CREDT [0.92, 0.95], FDEI [0.90, 0.97]) pinning lapse/temperature. N1 (one 50% expert won by each side): BOTH theories must predict exactly 0.50 for every parameter -- a side/display-bias assay; note N1 also defeats a 'count the 1s' heuristic (B has more 1s), as do V1 and W2 where raw-1 counts favor A while both theories' evidence favors B. Tie patterns mix 0/0 and 1/1 throughout so count-based heuristics are unviable; no all-tie trials; every trial requires scanning all 10 experts; sides are balanced (7 A-tilted / 6 B-tilted / near-neutral remainder under CREDT centers; 9/10 under FDEI).

DIFFERENTIATION FROM PRIOR EXPERIMENTS. No prior design instantiated the composition-amplifier regime (c_frac = 0.5 > c_half; [6] had 2/8, [8] 3/11, [9] 0/10 chance cues), and no prior design tested the CONJUNCTION it produces: 0.99 anti-diagnostic AND chance cues at near-full POSITIVE weight AND a flat plateau, on one fixed panel. [4] tested chance-cue weighting vs annihilation but with a monotone advocated theory and no outlier inversion; [6]-[9] tested 0.99 cliffs with chance cues negative or absent. The K1 cell (unanimous trusted panel vs equal-count coin-flip panel -> CREDT coin flip, FDEI 0.90) and the X-family sign crossover are new quantitative signatures. PREDICTED PATTERN IF CREDT IS RIGHT (centers): Z .67/.80/.87, X .50/.66/.79, S1 .23, S1m .77, S2 .14, V1 .14, P2 = P2m = 0.50 exact, P4 ~ .50, W1 ~ .51, W2 .32, K1 .51, G3 .92, N1 0.50. IF FDEI IS RIGHT: Z = 0.50 exactly (all four), X flat at .33, S1 .69, S1m .31, S2 .52, V1 .69, P2 .55, P4 .58, W1 .58, W2 .58, K1 .90, G3 .93, N1 0.50. The boxes are disjoint on 14 of 24 cells and every family contains a parameter-free exact-equality prediction for one theory that the other breaks in a signed, graded, pre-registered direction.

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
**Description:** Flattened-Diagnosticity Evidence Integration (the extreme-compression corner of the Bayesian cue-counting family): people integrate ALL binary expert ratings into one evidence score, where each discriminating cue casts a sign vote weighted by w_j = max(log(v_j/(1-v_j)) - tau, 0)^delta — chance cues (v=0.5) are annihilated exactly, but among genuinely diagnostic cues the subjective weight is a HEAVILY flattened (delta ~ 0.15) version of the normative log-odds, and choice is a low-gain softmax (2*beta ~ 0.6) plus a small uniform lapse. The new theoretical claim is that the compression exponent delta is itself the structural margin-compression device: shrinking delta equalizes weights, which GROWS count-dominated conflict margins (where cues split across options) while SHRINKING alignment-dominated cumulative margins (where several strong cues pile onto one side). This differential action — not any global gain knob, and not trial-level normalization — is what lets a single small beta simultaneously produce moderate TTB-leaning conflict-trial behavior (Exp 1), a slightly positive tally-vs-top-cue regression slope (Exp 2), graded ~0.62 tally-following (Exp 3), and near-chance |d|=1 agreement driven by the cancellation between super-expert matches and super-expert reversals (Exp 4).

**Parameters:**
- delta: [0.1, 0.2]
- tau: [0.0, 0.04]
- beta: [0.27, 0.37]
- epsilon: [0.0, 0.05]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Diagnosticity-Weighted Evidence Integration (Bayesian cue-counting).
    # Stimulus: array of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are binary expert ratings (0/1).
    # Each feature j casts a sign vote sign(a_j - b_j); the vote is
    # weighted by the cue's log-likelihood-ratio diagnosticity
    # llr_j = log(v_j / (1 - v_j)), rectified by an attention floor
    # tau and compressed by an exponent delta:
    #   w_j = (max(llr_j - tau, 0))^delta
    # A v = 0.5 expert has llr = 0 -> exactly zero evidence.
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

    delta = float(parameters["delta"])
    tau = float(parameters["tau"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    a, b = stim[0], stim[1]

    # Feature-wise votes: +1 where A wins the feature, -1 where B
    # wins, 0 on ties (a tied cue is uninformative for the pair).
    votes = np.sign(a - b)

    # Normative diagnosticity: log-likelihood-ratio of each expert.
    # Clip away from v = 1.0 to keep the log finite; v = 0.5 maps to 0.
    v = np.clip(val, 0.5, 1.0 - 1e-9)
    llr = np.log(v / (1.0 - v))

    # Limited-attention floor: diagnosticity below tau is discarded;
    # the remainder is compressed by the exponent delta. delta = 1 is
    # the purely normative Bayesian weight; delta < 1 flattens the
    # diagnosticity gradient toward tallying over informative cues
    # (while v = 0.5 cues stay annihilated at any delta, since
    # llr(0.5) = 0 <= tau); delta > 1 sharpens it toward
    # lexicographic/TTB-like use of the best experts.
    w = np.power(np.maximum(llr - tau, 0.0), delta)

    # Weighted evidence score. B's score is the mirror of A's because
    # every vote is zero-sum across the two options.
    s_a = float(np.dot(w, votes))
    scores = np.array([s_a, -s_a])

    # Numerically stable low-temperature softmax. When all attended
    # cues tie (s_a == 0) the softmax is exactly uniform, which is the
    # correct behavior for undiscriminating evidence.
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
  "metric_source": "import numpy as np\nimport pandas as pd\n\n\ndef metric(data: pd.DataFrame) -> float:\n    \"\"\"\n    Theory-discriminant index for the 10-expert panel\n    validities = [0.99, 0.90, 0.80, 0.70, 0.60, 0.50, 0.50, 0.50, 0.50, 0.50].\n\n    Both theories reduce to a FIXED weight vector over the 10 experts on\n    this panel. At each theory's parameter-box midpoint we compute its\n    predicted p(choose A) for the observed trial, take the difference\n    d_t = p_CREDT(t) - p_FDEI(t), and average d_t * y_t over all trials,\n    where y_t = +1 if the subject chose A and -1 if B.\n\n    E[metric] = (1/T) * sum_t d_t * (2*p_true(t) - 1).\n    If CREDT generated the data this is strongly POSITIVE (~ +0.045);\n    if FDEI generated the data it is NEGATIVE (~ -0.022). The contrast\n    is driven by the cells where the boxes are disjoint: the Z-family\n    (chance-cue ladders: CREDT 0.59-0.74 vs FDEI exactly 0.50), the\n    0.99-expert trials (CREDT ~0.28-0.36 vs FDEI ~0.55-0.60), the\n    X-family sign flips, and K1 (four trusted experts vs four\n    coin-flippers: CREDT ~0.50 vs FDEI ~0.78).\n    \"\"\"\n    validities = np.array([0.99, 0.90, 0.80, 0.70, 0.60, 0.50, 0.50, 0.50, 0.50, 0.50])\n\n    # ---------- CREDT weights at parameter-box midpoints ----------\n    w_p, v_lo, v_hi, kappa_lo = 0.90, 0.53, 0.9815, 0.20\n    g0, s_gap = 0.043, 0.013\n    rho, d_w, lambda_ext = 1.36, 0.40, 0.40\n    g_star, s_l, s_r = 0.048, 0.012, 0.033\n    c_half, s_c, c_amp = 0.36, 0.06, 1.29\n    beta_c, eps_c = 0.42, 0.05\n\n    v = np.clip(validities, 0.5, 1.0)\n    v_top = float(v.max())\n    below = v[v < v_top - 1e-9]\n    v_second = float(below.max()) if below.size > 0 else 0.5\n    gap = v_top - v_second\n    outlier = v_top > v_hi\n    if outlier:\n        sig_gap = 0.5 * (1.0 + np.tanh(0.5 * ((gap - g0) / s_gap)))\n    else:\n        sig_gap = 0.0\n    gate = (lambda_ext + (1.0 - lambda_ext) * sig_gap) if outlier else 0.0\n\n    w_c = np.full(10, w_p, dtype=float)\n    flank = v < v_lo\n    if np.any(flank):\n        w_fl = -kappa_lo + (w_p + kappa_lo) * np.clip(v[flank] - 0.5, 0.0, None) / (v_lo - 0.5)\n        w_c[flank] = (1.0 - gate) * w_fl + gate * w_p\n    if outlier:\n        c_frac = float(flank.mean())\n        C = c_amp * 0.5 * (1.0 + np.tanh(0.5 * (c_frac - c_half) / s_c))\n        width = s_l if gap < g_star else s_r\n        P = float(np.exp(-0.5 * ((gap - g_star) / width) ** 2))\n        w_top = w_p - d_w * (v_top - v_hi) / (1.0 - v_hi) - rho * (P + C * sig_gap)\n        w_c[v >= v_top - 1e-9] = w_top\n\n    # ---------- FDEI weights at parameter-box midpoints ----------\n    delta, tau = 0.15, 0.02\n    beta_f, eps_f = 0.32, 0.025\n    vv = np.clip(validities, 0.5, 1.0 - 1e-9)\n    llr = np.log(vv / (1.0 - vv))\n    w_f = np.power(np.maximum(llr - tau, 0.0), delta)\n\n    def _p_choose_a(s, beta, eps):\n        # low-gain softmax over [s, -s] plus uniform lapse\n        z = beta * s\n        p_core = 1.0 / (1.0 + np.exp(-z))\n        return (1.0 - eps) * p_core + eps * 0.5\n\n    total = 0.0\n    n = 0\n    for a_cell, b_cell, resp in zip(data['option_a_ratings'],\n                                    data['option_b_ratings'],\n                                    data['response']):\n        a = np.asarray(a_cell, dtype=float)\n        b = np.asarray(b_cell, dtype=float)\n        votes = np.sign(a - b)\n        s_c = float(np.dot(w_c, votes))\n        s_f = float(np.dot(w_f, votes))\n        p_c = _p_choose_a(s_c, beta_c, eps_c)\n        p_f = _p_choose_a(s_f, beta_f, eps_f)\n        d_t = p_c - p_f\n        y = 1.0 if int(resp) == 0 else -1.0\n        total += d_t * y\n        n += 1\n\n    if n == 0:\n        return 0.0\n    return float(total / n)",
  "rationale": "Both theories collapse to a fixed 10-dimensional weight vector on this panel, so the optimal discriminating statistic is the linear contrast whose per-trial coefficient is the difference of the two theories' predicted p(A). Metric = mean_t [p_CREDT(t) - p_FDEI(t)] * y_t, with y_t = +1 for choosing A, -1 for B; the predictions are computed inside the metric from each theory's own source code at its parameter-box midpoints (CREDT weights ~ [-1.45, .90, .90, .90, .90, .883x5]; FDEI weights ~ [1.256, 1.124, 1.048, 0.972, 0.867, 0...0]). Expected value: if CREDT generated the data, E[metric] = (1/24) sum_t d_t (2 p_CREDT - 1) ~ +0.045 (positive contributions from the Z-family chance-cue ladders where CREDT says 0.59-0.74 and FDEI says exactly 0.50, from the 0.99-expert trials where CREDT says 0.28-0.36 and FDEI says 0.55-0.60, and from the X-family sign flips); if FDEI generated the data, E[metric] ~ -0.022 (negative contributions dominated by K1/K1m, where FDEI's 0.78/0.22 collides with CREDT's ~0.50, and by the 0.99 and X trials). The between-theory gap (~0.067) is large relative to per-subject sampling noise: each subject contributes 96 trials (4 reps of each of the 24 cells), giving a per-subject standard deviation of roughly 0.018-0.022 (binomial noise on the contrast, plus modest parameter-box heterogeneity whose sign structure is box-stable: CREDT's Z predictions stay in [0.66, 0.75], its 0.99 trials stay below 0.5, and FDEI's chance weight is identically zero for every parameter, so the contrast signs never flip). Welch's t at N=25 on (mean ~ +0.045, var ~ 4e-4) vs (mean ~ -0.022, var ~ 4e-4) gives |t| ~ 10+, far below alpha = 0.01. The metric works identically on the pooled DataFrame and on any single subject's slice (it is a per-trial average), satisfying the between-subject variance requirement. Trials where the theories agree (G3 anchors, N1, P-family) receive near-zero coefficients automatically, so they add no variance; the contrast concentrates its weight on the 14 box-disjoint cells."
}
```

## Usage

```json
{
  "input_tokens": 9408,
  "output_tokens": 23357,
  "reasoning_tokens": 21289,
  "provider": "Sail Research"
}
```
