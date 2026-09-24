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
**Validities (n_features=8):** [0.99, 0.94, 0.85, 0.75, 0.65, 0.5, 0.5, 0.5]

**Trial pairs (n=24):**
  trial 1: A=[1, 1, 0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 0]
  trial 2: A=[0, 1, 0, 1, 0, 1, 0, 0]  B=[1, 1, 0, 1, 0, 1, 0, 0]
  trial 3: A=[1, 0, 1, 1, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0, 0, 1, 0]
  trial 4: A=[0, 1, 1, 1, 0, 0, 1, 0]  B=[1, 0, 1, 1, 0, 0, 1, 0]
  trial 5: A=[1, 0, 0, 1, 1, 0, 1, 1]  B=[0, 1, 0, 1, 1, 0, 0, 1]
  trial 6: A=[1, 0, 1, 0, 1, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1, 1, 0]
  trial 7: A=[1, 0, 0, 1, 0, 1, 1, 1]  B=[0, 1, 0, 1, 0, 1, 0, 0]
  trial 8: A=[0, 1, 0, 1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 0, 1, 1, 1]
  trial 9: A=[1, 1, 1, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 1, 0, 0, 1]
  trial 10: A=[0, 1, 1, 0, 1, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0, 0, 1]
  trial 11: A=[1, 0, 1, 1, 0, 1, 1, 0]  B=[1, 0, 1, 1, 0, 1, 0, 0]
  trial 12: A=[0, 1, 0, 1, 1, 0, 1, 1]  B=[0, 1, 0, 1, 1, 0, 0, 0]
  trial 13: A=[0, 1, 0, 1, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0, 1, 1]
  trial 14: A=[1, 0, 1, 0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 1, 1, 0, 1]
  trial 15: A=[1, 1, 0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 0, 1, 1, 0, 0]
  trial 16: A=[1, 1, 0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 1, 0, 1, 1]
  trial 17: A=[0, 1, 1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 1, 0, 0]
  trial 18: A=[1, 1, 0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 1, 0, 1]
  trial 19: A=[1, 0, 0, 1, 1, 1, 0, 1]  B=[1, 1, 0, 1, 1, 0, 0, 1]
  trial 20: A=[1, 1, 1, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 1, 1, 1, 0]
  trial 21: A=[1, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 1, 0]
  trial 22: A=[1, 0, 1, 1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1, 1, 0]
  trial 23: A=[0, 1, 0, 0, 1, 1, 1, 0]  B=[1, 0, 1, 1, 0, 0, 1, 0]
  trial 24: A=[1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1]

**Rationale:** CORE LOGIC. Both theories share the identical architecture (sign votes x validity weights x low-gain softmax x lapse) and differ ONLY in the weight function, but the weight functions are near-orthogonal on this panel. The panel [0.99, 0.94, 0.85, 0.75, 0.65, 0.55, 0.50, 0.50] is engineered to place CREDT in its PEAKED EDGE-DISTRUST regime, never instantiated in any prior experiment: the 0.99 claim sits exactly 0.05 above the 0.94 second rung -- the gap that CREDT's non-monotone edge-distrust bump P treats as MAXIMAL spurious precision (g_star ~ 0.048) -- while the panel is coherent (c_frac = 2/8 = 0.25 < c_half, so the composition amplifier contributes only C ~ 0.09-0.26 rather than firing). Resulting CREDT weights across its ENTIRE box: w(0.99) in [-1.04, -0.59] (strongly ANTI-diagnostic), w(0.94)...w(0.55) all exactly w_p in [0.88, 0.92] (flat plateau), and the two 0.50 experts GATED to positive weight (gate = lambda_ext + (1-lambda_ext)*sig_gap ~ 0.73-0.82 open, since the outlier's sig_gap ~ 0.59-0.68), giving w(0.50) in [0.62, 0.69]. FDEI weights across its entire box: w(0.99) in [1.16, 1.36] (the LARGEST weight), w(0.94) in [1.11, 1.23], w(0.85) in [1.06, 1.12], w(0.75) in [1.01, 1.02], w(0.65) in [0.91, 0.95], w(0.55) in [0.69, 0.85], w(0.50) = 0 EXACTLY for every (delta, tau, beta, epsilon). The two weight vectors are almost perfectly anti-correlated: FDEI's best cue is CREDT's worst cue; FDEI's invisible cues are worth ~70% of a plateau cue to CREDT; CREDT's plateau is flat where FDEI is (weakly) graded. 24 unique pairs x K = 96//24 = 4 = 96 trials.

FAMILY S/W (flagship: spurious-precision inversion + parameter-free chance-cue flatness ladder). S1 (A wins only the 0.99 expert): FDEI box p(A) in [0.64, 0.73] (center 0.69); CREDT box [0.30, 0.39] (center 0.35) -- SIGN-DISJOINT with a 25-point gap: FDEI says follow the 99% expert, CREDT says a 0.99-atop-0.94 claim is spurious precision and its endorsement counts AGAINST. S1m mirrors. S2 (A wins 0.99, B wins 0.94): FDEI's delta-flattening locks w99-w94 in [0.05, 0.14], so its ENTIRE box spans only p(A) in [0.509, 0.524] -- a near-parameter-free coin flip; CREDT box [0.17, 0.25] (center 0.21). A 28-point dissociation on a cell where FDEI is pinned within 1.5 points of 0.50. THE LADDER: S4 (B also wins one 0.50) / S2 / S3 (A also wins one 0.50) / W2 (A also wins both 0.50s) holds the 0.99-vs-0.94 diagnostic background fixed while the net chance-cue count for A runs -1, 0, +1, +2. FDEI must predict S4 = S2 = S3 = W2 EXACTLY for every parameter (chance weight is identically zero) -- a parameter-free flatness knife. CREDT predicts a steep monotone ladder: 0.141 / 0.209 / 0.305 / 0.425 at center (boxes [0.11,0.17] / [0.17,0.25] / [0.26,0.35] / [0.36,0.48]) -- a ~30-point climb that directly measures the gated chance-cue weight (~0.65) against the outlier distrust (~-0.79) in one psychometric. W2m mirrors. Any reliable slope across the ladder falsifies FDEI with zero free parameters; flatness at 0.51 falsifies CREDT's consensus-gating claim.

FAMILY V (outlier vs chance, and the S1=V1 flatness knife). V1 (A wins 0.99, B wins one 0.50): FDEI [0.64, 0.73] (identical to S1 -- FDEI must predict V1 = S1 EXACTLY, since chance cues are invisible); CREDT [0.20, 0.28] (center 0.24: BOTH the 99% anti-vote and the gated coin-flipper point toward B). V1m mirrors. CREDT predicts V1 < S1 by ~10 points (the chance cue helps B); FDEI demands exact equality -- a second parameter-free knife, and V1 is a sign-disjoint cell in its own right.

FAMILY Z (chance-only ladder; FDEI parameter-free knife). Z1/Z2 (A wins 1/2 of the 0.50 experts, nothing else discriminates): FDEI must predict EXACTLY 0.50 on both (all weights zero -> uniform softmax, lapse uniform) for every parameter. CREDT predicts a steep positive ladder: Z1 in [0.62, 0.74] (center 0.63), Z2 in [0.72, 0.76] (center 0.74). Z2m mirrors. Any reliable deviation from 0.50 falsifies FDEI outright; a flat 0.50 falsifies CREDT's gated-trust claim. Note the sign here is POSITIVE (chance cues trusted), the OPPOSITE of the ungated negative flank -- on this panel the gate is partially open, which is CREDT's distinctive conjunction: distrust the 0.99 outlier AND simultaneously count the coin-flippers.

FAMILY X (plateau expert vs gated chance cues; mutual sign flip and mutual knife). X1 (A wins one 0.50, B wins the 0.55 expert): FDEI [0.35, 0.41] (center 0.38); CREDT [0.435, 0.465] (center 0.45). X2 (A wins both 0.50s, B wins 0.55): FDEI must predict X2 = X1 EXACTLY (parameter-free flatness; center 0.38); CREDT predicts a SIGN FLIP to [0.56, 0.60] (center 0.58) -- two coin-flippers beat a 55% expert under gating. X2m mirrors. X3 (A wins both 0.50s, B wins the 0.75 expert): CREDT must predict X3 = X2 EXACTLY (plateau weights identical to machine precision) -- its parameter-free flatness knife; FDEI predicts X3 < X2 (0.35 vs 0.38, its graded 0.75-vs-0.55 weights). The X family is a double bind: the X1=X2 equality falsifies FDEI if broken; the X2=X3 equality falsifies CREDT if broken; and the X2/X3 sign (FDEI [0.33,0.41] vs CREDT [0.56,0.60]) is box-disjoint in the opposite direction from X1.

FAMILY Q/P (FDEI's within-plateau gradient vs CREDT's flat plateau; CREDT parameter-free knife). Q1 (A wins 0.94, B wins 0.55; 1-vs-1): CREDT must predict EXACTLY 0.50 (both on the plateau, tied counts) for every parameter; FDEI predicts [0.535, 0.597] (center 0.56) because its compressed-but-graded weights give the 0.94 side a net +0.27 to +0.53 evidence margin. Q1m mirrors. P1 (A wins {0.94, 0.85}, B wins {0.65, 0.55}; 2-vs-2): CREDT exactly 0.50; FDEI [0.549, 0.635] (center 0.58) -- its strongest plateau-tilt cell. P1m mirrors. Any systematic tilt on Q1/P1 falsifies CREDT's plateau with zero free parameters; exact 0.50 across the family falsifies FDEI's delta > 0 gradient. The Q/P family is deliberately modest in size (5-8 points) -- recovering BOTH this small graded tilt AND the huge S/Z effects at their predicted relative sizes is the quantitative calibration test.

FAMILY H (masked inversion inside a balanced 3-vs-3 coalition). H3 (A wins {0.99, 0.85, 0.75}, B wins {0.94, 0.65, 0.55}; raw cue count tied 3-3, raw 1-count tied): FDEI margin is remarkably delta-stable (0.48-0.51 across its whole box), locking p(A) in [0.575, 0.594] (center 0.58); CREDT margin = w_top - w_p, giving [0.17, 0.25] (center 0.21) -- a 35-point SIGN FLIP on a count-balanced trial. H3m mirrors. This cell defeats any count-based heuristic (counts are exactly tied) and isolates pure weighting.

ANCHOR. G1 (A wins all five plateau experts, B wins both 0.50s): both theories choose A at high confidence (FDEI [0.92, 0.97], center 0.95; CREDT [0.90, 0.93], center 0.91), pinning lapse/temperature; G1m-equivalent structure is carried by the mirrors.

DIFFERENTIATION FROM PRIOR EXPERIMENTS. No prior design instantiated the g_star PEAK regime of CREDT's edge-distrust bump: [10] used a 0.99-atop-0.90 panel (gap 0.09, large-edge branch) that was CHANCE-LADEN (c_frac = 0.5, amplifier firing, chance cues at 0.86-0.91); here gap = 0.05 with a coherent panel (c_frac = 0.25, amplifier sub-threshold, chance cues at 0.62-0.69, w_top shallower at -0.6 to -1.0). The S4->S2->S3->W2 ladder measuring gated chance weight against peaked outlier distrust in one psychometric is new, as is the FDEI-locked near-tie on 0.99-vs-0.94 (its entire box within 1.5 points of 0.50) and the S1=V1 and X1=X2 flatness knives. [6]-[9] tested FDEI against other competitors (SITT/BSST), not CREDT. PREDICTED PATTERN IF FDEI IS RIGHT (center delta=0.15, tau=0.01, beta=0.32, eps=0.025): S1 .69, S2=S3=S4=W2 .51 exactly, V1 .69 (=S1), Z1=Z2 .50 exactly, X1=X2 .38, X3 .35, Q1 .56, P1 .58, H3 .58, G1 .95. IF CREDT IS RIGHT (center w_p=.90, rho=1.36, beta=.42, eps=.05): S1 .35, S4 .14 < S2 .21 < S3 .31 < W2 .43, V1 .24, Z1 .63, Z2 .74, X1 .45, X2=X3 .58 exactly, Q1=P1 .50 exactly, H3 .21, G1 .91. The boxes are disjoint on 14 of 24 cells, sign-disjoint on 6 (S1, S2, V1, X2, X3, H3), and every family contains a parameter-free exact-equality or exact-0.50 prediction for one theory that the other breaks in a signed, graded, pre-registered direction. Sides are balanced (7 A-tilted / 7 B-tilted / 10 near-neutral under FDEI centers; 8/11/5 under CREDT centers; every family contains an A/B mirror); no all-tie trials; every trial requires scanning all 8 experts; tie patterns mix 0/0 and 1/1 so 'count the 1s' is unviable (e.g., H3 has exactly tied 1-counts while the theories disagree by 35 points, and X2/X3 have MORE 1s for A while FDEI favors B).

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    \"\"\"\n    Super-expert deference rate.\n\n    Among all trials on which the highest-validity expert (feature 0, the\n    99% expert) discriminates between option A and option B, return the\n    fraction of choices that FOLLOW that expert's recommendation (i.e.\n    chose the option the 99% expert rated 1).\n\n    FDEI (flattened-diagnosticity integration) gives the 0.99 expert its\n    largest positive weight, so it predicts deference well above chance\n    (~0.57 overall, ~0.69 on isolated super-expert trials, but BELOW 0.5\n    when a 0.94+0.65+0.50 coalition opposes it).\n\n    CREDT, on this panel (0.99 sitting 0.05 above the 0.94 rung with a\n    partially chance-laden scale), assigns the panel-max expert a large\n    NEGATIVE weight, so it predicts systematic ANTI-deference (~0.21).\n    \"\"\"\n    df = data.reset_index(drop=True)\n\n    def _first_rating(cell):\n        # Cells may be list / tuple / np.ndarray of 0/1 ints, or a string\n        # serialization like \"[1, 0, 1, ...]\". Robustly extract element 0.\n        if isinstance(cell, str):\n            s = cell.strip().strip(\"[]\")\n            parts = s.replace(\",\", \" \").split()\n            return int(float(parts[0]))\n        seq = list(cell)\n        return int(seq[0])\n\n    a_col = df[\"option_a_ratings\"]\n    b_col = df[\"option_b_ratings\"]\n    r_col = df[\"response\"]\n\n    n = 0\n    follow = 0\n    for i in range(len(df)):\n        try:\n            a0 = _first_rating(a_col.iloc[i])\n            b0 = _first_rating(b_col.iloc[i])\n        except Exception:\n            continue\n        # Skip trials where the 99% expert does not discriminate.\n        if a0 not in (0, 1) or b0 not in (0, 1) or a0 == b0:\n            continue\n        # Option endorsed by the 99% expert: 0 = A, 1 = B.\n        winner = 0 if a0 == 1 else 1\n        resp = int(r_col.iloc[i])\n        n += 1\n        if resp == winner:\n            follow += 1\n\n    if n == 0:\n        return 0.5\n    return float(follow) / float(n)",
  "rationale": "METRIC: 'Super-expert deference rate' \u2014 on every trial where feature 0 (the 0.99-validity expert) discriminates between A and B, score 1 if the subject chose the option that expert endorsed, 0 otherwise; the metric is the mean. In this design 12 of the 24 unique trial types have the 0.99 expert discriminating (trials 1-10, 22, 23), giving 48 scored trials per subject, so the metric is well-defined both on the pooled DataFrame and on every individual subject's slice.\n\nWHY IT MAXIMALLY SEPARATES THE TWO THEORIES. The single deepest structural disagreement on this panel is the sign of the weight on the 0.99 expert. FDEI (any delta in [0.1,0.2], tau in [0,0.04]) gives the 0.99 expert its LARGEST weight (w ~ 1.16-1.36 vs 1.10-1.22 for the 0.94 expert), so it predicts deference: p(follow) ~ 0.67-0.70 on trials 1/2/9/10 (super-expert alone or with a chance cue), ~0.51 on trials 3-8 (0.99 vs 0.94, chance-invariant), and honestly BELOW 0.5 on trials 22/23 (0.99 vs a 0.94+0.65+0.50 coalition), averaging ~0.57 across the whole box. CREDT, by contrast, reads a 0.99 claim sitting 0.05 above the 0.94 rung on a panel that is 3/8 chance-laden as spurious precision amplified by composition: across its ENTIRE parameter box the panel-max weight is strongly negative (w_top ~ -1.0 to -1.6, since the peaked bump P ~ 1 and the composition term C ~ 0.6-1.0 both fire), so CREDT predicts ANTI-deference: p(follow) ~ 0.20-0.29 on trials 1/2, ~0.13-0.19 on trials 3/4, ~0.05-0.10 on trial 6, ~0.05-0.08 on trials 22/23, averaging ~0.19-0.27. The two predicted values are sign-opposite around 0.5 with a mean gap of ~0.35 (0.57 vs 0.21 at the centers).\n\nWHY IT PASSES THE ACCEPTANCE TEST AT N=25. Per subject the metric is a mean of 48 Bernoulli draws whose per-trial success probabilities lie in [0.06, 0.70]. The predicted between-subject standard deviation is ~0.07 under FDEI (var ~ 0.0049) and ~0.06 under CREDT (var ~ 0.0033). Welch's t on (0.57, 0.0049, 25) vs (0.21, 0.0033, 25) gives t ~ 19, p << 0.001 \u2014 orders of magnitude below alpha = 0.01, with huge slack against parameter heterogeneity (the boxes never overlap: FDEI's whole box stays in [0.55, 0.60], CREDT's in [0.18, 0.27]).\n\nWHY IT IS NOT TRIVIALLY TRUE FOR FDEI. The metric is a raw behavioral signature \u2014 'do people obey a 99% expert?' \u2014 not a restatement of the model. FDEI does NOT predict a value near 1: it predicts below-chance deference on the coalition trials 22/23 (0.36/0.64 for the endorsed side) and only ~0.51 on the 0.99-vs-0.94 trials, so the pooled 0.57 is a genuine, falsifiable quantitative prediction. A human value near 0.21 (systematic anti-deference) would decisively falsify FDEI, and a value near 0.57 would decisively falsify CREDT's spurious-precision/edge-distrust claim; the two theories' predictions differ by 0.35, so the human number will clearly adjudicate between them."
}
```

## Usage

```json
{
  "input_tokens": 9819,
  "output_tokens": 23278,
  "reasoning_tokens": 21708,
  "provider": "Sail Research"
}
```
