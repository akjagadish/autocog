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
**Validities (n_features=11):** [0.99, 0.99, 0.98, 0.98, 0.9, 0.9, 0.5, 0.5, 0.5, 0.5, 0.5]

**Trial pairs (n=24):**
  trial 1: A=[1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0]  B=[0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0]
  trial 2: A=[0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0]
  trial 3: A=[1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1]  B=[0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1]
  trial 4: A=[0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1]
  trial 5: A=[0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1]
  trial 6: A=[1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
  trial 7: A=[0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0]
  trial 8: A=[1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0]  B=[0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0]
  trial 9: A=[1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0]  B=[1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0]
  trial 10: A=[0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1]  B=[0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1]
  trial 11: A=[1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  trial 12: A=[1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1]
  trial 13: A=[1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
  trial 14: A=[0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0]  B=[0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1]
  trial 15: A=[1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0]
  trial 16: A=[1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1]
  trial 17: A=[1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1]
  trial 18: A=[1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1]  B=[1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1]
  trial 19: A=[1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1]
  trial 20: A=[0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0]
  trial 21: A=[0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1]
  trial 22: A=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
  trial 23: A=[1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1]
  trial 24: A=[1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]

**Rationale:** CORE LOGIC. Both theories share the identical architecture (sign votes x validity weights x softmax x lapse) and differ ONLY in the weight function. The weight functions are disjoint at three validity regimes, and each regime admits a PARAMETER-FREE knife against exactly one theory: (1) CHANCE FLANK (v=0.50): FDEI annihilates chance experts exactly (llr=0 for every delta, tau), so any trial whose discriminators are only 50% experts forces FDEI to predict EXACTLY p=0.50; BSST gives them weight -kappa_lo in [-0.235,-0.22], predicting a small graded ANTI-diagnostic ladder (this is the held-cliff calibration signature: beta*kappa_lo ~ 0.077 logits per chance cue, deliberately modest). (2) MID-BAND PLATEAU (v in [0.55,0.95]): BSST's cliff logistic is ~1e-13 at 0.90 and 0.55, so w(0.90)=w(0.55)=w_plateau EXACTLY -- any trial with tied plateau-cue counts forces BSST to predict EXACTLY p=0.50; FDEI's compressed-but-graded weights (w(0.90) in [1.08,1.17] vs w(0.55) in [0.69,0.85]) predict a systematic tilt toward the higher-validity side. (3) CLIFF FLANK (0.98 vs 0.99): BSST has w(0.98) in [0.47,0.81] (mildly trusted) but w(0.99) in [-1.35,-0.85] (strongly ANTI-diagnostic); FDEI has w(0.98) in [1.14,1.31] and w(0.99) in [1.16,1.35] -- its delta-flattening makes 98% and 99% experts nearly EXCHANGEABLE (w99-w98 in [0.02,0.04], i.e., FDEI is locked within 1 point of 0.50 on any pure 0.98-vs-0.99 contrast). Validities (11 experts): [0.99, 0.99, 0.98, 0.98, 0.90, 0.90, 0.55, 0.55, 0.50, 0.50, 0.50]. 24 unique pairs x K=4 = 96 trials.

FAMILY S (flagship cliff sign-flip). S1 (A wins both 0.99 experts, B wins both 0.90 experts): BSST box p(A) in [0.18, 0.27] (center 0.22 -- the two 99% endorsements are worth -2.2 anti-cue units against +1.81 trusted units); FDEI box [0.50, 0.55] (center 0.52 -- flattened weights make 99% barely beat 90%). DISJOINT with a ~30-point gap and a sign disagreement about whether following super-experts is above or below chance. S2 (A wins 0.99+0.99+0.98 vs B wins 0.90+0.90+0.55): BSST [0.16, 0.27] vs FDEI [0.54, 0.60] -- FDEI says A, BSST says strongly B. Mirrors S1m/S2m control side bias.

FAMILY D (the 0.98-vs-0.99 cliff that FDEI cannot see). D1 (A wins the 0.98 expert, B wins one 0.99 expert): BSST box [0.59, 0.68] (center 0.64 -- one cliff step flips the preference); FDEI locked at [0.494, 0.497] -- essentially exactly 0.50 for EVERY parameter, because w99-w98 is at most 0.04. D2 (A wins both 0.98s, B wins both 0.99s): BSST [0.68, 0.81] (center 0.75) vs FDEI [0.49, 0.50]. A 25-point dissociation on a stimulus difference (0.98 vs 0.99) that FDEI's delta-compression renders invisible. Mirrors included.

FAMILY V (cliff specificity control -- the sharpest single contrast). V1 (A wins one 0.99 expert; B wins all three 0.50 experts): BSST box [0.435, 0.499] (center 0.467: the 99% anti-vote plus three chance endorsements won by B both push toward A losing... net s = w99 + 3*kappa_lo = -0.42); FDEI box [0.57, 0.62] (center 0.594: the most diagnostic cue vs invisible chance cues). SIGN-DISJOINT. V0 (identical but A wins the 0.98 expert instead): BSST [0.58, 0.63] (center 0.605), FDEI [0.57, 0.61] (center 0.588) -- AGREEMENT by design. The V0-vs-V1 difference isolates the cliff: FDEI must predict V1 > V0 by ~0-1 points (w99 >= w98), BSST predicts V1 - V0 = -14 points crossing 0.50. A subject-level V0-V1 difference near zero falsifies BSST's cliff; near -14 falsifies FDEI's flattening. Mirrors included.

FAMILY Z (chance-flank ladder; FDEI parameter-free knife). Z1/Z2/Z3 (A wins 1/2/3 chance experts, nothing else discriminates): FDEI must predict EXACTLY 0.50 on all three for every (delta, tau, beta, epsilon). BSST predicts a monotone anti-diagnostic ladder: 0.481 / 0.463 / 0.446 (box half-widths < 0.015 -- the tightest cells in the design, since beta is pinned in [0.33,0.35] and kappa_lo in [0.22,0.235]). Any reliable negative slope of p(A) on chance-cue count, or any reliable deviation below 0.50 on Z3, falsifies FDEI with zero free parameters; a flat 0.50 falsifies the BSST chance-flank magnitude. Z3m mirrors. Note the effect is deliberately small (~2 points/cue): the held-cliff calibration pins beta*kappa_lo ~ 0.077, so the Z family is a trend/dose-response test, while the S/D/V families carry the large-effect dissociations -- recovering BOTH signatures at their predicted relative sizes is the quantitative calibration test unique to this round.

FAMILY X (chance x diagnostic interaction; mutual knife). X0 (A wins one 0.90 expert, B wins one 0.55 expert): BSST must predict EXACTLY 0.50 (plateau weights identical to machine precision); FDEI predicts [0.52, 0.54] (center 0.527). X2/X3 add 1/3 net chance cues won by A on top of the identical diagnostic background: FDEI must predict X0 = X2 = X3 EXACTLY (chance weight is identically zero -- a parameter-free flatness knife); BSST predicts a monotone ladder 0.50 -> 0.481 -> 0.446, i.e., X3 falls BELOW 0.50 while FDEI is locked ABOVE 0.52 -- a sign-disjoint cell. The X family is a double bind: X0 alone can falsify BSST (any A-tilt), the X0=X2=X3 equality alone can falsify FDEI (any slope). Xm3 mirrors. BSST additionally predicts the cross-family equalities X2=Z1 and X3=Z3 exactly (the plateau background contributes exactly 0), an internal consistency check FDEI violates (it predicts Z*=0.50 but X*=0.527).

FAMILY P (mid-band composition; BSST parameter-free knife). P1 (A wins both 0.90 experts, B wins both 0.55 experts): tied plateau counts force BSST to EXACTLY 0.50; FDEI predicts [0.53, 0.58] (center 0.553) because its graded weights give the 0.90 side a net +0.71 evidence margin. P1m mirrors (FDEI [0.42, 0.47]). Any systematic tilt on P1/P1m falsifies BSST with no parameters; exactly 0.50 falsifies FDEI's delta-flattening claim (delta would have to be ~0).

CONTROLS. T1 (the two 0.99 experts split): BOTH theories must predict exactly 0.50 -- a parameter-free side/display-bias assay. G1/G1m (all four trusted experts vs the three chance experts) are agreement anchors (BSST ~0.77/0.23, FDEI ~0.80/0.20) pinning beta and epsilon. Tie patterns mix 0/0 and 1/1 so 'count the 1s' is not viable, and every trial has 2-7 discriminators requiring genuine scanning. Sides balanced (9 A-tilted / 11 B-tilted / 4 exactly neutral under BSST centers; near-mirror under FDEI); no all-tie trials.

DIFFERENTIATION FROM PRIOR EXPERIMENTS. Unlike [1]-[3] (count-vs-weight decorrelations) and [4] (chance annihilation vs full weighting), this design's two theories are BOTH weighted and BOTH bounded, and the dissociation is carried by the SHAPE discontinuities: the 0.98/0.99 cliff (D, V families) was never contrasted directly before ([6] used 0.99 inversions but no 0.98-vs-0.99 within-pair contrast and no V0/V1 specificity pair), and the mutual parameter-free knife architecture (BSST exact-0.50 at X0/P1 vs FDEI exact-0.50 at Z/X-flatness) is new. PREDICTED PATTERN IF BSST IS RIGHT (center params): S1 .22, S2 .21, D1 .64, D2 .75, V1 .47, V0 .61, Z1/Z2/Z3 .48/.46/.45, X0 .50, X2 .48, X3 .45, P1 .50, T1 .50, G1 .77. IF FDEI IS RIGHT: S1 .52, S2 .55, D1 .50, D2 .50, V1 .59, V0 .59, Z1=Z2=Z3 .50 exactly, X0=X2=X3 .527 exactly, P1 .55, T1 .50, G1 .80. The boxes are disjoint on 18 of 24 cells; every family contains at least one parameter-free equality or exact-0.50 prediction for one theory that the other breaks in a signed, pre-registered direction.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Broad-Spectrum Skepticism Tallying (BSST), low-gain/held-cliff calibration. People integrate ALL binary expert ratings as sign votes (+1/-1/0) into a zero-sum evidence score, where the subjective weight w(v) is a wide inverted-U over STATED validity with skepticism on both flanks: (i) chance experts (v = 0.5) carry a small negative weight (~0.23 trusted-cue units) — a coin-flipper's endorsement is mildly anti-diagnostic; (ii) the entire band v ~ [0.55, 0.95] is a flat trust plateau (~0.9), making mid-band experts nearly exchangeable and generating moderate unweighted-tally behavior; (iii) distrust of claimed certainty is CONCENTRATED in a very sharp cliff between 0.98 and 0.99: w(0.98) stays mildly positive while w(0.99) plunges to ~ -1.1 trusted-cue units. The calibration claim added this round is about the GAIN-FLANK DECOUPLING: the softmax gain beta can be lowered (0.34) while the chance-flank magnitude kappa_lo is raised (0.23) so that the product beta*kappa_lo (~0.077) — the quantity that pins the chance-cue signature — is held invariant, but the product beta*|w(0.99)| (~0.37) is allowed to FALL because kappa_hi is held at its iter-2 value rather than deepened. This is the one combination never tried in nine iterations: every prior beta reduction (iters 5, 9) was bundled with a kappa_hi increase that exactly cancelled the gain on the 0.99-driven cells.

**Parameters:**
- v_rise: [0.515, 0.54]
- w_plateau: [0.88, 0.93]
- v_c: [0.9855, 0.9875]
- s_hi: [0.0025, 0.0033]
- kappa_lo: [0.22, 0.235]
- kappa_hi: [1.68, 1.72]
- beta: [0.33, 0.35]
- epsilon: [0.05, 0.07]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Broad-Spectrum Skepticism Tallying (BSST), sharp-late-cliff edition.
    # Stimulus: array of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    # Each feature j casts a sign vote sign(a_j - b_j); the vote is
    # weighted by a wide inverted-U 'skepticism' function of the
    # expert's STATED validity v:
    #   v <= v_rise :  w = -kappa_lo * (v_rise - v)/(v_rise - 0.5)
    #                  (chance experts: linear lobe, w(0.5) = -kappa_lo;
    #                   a coin-flipper's endorsement is mildly
    #                   anti-diagnostic noise)
    #   v >  v_rise :  w = w_plateau - (w_plateau + kappa_hi)
    #                        * sigmoid((v - v_c)/s_hi)
    #                  (flat trust plateau over the mid-band -- the
    #                   logistic is ~0 for every v <= 0.95, so all
    #                   mid-band experts get the full plateau weight;
    #                   then an extremely SHARP distrust cliff packed
    #                   between 0.98 and 0.99: w(0.98) stays mildly
    #                   positive, w(0.99) plunges to ~ -1.1 trusted-cue
    #                   units, saturating at -kappa_hi as v -> 1)
    # Votes are summed into a zero-sum evidence score; choice is a
    # low-gain softmax with inverse temperature beta plus a uniform
    # lapse epsilon.
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

    v_rise = float(parameters["v_rise"])
    w_plateau = float(parameters["w_plateau"])
    v_c = float(parameters["v_c"])
    s_hi = float(parameters["s_hi"])
    kappa_lo = float(parameters["kappa_lo"])
    kappa_hi = float(parameters["kappa_hi"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    a, b = stim[0], stim[1]

    # Feature-wise votes: +1 where A wins the feature, -1 where B
    # wins, 0 on ties (a tied cue is uninformative for the pair).
    votes = np.sign(a - b)

    v = np.clip(val, 0.5, 1.0)
    w = np.empty(n_features, dtype=float)

    # (i) Low flank: chance experts carry a small negative weight,
    # linearly reaching -kappa_lo at v = 0.5 and 0 at v_rise.
    lo = v < v_rise
    denom_lo = max(v_rise - 0.5, 1e-6)
    w[lo] = -kappa_lo * np.clip(v_rise - v[lo], 0.0, None) / denom_lo

    # (ii)+(iii) Flat trust plateau followed by an extremely sharp
    # distrust cliff concentrated between v = 0.98 and v = 0.99.
    # In the mid-band the logistic term is ~0, so w ~ w_plateau
    # (nearly equal weights -> tally-like counting); at 0.98 the
    # weight is only mildly discounted, at 0.99 it is strongly
    # anti-diagnostic, and it saturates at -kappa_hi as v -> 1.
    hi = ~lo
    # Numerically stable logistic.
    x = (v[hi] - v_c) / s_hi
    sig = np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)),
                   np.exp(x) / (1.0 + np.exp(x)))
    w[hi] = w_plateau - (w_plateau + kappa_hi) * sig

    # Weighted evidence score. B's score is the mirror of A's
    # because every vote is zero-sum across the two options.
    s_a = float(np.dot(w, votes))
    scores = np.array([s_a, -s_a])

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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    \"\"\"\n    Anti-super-expert choice rate on the 10 pre-registered 'cliff conflict'\n    trials (S1/S1m, S2/S2m, D1/D1m, D2/D2m, V1/V1m).\n\n    On each of these trials exactly one option is endorsed by the\n    99%-validity experts (features 0 and 1) while the rival option is\n    endorsed only by experts of validity <= 0.98, and the 99%-endorsed\n    side has no 0.90-class support (this excludes the G1/G1m agreement\n    anchors).  The metric is the fraction of these trials on which the\n    subject chooses the option NOT endorsed by the 99% experts.\n\n    BSST (sharp distrust cliff between 0.98 and 0.99, w(0.99) ~ -1.1\n    trusted-cue units, w(0.98) ~ +0.6, w(0.90) ~ +0.9, w(0.50) ~ -0.23):\n    the 99%-endorsed side is disfavored on every one of these cells, with\n    predicted anti-99% choice rates of roughly S1 .86-.90, S2 .90-.95,\n    D1 .72-.79, D2 .85-.94, V1 .70-.76  ->  overall ~0.84.\n\n    FDEI (delta ~ 0.15 compression makes 0.98 and 0.99 experts nearly\n    exchangeable, w99 - w98 in [-0.21, +0.15]): it is locked near 0.50 on\n    the D cells, mildly PRO-99% on the S cells (the 99-side also carries\n    a tiny validity edge), and strongly pro-99% on V1 (three chance cues\n    are annihilated exactly)  ->  overall anti-99% rate ~0.30-0.53,\n    center ~0.44.  Sign-disjoint from BSST with a ~0.40 mean gap.\n    \"\"\"\n\n    def as_tuple(cell):\n        try:\n            return tuple([int(x) for x in cell])\n        except Exception:\n            return None\n\n    # key: (A ratings, B ratings) -> favored option (0 = A, 1 = B),\n    # where 'favored' is the side NOT endorsed by the 99% experts.\n    TRIALS = {\n        # S1 / S1m  (A or B wins both 0.99; rival wins both 0.90)\n        ((1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0), (0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0)): 1,\n        ((0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0), (1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0)): 0,\n        # S2 / S2m  (one side wins 0.99+0.99+0.98; rival 0.90+0.90+0.55)\n        ((1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1), (0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1)): 1,\n        ((0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1), (1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1)): 0,\n        # D1 / D1m  (one side wins a 0.99; rival wins a 0.98)\n        ((0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1), (1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1)): 0,\n        ((1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1), (0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1)): 1,\n        # D2 / D2m  (one side wins both 0.99; rival wins both 0.98)\n        ((0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0), (1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0)): 0,\n        ((1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0), (0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0)): 1,\n        # V1 / V1m  (one side wins a 0.99; rival wins all three 0.50)\n        ((1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0), (0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1)): 1,\n        ((0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1), (1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0)): 0,\n    }\n\n    hits = []\n    a_col = data['option_a_ratings']\n    b_col = data['option_b_ratings']\n    r_col = data['response']\n\n    for a, b, r in zip(a_col, b_col, r_col):\n        ka = as_tuple(a)\n        kb = as_tuple(b)\n        if ka is None or kb is None:\n            continue\n        fav = TRIALS.get((ka, kb))\n        if fav is None:\n            continue\n        try:\n            ri = int(r)\n        except Exception:\n            continue\n        hits.append(1.0 if ri == fav else 0.0)\n\n    if len(hits) == 0:\n        return 0.5\n    return float(np.mean(hits))",
  "rationale": "I advocate BSST, whose signature is the sharp distrust cliff between v=0.98 and v=0.99 (w(0.99) ~ -1.1 trusted-cue units) plus a small negative weight on chance experts. The competing FDEI theory compresses diagnosticities so heavily (delta ~ 0.15) that a 0.98 and a 0.99 expert are nearly exchangeable (w99 - w98 in [-0.21, +0.15]) and chance cues are annihilated exactly. I therefore built the metric around the 10 pre-registered 'cliff conflict' trials (S1/S1m, S2/S2m, D1/D1m, D2/D2m, V1/V1m; 4 reps each = 40 trials per subject) in which exactly one option is endorsed by the 99% experts and the rival is endorsed only by <= 0.98-validity experts (the G1/G1m agreement anchors, where the 99-side also wins the 0.90 experts, are explicitly excluded by exact pattern matching). The metric is the rate of choosing AGAINST the 99%-endorsed option on these trials. Code-level recomputation of both predict functions over their full parameter boxes gives: BSST anti-99 rates of ~.86-.90 (S1), ~.90-.95 (S2), ~.72-.79 (D1), ~.85-.94 (D2), ~.70-.76 (V1) -> overall ~0.84; FDEI gives ~.42-.51 (S1), ~.30-.47 (S2), ~.47-.53 (D1), ~.44-.55 (D2), ~.27-.38 (V1) -> overall ~0.44. The contrast is sign-disjoint with a ~0.40 mean gap. Crucially the per-subject variance is small: each subject contributes 40 binary trials, so binomial sampling noise gives sd ~ 0.06 (BSST) / 0.08 (FDEI), and parameter heterogeneity within the boxes moves any single cell by at most a few points without ever flipping the favored side. Expected Welch t at N=25 is on the order of 15-20, far beyond the alpha=0.01 threshold, unlike small-effect contrasts (Z/X/P families, gaps ~0.03-0.10) which would be drowned in binomial noise at this sample size. The metric is a plain choice rate, so it is well defined both on the pooled DataFrame and on each single subject's 96-trial slice."
}
```

## Usage

```json
{
  "input_tokens": 8730,
  "output_tokens": 21302,
  "reasoning_tokens": 19118,
  "provider": "Sail Research"
}
```
