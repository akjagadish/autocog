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
**Validities (n_features=20):** [0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.75, 0.75, 0.75, 0.75, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

**Trial pairs (n=24):**
  trial 1: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  trial 3: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 7: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 9: A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
  trial 10: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 13: A=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  trial 14: A=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  trial 15: A=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
  trial 16: A=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 17: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 18: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 19: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 20: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 21: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 22: A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 23: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 24: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]

**Rationale:** NOVEL AXIS (vs. Exps 0-9): the first head-to-head between SVD (advocated) and VPS (competing) -- two theories that share the SAME distrust core. SVD's skeptic branch is functionally identical to VPS (penalty c0 + c1*v^kappa per endorsement, D = pen_B - pen_A, sigmoid(beta*D), uniform lapse), and SVD's parameter box is essentially NESTED inside VPS's (c0 [0.05,0.12] within [0.05,0.15]; c1 [0.95,1.3] within [0.7,1.3]; kappa [1.5,1.8] within [1.6,2.4] up to a 0.1 sliver). Two consequences drive the design. (1) A box-robust SIGN conflict is provably impossible: any penalty difference that is decisively signed across VPS's box is also signed on the shared region of SVD's box, so both cores always point the same way. The dissociation must therefore be QUANTITATIVE -- in CHANGES across cells and in saturation LEVELS. (2) Everything that distinguishes the theories flows from SVD's counter mixture (probability 1-w in [0.08,0.20]: a deterministic fewer-endorsement tally that is EXACTLY indifferent at matched counts) plus SVD's wider lapse. The design's keystone is a manipulation that is orthogonal to everything VPS computes: COUNT PARITY at fixed, saturated penalty evidence. Validities (20 experts): 0.95 x7, 0.75 x4, 0.50 x9. 12 base pairs, each exactly A/B-mirrored (24 unique pairs, K = 96//24 = 4, 96 trials, 8 samples per base cell per subject), perfectly side-balanced. Mid parameters used below: SVD (c0=.085, c1=1.125, kappa=1.65, beta=1.9, w=.86, eps=.18); VPS (c0=.10, c1=1.0, kappa=2, beta=2, eps=.10); box ranges verified at box corners. FAMILY 1 -- SATURATED COUNT-PARITY JUMP (pairs 1-4, primary): pair 1 is A = {50x7} vs B = {95x7}, a matched-count cell with a huge penalty gap (D = 7*c1*(0.95^k - 0.5^k) >= 2.90 on VPS's box, >= 3.81 on SVD's). Pair 2 adds ONE coin-flip endorsement to B (counts 7v8); pair 3 adds ONE coin-flip to A (8v7); pair 4 adds TWO to A (9v7). The added endorsements are 0.50 experts, whose penalty contribution (0.18-0.58 units) is trivial against the saturated D (3-6 units), so VPS's decision variable is essentially untouched: VPS's predicted P(A) is FLAT across all four pairs -- box-verified changes: Delta(pair2 - pair1) in [0, +0.06] points, Delta(pair3 - pair1) in [-0.7, 0], Delta(pair4 - pair1) in [-1.9, 0] -- while its LEVEL on pair 1 is [0.927, 0.965] (worst corner c1=.7, kappa=1.6, beta=1.6, eps=.13 still gives 0.927). SVD behaves completely differently: on pair 1 (matched counts) its counter branch contributes exactly 0.5, capping P(A) at (1-eps)(1+w)/2 + eps/2 <= 0.905 (box [0.798, 0.905], mid 0.853) -- DISJOINT from VPS's [0.927, 0.965]; the moment counts become unmatched, the counter branch flips to a deterministic vote: pair 2 (counter favors A) JUMPS to [0.876, 0.940] (mid 0.910), i.e., Delta in [+3.3, +8.8] points; pair 3 (counter favors B) DROPS to [0.717, 0.869] (mid 0.795), Delta in [-8.0, -3.5]; pair 4 drops further to [0.708, 0.869], Delta in [-8.9, -4.4]. THE CHANGES ARE BOX-DISJOINT: VPS moves at most 1.9 points where SVD moves at least 3.3 (pair 2) and at least 3.5 (pairs 3-4). A count-parity effect of >= 2 points at saturation falsifies VPS outright (no beta, kappa, c0, c1, or epsilon in its box can produce it); a flat profile (changes <= 2 points) falsifies SVD outright. The pair-1 LEVEL alone is nearly decisive: P(A) >= 0.92 falsifies SVD; <= 0.91 falsifies VPS. SVD additionally predicts the counter branch is a STEP at parity, not a gradient: pairs 3 and 4 are predicted nearly equal (mid 0.795 both). FAMILY 2 -- MATCHED-COUNT CEILING LADDER (pairs 10-12 + pair 1): k x 0.50 vs k x {0.75 or 0.95} at k = 2, 2, 4, 7. Both theories prefer the coin-flip side, but VPS climbs to its ceiling (mid: 0.750, 0.889, 0.945, 0.950; box at k=7: [0.927, 0.965]) while SVD is compressed toward 0.5 by the factor (1-eps)*w ~ 0.65 and plateaus (mid: 0.702, 0.803, 0.849, 0.853; box at k=7: [0.798, 0.905]). The asymptote LEVEL (>= 0.92 vs <= 0.91) and the ~10-point mid gap at every saturated rung is the signature; SVD's between-subject dispersion on the asymptote (wide eps in [0.12,0.24] and w in [0.80,0.92]) is a secondary heterogeneity signature VPS cannot produce (its eps box [0.07,0.13] spans 3 points of asymptote, SVD's spans 11). FAMILY 3 -- COUNTER BUFFERING ON CONFLICT CELLS (pairs 5-7): A = {95 x j} vs B = {50 x 2j}, j = 1, 2, 3. Here the skeptic core of BOTH theories prefers B (the coin-flip side: 2 coin-flips cost less suspicion than one 0.95 endorsement, sign-robustly: VPS's D = j*[c0 + c1*(2*0.5^k - 0.95^k)] < 0 at every box corner), but SVD's counter branch votes for A (fewer endorsements), dragging SVD back toward chance. VPS dives: mid P(A) = 0.368, 0.256, 0.176 (boxes [0.21,0.49], [0.09,0.48], [0.05,0.47]) -- a deepening B-majority, B-rate 0.63 -> 0.74 -> 0.82 at mid. SVD hovers: mid P(A) = 0.481, 0.411, 0.352 (boxes [0.44,0.51], [0.35,0.49], [0.30,0.47]) -- near chance at j=1 and never below 0.30. A B-rate >= 0.60 on pair 5 falsifies SVD; a B-rate <= 0.52 falsifies VPS; at j=3 the mid separation is 17.6 points. FAMILY 4 -- EXCHANGE-RIDGE CELLS (pairs 8-9): A = {75 x 2} vs B = {50 x 4}, and its x2 amplification A = {75 x 4} vs B = {50 x 8}. These sit exactly at the coin-flip/0.75 exchange ridge where VPS's penalty difference is pinned near zero (VPS mid D = +0.075 and +0.15, P(A) = 0.533 and 0.567; box [0.42,0.69] and [0.33,0.81] -- near chance), because 4 coin-flips cost roughly 2 x 0.75 in suspicion. SVD's exchange rate is slightly lower AND its counter branch votes for the 2-endorsement (or 4-vs-8) side, so SVD delivers a clear A-majority: mid 0.634 and 0.687 (boxes [0.59,0.72], [0.60,0.84]). SVD's P(A) rises with the amplification (+5 mid points, robust: SVD's D > 0 at every box corner since c0 + c1*(2*0.5^k - 0.75^k) >= 0.034 > 0 for kappa in [1.5,1.8], so doubling the coalition strictly increases its skeptic probability while the counter vote stays fixed). UNFAKEABILITY IN BOTH DIRECTIONS (joint profile): because the boxes are nested, single-cell mimicry is possible only at extreme corners, but the corners that let VPS imitate SVD on one family force it away on the others. The VPS corner that comes closest to SVD on Family 3 (c0=.15, c1=.7, kappa=1.6, the flattest corner, P(A)=0.489 on pair 5) simultaneously predicts P(A) = 0.927 on pair 1 with a count-jump of ~0 -- while SVD's box requires P(A) <= 0.905 AND a jump >= +3.3; conversely, the SVD corner that maximizes pair 1 (eps=.12, w=.92: 0.905, jump +3.5) predicts pair 5 at 0.51, far above VPS's ceiling there (0.49), and pair 7 at 0.47 vs VPS's 0.47 only at its flattest corner where pair 1 sits at 0.93. The conjunction {pair-1 level, count-jump size, Family-3 depth, ladder asymptote} is a profile neither box can approximate: VPS is a flat line at 0.93-0.97 across Family 1 with deep B-majorities in Family 3; SVD is a V-shaped step function on Family 1 (0.85 -> 0.91 -> 0.80 -> 0.80 at mid) with near-chance performance in Family 3 and an 0.85 asymptote. All 24 pairs are non-degenerate (every pair has discriminating endorsements, both options carry >= 1 endorsement and >= 11 non-endorsements, no all-tie pairs, and no pair leaves both theories guessing -- the closest cells, pairs 5 and 8, are exactly where the boxes maximally separate at mid), and every trial requires actually comparing the two rating vectors. Both theories are history-free (validities in the instructions), so K=4 repetition is harmless.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Softened Validity-Distrust with Count Mixture (SVD). Every expert endorsement is a liability whose magnitude increases monotonically in the endorser's RAW claimed validity: d_j = c0 + c1*v_j^kappa, with no dampening near v=0.5 and no trust refund. Decision making is a dual-process mixture at the trial level: with probability w the subject is a 'skeptic' who integrates the accumulated-suspicion difference D = P(B) - P(A) through sigmoid(beta*D); with probability (1-w) the subject is a 'counter' who simply chooses the option with FEWER total endorsements (indifferent at matched counts). Two heterogeneity mechanisms soften the pooled extremes and capture between-subject variability: (i) the sigmoid slope beta is sampled from a WIDE per-subject range, so the population is a mixture of shallow and steep sigmoids — sensitive near D=0 (matched-count validity-composition trials) yet compressed at large |D| (unequal-count trials); (ii) the uniform lapse epsilon is also sampled from a wide per-subject range, which pulls every saturated large-D assay toward 0.5 while adding genuine between-subject dispersion in overall choice rates.

**Parameters:**
- c0: [0.05, 0.12]
- c1: [0.95, 1.3]
- kappa: [1.5, 1.8]
- beta: [1.2, 2.6]
- w: [0.80, 0.92]
- epsilon: [0.12, 0.24]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Softened Validity-Distrust with Count Mixture (SVD).
    # Stimulus: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    #
    # Branch 1 (skeptic, prob w): each endorsement is a LIABILITY whose
    # magnitude grows with the endorser's RAW claimed validity:
    #   d_j = c0 + c1 * v_j^kappa   (kappa >= 1.5; NO dampening near
    #                                v = 0.5, NO trust refund)
    #   accumulated suspicion: P(X) = sum_j x_j * d_j
    #   D = P(B) - P(A)  (positive favors A)
    #   P_skeptic(A) = sigmoid(beta * D)
    # Branch 2 (counter, prob 1-w): pure tallying on endorsement COUNTS;
    #   choose the option with FEWER total endorsements; exactly
    #   indifferent (0.5) at matched counts.
    # Mixture: P_core(A) = w * P_skeptic(A) + (1-w) * P_count(A),
    # mixed with a uniform lapse epsilon.
    # History is ignored: validities are in the instructions, so the
    # distrust weights are fixed for the whole block. beta AND epsilon
    # are sampled per subject from WIDE ranges: between-subject
    # heterogeneity in the sigmoid slope and in the lapse rate are
    # explicit parts of the theory.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"SVD expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n_features = stim.shape[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.shape[0] != n_features:
        raise ValueError(
            f"validities length {v.shape[0]} != n_features {n_features}."
        )
    v = np.clip(v, 0.5, 1.0)  # defensive; no logs taken, v=1 harmless.

    kappa = float(parameters["kappa"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    c0 = float(parameters["c0"])
    c1 = float(parameters["c1"])
    w_mix = float(parameters["w"])

    # --- Skeptic branch: raw validity-proportional distrust charge ---
    d = c0 + c1 * np.power(v, kappa)
    pen_a = float(np.dot(d, a))
    pen_b = float(np.dot(d, b))

    # D > 0 favors A (B carries more accumulated suspicion).
    D = pen_b - pen_a
    x = beta * D
    p_skeptic_a = 0.5 * (1.0 + np.tanh(0.5 * x))  # stable sigmoid

    # --- Counter branch: pure endorsement-count tallying ---
    cnt_a = float(np.sum(a))
    cnt_b = float(np.sum(b))
    if cnt_a < cnt_b:
        p_count_a = 1.0      # A has fewer endorsements -> prefer A
    elif cnt_a > cnt_b:
        p_count_a = 0.0
    else:
        p_count_a = 0.5      # matched counts -> indifferent

    # --- Dual-process mixture ---
    p_a = w_mix * p_skeptic_a + (1.0 - w_mix) * p_count_a
    p_core = np.array([p_a, 1.0 - p_a], dtype=float)

    # Independent lapse: with probability epsilon pick uniformly.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Validity-Proportional Skepticism (VPS), a.k.a. Confidence-Distrust Defect Integration. Subjects treat every expert endorsement not as an asset but as a liability whose magnitude scales with the endorser's claimed validity: a highly valid expert's praise is a strong claim, and strong claims attract the most suspicion. On each trial the decision maker computes a penalty for each option, P(X) = sum_j x_j * (c0 + c1 * v_j^kappa), where c0 is a flat defect-count term (c1 = 0 recovers pure unweighted defect counting) and c1 * v_j^kappa is the confidence-distrust term, with penalty INCREASING in the endorser's validity and kappa controlling how steeply. The subject chooses the lower-penalty option via P(A) = sigmoid(beta * (P(B) - P(A))) mixed with a uniform lapse epsilon. History is ignored because validities are communicated in the instructions, so skepticism weights are fixed for the whole block. VPS reproduces the sign of every observed metric while fixing the one structural blind spot of pure defect counting (SDI): on matched-count trials with a large validity gap (e.g., a single 0.90 endorser vs a single 0.60 endorser), SDI is pinned at 0.5 while VPS predicts a clear majority for the LOWER-validity endorser — exactly what the Experiment 6 composite (observed 0.193) requires.

**Parameters:**
- kappa: [1.6, 2.4]
- beta: [1.6, 2.4]
- epsilon: [0.07, 0.13]
- c0: [0.05, 0.15]
- c1: [0.7, 1.3]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Validity-Proportional Skepticism (VPS) /
    # Confidence-Distrust Defect Integration.
    #
    # Stimulus: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    #
    # Each endorsement is a LIABILITY whose magnitude scales with the
    # endorser's claimed validity (a strong claim attracts suspicion):
    #   penalty for option X:  P(X) = sum_j x_j * (c0 + c1 * v_j^kappa)
    #     c0        -> flat defect-count term (c1 = 0 recovers pure
    #                  unweighted endorsement counting)
    #     c1*v^kappa -> confidence-distrust term, INCREASING in validity
    # Decision variable:  D = P(B) - P(A)   (positive favors A)
    # P(A) = sigmoid(beta * D), mixed with a uniform lapse epsilon.
    # History is ignored: validities are in the instructions, so the
    # skepticism weights are fixed for the whole block.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"VPS expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n_features = stim.shape[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.shape[0] != n_features:
        raise ValueError(
            f"validities length {v.shape[0]} != n_features {n_features}."
        )
    # Validities live in [0.5, 1.0]; clip defensively (no logarithms
    # are taken, so v = 1 is harmless here).
    v = np.clip(v, 0.5, 1.0)

    kappa = float(parameters["kappa"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    c0 = float(parameters["c0"])
    c1 = float(parameters["c1"])

    # Per-endorser liability: flat suspicion plus a
    # validity-proportional confidence-distrust charge.
    d = c0 + c1 * (v ** kappa)

    # Total accumulated suspicion for each option.
    pen_a = float(np.dot(d, a))
    pen_b = float(np.dot(d, b))

    # D > 0 favors A (B carries more accumulated suspicion than A).
    D = pen_b - pen_a

    # Numerically stable sigmoid: 0.5 * (1 + tanh(x/2)).
    # D == 0 (exact tie in penalties) -> exactly 0.5.
    x = beta * D
    p_a = 0.5 * (1.0 + np.tanh(0.5 * x))
    p_core = np.array([p_a, 1.0 - p_a], dtype=float)

    # Independent lapse: with probability epsilon pick uniformly.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Saturated-conflict compliance rate.\n    #\n    # Logic: using a FIXED reference parameterization of the shared skeptic\n    # core (c0=0.10, c1=1.0, kappa=2.0 -> d_j = 0.10 + v_j^2), compute the\n    # skeptic decision variable D = pen(B) - pen(A) for every trial, plus the\n    # raw endorsement counts cnt_A, cnt_B.\n    #\n    # Keep only trials that are (i) SATURATED for the skeptic core\n    # (|D| > 2.0, so any plausible beta in either theory's box drives the\n    # sigmoid to ~1) and (ii) in CONFLICT: the skeptic-preferred side is the\n    # side with MORE endorsements (the pure counter branch votes the other\n    # way). In this fixed design these are exactly the count-parity-jump\n    # cells pairs 3, 4 and their mirrors 15, 16 (16 trials per subject).\n    #\n    # On these cells the two theories make disjoint predictions:\n    #   VPS:  P(pick skeptic side) = (1-eps)*~1 + eps/2  ~ 0.95 (box 0.92-0.98)\n    #   SVD:  P = (1-eps)*(w*1 + (1-w)*0) + eps/2 = (1-eps)*w + eps/2\n    #         ~ 0.80 (box 0.73-0.87, since the counter mixture drags the\n    #         saturated response toward the fewer-endorsement side).\n    # The metric is the observed fraction of choices that follow the\n    # skeptic-preferred side on these cells.\n\n    validities = [0.95] * 7 + [0.75] * 4 + [0.50] * 9\n    d_ref = [0.10 + 1.0 * (v ** 2) for v in validities]\n\n    hits = []\n    for _, row in data.iterrows():\n        a_raw = row[\"option_a_ratings\"]\n        b_raw = row[\"option_b_ratings\"]\n        a = [int(float(x)) for x in a_raw]\n        b = [int(float(x)) for x in b_raw]\n\n        pen_a = sum(di for di, xi in zip(d_ref, a) if xi == 1)\n        pen_b = sum(di for di, xi in zip(d_ref, b) if xi == 1)\n        D = pen_b - pen_a  # >0 favors A\n\n        cnt_a = sum(1 for x in a if x == 1)\n        cnt_b = sum(1 for x in b if x == 1)\n\n        # (i) saturation for the skeptic core\n        if abs(D) <= 2.0:\n            continue\n        # counter branch must be decisive (not indifferent)\n        if cnt_a == cnt_b:\n            continue\n\n        skeptic_prefers_a = D > 0\n        counter_prefers_a = cnt_a < cnt_b  # fewer endorsements preferred\n\n        # (ii) conflict: counter branch votes AGAINST the saturated skeptic core\n        if skeptic_prefers_a == counter_prefers_a:\n            continue\n\n        resp = int(row[\"response\"])  # 0 = chose A, 1 = chose B\n        followed_skeptic = (resp == 0) if skeptic_prefers_a else (resp == 1)\n        hits.append(1.0 if followed_skeptic else 0.0)\n\n    if len(hits) == 0:\n        return 0.5\n    return float(np.mean(hits))",
  "rationale": "TARGETED CONTRAST: saturated conflict cells. The design's Family-1 count-parity cells 3, 4 (and mirrors 15, 16) are the one place where the two theories' boxes make DISJOINT LEVEL predictions, not just different change scores. On these trials the shared skeptic core is saturated (|D| ~ 3.9-4.2, so sigmoid(beta*D) ~ 1 for every beta in either theory's box) and it favors the side with MORE endorsements, while SVD's counter branch (prob 1-w) deterministically votes for the FEWER-endorsement side. Hence: VPS predicts P(follow skeptic) = (1-eps)*1 + eps/2 \u2248 0.95 (box 0.92-0.98, tight because VPS's eps box is narrow), whereas SVD predicts P = (1-eps)*w + eps/2 \u2248 0.80 (box 0.73-0.87). Mid gap \u2248 15.5 points, and even the extreme corners of the two boxes do not overlap (SVD max 0.87 < VPS min ~0.92).\n\nWHY THIS METRIC HAS LOW BETWEEN-SUBJECT VARIANCE: (a) Each subject contributes 16 trials of exactly this cell type (4 unique pairs x 4 reps), so the binomial component of per-subject variance is only ~p(1-p)/16 (sd ~ 0.10). (b) The cell is saturated, so SVD's wide beta range [1.2, 2.6] contributes almost nothing (even the weakest corner gives sigmoid ~ 0.99). (c) The remaining heterogeneity (eps in [0.12,0.24], w in [0.80,0.92]) moves the SVD per-subject value only over ~[0.73, 0.87] (sd ~ 0.03), far less than the 15-point mean gap. VPS's per-subject spread is even tighter (~0.05 sd).\n\nEXPECTED NUMBERS: SVD point estimate ~ 0.79-0.80 with between-subject variance ~ 0.011; VPS ~ 0.95 with variance ~ 0.003. Welch's t at N=25: t \u2248 0.155 / sqrt((0.011+0.003)/25) \u2248 6.5, p << 0.001, comfortably below alpha = 0.01.\n\nROBUSTNESS OF CELL IDENTIFICATION: cells are selected by a generic rule (reference-weighted |D| > 2.0, count mismatch, sign(D) opposite to the fewer-endorsement side) rather than by hardcoded trial indices; with this design's fixed validities and pairs, the rule selects exactly pairs 3, 4, 15, 16 \u2014 the next-largest conflicting |D| in the whole design is 0.91 (pair 19), and all other large-|D| cells are either count-matched (counter indifferent: pairs 1, 12, 13, 24) or counter-agreeing (pairs 2, 14), so the selection is unambiguous. The metric works identically on the pooled DataFrame and on a single subject's slice (each slice contains all 96 trials including the 16 target trials)."
}
```

## Usage

```json
{
  "input_tokens": 9717,
  "output_tokens": 15756,
  "reasoning_tokens": 14163,
  "provider": "Sail Research"
}
```
