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
**Validities (n_features=22):** [1.0, 1.0, 1.0, 1.0, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

**Trial pairs (n=24):**
  trial 1: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  trial 2: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 3: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  trial 4: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  trial 5: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 6: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 10: A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 13: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 14: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 15: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 16: A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 17: A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 18: A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 19: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 20: A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 21: A=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
  trial 22: A=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
  trial 23: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 24: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

**Rationale:** ADVERSARIAL LOGIC. HEAI (advocated) and SCWI (competing) are mechanistically identical anti-validity additive integrators: a rating of 1 is a defect, tied cues cancel exactly, choice is a softmax plus symmetric lapse, and both carry a 16% true-guesser subpopulation. The only wedge is the validity-to-weight map. Every prior experiment ([4]-[8]) attacked this wedge from the STEEP side (the old gamma-box [1.4, 2.4] and HEAI's steep mode), using count ratios at or above each theory's flip window. The unexploited territory is HEAI's SHALLOW mode: 75% of engaged subjects with gamma ~ N(1.04, 0.22) on [0.60, 1.42] - i.e. mostly SUBLINEAR or barely supralinear. SCWI's implied power exponent is pinned by its tau-box to [1.10, 1.23] at the (1.00, 0.50) pair (ln[(1+tau)/(0.5+0.25tau)]/ln 2) and [1.09, 1.22] at (1.00, 0.60). Hence ~60% of shallow subjects have validity gradients SHALLOWER than ANY admissible SCWI subject: they prefer the high-validity stack at count ratios where SCWI, for every (tau, attention, beta, epsilon), prefers the low-validity stack. No prior design sampled this ratio region: [4] and [8] placed rungs at r >= 2.5-2.67, [5] swept low-cue validity at fixed r, and their kills all targeted gamma >= 1.4.

INSTRUMENT: THE RECIPROCAL-RATIO LADDER. With a endorsements on v=1.00 cues (where HEAI's weight is EXACTLY 1, gamma-invariant) against b endorsements on v_l cues, an anti-validity integrator prefers the top-stack option H iff w(1)/w(v_l) < b/a = r. SCWI's flip ratio is (1+tau)/(v_l + tau*v_l^2): [2.14, 2.35] at v_l=0.50, [1.76, 1.89] at v_l=0.60. HEAI's flip ratio is (1/v_l)^gamma, so the cell with b = a/v_l (r = 2.0 on ladder 1: 4x1.00 vs 8x0.50; r = 5/3 on ladder 2: 3x1.00 vs 5x0.60) flips EXACTLY at gamma = 1 - the shallow mode's median. These TWIN KILLS (pairs 3, 9 and 7, 10, plus mirrors) are parameter-free for SCWI: its margin is 2*tau (ladder 1) and 1.2*tau (ladder 2), strictly positive for ALL tau in [0.15, 0.42], so EVERY engaged SCWI subject prefers the low stack (per-trial P(H) <= 0.37 even at the tau=0.15 edge with mid beta=1.8). HEAI instead splits its shallow mode at gamma=1: the 43% with gamma < 1 prefer H with margins 0.3-1.3 at beta median 2.6 (P(H) per trial 0.6-0.95), the steep mode prefers L, guessers sit at 0.5.

QUANTITATIVE POPULATION PREDICTIONS, P(H): L1a (r=1.5): both <= 0.12 (parameter-free agreement-L: flip needs gamma < 0.585, below HEAI's 0.60 truncation, and SCWI margin 1+2.5tau > 0). L1b (r=1.75): SCWI <= 0.15, HEAI ~0.16. L1c/K2rot (r=2.0): SCWI 0.20-0.33, HEAI ~0.36-0.40 - HEAI ABOVE SCWI's ceiling. L1d (r=2.25): SCWI ~0.50 (its internal tau-rung, splitting at tau=0.286 with margins <= 0.24), HEAI ~0.42-0.46. L1f (r=3.0, mode separator): SCWI 0.86-0.92 (margin 2-tau in [1.58, 1.85], every engaged subject >= 0.9), HEAI 0.66-0.72 - HEAI BELOW SCWI's floor. L2a (r=4/3): both <= 0.12 (agreement-L; HEAI flip needs gamma < 0.563, outside its box). L2b/K4rot (r=5/3): SCWI 0.18-0.30, HEAI ~0.38. L2c (r=2.0): SCWI 0.85-0.90, HEAI 0.55-0.60. So the two theories' psychometric curves CROSS TWICE with opposite signs in different ratio regions: HEAI exceeds SCWI exactly where the shallow mode's sub-linearity bites (r = 1/v_l), and SCWI exceeds HEAI where HEAI's steep mode (21% of the population, confidently L everywhere) and guessers cap the composite. SCWI cannot be simultaneously above 0.84 at r=3.0/L2c and below... it is PINNED: P(H) <= 0.33 on the twin kills and >= 0.84 on L1f for ALL parameters. Observing twin-kill >= 0.35 OR L1f <= 0.80 falsifies SCWI outright; observing L1f >= 0.84 with twin-kill <= 0.30 falsifies HEAI (whose L1f ceiling is 0.71 even if every shallow subject were deterministic).

SUBJECT-LEVEL (PRIMARY) STATISTICS. (S1) H-majority (>= 60% H choices) on the 32-trial twin-kill composite (pairs 3, 7, 9, 10 + mirrors, 4 reps each): HEAI predicts 20-27% of subjects (shallow gamma <= ~0.9); SCWI predicts < 3-5% (only guesser binomial flukes - an engaged SCWI subject's per-trial P(H) never exceeds 0.37, so a 32-trial majority is vanishingly unlikely). (S2) L-majority at the mode separator L1f (8 trials): HEAI ~20-22% (the steep mode; gamma* = log2(3) = 1.585 lies in the EMPTY inter-mode band (1.42, 1.65), so HEAI's per-subject distribution there is bimodal with an empty middle - shallow >= 0.9, steep <= 0.1); SCWI ~0-2%, unimodal-high. (S3) FLIP LOCATION on ladder 1: SCWI's every engaged subject steps L (r=2.0) -> chance (2.25) -> H (3.0), flip confined to [2.14, 2.35] with near-zero between-subject spread; HEAI's flips spread over r = 2^gamma in [1.52, 2.68] (graded across the 1.75/2.0/2.25 rungs for roughly half of engaged subjects), and 21% of subjects NEVER flip (steep: L even at r=3.0, since 1.65 > 1.585). (S4) CROSS-LADDER CONSISTENCY WEB: a shallow subject's two flip ratios must satisfy r2* = r1*^(ln(5/3)/ln 2) = r1*^0.737 with a single gamma in [0.6, 1.4]; e.g. H at BOTH twin kills implies gamma < 1; H at L1c but L at L2b violates any single gamma and falsifies HEAI's per-subject homogeneity. Under SCWI, H at either twin kill is itself near-impossible, so SCWI generates no such covariance. (S5) STEP-SIZE profile: SCWI concentrates essentially its entire rise into single inter-rung intervals (ladder 2: P(2.0) - P(5/3) ~ 0.65; ladder 1: P(3.0) - P(2.25) ~ 0.4); HEAI's rise is spread and capped (ladder 2 difference ~0.19; ladder 1 rise ~0.26 with ceiling 0.70).

CONFOUNDS AND THIRD PARTIES. Bottom anchors L1a/L2a are parameter-free agreement-L cells for BOTH theories (all shallow gamma >= 0.60 > 0.585/0.563, all SCWI margins >= 0.83): a TTB subject (follows the top discriminating cue) and a fewer-1s heuristic both pick the 4-1 top stack there and are exposed; a tallying/more-1s subject picks the low stack on every ladder cell, mimicking anti-validity, but is exposed by the ANCHOR (all-1 vs all-0), where both target theories predict P(choose all-1) ~ epsilon/2 while tallying predicts ~1; pro-validity integrators are likewise exposed. TIE pairs pin side/response bias at exactly 0.500. All critical pairs are mirrored. SCWI's per-cue attention (U[0.9, 1.1]) is defeated by cue rotation: the twin kills use four different cue subsets spanning all three validity classes (top cues 0-3 vs 1-3; 0.50-cues 10-17 vs 14-21; 0.60-cues 4-8 vs 5-9), so an H-majority on the composite requires ~2 SD adverse class-mean attention on three independent classes simultaneously PLUS tau at its floor; the joint probability is < 2-3%. L1f's SCWI margin (1.58-1.85) needs > 5 SD to flip. HEAI has no attention parameter at all. No feedback keeps weights anchored to the instructed validities; every critical trial requires comparing large multi-cue stacks (4-12 cues per side), satisfying the min-RT viewing requirement.

DESIGN. 22 experts: 4 at 1.00, 6 at 0.60, 12 at 0.50. 24 unique pairs x K = 96//24 = 4 repetitions = 96 trials (40 ladder-1, 24 ladder-2, 16 rotation, 8 tie, 8 anchor trials per subject).

DECISION RULES. (i) HEAI is supported, and SCWI falsified, if >= 15% of subjects show H-majority on the twin-kill composite (SCWI ceiling ~5%), psychometric flips appear inside r in (1.6, 2.1) and are jointly consistent across both ladders with a single gamma-hat in [0.6, 1.4], ~20% of subjects show the steep signature (L at every rung including r=3.0, producing the bimodal empty-middle distribution at L1f), and the population profile shows P(H|twin kill) >= 0.35 with P(H|r=3.0) <= 0.80 - a joint pattern outside SCWI's closed parameter box. (ii) SCWI is supported, and HEAI falsified, if the twin-kill composite <= 0.30 with no H-majority subjects beyond guesser noise, L1f >= 0.84 and L2c >= 0.85, every subject's flip confined to (2.0, 3.0) on ladder 1 and (5/3, 2.0) on ladder 2 with near-zero between-subject spread, and the whole rise concentrated in single inter-rung steps. (iii) BOTH are falsified if the top stack is chosen on the bottom anchors (TTB / fewer-1s / pro-validity), the all-1 option is chosen well above lapse on the ANCHOR (tallying / pro-validity), everything sits flat at 0.5 (pure guessing), flips appear below r = 1.6 (gamma < 0.68, outside HEAI's truncation and below SCWI's window), or the profile fits neither map.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Heterogeneous-Elasticity Anti-Validity Integration (HEAI), bimodal-population refinement. Subjects integrate ALL expert ratings additively into a per-option subjective value, but a rating of 1 is comprehended as a DEFECT (evidence against the option) — a sign inversion of the rating direction. The anti-weight of expert j is a per-subject power law of the instructed validity, w_j = -v_j^gamma, where gamma ('validity elasticity') is drawn from a BIMODAL continuous population: a shallow majority (75% of engaged subjects, gamma ~ N(1.04, 0.22) truncated [0.60, 1.42]) that sits cleanly below the kill-design count-ratio thresholds (~1.31-1.34) and therefore follows the high-validity stack, and a steep minority (25% of engaged subjects, gamma ~ N(2.10, 0.26) truncated [1.65, 2.75]) that sits cleanly above them, whose near-lexicographic top-cue dominance drives the Exp-2 dissociation and keeps the Exp-8/Exp-4 composites from overshooting. Inverse temperature is mode-linked (shallow: lognormal median ~2.6; steep: lognormal median ~3.3; both heavy-tailed), a 16% guesser subpopulation (beta ~ 0) and a symmetric lapse epsilon are retained unchanged. The population mean elasticity (~1.30) stays below the arbiter's 1.4 ceiling. The key change from the falsified wide-unimodal shape is that almost no mass is wasted in the ambiguous 1.2-1.6 band where subjects sit at chance on the diagnostic cells: the shallow mode's thin upper shoulder (1.32-1.42) is retained deliberately, because subjects just above the kill thresholds miss Exp 8 while still hitting the higher-threshold Exp 5/7 ladder cells — the lever that lets Exp 8's mean and variance stay calibrated without capping Exp 5/6/7.

**Parameters:**
- regime_seed: [0, 1]
- gamma_seed: [0, 1]
- beta_seed: [0, 1]
- epsilon_seed: [0, 1]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Heterogeneous-Elasticity Anti-Validity Integration (HEAI), iter 3.
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    # Mechanism (UNCHANGED from the accepted iter-2 base): every cue
    # contributes signed evidence w_j = -v_j**gamma per unit rating
    # (a rating of 1 is a defect; the anti-weight is a per-subject
    # power law of the instructed validity). Choice is a numerically
    # stable softmax over beta * option values, mixed with a symmetric
    # lapse epsilon. History is ignored (no trial-by-trial feedback).
    #
    # Population edits on top of the accepted iter-2 base (mechanism
    # untouched), implementing the iter-2 critic's diagnosis that a
    # wide UNIMODAL gamma dumps its spread into the ambiguous
    # threshold band (gamma ~ 1.2-1.6) where subjects sit at ~chance
    # on every diagnostic cell, capping the means on BOTH sides:
    #   * GAMMA MADE BIMODAL: the engaged population is now a mixture
    #     of a SHALLOW majority (75%: N(1.04, 0.22) trunc [0.60, 1.42])
    #     and a STEEP minority (25%: N(2.10, 0.26) trunc [1.65, 2.75]),
    #     with minimal mass in between. The shallow mode sits cleanly
    #     below the kill-design count-ratio thresholds (~1.31-1.34 on
    #     Exp 8; ~1.32+ on the Exp 5 ladder), so it drives high-stack
    #     adherence on Exp 5/6/7/8; its thin upper shoulder (1.32-1.42,
    #     ~10% of the mode) deliberately remains to keep Exp 8's mean
    #     and between-subject variance calibrated. The steep mode sits
    #     cleanly above ~1.8 (critic's prescription; extends to 2.75 so
    #     the top cue can dominate three lower cues), which is what
    #     lifts the Exp 2 dissociation mean/variance and holds Exp 4
    #     down. Population mean elasticity ~ 1.30 < 1.4 (arbiter box).
    #   * BETA MODE-LINKED AND HOTTER: shallow subjects draw lognormal
    #     beta, median exp(0.95) ~ 2.6 (up from ~2.3), sigma 0.70,
    #     clip [0.6, 7.0]; steep subjects draw a hotter lognormal,
    #     median exp(1.20) ~ 3.3, sigma 0.60, clip [1.2, 8.0] (the
    #     steep 'confident lexicographic' subjects are also the
    #     sharpest). Hotter shallow beta sharpens engaged adherence on
    #     Exp 5/6/7; hotter steep beta maximizes the Exp 2 contribution
    #     per steep subject while pushing steep Exp 8/Exp 4 hits down.
    #   * Guesser fraction (16%), epsilon ranges, seeds, and everything
    #     else are untouched.
    import math
    import numpy as np
    from scipy.special import ndtri

    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"HEAI expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    def _to_vec(raw):
        if raw is None:
            return None
        if isinstance(raw, str):
            s = raw.strip().strip('[]()')
            toks = [t for t in s.replace(',', ' ').split() if t != '']
            if not toks:
                return None
            return np.array([float(t) for t in toks])
        arr = np.asarray(raw, dtype=float).ravel()
        return arr if arr.size > 0 else None

    val = _to_vec(parameters.get("validities"))
    if val is None or val.size != n_features:
        # Defensive fallback: uniform mid-range validities.
        val = np.full(n_features, 0.75)
    val = np.clip(val, 1e-3, 1.0)

    def _seed(name):
        try:
            x = float(parameters[name])
        except (KeyError, TypeError, ValueError):
            x = 0.5
        if not np.isfinite(x):
            x = 0.5
        return min(max(x, 0.0), 1.0)

    u = _seed("regime_seed")
    # Keep CDF seeds strictly inside (0, 1) so ndtri stays finite.
    gs = min(max(_seed("gamma_seed"), 1e-6), 1.0 - 1e-6)
    bs = min(max(_seed("beta_seed"), 1e-6), 1.0 - 1e-6)
    es = _seed("epsilon_seed")

    if u < 0.16:
        # Guesser subpopulation (~16%): beta ~ 0 makes the softmax core
        # essentially uniform on every trial; lapse is elevated.
        t = u / 0.16
        beta = 0.02 + 0.08 * t          # 0.02 .. 0.10
        epsilon = 0.12 + 0.13 * es      # 0.12 .. 0.25
        gamma = 1.12                    # irrelevant at beta ~ 0
    else:
        if gs < 0.75:
            # SHALLOW majority (~75% of engaged): gamma ~ N(1.04, 0.22)
            # truncated to [0.60, 1.42]. Cleanly below the kill-design
            # thresholds (~1.31-1.34), with a thin upper shoulder that
            # keeps Exp 8's mean/variance calibrated.
            t = gs / 0.75
            tt = min(max(t, 1e-4), 1.0 - 1e-4)
            gamma = 1.04 + 0.22 * float(ndtri(tt))
            gamma = min(max(gamma, 0.60), 1.42)

            # Hot shallow beta: lognormal median ~2.6, heavy tail.
            z_b = float(ndtri(bs))
            beta = math.exp(0.95 + 0.70 * z_b)
            beta = min(max(beta, 0.6), 7.0)
        else:
            # STEEP minority (~25% of engaged): gamma ~ N(2.10, 0.26)
            # truncated to [1.65, 2.75]. Cleanly above ~1.8 so the top
            # cue can dominate three lower cues (the Exp 2 lever).
            t = (gs - 0.75) / 0.25
            tt = min(max(t, 1e-4), 1.0 - 1e-4)
            gamma = 2.10 + 0.26 * float(ndtri(tt))
            gamma = min(max(gamma, 1.65), 2.75)

            # Steep subjects are also sharper: lognormal median ~3.3.
            z_b = float(ndtri(bs))
            beta = math.exp(1.20 + 0.60 * z_b)
            beta = min(max(beta, 1.2), 8.0)

        # Symmetric lapse for engaged subjects (unchanged).
        epsilon = 0.03 + 0.14 * es      # 0.03 .. 0.17

    # Anti-validity power-law cue weights: a rating of 1 counts AGAINST
    # the option, with magnitude v_j**gamma.
    w = -np.power(val, gamma)

    scores = np.array([float(np.dot(w, stim[0])), float(np.dot(w, stim[1]))])

    # Numerically stable softmax over the two option values.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Guard against float drift.
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Signed-Cue Weighted Integration (SCWI). When choosing between two options described by binary expert ratings, subjects do not use a lexicographic cascade (TTB) and do not merely count wins (Tallying). Instead they integrate ALL cues additively into a single subjective value per option — but the mapping from a rating to evidence carries a systematic SIGN INVERSION: a rating of 1 is treated as a defect/criticism rather than an endorsement (a comprehension inversion of the rating direction). Moreover, the inversion is not uniform across cues: the subjective weight of cue j is w_j = -(v_j + tau * v_j^2), i.e. an anti-validity weighting that grows SUPRALINEARLY with the instructed validity — the endorsements of the most trusted experts are avoided most strongly, while low-validity experts are nearly ignored (their anti-weight is small in magnitude). The option with the higher (less negative) signed value is preferred, with softmax noise (inverse temperature beta) and an independent lapse (epsilon) capturing response variability. This single task-invariant mechanism predicts both experimental signatures simultaneously: in Experiment 1, the top-cue-favored option carries more 1s and hence more 'defects', so P(choose the top-cue-favored option) DECREASES with its tally margin (negative slope); in Experiment 2, on margin-1 trials where TTB and the tally disagree the two options' signed values are close together (so noise frequently flips the subject onto the tally winner), whereas when they agree the value gap is large (so the subject reliably anti-follows) — yielding p_disagree > p_agree, a positive metric.

**Parameters:**
- beta_seed: [0, 1]
- tau: [0.15, 0.42]
- attention: [(0.90, 1.10)] * n_features
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Signed-Cue Weighted Integration (SCWI) -- corrected three-regime
    # population, iteration 7 (minimal-diff on the accepted iter-6 base).
    # Mechanism is UNCHANGED: subjective validity of cue j = a_j * v_j
    # (per-cue attention factor, fixed for the whole run), signed cue
    # weight w_j = -(v'_j + tau * v'_j^2) (a rating of 1 is a defect; the
    # anti-weight grows SUPRALINEARLY in the subjective instructed
    # validity), softmax over beta * values plus a symmetric lapse
    # epsilon. History is ignored (no trial-by-trial feedback).
    #
    # Population edits on top of the accepted iter-6 base (mechanism
    # untouched), per the iter-6 critic diagnosis that the remaining
    # gaps are pure mixture-calibration problems:
    #   * FRACTIONS 13/57/30 -> 16/50/34: the mid regime (57%) was the
    #     underperformer on Exp 5 and contributes almost nothing on the
    #     Exp-4 Probe-B conflict cells, so its mass is redistributed to
    #     BOTH tails -- the guesser tail lifts Exp 4 (guessers contribute
    #     0.5 on conflict cells vs mid's ~0.10) and the deterministic
    #     tail lifts Exp 5/6 (det ~0.9 vs mid's ~0.53).
    #   * MID BETA raised a small step, 1.6-2.2 -> 1.8-2.4 (NOT a swing
    #     back to iter-5's 1.8-2.8, which failed without real guessers;
    #     with 16% pinned at 0.5 the Exp-4 floor is higher).
    #   * WITHIN-REGIME SPREAD WIDENED for between-subject variance:
    #     deterministic beta 4.2-5.6 -> 3.8-6.0, mid epsilon 0.06-0.10
    #     -> 0.05-0.11, guesser epsilon 0.15-0.25 -> 0.12-0.28. These
    #     widen the per-subject hit-rate distribution without materially
    #     moving the population means.
    #   * Attention jitter stays exactly at the validated (0.90, 1.10);
    #     tau stays at [0.15, 0.42]; the beta~0 guesser implementation
    #     (the validated decoupling lever) is untouched.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"SCWI expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    def _to_vec(raw):
        if raw is None:
            return None
        if isinstance(raw, str):
            s = raw.strip().strip('[]()')
            toks = [t for t in s.replace(',', ' ').split() if t != '']
            if not toks:
                return None
            return np.array([float(t) for t in toks])
        arr = np.asarray(raw, dtype=float).ravel()
        return arr if arr.size > 0 else None

    val = _to_vec(parameters.get("validities"))
    if val is None or val.size != n_features:
        # Defensive fallback: uniform mid-range validities.
        val = np.full(n_features, 0.75)
    val = np.clip(val, 1e-3, 1.0)

    att = _to_vec(parameters.get("attention"))
    if att is None or att.size != n_features:
        # Defensive fallback: no attention distortion.
        att = np.ones(n_features)
    att = np.clip(att, 0.6, 1.4)

    # --- Population mixture over (beta, epsilon) from a single uniform
    # seed. Three regimes: TRUE guessers (beta ~ 0 -> uniform core on
    # every trial), a mid anti-validity regime, and a deterministic
    # anti-validity regime. Iter-7 fractions: 16% / 50% / 34%.
    try:
        u = float(parameters["beta_seed"])
    except (TypeError, ValueError):
        u = 0.5
    u = min(max(u, 0.0), 1.0)
    if u < 0.16:
        # True guessing subpopulation (~16%): beta ~ 0 makes the
        # softmax core essentially uniform on every trial, so these
        # subjects sit at ~0.5 on every diagnostic cell.
        t = u / 0.16
        beta = 0.02 + 0.08 * t         # 0.02 .. 0.10
        epsilon = 0.12 + 0.16 * t      # 0.12 .. 0.28
    elif u < 0.66:
        # Mid subpopulation (~50%): moderate beta (small step up from
        # the iter-6 1.6-2.2 to target the Exp-5 cells where mid
        # subjects sit too close to chance), low lapse.
        t = (u - 0.16) / 0.50
        beta = 1.8 + 0.6 * t           # 1.8 .. 2.4
        epsilon = 0.11 - 0.06 * t      # 0.11 .. 0.05
    else:
        # Deterministic subpopulation (~34%): high beta with widened
        # within-regime spread, minimal lapse. Fraction (not beta
        # ceiling) guards Exp 6.
        t = (u - 0.66) / 0.34
        beta = 3.8 + 2.2 * t           # 3.8 .. 6.0
        epsilon = 0.04 - 0.02 * t      # 0.04 .. 0.02

    tau = float(parameters["tau"])

    # Subjective validities (instructed validity x per-cue attention),
    # then anti-validity weights: the anti-weight of expert j is
    # v'_j + tau * v'_j^2, supralinear in the subjective validity, so
    # the endorsements of the most trusted experts are avoided most
    # strongly while low-validity experts are nearly ignored.
    v_eff = val * att
    w = -(v_eff + tau * np.square(v_eff))

    scores = np.array([float(np.dot(w, stim[0])), float(np.dot(w, stim[1]))])

    # Numerically stable softmax over the two option values.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Guard against float drift.
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    # Cell patterns are (top-cue count on H option, mid-cue count on L option,\n    # low-cue count on L option), where H = the option whose 1s sit exclusively\n    # on the v=1.00 cues and L = the option whose 1s sit exclusively on the\n    # v<=0.60 cues.\n    # TWIN-KILL cells (r = 1/v_l, the gamma=1 flip point for HEAI):\n    #   4x1.00 vs 8x0.50  (trials 3, 9, 13, 19)   and\n    #   3x1.00 vs 5x0.60  (trials 7, 10, 17, 20)  -> 32 rows/subject.\n    # HIGH-RATIO cells (r = 3.0 on ladder 1, r = 2.0 on ladder 2):\n    #   4x1.00 vs 12x0.50 (trials 5, 15)          and\n    #   3x1.00 vs 6x0.60  (trials 8, 18)          -> 16 rows/subject.\n    TWIN_PATTERNS = {(4.0, 0.0, 8.0), (3.0, 5.0, 0.0)}\n    HIGH_PATTERNS = {(4.0, 0.0, 12.0), (3.0, 6.0, 0.0)}\n\n    def _counts(ratings):\n        try:\n            arr = np.asarray(list(ratings), dtype=float).ravel()\n        except Exception:\n            return None\n        if arr.size != 22:\n            return None\n        return (float(arr[0:4].sum()), float(arr[4:10].sum()), float(arr[10:22].sum()))\n\n    def _is_H(c):\n        # H option: all its 1s are on the four v=1.00 cues.\n        return c[0] > 0 and c[1] == 0.0 and c[2] == 0.0\n\n    def _is_L(c):\n        # L option: no top-cue 1s, at least one 1 on the v<=0.60 cues.\n        return c[0] == 0.0 and (c[1] > 0 or c[2] > 0)\n\n    twin_hits = []\n    high_hits = []\n\n    responses = data[\"response\"].values\n    a_list = data[\"option_a_ratings\"].values\n    b_list = data[\"option_b_ratings\"].values\n\n    for resp, ra, rb in zip(responses, a_list, b_list):\n        ca = _counts(ra)\n        cb = _counts(rb)\n        if ca is None or cb is None:\n            continue\n        try:\n            r = int(float(resp))\n        except Exception:\n            continue\n        if _is_H(ca) and _is_L(cb):\n            h_cnt, l_cnt, h_is_a = ca, cb, True\n        elif _is_H(cb) and _is_L(ca):\n            h_cnt, l_cnt, h_is_a = cb, ca, False\n        else:\n            continue\n        pattern = (float(h_cnt[0]), float(l_cnt[1]), float(l_cnt[2]))\n        chose_h = (r == 0) if h_is_a else (r == 1)\n        val = 1.0 if chose_h else 0.0\n        if pattern in TWIN_PATTERNS:\n            twin_hits.append(val)\n        elif pattern in HIGH_PATTERNS:\n            high_hits.append(val)\n\n    if len(twin_hits) == 0 or len(high_hits) == 0:\n        return 0.0\n\n    p_twin = float(np.mean(twin_hits))\n    p_high = float(np.mean(high_hits))\n    # Signed psychometric rise between the twin-kill ratio region and the\n    # high-ratio region. SCWI (step function pinned in [2.14, 2.35] ladder 1\n    # and [1.76, 1.89] ladder 2) produces a LARGE rise (~0.55-0.60); HEAI\n    # (bimodal gamma, shallow mode straddling the gamma=1 flip) produces a\n    # SMALL rise (~0.20).\n    return p_high - p_twin",
  "rationale": "METRIC: the signed rise of P(choose the high-validity stack H) between the twin-kill ratio cells and the high-ratio cells: D = P(H | r in {2.0 on 0.50, 5/3 on 0.60}) minus... specifically D = P(H | high-r composite) - P(H | twin-kill composite), where the twin-kill composite is the 32 trials pitting 4x1.00 vs 8x0.50 (trials 3, 9, 13, 19) and 3x1.00 vs 5x0.60 (trials 7, 10, 17, 20), and the high-r composite is the 16 trials pitting 4x1.00 vs 12x0.50 (trials 5, 15) and 3x1.00 vs 6x0.60 (trials 8, 18). H is always the option whose 1s fall exclusively on the v=1.00 cues; trials are identified purely from the rating vectors (top/mid/low cue counts), so the metric is robust to column-format quirks and works identically on the pooled frame and on any single subject's 96-trial slice.\n\nWHY IT DISCRIMINATES: the two theories' psychometric curves over the endorsement ratio r CROSS TWICE with opposite signs, and D is exactly the signed distance between the two crossing regions, so it stacks both opposing gaps into one scalar. (1) On the twin kills (r = 1/v_l, HEAI's flip point at gamma = 1 exactly), every engaged SCWI subject prefers the LOW stack: its margin is -2*tau (ladder 1) and -1.2*tau (ladder 2), strictly negative for all tau in [0.15, 0.42] and robust to the attention jitter (worst-case attention still leaves the margin negative), so SCWI's twin-kill P(H) is ~0.20-0.30. HEAI's shallow mode straddles gamma = 1: the ~43% of shallow subjects with gamma < 1 take H with per-trial probability 0.6-0.95, lifting HEAI's twin-kill P(H) to ~0.37-0.40 \u2014 HEAI is ABOVE SCWI here. (2) On the high-r cells, every engaged SCWI subject prefers H (L1f margin 2 - tau in [1.58, 1.85]; L2c margin 0.6 - 0.84*tau in [0.25, 0.47]), giving P(H) ~0.78-0.85; HEAI's steep mode (21% of the population, gamma >= 1.65 > log2(3) = 1.585) stays on L at r = 3.0 and its shallow mode's upper shoulder fades at L2c (flip at gamma = 1.357), capping HEAI at ~0.55-0.63 \u2014 SCWI is ABOVE HEAI here. Net: D_SCWI ~ 0.55-0.60 vs D_HEAI ~ 0.18-0.25, a mean gap of ~0.35.\n\nWHY IT SURVIVES N=25: the per-subject statistic is a difference of two within-subject proportions (32 and 16 trials), which cancels each subject's overall response bias and side bias (all cells are mirrored). The residual between-subject spread is mostly genuine mixture heterogeneity, estimated at sd ~0.27 (HEAI: guessers ~0, steep ~0.05, shallow-gamma<1 ~0.18, shallow-gamma>1 ~0.55) and sd ~0.31 (SCWI: guessers ~0, mid ~0.53, deterministic ~0.90). Welch's t at N=25 per arm: t ~ 0.38 / sqrt((0.075 + 0.10)/25) ~ 4.5-5, p well below 0.001 \u2014 comfortably inside the alpha = 0.01 admission threshold, unlike pooled single-cell rates (twin-kill alone: gap only ~0.13; L1f alone: gap ~0.2 with a bimodal per-subject distribution, both of which fail at N=25). The difference metric concentrates the full double-crossing contrast into one number while keeping per-subject noise at binomial level."
}
```

## Usage

```json
{
  "input_tokens": 12205,
  "output_tokens": 18012,
  "reasoning_tokens": 15974,
  "provider": "Sail Research"
}
```
