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
**Validities (n_features=26):** [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]

**Trial pairs (n=24):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 13: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 14: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 15: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 16: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 17: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 18: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 19: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 20: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 21: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 22: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  trial 23: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 24: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

**Rationale:** ADVERSARIAL LOGIC. CAVI-K (advocated) and CNNAI (competing) are near-degenerate anti-validity additive integrators: a rating of 1 is a defect, tied cues cancel exactly, both use the SAME bucketed-mean-endorsed-validity context key with a per-bucket eta redraw, both have softmax + symmetric lapse + a true-guesser subpopulation, and their effective validity exponents overlap almost everywhere (CAVI-K's w(v)=v+tau*v^2 implies exponent ~1.1-1.3; CNNAI's 1+eta is centered at 1.12). Experiments [10] and [11] already took the context-redraw and beta-mixture wedges. The ONE hard structural difference no prior design has touched is CAVI-K's TWO-REGIME COMPRESSIVE LOAD MAP: defect loads are identity below a knee at 4.45 and power-compressed above it with kappa in [0.55,0.68] (engaged) and [0.78,0.85] (deterministic tail) -- ALWAYS strictly below 1. CNNAI is exactly linear in the ratings. Because compress() is strictly increasing it can never flip a preference; it only shrinks margins. So the dissociation is a CONFIDENCE-vs-LOAD profile, and it can be engineered to be parameter-free on both sides.

THE INSTRUMENT: THE TIED-FILLER LOAD LADDER AT SINGLE-VALIDITY ACTIVE SETS. On a trial where one option carries k extra 1s (defects) on cues of validity v and both options share f tied 1s on other cues of the SAME validity v: (i) every active cue has v_j = v_ref, so (v_j/v_ref)^eta = 1 for EVERY eta in BOTH theories -- the entire context/elasticity machinery (bucket redraws, stable components, pinned tail eta) is exactly inert; (ii) tied cues cancel exactly in CNNAI's score difference, and the bucket key is identical at every rung, so CNNAI's per-trial choice probability is provably IDENTICAL at every filler count (up to mean-zero per-trial attention noise on the discriminating cues only) -- a horizontal line with zero free parameters; (iii) in CAVI-K the tied cues add weight f*w to BOTH options' raw defect loads, and once the loads cross 4.45 the power compression shrinks the margin like s^(kappa-1): as f grows, engaged confidence falls from sigmoid(beta*k*w) toward 0.5, for EVERY admissible (tau, kappa, beta, epsilon, eta) and every non-guesser subject. The direction of choice never changes; only its confidence. CNNAI is pinned flat; CAVI-K is pinned decreasing.

VALIDITY ENGINEERING. 26 experts: 9 at 0.90 (family H), 17 at 0.60 (family L). Per-cue defect weight w_H = 0.9+0.81*tau in [1.18, 1.43] over the whole tau box (engaged [0.35,0.65], deterministic [0.55,0.65]); w_L = 0.6+0.36*tau in [0.726, 0.834]. PARAMETER-FREE FLAT ZONE: family H rungs f<=2 give max load 3*w_H = 4.28 < 4.45 and family L rungs f<=4 give max load 5*w_L = 4.17 < 4.45 for EVERY admissible tau (even a 2-SD adverse class-mean attention draw cannot cross the knee), so CAVI-K is exactly flat there too. PARAMETER-FREE COMPRESSED ZONE: H f=8 gives min load 8*1.18 = 9.47 and L f=16 gives min load 16*0.726 = 11.6, both far above 4.45 for the entire box. The knee is crossed between f=2 and f=3 in family H and between f=4 and f=6 in family L.

HEADLINE MINIMAL PAIR (count-matched, load-mismatched): H-f4 vs L-f4 have the IDENTICAL count structure -- option A shows 5 ones, option B shows 4, one extra defect for A, same display complexity -- but H-f4's load (5*w_H in [5.9, 7.15]) is above the knee while L-f4's (5*w_L in [3.63, 4.17]) is below it. CAVI-K predicts a confidence DROP in H-f4 but ZERO change in L-f4; CNNAI predicts zero change in both; any raw-count, total-1s, or search-difficulty heuristic treats the two cells identically and is thereby excluded. Symmetrically, going from the bottom rung to f=4 adds MORE ones in family L (1->5 per option) than in family H (3->5), so a difficulty account predicts a LARGER drop in L -- the exact opposite of CAVI-K's prediction.

PREDICTED POPULATION RATES, P(choose the defect-free option), midpoint parameters (engaged tau=.5, kappa=.615, beta=2.65, eps=.06; deterministic tau=.6, kappa=.815, beta=5.25, eps=.03; mixture 14/70/16; CNNAI engaged beta=2.14, eps=.06, mixture 13/87): CAVI-K family H: f=0: 0.87, f=2: 0.87 (flat -- sub-knee), f=4: 0.81, f=8: 0.78. Family L: f=0: 0.82, f=4: 0.82 (flat -- sub-knee), f=8: 0.73, f=16: 0.70. k=2 probe (two defects, family L): f=0: 0.90, f=16: 0.80. CNNAI: family H flat at 0.80, family L flat at 0.72, k=2 flat at 0.85 -- at EVERY rung, for every parameter. In LOG-ODDS the dissociation is large: CAVI-K's engaged per-subject logit falls by ~1.2 (family L: 2.65*0.78 -> 2.65*0.32) to ~2.0 (family H: 3.46 -> 1.50) across the ladder; CNNAI's falls by exactly 0.00. Per-subject drop distribution: CAVI-K predicts a negative slope for EVERY non-guesser (large for the 70% engaged, small for the 16% deterministic tail whose kappa=0.78-0.85 compresses weakly -- a bimodal drop-magnitude distribution plus a zero-slope guesser mass at 0.5); CNNAI predicts zero slope for every subject. KNEE LOCALIZATION: CAVI-K's drop onset sits at ~4 total ones on the defect option in family H and ~6-7 in family L -- the same WEIGHTED load (~4.45 in (v+tau*v^2) units) at different raw counts, a quantitative signature no count-based heuristic can reproduce. MARGIN SCALING: the L2 rungs (k=2 at f=0 and f=15 fillers) show the logit drop scales with the raw margin (compressed margin ~= kappa*(s/4.45)^(kappa-1)*k*w), replicating the drop at a second confidence level; CNNAI stays flat there too.

CONTROLS AND THIRD PARTIES. TIE trials (identical vectors, one all-1 and one mixed) pin side/response bias at exactly 0.500 for both theories. ANCHOR (all-1 vs all-0, mirrored) calibrates the lapse and screens tallying and pro-validity integration (both target theories predict P(choose all-1) ~ 0.09 including the guesser mass at 0.5; tallying/pro-validity predicts ~1). The sub-knee rungs are an internal artifact screen: any drop appearing at H f<=2 or L f<=4 falsifies the compression interpretation (and both theories), since those loads are below the knee for the entire parameter box -- this catches general dilution/complexity effects. Mirroring every critical pair absorbs side bias and leaves v_ref and the bucket key unchanged. All 26 expert rows are displayed on every trial, so display complexity is constant across rungs; the discriminating 1 is embedded among many fillers at high rungs, so every trial genuinely requires comparing the two rating vectors (min-RT requirement). Attention robustness: per-cue attention is mean-1 multiplicative; flipping a sub-knee rung across the knee requires a 4-7% class-mean distortion (2-4 SD), and the population-level slope predictions are immune because attention is symmetric. No feedback keeps the instructed validities anchored.

DESIGN. 26 experts (9 at 0.90, 17 at 0.60); 24 unique pairs x K = 96//24 = 4 repetitions = 96 trials per subject (8 trials per rung per family, 32 ladder trials per family, 8 k=2 trials, 8 tie, 8 anchor).

DECISION RULES. (i) CAVI-K is supported, and CNNAI falsified, if P(defect-free option) shows a reliable negative slope over filler load within both families (population-level and/or a majority of non-guesser subjects), with the sub-knee rungs flat (H f<=2, L f<=4), the drop onset located between the sub-knee and above-knee rungs at load-matched positions across families (~4 ones in H, ~6-7 in L), the H-f4-vs-L-f4 count-matched contrast showing a drop only in H, and the k=2 rungs replicating the drop at higher base confidence. CNNAI cannot produce ANY of these slopes for any admissible parameters: tied cues cancel exactly and its context machinery is inert by construction. (ii) CNNAI is supported, and CAVI-K falsified, if confidence is flat at every rung in both families (slope indistinguishable from 0 beyond binomial noise): CAVI-K's compression with kappa < 1 in every regime is hard-wired, and the top-rung loads (9.5-14.2) exceed the knee 4.45 across its entire parameter box, so it cannot stay flat. (iii) BOTH are falsified if the all-1 option is chosen well above ~0.15 on ANCHOR (tallying/pro-validity), TIE deviates from 0.5 (side bias), the drop is monotone in RAW filler count or total ones rather than weighted load (count/difficulty heuristic -- exposed by the H-f4/L-f4 count-matched contrast and the cross-family onset locations), or the drop appears already in the sub-knee rungs (dilution artifact).

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Kneed Compressive Anti-Validity Integration with a Moderate Deterministic Tail (CAVI-K), beta-homogenization refinement. Subjects integrate all binary expert ratings into a per-option subjective DEFECT LOAD, comprehending a rating of 1 as a defect (sign inversion). Per-cue defect magnitude is supralinear in the instructed validity, d_j = v_j + tau*v_j^2 (tau in [0.35, 0.65]), modulated by a context elasticity (v_j/v_ref)^eta keyed on the bucketed mean endorsed validity, with a zero-centered stable component (sd 0.10) plus a per-context redraw (sd 0.20), fixing latent cross-context consistency at r ~ 0.20. The defect load is compressed by a two-regime map with the knee at 4.45 — above every near-knife small stack (AL cells, 2-top-cue adherence cells, which keep full raw margins) and below the large diagnostic stacks (which grow sub-proportionally aversive under kappa in [0.55, 0.68]). Population: ~14% true guessers, ~70% engaged with a HOMOGENIZED lognormal beta (median 2.65, log-sd 0.29, clip [0.8, 6.0]) — the tightened spread removes both the saturating hot-engaged bloc (which was driving the Exp-9 cross-ladder correlation to 0.388 via beta-saturation, not latent consistency) and the soft low-beta tail (which was diluting anti-validity adherence on Exps 2/5/6/8/11) — and ~16% deterministic tail (beta 4.0-6.5, near-linear kappa, pinned eta). Choice is softmax(beta*(D_B - D_A)) with symmetric lapse and per-trial attention diffusion (sd 0.035).

**Parameters:**
- regime_seed: [0, 1]
- tau_seed: [0, 1]
- kappa_seed: [0, 1]
- beta_seed: [0, 1]
- epsilon_seed: [0, 1]
- eta_base_seed: [0, 1]
- subject_seed: [0, 1]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Kneed Compressive Anti-Validity Integration (CAVI-K), iter 5.
    # Minimal single-lever diff on the accepted iter-4 base, applying
    # the iter-4 critic diagnosis. Changes vs the accepted base:
    #   (A) ENGAGED BETA LOG-SD TRIMMED 0.35 -> 0.29 with the median
    #       nudged 2.5 -> 2.65 (clip unchanged [0.8, 6.0]). The Exp-9
    #       trajectory isolated the failure: at a CONSTANT eta split
    #       (0.10/0.20), raising the beta median 2.0 -> 2.5 swung the
    #       cross-ladder correlation 0.067 -> 0.388, i.e. the correlation
    #       is driven by the saturating HOT END of the engaged beta
    #       distribution, not by latent context consistency. Trimming the
    #       log-sd removes that bloc (pulling Exp 9 back toward its
    #       [0.15, 0.30] window) while ALSO removing the soft low-beta
    #       tail that was diluting anti-validity adherence on
    #       Exps 2/5/6/8/11 — the shared undershoot of all four prior
    #       iterations. The small median nudge (2.5 -> 2.65, well short
    #       of the critic's 3.0+ that the AL-sigmoid arithmetic rules
    #       out) targets Exp 11's remaining shortfall (2.24 vs the
    #       [2.4, 3.2] window).
    #   (B) ENGAGED KAPPA UPPER BOUND 0.70 -> 0.68 (range now
    #       [0.55, 0.68]): a single small pre-emptive guardrail step.
    #       Removing the low-beta tail raises mean adherence broadly,
    #       which will push Exp 10's rise (0.350 at iter 4, window
    #       [0.22, 0.36]) toward its ceiling; slightly stronger
    #       compression on the large diagnostic stacks holds it inside
    #       the window. This is deliberately NOT the iter-3 confounded
    #       trim (which fought a simultaneous knee raise) — the knee
    #       stays at 4.45 and only the upper bound moves by 0.02.
    #   Everything else is held verbatim from the accepted iter-4 base:
    #   knee 4.45 (above the max AL stack 4.28 for every tau), tau
    #   [0.35, 0.65], the 0.10/0.20 eta split with pinned tail eta,
    #   attention diffusion 0.035, the 14/70/16 mixture, and all lapse
    #   ranges — each either in-window or explicitly validated at
    #   iter 4.
    import hashlib
    import math
    import numpy as np
    from scipy.special import ndtri

    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"CAVI-K expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]
    ra, rb = stim[0], stim[1]

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
    ts = _seed("tau_seed")
    ks = _seed("kappa_seed")
    bs = min(max(_seed("beta_seed"), 1e-6), 1.0 - 1e-6)
    es = _seed("epsilon_seed")
    ebs = min(max(_seed("eta_base_seed"), 1e-6), 1.0 - 1e-6)
    ss = _seed("subject_seed")

    # --- Population mixture ---
    if u < 0.14:
        # True guessers (~14%).
        t = u / 0.14
        beta = 0.02 + 0.08 * t                 # 0.02 .. 0.10
        epsilon = 0.12 + 0.13 * es              # 0.12 .. 0.25
        tau = 0.50
        kappa = 0.75
        eta_base = 0.00
    elif u < 0.84:
        # Engaged bulk (~70%): HOMOGENIZED beta (median 2.65, log-sd
        # 0.29), kappa acting only above the knee, zero-centered stable
        # elasticity. The tightened beta spread removes both the
        # saturating hot bloc (Exp 9) and the soft low-beta tail
        # (adherence dilution on Exps 2/5/6/8/11).
        tau = 0.35 + 0.30 * ts                 # 0.35 .. 0.65
        kappa = 0.55 + 0.13 * ks                # 0.55 .. 0.68
        beta = math.exp(math.log(2.65) + 0.29 * float(ndtri(bs)))
        beta = min(max(beta, 0.8), 6.0)
        epsilon = 0.03 + 0.06 * es             # 0.03 .. 0.09
        eta_base = 0.10 * float(ndtri(ebs))
        eta_base = min(max(eta_base, -0.15), 0.15)
    else:
        # Deterministic tail (~16%): hot, low-lapse, strongly inverted,
        # least compressive, PINNED stable elasticity (no ndtri draw) so
        # the tail cannot act as a hidden cross-context coupling channel.
        tau = 0.55 + 0.10 * ts                 # 0.55 .. 0.65
        kappa = 0.78 + 0.07 * ks                # 0.78 .. 0.85
        beta = 4.0 + 2.5 * bs                   # 4.0 .. 6.5
        epsilon = 0.02 + 0.02 * es              # 0.02 .. 0.04
        eta_base = 0.05                         # pinned

    # --- Context: bucketed mean endorsed validity ---
    active = np.flatnonzero((ra > 0.5) | (rb > 0.5))
    if active.size == 0:
        return np.full(2, 0.5)

    v_ref = float(np.mean(val[active]))
    bucket = float(np.round(v_ref, 1))
    key = f"{bucket:.1f}"
    h = int(hashlib.md5(f"{ss:.9f}|{key}".encode("utf-8")).hexdigest()[:14], 16)
    uc = ((h % 999983) + 0.5) / 999984.0
    uc = min(max(uc, 1e-6), 1.0 - 1e-6)

    # Per-context redraw, zero-centered; stable/redraw sd ratio 0.10/0.20
    # fixes latent cross-context consistency at r ~ 0.20 by construction.
    eta_ctx = eta_base + 0.20 * float(ndtri(uc))
    eta_ctx = min(max(eta_ctx, -0.30), 0.50)

    # --- Defect weights: supralinear anti-validity, context-normalized ---
    d = val + tau * np.square(val)
    w = d * np.power(val / v_ref, eta_ctx)

    # Per-trial multiplicative attention diffusion (sd 0.035, the
    # iter-3-proven level that lets Exp-12 fitted bhat reach the ramp).
    att = np.exp(0.035 * np.random.randn(n_features))
    w = w * att

    # Raw defect loads.
    s_a = float(np.dot(w, ra))
    s_b = float(np.dot(w, rb))

    # TWO-REGIME (offset) compression: identity below the knee, power
    # compression above. Knee 4.45 > max AL stack 4.28 (at tau = 0.65),
    # so near-knife small stacks keep full raw margins for every tau;
    # the large Exp-10/12 stacks (4.7-11.5) are compressed.
    S_KNEE = 4.45

    def _compress(s):
        if s <= S_KNEE:
            return s
        return S_KNEE * ((s / S_KNEE) ** kappa)

    d_a = _compress(s_a)
    d_b = _compress(s_b)

    scores = np.array([-d_a, -d_b])

    # Numerically stable softmax over the two option values.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    return (1.0 - epsilon) * p_core + epsilon * 0.5 * np.ones(2)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Guard against float drift.
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Context-Normalized Noisy Anti-Validity Integration (CNNAI), knife-avoidance recalibration. Subjects integrate all binary expert ratings additively into a per-option value, but a rating of 1 is comprehended as a DEFECT (sign inversion). The anti-weight of expert j is context-normalized: w_j = -v_j * (v_j / v_ref)^eta, where v_ref is the mean instructed validity of the cues endorsed on the current trial and eta is a per-context elasticity drawn at the level of coarse validity CONTEXT BUCKETS (the mean endorsed validity rounded to 0.1), so all trials sharing a bucket inherit one shared per-subject draw. The elasticity decomposes into a small stable per-subject component (eta_base ~ N(0.02, 0.09)) and a larger per-bucket redraw (~N(0.10, 0.19)), total hard-clipped to [-0.25, 0.70]; the variance ratio fixes latent cross-context consistency at r ~ 0.18 by construction. The engaged eta distribution (mean ~0.12, sd ~0.21) is deliberately centered well BELOW the kill-design knife region (eta ~ 0.31-0.34 where high-validity stacks flip to low-validity stacks), so the engaged bulk behaves SCWI-like in effective curvature while margins on the below-knife side stay large. A modest per-trial multiplicative attention diffusion (sd 0.05) supplies within-context stochasticity, choice is softmax(beta * values) with nearly homogeneous engaged beta (lognormal median ~2.14, log-sd 0.20) plus a narrow symmetric lapse (0.03-0.09), and a 13% true-guesser subpopulation is retained, uncorrelated with eta.

**Parameters:**
- subject_seed: [0, 1]
- regime_seed: [0, 1]
- eta_base_seed: [0, 1]
- beta_seed: [0, 1]
- epsilon_seed: [0, 1]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Context-Normalized Noisy Anti-Validity Integration (CNNAI),
    # iter 6 -- knife-avoidance recalibration of the accepted iter-5 base
    # per the iter-5 critic diagnosis. Mechanism family UNCHANGED:
    # additive anti-validity integration, context-referenced elasticity
    # w_j = -v_j * (v_j / v_ref)^eta with v_ref = mean validity of the
    # endorsed cues, per-trial attention diffusion, softmax + symmetric
    # lapse, guesser mixture. Changes vs the accepted iter-5 base:
    #   (A) ETA CENTER LOWERED 0.23 -> 0.12 (eta_base ~ N(0.02, 0.09) plus
    #       per-context redraw ~ N(0.10, 0.19), total clipped [-0.25, 0.70]).
    #       The kill composites (Exps 5/6/8) are near-knife statistics:
    #       their flip knives sit at eta ~ 0.30-0.34, and on the
    #       below-knife side the score margins GROW as eta decreases
    #       (e.g. Exp 5's 2x1.00-vs-5x0.50 cell: margin 0.14 at eta 0.23
    #       vs 0.33 at eta 0.10). Centering at 0.12 both moves engaged
    #       mass below the knives (P(eta < 0.33) ~ 0.83) and enlarges
    #       the margins, lifting the undershooting composites.
    #   (B) ATTENTION DIFFUSION CUT 0.10 -> 0.05: at 0.10 the per-trial
    #       margin noise (sd ~ 0.25 on a 12-cue cell) was comparable to
    #       the near-knife signals (~0.15-0.35), damping below-knife
    #       subjects to p ~ 0.55-0.6 -- the main softening channel on
    #       Exps 5/6/8. Per-trial noise averages out within a subject
    #       (iter-2 critic), so cutting it lifts means without touching
    #       between-subject variance.
    #   (C) CONTEXT KEY COARSENED to the mean endorsed validity rounded
    #       to 0.1 (a bucketed v_ref / ladder identifier). The critic's
    #       'rounded MAX endorsed validity' variant was NOT used: both
    #       Exp-9 ladders ({1.00,0.50} and {1.00,0.60} cells) have max
    #       validity 1.0, so a max-based key would map them to the SAME
    #       draw and re-couple them -- recreating the iter-1 r = 0.44
    #       failure the critic itself warns against. Bucketed mean keeps
    #       them distinct (0.6 vs 0.7) while merging within-experiment
    #       fragmentation (Exp 2: ~2-3 keys instead of ~6; Exp 5: 2 keys),
    #       so per-subject composites stop averaging over many
    #       independent eta draws.
    #   (D) EXP-9 COUPLING CHANNELS CLOSED: engaged epsilon narrowed
    #       0.02-0.17 -> 0.03-0.09 (stable lapse was the dominant
    #       residual cross-ladder trait), eta_base sd cut 0.13 -> 0.09
    #       (latent r = 0.0081/(0.0081+0.0361) ~ 0.18), beta median
    #       trimmed 2.34 -> 2.14 with log-sd 0.25 -> 0.20 (guards Exps
    #       1/3/10 against the noise cut, and removes the last sizeable
    #       stable-trait correlation channel).
    #   (E) v_ref STAYS MEAN-REFERENCED (not max): the weight ratio
    #       w(v_h)/w(v_l) = (v_h/v_l)^(1+eta) is invariant to v_ref, and
    #       max-referencing multiplies every margin by v_max^(-eta) <= 1,
    #       which would DEFLATE exactly the near-knife margins the
    #       composites need. The critic's goal (steeper high-validity
    #       region) is achieved via (A)+(B) instead.
    import hashlib
    import math
    import numpy as np
    from scipy.special import ndtri

    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"CNNAI expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]
    ra, rb = stim[0], stim[1]

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
    es = _seed("eta_base_seed")
    bs = min(max(_seed("beta_seed"), 1e-6), 1.0 - 1e-6)
    ps = _seed("epsilon_seed")
    ss = _seed("subject_seed")

    # --- Population mixture (uncorrelated with eta). ---
    if u < 0.13:
        # Guesser subpopulation (~13%): beta ~ 0 -> uniform core on
        # every trial; elevated lapse. These subjects sit at ~0.5 on
        # every diagnostic cell and dilute population means/slopes.
        t = u / 0.13
        beta = 0.02 + 0.08 * t            # 0.02 .. 0.10
        epsilon = 0.12 + 0.13 * ps       # 0.12 .. 0.25
        eta_base = 0.10                  # irrelevant at beta ~ 0
    else:
        # Engaged subpopulation (~87%). Small stable elasticity
        # component (sd 0.09) paired with a larger per-context redraw
        # (sd 0.19): the variance ratio fixes latent cross-context
        # consistency at r ~ 0.18 by construction. The stable lapse
        # range is NARROW (0.03-0.09) so it can no longer act as a
        # hidden coupling channel across contexts, and beta is nearly
        # homogeneous (lognormal sd 0.20).
        zs = min(max(es, 1e-6), 1.0 - 1e-6)
        eta_base = 0.02 + 0.09 * float(ndtri(zs))
        eta_base = min(max(eta_base, -0.15), 0.20)

        beta = math.exp(0.76 + 0.20 * float(ndtri(bs)))  # median ~2.14
        beta = min(max(beta, 0.9), 4.5)
        epsilon = 0.03 + 0.06 * ps       # 0.03 .. 0.09

    # --- Context: bucketed mean endorsed validity. ---
    active = np.flatnonzero((ra > 0.5) | (rb > 0.5))
    if active.size == 0:
        # Nothing endorsed by either option: no evidence either way.
        return np.full(2, 0.5)

    # v_ref = mean validity of the cues actually displayed (endorsed)
    # this trial; the context key is v_ref rounded to one decimal, so
    # all cells of one validity ladder share a single per-subject eta
    # draw while different ladders (e.g. Exp 9's 0.50-ladder at bucket
    # 0.6 vs its 0.60-ladder at bucket 0.7) draw independently.
    v_ref = float(np.mean(val[active]))
    bucket = float(np.round(v_ref, 1))
    key = f"{bucket:.1f}"
    h = int(hashlib.md5(f"{ss:.9f}|{key}".encode("utf-8")).hexdigest()[:14], 16)
    uc = ((h % 999983) + 0.5) / 999984.0
    uc = min(max(uc, 1e-6), 1.0 - 1e-6)

    # Per-context redraw, centered low so the engaged bulk sits below
    # the kill knives (~0.31-0.34) with large below-knife margins:
    # engaged eta ~ N(0.12, 0.21), P(eta < 0.33) ~ 0.83.
    eta_ctx = eta_base + 0.10 + 0.19 * float(ndtri(uc))
    eta_ctx = min(max(eta_ctx, -0.25), 0.70)

    # Context-normalized anti-validity weights.
    w = -val * np.power(val / v_ref, eta_ctx)

    # Per-trial multiplicative attention diffusion on cue weights,
    # sd cut 0.10 -> 0.05: at 0.10 the margin noise on near-knife cells
    # was comparable to the signal and was the main softening channel
    # on the Exp 5/6/8 composites.
    att = np.exp(0.05 * np.random.randn(n_features))
    w = w * att

    scores = np.array([float(np.dot(w, ra)), float(np.dot(w, rb))])

    # Numerically stable softmax over the two option values.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    return (1.0 - epsilon) * p_core + epsilon * 0.5 * np.ones(2)
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    \"\"\"Median across subjects of the within-subject LOG-ODDS CONFIDENCE DROP\n    on the tied-filler load ladder.\n\n    For every ladder trial (one validity family active, options differ by\n    1-2 'defect' ratings, plus f tied ratings shared by both options) we\n    record whether the subject chose the option with FEWER 1s (the\n    'defect-free' option).  Trials are split by the defect option's total\n    one-count on its active family:\n      - LOW  (sub-knee):  family H (0.90 cues, first 9) with <=3 ones,\n                          family L (0.60 cues) with <=5 ones.  For the whole\n                          CAVI-K parameter box the weighted loads (count *\n                          (v + tau*v^2), tau in [0.35,0.65]) stay below the\n                          compression knee 4.45, so CAVI-K is exactly linear\n                          here too.\n      - HIGH (above-knee): family H with >=4 ones, family L with >=8 ones.\n                          Loads are 4.7-13.9, strictly above 4.45 for every\n                          admissible tau, so CAVI-K's power compression\n                          (kappa in [0.55,0.85], always < 1) shrinks the\n                          margin; CNNAI, being exactly linear in the\n                          ratings, is unaffected.\n    Ties (A==B) and anchor trials (both validity families active) are\n    excluded.  Per subject: p_low and p_high are the smoothed proportions\n    of defect-free choices in each pool; the statistic is\n        logit(p_low) - logit(p_high).\n    The metric returns the MEDIAN of this drop across subjects.\n\n    CNNAI: within a family every rung has identical active-validity set\n    (same bucket key, same eta draw), and tied cues cancel exactly in the\n    score difference, so p_low and p_high are draws from the SAME\n    distribution -> the drop is symmetric around exactly 0 for every\n    subject (mirrored pairs also cancel side bias).\n    CAVI-K: every non-guesser subject has kappa < 1 and top-rung loads far\n    above the knee, so the drop is strictly positive (engaged ~1 logit,\n    deterministic tail ~0.5, guessers ~0).\n    \"\"\"\n    import numpy as np\n    import pandas as pd\n\n    df = data.reset_index(drop=True)\n\n    # --- response as 0 (chose A) / 1 (chose B) ---\n    try:\n        resp = pd.to_numeric(df['response']).to_numpy(dtype=float)\n    except Exception:\n        resp = np.array(\n            [0.0 if str(r).strip().upper().startswith('A') else 1.0\n             for r in df['response']]\n        )\n\n    n_rows = len(df)\n    grp = np.zeros(n_rows, dtype=int)      # 0 = ignore, 1 = LOW, 2 = HIGH\n    clean = np.zeros(n_rows, dtype=float)  # 1 = chose the fewer-ones option\n\n    for i in range(n_rows):\n        a = np.asarray(df.at[i, 'option_a_ratings'], dtype=float).ravel()\n        b = np.asarray(df.at[i, 'option_b_ratings'], dtype=float).ravel()\n        if a.size == 0 or a.size != b.size:\n            continue\n        na = float(a.sum())\n        nb = float(b.sum())\n        if na == nb:\n            continue  # exact-tie trials (21, 22)\n        hs = min(9, a.size)  # first 9 experts at 0.90, remaining at 0.60\n        aH = float(a[:hs].sum()); aL = float(a[hs:].sum())\n        bH = float(b[:hs].sum()); bL = float(b[hs:].sum())\n        totH = aH + bH\n        totL = aL + bL\n        if (totH > 0 and totL > 0) or (totH == 0 and totL == 0):\n            continue  # anchor trials spanning both families (23, 24)\n        if totH > 0:\n            cnt = max(aH, bH)\n            is_high = cnt >= 4.0  # >=4 ones on 0.90-cues: load > knee 4.45 for all tau\n        else:\n            cnt = max(aL, bL)\n            is_high = cnt >= 8.0  # >=8 ones on 0.60-cues: load > knee; <=5 stays below\n        r = resp[i]\n        chose_clean = (r == 0.0) if (na < nb) else (r == 1.0)\n        grp[i] = 2 if is_high else 1\n        clean[i] = 1.0 if chose_clean else 0.0\n\n    def _logit(p):\n        p = min(max(p, 0.02), 0.98)\n        return float(np.log(p / (1.0 - p)))\n\n    subj = df['subject_id'].to_numpy()\n    drops = []\n    for s in pd.unique(subj):\n        m = (subj == s)\n        lo = m & (grp == 1)\n        hi = m & (grp == 2)\n        nl = int(lo.sum())\n        nh = int(hi.sum())\n        if nl < 5 or nh < 5:\n            continue\n        pl = (float(clean[lo].sum()) + 0.5) / (nl + 1.0)\n        ph = (float(clean[hi].sum()) + 0.5) / (nh + 1.0)\n        drops.append(_logit(pl) - _logit(ph))\n\n    if len(drops) == 0:\n        return 0.0\n    return float(np.median(drops))",
  "rationale": "The one hard structural difference between CAVI-K and CNNAI is CAVI-K's two-regime compressive load map (knee 4.45, kappa < 1 in every regime) versus CNNAI's exact linearity in the ratings. This metric reads that difference directly off the tied-filler load ladder. Per subject I pool the sub-knee rungs (family-H counts <=3, family-L counts <=5: weighted loads 4.17-4.29 < 4.45 over the ENTIRE CAVI-K tau box, so both theories are linear there) into a LOW pool and the above-knee rungs (H counts >=4, L counts >=8, plus the k=2 high filler cell: loads 4.7-13.9 > 4.45 for every admissible tau) into a HIGH pool \u2014 40 vs 40 trials per subject \u2014 and compute the log-odds of choosing the defect-free option in each pool, taking LOW minus HIGH. Ties and anchors are excluded; mirrored pairs cancel side bias. KEY IDENTIFICATION: within a family every rung activates only same-validity cues, so (v_j/v_ref)^eta = 1 for every eta \u2014 the entire context-elasticity machinery (bucket redraws, stable components) is inert, and for CNNAI the tied cues cancel exactly in the score difference, so its per-trial choice probability is IDENTICAL in the two pools. Hence for CNNAI p_low and p_high are i.i.d. draws from the same distribution: the per-subject logit drop is symmetric around exactly 0 for every subject and every parameter, and the median across subjects is ~0 (sd of the median at N=25 about 0.14). CAVI-K, by contrast, must show a positive drop for every non-guesser: once loads cross 4.45 the power compression shrinks the margin like s^(kappa-1), giving an engaged per-subject logit drop around 0.7-1.1 (deterministic tail ~0.5, guessers ~0), so the median across subjects lands in the engaged bulk around 0.8-1.0. The median (rather than the mean) is used because the diluting masses (14% guessers at 0, near-saturated deterministic tail whose drop is noisy in logit space) sit below the engaged bulk; the median is a robust 'typical subject' estimator that stays inside the engaged mode. Proportions are smoothed ((x+0.5)/(n+1)) and clipped to [0.02, 0.98] to keep the logit bounded for near-deterministic subjects; because the same transform is applied to both pools, it cancels exactly under CNNAI (identical pool distributions) and only trivially attenuates CAVI-K's drop. Expected separation: self median ~0.8-1.0 with between-subject variance ~0.5 (dominated by irreducible binomial noise over 80 ladder trials), adv ~0.0 with variance ~0.3, giving a Welch t around 4-5 at N=25, far below alpha = 0.01. The metric is also diagnostic in the right way: it cannot be produced by any raw-count or difficulty heuristic (the LOW pool contains MORE total ones in family L than the HIGH pool does in family H), and a spurious dilution artifact would have to appear in the sub-knee rungs, which are pooled as the baseline."
}
```

## Usage

```json
{
  "input_tokens": 13720,
  "output_tokens": 39509,
  "reasoning_tokens": 37369,
  "provider": "Sail Research"
}
```
