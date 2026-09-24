# metric_exp01_attempt_00

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
**Validities (n_features=12):** [0.9, 0.9, 0.9, 0.9, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]

**Trial pairs (n=32):**
  trial 1: A=[1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 2: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
  trial 3: A=[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 4: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1]
  trial 5: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 6: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
  trial 7: A=[1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 8: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1]
  trial 9: A=[1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 10: A=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
  trial 11: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]  B=[1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 12: A=[1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]
  trial 13: A=[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 14: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 15: A=[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 16: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0]
  trial 17: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 18: A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1]
  trial 19: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 20: A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  trial 21: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 22: A=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1]
  trial 23: A=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 24: A=[1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 25: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 26: A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1]
  trial 27: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 28: A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 29: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 30: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 31: A=[1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  trial 32: A=[0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]

**Rationale:** ADVERSARIAL LOGIC. CNNAI (advocated) and SCWI (competing) are near-degenerate anti-validity additive integrators, and CNNAI's own documentation concedes this: its engaged eta is centered at 0.12 precisely so that 'the engaged bulk behaves SCWI-like in effective curvature' while sitting below the knife region eta~0.31-0.34. At the validity pair (0.90, 0.60), SCWI's weight ratio w(0.90)/w(0.60) = (0.9+0.81*tau)/(0.6+0.36*tau) is confined by tau in [0.15, 0.42] to [1.562, 1.651] (implied power exponent 1.10-1.24), while CNNAI's ratio is 1.5^(1+eta) with eta ~ N(0.12, 0.21) clipped [-0.25, 0.70]; the engaged bulk overlaps and no single-cell population sign kill exists. Experiment [10] already took the context-redraw wedge (bucket twinning and transfer). This design attacks the two flanks CNNAI's knife-avoidance recalibration left un-fortified and that no prior experiment instrumented. FLANK 1 -- THE DETERMINISTIC BLOC (primary, distributional kill). SCWI hard-codes a trimodal response mixture: 16% true guessers (beta 0.02-0.10), 50% mid (beta 1.8-2.4, eps 0.05-0.11), and 34% DETERMINISTIC (beta 3.8-6.0, eps 0.02-0.04). CNNAI hard-codes a homogeneous engaged beta: lognormal median 2.14, log-sd 0.20, hard-clipped to [0.9, 4.5] -- P(beta>3.2) ~ 2%, P(beta>4) ~ 0.1% -- with eps 0.03-0.09. Instrument: the M-cell, where one option carries a single extra 0.60-cue endorsement and the other 11 cues are tied (both options show 11-12 ones). The choice margin is exactly |w(0.60)|: 0.654-0.751 for SCWI (only 15% variation over the whole tau box) and 0.54-0.62 for CNNAI (eta-robust). Both theories prefer the defect-free option, so the cell measures CONFIDENCE, not direction. Per-trial p: SCWI deterministic regime 0.89-0.99; SCWI mid 0.71-0.86; CNNAI engaged 0.68-0.86 (essentially nobody above 0.87, since even beta=4.5 gives only ~0.90 and that beta has ~0.01% prior mass). Pre-registered statistic: the fraction of subjects choosing the defect-free option on >= 22 of 24 M-trials, gated on S-cell anti-validity consistency (>= 8/12; see screens below) to exclude count-heuristics. SCWI predicts 0.22-0.35 (0.34 x P(>=22/24 | p~0.95) plus a small mid leak; even the worst corner of its box -- tau=0.15, beta=3.8, eps=0.04 -- yields >= 0.20). CNNAI predicts 0.005-0.03 (its beta ceiling makes a sustained 0.917 rate essentially impossible; guesser binomial flukes add ~0.0001). The boxes are disjoint by ~7x. Crucially, the population MEANS are nearly identical (SCWI ~0.79, CNNAI ~0.73): a mean-level test -- the kind CNNAI's calibration was built to survive -- is underpowered here by construction; the dissociation lives in the upper tail and in the trimodal-vs-bimodal shape of the per-subject M-rate distribution (SCWI: modes near 0.50 / 0.78 / 0.95; CNNAI: 0.50 / ~0.76 with only a thin continuous shoulder). FLANK 2 -- THE SUBLINEAR TAIL (secondary; level-and-step kill on a count-ratio ladder). SCWI's map is everywhere steeper than linear; CNNAI places 28% of its engaged mass at eta < 0, i.e., SUBLINEAR anti-validity weighting (w(0.90)/w(0.60) < 1.5: high-validity defects weighted less than proportionally). The rung at exactly the linear knife -- 4x0.90 vs 6x0.60 endorsements (ratio 3/2) -- pins the two: every engaged SCWI subject prefers the low-validity stack (margin 0.16-0.45 across the entire tau box, never flipping), while CNNAI's eta<0 subjects (~24% of the population) prefer the HIGH-validity stack (margins up to 0.36). Population P(H-stack): SCWI 0.30-0.40 vs CNNAI 0.42-0.47. The adjacent rung at 5/3 (3x0.90 vs 5x0.60) sits just above SCWI's whole flip window [1.562, 1.651]: ALL engaged SCWI subjects flip between the two adjacent rungs (step +0.26 at the box center, range +0.15..+0.34), whereas CNNAI rises only +0.10..+0.18 (25% of engaged still prefer the L-stack at 5/3, i.e., eta>0.26; 14% at 7/4). Rungs 7/4 (4 vs 7) and 2/1 (4 vs 8) complete the ladder: SCWI ~0.72 / ~0.81 vs CNNAI ~0.65 / ~0.80. SINGLE-CONTEXT CONSTRUCTION (conservative by design). Every critical trial endorses all 12 experts (each cue is 1 for at least one option), so CNNAI's v_ref = 0.70 exactly and its bucket key is '0.7' on EVERY trial: one eta draw per subject for the whole experiment. CNNAI therefore runs in its most SCWI-like fixed configuration -- no bucket redraws, no v_ref rescaling differences -- so any observed dissociation is conservative and cannot be attributed to CNNAI's context stochasticity. This deliberately does NOT repeat [10]'s twinning/transfer logic; the wedges here (beta-mixture shape; sublinear curvature tail) are untouched by [0]-[10]. MARGIN RUNGS AND SCREENS. S-cells (one exclusive 0.90-endorsement vs one exclusive 0.60-endorsement, 10 tied cues; margin w90-w60 = 0.37-0.49 SCWI, 0.22-0.54 CNNAI): both theories prefer the low-validity-endorsed option (population ~0.72 vs ~0.67) -- and the 1-count is exactly tied (11-11), so Tallying sits at exactly 0.500 while TTB and pro-validity integrators pick the high-validity option: a triple screen that simultaneously verifies the anti-validity direction and gates the bloc statistic against 'fewer-1s' heuristics (which are at chance on tally-tied S-cells). L-cells (single 0.90 defect, margin 1.02-1.24) and XL-cells (two 0.60 defects, margin 1.31-1.50) are confident agreement anchors calibrating the softmax ceiling (both theories 0.85-0.93). ANCHOR (all-1 vs all-0): both theories predict P(choose all-1) ~ eps/2 (population ~0.10 including guessers); tallying and pro-validity integration predict ~1 -- an exclusion screen. TIE (identical vectors): exactly 0.500 for both -- side-bias calibration. Every critical pair is mirrored (mirroring leaves v_ref and the bucket key unchanged). ATTENTION ROBUSTNESS. M-cells rotate the defect cue across four different 0.60-experts, and knife cells rotate cue subsets, so SCWI's fixed per-cue attention (U[0.9, 1.1]) enters margins multiplicatively and averages out (class-mean sd ~0.03 over rotations); a deterministic-regime SCWI subject's M-cell p cannot drop below ~0.88. CNNAI's per-trial attention diffusion (sd 0.05) shifts margins by ~0.03 against margins of 0.54-0.75 -- negligible. The bloc kill requires only a sustained high rate, not a sign flip, so attention cannot sabotage it from either side. DESIGN. 12 experts: 4 at 0.90, 8 at 0.60. 32 unique pairs x K = 96//32 = 3 = 96 trials per subject: 24 M (bloc), 12 S (direction + screens), 6 L + 6 XL (agreement anchors), 12 K(3/2) + 12 K(5/3) + 6 K(7/4) + 6 K(2/1) (knife ladder), 6 ANCHOR, 6 TIE. Every trial requires scanning 9-11 tied cues to locate the discriminating endorsements, satisfying the min-RT viewing requirement. APPROXIMATE POPULATION PREDICTIONS (integrated over each theory's stated parameter boxes): M-cell mean 0.79 (SCWI) vs 0.73 (CNNAI); M-bloc fraction (>=22/24, S-gated) 0.22-0.35 vs 0.005-0.03; M-rate distribution trimodal vs bimodal; S-cell 0.72 vs 0.67; L 0.88 vs 0.85; XL 0.93 vs 0.90; K(3/2) P(H-stack) 0.30-0.40 vs 0.42-0.47; K(5/3) ~0.59 both, step from K(3/2) +0.26 vs +0.15; K(7/4) 0.66-0.80 (point 0.72) vs ~0.65; K(2/1) ~0.81 vs ~0.80; ANCHOR P(all-1) ~0.10 both; TIE 0.500 both. DECISION RULES. (i) SCWI is supported and CNNAI falsified if the S-gated M-bloc fraction is >= 0.18 with a visible near-ceiling mode (~0.9+) in the per-subject M-rate distribution -- CNNAI cannot exceed ~0.03-0.06 for ANY admissible parameters (beta capped at 4.5 with 98% of engaged subjects below 3.2) -- together with K(3/2) <= 0.40 and a ladder step K(5/3)-K(3/2) >= 0.20. (ii) CNNAI is supported and SCWI falsified if the M-bloc fraction is <= 0.07 with a unimodal engaged M-rate distribution and M-mean <= 0.75, K(3/2) >= 0.42 (above SCWI's whole-box ceiling), and step <= 0.16. (iii) BOTH are falsified if the all-1 option is chosen well above 0.15 on ANCHOR (tallying/pro-validity), TIE deviates from 0.5 (side bias), the high-validity-endorsed option wins the S-cells (TTB/pro-validity), or the M-rate distribution fits neither shape (e.g., mass only at 0.5 and 1.0). TERTIARY (supportive only, weak): the between-subject correlation of K(3/2) H-rate with M-rate among non-guessers is NEGATIVE under SCWI (a single tau raises both M-confidence and the L-stack margin at the linear knife) and POSITIVE under CNNAI (a single low eta both enlarges w(0.60) and flips the linear knife toward the H-stack); both mechanisms are attenuated by beta/binomial noise, so this is corroborative, not decisive.

**Computed schedule:** 32 unique pairs × 3 reps = 96 trials per subject.



## ADVOCATED THEORY
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    df = data.reset_index(drop=True)\n\n    # --- Design constants for this experiment --------------------------\n    # 12 experts: validities [0.9 x 4, 0.6 x 8].\n    VALID = np.array([0.9] * 4 + [0.6] * 8)\n\n    # Reference anti-validity defect magnitudes: a rating of 1 is treated\n    # as a defect with weight -d_j. The reference ratio d(0.9)/d(0.6)\n    # ~ 1.58 sits between the two candidate anti-validity maps, so the\n    # per-trial signed margins fed to the fit are theory-neutral.\n    D_HI, D_LO = 1.0, 0.633\n\n    # Grid for the per-subject sensitivity (softmax inverse temperature)\n    # maximum-likelihood fit, and the soft-threshold ramp applied to it.\n    GRID = np.linspace(0.0, 10.0, 501)\n    RAMP_LO, RAMP_HI = 3.0, 4.2\n\n    def _vec(cell):\n        try:\n            return np.asarray(cell, dtype=float).ravel()\n        except Exception:\n            return None\n\n    def _subject_score(sub):\n        ms = []\n        ys = []\n        for a_cell, b_cell, resp in zip(\n            sub[\"option_a_ratings\"], sub[\"option_b_ratings\"], sub[\"response\"]\n        ):\n            A = _vec(a_cell)\n            B = _vec(b_cell)\n            if A is None or B is None or A.size == 0 or A.size != B.size:\n                continue\n            if A.size == VALID.size:\n                d = np.where(VALID > 0.75, D_HI, D_LO)\n            else:\n                d = np.full(A.size, 0.8)\n            w = -d\n            va = float(np.dot(w, A))\n            vb = float(np.dot(w, B))\n            m = abs(vb - va)\n            if m < 1e-9 or m > 2.0:\n                # identical pairs (ties) and the all-ones-vs-all-zeros\n                # anchor screens carry no usable sensitivity information\n                # (both theories put them near the lapse rate / 0.5),\n                # and the huge anchor margin would corrupt the fit.\n                continue\n            b_pred = vb > va\n            r = float(resp)\n            y = 1.0 if ((r > 0.5) == b_pred) else 0.0\n            ms.append(m)\n            ys.append(y)\n        if len(ms) < 10:\n            return 0.0\n        m = np.asarray(ms, dtype=float)\n        y = np.asarray(ys, dtype=float)\n        # One-parameter logistic MLE: P(choose reference-predicted\n        # option) = sigmoid(beta * margin), over a dense grid (robust,\n        # no convergence issues; perfectly consistent subjects cap at 10).\n        z = np.outer(m, GRID)\n        ll = -(\n            y * np.logaddexp(0.0, -z) + (1.0 - y) * np.logaddexp(0.0, z)\n        ).sum(axis=0)\n        bhat = float(GRID[int(np.argmax(ll))])\n        g = (bhat - RAMP_LO) / (RAMP_HI - RAMP_LO)\n        return float(min(max(g, 0.0), 1.0))\n\n    subjects = pd.unique(df[\"subject_id\"])\n    if len(subjects) == 0:\n        return 0.0\n    vals = []\n    for sid in subjects:\n        sub = df[df[\"subject_id\"] == sid]\n        vals.append(_subject_score(sub))\n    return float(np.mean(vals))",
  "rationale": "CORE IDEA (Flank 1 of the design, implemented with an efficient estimator). The two theories nearly agree on every population MEAN (M-cell 0.79 vs 0.73, S 0.72 vs 0.67, etc.), so mean-level statistics are underpowered by construction. The one structural difference with a large effect size is SCWI's hard-coded DETERMINISTIC subpopulation: 34% of subjects with softmax beta uniformly in [3.8, 6.0] and lapse 0.02-0.04. CNNAI's engaged beta is a homogeneous lognormal with median ~2.14 (log-sd 0.20), hard-clipped to [0.9, 4.5], with 98% of engaged subjects below 3.2. ESTIMATOR. For each subject I fit a one-parameter logistic sensitivity beta_hat by maximum likelihood over all 84 usable trials (ties and the all-ones/all-zeros anchors excluded: ties carry no signal, anchors sit at the lapse rate for both theories and their ~9-unit margin would corrupt the fit). The per-trial signed margins come from a FIXED reference anti-validity map (d90=1.0, d60=0.633, ratio 1.58, i.e. the midpoint between SCWI's tau-box ratio 1.56-1.65 and CNNAI's median ratio 1.575), so the reference is theory-neutral. Fitting beta over all margin levels (M=0.63, S=0.37, L=1.0, XL=1.27, knife rungs 0.17-1.06) is much more efficient than a raw M-cell hit rate: it borrows strength across the whole saturation curve (Fisher information ~5 -> sd(beta_hat) ~0.45 for typical subjects). The metric is then the population mean of a soft indicator g(beta_hat) = clip((beta_hat-3.0)/1.2, 0, 1) of 'near-deterministic' sensitivity. PREDICTED VALUES. SCWI: deterministic-regime subjects (34%) have beta_hat ~ N(5.2, 1.0) -> g ~ 0.9; mid-regime (50%) have beta_hat ~ N(2.3, 0.45) -> g ~ 0.03; guessers (16%) -> 0. Pooled ~ 0.32, between-subject variance ~ 0.18 (bimodality of the mixture). CNNAI: engaged subjects (87%) have beta_hat ~ N(1.95, 0.46) -> g ~ 0.01 (the reference scale slightly deflates their fitted sensitivity, and eta-spread misfit on the knife cells pulls it further down); guessers (13%) -> 0. Pooled ~ 0.01, between-subject variance ~ 0.005. Welch t at N=25: t ~ 3.5-3.7, p ~ 0.001 << 0.01. Robustness margins: even if CNNAI's engaged beta_hat tail above 3.0 were 3x my estimate AND SCWI's deterministic g averaged only 0.7, t stays ~ 3.0 (p ~ 0.006). The statistic is a genuine distributional signature (presence of a near-ceiling response regime), not a trivial consequence of CNNAI's assumptions, and it works identically on pooled data (mean of g) and single-subject slices (that subject's g)."
}
```

## Usage

```json
{
  "input_tokens": 11844,
  "output_tokens": 84396,
  "reasoning_tokens": 82709,
  "provider": "Sail Research"
}
```
