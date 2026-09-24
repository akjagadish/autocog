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
**Validities (n_features=32):** [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.64, 0.76]

**Trial pairs (n=24):**
  trial 1: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  trial 10: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0]
  trial 12: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 13: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1]
  trial 14: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 15: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
  trial 16: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 17: A=[1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 18: A=[0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 19: A=[1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 20: A=[0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 21: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 22: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 23: A=[1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 24: A=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]

**Rationale:** STRUCTURAL SETUP. HT-AEI-r6 and RSD-vgp share the identical anti-endorsement core: p(A) = 0.5 + A_eff*(sigmoid(beta*E) - 0.5), E = sum_j v_j^gamma*(b_j - a_j), RAW weights, and nearly identical m/epsilon boxes (A_eff support is NOT a usable lever). The theories are the SAME model with different population SUPPORTS: (i) GAMMA. RSD's two-component mixture is algebraically EXACTLY Uniform[0,2] (density 0.5 throughout, hard edge at 2.0). HT-AEI is a three-segment density: 0.40 on [0,1] (40% low bulk), 0.833 on [1.05,1.35] (25% mid segment), 3.5 on [2.0,2.10] (35% heavy tail), with EMPTY gaps (1.0,1.05) and (1.35,2.0) and a hard cliff at 2.10. (ii) BETA. RSD: log-uniform [1.8,6] independent of gamma. HT: every segment has beta >= 2.2 (low [2.2,4.0], mid [2.2,3.2], tail [2.2,2.8]) -- beta in [1.8,2.2) holds 16.7% of RSD subjects and 0% of HT subjects, and the tail's beta is capped at 2.8. Unlike the prior HT-vs-RSD design ([16]: a coarse 6-rung ladder at 0.585/1.0/1.322/1.585/1.807/2.0 that could see the plateau and the 2.0 edge but NOT the internal structure of any segment), this design is a FINE-GRAINED GAMMA-DENSITY PROFILOMETER: 8 rungs spaced 0.04-0.26 apart that recover the entire piecewise density -- the mid-segment bump, the empty gap, the tail's internal uniform density, and the exact location of the 2.10 cliff.

ARCHITECTURE (32 experts): six cues at validity 1.00 (power weight EXACTLY 1 for every gamma -- a gamma-invariant reference block of weight exactly 6) and 24 cues at 0.50, plus two tuning cues (0.64, 0.76). On every rung trial A is endorsed on the six unit cues (endorsement mass exactly 6 for all gamma) and B on a block of low cues, so E(gamma) = W_block(gamma) - 6 is strictly decreasing with a single engineered sign change at flip point g*: gamma < g* subjects prefer A (the count-sparser side, 6 endorsements); gamma > g* prefer B. The W=6 architecture makes the NATURAL all-0.50 rung grid hit exactly the points this dissociation needs: 13 cues -> g*=1.1155 (inside HT's mid segment); 15 -> 1.3219 (top of mid segment); 18 -> 1.585 and 21 -> 1.807 (inside HT's empty gap); 24 -> 2.000 (the tail's lower edge). Two TUNED rungs probe the tail's interior: 23x0.50 + 0.64 -> g*=2.040; 23x0.50 + 0.76 -> g*=2.082. A beyond-cliff control: 24x0.50 + 0.76 -> g*=2.141, just past HT's 2.10 tail edge (E(2.10) = +0.16 > 0, so every subject of BOTH theories has p >= 0.5 there).

DISSOCIATION 1 -- DENSITY PROFILOMETRY (headline). Because both gamma distributions are mixtures of uniforms, the population inversion rate at a rung equals the survival function S(g*) = P(gamma > g*), and adjacent rungs estimate LOCAL GAMMA DENSITY. Predicted inversion rates, HT vs RSD: R1 (1.1155): 0.545 / 0.442; R2 (1.3219): 0.373 / 0.339; R3 (1.585): 0.350 / 0.207; R4 (1.807): 0.350 / 0.097; R5 (2.000): 0.350 / 0.000; R6 (2.040): 0.210 / 0.000; R7 (2.082): 0.063 / 0.000; R8 (2.141): 0.000 / 0.000. Local densities (survival drop per unit flip-point): [1.1155,1.3219]: 0.833 (HT) vs 0.5 (RSD) -- the mid-segment 1.67x BUMP; [1.3219,1.585]: 0.087 vs 0.5; [1.585,1.807] and [1.807,2.0]: 0.000 vs 0.5 -- the EMPTY GAP (a 3-point FLAT plateau at exactly 0.35 across R3/R4/R5 is a signature RSD's uniform density cannot produce at any parameterization); [2.0,2.04] and [2.04,2.082]: 3.5 vs 0 -- the tail's internal density, ~7x RSD's maximum; [2.082,2.141]: cliff to 0 exactly at HT's 2.10 edge vs RSD's edge already at 2.0. Per-subject crossing points (the rung interval where a subject's psychometric crosses 0.5) give the full histogram: HT predicts 45.5% below 1.1155, 17.2% in (1.1155,1.3219], 2.3% in (1.3219,1.585], 0% in (1.585,2.0], and 35% in (2.0,2.141] (14.0% / 14.7% / 6.3% across the three tail sub-intervals -- a uniform 3.5-density fill); RSD predicts 55.8% / 10.3% / 13.2% / 20.8% / 0% -- a 15x contrast on the gap mass (2.3% vs 34.0%) and a hard 35%-vs-0% contrast above 2.0.

DISSOCIATION 2 -- HARD TAIL CELLS AND THE CLIFF. On R5-R8, E(gamma) >= 0 for EVERY gamma <= 2.0 (R5: E = 24*0.5^gamma - 6 >= 0 with equality only at the measure-zero point gamma = 2; R6-R8 flip above 2.0), and m >= 0.74 > 0.5 keeps p >= 0.5 whenever E >= 0, so NO RSD subject can truly prefer B on any of R5-R8 -- a hard floor at exactly 50%. HT-AEI REQUIRES 35% of subjects below 0.5 on R5 (tail depth: E in [-0.40, 0], p in [0.39, 0.50], mean ~0.44), 21% on R6 (p ~ 0.44-0.50), and 6.3% on R7. The WITHIN-TAIL LINEAR DECLINE 35% -> 21% -> 6.3% -> 0% across [2.0, 2.141] is a quantitative fingerprint no RSD parameterization can approximate (its gamma is hard-capped at 2.0), and rung R8 locates the cliff: inversions at R8 falsify BOTH theories (HT's tail ends at 2.10; RSD's at 2.0), making the design adversarial in both directions. CONDITIONAL COUPLING: under HT, P(p(R5) < 0.5 | p(R4) < 0.5) = 1.0 (every R4 inverter is a tail subject); under RSD it is 0 HARD (no RSD subject can be below 0.5 on R5). Under HT, P(invert R6 | invert R5) = 0.60; under RSD the conditioning event has probability 0. Within-subject monotonicity (inverting a higher rung implies inverting all lower rungs) holds under both and is a data-quality check.

DISSOCIATION 3 -- BETA SUPPORT VIA EXACT-E CELLS. B1/B2 use only unit cues with exactly canceling shared endorsements (shared endorsements on identical cue indices contribute w*(1-1) = 0), giving E = +1 and +2 EXACTLY for every subject at every gamma. The per-subject ratio r12 = (p_sparse,B2 - 0.5)/(p_sparse,B1 - 0.5) = (sigmoid(2*beta)-0.5)/(sigmoid(beta)-0.5) is exactly A_eff-free, lapse-free, and gamma-free. RSD support: r12 in [1.005, 1.322] (beta in [1.8,6]); HT-AEI: r12 <= 1.219 for ALL subjects (beta >= 2.2 in every segment). The region r12 in (1.219, 1.322] is RSD-EXCLUSIVE: 16.7% of RSD subjects, 0% of HT -- any subject credibly there categorically falsifies HT-AEI; a hard zero there falsifies RSD's beta floor. SEGMENT-COUPLED BETA (new readout on this design's deep rungs): among R4/R5 inverters, HT-AEI's tail caps beta at 2.8 (r12 >= 1.121 for all of them), whereas RSD draws beta independently, so ~63% of its high-gamma subjects have beta > 2.8 (r12 < 1.121) -- a population-level split of the inverter group that HT's coupled supports forbid. HT's beta ceiling of 4.0 (vs RSD's 6.0) moves r12 only between 1.005 and 1.037, below per-subject resolution -- honestly acknowledged as one-sided and not relied upon.

POOLED PREDICTIONS (approximate mid-box planning values, p(A-side), HT vs RSD): R1 ~0.47/0.47 (the low/mid mass and inverter depths nearly cancel -- R1 is a shape probe, not a pooled probe; honestly acknowledged); R2 ~0.57/0.64; R3 ~0.61/0.72; R4 ~0.63/0.76; R5 ~0.70/0.80; R6 ~0.72/0.79; R7 ~0.73/0.82; R8 ~0.78/0.83; B1 ~0.80/0.80; U ~0.85/0.85; ties 0.50/0.50. Pooled differences of 0.05-0.13 on R2-R7 are corroborating; the decisive evidence is SUBJECT-LEVEL population structure.

CONTROLS. U/Um (all-0 vs all-1, mirrored) saturate every subject (|E| = S(gamma) >= 12.5 at gamma = 2.1, beta >= 1.8), reading out A_eff = 2*(p_sparse - 0.5) to within 0.0002 -- required to normalize the beta probes and rung psychometrics. Uc is a CONFLICT-STYLE saturated twin with the OPPOSITE sparse side (A endorsed on 31 cues, B on a single 0.50 cue; |E| >= 11): p_sparse(Uc) must equal p_sparse(U), which detects any 'pick the empty bar' heuristic while its reversed side assignment cancels side bias against U. T1 (identical non-trivial vectors) verifies both theories' exact-50% prediction and catches key bias. Every rung and beta cell has a mirrored twin (A/B swapped) so side bias cancels; analysis pools by unit-cue side (rungs) or sparse side (B-cells, U, Uc). The ladder itself rules out pure counting (counting predicts flat ~100% preference for the 6-endorsement side on every rung) and conventional pro-endorsement polarity (flat ~0%). All informative trials carry nonempty, distinct bars on both sides, satisfying the locked-prompt comparison requirement.

POWER AND ANALYSIS. 24 unique pairs x K = 4 = 96 trials exactly; 8 observations per pooled mirrored cell per subject; 64 rung observations per subject. Requires N >= 100 subjects with hierarchical Bayesian fitting of per-subject (gamma, beta, A_eff) under each theory's population prior (RSD: gamma exactly U[0,2], beta log-U[1.8,6] independent; HT-AEI: the three-segment gamma mixture with segment-coupled beta boxes), plus simulation-based model recovery.

DECISIVE QUANTITATIVE SIGNATURES: (a) any subject credibly below 50% on R5, R6, or R7 categorically falsifies RSD-vgp (E >= 0 over its entire gamma box; hard floor at exactly 0.5), while HT-AEI requires 35%/21%/6.3% of subjects there; (b) inversion-rate survival curve over g* = 1.1155/1.3219/1.585/1.807/2.000/2.040/2.082/2.141: HT 0.545/0.373/0.350/0.350/0.350/0.210/0.063/0 (flat 0.35 plateau on [1.35,2.0], linear 3.5-density decline inside the tail, cliff at 2.10) vs RSD 0.442/0.339/0.207/0.097/0/0/0/0 (constant 0.5 density, hard edge at 2.0); (c) per-subject crossing-point histogram: gap mass (1.3219,2.0] at 2.3% (HT) vs 34.0% (RSD), tail mass (2.0,2.141] at 35% (HT) vs 0% hard (RSD), mid-bump density 0.833 vs 0.5 on [1.1155,1.3219]; (d) P(p(R5) < 0.5 | p(R4) < 0.5) = 1.0 (HT) vs 0 hard (RSD); (e) per-subject r12 in (1.219, 1.322]: ~16.7% (RSD) vs 0% hard (HT-AEI); (f) among R4/R5 inverters, r12 >= 1.121 for all (HT tail beta <= 2.8) vs ~63% below 1.121 (RSD independent supports); (g) within-subject monotone flip pattern (inverting R5 implies inverting R4..R1); (h) inversions at R8 (g* = 2.141) falsify BOTH theories -- HT's tail edge is at 2.10, RSD's at 2.0; (i) U = Uc in sparse-side rate (empty-bar control) and identical-vector ties at exactly 50% (key-bias control).

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Heavy-Tail Anti-Endorsement Integration, revision 6 (HT-AEI-r6). Subjects reading a product described by binary expert ratings treat an endorsement (rating = 1) as evidence AGAINST the option carrying it (a 'fewer red flags' reading), so the count-sparser option is perceived as higher quality. Within a trial, all cues are integrated additively with RAW (unnormalized) validity weights: the evidence for A over B is E = sum_j v_j^gamma (b_j - a_j), so the absolute endorsement-count margin sets the evidence scale independent of feature count, and identical rating vectors give exactly 50%. Choice follows p(A) = (1-eps)*[m*sigmoid(beta*E) + (1-m)*(1-sigmoid(beta*E))] + eps/2, with effective attenuation A_eff = (1-eps)(2m-1) concentrated near 0.69 — the 'steep normalized slope x moderate asymptote' family. The population is a three-segment mixture with SEGMENT-SPECIFIC PRECISION, all segments now at moderate-to-high precision (no soft-beta segment remains, since weakly identified slopes were shown to leak Exp 10/14 posteriors below threshold and to donate RFI-ward on Exp 9): (1) a 40% low bulk on gamma in [0, 1.0] with beta log-uniform on [2.2, 4.0] — the validated shared-core bulk, concentrated below the ~0.73/1.0 flip points of Exp 16's ladder, with the beta floor raised so every low-bulk subject's normalized slope is sharply identified; (2) a 25% mid segment on gamma in [1.05, 1.35] with beta log-uniform on [2.2, 3.2] — a band where RSD's uniform density (0.5) beats SGAI's (0.4) and EPAI's (0.375), sitting fully inside GSP's gamma box with GSP-compatible beta, so these subjects donate RSD-ward on Exp 15 while pulling Exp 11 and Exp 9 toward their observed values; (3) a 35% heavy tail on gamma in [2.0, 2.10] with beta log-uniform on [2.2, 2.8] — an extreme edge-hug of the RSD gamma boundary (average distance <= 0.05) at sharpened precision, so tail subjects' gamma posteriors concentrate just above 2.0 (feeding Exp 14), their slopes stay above the s>9.5 threshold (feeding Exp 10), and their near-deterministic past-flip-point behavior pulls Exp 16 down toward 0.4335. No per-trial gamma jitter. Falsifiable signatures: ~35% of subjects carry gamma within 0.10 of 2.0 with per-subject posteriors extending smoothly past 2.0; Exps 13/15 small and positive; Exp 11 in the 50-100 range; Exp 14 at or above 0.21; Exp 16 in [0.42, 0.45]; Exp 10 at or above 0.85.

**Parameters:**
- gamma_u: [0, 1]
- log_beta_low: [0.788, 1.386]
- log_beta_mid: [0.788, 1.163]
- log_beta_tail: [0.788, 1.030]
- epsilon: [0.03, 0.10]
- m: [0.75, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Heavy-Tail Anti-Endorsement Integration, revision 6 (HT-AEI-r6).
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are 0/1 cue values. History is ignored
    # (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"HT-AEI-r6 expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    # ---- Validity gating: three-segment mixture with SEGMENT-SPECIFIC
    # beta. All segments now carry moderate-to-high precision (no
    # soft-beta segment: weakly identified slopes were shown to leak
    # Exp 10/14 posteriors below threshold and donate RFI-ward on Exp 9).
    # gamma_u ~ U[0, 1] is mapped through:
    #   u in [0.00, 0.40): gamma ~ U[0.0, 1.0], beta log-U[2.2, 4.0]
    #            (40%; the validated low bulk, UNTOUCHED in gamma support.
    #            Beta floor raised 1.8 -> 2.2 per the pre-specified Exp 10
    #            contingency: sharpens per-subject slope identification
    #            without touching any other mechanism.)
    #   u in [0.40, 0.65): gamma ~ U[1.05, 1.35], beta log-U[2.2, 3.2]
    #            (25%; ENLARGED from 18%. On [1.05, 1.3] RSD's uniform
    #            density 0.5 beats SGAI's 0.4 and EPAI's 0.375, so these
    #            subjects donate RSD-ward on Exp 15; the band sits fully
    #            inside GSP's gamma box with GSP-compatible beta, pulling
    #            Exp 11 down toward the observed 52.6 and Exp 9 GSP-ward
    #            toward -1.76; it is above the 1.0 flip point
    #            (Exp-16-neutral); gamma near 1.2 matches the
    #            validity-proportional metrics of Exps 2 and 5.)
    #   u in [0.65, 1.00]: gamma ~ U[2.0, 2.10], beta log-U[2.2, 2.8]
    #            (35%; TRIMMED from 42% and RE-SHARPENED. Extreme edge-hug
    #            of the RSD gamma boundary (average distance <= 0.05), with
    #            beta raised from [1.9, 2.4] to [2.2, 2.8]: sharper gamma
    #            identification concentrates tail posteriors just above
    #            2.0 (Exp 14 recovers), keeps slopes above the s>9.5
    #            threshold (Exp 10), makes tail subjects near-deterministic
    #            past the flip points (Exp 16 pulled down toward 0.4335),
    #            and removes RFI-fittable soft slopes (Exp 9 back toward
    #            -1.76). The fraction trim cuts the residual Exp 13 edge
    #            cost ~17% while the mid segment's positive donations
    #            hold the total near the observed +6.)
    u = float(parameters["gamma_u"])
    if u < 0.40:
        gamma = 1.0 * (u / 0.40)
        log_b = float(parameters["log_beta_low"])
    elif u < 0.65:
        gamma = 1.05 + 0.30 * ((u - 0.40) / 0.25)
        log_b = float(parameters["log_beta_mid"])
    else:
        gamma = 2.0 + 0.10 * ((u - 0.65) / 0.35)
        log_b = float(parameters["log_beta_tail"])

    beta = float(np.exp(log_b))
    epsilon = float(parameters["epsilon"])
    m = float(parameters["m"])

    # RAW validity weighting: w_j = v_j^gamma, NO normalization.
    # This preserves the absolute endorsement-margin scale: a margin-2
    # conflict carries twice the evidence of a margin-1 conflict,
    # independent of n_features (the cross-experiment scale separation
    # that per-experiment normalization provably erases).
    w = np.power(val, gamma)

    a, b = stim[0], stim[1]
    # Anti-endorsement (sparse-preferring) evidence for A over B:
    # each endorsement carried by B counts FOR A, each endorsement
    # carried by A counts AGAINST A. E = 0 on identical rating vectors.
    E = float(np.sum(w * (b - a)))

    # Numerically stable logistic (both branches exponentiate a
    # non-positive argument, so no overflow is possible).
    x = beta * E
    if x >= 0.0:
        sig = 1.0 / (1.0 + np.exp(-x))
    else:
        ex = np.exp(x)
        sig = ex / (1.0 + ex)

    # Graded polarity: with per-trial probability m the subject responds
    # with the INVERTED (sparse-preferring) polarity, with probability
    # (1-m) with the conventional polarity. Continuous, no empty middle,
    # no mirror cluster. NO per-trial gamma jitter (delta = 0): the
    # stochastic attention gate is a confirmed dead lever.
    p_core = m * sig + (1.0 - m) * (1.0 - sig)

    # Independent lapse to uniform choice. Ties (E = 0) give exactly 50%.
    p_a = (1.0 - epsilon) * p_core + 0.5 * epsilon

    return np.array([p_a, 1.0 - p_a])
```

**`policy source code`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # Guard against float drift.
    return np.random.choice(len(probs), p=probs)
```


## COMPETING THEORY
**Description:** Raw-Evidence Sparse Dominance with Validity-Gated Precision (RSD-vgp). When choosing between two products described by binary expert ratings, subjects read an endorsement (rating = 1) as evidence AGAINST the option carrying it, so the count-sparser option is perceived as higher quality. Within a trial, ALL cues are integrated additively with RAW (unnormalized) validity weights: the evidence for A over B is E = sum_j v_j^gamma (b_j - a_j), so the absolute endorsement-count margin sets the evidence scale independent of feature count, and identical vectors give exactly 50%. The population is a continuum of graded polarity (no bimodal trait clusters): each subject carries a per-trial inverted-polarity probability m, a lapse epsilon, and — the new discriminative dimension — validity-gated precision: a per-subject validity exponent gamma dispersed over [0, 2] with a substantial steep-gamma subpopulation, and a steep log-uniformly dispersed raw inverse temperature beta over [1.8, 6.0], calibrated so that essentially every subject's effective normalized psychometric slope beta * sum(v^gamma) exceeds 9.5. The resulting subject family is 'steep slope x moderate asymptote': p(A) = (1-eps)*[m*sigmoid(beta*E) + (1-m)*(1-sigmoid(beta*E))] + eps/2, with effective attenuation A_eff = (1-eps)(2m-1) concentrated near 0.69. This decomposition is what the data jointly demand: the slope-sensitive diagnostics (psychometric slope posterior, evidence-usage slopes, mixture log-likelihood ratios) require steep normalized slopes, while the extremity/consistency diagnostics (saturated sparse-choice rates) require moderate asymptotic preference — a combination no single shallow-slope, high-extremity or steep-slope, high-extremity parameterization can produce.

**Parameters:**
- gamma_u: [0, 1]
- log_beta: [0.588, 1.792]
- epsilon: [0.03, 0.11]
- m: [0.74, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Raw-Evidence Sparse Dominance with Validity-Gated Precision (RSD-vgp).
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are 0/1 cue values. History is ignored
    # (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"RSD-vgp expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    # ---- Validity gating: per-subject gamma via a two-component mixture ----
    # gamma_u ~ U[0, 1]. With probability 0.65 the subject draws gamma
    # uniformly from [0, 1.3] (near-flat validity weighting); with
    # probability 0.35 from [1.3, 2.0] (the steep-gating subpopulation
    # that validity-conflict cells discriminate).
    u = float(parameters["gamma_u"])
    if u < 0.65:
        gamma = 1.3 * (u / 0.65)
    else:
        gamma = 1.3 + 0.7 * ((u - 0.65) / 0.35)

    # ---- Precision: log-uniform raw inverse temperature ----
    # beta = exp(log_beta), log_beta ~ U[ln 1.8, ln 6.0]  =>  beta
    # log-uniform over [1.8, 6.0]. Calibrated against the RAW evidence
    # scale below so that every subject's effective normalized slope
    # beta * sum(v^gamma) comfortably exceeds 9.5 on any plausible
    # validity vector (min ~ 1.8 * 7.4 ~ 13).
    beta = float(np.exp(float(parameters["log_beta"])))
    epsilon = float(parameters["epsilon"])
    m = float(parameters["m"])

    # RAW validity weighting: w_j = v_j^gamma, NO normalization.
    # This preserves the absolute endorsement-margin scale: a margin-2
    # conflict carries twice the evidence of a margin-1 conflict,
    # independent of n_features (the cross-experiment scale separation
    # that per-experiment normalization provably erases).
    w = np.power(val, gamma)

    a, b = stim[0], stim[1]
    # Anti-endorsement (sparse-preferring) evidence for A over B:
    # each endorsement carried by B counts FOR A, each endorsement
    # carried by A counts AGAINST A. E = 0 on identical rating vectors.
    E = float(np.sum(w * (b - a)))

    # Numerically stable logistic (both branches exponentiate a
    # non-positive argument, so no overflow is possible).
    x = beta * E
    if x >= 0.0:
        sig = 1.0 / (1.0 + np.exp(-x))
    else:
        ex = np.exp(x)
        sig = ex / (1.0 + ex)

    # Graded polarity: with per-trial probability m the subject responds
    # with the INVERTED (sparse-preferring) polarity, with probability
    # (1 - m) with the conventional polarity. Continuous, no empty
    # middle, no mirror cluster.
    p_core = m * sig + (1.0 - m) * (1.0 - sig)

    # Independent lapse to uniform choice. Ties (E = 0) give exactly 50%.
    p_a = (1.0 - epsilon) * p_core + 0.5 * epsilon

    return np.array([p_a, 1.0 - p_a])
```

**`policy source code`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # Guard against float drift.
    return np.random.choice(len(probs), p=probs)
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Crossing-point population statistic.\n    # Per subject: fit the shared core model p(B) = 0.5 - A*(sigmoid(beta*E(gamma)) - 0.5),\n    # E(gamma) = sum_j v_j^gamma * (b_j - a_j), by grid MLE over (gamma, beta, A),\n    # then score s = P(gamma > 1.95) - P(1.40 < gamma <= 1.95) under the flat-prior\n    # profile-likelihood posterior over gamma. Metric = mean of s over subjects.\n    a_list = data[\"option_a_ratings\"].to_numpy()\n    b_list = data[\"option_b_ratings\"].to_numpy()\n    A_mat = np.vstack([np.asarray(r, dtype=float) for r in a_list])\n    B_mat = np.vstack([np.asarray(r, dtype=float) for r in b_list])\n    y = data[\"response\"].to_numpy(dtype=float)\n    subj = data[\"subject_id\"].to_numpy()\n\n    n_feat = A_mat.shape[1]\n    if n_feat >= 8:\n        # design validities: six 1.00 cues, twenty-four 0.50 cues, 0.64, 0.76\n        V = np.array([1.0] * 6 + [0.5] * (n_feat - 8) + [0.64, 0.76], dtype=float)\n    else:\n        V = np.ones(n_feat, dtype=float)\n\n    gam = np.linspace(0.0, 2.5, 126)\n    bet = np.geomspace(1.6, 7.5, 8)\n    amps = np.linspace(0.25, 0.97, 9)\n\n    Wg = V[None, :] ** gam[:, None]  # (G, F) validity-power weights per gamma\n\n    scores = []\n    for s_id in pd.unique(data[\"subject_id\"]):\n        m = subj == s_id\n        D = (B_mat[m] - A_mat[m]).astype(float)  # (n, F)\n        yy = y[m]  # 1 = chose B\n        E = D @ Wg.T  # (n, G) raw evidence for A over B\n        X = E[:, :, None] * bet[None, None, :]  # (n, G, B)\n        d = 0.5 * np.tanh(0.5 * X)  # sigmoid(X) - 0.5, stable, in [-0.5, 0.5]\n        ll_prof = np.full(gam.shape[0], -np.inf)  # profile log-likelihood over gamma\n        for a in amps:\n            P = np.clip(0.5 - a * d, 1e-9, 1.0 - 1e-9)  # p(choose B)\n            ll = np.sum(\n                yy[:, None, None] * np.log(P)\n                + (1.0 - yy)[:, None, None] * np.log(1.0 - P),\n                axis=0,\n            )  # (G, B)\n            ll_prof = np.maximum(ll_prof, ll.max(axis=1))\n        w = np.exp(ll_prof - ll_prof.max())\n        w = w / w.sum()  # posterior over gamma (flat prior on [0, 2.5])\n        p_hi = float(w[gam > 1.95].sum())          # tail mass above RSD's hard cap\n        p_gap = float(w[(gam > 1.40) & (gam <= 1.95)].sum())  # HT's empty-gap mass\n        scores.append(p_hi - p_gap)\n    return float(np.mean(scores))",
  "rationale": "Both theories share the identical per-trial core p(B) = 0.5 - A_eff*(sigmoid(beta*E(gamma)) - 0.5) with E(gamma) = sum_j v_j^gamma*(b_j - a_j); they differ ONLY in the population distribution of gamma. HT-AEI-r6 puts 35% of subjects in a heavy tail on gamma in [2.0, 2.10] and leaves (1.35, 2.0) essentially EMPTY (2.3%), while RSD-vgp spreads 35% of subjects uniformly over [1.3, 2.0] with a hard cap at 2.0. The rung ladder was built so that per-subject gamma is recoverable from the 64 rung observations (flip points g* = 1.12/1.32/1.59/1.81/2.00/2.04/2.08/2.14; the deep rungs R5-R8 straddle the 2.0 boundary with 0.04 spacing, localizing a tail subject's gamma to within ~0.05-0.08).\n\nThe metric therefore fits the (shared, correctly specified) model per subject by grid MLE over gamma in [0, 2.5], beta in [1.6, 7.5], A_eff in [0.25, 0.97], converts the profile likelihood into a flat-prior posterior over gamma, and scores each subject as s = P(gamma > 1.95) - P(1.40 < gamma <= 1.95). The first term captures mass above RSD's hard gamma ceiling (HT tail subjects score ~ +0.4 to +1.0; RSD can only place ~2.5% true mass there plus a thin estimation smear), and the second term captures mass in HT's empty gap (1.40, 1.95), where RSD concentrates ~27% of its population (those subjects invert rungs R2-R4 but not R5, scoring ~ -0.9) but HT has almost none (its mid segment ends at 1.35). The statistic is continuous, so boundary subjects contribute partial credit instead of all-or-nothing noise, keeping per-subject variance moderate (~0.15-0.25) while the between-theory mean gap is large: expected pooled value ~ +0.22 to +0.30 under HT-AEI-r6 versus ~ -0.19 to -0.24 under RSD-vgp, giving Welch t ~ 3.5-4 at N = 25 (p < 0.002). The two-sided construction also makes the metric adversarial in both directions: it rewards HT only for genuinely above-2.0 precision AND punishes it if the (1.4, 1.95) gap fills in, so it is a real test of the population structure, not a tautology. It works identically on a single subject's slice (returns that subject's score), so between-subject variance is well defined."
}
```

## Usage

```json
{
  "input_tokens": 13673,
  "output_tokens": 63926,
  "reasoning_tokens": 62387,
  "provider": "Sail Research"
}
```
