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
**Validities (n_features=29):** [0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.8, 0.8, 0.8, 0.8, 0.8, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

**Trial pairs (n=24):**
  trial 1: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 4: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 12: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 13: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  trial 14: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 15: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 16: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 17: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 18: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 19: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  trial 20: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 21: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 22: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 23: A=[1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 24: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** ADVERSARIAL LOGIC. SCWI (advocated) and PAHI (competing) agree on everything except the validity-to-weight map: SCWI uses w(v) = v + tau*v^2 with tau in [0.15, 0.42]; PAHI uses w(v) = v^gamma with gamma in [1.4, 2.4]. Both are anti-validity additive integrators (a rating of 1 is a defect, tied cues cancel, softmax plus symmetric lapse), so the ONLY exploitable wedge is the SHAPE of w(v). The key mathematical fact: the local log-elasticity of SCWI's map, (1+2*tau*v)/(1+tau*v), lies in [1.13, 1.30] for ALL v in [0.5, 1] and ALL tau in its box, while PAHI's elasticity is exactly gamma >= 1.4. The ranges are DISJOINT, so for every validity pair (v_hi, v_lo) SCWI's weight ratio w(v_hi)/w(v_lo) is strictly SMALLER than PAHI's. Experiments [4] and [5] exploited this at the v = 1.00 anchor (where PAHI's weight is pinned at exactly 1). This design is the TRANSPOSE into the mid-validity region, using NO validity at 1.00: the knife-edge becomes a RATIONAL COUNT RATIO. For a trial pitting a endorsements on v_hi-cues against b endorsements on v_lo-cues, both theories prefer the high-validity-stack option iff their weight ratio is below b/a. SCWI ratio windows (over its full tau box): (0.75, 0.50): [1.552, 1.630]; (0.80, 0.50): [1.667, 1.766]; (0.85, 0.50): [1.783, 1.907]. PAHI windows (over gamma in [1.4, 2.4]): 1.5^gamma = [1.764, 2.646]; 1.6^gamma = [1.931, 3.090]; 1.7^gamma = [2.102, 3.573]. Each inter-window gap contains a simple rational: 5/3 = 1.667 and 7/4 = 1.75 in (1.630, 1.764); 9/5 = 1.80 in (1.766, 1.931); 2 in (1.907, 2.102). Placing the count ratio exactly at these rationals yields KILL CELLS with OPPOSITE sign predictions for EVERY admissible parameter value of BOTH theories.

KILL CELLS (evidence for the stated option, positive for all parameters): B3 = 6x0.75 vs 10x0.50 (ratio 5/3): SCWI prefers the 0.75-stack with evidence 0.5 - 0.875*tau in [0.13, 0.37]; PAHI prefers the 0.50-stack with evidence 0.5^gamma*(6*1.5^gamma - 10) in [0.22, 1.11]. B4 = 4x0.75 vs 7x0.50 (ratio 7/4): SCWI evidence 0.5 - 0.5*tau in [0.29, 0.43]; PAHI evidence in [0.02, 0.68]. K3 = 5x0.80 vs 9x0.50 (ratio 9/5): SCWI evidence 0.5 - 0.95*tau in [0.10, 0.36]; PAHI evidence in [0.25, 1.23]. A3 = 4x0.85 vs 8x0.50 (ratio 2): SCWI evidence 0.6 - 0.89*tau in [0.23, 0.47]; PAHI evidence in [0.16, 1.19]. Intuition: PAHI's steep power law makes mid-validity endorsements relatively heavy, so for PAHI a 1 on a 0.75-cue outweighs 5/3 endorsements on 0.50-cues; SCWI's near-linear-quadratic map makes them equivalent-or-lighter, so SCWI needs the count ratio to reach ~1.75-2.1 before flipping. POPULATION PREDICTIONS (SCWI mixture 16% guessers / 50% mid / 34% deterministic; PAHI with beta in [1.5,4], gamma in [1.4,2.4], epsilon in [0.06,0.2]): kill composite (48 trials/subject, pairs 1-12) SCWI = 0.62-0.72 (point ~0.66); PAHI = 0.15-0.45 (point ~0.25). Every non-guesser SCWI subject sits >= 0.55; every PAHI subject sits <= 0.50.

LADDER / FLIP-LOCATION DISSOCIATION. Bracketing rungs pin each theory's crossing point, per subject, at two independent validity pairs. On the 0.75/0.50 ladder: SCWI's flip ratio lies in (3/2, 5/3) for EVERY subject (low-stack at r = 3/2 [pair 13], high-stack at r = 5/3 [pairs 1-4]); PAHI's flip ratio is >= 1.764, hence at or above the 7/4 rung (low-stack at 5/3 AND at 7/4 [pairs 5-6], high-stack at r = 2 only if gamma < 1.71 [pairs 15-16]). On the 0.85/0.50 ladder: SCWI flips in (7/4, 2) (low-stack at 7/4 [pairs 17-18], high-stack at 2 [pairs 9-12]); PAHI flips above 2 (low-stack at 2, high-stack at 9/4 only if gamma < 1.53 [pairs 19-20]). A systematic ONE-RUNG SHIFT in the psychometric crossing over count ratio, replicated across two validity mixtures -- a quantitative signature SCWI's tau-box cannot cross into and PAHI's gamma-box cannot cross out of.

GAMMA-RUNGS (PAHI-specific consistency test). B5 (6x0.75 vs 12x0.50) splits PAHI exactly at gamma = 1.71; A4 (4x0.85 vs 9x0.50) splits PAHI at gamma = 1.53. SCWI predicts the HIGH-stack on both with huge margins (1.34-1.44 and 0.83-1.00): population P(high-stack) ~ 0.85 vs PAHI's ~0.30-0.40. A PAHI subject's rung and kill choices must be jointly consistent with a single gamma (e.g., high-stack on A4 implies gamma < 1.53, which forces high-stack on B5); SCWI predicts high-stack on all rungs uniformly, with essentially zero between-subject spread in crossing location (the whole tau box moves the 0.75-ladder crossing ratio by only 0.078, all inside the same rung gap), whereas PAHI's crossings spread over ~0.9 ratio units.

POPULATION-MIXTURE PROBE (novel; untouched by experiments [0]-[5]). SCWI's beta_seed mixture places 16% of subjects in a true-guesser regime (beta in [0.02, 0.10], epsilon in [0.12, 0.28]). On the all-1-vs-all-0 anchors (pairs 21-22) the total anti-load is ~23.5 (SCWI units) or ~17 (PAHI units), so these guessers sit at P(choose all-1) in [0.22, 0.35] -- and at ~0.5 on every kill and rung. PAHI has NO such regime: every subject has beta >= 1.5 and epsilon <= 0.20, so P(choose all-1) <= epsilon/2 <= 0.10 for EVERY subject. Statistic: fraction of subjects choosing the all-1 option on >= 3 of 8 anchor trials: SCWI ~ 0.08-0.13; PAHI ~ 0.01 (and no PAHI subject can exceed ~0.12 in true probability). SCWI additionally predicts a TRIMODAL per-subject anchor distribution (masses near 0.02, 0.05, and 0.3) with guesser-flagged subjects also at chance on kills/rungs; PAHI predicts a unimodal-low continuum.

CONTROLS. Pairs 13-14 (B2: 4x0.75 vs 6x0.50, r = 3/2) and 17-18 (A2: 4x0.85 vs 7x0.50, r = 7/4) are agreement cells where BOTH theories prefer the low-stack (SCWI evidence 0.75*tau in [0.11, 0.32] and 1.14*tau - 0.1 in [0.07, 0.38]; PAHI evidence 0.40-0.87 and 0.53-1.31): they calibrate the shared softmax/lapse on the anti-validity side and identify non-inverted TTB subjects (who pick the high-stack because the top discriminating cue favors it -- note TTB also picks the high-stack on the kills, so B2/A2 and the anchors are what exclude it) and pro-validity integrators (who pick all-1 on anchors). Pair 24 (CAL: one 0.85-endorsement vs one 0.50-endorsement) is a single-cue anti-validity calibration (both theories prefer the 0.50-1 option, P ~ 0.75-0.85). Pair 23 (TIE: identical mixed vectors) pins side/response bias at exactly 0.500 for both. All critical pairs are mirrored; B3 and A3 include cue-rotation variants (different low-cue subsets) to average within-subject per-cue attention (drawn from [0.90, 1.10]); kills that use every cue of a class (K3, B5) average attention by construction, and the population-level sign predictions are immune to attention because it is symmetric with mean 1. Worst-case attention analysis: flipping B3 requires ~8% adverse class-mean attention (~2 SD of a 10-cue class mean); B4/B5/A4 margins (0.29-1.44) require 15-30% and cannot flip.

DESIGN. 29 experts (6 at 0.85, 5 at 0.80, 6 at 0.75, 12 at 0.50); 24 unique pairs x K = 96//24 = 4 repetitions = 96 trials. Per subject: 48 kill trials, 32 rung trials, 8 anchor trials, 4 tie trials, 4 calibration trials.

DECISION RULES. (i) SCWI is supported, and PAHI falsified, if the kill composite P(high-stack) > 0.55 with per-subject high-stack majorities (PAHI cannot exceed 0.50 on any kill for any admissible parameters), the rung composite >= 0.75, the 0.75-ladder crossing lies in (1.5, 1.667) and the 0.85-ladder crossing in (1.75, 2) with tight between-subject clustering, and the anchor-tail fraction (subjects >= 3/8 all-1 choices, also near chance on kills) is ~0.10-0.15 where PAHI predicts ~0.01. (ii) PAHI is supported, and SCWI falsified, if all four kills show low-stack preference (composite < 0.45), the rung pattern is jointly consistent with a single gamma per subject (A4 high-stack => B5 high-stack; A4 low-stack with gamma > 1.53 => correspondingly graded B5), crossings sit at or above 5/3 (0.75-ladder) and at or above 2 (0.85-ladder) with wide between-subject spread, and the anchor distribution is unimodal-low with no subject above ~0.12. (iii) BOTH pure forms are falsified if subjects prefer the option with MORE high-validity endorsements (pro-validity: all-1 chosen on anchors, low-stack rejected on kills), or if a flat anti-tallying pattern appears (low-stack on EVERY cell including the B5/A4 rungs, which both theories reject), or if the profile fits neither map.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
**Description:** Power-Law Anti-Validity Integration with Heterogeneous Inversion (PAHI). Subjects integrate all cues additively into a per-option subjective value, but a rating of 1 is treated as evidence AGAINST the option (comprehension inversion, s = -1 for ~99% of subjects). The anti-weight of cue j is a free power law of the instructed validity, |w_j| = v_j^gamma, with gamma a free per-subject parameter drawn from a heterogeneous population distribution (gamma ~ 1.4-2.4, mildly supralinear on average). Choice is a softmax over beta * s * sum_j v_j^gamma * rating_j with a symmetric lapse epsilon. Because gamma, beta, and epsilon are drawn broadly per subject, the population reproduces both the mean-level signatures (negative Exp-1 slope, positive Exp-2 dissociation, positive Exp-3 slope, low Exp-4 conflict-following) and the large between-subject variance seen in the real data.

**Parameters:**
- beta: [1.5, 4.0]
- gamma: [1.4, 2.4]
- epsilon: [0.06, 0.20]
- sign_seed: [0, 1]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Power-Law Anti-Validity Integration with Heterogeneous Inversion (PAHI).
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    # Each cue contributes signed evidence w_j = s * v_j**gamma per unit
    # rating, where s = -1 for the (vast majority of) comprehension-inverted
    # subjects and +1 for the rare pro-validity subject; gamma is a free
    # power-law exponent applied to the instructed validity. Option values
    # are the cue-weighted sums; choice is a numerically stable softmax over
    # beta * values, mixed with a symmetric lapse epsilon. History is
    # ignored (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"PAHI expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    raw = parameters.get("validities")
    if isinstance(raw, str):
        s_txt = raw.strip().strip('[]()')
        val = np.array([float(t) for t in s_txt.replace(',', ' ').split()])
    else:
        val = np.asarray(raw, dtype=float).ravel()
    if val.size != n_features:
        # Defensive fallback: uniform mid-range validities.
        val = np.full(n_features, 0.75)

    beta = float(parameters["beta"])
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    sign_seed = float(parameters["sign_seed"])

    # Heterogeneous comprehension inversion: with population probability
    # 0.99 a subject inverts the rating direction (a 1 counts AGAINST the
    # option); the remaining subjects read ratings pro-validity.
    s = -1.0 if sign_seed < 0.99 else 1.0

    # Power-law anti-validity cue weights: the anti-weight of expert j is
    # v_j**gamma (on average mildly supralinear in the instructed validity).
    w = s * np.power(val, gamma)

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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n\n    # Design-time constants (fixed by this experiment: 29 experts,\n    # 6 @ .85, 5 @ .80, 6 @ .75, 12 @ .50).\n    V = np.array([0.85] * 6 + [0.80] * 5 + [0.75] * 6 + [0.50] * 12, dtype=float)\n\n    # Reference (midpoint) parameter values. The design guarantees that on\n    # every critical cell the SIGN of each theory's evidence is constant\n    # over its ENTIRE parameter box, so midpoint values classify trials\n    # identically to any admissible parameter.\n    TAU = 0.28      # SCWI box [0.15, 0.42]\n    GAMMA = 1.9     # PAHI box [1.4, 2.4]\n    w_scwi = -(V + TAU * np.square(V))   # SCWI anti-validity weights\n    w_pahi = -(np.power(V, GAMMA))       # PAHI anti-validity weights\n\n    def _to_arr(x):\n        if isinstance(x, np.ndarray):\n            return x.astype(float)\n        if isinstance(x, (list, tuple)):\n            return np.asarray(x, dtype=float)\n        s = str(x).strip().strip('[]()')\n        toks = [t for t in s.replace(',', ' ').split() if t != '']\n        return np.array([float(t) for t in toks])\n\n    hits = []\n    for a_raw, b_raw, r_raw in zip(data['option_a_ratings'],\n                                    data['option_b_ratings'],\n                                    data['response']):\n        try:\n            a = _to_arr(a_raw)\n            b = _to_arr(b_raw)\n            r = int(r_raw)\n        except Exception:\n            continue\n        if r not in (0, 1):\n            continue\n        n = int(min(a.size, b.size, V.size))\n        if n == 0:\n            continue\n        diff = a[:n] - b[:n]\n        d_scwi = float(np.dot(w_scwi[:n], diff))   # value_A - value_B under SCWI\n        d_pahi = float(np.dot(w_pahi[:n], diff))   # value_A - value_B under PAHI\n\n        # Keep only trials where the two theories make OPPOSITE predictions\n        # with a non-trivial margin on both sides (this automatically selects\n        # the knife-edge count-ratio cells and discards controls, ties,\n        # anchors, and calibration trials, where the theories agree).\n        if abs(d_scwi) <= 0.05 or abs(d_pahi) <= 0.05:\n            continue\n        if (d_scwi > 0) == (d_pahi > 0):\n            continue\n\n        scwi_option = 0 if d_scwi > 0 else 1   # 0 = A, 1 = B\n        hits.append(1.0 if r == scwi_option else 0.0)\n\n    if not hits:\n        return 0.5\n    return float(np.mean(hits))",
  "rationale": "The ONLY wedge between SCWI (w(v) = v + tau*v^2, tau in [0.15,0.42]) and PAHI (w(v) = v^gamma, gamma in [1.4,2.4]) is the shape of the validity-to-weight map: SCWI's weight ratio for any validity pair is strictly smaller than PAHI's (disjoint elasticity ranges). The design places 16 of the 24 unique pairs exactly on knife-edge count ratios (5/3, 7/4, 9/5, 2, and the gamma-rungs 9/4 and 12/6) where the two theories prefer OPPOSITE options for every admissible parameter value. The metric recomputes each trial's signed evidence under midpoint parameters of each theory (tau=0.28, gamma=1.9; sign is box-constant by design, so the classification is parameter-free in practice), keeps only trials where the two evidences have opposite signs with margin > 0.05 on both sides (this auto-selects the kill cells B3/B4/K3/A3 and rungs B5/A4, and auto-discards agreement controls B2/A2, anchors, ties, and the single-cue calibration, where the theories agree), and returns the fraction of choices falling on the SCWI-predicted option. Expected values: under SCWI (16% guessers at 0.5, 50% mid at ~0.70, 34% deterministic at ~0.83) the pooled value is ~0.70 with per-subject SD ~0.14; under PAHI (all subjects anti-validity with gamma in [1.4,2.4], whose steep power law favors the low-validity stacks) the pooled value is ~0.25 with per-subject SD ~0.13. Welch's t at N=25 on (0.70, 0.019) vs (0.25, 0.017) gives t ~ 11-13, p < 1e-12, far below alpha = 0.01. The metric works identically on a single subject's slice (64 disagreement trials per subject), so between-subject variance is well-defined. It is not trivially true for SCWI: it would fail (return ~0.25) if subjects instead followed a power-law anti-validity map, and it directly operationalizes the decision rule's kill composite."
}
```

## Usage

```json
{
  "input_tokens": 11794,
  "output_tokens": 23737,
  "reasoning_tokens": 22471,
  "provider": "Sail Research"
}
```
