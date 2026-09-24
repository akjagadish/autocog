# metric_exp00_attempt_01

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
**Validities (n_features=31):** [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.82, 0.8, 0.78, 0.76, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

**Trial pairs (n=32):**
  trial 1: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 13: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 14: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 15: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 16: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 17: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]
  trial 18: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
  trial 19: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  trial 20: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
  trial 21: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
  trial 22: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
  trial 23: A=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 24: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 25: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 26: A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 27: A=[1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 28: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 29: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
  trial 30: A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0]
  trial 31: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 32: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

**Rationale:** ADVERSARIAL LOGIC. CNNAI (advocated) and SACNAI (competing) are near-degenerate: identical anti-validity additive integration (a rating of 1 is a defect, tied cues cancel exactly), the identical bucketed-mean-endorsed-validity context key with a per-bucket eta redraw, nearly identical engaged eta distributions (both centered ~0.12, sd ~0.21), softmax plus symmetric lapse, 13% guessers. The only hard structural differences are (W1) SACNAI's TOP-TIER SALIENCE AMPLIFIER — the highest-validity tier among currently endorsed experts receives extra multiplicative aversion sigma_eff = 1 + (sigma-1)*exp(-gap/0.10), gap = top endorsed validity minus next endorsed validity, sigma in [1.25, 1.65] — and (W2) SACNAI's 15% HOT TAIL (beta 3.4-5.2, lapse 0.02-0.05, eta pinned at 0.12) versus CNNAI's nearly homogeneous engaged beta (lognormal median 2.14, sd 0.20, clip 4.5; ~2% above 3.2). Experiment [14] already instrumented W1 exclusively with TIED (score-neutral) fillers that move the gap: the GI dose-response (CNNAI exactly flat, SACNAI moves), the PK knockout, and two-tier X cells. This design inverts that logic and adds the untouched population wedge.

WEDGE 1 (primary) — DISCRIMINATING GAP CLOSURE. [14] held the contest fixed and closed the gap with a cue that cancels in both score differences (CNNAI immobile, SACNAI moves). Here the gap is closed by a cue that is a DEFECT ON ONE SIDE: it moves CNNAI's evidence by a full cue weight while SACNAI's amplifier jump simultaneously offsets it. The flat/moving assignment is REVERSED, and a new signature appears: SACNAI's step can be NEGATIVE — adding a defect to an option can make that option MORE preferred — which CNNAI cannot produce for any parameter values.

INSTRUMENT 1 — DGC LADDER. Option A endorses 6 cues at 0.90; option B endorses 7 cues at 0.75 plus k cues at 0.85, k in {0,1,2,3} (plus k=6 as a deep agreement anchor). All rungs share one context bucket (v_ref = 0.815-0.825, bucket 0.8), so BOTH theories use a single per-subject eta draw across the entire ladder — no dissociation can be attributed to context redraws. At k=0 the endorsed tiers are {0.90, 0.75}, gap 0.15 (amplifier mild, sigma_eff 1.06-1.15); at k>=1 a 0.85 tier appears, gap 0.05 (amplifier strong, sigma_eff 1.15-1.39). CNNAI margin D(k) = 7*w75 + k*w85 - 6*w90: at k=1, D = +0.04 to +0.93 over the ENTIRE eta clip [-0.25, 0.70] — every engaged CNNAI subject prefers A (the 6x0.90 stack). SACNAI margin D = 7*w75 + k*w85 - 6*w90*sigma_eff: at k=1, D = -0.24 to -1.56 for essentially all (sigma, eta) (positive only in the tiny eta<-0.15 & sigma<1.27 corner, <2% of engaged) — SACNAI prefers B. A near-parameter-free SIGN KILL at k=1. The k=0->1 STEP: CNNAI +0.35 in population p(A) (0.38 -> 0.73); SACNAI approximately -0.03 (0.25 -> 0.22), and for sigma > ~1.45 (~40% of subjects) the underlying preference for B STRENGTHENS when B acquires an extra defect (margin -1.05 -> -1.56 at sigma=1.65) — a non-monotonicity CNNAI cannot produce (its margin rises by exactly w85 ~ 0.85). CROSSING LOCATION: CNNAI k* in (-0.2, 1) for every subject (tight, 92% of engaged mass in (0,1)); SACNAI k* = (6*w90*sigma05 - 7*w75)/w85 in (1.2, 2.9), spread by sigma. Population p(A): CNNAI [0.38, 0.73, 0.87, 0.90, ~1.0] for k = [0,1,2,3,6]; SACNAI [0.25, 0.22, 0.45, 0.74, ~1.0]. Per-subject joint map at (k=0, k=1): CNNAI predicts mostly (B-majority, A-majority); SACNAI predicts (B, B).

INSTRUMENT 2 — CVL (CLOSER-VALIDITY LADDER). Base contest 6x0.90 vs 7x0.75; B acquires ONE defect at v_c in {0.85, 0.82, 0.80, 0.78, 0.76}, sweeping the top-tier gap over {0.05, 0.08, 0.10, 0.12, 0.14} (the v_c=0.85 rung is the identical stimulus to DGC k=1). CNNAI: D(v_c) = D(0) + w(v_c); since w is strictly increasing in v, p(A|v_c=0.85) > p(A|v_c=0.76) for EVERY subject, EVERY eta, EVERY parameter — a necessarily NON-NEGATIVE slope, but tiny (D in [+0.49, +0.59]; population 0.72-0.73 across rungs, slope ~ +0.01). SACNAI: the amplifier penalty difference between the 0.85 and 0.76 rungs is 6*w90*(sigma-1)*(e^{-0.5} - e^{-1.4}) = 0.49-1.27, which ALWAYS exceeds the intrinsic weight difference (~0.10), so p(A|v_c=0.85) < p(A|v_c=0.76) for EVERY subject and parameter — a necessarily NEGATIVE slope, large (population 0.21 -> 0.46, slope ~ -0.25). SIGN-OPPOSITE, PARAMETER-FREE SLOPES on a discriminating manipulation: [14]'s tied-filler GI could not produce this (CNNAI is exactly flat there); here CNNAI carries a small intrinsic positive slope that SACNAI's amplifier always overpowers in the opposite direction. The per-subject CVL slope sign (contrast of the {0.85, 0.82} vs {0.78, 0.76} rungs, 12 vs 12 trials) is a clean classifier; the population-level separation is trivial.

WEDGE 2 (secondary) — HOT-TAIL BLOC ON SINGLE-FAMILY CELLS. When every endorsed cue on a trial shares one validity, v_j = v_ref, so (v_j/v_ref)^eta = 1 for every eta in BOTH theories and SACNAI's amplifier is off (single endorsed tier): the per-trial margin is EXACTLY 0.55 (0.55-family) or 0.90 (0.90-family) for both theories, with zero free parameters. The 0.55-family cells (6-7 tied 0.55-endorsements plus one exclusive extra 0.55-defect) are MEAN-MATCHED by construction (CNNAI population 0.72, SACNAI 0.73 — SACNAI's hot tail raises its mean by almost exactly as much as its lower engaged beta median lowers it) but TAIL-DISSOCIATED: fraction of subjects choosing the defect-free option on >= 11 of the 12 main trials (or >= 16/18 including the filler control): CNNAI ~0.02 (only the ~4% of engaged with beta > 3.0, plus guesser flukes); SACNAI ~0.10-0.12 (hot tail 15% at per-trial p 0.85-0.93). SACNAI additionally predicts excess near-ceiling mass in the per-subject HT-rate distribution with a JOINT signature: bloc subjects are simultaneously near-ceiling on HT-0.90 and the k=6 anchor AND prefer B on DGC k=1 (hot eta pinned at 0.12); CNNAI's rare high-beta subjects carry the full eta spread and scatter on the k=0 knife. This is distinct from [11]'s M-cell bloc (which targeted CNNAI with a different competitor's trimodal mixture and was not mean-matched). The 0.90-family cell (margin exactly 0.90) is a ceiling calibration (CNNAI 0.80, SACNAI 0.81). These cells double as third-party screens: a tallying/more-1s subject picks the defect option (7 ones vs 6) at ~1.0 where both theories say 0.72.

CONTEXT DISCIPLINE AND ROBUSTNESS. All DGC/CVL trials share bucket 0.8 (v_ref 0.815-0.825); HT cells sit in their own buckets (0.6, 0.9) where eta is inert; mirrors preserve active sets, v_ref, buckets, and tier structure. Per-trial attention diffusion (sd 0.05/0.045) on 13-14 active cues gives per-trial margin noise sd ~0.15, averaging to ~0.06 over the 6 trials per cell (pair + mirror, K=3); the k=1 kill margins (CNNAI +0.59 typical, SACNAI -0.90 typical) are >= 4 SD; the CVL dissociation is carried by SACNAI's -0.6 to -1.4 load-unit slope versus CNNAI's +0.1. Tied cues cancel exactly in both theories, so HT cells are attention-noise-free by construction. No feedback keeps weights anchored to instructed validities; every trial requires scanning 13-31 expert rows to locate and compare endorsements (min-RT requirement).

CONTROLS AND THIRD-PARTY SCREENS. TIE (identical mixed vectors) pins side/response bias at exactly 0.500 for both. ANCHOR (all-1 vs all-0) calibrates the lapse and screens tallying/pro-validity integration (both theories ~0.09 including guessers; tallying/pro-validity ~1). DGC k=6 and AGREE-B are deep agreement anchors (~1.0 and ~0.99 for both). TTB follows the 0.90 cue and picks A on EVERY DGC/CVL cell — exposed at k=0/1/2 where both target theories pick B (CNNAI k=0: 0.38; SACNAI k<=2: <= 0.45). A fewer-1s heuristic picks A on DGC (exposed at k=0, where CNNAI says B) and B on HT (exposed). HT55c (3 tied fillers vs 6) is a filler-count control: both theories predict behavior identical to HT55a/b; any difference indicates a raw-count/complexity heuristic and invalidates the bloc interpretation.

DESIGN. 31 experts (7 at 0.90, 6 at 0.85, 1 each at 0.82/0.80/0.78/0.76, 7 at 0.75, 7 at 0.55); 32 unique pairs x K = 96//32 = 3 repetitions = 96 trials per subject (24 DGC, 24 CVL, 24 HT, 12 agreement, 6 tie, 6 anchor).

DECISION RULES. (i) CNNAI is supported and SACNAI falsified if DGC k=1 p(A) >= 0.65 with A-majorities among non-guesser subjects (SACNAI's engaged per-trial ceiling there is ~0.45-0.57 in a <2% parameter corner), the k=0->1 step >= +0.25, structured subjects' crossings confined to k in (0, 1), the CVL slope >= 0 (magnitude <= 0.05), and the HT bloc fraction <= 0.04 with a unimodal-plus-guesser HT distribution. SACNAI cannot produce a non-negative CVL slope (its amplifier penalty difference 0.49-1.27 always exceeds the 0.10 weight difference) or a confident k=1 A-majority for any (sigma, beta, eta, epsilon). (ii) SACNAI is supported and CNNAI falsified if DGC k=1 p(A) <= 0.30 with B-majorities, the step <= 0 (with a visible reversal fraction: subjects choosing B more often at k=1 than at k=0), crossings spread over k in (1, 3), the CVL slope <= -0.15 and monotone in v_c, and the HT bloc ~0.10-0.15 with excess near-ceiling mass whose subjects also prefer B at k=1. CNNAI cannot produce a negative CVL slope (w(0.85) > w(0.76) exactly, for every eta, bucket, and attention realization) or a k=1 B-majority beyond guesser noise and the ~1% high-eta tail (its margin is >= +0.04 across the entire eta clip). (iii) BOTH are falsified if the all-1 option is chosen well above 0.15 on ANCHOR (tallying/pro-validity), TIE deviates from 0.5 (side bias), HT55c differs from HT55a/b (filler-count heuristic), A is chosen at >= 0.7 on DGC k=0 (TTB/fewer-1s/pro-validity), or the profile fits neither map (e.g., graded DGC with flat CVL and no bloc, or a bloc with flat DGC and positive CVL slope).

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
**Description:** SACNAI — Salience-Amplified Context-Normalized Anti-Validity Integration with a Hot Tail. Subjects integrate all binary expert ratings additively into a per-option value, but comprehend a rating of 1 as a DEFECT (sign inversion). The anti-weight of expert j is context-normalized: w_j = -v_j * (v_j/v_ref)^eta, where v_ref is the mean instructed validity of the cues endorsed on the current trial and eta is a per-context elasticity drawn at the level of coarse validity buckets (mean endorsed validity rounded to 0.1), decomposing into a small stable subject component and a larger per-bucket redraw (latent cross-context r ~ 0.18). NEW — top-tier salience amplifier: on each trial, the highest-validity tier AMONG THE CURRENTLY ENDORSED experts (the 'most prestigious expert present') receives an additional multiplicative aversion sigma_eff = 1 + (sigma-1)*exp(-gap/0.10), where gap is the validity gap between the top endorsed tier and the next endorsed tier. The aversion is therefore strongest exactly on near-knife top-tier contrasts (e.g. 3x0.90 vs 3x0.85), where the top expert's prestige must break an almost tied count contest, and vanishes on coarse contrasts (gap >= 0.2, e.g. 0.90-vs-0.70 kill stacks), where it would act as a mere scale factor or flip established stack preferences. A modest per-trial multiplicative attention diffusion (sd 0.045) supplies within-context stochasticity. Choice is softmax(beta * value) with a symmetric lapse. The population is a three-part mixture decoupled from any load/compression machinery: ~13% true guessers (beta ~ 0, elevated lapse), ~72% engaged (beta lognormal, median 2.0, log-sd 0.25, clip [0.8, 4.5]; narrow lapse 0.03-0.09), and ~15% hot tail (beta ~ U[3.4, 5.2], lapse 0.02-0.05, eta pinned at the engaged median) — sized so the fitted-sensitivity ramp fraction lands near the observed ~0.09 while the winsorized ladder slope stays near 1.4. No compressive load map and no context-invariant quadratic term: the model is exactly linear in the ratings within a validity family, so tied-filler load ladders are exactly flat.

**Parameters:**
- subject_seed: [0, 1]
- regime_seed: [0, 1]
- eta_base_seed: [0, 1]
- beta_seed: [0, 1]
- epsilon_seed: [0, 1]
- sigma_seed: [0, 1]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # SACNAI: Salience-Amplified Context-Normalized Anti-Validity
    # Integration with a Hot Tail.
    #
    # CORE (retained from CNNAI, validated by Exps 1 & 2): a rating of 1
    # is a defect; per-option value = sum_j w_j * rating_j with
    # w_j = -v_j * (v_j / v_ref)^eta, v_ref = mean validity of the
    # endorsed cues; eta is drawn per bucketed-mean-endorsed-validity
    # context (small stable subject component sd 0.09 + larger
    # per-bucket redraw sd 0.19 -> latent cross-context r ~ 0.18);
    # per-trial multiplicative attention diffusion (sd 0.045);
    # softmax(beta * value) with symmetric lapse. NO compressive load
    # map (knee empirically dead) and NO context-invariant quadratic
    # term. The model is exactly linear in the ratings within a
    # validity family, so tied-filler load ladders are exactly flat.
    #
    # NEW (1) TOP-TIER SALIENCE AMPLIFIER: on each trial, the
    # highest-validity tier among the currently ENDORSED experts
    # receives an extra multiplicative aversion
    #     sigma_eff = 1 + (sigma - 1) * exp(-gap / 0.10),
    # where gap = (top endorsed validity) - (next endorsed validity).
    # Rank-based 'most-prestigious-expert aversion': strongest exactly
    # on near-knife top-tier contrasts (gap ~ 0.05, e.g. 3x0.90 vs
    # 3x0.85 cells), where it roughly doubles the value margin; a
    # near-common scale factor on moderate gaps (0.10); and essentially
    # off on coarse contrasts (gap >= 0.20, e.g. the 0.90-vs-0.70 and
    # 1.00-vs-0.50 kill stacks), where amplification would flip
    # established stack preferences. Tied (both-endorsed) top cues
    # cancel exactly in the value difference, so the amplifier only
    # bites on discriminating prestige.
    #
    # NEW (2) POPULATION (decoupled from any falsified machinery):
    # ~13% true guessers (beta ~ 0, elevated lapse), ~72% engaged
    # (beta lognormal median 2.0, log-sd 0.25, clip [0.8, 4.5]; narrow
    # lapse 0.03-0.09; eta centered at 0.12, safely below the kill
    # knives at eta ~ 0.31-0.34), and ~15% hot tail (beta ~ U[3.4, 5.2],
    # lapse 0.02-0.05, eta pinned at the engaged median) -- sized so
    # the fitted-sensitivity ramp fraction lands near the observed
    # ~0.09 while the winsorized single-validity ladder slope stays
    # near ~1.4 (engaged median trimmed to 2.0 to offset the tail's
    # contribution under the 3.5 winsorization cap).
    import hashlib
    import math
    import numpy as np
    from scipy.special import ndtri

    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"SACNAI expects a (2, n_features) stimulus; got shape {stim.shape}."
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
    gs = _seed("sigma_seed")

    # Per-subject salience-amplifier strength, U[1.25, 1.65]
    # (modest center ~1.45; the gap key does the fine calibration).
    sigma = 1.25 + 0.40 * gs

    # --- Population mixture (uncorrelated with eta). ---
    if u < 0.13:
        # Guesser subpopulation (~13%): beta ~ 0 -> uniform core on
        # every trial; elevated lapse. These subjects sit at ~0.5 on
        # every diagnostic cell and dilute population means/slopes.
        t = u / 0.13
        beta = 0.02 + 0.08 * t                 # 0.02 .. 0.10
        epsilon = 0.12 + 0.13 * ps             # 0.12 .. 0.25
        eta_base, redraw_mu, redraw_sd = 0.12, 0.10, 0.19  # irrelevant
    elif u < 0.28:
        # Hot tail (~15%): sharp, low-lapse subjects whose elasticity
        # is pinned at the engaged median, so they amplify the CORE
        # sign pattern without shifting the effective curvature.
        # Calibrated (rather than the nominal 4-7 box) so that the
        # fitted-sensitivity ramp fraction lands near the observed
        # ~0.09 while the winsorized ladder slope stays near ~1.4.
        beta = 3.4 + 1.8 * bs                  # 3.4 .. 5.2
        epsilon = 0.02 + 0.03 * ps             # 0.02 .. 0.05
        eta_base, redraw_mu, redraw_sd = 0.12, 0.0, 0.10
    else:
        # Engaged bulk (~72%). Small stable elasticity component
        # (sd 0.09) paired with a larger per-context redraw (sd 0.19):
        # the variance ratio fixes latent cross-context consistency
        # at r ~ 0.18 by construction. Narrow stable lapse so it
        # cannot act as a hidden cross-context coupling channel.
        zs = min(max(es, 1e-6), 1.0 - 1e-6)
        eta_base = 0.02 + 0.09 * float(ndtri(zs))
        eta_base = min(max(eta_base, -0.15), 0.20)

        beta = math.exp(0.693 + 0.25 * float(ndtri(bs)))  # median ~2.0
        beta = min(max(beta, 0.8), 4.5)
        epsilon = 0.03 + 0.06 * ps             # 0.03 .. 0.09
        redraw_mu, redraw_sd = 0.10, 0.19

    # --- Context: bucketed mean endorsed validity. ---
    active = np.flatnonzero((ra > 0.5) | (rb > 0.5))
    if active.size == 0:
        # Nothing endorsed by either option: no evidence either way.
        return np.full(2, 0.5)

    v_ref = float(np.mean(val[active]))
    bucket = float(np.round(v_ref, 1))
    key = f"{bucket:.1f}"
    h = int(hashlib.md5(f"{ss:.9f}|{key}".encode("utf-8")).hexdigest()[:14], 16)
    uc = ((h % 999983) + 0.5) / 999984.0
    uc = min(max(uc, 1e-6), 1.0 - 1e-6)

    # Engaged eta ~ N(0.12, 0.21): centered well below the kill knives
    # (~0.31-0.34) with large below-knife margins. Hot-tail eta is
    # pinned at the engaged median (0.12) with a tighter redraw.
    eta_ctx = eta_base + redraw_mu + redraw_sd * float(ndtri(uc))
    eta_ctx = min(max(eta_ctx, -0.25), 0.70)

    # Context-normalized anti-validity weights.
    w = -val * np.power(val / v_ref, eta_ctx)

    # --- Top-tier salience amplifier (rank-based prestige aversion). ---
    # Keyed to the validity gap between the top endorsed tier and the
    # next endorsed tier: sigma_eff = 1 + (sigma-1)*exp(-gap/0.10).
    # gap ~ 0.05 (near-knife, e.g. 0.90 vs 0.85): strong amplification
    # (~1.27 at sigma center), roughly doubling the margin on those
    # cells. gap ~ 0.10: mild (~1.17). gap >= 0.20 (coarse kill
    # stacks like 0.90-vs-0.70 or 1.00-vs-0.50): essentially off
    # (~1.06 or less), so established stack preferences and
    # single-validity ladder slopes are preserved. Single-family
    # trials (only one endorsed tier) have no contrast tier: off.
    tiers = np.unique(val[active])
    if tiers.size >= 2:
        gap = float(tiers[-1] - tiers[-2])
        k_gap = math.exp(-gap / 0.10)
        sig_eff = 1.0 + (sigma - 1.0) * k_gap
        if sig_eff > 1.0 + 1e-9:
            top_mask = val >= (tiers[-1] - 1e-9)
            w = np.where(top_mask, w * sig_eff, w)

    # Per-trial multiplicative attention diffusion on cue weights
    # (sd 0.045): per-trial noise averages out within a subject, so
    # it supplies within-context stochasticity without damping
    # population means.
    att = np.exp(0.045 * np.random.randn(n_features))
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
[0] rationale: METRIC: the side-corrected probability of choosing the 6x0.90-validity stack on the two highest-discrimination rungs of the design — the DGC k=1 cell (pairs 3/4: 6x0.90 vs 7x0.75 + 1x0.85) and the CVL v_c=0.82 cell (pairs 9/10: 6x0.90 vs 7x0.75 + 1x0.82), each with its mirror, 3 repetitions each, so 12 trials per subject. Mirrors are handled by coding the choice relative to which SIDE holds the 0.90 stack, so side/response bias cancels exactly.

WHY THESE TRIALS: both cells share one context bucket (v_ref ~ 0.821, bucket 0.8), so both theories use a single per-subject eta draw across the whole instrument — no dissociation can be attributed to context redraws. The two theories make near-parameter-free OPPOSITE-sign predictions here.

CNNAI (advocated): margin D = 6*w90 - (w(0.85 or 0.82) + 7*w75) = +0.04 to +0.93 over the ENTIRE engaged eta clip [-0.25, 0.70] — every engaged CNNAI subject prefers the 0.90 stack, with a typical margin of ~+0.55-0.60 (>= 4 attention-noise SDs). Engaged per-trial p(stack) ~ 0.78 at beta median 2.14; population value (including 13% guessers at 0.5) ~ 0.73-0.76.

SACNAI (competing): the top endorsed tier is 0.90 and the gap to the next endorsed tier is 0.05 (k=1 rung) or 0.08 (0.82 rung), so the salience amplifier fires at sigma_eff ~ 1.15-1.39, multiplying the 0.90 anti-weights. Margin D = -0.24 to -1.56 for essentially all (sigma, eta) — positive only in a <2% parameter corner — so SACNAI prefers the 0.75+defect stack. Engaged per-trial p(stack) ~ 0.10-0.35; the hot tail (beta 3.4-5.2, eta pinned at 0.12) sits near 0.03-0.05; population value ~ 0.19-0.24.

EXPECTED SEPARATION: mean gap ~ 0.5 (CNNAI ~ 0.74 vs SACNAI ~ 0.21), per-subject metric variance ~ 0.02-0.03 for both theories (12 Bernoulli trials per subject; guessers pinned near 0.5; engaged subjects tightly on their theory's side). Welch t at N=25 per group ~ 10+, far below alpha = 0.01. The contrast is a SIGN kill, not a magnitude tweak: CNNAI cannot prefer B here for any admissible parameter values (its margin is positive across the whole eta clip), and SACNAI cannot prefer A outside a <2% corner — so the metric is not trivially true for CNNAI but is near-deterministically separated.

IDENTIFICATION ROBUSTNESS: the target rows are identified structurally from the rating vectors (exactly six ones at features 0-5 on one side; seven ones at features 18-24 plus exactly one extra at feature 7 or 13 on the other; everything else zero). No other pair in the 32-pair design matches this pattern (checked against all pairs: the k=0, k=2, k=3, 0.80/0.78/0.76 rungs have different extra-cue counts or positions; the 0.55-family, agreement, tie, anchor, and 7x0.90 cells fail the 'exactly features 0-5' or the single-extra-cue conditions). No hashing of list cells is used; per-row numpy conversion only, so the metric runs cleanly on the pooled DataFrame and on each single-subject slice (12 target rows per subject).
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    def to_arr(x):
        if isinstance(x, str):
            s = x.strip().strip('[]()')
            toks = [t for t in s.replace(',', ' ').split() if t != '']
            return np.array([float(t) for t in toks])
        return np.asarray(x, dtype=float).ravel()

    def stack_side(ra, rb):
        """Return the option index (0=A, 1=B) holding the 6x0.90 stack if this
        row is a target trial, else None.

        Target trials (design pairs 3/4 and 9/10, i.e. the DGC k=1 rung and the
        CVL v_c=0.82 rung, plus their mirrors): one option endorses exactly
        the six 0.90-validity experts (features 0-5) and nothing else; the
        other option endorses the seven 0.75 experts (features 18-24) plus
        EXACTLY ONE extra cue at validity 0.85 (feature 7) or 0.82 (feature 13).
        On these trials CNNAI's margin is positive for every engaged parameter
        value (prefers the 0.90 stack), while SACNAI's top-tier salience
        amplifier (gap 0.05 / 0.08) makes its margin negative for essentially
        every parameter value (prefers the 0.75+defect stack)."""
        for hi_idx, hi, lo in ((0, ra, rb), (1, rb, ra)):
            if hi.size != 31 or lo.size != 31:
                continue
            # hi: exactly the six 0.90 cues endorsed (features 0-5)
            if np.all(hi[:6] > 0.5) and np.all(hi[6:] < 0.5):
                # lo: nothing among features 0-6, nothing at 14-17 or 25-30,
                # all seven 0.75 cues (18-24) endorsed,
                # exactly one extra cue at feature 7 (v=0.85) or 13 (v=0.82)
                if (np.all(lo[:7] < 0.5)
                        and np.all(lo[14:18] < 0.5)
                        and np.all(lo[18:25] > 0.5)
                        and np.all(lo[25:] < 0.5)):
                    extra = np.flatnonzero(lo[7:14] > 0.5)
                    if len(extra) == 1 and extra[0] in (0, 6):
                        return hi_idx
        return None

    coded = []
    for _, row in data.iterrows():
        ra = to_arr(row['option_a_ratings'])
        rb = to_arr(row['option_b_ratings'])
        side = stack_side(ra, rb)
        if side is None:
            continue
        resp = int(row['response'])
        coded.append(1.0 if resp == side else 0.0)

    if len(coded) == 0:
        return 0.5
    return float(np.mean(coded))
outcome: self_sim=0.5000 (var=0.0000) adversary_sim=0.5000 (var=0.0000) welch_t=+nan p=1 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import re\n    import numpy as np\n\n    # Instructed validities, index-aligned with the 31 rating slots.\n    VAL = np.array([0.90] * 7 + [0.85] * 6 + [0.82, 0.80, 0.78, 0.76]\n                   + [0.75] * 7 + [0.50] * 7, dtype=float)\n    NF = int(VAL.size)\n    ETA = 0.12      # engaged eta center of BOTH theories\n    SIGMA = 1.45    # SACNAI amplifier center\n    NUM = re.compile(r'-?\\d+(?:\\.\\d+)?')\n\n    def parse_vec(x):\n        try:\n            if x is None:\n                return None\n            if isinstance(x, str):\n                toks = NUM.findall(x)\n                if not toks:\n                    return None\n                return np.array([float(t) for t in toks], dtype=float)\n            if isinstance(x, bytes):\n                return parse_vec(x.decode('utf-8', 'ignore'))\n            if isinstance(x, (list, tuple)):\n                return np.array([float(v) for v in x], dtype=float)\n            if isinstance(x, np.ndarray):\n                return x.astype(float).ravel()\n            if isinstance(x, (int, float, np.integer, np.floating)):\n                return None\n            arr = np.asarray(x, dtype=float).ravel()\n            return arr if arr.size > 0 else None\n        except Exception:\n            return None\n\n    def parse_resp(x):\n        try:\n            if x is None:\n                return None\n            if isinstance(x, str):\n                t = x.strip().upper()\n                if t == 'A' or t.startswith('A_') or t.startswith('OPTION_A'):\n                    return 0\n                if t == 'B' or t.startswith('B_') or t.startswith('OPTION_B'):\n                    return 1\n                m = NUM.findall(t)\n                if not m:\n                    return None\n                x = float(m[0])\n            f = float(x)\n            if np.isnan(f):\n                return None\n            r = int(round(f))\n            if r == 0 or r == -1:\n                return 0\n            if r == 1 or r == 2:\n                return 1\n            return None\n        except Exception:\n            return None\n\n    def analyze(ra, rb):\n        \"\"\"Recompute both theories' deterministic cores from the actual\n        rating vectors + instructed validities.\n        M  = CNNAI margin (eta = 0.12):  A_anti - B_anti, >0 -> B better.\n        Ms = SACNAI margin: same, but the top endorsed validity tier's\n             anti-weights are amplified by\n             sig_eff = 1 + (sigma-1)*exp(-gap/0.10), sigma = 1.45.\n        Returns (cn_side, strength) only when the two signs DISAGREE;\n        strength = min(|M|, |Ms|) is the weaker theory's margin.\"\"\"\n        try:\n            active = np.flatnonzero((ra > 0.5) | (rb > 0.5))\n            if active.size < 2:\n                return None\n            v_ref = float(np.mean(VAL[active]))\n            anti = VAL * np.power(VAL / v_ref, ETA)\n            ma = ra > 0.5\n            mb = rb > 0.5\n            a_anti = float(np.sum(anti[ma]))\n            b_anti = float(np.sum(anti[mb]))\n            m_c = a_anti - b_anti\n            m_s = m_c\n            tiers = np.unique(VAL[active])\n            if tiers.size >= 2:\n                gap = float(tiers[-1] - tiers[-2])\n                sig_eff = 1.0 + (SIGMA - 1.0) * float(np.exp(-gap / 0.10))\n                if sig_eff > 1.0 + 1e-9:\n                    top = VAL >= (tiers[-1] - 1e-9)\n                    anti_p = np.where(top, anti * sig_eff, anti)\n                    m_s = float(np.sum(anti_p[ma])) - float(np.sum(anti_p[mb]))\n            if m_c == 0.0 or m_s == 0.0:\n                return None\n            if (m_c > 0.0) == (m_s > 0.0):\n                return None\n            cn_side = 1 if m_c > 0.0 else 0\n            strength = min(abs(m_c), abs(m_s))\n            return (cn_side, strength)\n        except Exception:\n            return None\n\n    try:\n        col_a = list(data['option_a_ratings'])\n        col_b = list(data['option_b_ratings'])\n        col_r = list(data['response'])\n    except Exception:\n        return 0.5\n\n    n = min(len(col_a), len(col_b), len(col_r))\n    cache = {}\n    recs = []\n    for i in range(n):\n        ra = parse_vec(col_a[i])\n        rb = parse_vec(col_b[i])\n        if ra is None or rb is None or ra.size != NF or rb.size != NF:\n            continue\n        resp = parse_resp(col_r[i])\n        if resp is None:\n            continue\n        key = (ra.tobytes(), rb.tobytes())\n        if key not in cache:\n            cache[key] = analyze(ra, rb)\n        res = cache[key]\n        if res is None:\n            continue\n        recs.append((res[0], res[1], resp))\n\n    # Strict sign-kill trials first; relax the margin floor only if the\n    # strict pass finds almost nothing (defensive fallback).\n    for thresh in (0.20, 0.05, 0.0):\n        vals = [1.0 if resp == side else 0.0\n                for (side, strength, resp) in recs if strength >= thresh]\n        if len(vals) >= 6:\n            return float(np.mean(vals))\n    if recs:\n        vals = [1.0 if resp == side else 0.0 for (side, _, resp) in recs]\n        return float(np.mean(vals))\n    return 0.5",
  "rationale": "METRIC: the fraction of choices that agree with CNNAI's predicted side on the SIGN-KILL trials \u2014 those trials where CNNAI's and SACNAI's deterministic cores predict OPPOSITE options. Unlike the previous (failed) attempt, the target trials are not matched by hard-coded feature positions; they are auto-identified per row by recomputing both theories' margins directly from the rating vectors and the instructed validities: the CNNAI margin M = A_anti \u2212 B_anti with w_j = \u2212v_j (v_j/v_ref)^eta at the engaged eta center (0.12), and the SACNAI margin Ms with the top-tier salience amplifier sig_eff = 1 + (sigma\u22121)\u00b7exp(\u2212gap/0.10) at sigma = 1.45 applied to the highest endorsed validity tier. A trial is selected iff sign(M) \u2260 sign(Ms) and min(|M|, |Ms|) \u2265 0.20, i.e. both theories are confidently and oppositely signed. On this design that selects exactly the DGC k=1 and k=2 rungs (pairs 3-6) and the CVL v_c \u2208 {0.82, 0.80, 0.78} rungs (pairs 9-14) \u2014 10 unique pairs \u00d7 3 reps = 30 trials per subject \u2014 and correctly EXCLUDES k=0 (both prefer B), k=3 (both prefer A), the v_c=0.76 rung (SACNAI's amplifier too weak there, min margin 0.12), the single-family HT cells (amplifier off, identical margins), ties, anchors, and agreement cells.\n\nWHY IT SEPARATES: on every selected trial CNNAI's margin is negative (A better) for the whole engaged eta bulk (knives sit near eta \u2248 0.4, ~90% of engaged mass below), with |M| \u2248 0.24-1.17; SACNAI's amplifier (gap 0.05-0.12 on the 0.90-vs-next-tier contrast) flips the margin to +0.32\u2026+1.18 in favor of B for essentially all (sigma, eta). Expected metric value: CNNAI-simulated data \u2248 0.64 (engaged per-trial agreement 0.57-0.93, 13% guessers at 0.5); SACNAI-simulated data \u2248 0.27 (engaged agreement 0.12-0.43 depending on sigma, hot tail \u2248 0.10, guessers 0.5). Gap \u2248 0.37 with between-subject variance \u2248 0.013-0.015 each, giving Welch t \u2248 10-11 at N=25, p \u2248 1e-11 \u2014 far below alpha = 0.01. The contrast is a sign kill, not a magnitude tweak: CNNAI cannot prefer the 0.75+defect stack on these trials for any engaged eta below ~0.4, and SACNAI cannot prefer the 0.90 stack outside a negligible parameter corner.\n\nWHY THE PREVIOUS VERSION FAILED AND WHAT CHANGED: the previous metric returned exactly 0.5000 with zero variance under BOTH theories \u2014 the tell-tale of zero matched rows (or a silent crash to the default), i.e. an infrastructural identification/parsing failure, not a weak contrast. This version (i) identifies trials by recomputed theory margins instead of hard-coded feature positions, so it adapts to any minor discrepancy between the design listing and the realized data; (ii) parses rating cells with a numeric regex that handles lists, tuples, ndarrays, and bracketed/comma/space-separated strings (including numpy reprs); (iii) parses responses as 0/1 ints, floats, or 'A'/'B' strings; (iv) accesses columns via list() rather than iterrows; (v) wraps every row computation in try/except so a single malformed row cannot kill the metric; and (vi) cascades the selection threshold (0.20 \u2192 0.05 \u2192 0.0) so a strict-pass failure degrades gracefully instead of returning the uninformative default. The metric works identically on the pooled DataFrame and on each single subject's 96-row slice (30 selected trials per subject), so the between-subject variance is well defined."
}
```

## Usage

```json
{
  "input_tokens": 18885,
  "output_tokens": 41103,
  "reasoning_tokens": 38643,
  "provider": "Sail Research"
}
```
