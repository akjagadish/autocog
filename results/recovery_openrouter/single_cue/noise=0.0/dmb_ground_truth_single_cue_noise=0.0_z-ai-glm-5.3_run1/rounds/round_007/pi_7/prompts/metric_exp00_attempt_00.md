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
**Validities (n_features=10):** [0.99, 0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.5, 0.5, 0.5]

**Trial pairs (n=32):**
  trial 1: A=[1, 1, 0, 0, 1, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
  trial 2: A=[0, 1, 0, 0, 1, 0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 1, 0, 0, 1, 0, 0]
  trial 3: A=[1, 1, 0, 1, 0, 0, 1, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 1, 0, 0, 1]
  trial 4: A=[0, 0, 0, 1, 0, 0, 1, 0, 0, 1]  B=[1, 1, 0, 1, 0, 0, 1, 0, 0, 1]
  trial 5: A=[1, 0, 1, 0, 0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 0, 0, 1, 0, 0]
  trial 6: A=[1, 1, 0, 0, 1, 0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1, 0, 0, 0, 1, 0]
  trial 7: A=[0, 1, 0, 0, 1, 0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0, 0, 0, 1, 0]
  trial 8: A=[1, 0, 1, 1, 0, 0, 1, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 1, 0, 0, 1]
  trial 9: A=[0, 0, 1, 1, 0, 0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 0, 1, 0, 0, 1]
  trial 10: A=[1, 1, 1, 0, 0, 1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 1, 0, 0, 1, 0]
  trial 11: A=[0, 1, 0, 0, 0, 1, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0, 1, 0, 0, 1, 0]
  trial 12: A=[1, 1, 1, 0, 0, 0, 1, 0, 0, 1]  B=[0, 0, 0, 1, 1, 1, 1, 0, 0, 1]
  trial 13: A=[0, 0, 0, 1, 1, 1, 1, 0, 0, 1]  B=[1, 1, 1, 0, 0, 0, 1, 0, 0, 1]
  trial 14: A=[1, 0, 1, 0, 1, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 1, 0]
  trial 15: A=[0, 0, 0, 0, 1, 0, 0, 1, 1, 0]  B=[1, 0, 1, 0, 1, 0, 0, 0, 1, 0]
  trial 16: A=[1, 0, 1, 1, 0, 0, 1, 1, 1, 0]  B=[0, 0, 0, 1, 0, 0, 1, 0, 0, 0]
  trial 17: A=[0, 1, 0, 0, 0, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 0, 1, 0, 0, 0, 1]
  trial 18: A=[1, 0, 0, 0, 1, 0, 0, 1, 1, 0]  B=[1, 0, 1, 0, 1, 0, 0, 0, 0, 0]
  trial 19: A=[0, 1, 0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0, 1, 0, 0, 0]
  trial 20: A=[0, 1, 1, 0, 0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 1, 1, 1, 1]
  trial 21: A=[1, 0, 0, 1, 0, 0, 0, 1, 0, 1]  B=[1, 0, 0, 1, 0, 0, 0, 0, 0, 1]
  trial 22: A=[0, 0, 1, 0, 0, 1, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
  trial 23: A=[0, 0, 1, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 1, 0, 1, 1, 0]
  trial 24: A=[0, 1, 0, 0, 1, 0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
  trial 25: A=[1, 1, 1, 1, 0, 1, 0, 1, 0, 1]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 1]
  trial 26: A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0, 1, 0, 1, 0, 1]
  trial 27: A=[1, 0, 1, 0, 1, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 1, 0, 1, 0, 1, 0]
  trial 28: A=[1, 0, 0, 0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0, 0, 0, 1, 0]
  trial 29: A=[0, 1, 1, 1, 0, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 1, 1, 1, 0, 0]
  trial 30: A=[1, 0, 0, 1, 0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0, 0, 1, 0, 1, 0]
  trial 31: A=[1, 0, 0, 1, 0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0, 1, 0, 1, 0]
  trial 32: A=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 1, 1, 1]

**Rationale:** CORE LOGIC. CREDT (advocated) and BODT (competing) share the identical architecture (sign votes x panel-relative weights x low-gain softmax x lapse), the same plateau (w_p in [0.88,0.92] over [v_lo,v_hi]), the same trust ceiling (~0.981), and the same consensus-gating machinery (same g0/s_gap/lambda_ext). Prior head-to-head rounds ([12],[13]) probed them only at gap = 0.05 (the spurious-precision peak). This design moves to the LARGE-EDGE regime (gap = 0.09: a 0.99 claim atop a 0.90 second rung), where the two theories' formulas diverge in TWO orthogonal, previously unexploited ways. (1) OUTLIER WEIGHT: CREDT's distrust is an UNBOUNDED additive discount with a NARROW right flank (s_r in [0.030,0.036]) plus a half-engaged composition amplifier (c_frac = 3/10 = 0.30 sits just below its c_half in [0.34,0.38], contributing C ~ 0.30-0.42), giving w(0.99) in [-0.70, -0.04] across its ENTIRE box -- persistent anti-diagnostic distrust even at a large edge. BODT's distrust is a BOUNDED saturating shrinkage toward w_floor with a WIDE right flank (s_r in [0.062,0.088], so its P-bump stays ~0.78-0.90 at gap 0.09) but the saturation caps the demotion: w(0.99) in [+0.22, +0.44] across its ENTIRE box (its composition route is strangled by comp_gate ~ 0.01 at c_frac 0.30). THE w_top BOXES ARE SIGN-DISJOINT. (2) GATED CHANCE WEIGHT: at gap 0.09 the consensus gate is ~0.98 OPEN for both, exposing a structural difference never exploited before: CREDT promotes gated chance cues to FULL plateau weight, w(0.50) in [0.85, 0.91], while BODT caps the promotion at kappa_gate*w_p, w(0.50) in [0.55, 0.71] -- DISJOINT by 0.14-0.35 per cue. ([12]/[13] worked at gap 0.05 where the gate was only ~0.78 open and treated chance weight as common currency; here it is a second lever at maximum separation.) Everything else is common currency: plateau experts (0.55-0.90) get ~w_p in both (BODT adds only a shallow tilt of 0.04-0.12). The panel [0.99, 0.99, 0.90, 0.80, 0.70, 0.60, 0.55, 0.50, 0.50, 0.50] therefore reduces the two theories to nearly anti-correlated weight vectors: CREDT ~ [-0.35, -0.35, 0.90, 0.90, 0.90, 0.90, 0.90, 0.88, 0.88, 0.88] ('count everyone EXCEPT the 99% experts, whose endorsements count against') vs BODT ~ [+0.32, +0.32, 0.89, 0.87, 0.85, 0.84, 0.83, 0.63, 0.63, 0.63] ('the 99% experts are weak positives; coin-flippers are 70% as good as real experts'). 32 unique pairs x K = 96//32 = 3 = 96 trials.

FAMILY M (flagship: masked lone/double outlier endorsement). M1 (A wins one 0.99 expert, all else tied): s = w_top exactly. CREDT box p(A) in [0.37, 0.49]; BODT box [0.54, 0.59] -- SIGN-DISJOINT around 0.50: CREDT says a lone 99%-atop-90% endorsement on this panel pushes choice AGAINST the endorsed option; BODT says weakly for it. M3 (A wins both 0.99s): CREDT [0.25, 0.48] vs BODT [0.58, 0.67] -- sign-disjoint with a doubled lever. M2 (A wins {0.99, 0.60} vs B wins {0.70}) replicates M1 with a masked balanced background: CREDT must predict M1 = M2 EXACTLY (flat plateau cancels), a within-theory consistency check; BODT predicts M2 ~ M1 up to its small tilt. Mirrors M1m/M3m control side bias.

FAMILY V/W (outlier head-to-heads). V1 (A wins 0.99, B wins one coin-flipper): CREDT [0.23, 0.33] vs BODT [0.41, 0.48] -- 'a coin-flipper's endorsement beats a 99% expert's': CREDT says clearly yes, BODT says near-tie. W1 (A wins 0.99, B wins the 0.90 expert): CREDT [0.23, 0.32] vs BODT [0.38, 0.42]. Both cells have TIED raw cue tallies (1-vs-1), so they also defeat any count-the-1s heuristic. Mirrors included.

FAMILY O (stacks). O1 (A wins {0.99, 0.90}): CREDT [0.54, 0.67] vs BODT [0.68, 0.75] -- even a unanimous 99+90 endorsement is only weakly followed under CREDT (the anti-vote nearly cancels the 90% endorsement) but clearly followed under BODT. O2 (A wins {0.99, 0.99, 0.90} vs B wins {0.80, 0.70, 0.60}; raw tally TIED 3-3): CREDT [0.08, 0.20] vs BODT [0.27, 0.37] -- a 12-25 point dissociation on a count-balanced trial, the strongest heuristic-defeating cell. Mirrors included.

FAMILY R (outlier + chance-cue ladder vs the 0.90 expert). R1 (A wins {0.99, one coin-flipper} vs B wins 0.90): CREDT [0.36, 0.50] vs BODT [0.47, 0.56] -- the crossover cell: CREDT says the 99%+coin-flipper stack still loses to one 90% expert; BODT says it wins. R2 (two coin-flippers): centers 0.60 vs 0.63 (converging dose point). The R-ladder crossover location (in coin-flipper units) is a direct estimate of w_top: CREDT crosses at k* ~ 1.4, BODT at k* ~ 0.9.

FAMILY X (chance-vs-plateau exchange rate -- the kappa_gate ladder; outlier held TIED so w_top contributes exactly zero). B wins the 0.90 expert; A wins k = 1, 2, 3 coin-flippers. CREDT: k=1 p in [0.50, 0.52] (a single gated coin-flipper is worth essentially ONE FULL 90% expert -- full promotion to w_p), k=2 [0.65, 0.68], k=3 [0.78, 0.81]. BODT: k=1 [0.44, 0.47] (the capped coin-flipper clearly loses), k=2 [0.53, 0.61], k=3 [0.63, 0.74]. DISJOINT AT EVERY RUNG; the X-ladder crossover (CREDT k* ~ 1.0 vs BODT k* ~ 1.4) measures the gated chance weight directly, and the DIFFERENCE between the R-crossover and the X-crossover measures w_top in chance-cue units (CREDT ~ -0.4 cues, BODT ~ +0.5 cues -- opposite signs). Mirror Xm3c included.

FAMILY Z (chance-only ladder). A wins 1/2/3 coin-flippers, nothing else discriminates (the outlier is present in the panel, keeping the gate ~0.98 open, but tied). CREDT: [0.66,0.68] / [0.79,0.81] / [0.87,0.89]; BODT: [0.60,0.64] / [0.69,0.76] / [0.77,0.84]. Disjoint at every rung (tight at k=1, widening with k): under CREDT, gated coin-flippers are near-exchangeable with plateau experts, so a 2-0 coin-flip sweep is followed at ~0.80; under BODT's partial cap, only ~0.72. Mirror Z2m included.

FAMILY K (coalition). K1 (A wins {0.99, 0.99, one coin-flipper} vs B wins {0.90, 0.80}): CREDT [0.15, 0.32] vs BODT [0.36, 0.48] -- disjoint. Raw tally favors A (3-vs-2) while CREDT predicts A loses: another count-heuristic defeat. Mirror included.

FAMILY T (honest parameter-free knives against MY OWN theory, CREDT). With the outlier tied and only plateau cues discriminating with tied counts, CREDT's exactly-flat plateau forces p = 0.50 EXACTLY on T1 (A wins 0.90 vs B wins 0.55), T1m, and T2 (A wins {0.90,0.80} vs B wins {0.60,0.55}) for EVERY parameter setting. BODT's within-plateau tilt predicts 0.52-0.54 (T1), 0.46-0.48 (T1m), 0.52-0.55 (T2). Any reliable tilt on these cells falsifies CREDT's flat-plateau claim with zero free parameters; exact 0.50 across the family falsifies BODT's tilt claim. Conversely, X1c is a knife against CREDT's full-promotion claim: if subjects favor the 90% expert over a single coin-flipper at p < 0.47, CREDT (which demands 0.50-0.52) is falsified; and M1/M3 falsify CREDT outright if p(A) >= 0.50/0.52. Knives against BODT: M1 (demands >= 0.54), X2c (demands <= 0.61), O1 (demands >= 0.68), Z2 (demands <= 0.76), K1 (demands >= 0.36).

CONTROLS. N1 (one coin-flipper won by each side) and N2 (the two 0.99 experts split between the options): BOTH theories must predict EXACTLY 0.50 for every parameter -- parameter-free side/display-bias assays; N2 additionally controls any attentional asymmetry induced by the 99% experts. G1 (all five plateau experts vs all three coin-flippers) is a high-confidence agreement anchor (CREDT ~0.81, BODT ~0.85) pinning lapse/temperature. Tie patterns mix 0/0 and 1/1 throughout so 'count the 1s' reduces to raw tallying, which the design defeats on M1/M3/M2 (tally favors A, CREDT predicts B), V1/W1/O2 (exactly tied tallies, large dissociations), K1 and R1 (tally favors A, CREDT predicts B). No all-tie trials; every trial requires scanning all 10 experts to locate 1-8 discriminators.

KEY QUANTITATIVE METRICS: (i) p(A) on the sign-flip cells M1/M1m/M3/M3m/M2/R1/R1m (CREDT .43/.57/.36/.64/.43/.43/.57 vs BODT .56/.44/.62/.38/.56/.52/.48 at centers); (ii) the X-ladder crossover k* (CREDT ~1.0 vs BODT ~1.4) -- a direct readout of the kappa_gate cap; (iii) the R-ladder crossover (~1.4 vs ~0.9) and the R-minus-X crossover difference -- a sign-flipped estimate of w_top in chance-cue units; (iv) the Z-ladder levels (CREDT .67/.80/.88 vs BODT .62/.72/.80); (v) O2/K1 confidence on tally-tied and tally-reversed cells; (vi) parameter-free exact-0.50 cells (N1, N2 for both; T1/T1m/T2 for CREDT) and the parameter-free equality M1 = M2 (CREDT). PREDICTED PATTERN IF CREDT IS RIGHT (centers: w_p=.90, rho=1.36, d_w=.40, beta=.42, eps=.05): M1 .43, M1m .57, M3 .36, M3m .64, M2 .43, V1 .27, V1m .73, W1 .27, W1m .73, O1 .61, O1m .39, O2 .13, O2m .87, R1 .43, R1m .57, R2 .60, X1c .50, X2c .67, X3c .80, Xm3c .20, Z1 .67, Z2 .80, Z2m .20, Z3 .88, K1 .22, K1m .78, T1/T1m/T2 .50 exact, N1/N2 .50 exact, G1 .81 -- subjects anti-follow the 99% endorsements and treat coin-flippers as full experts. IF BODT IS RIGHT (centers: w_floor=-.34, rho_m=.61, kappa_gate=.715, tilt=.08, beta=.395, eps=.0425): M1 .56, M1m .44, M3 .62, M3m .38, M2 .56, V1 .44, V1m .56, W1 .40, W1m .60, O1 .71, O1m .29, O2 .31, O2m .69, R1 .52, R1m .48, R2 .63, X1c .45, X2c .57, X3c .68, Xm3c .32, Z1 .62, Z2 .72, Z2m .28, Z3 .80, K1 .41, K1m .59, T1 .53, T1m .47, T2 .54, N1/N2 .50 exact, G1 .85 -- subjects weakly follow the 99% endorsements and discount coin-flippers to ~70% of an expert. The boxes are disjoint on ~24 of 32 cells and sign-disjoint on 7. DIFFERENTIATION FROM PRIOR EXPERIMENTS: [12]/[13] probed CREDT vs BODT only at gap = 0.05, where both distrust the outlier and the dispute is depth; no prior design instantiated the LARGE-EDGE regime (gap 0.09), where CREDT's narrow s_r plus additive rho plus half-engaged amplifier keeps w_top negative while BODT's bounded saturation recovers it to positive -- a SIGN flip, not a depth difference. The kappa_gate cap on gated chance cues (full promotion vs partial cap) was never exploited in any prior round, and at gap 0.09 the ~0.98-open gate maximizes it. The X-vs-R crossover-difference estimator for w_top and the tied-tally O2/K1 cells are new quantitative signatures. Sides are balanced via mirrors in every family (15 A-tilted / 11 B-tilted / 6 near-neutral under CREDT centers; near-mirrored under BODT centers); no all-tie trials; every trial requires genuine comparison of the ratings.

**Computed schedule:** 32 unique pairs × 3 reps = 96 trials per subject.



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
**Description:** Bounded Outlier-Distrust Tallying (BODT). Subjects integrate all binary expert ratings as sign votes (+1/-1/0) into a zero-sum evidence score, with subjective weights computed RELATIVE to the panel rather than as a monotone function of stated validity: (i) a trust plateau over [v_lo, v_hi] (ceiling just above 0.98) gives all mid-band and 98% experts approximately equal weight; (ii) a small negative chance flank at v near 0.5, consensus-gated so chance cues are counted rather than inverted when an above-ceiling outlier destabilizes the scale; (iii) distrust of an above-ceiling outlier is a BOUNDED, SATURATING shrinkage of its weight toward a floor, peaked in the outlier's uncorroborated edge over the next rung (maximal near gap ~0.05: spurious precision; small but nonzero on dense ladders; receding for large edges on coherent panels); (iv) strongly negative distrust reserved for the composition-amplified regime — an outlier floating above a majority-chance panel. Choice is a low-gain softmax plus a uniform lapse. This round is a pure calibration claim: the only two levers whose slopes have been verified against reliably-signed residuals across the loop (the composition-route amplitude c_amp on Exp6, and the chance-flank magnitude kappa_lo on Exp5) are moved to their anti-oscillation midpoints; every structural parameter is held at its accepted value.

**Parameters:**
- w_p: [0.88, 0.92]
- v_lo: [0.52, 0.54]
- v_hi: [0.9805, 0.9825]
- kappa_lo: [0.20, 0.235]
- g0: [0.041, 0.045]
- s_gap: [0.012, 0.014]
- lambda_ext: [0.35, 0.45]
- g_star: [0.046, 0.050]
- s_l: [0.024, 0.029]
- s_r: [0.062, 0.088]
- w_floor: [-0.40, -0.28]
- rho_m: [0.56, 0.66]
- d_w: [0.08, 0.12]
- g_far: [0.068, 0.082]
- s_far: [0.028, 0.036]
- c_half: [0.44, 0.48]
- s_c: [0.030, 0.040]
- c_amp: [3.8, 4.4]
- kappa_gate: [0.65, 0.78]
- tilt: [0.04, 0.12]
- beta: [0.365, 0.425]
- epsilon: [0.035, 0.05]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Bounded Outlier-Distrust Tallying (BODT), recombination edition.
    # Mechanism family is UNCHANGED from the accepted iter-3 base:
    #   (i) TRUST PLATEAU: all v in [v_lo, v_hi] (v_hi just above
    #       0.98, so a 98% expert stays INSIDE the plateau) get
    #       approximately equal positive weight w_p -- now with a
    #       SHALLOW WITHIN-PLATEAU TILT (the iter-4 round's only
    #       unambiguous structural win, kept per the critic):
    #       w(v) = w_p * (1 - tilt*(v_hi - v)/(v_hi - v_lo)), so a
    #       0.92 cue modestly outranks a 0.75 cue while the band
    #       remains 'approximately equal'.
    #  (ii) CHANCE FLANK + CONSENSUS GATE: v near 0.5 gets a small
    #       negative weight (-kappa_lo at v=0.5, rising linearly to
    #       the tilted plateau value at v_lo). When an above-ceiling
    #       claim is present, the flank is gated: gate = lambda_ext
    #       + (1-lambda_ext)*sig_gap, and the gated promotion is
    #       capped at a PARTIAL ceiling kappa_gate*w_p, so counted
    #       chance cues stay clearly weaker than genuine mid-band
    #       experts.
    # (iii) BOUNDED OUTLIER DISTRUST: the discount of the panel-max
    #       expert is a SATURATING SHRINKAGE toward a floor, never
    #       an unbounded sign flip:
    #         w_top_mod = w_floor + (w_p - w_floor) * exp(-rho_m * P)
    #       with P a PEAKED bump in the rung gap (maximal at gap ~
    #       g_star ~ 0.05; small on dense ladders via s_l; receding
    #       for large edges on coherent panels via s_r). rho_m and
    #       d_w are REVERTED to the midpoint between the flanking
    #       iterations (the iter-4 trims failed their primary target,
    #       Exp 14, while regressing Exps 12/13).
    #  (iv) COMPOSITION-AMPLIFIED ROUTE: comp = c_amp * sig_gap *
    #       far(gap) * gate(c_frac), with the sharpened engagement
    #       region (c_half ~ 0.44-0.48, g_far ~ 0.068-0.082). c_amp
    #       is kept at the RAISED level: its predicted benefits on
    #       the gap-0.05 chance-laden panels (Exps 6/12/13) were
    #       masked, not refuted, by the iter-4 rho_m trim.
    # A small bounded extremity term d_w*(v_top - v_hi)/(1 - v_hi)
    # is retained. Choice is a low-gain softmax (inverse temperature
    # beta) over the two mirrored scores, plus a uniform lapse
    # (epsilon).
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
    lambda_ext = float(parameters["lambda_ext"])
    g_star = float(parameters["g_star"])
    s_l = float(parameters["s_l"])
    s_r = float(parameters["s_r"])
    w_floor = float(parameters["w_floor"])
    rho_m = float(parameters["rho_m"])
    d_w = float(parameters["d_w"])
    g_far = float(parameters["g_far"])
    s_far = float(parameters["s_far"])
    c_half = float(parameters["c_half"])
    s_c = float(parameters["s_c"])
    c_amp = float(parameters["c_amp"])
    kappa_gate = float(parameters["kappa_gate"])
    tilt = float(parameters["tilt"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    def _sigmoid(x):
        # Numerically stable logistic via the identity
        # sigmoid(x) = 0.5 * (1 + tanh(x / 2)).
        return 0.5 * (1.0 + np.tanh(0.5 * x))

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
    # a 0.98-topped panel stays inside the trusted plateau; only
    # 99%+ claims enter the distrust branch.
    outlier = v_top > v_hi

    if outlier:
        sig_gap = _sigmoid((gap - g0) / s_gap)
    else:
        sig_gap = 0.0

    # ---- (ii): consensus gate for the chance flank. ----
    if outlier:
        gate = lambda_ext + (1.0 - lambda_ext) * float(sig_gap)
    else:
        gate = 0.0

    # ---- (i) + (ii): plateau with a shallow within-plateau tilt
    # and a consensus-gated chance flank whose promotion is capped
    # at the PARTIAL ceiling kappa_gate * w_p. ----
    w = np.full(n_features, w_p, dtype=float)
    span = max(v_hi - v_lo, 1e-6)
    plateau = (v >= v_lo) & (v <= v_hi)
    if np.any(plateau):
        # Shallow within-plateau validity gradient: higher-validity
        # experts slightly outrank lower ones (tilt ~ 0.04-0.12),
        # so e.g. a 0.92 cue modestly outweighs a 0.75 cue while
        # the band remains 'approximately equal'.
        w[plateau] = w_p * (
            1.0 - tilt * (v_hi - v[plateau]) / span
        )
    flank = v < v_lo
    if np.any(flank):
        denom = max(v_lo - 0.5, 1e-6)
        # Flank rises linearly to the TILTED plateau value at v_lo,
        # keeping the weight profile continuous at the band edge.
        w_lo_eff = w_p * (1.0 - tilt)
        w_flank = -kappa_lo + (w_lo_eff + kappa_lo) * np.clip(
            v[flank] - 0.5, 0.0, None
        ) / denom
        # With an above-ceiling claim in the panel, chance cues are
        # counted toward a partial ceiling kappa_gate*w_p rather
        # than inverted (and rather than fully trusted).
        w[flank] = (1.0 - gate) * w_flank + gate * kappa_gate * w_p

    # ---- (iii) + (iv): bounded saturating outlier distrust plus
    # the composition-amplified route with its sharpened
    # engagement region. ----
    if outlier:
        # (a) Peaked bump in the uncorroborated edge: maximal near
        # gap ~ g_star (spurious precision), small on dense
        # ladders (left width s_l), receding for large edges on
        # coherent panels (right width s_r).
        width = s_l if gap < g_star else s_r
        P = float(np.exp(-0.5 * ((gap - g_star) / width) ** 2))

        # (b) BOUNDED, SATURATING shrinkage toward the floor:
        # w_top_mod in [w_floor, w_p] for every P.
        w_top_mod = w_floor + (w_p - w_floor) * float(np.exp(-rho_m * P))

        # (c) Composition-amplified route: the c_frac logistic
        # (c_half ~ 0.44-0.48, s_c ~ 0.03-0.04) stays essentially
        # closed below c_frac ~ 0.375 and open above ~0.5; the far
        # logistic centered at g_far ~ 0.068-0.082 stays
        # essentially closed at the spurious-precision gap. c_amp
        # is held at the raised level so the gap-0.05
        # majority-chance panels keep a deeply negative w_top.
        c_frac = float(np.mean(flank))
        comp_gate = _sigmoid((c_frac - c_half) / s_c)
        far = _sigmoid((gap - g_far) / s_far)
        comp = c_amp * float(sig_gap) * float(far) * float(comp_gate)

        # (d) Small bounded extremity term.
        ext = d_w * (v_top - v_hi) / max(1.0 - v_hi, 1e-6)

        w_top = w_top_mod - ext - comp
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    \"\"\"Anti-follow rate of the 99%-expert endorsements.\n\n    On every trial where the two top-validity (0.99) experts cast a\n    net nonzero vote (i.e., they jointly endorse one option), compute\n    the probability that the subject chooses the NON-endorsed option.\n    Under CREDT (negative subjective weight for the uncorroborated\n    0.99 outlier on this chance-laden panel) this rate is well above\n    0.5 (predicted ~0.56-0.65); under BODT (bounded distrust, w_top\n    near zero) it is below 0.5 (predicted ~0.38-0.48). The two\n    predicted boxes are disjoint.\n    \"\"\"\n\n    def _to_vec(cell):\n        if isinstance(cell, np.ndarray):\n            return cell\n        if isinstance(cell, (list, tuple)):\n            return np.asarray(cell, dtype=float)\n        # Fallback: parse string-encoded sequences like \"[1, 0, ...]\"\n        s = str(cell).strip().replace(\"[\", \"\").replace(\"]\", \"\")\n        return np.asarray([float(v) for v in s.split(\",\") if v.strip() != \"\"], dtype=float)\n\n    a = np.vstack([_to_vec(c) for c in data[\"option_a_ratings\"]])\n    b = np.vstack([_to_vec(c) for c in data[\"option_b_ratings\"]])\n    resp = np.asarray(data[\"response\"], dtype=float)\n\n    # Net vote of the two 0.99-validity experts (features 0 and 1,\n    # fixed by this experiment's validity vector).\n    net = np.sign(a[:, 0] - b[:, 0]) + np.sign(a[:, 1] - b[:, 1])\n\n    mask = net != 0\n    if not np.any(mask):\n        return 0.5\n\n    net_m = net[mask]\n    r = np.rint(resp[mask]).astype(int)\n    endorsed_a = net_m > 0  # True -> 0.99 experts endorse A\n\n    # 'against' = subject picked the option the 0.99 experts did NOT endorse\n    against = np.where(endorsed_a, r == 1, r == 0)\n\n    return float(np.mean(against))",
  "rationale": "The single sharpest, most robust discriminator between CREDT and BODT on this panel is the SIGN of the subjective weight assigned to the two 0.99 experts. Working through both predict functions at parameter-box centers and corners: CREDT's unbounded additive distrust plus the half-engaged composition amplifier (c_frac = 0.4 with c_half ~0.34-0.38, gate ~0.98 open) yields w_top in roughly [-1.30, -0.79] across its ENTIRE box, while BODT's bounded saturating shrinkage (w_floor + (w_p-w_floor)*exp(-rho_m*P), with its composition route strangled by comp_gate ~0.15 at c_frac 0.40) yields w_top in roughly [-0.35, +0.17] -- the boxes are disjoint with no overlap. Consequently, on the 18 unique trials (x3 reps = 54 trials per subject) where the two 0.99 experts cast a net vote (trials 1-16 plus 25/26), CREDT predicts subjects choose AGAINST the 99%-endorsed option ~56-65% of the time, whereas BODT predicts ~38-48%. Expected gap ~0.18 in the metric mean. Per-subject variability is modest: binomial noise over 54 trials gives sd ~0.06, parameter heterogeneity adds only ~0.03 (because the sign of w_top never flips within either box), so between-subject variance ~0.005; with N=25 the Welch t-statistic is ~9-10, far below the alpha=0.01 threshold. The metric is a direct behavioral readout (a proportion in [0,1]) that works identically on the pooled DataFrame and on any single subject's slice, is immune to side/display bias because mirror trials (2/4/7/11/13/15/26) contribute symmetrically, and cannot be produced by raw tallying heuristics since it conditions only on the 0.99 experts' votes regardless of the remaining cue pattern. Unlike a theory-difference-weighted composite, it is transparent, parameter-free, and captures the core qualitative dispute: does a lone 99% endorsement on a chance-laden panel push choice toward or against the endorsed option?"
}
```

## Usage

```json
{
  "input_tokens": 13353,
  "output_tokens": 24430,
  "reasoning_tokens": 23384,
  "provider": "Sail Research"
}
```
