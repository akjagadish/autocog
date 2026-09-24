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
**Validities (n_features=11):** [0.99, 0.99, 0.9, 0.85, 0.8, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

**Trial pairs (n=32):**
  trial 1: A=[1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
  trial 2: A=[0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
  trial 3: A=[1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
  trial 4: A=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
  trial 5: A=[1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
  trial 6: A=[0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
  trial 7: A=[1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
  trial 8: A=[0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1]
  trial 9: A=[1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
  trial 10: A=[0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1]
  trial 11: A=[1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0]
  trial 12: A=[0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0]  B=[1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0]
  trial 13: A=[1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0]
  trial 14: A=[1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1]
  trial 15: A=[1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0]
  trial 16: A=[1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1]
  trial 17: A=[1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0]  B=[0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
  trial 18: A=[0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0]  B=[1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0]
  trial 19: A=[1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0]  B=[0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0]
  trial 20: A=[1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1]  B=[0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1]
  trial 21: A=[1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0]
  trial 22: A=[1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]
  trial 23: A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
  trial 24: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  trial 25: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 26: A=[1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1]
  trial 27: A=[1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1]
  trial 28: A=[1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]
  trial 29: A=[1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0]
  trial 30: A=[1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1]
  trial 31: A=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
  trial 32: A=[1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

**Rationale:** CORE LOGIC. EOBOT (advocated) and BODT (competing) share the identical architecture (sign votes x panel-relative weights x low-gain softmax x lapse), the same plateau (w_p in [0.88,0.92] over [v_lo,v_hi], ceiling ~0.981), the same consensus-gating machinery (same g0/s_gap/lambda_ext), and nearly identical chance-flank calibration (kappa_lo 0.19-0.22 vs 0.20-0.235; kappa_gate 0.66-0.76 vs 0.65-0.78). They differ in WHERE distrust of an above-ceiling outlier lives as a function of the rung gap, and in WHERE the composition-amplified 'fraud' route engages. EOBOT's peaked kernel is RELOCATED to g_star ~ 0.088 with a steep left wall (gap-0.05 outliers taken near face value) and its far-fraud route is a sharp logistic at g_far ~ 0.183 that is essentially CLOSED at gap 0.09; its comp_near kernel (centered 0.036) is inert at gap 0.09. BODT's bump peaks at g_star ~ 0.048 (receded to P ~ 0.78-0.90 at gap 0.09, giving only mild base distrust) but its composition route fires through a far logistic centered at g_far ~ 0.068-0.082 that is OPEN (0.57-0.65) at gap 0.09, gated by comp_gate at c_frac 0.545 (0.84-0.97 open). The design therefore engineers the one panel conjunction that maximally separates these two formulas: a 0.99 claim sitting 0.09 above its 0.90 next rung on a MAJORITY-CHANCE panel. Panel: [0.99, 0.99, 0.90, 0.85, 0.80, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50] (11 experts; c_frac = 6/11 = 0.545; two tied 0.99 experts so the disputed lever applies twice). Resulting weight on each 0.99 expert across the ENTIRE parameter boxes: EOBOT w(0.99) in [-0.48, -0.11] (center -0.28: the relocated peak fires, but the bounded saturating shrinkage toward w_floor ~ -0.55 caps the demotion at a MILD anti-vote, and both composition kernels are structurally closed at gap 0.09); BODT w(0.99) in [-2.47, -1.28] (center -1.91: its far gate is open at gap 0.09 and comp_gate is engaged at c_frac 0.545, so the raised c_amp ~ 4.1 route fires fully). THE BOXES ARE DISJOINT BY >= 0.80 WEIGHT UNITS PER 99% EXPERT. Every other weight is common currency: plateau experts (0.90/0.85/0.80) get 0.86-0.91 in both (tilt 0.08-0.12 vs 0.04-0.12, overlapping), and the gated chance weight is 0.56-0.69 (EOBOT) vs 0.55-0.71 (BODT) -- the consensus gate is ~0.98 open at gap 0.09 for both, suppressing the kappa_lo difference to < 0.001. The design is therefore a psychometric isolation of ONE structural disagreement -- whether a 0.09-edge 99% claim floating over a coin-flip-laden panel is a mild anti-cue (EOBOT: worth about half a chance cue AGAINST) or a fraud signal (BODT: worth about three chance cues AGAINST) -- measured in the common currency of chance-cue units, from both sides of 0.50. 32 unique pairs x K = 96//32 = 3 = 96 trials.

FAMILY M (masked endorsement: balanced/tied backgrounds leave s = w(0.99) or 2*w(0.99) alone). M1 (A wins both 99s, all else tied): s = 2*w_top. EOBOT box p(A) in [0.30, 0.46] (center 0.39); BODT box [0.03, 0.15] (center 0.07) -- DISJOINT. M2 (A wins one 99, all else tied): EOBOT [0.40, 0.48] (center 0.44, a near-tie); BODT [0.13, 0.29] (center 0.19) -- DISJOINT. M3 (A wins {99, one coin} vs B wins one coin; the chance cues cancel exactly): both theories must predict M3 = M2 EXACTLY (s = w_top on both surfaces) -- a within-theory surface-invariance check that isolates count- or surface-based heuristics outside both models. Mirrors M1m/M2m/M3m control side bias (e.g. M2m: EOBOT 0.56 vs BODT 0.81). DOUBLE BIND: p(M2) >= 0.35 falsifies BODT outright (its box caps at 0.29); p(M2) <= 0.38 falsifies EOBOT outright (its box floor is 0.40); observed values in the dead zone (0.29, 0.40) strain BOTH theories -- included deliberately as an honest adversarial region.

FAMILY V (head-to-heads). V1 (A wins one 99 vs B wins the 0.90 expert): EOBOT [0.29, 0.40] (center 0.28) vs BODT [0.08, 0.30] (center 0.12) -- disjoint at the edges, centers 16 points apart, on a trial with an exactly tied raw tally and tied 1-count. V2 (A wins one 99 vs B wins one coin-flipper): 'does a COIN-FLIPPER's endorsement beat a 99% expert's?' -- both say yes, but EOBOT box [0.35, 0.40] (center 0.33) vs BODT [0.09, 0.27] (center 0.14) -- DISJOINT confidence. V3 (A wins both 99s vs B wins all three plateau experts): agreement-direction cell with disjoint confidence (EOBOT 0.08 vs BODT 0.02) probing how deep the anti-vote goes inside a fully trusted counter-stack. Mirrors included.

FAMILY R (flagship crossover ladder: one 99 + k coin-flippers vs the 0.90 expert, k = 1..5). s = w_top + k*w_ch - w_p. EOBOT centers: R1 .39, R2 .52, R3 .64, R4 .74, R5 .82 -- crossing p = 0.5 between k = 1 and k = 2 (j* ~ 1.9). BODT centers: R1 .17, R2 .24, R3 .34, R4 .45, R5 .57 -- crossing between k = 4 and k = 5 (j* ~ 4.4). A ~2.5-CHANCE-CUE SEPARATION IN CROSSOVER LOCATION that no parameter setting in either box can close (at k=2 the boxes are [0.43,0.60] vs [0.16,0.36], disjoint; at k=3, EOBOT center .64 vs BODT ceiling .49 -- BODT can never favor the 99+3-coin stack over one 90% expert, EOBOT always does at center). R3 and R4 are SIGN-STRUCTURED cells: 'does a 99% expert backed by 3-4 coin-flippers beat one 90% expert?' EOBOT says yes (.64/.74), BODT says no (.34/.45). R3m mirrors.

FAMILY D (double-99 ladder: both 99s + k coin-flippers vs the 0.90 expert). EOBOT centers: D2(k=2) .47, D3 .59, D4 .70, D5 .79 -- crossing ~k = 2.3. BODT centers: D2 .09, D3 .12, D4 .17, D5 .24 -- NEVER CROSSES (its crossover needs k > 6, outside the design range): under BODT, two 99% endorsements backed by five coin-flippers still lose to a single 90% expert at p ~ 0.24, while EOBOT follows the same stack at p ~ 0.79. D2 is box-disjoint (EOBOT [0.36,0.56] vs BODT [0.05,0.24]); D3/D4 are box-disjoint at the edges and center-sign-flipped by 47-53 points. D4m mirrors.

FAMILY K (unanimity cell). K1 (A wins ALL FIVE genuine experts plus one coin-flipper; B wins five coin-flippers; raw tally 6-vs-5 for A): EOBOT center .41 (near-tie -- the two mild anti-votes nearly cancel the trusted stack) vs BODT center .07 (the double fraud signal sinks the unanimous panel) -- a 34-point center dissociation on the exact surface where any follow-the-experts heuristic is maximal, with the theories' roles REVERSED relative to [12]'s K1 (there the additive-discount theory was the anti-follower at gap 0.05; here at gap 0.09 on a chance-laden panel the bounded-relocation theory is the mild one). K1m mirrors.

FAMILY X/Z (common-currency anchors + the paired endorsement-cost estimator). X2/X3/X4 (k = 2/3/4 coin-flippers vs the 0.90 expert, outliers TIED) and Z2/Z3 (coin-only ladders): both theories agree within ~1 point (X2 .57, X3 .68-.69, X4 .75, Z2 .62-.63, Z3 .72-.73), pinning the shared gated-chance weight w_ch and lapse/temperature -- deviation on these cells indicts the SHARED machinery, not the disputed distrust. THE KEY COMPOSITE METRIC: the paired X-vs-R difference at matched k isolates the 99% endorsement's value in probability units -- EOBOT predicts the endorsement COSTS A only ~0-6 points at every rung (X2 .57 -> R2 .52; X3 .69 -> R3 .64; X4 .75 -> R4 .74), while BODT predicts it costs 30-34 points at every rung (X2 .57 -> R2 .24; X3 .68 -> R3 .34; X4 .75 -> R4 .45). A within-subject endorsement-cost estimate of ~5 vs ~32 points separates the theories by a factor of ~6 and is robust to all shared parameters.

CONTROLS. T1 (the two 99% experts split between A and B) and T2 (one coin-flipper won by each side): BOTH theories must predict EXACTLY 0.50 for every parameter (tied softmax is uniform; the lapse injects uniform) -- parameter-free side/display-bias assays; T1 additionally controls attentional asymmetry induced by the 99% experts. Tie patterns mix 0/0 and 1/1 throughout so 'count the 1s' is unviable: V1/V1m/T1 have exactly tied 1-counts with 16-point dissociations; M1, R2-R5, D2-D5, K1 have MORE 1s and larger raw tallies for A while BODT predicts A loses by 2-4 evidence units; M2/M3 have more 1s for A while both theories favor B. No all-tie trials; every trial requires scanning all 11 experts to locate 1-11 discriminators.

KEY QUANTITATIVE METRICS: (i) masked cells M1/M2 (EOBOT .39/.44 vs BODT .07/.19, box-disjoint); (ii) R-ladder crossover j* (EOBOT ~1.9 vs BODT ~4.4 chance cues) and D-ladder crossover (EOBOT ~2.3 vs BODT out of range > 6); (iii) the paired X-vs-R endorsement cost (~5 vs ~32 points, at every rung k = 2, 3, 4); (iv) sign-structured cells R3/R4/D3/D4/D5 (EOBOT .64/.74/.59/.70/.79 vs BODT .34/.45/.12/.17/.24); (v) V2 confidence (coin-flipper vs 99%: .33 vs .14, box-disjoint); (vi) K1 unanimity (.41 vs .07); (vii) parameter-free exact-0.50 cells (T1, T2) and the M2 = M3 surface equality. PREDICTED PATTERN IF EOBOT IS RIGHT (centers): M1 .39, M2 = M3 .44, V1 .28, V2 .33, V3 .08, R1-R5 .39/.52/.64/.74/.82, D2-D5 .47/.59/.70/.79, K1 .41, X2/X3/X4 .57/.69/.75, Z2/Z3 .62/.73, T1/T2 .50 -- a 99%-atop-90% claim on a coin-flip-laden panel is worth about half a coin-flipper AGAINST: masked near-ties, crossovers at ~2 chance cues, unanimous panels near chance. IF BODT IS RIGHT: M1 .07, M2 = M3 .19, V1 .12, V2 .14, V3 .02, R1-R5 .17/.24/.34/.45/.57, D2-D5 .09/.12/.17/.24, K1 .07, X2/X3/X4 .57/.68/.75, Z2/Z3 .63/.72, T1/T2 .50 -- the same claim is a fraud signal worth ~3 coin-flippers AGAINST: strong masked anti-following, crossovers beyond 4-6 chance cues, unanimous panels rejected. The w(0.99) boxes are disjoint by construction, so the patterns are mutually exclusive on ~20 of 32 cells.

DIFFERENTIATION FROM PRIOR EXPERIMENTS. [12] tested gap 0.05 chance-laden, where EOBOT and BODT nearly AGREE (both w_top ~ -0.4: EOBOT's comp_near fires at the 0.036-centered kernel while its relocated peak is inert, matching BODT's peaked bump plus partially-open far route) -- deliberately avoided here as an uninformative regime. [13] tested gap 0.05 coherent. [14] tested gap 0.09 but with c_frac = 0.30, which kept BODT's comp_gate CLOSED (its w_top was POSITIVE there -- a different sign structure entirely). [15] tested gap 0.14 chance-laden, where BOTH theories' far routes are open. No prior design instantiated the conjunction gap = 0.09 + c_frac = 0.545: the unique configuration in which BODT's far-fraud gate (centered 0.068-0.082) is OPEN while EOBOT's (centered 0.178-0.188, sharp s_far 0.018-0.022) is CLOSED -- a pure test of the LOCATION of the composition route, doubled by EOBOT's relocated peak firing exactly where BODT's bump has receded. The paired X-vs-R endorsement-cost estimator, the out-of-range D-ladder crossover for BODT, and the role-reversed unanimity cell K1 are new quantitative signatures. Sides are balanced via mirrors in every family (18 A-tilted / 11 B-tilted / 3 neutral under EOBOT centers; the B-tilt under BODT centers is itself its signature prediction, controlled by mirrors); no all-tie trials; every trial requires genuine comparison of the ratings.

**Computed schedule:** 32 unique pairs × 3 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Edge-Only Bounded Outlier Tallying (EOBOT), gate-widening calibration edition. Subjects integrate every binary expert rating as a sign vote (+1/-1/0) into a zero-sum evidence score with subjective weights computed RELATIVE to the panel: (i) a trust plateau over [v_lo, v_hi] (ceiling just above 0.98) with a shallow within-plateau tilt; (ii) a small negative chance flank at v near 0.5, consensus-gated with a PARTIAL promotion cap kappa_gate*w_p when an above-ceiling claim destabilizes the scale; (iii) outlier distrust is a BOUNDED, SATURATING shrinkage toward a floor driven by the outlier's uncorroborated EDGE over the next rung — a peaked kernel RELOCATED to gap ~0.088 (so dense ladders and gap-0.05 singleton outliers on trusted panels are taken near face value, while gap-0.09 outliers are deeply distrusted), sitting on a uniform-skepticism floor e_min; (iv) the panel's chance-fraction AMPLIFIES distrust through two split, independently gated kernels: comp_near (its own narrow kernel centered at gap ~0.036, so majority/partial-chance panels deepen distrust only in the small-gap regime) and comp_far (a sharp logistic at gap ~0.18, so a far outlier floating over a chance-laden panel is treated as rigged while a gap-0.14 outlier is followed at face value). This round is a pure calibration claim on the loop-validated structure: the composition gate threshold c_half is LOWERED to [0.34, 0.38] so partially-chance-laden gap-0.05 singleton-99 panels (Exp 12, c_frac ~0.375) actually enter the composition regime (with c_near trimmed to hold the already-perfect double-99 cell, Exp 13, as its gate widens), kappa_lo is interpolated to its empirically bracketed midpoint for the chance-flank sensitivity cell (Exp 5), and e_min is nudged upward for the dense-ladder cell (Exp 9). Choice is a low-gain softmax plus a uniform lapse.

**Parameters:**
- w_p: [0.88, 0.92]
- v_lo: [0.52, 0.55]
- v_hi: [0.9805, 0.9825]
- kappa_lo: [0.19, 0.22]
- kappa_gate: [0.66, 0.76]
- tilt: [0.08, 0.12]
- g0: [0.041, 0.045]
- s_gap: [0.012, 0.014]
- lambda_ext: [0.30, 0.42]
- g_star: [0.084, 0.092]
- s_l: [0.011, 0.015]
- s_r: [0.027, 0.033]
- e_min: [0.12, 0.16]
- w_floor: [-0.58, -0.52]
- rho_e: [1.15, 1.45]
- d_w: [0.08, 0.14]
- g_near: [0.032, 0.040]
- s_near: [0.016, 0.022]
- c_near: [1.9, 2.3]
- c_half: [0.34, 0.38]
- s_c: [0.05, 0.07]
- g_far: [0.178, 0.188]
- s_far: [0.018, 0.022]
- c_far: [4.2, 5.2]
- beta: [0.38, 0.46]
- epsilon: [0.035, 0.055]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Edge-Only Bounded Outlier Tallying (EOBOT), gate-widening edition.
    # Stimulus: array of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are binary expert ratings (0/1).
    # Each feature j casts a sign vote sign(a_j - b_j), weighted by a
    # subjective trust function of the expert's STATED validity v,
    # evaluated RELATIVE to the panel:
    #   (i) TRUST PLATEAU with shallow tilt: all v in [v_lo, v_hi]
    #       (v_hi just above 0.98, so a 98% expert stays INSIDE the
    #       plateau) get approximately equal weight w_p, with a
    #       within-band tilt so higher-validity cues slightly
    #       outrank lower ones.
    #  (ii) CHANCE FLANK + CONSENSUS GATE: v near 0.5 gets a small
    #       negative weight (-kappa_lo at v=0.5, rising linearly to
    #       the tilted plateau at v_lo). When an above-ceiling claim
    #       is present the flank is gated and chance cues are
    #       PROMOTED to a partial ceiling kappa_gate*w_p (counted,
    #       but weaker than genuine mid-band experts).
    # (iii) EDGE-ONLY BOUNDED OUTLIER DISTRUST, RELOCATED PEAK: the
    #       discount of the panel-max expert is a saturating
    #       shrinkage toward a floor,
    #         w_top_mod = w_floor + (w_p - w_floor)*exp(-rho_e*E(gap))
    #       with E a PEAKED kernel centered at g_star ~ 0.088 sitting
    #       on a uniform-skepticism floor e_min: near zero distrust
    #       on dense ladders AND on gap-0.05 singleton outliers
    #       (steep left wall s_l), deep distrust at gap ~0.09 (peak),
    #       receding by gap ~0.14 (right wall s_r).
    #  (iv) SPLIT COMPOSITION AMPLIFICATION, both gated by
    #       comp_gate(c_frac) with threshold c_half ~ 0.36 (LOWERED
    #       this round so partially-chance-laden gap-0.05 panels
    #       enter the regime):
    #         comp_near = c_near * sig_gap * L(gap) * comp_gate
    #       with L a NARROW kernel centered at g_near ~ 0.036 (fires
    #       only in the small-gap regime; inert at gap >= 0.09), and
    #         comp_far  = c_far * sig_gap * far(gap) * comp_gate
    #       with far() a SHARP logistic centered at g_far ~ 0.183:
    #       closed at gap 0.14 (a 0.99 outlier over a 0.85 rung is
    #       followed even on a majority-chance panel), fully open by
    #       gap ~0.24 (a far outlier over a chance-laden panel is a
    #       fraud signal).
    # A small bounded extremity term d_w*(v_top - v_hi)/(1 - v_hi)
    # is retained. All experts tied at the panel maximum share
    # w_top (double-99 panels double the anti-vote). Choice is a
    # low-gain softmax (inverse temperature beta) over the two
    # mirrored scores, plus a uniform lapse (epsilon).
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
    kappa_gate = float(parameters["kappa_gate"])
    tilt = float(parameters["tilt"])
    g0 = float(parameters["g0"])
    s_gap = float(parameters["s_gap"])
    lambda_ext = float(parameters["lambda_ext"])
    g_star = float(parameters["g_star"])
    s_l = float(parameters["s_l"])
    s_r = float(parameters["s_r"])
    e_min = float(parameters["e_min"])
    w_floor = float(parameters["w_floor"])
    rho_e = float(parameters["rho_e"])
    d_w = float(parameters["d_w"])
    g_near = float(parameters["g_near"])
    s_near = float(parameters["s_near"])
    c_near = float(parameters["c_near"])
    c_half = float(parameters["c_half"])
    s_c = float(parameters["s_c"])
    g_far = float(parameters["g_far"])
    s_far = float(parameters["s_far"])
    c_far = float(parameters["c_far"])
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

    # ---- (i) + (ii): tilted plateau with a consensus-gated
    # chance flank whose promotion is capped at the PARTIAL
    # ceiling kappa_gate * w_p. ----
    w = np.full(n_features, w_p, dtype=float)
    span = max(v_hi - v_lo, 1e-6)
    plateau = (v >= v_lo) & (v <= v_hi)
    if np.any(plateau):
        # Shallow within-plateau validity gradient: higher-validity
        # experts slightly outrank lower ones while the band
        # remains 'approximately equal'.
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

    # ---- (iii) + (iv): relocated edge-only bounded distrust with
    # the split, gate-widened composition amplification. ----
    if outlier:
        # (a) Peaked kernel RELOCATED to g_star ~ 0.088, sitting on
        # the uniform-skepticism floor e_min: near-zero distrust on
        # dense ladders (gap ~ 0.01) and on gap-0.05 singleton
        # outliers (steep left wall s_l), deep distrust at the
        # gap ~ 0.09 peak, receding right flank (s_r).
        width = s_l if gap < g_star else s_r
        E_kernel = float(np.exp(-0.5 * ((gap - g_star) / width) ** 2))
        E = max(E_kernel, e_min)

        # (b) BOUNDED, SATURATING shrinkage toward the floor:
        # w_top_mod in [w_floor, w_p] for every E.
        w_top_mod = w_floor + (w_p - w_floor) * float(np.exp(-rho_e * E))

        # (c) Split composition amplification. The chance-fraction
        # gate threshold c_half is LOWERED (~0.36) so PARTIALLY
        # chance-laden panels (c_frac ~ 0.375) enter the regime:
        #   comp_near fires only in the small-gap regime via its own
        #   narrow kernel L (centered g_near ~ 0.036), deepening the
        #   anti-vote for an uncorroborated near-rung outlier on a
        #   chance-laden scale; it is inert at gap >= 0.09.
        #   comp_far is the far-fraud route: a SHARP logistic at
        #   g_far ~ 0.183, closed at gap 0.14, fully open by 0.24.
        c_frac = float(np.mean(flank))
        comp_gate = _sigmoid((c_frac - c_half) / s_c)
        L = float(np.exp(-0.5 * ((gap - g_near) / s_near) ** 2))
        comp_near = c_near * float(sig_gap) * L * float(comp_gate)
        far = _sigmoid((gap - g_far) / s_far)
        comp_far = c_far * float(sig_gap) * float(far) * float(comp_gate)

        # (d) Small bounded extremity term.
        ext = d_w * (v_top - v_hi) / max(1.0 - v_hi, 1e-6)

        w_top = w_top_mod - ext - comp_near - comp_far
        # All experts tied at the panel maximum share the distrust
        # (e.g., a pair of 0.99 super-experts doubles the anti-vote).
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Double-99 endorsement-following rate on the disputed panel configuration.\n    #\n    # The design's single structural disagreement is the weight assigned to\n    # the 0.99 experts on a majority-chance panel with a 0.09 edge over the\n    # 0.90 rung: EOBOT's relocated bounded kernel yields only a MILD anti-vote\n    # (w(0.99) ~ -0.28, about half a chance cue against), while BODT's open\n    # far-fraud composition route yields a deep anti-vote (w(0.99) ~ -1.9,\n    # about three chance cues against). The cleanest behavioral readout of\n    # that single lever is: on trials where ONE option wins BOTH 0.99 experts\n    # (features 0 and 1) and the other wins neither, how often does the\n    # subject follow the double-99 endorsement?\n    #\n    # These are exactly trials 1, 2 (M1), 11, 12 (V3), 19-23 (D-family),\n    # 24, 25 (K1) -- 11 unique trials x 3 reps = 33 observations per subject.\n    # Predicted values: EOBOT ~ 0.45 (masked cells near-tie, D-ladder\n    # followed, unanimous panel near chance), BODT ~ 0.10 (strong anti-\n    # following everywhere). The w(0.99) boxes are disjoint by >= 0.8 weight\n    # units, so no parameter setting in either box can close the gap, while\n    # every shared parameter (w_p, w_ch, beta, epsilon) enters both theories\n    # identically and thus largely cancels in the contrast.\n\n    resp_col = data[\"response\"].to_numpy()\n    a_col = data[\"option_a_ratings\"].to_numpy()\n    b_col = data[\"option_b_ratings\"].to_numpy()\n\n    follow = []\n    for i in range(len(data)):\n        a = np.asarray(a_col[i], dtype=float).ravel()\n        b = np.asarray(b_col[i], dtype=float).ravel()\n        if a.size < 2 or b.size < 2 or a.size != b.size:\n            continue\n\n        # Sign votes on the two 0.99-expert features (design fixes features\n        # 0 and 1 as the tied panel-maximum 0.99 experts).\n        v0 = np.sign(a[0] - b[0])\n        v1 = np.sign(a[1] - b[1])\n\n        # Keep only trials where a single option wins BOTH 99% experts\n        # (the double-99 endorsement cells: M1, V3, D2-D5, K1 and mirrors).\n        if v0 == 0 or v1 == 0 or v0 != v1:\n            continue\n\n        chose_a = (int(resp_col[i]) == 0)\n        if v0 > 0:\n            # A holds the double-99 endorsement.\n            follow.append(1.0 if chose_a else 0.0)\n        else:\n            # B holds the double-99 endorsement (mirror trials).\n            follow.append(0.0 if chose_a else 1.0)\n\n    if not follow:\n        return float(\"nan\")\n    return float(np.mean(follow))",
  "rationale": "METRIC: the rate at which subjects FOLLOW a double-0.99 endorsement on the experiment's disputed panel ([0.99, 0.99, 0.90, 0.85, 0.80, 0.50 x 6]; gap 0.09, c_frac 0.545). Computed over the 11 unique trials (x3 reps = 33 obs/subject) where one option wins both 0.99 experts and the opponent wins neither: M1/M1m (masked, all else tied), V3/V3m (99s vs the full trusted stack), D2-D5 and mirrors (both 99s + k coin-flippers vs the 0.90 expert), K1/K1m (unanimous genuine panel + one coin vs five coins). This is precisely the cell family engineered to isolate the single disjoint lever, w(0.99), in the common currency of choice probability. PREDICTED VALUES: EOBOT ~ 0.45 (centers: M1 .39, V3 .08, D2-D5 .47/.59/.70/.79, K1 .41 -- the bounded saturating shrinkage caps the demotion at a mild anti-vote worth about half a chance cue, so double-99 stacks are near-ties or followed); BODT ~ 0.10 (centers: M1 .07, V3 .02, D2-D5 .09/.12/.17/.24, K1 .07 -- the composition-amplified fraud route is fully open at gap 0.09 / c_frac 0.545, making the 99% endorsement worth ~3 chance cues AGAINST). The w(0.99) boxes are disjoint by >= 0.8 weight units per expert (EOBOT [-0.48,-0.11] vs BODT [-2.47,-1.28]), so the ~0.35 probability gap cannot be closed by any admissible parameter setting of either theory. WHY THE t-TEST PASSES AT N=25: (a) the mean gap is large (~0.35) while per-subject noise is small -- 33 binary observations per subject give binomial variance ~0.006 (EOBOT) / ~0.003 (BODT); (b) parameter heterogeneity contributes only modestly because every SHARED parameter (plateau weight, gated chance weight, softmax gain, lapse) affects both theories' predictions on these same trials identically and the residual w_top-driven spread within each box keeps subject-level means inside roughly [0.40, 0.52] (EOBOT) and [0.05, 0.25] (BODT); combined variance ~0.012 gives t ~ 0.35 / sqrt(0.012/25) ~ 15, far beyond the alpha = 0.01 threshold even with several-fold variance inflation. (c) The metric is well defined on any single subject's slice (33 rows) and on the pooled data, so the between-subject variance is always available. (d) Orientation is computed from the ratings themselves (the side winning both feature-0 and feature-1 votes), so mirror trials are handled automatically and no side-bias or display-position confound can mimic the effect: a pure side bias cancels across the A-tilted and B-tilted mirrors, and a 'count the 1s' heuristic is uninformative here because M1/V3 have exactly tied 1-counts while K1 has MORE 1s for the 99-side that BODT says subjects reject. INTERPRETATION: values >= ~0.35 support EOBOT (mild bounded distrust of a 0.09-edge 99% claim over a coin-laden panel); values <= ~0.25 support BODT (composition-amplified fraud signal); the dead zone in between strains both."
}
```

## Usage

```json
{
  "input_tokens": 14428,
  "output_tokens": 18638,
  "reasoning_tokens": 17019,
  "provider": "Sail Research"
}
```
