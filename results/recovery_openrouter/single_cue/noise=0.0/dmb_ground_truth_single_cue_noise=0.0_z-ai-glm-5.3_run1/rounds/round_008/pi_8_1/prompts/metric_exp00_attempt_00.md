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
**Validities (n_features=10):** [0.99, 0.99, 0.9, 0.85, 0.8, 0.75, 0.7, 0.5, 0.5, 0.5]

**Trial pairs (n=32):**
  trial 1: A=[1, 1, 1, 1, 0, 0, 1, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1, 0, 1, 0, 0]
  trial 2: A=[0, 0, 1, 1, 0, 0, 1, 1, 0, 0]  B=[1, 1, 1, 0, 1, 1, 0, 1, 0, 0]
  trial 3: A=[1, 1, 1, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
  trial 4: A=[1, 0, 1, 1, 0, 0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1, 1, 0, 0, 1, 0]
  trial 5: A=[0, 0, 1, 1, 0, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 1, 0, 0, 1, 0]
  trial 6: A=[1, 0, 1, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
  trial 7: A=[1, 0, 1, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 1, 1, 0]
  trial 8: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 1, 1]
  trial 9: A=[0, 0, 1, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 10: A=[1, 1, 1, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 0, 0, 1, 1, 0]
  trial 11: A=[1, 1, 0, 0, 0, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 0, 0, 1, 1, 0]
  trial 12: A=[0, 0, 1, 0, 0, 0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1, 1, 0]
  trial 13: A=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
  trial 14: A=[1, 1, 0, 0, 0, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
  trial 15: A=[0, 0, 1, 0, 0, 0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1, 1, 0]
  trial 16: A=[1, 1, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
  trial 17: A=[0, 0, 1, 0, 0, 0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1, 1, 1]
  trial 18: A=[1, 1, 0, 1, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  trial 19: A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 1, 0, 0, 0, 1, 0, 0]
  trial 20: A=[1, 1, 1, 1, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  trial 21: A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 1, 1]
  trial 22: A=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 1, 0, 0]
  trial 23: A=[0, 0, 1, 1, 0, 0, 0, 1, 0, 0]  B=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
  trial 24: A=[1, 1, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 1, 0, 0, 0, 1, 0, 0]
  trial 25: A=[0, 0, 1, 1, 0, 0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1, 1, 1]
  trial 26: A=[1, 1, 0, 0, 0, 0, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 1, 0, 0]
  trial 27: A=[1, 1, 0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 1, 0, 0]
  trial 28: A=[1, 1, 0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 1, 0, 0]
  trial 29: A=[1, 0, 1, 0, 0, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 1, 0, 0]
  trial 30: A=[0, 0, 1, 0, 0, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 0, 0, 1, 0, 1]
  trial 31: A=[1, 1, 1, 1, 0, 0, 1, 0, 0, 0]  B=[1, 1, 1, 0, 1, 1, 0, 0, 0, 0]
  trial 32: A=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 1, 1]

**Rationale:** CORE LOGIC. BODT (advocated) and EOBOT (competing) share the identical architecture (sign votes x panel-relative weights x low-gain softmax x lapse), the same plateau (w_p in [0.88,0.92] over [v_lo,v_hi], ceiling ~0.981), the same consensus-gating machinery (same g0/s_gap/lambda_ext), and near-identical chance-flank calibration (kappa_lo 0.20-0.235 vs 0.19-0.22; kappa_gate 0.65-0.78 vs 0.66-0.76). They disagree about WHERE above-ceiling distrust lives as a function of the rung gap: BODT's bump peaks at g_star ~ 0.048 with a SHALLOW saturating shrinkage (rho_m 0.56-0.66 toward w_floor ~ -0.34); EOBOT's kernel is RELOCATED to g_star ~ 0.088 with a DEEPER shrinkage (rho_e 1.15-1.45 toward w_floor ~ -0.55) on an e_min floor. The prior head-to-head [16] tested gap 0.09 only on a MAJORITY-CHANCE panel (c_frac 0.545), where BODT's composition route fired (w_top ~ -1.9) and EOBOT's was inert (~ -0.2) -- both negative, an amplitude dissociation. This design instantiates the one untested conjunction: gap = 0.09 on a COHERENT panel (c_frac = 2/10 = 0.20), where BOTH composition routes are structurally silenced (BODT comp_gate = sigmoid((0.20-0.46)/0.035) ~ 0.0006; EOBOT gate ~ 0.065 with its comp_near kernel inert at gap 0.09 and comp_far closed), leaving the PURE kernel-location disagreement. Panel: [0.99, 0.99, 0.90, 0.85, 0.80, 0.75, 0.70, 0.55, 0.50, 0.50] -- two tied 0.99 experts (doubling the disputed lever) sitting exactly 0.09 above the 0.90 rung. Resulting weight on each 99% expert across the ENTIRE parameter boxes: BODT w(0.99) in [+0.25, +0.45] (its bump has receded to P in [0.81,0.88] and the shallow saturation caps the demotion -- a weak POSITIVE, ~40% of a plateau cue); EOBOT w(0.99) in [-0.34, -0.09] (its relocated kernel is at its PEAK, E ~ 0.98-1.0, and the deeper shrinkage demotes it below zero -- a mild NEGATIVE). THE w_top BOXES ARE SIGN-DISJOINT with a 0.34-unit gap between edges -- the mirror image of [16], where both were negative. Every other weight is common currency: plateau experts (0.55-0.90) get 0.77-0.91 in both (tilt 0.04-0.12 vs 0.08-0.12, overlapping), and because sig_gap(0.09) ~ 0.98 the consensus gate is ~0.98 open for both, giving gated chance weight 0.56-0.74 (BODT) vs 0.57-0.71 (EOBOT) -- overlapping. The experiment is therefore a psychometric isolation of ONE signed scalar -- the value of a 99%-atop-90% endorsement on a coherent panel -- measured in plateau-cue and chance-cue units from both sides of 0.50. 32 unique pairs x K = 96//32 = 3 = 96 trials.

FAMILY M (masked endorsement; tilt-cancelled backgrounds). Because both theories' plateau weights are affine in v, the background {0.85, 0.70} vs {0.80, 0.75} (equal validity sums 1.55) cancels to machine precision, leaving s = w(0.99) or 2*w(0.99) exactly. M1 (A wins both 99s + background): BODT box p(A) in [0.586, 0.675] (center 0.63); EOBOT box [0.359, 0.469] (center 0.42) -- SIGN-DISJOINT with a dead zone (0.469, 0.586) that falsifies BOTH theories. M2 (single 99): BODT [0.543, 0.590] vs EOBOT [0.428, 0.492] -- sign-disjoint, dead zone (0.492, 0.543). M1lone/M2lone (99s alone, everything else tied) share the exact predictions, giving the parameter-free surface-invariance equalities M1 = M1lone and M2 = M2lone that both theories demand; inequality indicates a count/surface heuristic outside both models. Mirrors M1m/M2m control side bias.

FAMILY R (flagship crossover ladder: both 99s + k coin-flippers vs the 0.90 expert). R1 (k=1): BODT box [0.526, 0.654] (center 0.58) vs EOBOT [0.295, 0.429] (center 0.37) -- SIGN-DISJOINT with dead zone (0.429, 0.526): 'do two 99% experts backed by one coin-flipper beat one 90% expert?' BODT says yes for every parameter; EOBOT says no for every parameter. R2 (k=2): BODT [0.620, 0.774] vs EOBOT [0.406, 0.586] -- disjoint. The R-ladder crossover k* (chance cues needed to make the double-99 stack beat one 90% expert) is BODT ~0.4 vs EOBOT ~2.0 -- a ~1.6-cue separation no parameter setting can close. Mirrors R1m/R2m are also box-disjoint and reverse the pattern.

FAMILY S/W/V (head-to-heads). S90 (both 99s vs the 0.90 expert): BODT [0.429, 0.505] (near-tie) vs EOBOT [0.205, 0.320]; S90m (one 90% expert vs both 99s): BODT [0.499, 0.588] vs EOBOT [0.710, 0.795] -- disjoint with a sign-structured pair: BODT says 90%-vs-double-99 is a coin flip; EOBOT says the 90% expert wins clearly. S55: BODT [0.434, 0.521] vs EOBOT [0.216, 0.307]. W1 (one 99 vs the 55% expert): BODT [0.392, 0.442] vs EOBOT [0.266, 0.322]. V1 (one 99 vs one coin-flipper): BODT [0.416, 0.477] vs EOBOT [0.289, 0.382]; V2 (both 99s vs both coin-flippers, raw tally tied): BODT [0.337, 0.454] vs EOBOT [0.148, 0.245]. All box-disjoint; on V1/V2 BOTH theories say the coin-flipper(s) beat the 99%(s), but at disjoint confidences -- pure amplitude readouts of the sign-flipped w_top.

FAMILY O/K/D (stacks and coalitions). O1 (both 99s + the 55% expert vs the 0.90 expert): BODT [0.562, 0.678] vs EOBOT [0.332, 0.459] -- sign-disjoint. K1 (A wins {99,99,90,85} vs B wins {80,75,70,55}; raw tally and 1-count EXACTLY TIED): BODT [0.281, 0.393] vs EOBOT [0.114, 0.204] -- a count-balanced cell with disjoint boxes, defeating any tally heuristic. D1 (both 99s vs {90,55}, tally tied): BODT [0.291, 0.354] vs EOBOT [0.119, 0.179]. D2 (both 99s + both coin-flippers vs {90,85}): BODT [0.465, 0.632] (center 0.54) vs EOBOT [0.240, 0.401] (center 0.32) -- sign-disjoint at center. Mirrors for every cell control side bias.

HONEST KNIVES BOTH WAYS. Against MY theory (BODT): observed p(M2) <= 0.49, p(M1) <= 0.50, p(R1) <= 0.49, p(O1) <= 0.50, or p(S90m) >= 0.60 falsifies BODT outright with zero free parameters (its w_top cannot drop below +0.25 on this panel -- the shallow saturation is the claim). Against EOBOT: observed p(M2) >= 0.50, p(M1) >= 0.52, p(R1) >= 0.45, p(S90m) <= 0.65, p(O1) >= 0.50, p(K1) >= 0.25, or p(D1) >= 0.22 falsifies EOBOT outright (its kernel-at-peak plus deep saturation cannot lift w_top above -0.09 here). The dead zones on M1 (0.469-0.586), M2 (0.492-0.543) and R1 (0.429-0.526) falsify BOTH theories -- included deliberately.

CONTROLS/ANCHORS. X1/X2 (k = 1/2 coin-flippers vs the 0.90 expert, 99s TIED so w_top contributes exactly zero): both theories agree (X1 ~ 0.45, X2 ~ 0.57, boxes overlapping) -- the shared kappa_gate currency; deviation indicts the COMMON gating machinery, not the disputed distrust. Z2 (both coin-flippers alone): both ~ 0.72-0.73. G1 (all six plateau experts vs both coin-flippers): both ~ 0.92, pinning lapse/temperature. T1 (the two 99s split), T2 (one coin-flipper each), and T3 ({85,70} vs {80,75}, equal validity sums): BOTH theories must predict EXACTLY 0.50 for every parameter -- parameter-free side/display-bias assays. THE KEY COMPOSITE METRIC: the paired X-vs-R endorsement cost at matched k -- adding the double-99 endorsement to A's side shifts p by +13 points at k=1 and +12 at k=2 under BODT (0.453->0.583; 0.574->0.693) but by -8 points under EOBOT (0.449->0.368; 0.569->0.491): a within-subject, sign-opposite estimator of w_top robust to all shared parameters.

KEY QUANTITATIVE METRICS: (i) masked cells M1/M2 (0.63/0.57 vs 0.42/0.46, box-disjoint); (ii) R-ladder crossover k* (~0.4 vs ~2.0 chance cues) against the shared X-ladder crossover (~1.5 for both); (iii) the paired endorsement cost (+12-13 vs -8 points); (iv) sign-structured cells R1/O1/D2 and the S90/S90m pair; (v) count-tied cells K1/D1/V2; (vi) parameter-free exact-0.50 cells (T1/T2/T3) and surface equalities (M1=M1lone, M2=M2lone). PREDICTED PATTERN IF BODT IS RIGHT (centers): M1/M1lone .63, M1m .37, M2/M2lone .57, M2m .44, V1 .44, V2 .39, V2m .61, S55 .47, S90 .46, S90m .54, W1 .41, R1 .58, R1m .42, R2 .69, R2m .31, O1 .62, O1m .38, K1 .33, K1m .67, D1 .32, D1m .68, D2 .54, D2m .46, X1 .45, X2 .57, Z2 .72, T1/T2/T3 .50, G1 .92 -- a 99%-atop-90% endorsement on a coherent panel is worth ~40% of a trusted cue FOR: near-ties against single experts, crossovers at <1 coin-flipper, stacks followed. IF EOBOT IS RIGHT: M1/M1lone .42, M1m .58, M2/M2lone .46, M2m .54, V1 .34, V2 .21, V2m .79, S55 .27, S90 .26, S90m .74, W1 .31, R1 .37, R1m .63, R2 .49, R2m .51, O1 .40, O1m .60, K1 .16, K1m .84, D1 .16, D1m .84, D2 .32, D2m .68, X1 .45, X2 .57, Z2 .73, T .50, G1 .92 -- the same endorsement is worth ~20% of a cue AGAINST: anti-following everywhere, crossovers at ~2 coin-flippers, the 90% expert beating the double-99 stack at 0.74. The w_top boxes are sign-disjoint by construction, so the patterns are mutually exclusive on 25 of 32 cells.

DIFFERENTIATION FROM PRIOR EXPERIMENTS. [16] tested this exact theory pair at gap 0.09 but on a MAJORITY-CHANCE panel (c_frac 0.545), where BODT's composition route fired and BOTH theories predicted anti-following (an amplitude dissociation, w_top -1.9 vs -0.2); no prior design instantiated gap 0.09 with BOTH composition routes structurally closed, where the disagreement reduces to a pure SIGN flip of the kernel-location claim (BODT's receded bump + shallow saturation vs EOBOT's kernel at its peak + deep saturation). [14] used gap 0.09 against a different competitor (CREDT) at c_frac 0.30; [12]/[13] gap 0.05; [15] gap 0.14. The tilt-cancelled masked architecture (equal-validity-sum backgrounds that cancel exactly under any affine tilt), the paired X-vs-R endorsement-cost estimator, and the S90/S90m sign-structured pair are new quantitative signatures. Sides are balanced via mirrors in every family; no all-tie trials; tie patterns mix 0/0 and 1/1 so 'count the 1s' is unviable (K1/D1/V2/T3 have exactly tied tallies and 1-counts with 12-17-point dissociations; M1/M1lone/R1/O1 have MORE 1s for A while EOBOT predicts A loses); every trial requires scanning all 10 experts to locate 1-6 discriminators; the two 99% experts are displayed first (validity-salient), an adversarial touch favoring any follow-the-best-expert heuristic that would push against EOBOT's anti-following signature.

**Computed schedule:** 32 unique pairs × 3 reps = 96 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n\n    def _to_mat(col):\n        vals = list(col)\n        rows = [np.asarray(v, dtype=float).ravel() for v in vals]\n        return np.array(rows, dtype=float)\n\n    a = _to_mat(data['option_a_ratings'])\n    b = _to_mat(data['option_b_ratings'])\n    resp = np.asarray(data['response'], dtype=float).ravel()\n\n    # The two 99%-validity experts occupy features 0 and 1 of the fixed\n    # validity vector [0.99, 0.99, 0.90, 0.85, 0.80, 0.75, 0.70, 0.5, 0.5, 0.5].\n    # Net vote of the double-99 endorsement on this trial:\n    #   +2 / +1 : the 99(s) favor A,  -2 / -1 : favor B,  0 : tied/split.\n    e = (a[:, 0] - b[:, 0]) + (a[:, 1] - b[:, 1])\n\n    mask = e != 0\n    if not np.any(mask):\n        return 0.0\n\n    e_m = e[mask]\n    r_m = resp[mask]  # 0 = chose A, 1 = chose B\n\n    # \"Follow\" = the chosen option is the one the 99% experts endorse.\n    follow = np.where(e_m > 0, 1.0 - r_m, r_m)\n\n    return float(np.mean(follow))",
  "rationale": "METRIC: the double-99 FOLLOW INDEX \u2014 over every trial where the two 99%-validity experts cast a net vote for one option (75 of 96 trials per subject; only the 99-tied/split cells X1/X2/Z2/T1/T2/T3/G1 are dropped), the rate at which the subject's choice goes in the direction the 99% experts favor. It is a pure behavioral summary (no model fitting), works identically on the pooled frame and on a single subject's slice, and is automatically side-bias-immune because every cell has a mirror and 'follow' is defined relative to the 99-direction, not to A/B.\n\nWHY IT MAXIMALLY SEPARATES THE TWO THEORIES. On this panel (v_top=0.99, next rung 0.90, gap=0.09, c_frac=0.30) both theories' composition routes are structurally closed and every shared quantity (plateau weights 0.84-0.92, gated chance weight ~0.56-0.74, beta, lapse) overlaps. The ONLY live disagreement is the sign of w(0.99): BODT's receded bump + shallow saturation gives w(0.99) in [+0.25, +0.45] (verified at box corners: +0.27 to +0.39); EOBOT's relocated peak (g_star~0.088) + deep shrinkage gives w(0.99) in [-0.34, -0.09] (corners: -0.30 to -0.12). The boxes are SIGN-DISJOINT, so the follow index inherits a sign-opposite gap from every one of the 25 discriminating unique cells: tilt-cancelled masked cells (s = 2*w99 or w99 exactly: M1/M1lone/M2/M2lone), the R-ladder, S90, V1/V2, W1, D1/D2, K1. Expected values (parameter-box centers, computed from both predict() functions): BODT follow rate ~0.55 (range ~0.52-0.58 across its full box); EOBOT ~0.34 (range ~0.30-0.37). Mean gap ~0.21-0.22, and no parameter setting in either box can close it because the gap's sign is fixed by the sign-disjoint w_top boxes.\n\nWHY IT PASSES THE N=25 VARIANCE GATE. Per-subject binomial noise: 75 trials, mean p(1-p) ~ 0.24, so sd ~ sqrt(0.24/75) ~ 0.057. Parameter heterogeneity across the boxes moves the index by only ~0.02-0.03 sd (the w99 lever shifts each double-99 cell by ~0.04 in probability; w_coin/beta/tilt shifts are smaller and partly cancel across the mirrored, tilt-cancelled cells). Total per-subject sd ~ 0.06 on each side, giving Welch t ~ 0.217 / sqrt(2*0.0036/25) ~ 12 at N=25 \u2014 orders of magnitude below alpha=0.01, with the two distributions (0.52-0.58 vs 0.30-0.37) not even overlapping. This is far safer than any single-cell or paired-difference metric (e.g., M1 alone or the X-vs-R endorsement cost uses only 3-6 reps per subject, per-subject sd ~ 0.16-0.29, t ~ 2.5 \u2014 marginal); pooling all 25 discriminating cells triples the effective trial count while the per-cell gaps (0.11-0.21) all point the same way, since every cell's discrimination reduces to the same signed scalar w(0.99)."
}
```

## Usage

```json
{
  "input_tokens": 14204,
  "output_tokens": 22595,
  "reasoning_tokens": 21382,
  "provider": "Sail Research"
}
```
