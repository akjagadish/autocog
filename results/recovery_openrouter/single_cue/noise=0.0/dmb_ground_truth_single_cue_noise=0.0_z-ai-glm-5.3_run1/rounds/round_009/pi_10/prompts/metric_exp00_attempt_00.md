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
**Validities (n_features=11):** [0.99, 0.8, 0.75, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

**Trial pairs (n=32):**
  trial 1: A=[1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  trial 2: A=[1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0]
  trial 3: A=[1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1]
  trial 4: A=[1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0]
  trial 5: A=[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0]
  trial 6: A=[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0]  B=[1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  trial 8: A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 9: A=[1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0]
  trial 10: A=[1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0]
  trial 11: A=[1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 12: A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]
  trial 13: A=[1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
  trial 14: A=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
  trial 15: A=[1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0]
  trial 16: A=[1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0]
  trial 17: A=[0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0]  B=[1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0]
  trial 18: A=[1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]  B=[0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1]
  trial 19: A=[1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
  trial 20: A=[1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0]  B=[0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0]
  trial 21: A=[1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1]  B=[0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1]
  trial 22: A=[0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0]  B=[1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0]
  trial 23: A=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  trial 24: A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  trial 25: A=[1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1]  B=[1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1]
  trial 26: A=[1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0]
  trial 27: A=[1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0]
  trial 28: A=[1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0]  B=[1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0]
  trial 29: A=[1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0]  B=[1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0]
  trial 30: A=[0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1]
  trial 31: A=[1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1]  B=[1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1]
  trial 32: A=[1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0]

**Rationale:** CORE LOGIC. SG-CGD (advocated) and EOBOT (competing) share the identical architecture (sign votes x panel-relative weights x low-gain softmax x lapse), the same plateau (w_p in [0.88,0.92] over [v_lo,v_hi], ceiling ~0.981), the same consensus-gating machinery (same g0/s_gap/lambda_ext), and near-identical chance-flank calibration (kappa_lo 0.19-0.22 both; kappa_gate 0.66-0.74 vs 0.66-0.76). Every prior round tested this family at gap <= 0.14. The single untested conjunction -- and the widest structural separation between these two theories -- is the FAR-OUTLIER regime: a 0.99 claim floating 0.19 above its next rung on a majority-chance panel. There, both theories' far-fraud logistics are equally half-open (both centered ~0.183, F in [0.50,0.66] at gap 0.19), so the dissociation isolates ONE parameter: the far-route AMPLITUDE. SG-CGD's far-fraud route carries c_far in [20,26] with its own chance gate (c_half_far 0.57-0.63, ~0.77-0.93 open at c_frac 0.727); EOBOT's carries c_far in [4.2,5.2] through its shared comp_gate (~1.0 open at c_frac 0.727). Panel: [0.99, 0.80, 0.75, 0.50 x 8] (11 experts; gap = 0.19; c_frac = 8/11 = 0.727). Resulting weight on the 0.99 expert across the ENTIRE parameter boxes: SG-CGD w(0.99) in [-14.1, -5.8] (center -9.5: the far-fraud route fires at 8-15 chance-cue units AGAINST, swamping the +0.5 base trust, the +1.1 credibility ramp, and the -0.05 extremity term); EOBOT w(0.99) in [-2.92, -0.48] (center -2.15). THE BOXES ARE DISJOINT BY ~2.9 WEIGHT UNITS. Every other weight is common currency: w(0.80) ~ 0.86, w(0.75) ~ 0.85 (tilt 0.09-0.13 vs 0.08-0.12, overlapping), and because sig_gap(0.19) ~ 1 the consensus gate is fully open for both, giving gated chance weight 0.58-0.68 (SG) vs 0.58-0.70 (EOBOT) -- overlapping. Both near-kernels are structurally inert at gap 0.19 (L ~ 0). The experiment is therefore a psychometric isolation of ONE scalar -- the far-fraud amplitude -- measured in the common currency of chance cues (~0.63) from both sides of 0.50. 32 unique pairs x K = 96//32 = 3 = 96 trials.

FAMILY R (flagship crossover ladder: the 0.99 endorsement + k coin-flippers vs the 0.80 expert, k = 0,2,4,5,6,8). EOBOT centers: R0 .09, R2 .20, R4 .41, R5 .54, R6 .65, R8 .83 -- a psychometric CROSSING p = 0.5 at k* ~ 4.8 chance cues. SG-CGD centers: .02/.02/.02/.02/.02/.02 -- a FLAT LAPSE FLOOR at every rung (its crossover needs k ~ 16, far outside any feasible panel). The crossover separation exceeds 10 chance cues -- the largest any round has achieved -- and no parameter setting in either box can close it (at R5, SG's box tops out at 0.09 while EOBOT's floor is 0.25; at R8, SG <= 0.30 vs EOBOT >= 0.63). R5/R6/R8 are SIGN-FLIP cells: 'does a 99% expert backed by 5-8 coin-flippers beat one 80% expert?' EOBOT says yes; SG-CGD says no by 8+ evidence units. Mirrors R5m/R8m reverse the pattern (SG .98 vs EOBOT .17/.46).

FAMILY M (masked endorsement + the tightest cell in the design). M1 (A wins ONLY the 0.99 expert, all else tied): s = w_top exactly. SG-CGD box p(A) in [0.024, 0.030] -- a NEAR-PARAMETER-FREE FLOOR (the far route saturates the softmax for every admissible parameter; only the lapse remains). EOBOT box [0.088, 0.413] (center 0.16). DISJOINT with a dead zone (0.03, 0.09) that falsifies BOTH theories -- included deliberately. M2 (A wins {0.99, one coin} vs B wins one coin) reduces to the same s = w_top: both theories demand the parameter-free equality M1 = M2, isolating any count/surface heuristic outside both models. M1m mirrors (SG .98 vs EOBOT .84).

FAMILY D (double counterweight ladder: 0.99 + k coin-flippers vs BOTH plateau experts). EOBOT centers: D5 .36, D6 .49, D8 .61 (crossover k* ~ 6.1); SG-CGD: .02 at every rung (box at D8 <= 0.11 vs EOBOT >= 0.48). D6m mirrors.

FAMILY V (head-to-heads, all with tied or near-tied raw tallies). V1 (0.99 vs one coin-flipper): SG .02 vs EOBOT .11 -- both say the coin-flipper wins, at disjoint confidences (the amplitude question in its purest form). V1m: .98 vs .89. V2 (0.99+0.75 vs 0.80): SG .02 vs EOBOT .15. V3 (0.99 vs 0.75): SG .02 vs EOBOT .09, with exactly tied 1-counts.

FAMILY O/K (stacks and the unanimity reversal). O1 (unanimous 99+80+75 endorsement vs one coin-flipper): SG .02 vs EOBOT .30 -- EOBOT says a unanimous genuine-expert panel LOSES to a single coin-flipper; SG says it loses catastrophically. O2 (vs three coin-flippers): .02 vs .11. O1m mirrors. K1 (A wins ALL THREE genuine experts plus FOUR coin-flippers vs B's four coin-flippers; raw tally 7-vs-4 and 1-counts 7-vs-4 for A): both theories favor B, but at .02 (SG) vs .41 (EOBOT) -- a 39-point confidence dissociation on the surface where any follow-the-experts or count-the-1s heuristic is maximal. K1m mirrors.

HONEST KNIVES BOTH WAYS. Against MY theory (SG-CGD): its M1 box spans only [0.024, 0.030], so ANY reliable following of a lone far-floating 99% endorsement (p >= 0.04) falsifies the c_far 20-26 amplitude outright; likewise p(R8) >= 0.35 (its box caps at 0.30 even when all eight coin-flippers back the 99 against one 80% expert) or p(K1) >= 0.15. Against EOBOT: p(M1) <= 0.08, p(R5) <= 0.09, p(R6) <= 0.14, p(R8) <= 0.60, or p(D8) <= 0.30 falsifies it with zero free parameters (its far route cannot exceed ~2.9 chance-cue units). The dead zones on M1 (0.03-0.09), R5 (0.09-0.25) and R6 (0.14-0.36) falsify BOTH theories -- included deliberately as honest adversarial regions.

CONTROLS/ANCHORS. X1/X2/X3 (k = 1/2/3 coin-flippers vs the 0.80 expert with the 99 TIED, so w_top contributes exactly zero) and Z2, G2 (chance-only stacks, 99 tied): both theories agree within ~1-2 points (X: .46/.58/.69; Z2 .62-.63; G2 .62-.63), pinning the SHARED gated-chance currency and beta/epsilon -- deviation on these cells indicts the common gating machinery, not the disputed far amplitude. G1 (both plateau experts + two coins vs two coins): agreement anchor ~.79-.80. T1/T2 (one/two coin-flippers won by each side, equal 1-counts): BOTH theories must predict EXACTLY 0.50 for every parameter -- parameter-free side/display-bias assays. Tie patterns mix 0/0 and 1/1 throughout so 'count the 1s' is unviable: M1 has MORE 1s for A (6 vs 5) while both theories predict A loses by 6-9 evidence units; K1 has a 7-vs-4 cue and 1-count sweep for A with the same prediction; V3 and T1/T2 have exactly tied 1-counts; R0-R8, D5-D8, O1-O2 have larger tallies for A while SG predicts A loses. No all-tie trials; every trial requires scanning all 11 experts to locate 1-9 discriminators; the 99% expert is displayed first (validity-salient), an adversarial touch favoring any follow-the-best-expert heuristic, which pushes against both theories' anti-following signature and most strongly against SG-CGD's extreme one.

KEY QUANTITATIVE METRICS: (i) the R-ladder crossover k* (EOBOT ~4.8 vs SG-CGD out of range > 10 -- a > 2x separation in chance-cue units); (ii) the masked floor p(M1) (SG 0.024-0.030, a near-parameter-free point prediction, vs EOBOT 0.09-0.41); (iii) sign-flip cells R5/R6/R8 and D8 (centers .02 vs .54/.65/.83 and .61); (iv) the unanimity cell K1 (.02 vs .41) and stack cell O1 (.02 vs .30); (v) confidence on V1/V2/V3 where both anti-follow but at disjoint depths; (vi) parameter-free exact-0.50 cells (T1, T2) and the surface equality M1 = M2; (vii) the anchor cells X/Z/G that verify the shared machinery before interpreting the disjoint cells. PREDICTED PATTERN IF SG-CGD IS RIGHT (centers): every 99-discriminating trial sits at the lapse floor (p(A) ~ .02 when A wins the 99, ~ .98 when B does), the R- and D-ladders are FLAT, and the anchors land at X1/X2/X3 .46/.58/.69, Z2 .62, G1 .79, G2 .62, T .50. IF EOBOT IS RIGHT: a crossing R-psychometric (.09/.20/.41/.54/.65/.83), a crossing D-psychometric (.36/.49/.61), masked anti-following at .16, near-tie unanimity K1 .41, O1 .30, V1 .11, and the same anchors. The w(0.99) boxes are disjoint by construction, so the patterns are mutually exclusive on 20 of 32 cells.

DIFFERENTIATION FROM PRIOR EXPERIMENTS. [15] tested gap 0.14 chance-laden but for BODT vs CREDT (far amplitudes 2.8 vs 1.0 -- both theories' far routes open, a modest amplitude ratio); [16] gap 0.09 c_frac 0.545 and [17] gap 0.09 c_frac 0.20 for EOBOT vs BODT; [18] tested EOBOT vs SG-CGD only at gap 0.09 c_frac 0.692 (credibility-ramp vs peak-distrust sign flip). No prior design instantiated gap >= 0.18, the unique regime in which SG-CGD's c_far 20-26 route fires at full gain (8-15 chance-cue units) while EOBOT's c_far 4.2-5.2 route yields a WITHIN-RANGE crossover -- producing a > 10-chance-cue crossover separation, a parameter-free lapse-floor prediction on the masked cell, and a unanimity cell where a 7-cue sweep is predicted to lose. The far-gate location being ~identical across theories (both ~0.183) makes this a pure AMPLITUDE test, uncontaminated by gate-position differences. Sides are balanced via mirrors in every family (12 A-tilted / 18 B-tilted / 2 neutral under SG centers -- the B-tilt excess is itself SG-CGD's signature prediction, controlled by mirrors; near-balanced under EOBOT centers); no all-tie trials; every trial requires genuine comparison of the ratings.

**Computed schedule:** 32 unique pairs × 3 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** SG-CGD (Small-Gap Composition-Gated Outlier Distrust), half-depth uniform-skepticism edition. Subjects integrate every binary expert rating as a sign vote into a zero-sum evidence score with subjective weights computed relative to the panel: a tilted trust plateau over [v_lo, v_hi] (ceiling just above 0.98), a consensus-gated chance flank with partial-ceiling promotion, and an outlier branch for above-ceiling claims consisting of (a) a bounded, saturating base distrust peaked at gap ~0.088 sitting on a uniform-skepticism floor, (b) a near-kernel composition amplification (narrow Gaussian at gap ~0.035, chance-fraction gate, sig_gap-throttled) that deepens distrust only of small-edge outliers on chance-laden scales, (c) a far-fraud route at gap ~0.185, and (d) a composition-gated edge-credibility ramp that promotes a well-separated 99% expert back toward plateau trust when the surrounding scale is unreliable. The single theoretical claim updated this round: the uniform-skepticism floor e_min — the residual distrust applied to a panel-max expert even when the gap kernel is silent (dense ladders, gap ~0.05 cells) — sits at a MODERATE depth (~0.36), roughly halfway between the near-zero floor of the accepted calibration and the deep floor of the rejected overshoot. This floor is what generates the empirically required mild anti-follow of 99% experts on dense validity ladders (Exp 7) without unbounded sign flips. Choice is a low-gain softmax plus uniform lapse.

**Parameters:**
- w_p: [0.88, 0.92]
- v_lo: [0.52, 0.55]
- v_hi: [0.9805, 0.9825]
- tilt: [0.09, 0.13]
- kappa_lo: [0.19, 0.22]
- kappa_gate: [0.66, 0.74]
- lambda_ext: [0.30, 0.42]
- g0: [0.041, 0.045]
- s_gap: [0.012, 0.014]
- g_star: [0.084, 0.092]
- s_l: [0.013, 0.017]
- s_r: [0.040, 0.050]
- e_min: [0.32, 0.40]
- w_floor: [-0.24, -0.17]
- rho_e: [1.05, 1.30]
- d_w: [0.08, 0.12]
- g_near: [0.032, 0.038]
- s_near: [0.014, 0.018]
- c_near: [3.5, 4.0]
- c_half: [0.44, 0.48]
- s_c: [0.085, 0.105]
- g_far: [0.180, 0.190]
- s_far: [0.018, 0.022]
- c_far: [20, 26]
- c_half_far: [0.57, 0.63]
- s_c_far: [0.06, 0.08]
- c_cred: [1.15, 1.40]
- g_edge: [0.070, 0.080]
- s_edge: [0.018, 0.022]
- c_cred_half: [0.57, 0.63]
- s_cred: [0.05, 0.06]
- beta: [0.38, 0.46]
- epsilon: [0.035, 0.055]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # SG-CGD: Small-Gap Composition-Gated Outlier Distrust with a
    # composition-gated edge-credibility ramp, half-depth-floor edition.
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
    #       PROMOTED to a partial ceiling kappa_gate*w_p.
    # (iii) BOUNDED BASE DISTRUST of the panel-max expert: a
    #       saturating shrinkage toward a SHALLOW floor,
    #         w_base = w_floor + (w_p - w_floor)*exp(-rho_e*E)
    #       with E = max(peaked gap kernel, e_min); the kernel peaks
    #       at gap ~0.088 (steep left wall s_l, wide right shoulder
    #       s_r). The UNIFORM-SKEPTICISM FLOOR e_min is the single
    #       lever moved this round: raised from ~0.15 to ~0.36
    #       (half the rejected overshoot depth), so dense-ladder
    #       and gap-0.05 outliers carry a moderate residual
    #       distrust even where the gap kernel is silent.
    #  (iv) NEAR-KERNEL COMPOSITION AMPLIFICATION (small-gap gate):
    #         comp_near = c_near * sig_gap * L(gap) * G(c_frac)
    #       L is a NARROW Gaussian centered g_near ~ 0.035 (inert for
    #       gap >= 0.07); G is a chance-fraction logistic. The
    #       sig_gap factor is ESSENTIAL and retained: it throttles
    #       the route on dense ladders (gap ~ 0.01), which protects
    #       the cliff-conflict cells from spurious anti-votes.
    #  (v) FAR-FRAUD ROUTE: comp_far = c_far * sig_gap * F(gap) *
    #       G_far(c_frac), F a SHARP logistic at g_far ~ 0.185 with
    #       its own chance gate: closed on every gap 0.09-0.14 panel
    #       tested here, open for far outliers over chance-laden
    #       panels (gap >= ~0.18).
    # (vi) COMPOSITION-GATED EDGE-CREDIBILITY RAMP: for gap >= ~0.075,
    #       cred = c_cred * R(gap) * Q(c_frac), capped at the plateau
    #       value. On a chance-laden panel a well-separated 99%
    #       expert is the only credible voice and is promoted back
    #       toward plateau trust; on a coherent-ladder panel the ramp
    #       stays closed and residual kernel distrust persists.
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
    tilt = float(parameters["tilt"])
    kappa_lo = float(parameters["kappa_lo"])
    kappa_gate = float(parameters["kappa_gate"])
    lambda_ext = float(parameters["lambda_ext"])
    g0 = float(parameters["g0"])
    s_gap = float(parameters["s_gap"])
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
    c_half_far = float(parameters["c_half_far"])
    s_c_far = float(parameters["s_c_far"])
    c_cred = float(parameters["c_cred"])
    g_edge = float(parameters["g_edge"])
    s_edge = float(parameters["s_edge"])
    c_cred_half = float(parameters["c_cred_half"])
    s_cred = float(parameters["s_cred"])
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

    # Outlier = a claim ABOVE the trust ceiling. With v_hi ~ 0.981 a
    # 0.98-topped panel stays INSIDE the trusted plateau; only 99%+
    # claims enter the distrust branch.
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
        # experts slightly outrank lower ones while the band remains
        # 'approximately equal'.
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

    # ---- (iii)-(vi): outlier distrust / credibility architecture. ----
    if outlier:
        # Chance fraction of the panel (share of experts below the
        # trust band) drives both composition gates.
        c_frac = float(np.mean(flank))

        # (iii) Bounded base distrust: saturating shrinkage toward a
        # shallow floor, driven by a peaked kernel in the
        # uncorroborated edge (peak at g_star, steep left wall,
        # wide right shoulder), sitting on the uniform-skepticism
        # floor e_min (RAISED to ~0.36 this round: half the rejected
        # overshoot depth, per the endpoint interpolation).
        width = s_l if gap < g_star else s_r
        E_kernel = float(np.exp(-0.5 * ((gap - g_star) / width) ** 2))
        E = max(E_kernel, e_min)
        w_base = w_floor + (w_p - w_floor) * float(np.exp(-rho_e * E))

        # (iv) Near-kernel composition amplification: fires only in
        # the small-gap regime (L centered g_near, narrow width),
        # only on chance-laden scales (G threshold c_half), and is
        # THROTTLED by sig_gap so dense ladders (gap ~ 0.01, where
        # sig_gap ~ 0.08) are protected from spurious anti-votes.
        L = float(np.exp(-0.5 * ((gap - g_near) / s_near) ** 2))
        G_near = _sigmoid((c_frac - c_half) / s_c)
        comp_near = c_near * float(sig_gap) * L * float(G_near)

        # (v) Far-fraud route: sharp logistic at g_far with its own
        # chance gate; closed on every gap 0.09-0.14 panel tested
        # here, open for far outliers over chance-laden panels.
        F_far = _sigmoid((gap - g_far) / s_far)
        G_far = _sigmoid((c_frac - c_half_far) / s_c_far)
        comp_far = c_far * float(sig_gap) * float(F_far) * float(G_far)

        # (vi) Composition-gated edge-credibility ramp: for
        # well-separated outliers (gap >= ~g_edge) on chance-laden
        # scales, the panel-max expert is promoted back toward
        # plateau trust (capped at w_p); on coherent ladders the
        # ramp stays closed and residual kernel distrust persists.
        R_cred = _sigmoid((gap - g_edge) / s_edge)
        Q_cred = _sigmoid((c_frac - c_cred_half) / s_cred)
        cred = c_cred * float(R_cred) * float(Q_cred)

        # Small bounded extremity term.
        ext = d_w * (v_top - v_hi) / max(1.0 - v_hi, 1e-6)

        w_top = w_base - ext - comp_near - comp_far + cred
        # The outlier is at most as trusted as a plateau expert.
        w_top = min(w_top, w_p)
        # All experts tied at the panel maximum share the same
        # weight (a pair of 0.99 super-experts doubles the vote).
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Fraction of choices that FOLLOW the 99%-validity expert's endorsement,\n    # computed only on trials where that expert discriminates between A and B.\n    #\n    # The 0.99 expert is feature index 0 (validities = [0.99, 0.80, 0.75, 0.5 x 8],\n    # displayed first). On trials 1-24 of this design the 0.99 expert's rating\n    # differs between A and B; on trials 25-32 it is tied and carries no vote.\n    #\n    # The two theories make sharply disjoint predictions for this quantity:\n    #  - SG-CGD: w(0.99) in [-14.1, -5.8] on this panel, so the far-fraud route\n    #    swamps every other contribution; the 99-endorsed side loses on every\n    #    discriminating trial and the follow rate sits at the lapse floor\n    #    (~0.02-0.03) for ANY admissible parameter setting.\n    #  - EOBOT: w(0.99) in [-2.92, -0.48]; the anti-vote is comparable in size\n    #    to a few chance cues, so backing coin-flippers / plateau experts can\n    #    override it. Follow rates range from ~0.09 (lone 99 vs 0.80) up to\n    #    ~0.75+ (99 + 6-8 coin-flippers vs 0.80), averaging roughly 0.4-0.5\n    #    across the 24 discriminating cells.\n    # The gap (~0.45) is far larger than the per-subject sampling noise\n    # (72 discriminating observations per subject), so the metric is tight\n    # within each theory's simulated population.\n\n    a_col = data['option_a_ratings']\n    b_col = data['option_b_ratings']\n    resp = data['response'].values\n\n    follows = []\n    for i in range(len(data)):\n        ra = a_col.iloc[i]\n        rb = b_col.iloc[i]\n        # Rating of the 99% expert (feature index 0) for each option.\n        a0 = int(ra[0])\n        b0 = int(rb[0])\n        if a0 == b0:\n            # The 99% expert is tied on this trial: no endorsement to follow.\n            continue\n        r = int(resp[i])\n        if a0 == 1 and b0 == 0:\n            # A carries the 99% endorsement.\n            follows.append(1.0 if r == 0 else 0.0)\n        else:\n            # B carries the 99% endorsement.\n            follows.append(1.0 if r == 1 else 0.0)\n\n    if len(follows) == 0:\n        return 0.5\n    return float(np.mean(follows))",
  "rationale": "This metric is the direct behavioral readout of the single parameter the experiment was built to isolate: the far-fraud amplitude applied to the 0.99 expert on a majority-chance panel. On every trial where the 0.99 expert discriminates (24 of 32 unique trials, 72 of 96 observations per subject), SG-CGD's far route contributes 8-15 chance-cue units AGAINST the endorsed side, so its predicted follow rate is pinned at the lapse floor (~0.02-0.03) for every parameter setting in its box \u2014 a near-parameter-free point prediction. EOBOT's far route contributes at most ~2.9 chance-cue units, so plateau experts (w~0.86) and stacks of consensus-promoted coin-flippers (w~0.58-0.70 each) can and do override it: EOBOT's predicted follow rate climbs from ~0.09 (99 vs 80 alone) through ~0.54-0.75 (99 + 5-8 coin-flippers vs 80) and averages roughly 0.4-0.5 across the discriminating cells. The between-theory mean gap (~0.45) is enormous relative to per-subject noise: each subject contributes 72 binary discriminating trials, so the binomial component of the per-subject variance is at most ~p(1-p)/72 (~0.003), and even after adding parameter-box heterogeneity the Welch t at N=25 is expected to be in the double digits. The metric is well defined both on the pooled DataFrame and on any single subject's slice (it is just a follow fraction), so the between-subject variance is always available. It is not tautological: it is an observable choice statistic (do subjects pick the product endorsed by the 99% expert?) that any follow-the-best-expert heuristic, both theories, and human data can all be scored on; SG-CGD predicts near-zero following, EOBOT predicts substantial following, and human data will land visibly closer to one or the other."
}
```

## Usage

```json
{
  "input_tokens": 14421,
  "output_tokens": 7460,
  "reasoning_tokens": 6333,
  "provider": "Sail Research"
}
```
