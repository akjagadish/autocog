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
**Validities (n_features=9):** [0.99, 0.85, 0.8, 0.75, 0.5, 0.5, 0.5, 0.5, 0.5]

**Trial pairs (n=32):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 1, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[1, 0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 1, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[1, 0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1, 0, 0]
  trial 8: A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1, 1, 1]
  trial 9: A=[1, 0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 10: A=[1, 0, 0, 0, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[1, 0, 1, 0, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 1, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 13: A=[0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 1, 0, 0, 0, 0, 0]
  trial 14: A=[1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 1, 1, 0, 0]
  trial 15: A=[0, 1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1]
  trial 16: A=[1, 0, 0, 0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 17: A=[1, 1, 0, 0, 1, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 18: A=[1, 0, 1, 0, 1, 1, 1, 1, 0]  B=[1, 0, 1, 0, 0, 0, 0, 0, 0]
  trial 19: A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 1, 1, 1, 0, 0]
  trial 20: A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0]
  trial 21: A=[1, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1, 0, 0, 0]
  trial 22: A=[0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 23: A=[1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0]
  trial 24: A=[1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0]
  trial 25: A=[1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0]
  trial 26: A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0]
  trial 27: A=[0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 28: A=[1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0]
  trial 29: A=[1, 0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 0, 0, 0]
  trial 30: A=[0, 1, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 1, 0, 0, 0]
  trial 31: A=[1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 0, 0]
  trial 32: A=[1, 0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0]

**Rationale:** CORE LOGIC. BODT (advocated) and CREDT (competing) share the identical architecture (sign votes x panel-relative weights x low-gain softmax x lapse), the same plateau, the same ~0.981 trust ceiling, and the same consensus-gating machinery. Prior head-to-head rounds left one configuration untested: a FAR outlier (gap 0.14) floating above a MAJORITY-CHANCE panel. On this configuration the two theories' composition routes separate maximally and in a REVERSED ordering relative to every prior round: BODT's composition term is c_amp*sig_gap*far(gap)*comp_gate with c_amp in [3.8,4.4] (the raised, 'verified' level), its far logistic WIDE OPEN at gap 0.14 (far in [0.83,0.93]) and its comp_gate essentially engaged at c_frac = 5/9 = 0.556 (sigmoid in [0.93,0.95]); CREDT's composition term is c_amp*sigmoid(c_frac) with c_amp in [1.22,1.36], and its P-bump has fully receded at gap 0.14 (P in [0.007,0.044], narrow s_r 0.030-0.036). Panel: [0.99, 0.85, 0.80, 0.75, 0.50, 0.50, 0.50, 0.50, 0.50]. Resulting weight on the 0.99 expert across the ENTIRE parameter boxes: BODT w(0.99) in [-3.46, -1.84] (center -2.82, i.e. 2-4 trusted cues AGAINST); CREDT w(0.99) in [-1.31, -0.71] (center -1.00, about one trusted cue against). DISJOINT BY 0.53 UNITS. This is the first regime where BODT predicts DEEPER outlier distrust than CREDT — the exact reversal of [12] (gap 0.05, chance-laden: CREDT -1.6 vs BODT -0.3) — because BODT's far-gated route OPENS with gap while CREDT's P-bump CLOSES with gap. Simultaneously, the consensus gate is essentially fully open (sig_gap ~ 0.999 at gap 0.14 for both, shared g0/s_gap), exposing the second disjoint lever: gated chance weight w(0.50) = kappa_gate*w_p in [0.57,0.72] for BODT (partial cap) vs full promotion w_p in [0.88,0.92] for CREDT. Plateau weights are common currency by design (0.85/0.80/0.75 clustered high, so BODT's tilt is inert: 0.85-0.91 vs CREDT's flat 0.88-0.92). The design therefore reduces to two disjoint weights — w_top and w_chance — measured psychometrically in a common currency (chance-cue units) from both sides of 0.50. 32 unique pairs x K = 96//32 = 3 = 96 trials.

FAMILY R (flagship crossover ladder: the 0.99 endorsement + k coin-flippers vs one 0.85 expert). R0-R5: A wins {0.99 + k of the 0.50 experts}, B wins the 0.85 expert, k = 0..5. Every rung is box-disjoint: R0 BODT [0.18,0.25] vs CREDT [0.30,0.35]; R1 [0.21,0.31] vs [0.37,0.43]; R2 [0.24,0.37] vs [0.45,0.53]; R3 [0.28,0.44] vs [0.54,0.62]; R4 [0.33,0.51] vs [0.60,0.70]; R5 [0.37,0.59] vs [0.68,0.77]. R3 and R4 are SIGN-DISJOINT: 'does a 99% expert backed by 3-4 coin-flippers beat one 85% expert?' BODT says no (its raised c_amp forces comp >= 2.9, so w_top <= -1.84 and w_ch <= 0.72 cap the stack); CREDT says yes from k=3 on. The crossover location k* is the headline statistic: CREDT crosses 0.5 between k=2 and k=3 (box-level, center k* ~ 2.1); BODT's center crossover is k* ~ 5.8 — OUTSIDE the design range (p_A still <= 0.59 at k=5 across its whole box). A ~3-cue separation in crossover location that no parameter setting in either box can close. Mirrors R3m/R5m (centers 0.66/0.55 under BODT vs 0.42/0.26 under CREDT) control side bias and reverse the pattern.

FAMILY X (kappa_gate ladder; outlier TIED so w_top contributes exactly zero). B wins the 0.85 expert; A wins k = 1..5 coin-flippers. X2: BODT [0.52,0.56] vs CREDT [0.58,0.60]; X3: [0.57,0.63] vs [0.66,0.68]; X4: [0.62,0.69] vs [0.73,0.76]; X5: [0.66,0.75] vs [0.79,0.82] — box-disjoint at every rung k>=2. The X-crossover (how many coin-flippers equal one 85% expert) is BODT k* ~ 1.4 vs CREDT k* ~ 1.0. CRITICAL COMPOSITE METRIC: the R-minus-X crossover difference estimates w_top directly in chance-cue units — BODT (5.8-1.4) = 4.4 cues x 0.64 ~ 2.8 evidence units ~ |w_top|; CREDT (2.1-1.0) = 1.1 cues x 0.90 ~ 1.0 ~ |w_top|. The ratio of these two estimates (~2.8) is a direct readout of the composition-amplitude calibration (BODT c_amp 3.8-4.4 vs CREDT 1.22-1.36) — the exact lever BODT's description claims was slope-verified on Exp6 and CREDT's description caps at 1.36. Mirrors X3m/X5m included.

FAMILY Z (chance-only ladder; outlier present but tied, keeping the gate ~fully open). A wins 2/3/4 coin-flippers, nothing else discriminates. Z2: BODT [0.60,0.64] vs CREDT [0.66,0.68]; Z3: [0.65,0.71] vs [0.74,0.76]; Z4: [0.69,0.76] vs [0.79,0.82] — disjoint at every rung. Under CREDT, gated coin-flippers are near-exchangeable with plateau experts (a 4-0 coin-flip sweep is followed at ~0.80); under BODT's partial cap only ~0.72. Mirror Z3m included.

FAMILY V/O (outlier head-to-heads). V1 (A wins 0.99, B wins one coin-flip): BODT [0.19,0.27] vs CREDT [0.30,0.35] — both theories say a COIN-FLIPPER's endorsement beats a 99% expert's, but at disjoint confidences (the amplitude question in its purest form). V2 and mirror V1m extend. O0 (A wins ONLY the 0.99 expert, plateau tied 1/1): s = w_top exactly — BODT [0.23,0.32] vs CREDT [0.38,0.43], disjoint. O1 (A wins 0.99+0.85 vs B wins 0.80+0.75; raw tally TIED 2-2): BODT [0.18,0.26] vs CREDT [0.30,0.35] — a count-balanced trial where both theories anti-follow but by disjoint amounts; mirror O1m reverses it. O2 (A wins 0.99+0.85+0.80 vs B wins 0.75+one 0.50): centers 0.27 vs 0.40.

FAMILY K (unanimity reversal cell). K1 (A wins ALL FOUR genuine experts 0.99/0.85/0.80/0.75; B wins one coin-flip): raw tally 4-vs-1 for A, yet BODT center p_A = 0.42 (BELOW 0.50 — the 99% anti-vote plus the capped coin-flip discount sink the unanimous panel) vs CREDT 0.58; boxes [0.36,0.53] vs [0.54,0.62], near-disjoint with a center sign flip. This is the mirror image of [12]'s K1 (there CREDT was the anti-follower at 0.20 vs BODT 0.47) — the same unanimous-expertise surface with the theories' roles REVERSED, which is itself a strong cross-experiment signature: only BODT predicts that increasing the gap from 0.05 to 0.14 on a chance-laden panel flips the deeper anti-follower from CREDT to BODT. K2 (B wins two coin-flippers): centers 0.36 vs 0.49.

HONEST KNIVES BOTH WAYS. Against MY theory (BODT): (i) p(R3) >= 0.50 falsifies BODT outright (its box caps at 0.44 — comp >= 2.93 forces w_top <= -1.84 and w_ch <= 0.718); (ii) p(X2) >= 0.60 or p(X3) >= 0.66 falsifies the kappa_gate cap; (iii) p(Z3) >= 0.74 falsifies it again; (iv) p(R5) >= 0.62 falsifies the raised c_amp. Against CREDT: (i) p(R3) <= 0.52 falsifies CREDT (its box floor is 0.54); (ii) p(X2) <= 0.55, p(Z3) <= 0.71, p(V1) <= 0.27 falsify its full-promotion claim; (iii) p(O0) <= 0.36 falsifies its additive discount (cannot exceed -0.71... cannot shrink below 0.71 in magnitude on this panel). The interval roughly (0.46, 0.52) on R3 falsifies BOTH theories — included deliberately as an honest adversarial region.

CONTROLS. N1 (one coin-flipper won by each side) and T2 (A wins 0.85+one 0.50 vs B wins 0.80+one 0.50; BODT's tilt difference is 0.009, CREDT's is exactly 0) are cells where BOTH theories must predict ~exactly 0.50 for every parameter — parameter-free side/display-bias assays. G1/G1m (three plateau experts vs one coin-flipper) are agreement anchors with OVERLAPPING boxes (BODT [0.65,0.70], CREDT [0.66,0.68]) pinning the shared currency w_p - w_ch and lapse/temperature — a deliberate check that the common machinery is intact before interpreting the disjoint cells. Tie patterns mix 0/0 and 1/1 throughout so 'count the 1s' is unviable: R0-R5 and K1/K2 have LARGER raw tallies and 1-counts for A while BODT predicts A loses; R3m/R5m reverse it; O1 has exactly tied tallies with a 12-point center dissociation. No all-tie trials; every trial requires scanning all 9 experts to locate 1-6 discriminators, with the five coin-flippers displayed last.

KEY QUANTITATIVE METRICS: (i) R-ladder crossover k* (CREDT ~2.1 vs BODT >= 5, center 5.8) — a ~3.7-cue separation; (ii) the R-minus-X crossover difference, a direct estimate of |w_top| in chance-cue units (BODT ~2.8 vs CREDT ~1.0 evidence units — the composition-amplitude ratio); (iii) X- and Z-ladder levels (kappa_gate cap vs full promotion), box-disjoint at every rung k>=2; (iv) sign flips at R3/R4 and center sign flip at K1; (v) disjoint confidence on V1/O0/O1 where both theories anti-follow; (vi) parameter-free 0.50 cells (N1, T2) and the G1 agreement anchor.

PREDICTED PATTERN IF BODT IS RIGHT (centers): R0-R5 = .20/.24/.29/.34/.39/.45 (never crossing 0.5), R3m .66, R5m .54, X1-X5 = .48/.54/.59/.65/.70, Z2-Z4 = .61/.67/.72, V1 .21, V2 .17, V1m .79, K1 .42, K2 .36, O0 .27, O1 .20, O1m .80, O2 .27, N1/T2 .50, G1 .67 — the 99% endorsement is worth ~3 trusted cues AGAINST, coin-flippers are worth ~70% of an expert, and even a unanimous genuine-expert panel backed by nothing loses to a single coin-flip. IF CREDT IS RIGHT: R0-R5 = .32/.40/.49/.58/.66/.74 (crossing at k~2), R3m .42, R5m .26, X1-X5 = .50/.59/.67/.73/.79, Z2-Z4 = .67/.73/.79, V1 .31, V2 .24, V1m .69, K1 .58, K2 .49, O0 .40, O1 .31, O1m .69, O2 .40, N1/T2 .50, G1 .67 — the same endorsement is worth only ~1 cue against, coin-flippers are near-exchangeable with experts, and the unanimous panel wins. The w_top and w_chance boxes are disjoint by construction, so the patterns are mutually exclusive on ~24 of 32 cells.

DIFFERENTIATION FROM PRIOR EXPERIMENTS. [12] tested gap 0.05 chance-laden (CREDT the deeper anti-follower, BODT's far-gate strangled); [13] gap 0.05 coherent (bounded-saturation vs additive depth at the P-peak); [14] gap 0.09 with c_frac 0.30 (BODT's comp_gate ~0.01 CLOSED, so its w_top was POSITIVE — a sign flip vs CREDT). No prior design instantiated the conjunction where BODT's composition route is FULLY engaged (c_frac 0.556 opens comp_gate to 0.93-0.95) at a gap where its far-gate is wide open (0.83-0.93) AND CREDT's P-bump has receded to ~0.02 — the unique configuration in which BOTH theories collapse the outlier but at disjoint AMPLITUDES (a pure calibration test of c_amp, not a sign test), and the unique configuration in which BODT is the deeper anti-follower (ordering reversed vs [12]). The R-minus-X crossover-difference estimator of w_top in chance-cue units, the out-of-range BODT crossover (k* ~ 5.8 with only 5 coin-flippers available), and the role-reversed unanimity cell K1 are new quantitative signatures. Sides are balanced via mirrors in every family (12 A-tilted / 17 B-tilted / 3 neutral under BODT centers — the B-tilt excess is itself BODT's signature prediction, controlled by mirrors; near-balanced under CREDT centers); no all-tie trials; every trial requires genuine comparison of the ratings.

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
  "metric_source": "```python\ndef metric(data: pd.DataFrame) -> float:\n    \"\"\"\n    'Low-drama side' choice rate on the sign-disjoint cells (R2-R5, R3m, R5m, K1, K2).\n\n    These eight trial types pit the 0.99 outlier's endorsement (backed by 2-5\n    coin-flip experts, or by the full unanimous genuine panel) against a single\n    0.85 expert (R-family) or a single/two coin-flipper(s) (K-family). The\n    metric is the fraction of choices falling on the 'low-drama' side: the\n    option endorsed ONLY by the lone 0.85 expert (R-cells) or ONLY by the\n    coin-flipper(s) against the unanimous panel (K-cells).\n\n    BODT (deep outlier distrust: w(0.99) ~ -2.4 to -3.2, capped coin weight\n    ~0.57-0.72) predicts subjects REJECT the outlier-endorsed side on these\n    cells -> metric ~0.70-0.75. CREDT (shallow distrust: w(0.99) ~ -0.8 to\n    -1.2, coin weight ~0.88-0.92) predicts subjects FOLLOW the outlier-backed\n    side -> metric ~0.28-0.35. The predicted s_a ranges over the FULL\n    parameter boxes do not cross zero on R3/R4/R5/R3m/R5m/K1, so the contrast\n    is sign-disjoint for every admissible parameter setting.\n    \"\"\"\n    # (A_ratings, B_ratings) -> response code of the 'low-drama' side\n    # (0 = chose A, 1 = chose B)\n    targets = {\n        # R2: A = 0.99 + 2 coin-flips, B = 0.85 -> low-drama side is B\n        ((1, 0, 0, 0, 1, 1, 0, 0, 0), (0, 1, 0, 0, 0, 0, 0, 0, 0)): 1,\n        # R3: A = 0.99 + 3 coin-flips, B = 0.85 -> B\n        ((1, 0, 0, 0, 1, 1, 1, 0, 0), (0, 1, 0, 0, 0, 0, 0, 0, 0)): 1,\n        # R4: A = 0.99 + 4 coin-flips, B = 0.85 -> B\n        ((1, 0, 0, 0, 1, 1, 1, 1, 0), (0, 1, 0, 0, 0, 0, 0, 0, 0)): 1,\n        # R5: A = 0.99 + 5 coin-flips, B = 0.85 -> B\n        ((1, 0, 0, 0, 1, 1, 1, 1, 1), (0, 1, 0, 0, 0, 0, 0, 0, 0)): 1,\n        # R3m: A = 0.85, B = 0.99 + 3 coin-flips -> A\n        ((0, 1, 0, 0, 0, 0, 0, 0, 0), (1, 0, 0, 0, 1, 1, 1, 0, 0)): 0,\n        # R5m: A = 0.85, B = 0.99 + 5 coin-flips -> A\n        ((0, 1, 0, 0, 0, 0, 0, 0, 0), (1, 0, 0, 0, 1, 1, 1, 1, 1)): 0,\n        # K1: A = unanimous 0.99/0.85/0.80/0.75 panel, B = one coin-flip -> B\n        ((1, 1, 1, 1, 0, 0, 0, 0, 0), (0, 0, 0, 0, 1, 0, 0, 0, 0)): 1,\n        # K2: A = unanimous panel, B = two coin-flips -> B\n        ((1, 1, 1, 1, 0, 0, 0, 0, 0), (0, 0, 0, 0, 1, 1, 0, 0, 0)): 1,\n    }\n\n    hits = 0\n    n = 0\n    a_col = data['option_a_ratings'].tolist()\n    b_col = data['option_b_ratings'].tolist()\n    r_col = data['response'].tolist()\n\n    for a_cell, b_cell, resp in zip(a_col, b_col, r_col):\n        try:\n            a_key = tuple(int(v) for v in a_cell)\n            b_key = tuple(int(v) for v in b_cell)\n        except (TypeError, ValueError):\n            continue\n        side = targets.get((a_key, b_key))\n        if side is None:\n            continue\n        n += 1\n        if int(resp) == side:\n            hits += 1\n\n    if n == 0:\n        return 0.5\n    return float(hits) / float(n)\n```",
  "rationale": "The design's sharpest lever is the sign-disjoint set of cells where the 0.99 outlier's endorsement (plus a coin-flip stack, or the unanimous genuine panel) faces a single low-drama alternative. I recomputed both theories' implied evidence scores over their FULL parameter boxes on this panel: BODT's w(0.99) lies in roughly [-3.2, -2.4] with a capped coin weight in [0.57, 0.72], while CREDT's w(0.99) lies in [-1.2, -0.8] with coin weight equal to w_p in [0.88, 0.92] \u2014 both weights disjoint. Consequently, on R3/R4/R5 and their mirrors the signed evidence s_a stays negative under BODT (e.g. R3: s in [-2.4, -1.1]) and positive under CREDT (R3: s in [0.54, 1.10]) for EVERY parameter draw; K1 is likewise box-disjoint (BODT s in [-1.31, -0.21], CREDT s in [0.58, 1.06]). R2 and K2 add further gap (CREDT sits near 0.5 there, BODT strongly on the low-drama side). The metric is the simple choice rate of the low-drama side over these 8 trial types (24 Bernoulli draws per subject). Expected values: ~0.72 under BODT vs ~0.31 under CREDT \u2014 a ~0.40 mean gap against a per-subject SD of ~0.09-0.10 (binomial noise over 24 draws plus parameter heterogeneity), giving a Welch t of order 10+ at N=25, far below alpha=0.01. I deliberately excluded saturated 'confidence-only' cells (R0/R1, V, O families) where both theories anti-follow and the p-gap is small, and the X/Z ladders whose gaps (~0.08-0.11) are too weak per unit variance; the chosen 8 cells maximize the gap-to-variance ratio. The metric works identically on the pooled DataFrame and on single-subject slices (it is a plain mean of indicators), so the between-subject variance is well defined. Under BODT (my advocated theory) human data should yield ~0.70-0.75: subjects reject a 99% expert backed by up to five coin-flippers in favor of one 85% expert, and reject a unanimous genuine-expert panel in favor of a single coin-flipper. Under CREDT it should yield ~0.28-0.35. Values near 0.5 would falsify both theories' extremes on this panel."
}
```

## Usage

```json
{
  "input_tokens": 13164,
  "output_tokens": 34695,
  "reasoning_tokens": 32798,
  "provider": "Sail Research"
}
```
