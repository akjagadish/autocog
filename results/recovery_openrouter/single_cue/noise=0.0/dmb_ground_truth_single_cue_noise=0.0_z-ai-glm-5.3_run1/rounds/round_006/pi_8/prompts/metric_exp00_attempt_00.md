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
**Validities (n_features=8):** [0.99, 0.94, 0.85, 0.75, 0.65, 0.655, 0.5, 0.5]

**Trial pairs (n=24):**
  trial 1: A=[1, 1, 1, 0, 1, 0, 0, 1]  B=[0, 1, 1, 1, 0, 0, 0, 1]
  trial 2: A=[1, 0, 1, 0, 1, 1, 0, 1]  B=[0, 1, 0, 1, 1, 0, 0, 1]
  trial 3: A=[1, 0, 1, 0, 1, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 0, 1]
  trial 4: A=[0, 1, 1, 1, 0, 0, 0, 1]  B=[1, 1, 1, 0, 1, 0, 0, 1]
  trial 5: A=[0, 1, 0, 1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 1, 0, 1]
  trial 6: A=[0, 1, 1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 1, 0, 1]
  trial 7: A=[1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0]
  trial 8: A=[1, 0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0, 1, 0]
  trial 9: A=[1, 0, 1, 0, 1, 1, 1, 0]  B=[0, 1, 1, 0, 1, 1, 0, 0]
  trial 10: A=[1, 0, 1, 0, 1, 1, 1, 1]  B=[0, 1, 1, 0, 1, 1, 0, 0]
  trial 11: A=[0, 1, 1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0, 1, 0]
  trial 12: A=[0, 1, 1, 0, 1, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1, 1, 1]
  trial 13: A=[1, 1, 0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0, 1, 0]
  trial 14: A=[1, 1, 1, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 1, 0, 0, 1]
  trial 15: A=[0, 1, 1, 0, 1, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0, 0, 1]
  trial 16: A=[1, 1, 1, 0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1, 0, 1, 0]
  trial 17: A=[0, 0, 1, 0, 1, 0, 1, 0]  B=[1, 1, 1, 0, 1, 0, 1, 0]
  trial 18: A=[1, 0, 1, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 1, 1, 0, 0]
  trial 19: A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  trial 20: A=[1, 1, 0, 0, 0, 1, 1, 0]  B=[1, 0, 1, 0, 1, 0, 1, 0]
  trial 21: A=[1, 1, 0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 1, 0, 1, 0, 1]
  trial 22: A=[1, 0, 1, 0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 1, 0, 0, 0]
  trial 23: A=[1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1]
  trial 24: A=[1, 1, 0, 1, 0, 1, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0, 0]

**Rationale:** CORE LOGIC. BODT (advocated) and CREDT (competing) share the identical architecture (sign votes x panel-relative weights x low-gain softmax x lapse), identical plateau (w_p in [0.88,0.92] over [v_lo, v_hi]), identical trust ceiling (~0.981), and identical consensus-gating machinery (same g0/s_gap/lambda_ext). They differ structurally in exactly one place: the discount of an above-ceiling outlier. CREDT's discount is UNBOUNDED and additive: rho*(P + C*sig_gap) with rho in [1.30,1.42]. BODT's is a BOUNDED, SATURATING shrinkage toward w_floor: w_top_mod = w_floor + (w_p - w_floor)*exp(-rho_m*P) with rho_m in [0.9,1.2] -- and because exp(-rho_m) ~ 0.30-0.41, even at MAXIMAL peaked distrust (P ~ 1) the shrinkage only demotes the outlier from w_p ~ 0.90 to ~ +0.14, i.e., near-zero, never negative. The design engineers the panel that isolates this single disagreement at its peak: a 0.99 claim sitting exactly 0.05 above its 0.94 next rung (gap = g_star for both theories, so the spurious-precision bump P ~ 0.997-1.0 across BOTH entire boxes) on a COHERENT panel (c_frac = 2/8 = 0.25, below both c_half boxes, so the composition amplifier is sub-threshold for both: BODT comp_gate ~ 0.12-0.27 further strangled by far(0.05) ~ 0.06-0.19; CREDT C ~ 0.08-0.29). Panel: [0.99, 0.94, 0.85, 0.75, 0.65, 0.55, 0.50, 0.50]. Resulting weight on the 0.99 expert across the ENTIRE parameter boxes: BODT w(0.99) in [-0.14, +0.15] (essentially INERT -- a coin flip); CREDT w(0.99) in [-1.03, -0.61] (strongly ANTI-diagnostic). The boxes are DISJOINT by 0.46-1.0 weight units. Every other weight is common currency: plateau experts (0.55-0.94) get exactly w_p in both; the gated chance weight is 0.58-0.71 (BODT) vs 0.59-0.73 (CREDT) -- the kappa_lo difference (0.27 vs 0.20) is suppressed to ~0.015 by the ~0.78-open consensus gate, which is itself nearly identical across boxes. The design is therefore a psychometric isolation of ONE structural disagreement -- bounded saturation vs unbounded additive discount at the spurious-precision peak -- measured from both sides of 0.50. 24 unique pairs x K = 96//24 = 4 = 96 trials.

FAMILY M (flagship: the MASKED 0.99 -- balanced backgrounds cancel exactly, leaving s = w(0.99) alone). Because all plateau weights are exactly w_p in both theories, any background in which A and B win equal plateau evidence cancels to machine precision, and the trial's entire evidence score is w(0.99). M1 (A wins {0.99, 0.65} vs B wins {0.75}), M2 (A wins {0.99, 0.85, 0.55} vs B wins {0.94, 0.75}, 2-vs-2 plateau), M4 (A wins {0.99, 0.55} vs B wins {0.94}), and O1 (A wins ONLY the 0.99 expert, everything else tied -- the lone-endorsement cell) all have s = w(0.99) exactly. BODT box: p in [0.47, 0.53] -- PINNED AT A COIN FLIP for every parameter setting; the 99% expert's endorsement is worth nothing. CREDT box: p in [0.30, 0.38] -- subjects should choose AGAINST the 99% endorsement by 12+ points. DISJOINT with a 9-point gap between box edges. Mirrors M1m/M2m/M4m (s = -w(0.99)): BODT [0.47,0.53] vs CREDT [0.62,0.70]. Both theories additionally demand the parameter-free within-family equalities M1 = M2 = M4 = O1 (background invariance across 1-vs-1 plateau, 2-vs-2 plateau, and lone-endorsement surfaces) -- inequality would indicate a count- or surface-based heuristic outside both models. DOUBLE BIND: masked p in (0.38, 0.47) falsifies BOTH theories -- included deliberately as an honest adversarial region.

FAMILY R (crossover psychometric: 0.99 + j coin-flippers vs the 0.94 expert). R0/R1/R2: A wins {0.99} plus j of the 0.50 experts (j = 0, 1, 2), B wins the 0.94 expert. s = w(0.99) - w_p + j*w_c. BODT boxes: R0 [0.32, 0.38], R1 [0.41, 0.50], R2 [0.51, 0.62]. CREDT boxes: R0 [0.18, 0.24], R1 [0.25, 0.36], R2 [0.35, 0.49]. DISJOINT AT EVERY RUNG. R2 is the sign-structured sharpest cell: 'does a 99% expert backed by two coin-flippers beat a 94% expert?' BODT says yes (entire box >= 0.51); CREDT says no (entire box <= 0.49). The crossover location j* (where p(A) crosses 0.5) is a direct estimate of the distrust depth in chance-cue units: BODT j* ~ 1.4, CREDT j* ~ 2.7 -- a ~1.3-cue separation (~0.85 evidence units) that no parameter setting in either box can close. Mirrors R0m (BODT [0.62,0.68] vs CREDT [0.76,0.83]) and R2m (BODT [0.38,0.49] vs CREDT [0.51,0.64]) are also box-disjoint and control side bias. S85 (A wins 0.99 vs B wins 0.85) shares R0's exact prediction in both theories (plateau flatness -- a parameter-free cross-reference equality both models demand).

FAMILY Sc/O (one-on-one and endorsement-stack cells). Sc (A wins 0.99, B wins one 0.50): BODT [0.35, 0.43] vs CREDT [0.20, 0.28] -- CREDT says a COIN-FLIPPER's endorsement clearly beats a 99% expert's (p ~ 0.23); BODT says it is a near-tie (p ~ 0.39). Mirror Scm: BODT [0.57,0.65] vs CREDT [0.72,0.80]. O2 (both the 0.99 AND 0.94 experts endorse A, all else tied): s = w(0.99) + w_p. BODT [0.62, 0.68] (follow the stack); CREDT [0.47, 0.56] (the 99% anti-vote nearly cancels the 94% endorsement -- a coin flip, center 0.51). Mirror O2m: BODT [0.32,0.38] vs CREDT [0.44,0.53]. A 15-point dissociation on whether a unanimous 99+94 endorsement is informative at all. V1 (0.99 + both coin-flippers vs 0.94 + 0.55): BODT [0.35,0.47] vs CREDT [0.21,0.33]. Ddeep (0.99+0.94+0.85 vs all five others): both favor B but at disjoint CONFIDENCE -- BODT [0.15,0.23] vs CREDT [0.08,0.12] -- a pure quantitative dissociation on an agreement-direction cell.

HONEST KNIVES AGAINST MY OWN THEORY (BODT). (i) The masked family: BODT's entire box spans only [0.47, 0.53] -- observed masked p < 0.45 or > 0.55 falsifies the bounded-saturation claim outright (the saturating shrinkage cannot demote the outlier below -0.14 nor trust it above +0.15 at the peak). (ii) R2: BODT demands p >= 0.51; observed p <= 0.49 falsifies it. (iii) O2: BODT demands p >= 0.60; observed p <= 0.58 strains the box. Conversely, CREDT is falsified outright by masked p >= 0.42, R2 p >= 0.52, O2 p >= 0.60, or R0 p >= 0.28 (its additive discount cannot go below -1.03 on this panel, and cannot go above -0.61).

CONTROLS/ANCHORS. T1 (A wins {0.94, 0.55} vs B wins {0.85, 0.65}) and T2 (the two 0.50 experts split): BOTH theories must predict EXACTLY 0.50 for every parameter (tied softmax is uniform; lapse injects uniform) -- parameter-free side/display-bias assays. Z2 (A wins both coin-flippers, nothing else): agreement anchor (BODT [0.67,0.74], CREDT [0.71,0.76]) pinning the SHARED gated-chance currency -- deviation here indicts the common gating machinery, not the disputed distrust depth. P1 (0.94 vs one 0.50): agreement anchor (~0.54 both) pinning w_p - w_c. G1 (all five plateau experts vs both coin-flippers): high-confidence agreement anchor (BODT [0.83,0.90], CREDT [0.88,0.92]) pinning lapse/temperature. Tie patterns mix 0/0 and 1/1 throughout so 'count the 1s' is unviable (e.g., M2 shows MORE 1s and MORE cue wins for A while CREDT predicts A loses by ~0.86 evidence units; Ddeep shows fewer 1s for A consistent with both; T1 has exactly tied 1-counts and tied wins). No all-tie trials; every trial requires scanning all 8 experts to locate discriminators (2-7 per trial); the 0.99 expert is displayed first (validity-salient), an adversarial touch favoring any follow-the-best heuristic.

KEY QUANTITATIVE METRICS: (i) masked-cell p (BODT ~0.50 vs CREDT ~0.34, boxes disjoint); (ii) R-ladder crossover j* (1.4 vs 2.7 chance cues); (iii) R2 sign (BODT >= 0.51 vs CREDT <= 0.49); (iv) O2 stack confidence (0.65 vs 0.51); (v) Sc (coin-flipper vs 99%: 0.39 vs 0.23); (vi) Ddeep confidence (0.19 vs 0.09); (vii) within-family parameter-free equalities (M1=M2=M4=O1; R0=S85; T1=T2=0.50 exactly) that isolate heuristics outside both models. PREDICTED PATTERN IF BODT IS RIGHT (centers): M1/M2/M4/O1 = 0.50, mirrors 0.50, R0/S85 0.33, R1 0.43, R2 0.55, R0m 0.67, R2m 0.46, Sc 0.37, Scm 0.63, O2 0.63, O2m 0.37, V1 0.39, Ddeep 0.19, T1/T2 0.50, Z2 0.68, G1 0.86, P1 0.53 -- the 99%-atop-94% endorsement is worth ~nothing: masked coin flips, crossover at ~1.4 coin-flippers, stacks followed. IF CREDT IS RIGHT: M1/M2/M4/O1 0.34, mirrors 0.66, R0/S85 0.20, R1 0.29, R2 0.41, R0m 0.80, R2m 0.59, Sc 0.23, Scm 0.77, O2 0.51, O2m 0.49, V1 0.26, Ddeep 0.09, T1/T2 0.50, Z2 0.72, G1 0.89, P1 0.53 -- the same endorsement is worth ~0.86 cues AGAINST: anti-following everywhere, crossover beyond ~2.7 coin-flippers, stacks cancelled. The w(0.99) boxes are disjoint by construction, so the patterns are mutually exclusive on 19 of 24 cells.

DIFFERENTIATION FROM PRIOR EXPERIMENTS. [12] pitted CREDT vs BODT on a CHANCE-LADEN panel (c_frac = 5/9, amplifier firing) with TWO tied 0.99 experts -- CREDT's maximum-distrust configuration, where the disagreement was amplified-vs-strangled composition routes. No prior design has isolated the PURE bounded-vs-unbounded disagreement at the peak of the P bump on a COHERENT panel (amplifier sub-threshold for both), where CREDT's rho*P term alone gives -0.6 to -1.0 while BODT's exp(-rho_m) saturation caps the demotion at near-zero. The masked-cell architecture (balanced plateau backgrounds cancel exactly in both theories, isolating s = w(0.99) as the whole evidence score and pinning BODT at a parameter-free coin flip across four different surfaces), the lone-endorsement cell O1, and the R-ladder crossover-in-chance-cue-units are new quantitative signatures. [11] used a similar panel against a monotone weighted theory (FDEI), where the dissociation was graded-vs-flat weights; here both theories have flat plateaus and identical gating, and the single differing weight is probed from both sides of 0.5. Sides are balanced (6 A-tilted / 8 B-tilted / 10 near-neutral under BODT centers; the masked-family B-tilt under CREDT is itself its signature prediction, controlled by mirrors); no all-tie trials; every trial requires genuine comparison of the ratings.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Bounded Outlier-Distrust Tallying (BODT). Subjects integrate all binary expert ratings as sign votes (+1/-1/0) into a zero-sum evidence score, with subjective weights computed RELATIVE to the panel rather than as a monotone function of stated validity: (i) a trust plateau over [v_lo, v_hi] (ceiling just above 0.98) gives all mid-band and 98% experts approximately equal weight; (ii) a small negative chance flank at v near 0.5, consensus-gated so chance cues are counted rather than inverted when an above-ceiling outlier destabilizes the scale; (iii) distrust of an above-ceiling outlier is a BOUNDED, SATURATING shrinkage of its weight toward a floor, peaked in the outlier's uncorroborated edge over the next rung (maximal near gap ~0.05: spurious precision; small but nonzero on dense ladders; receding for large edges on coherent panels); (iv) strongly negative distrust reserved for the composition-amplified regime — an outlier floating above a majority-chance panel. Choice is a low-gain softmax plus a uniform lapse. This round is a pure calibration claim: the only two levers whose slopes have been verified against reliably-signed residuals across the loop (the composition-route amplitude c_amp on Exp6, and the chance-flank magnitude kappa_lo on Exp5) are moved to their anti-oscillation midpoints; every structural parameter is held at its accepted value.

**Parameters:**
- w_p: [0.88, 0.92]
- v_lo: [0.52, 0.54]
- v_hi: [0.9805, 0.9825]
- kappa_lo: [0.26, 0.28]
- g0: [0.041, 0.045]
- s_gap: [0.012, 0.014]
- lambda_ext: [0.35, 0.45]
- g_star: [0.046, 0.050]
- s_l: [0.018, 0.024]
- s_r: [0.045, 0.055]
- w_floor: [-0.30, -0.25]
- rho_m: [0.9, 1.2]
- d_w: [0.15, 0.22]
- g_far: [0.10, 0.12]
- s_far: [0.025, 0.035]
- c_half: [0.32, 0.35]
- s_c: [0.05, 0.07]
- c_amp: [2.5, 2.8]
- beta: [0.34, 0.39]
- epsilon: [0.05, 0.07]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Bounded Outlier-Distrust Tallying (BODT), engaged-shoulder edition.
    # Stimulus: array of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are binary expert ratings (0/1).
    # Each feature j casts a sign vote sign(a_j - b_j), weighted by a
    # subjective trust function of the expert's STATED validity v,
    # evaluated RELATIVE to the panel:
    #   (i) TRUST PLATEAU: all v in [v_lo, v_hi] (v_hi just above
    #       0.98, so a 98% expert stays INSIDE the plateau) get
    #       approximately equal positive weight w_p.
    #  (ii) CHANCE FLANK + CONSENSUS GATE: v near 0.5 gets a small
    #       negative weight (-kappa_lo at v=0.5, rising linearly to
    #       the plateau at v_lo). When an above-ceiling claim is
    #       present, the flank is gated: gate = lambda_ext +
    #       (1-lambda_ext)*sig_gap, so chance cues are COUNTED
    #       (weight -> w_p) rather than inverted.
    # (iii) BOUNDED OUTLIER DISTRUST (the core BODT claim): the
    #       discount of the panel-max expert is a SATURATING
    #       SHRINKAGE toward a floor, not an unbounded sign flip:
    #         w_top_mod = w_floor + (w_p - w_floor) * exp(-rho_m * P)
    #       where P is a PEAKED bump in the rung gap (maximal at
    #       gap ~ g_star ~ 0.05 -- spurious precision; a SMALL but
    #       nonzero discount on dense ladders gap ~ 0.01 via the
    #       widened left width s_l; receding for large edges on
    #       coherent panels via the right width s_r, WIDENED so
    #       the shoulder covers the moderate-gap regime gap ~ 0.09).
    #       Because of the exponential saturation, w_top_mod never
    #       leaves [w_floor, w_p].
    #  (iv) COMPOSITION-AMPLIFIED ROUTE: an additive discount
    #         comp = c_amp * sig_gap * far(gap) * gate(c_frac)
    #       where far(gap) is a logistic gate whose center g_far is
    #       LOW enough (~0.10-0.12) to partially engage on
    #       chance-laden moderate-gap panels (gap ~ 0.09 with
    #       c_frac > c_half), while remaining essentially closed at
    #       gap ~ 0.05 and fully open only for far outliers
    #       (gap >= ~0.15) on majority-chance panels.
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

    # ---- (i) + (ii): plateau with a consensus-gated chance flank. ----
    w = np.full(n_features, w_p, dtype=float)
    flank = v < v_lo
    if np.any(flank):
        denom = max(v_lo - 0.5, 1e-6)
        w_flank = -kappa_lo + (w_p + kappa_lo) * np.clip(
            v[flank] - 0.5, 0.0, None
        ) / denom
        # With an above-ceiling claim in the panel, chance cues are
        # counted (weight -> w_p) rather than inverted.
        w[flank] = (1.0 - gate) * w_flank + gate * w_p

    # ---- (iii) + (iv): bounded outlier distrust, with a
    # composition-amplified route that now partially engages on
    # chance-laden MODERATE-gap panels. ----
    if outlier:
        # (a) Peaked bump in the uncorroborated edge: maximal near
        # gap ~ g_star (spurious precision), a SMALL nonzero
        # discount on dense ladders (widened s_l), receding for
        # large edges on coherent panels (widened right shoulder
        # s_r covers the moderate-gap regime gap ~ 0.09).
        width = s_l if gap < g_star else s_r
        P = float(np.exp(-0.5 * ((gap - g_star) / width) ** 2))

        # (b) BOUNDED, SATURATING shrinkage toward the floor:
        # w_top_mod in [w_floor, w_p] for every P; even at peak
        # distrust the outlier is only demoted toward the floor
        # (~ -0.25..-0.30 plateau units), never unboundedly flipped.
        w_top_mod = w_floor + (w_p - w_floor) * float(np.exp(-rho_m * P))

        # (c) Composition-amplified route: a logistic far gate
        # centered at g_far ~ 0.10-0.12, so it PARTIALLY engages on
        # chance-laden moderate-gap panels (gap ~ 0.09, c_frac >
        # c_half) and is fully open only for far outliers
        # (gap >= ~0.15) on majority-chance panels.
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n\n    def _to_vec(x):\n        if isinstance(x, str):\n            import ast\n            x = ast.literal_eval(x)\n        return np.asarray(x, dtype=float).ravel()\n\n    follows = []\n    for a_raw, b_raw, resp in zip(\n        data[\"option_a_ratings\"], data[\"option_b_ratings\"], data[\"response\"]\n    ):\n        try:\n            a = _to_vec(a_raw)\n            b = _to_vec(b_raw)\n        except Exception:\n            continue\n        if a.size == 0 or a.size != b.size:\n            continue\n        # The 0.99-validity expert is feature index 0 (displayed first).\n        d0 = float(a[0]) - float(b[0])\n        if d0 == 0.0:\n            # Top expert tied between options: endorsement undefined.\n            continue\n        endorsed_a = d0 > 0.0\n        try:\n            chose_a = (int(resp) == 0)\n        except Exception:\n            continue\n        follows.append(1.0 if (endorsed_a == chose_a) else 0.0)\n\n    if len(follows) == 0:\n        return 0.5\n    return float(np.mean(follows))",
  "rationale": "METRIC: endorsement-follow rate of the 0.99-validity expert (feature 0), computed over every trial where that expert casts a non-tied vote (trials 1-19 of the 24 unique pairs, i.e. ~76 of 96 trials per subject). For each such trial we record 1 if the subject chose the option the 99% expert endorsed, 0 otherwise, and average.\n\nWHY IT DISCRIMINATES: The two theories share every weight except w(0.99). On this panel (0.99 atop 0.94, gap = 0.05 at the spurious-precision peak, coherent panel so the composition amplifier is sub-threshold for both), BODT's bounded saturating shrinkage yields w(0.99) in [-0.14, +0.15] across its entire box \u2014 the 99% endorsement is inert, so the follow rate is pinned near chance (~0.47-0.50 given the trial mix). CREDT's unbounded additive discount rho*(P + C*sig_gap) yields w(0.99) in [-1.03, -0.61] \u2014 the endorsement is strongly ANTI-diagnostic, so subjects go against it, giving a follow rate ~0.30-0.36. Because the follow statistic is defined relative to the endorsed option (not to side A/B), the shared plateau/chance machinery (w_p, gated w_c, beta, lapse) largely cancels: on masked trials (s = w(0.99) alone) the two theories give 0.50 vs ~0.35; on R-ladder, Sc, O2, V1 and Ddeep cells the same w(0.99) gap produces follow-rate gaps of ~0.12-0.17 per cell in the SAME direction (anti-following under CREDT). Pooling all 19 voting cells averages these gaps (~0.15) while cutting per-subject binomial noise to sqrt(p(1-p)/76) \u2248 0.055.\n\nEXPECTED NUMBERS: BODT-simulated data: follow rate \u2248 0.47-0.50 (per-subject SD \u2248 0.055). CREDT-simulated data: follow rate \u2248 0.30-0.36 (per-subject SD \u2248 0.052). Mean gap \u2248 0.14, Welch t at N=25 \u2248 8-9, p << 0.001. The contrast is robust to side bias (endorsement direction is balanced across the schedule) and to the shared lapse/temperature parameters, which affect both theories' follow rates nearly identically because the trial set is mirror-balanced around the disputed weight.\n\nINTERPRETATION: A human follow rate near 0.5 means the 99% expert's endorsement is worth ~nothing (bounded saturation \u2014 BODT); a rate near 0.33 means subjects systematically contradict it (unbounded additive discount \u2014 CREDT). Rates in (0.38, 0.45) would strain both boxes."
}
```

## Usage

```json
{
  "input_tokens": 11956,
  "output_tokens": 18857,
  "reasoning_tokens": 17868,
  "provider": "Sail Research"
}
```
