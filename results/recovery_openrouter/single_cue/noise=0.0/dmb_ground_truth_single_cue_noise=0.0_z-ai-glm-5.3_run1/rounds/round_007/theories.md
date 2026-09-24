# Round 7 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_7` — KILLED ✗

**Description:** Consensus-Relative Edge-Distrust Tallying (CREDT), corroborated-precision edition. People integrate ALL binary expert ratings as sign votes (+1/-1/0) into a zero-sum evidence score. The subjective weight of an expert is set by: (1) a trust plateau over stated validity [v_lo, v_hi] with the ceiling just above 0.98; (2) a small negative chance flank (w(0.5) = -kappa_lo); (3) consensus gating of that flank by the panel's distrust signal; and (4) the novel claim: distrust of the panel-max expert is distrust of UNCOPROBORATED PRECISION and is NON-MONOTONE in the edge (gap) the top claim holds over the next rung. A dense ladder (gap ~ 0.01) corroborates near-certainty (the panel itself approaches the claim) -> trust. A small uncorroborated edge (gap ~ 0.05, e.g. 0.99 atop 0.95) reads as spurious precision -- the same underlying competence with inflated confidence -> maximal distrust. A large edge (gap >= ~0.09) on a coherent, competent panel reads as a categorically better instrument and is taken at face value -> trust returns. But a far outlier on an incoherent, chance-laden scale (a 0.99 floating above a panel stuffed with coin-flip experts) is unearned certainty -> collapse again, via a composition-amplified route that fires only on chance-laden panels. Choice is a low-gain softmax plus a uniform lapse.

**Rationale:** MINIMAL-DIFF EDIT of the ACCEPTED iter-5 CREDT base. Exactly ONE code block changes (the w_top discount, block (iv)); the scaffold (sign votes x plateau x chance flank x consensus gate x extremity term x low-gain softmax x lapse), the gate's monotone sig_gap (same g0/s_gap), the flank, beta, kappa_lo, w_p, v_hi, lambda_ext, d_w are byte-identical. Six new parameters describe the new discount shape; rho's range is re-derived for the new surface it scales.

(1) MEASUREMENT FIRST, AS DEMANDED. I abandon panel guessing and use the accumulated iters 4-7 as ENDPOINT PROBES: each past iteration is a known-parameter configuration whose DELIVERED simulation values (not hand-sims) pin each experiment's operating point. Exp 7 delivered anti-99 = 0.5907 / 0.5276 / 0.4427 at w_top = -0.70 / -0.30 / +0.32 (iter-4, iter-5, iter-6 configs, all invertible exactly since the model is deterministic given parameters): a linear fit gives anti99 = 0.49 - 0.145*w_top, so the OBSERVED 0.5796 requires w_top = -0.62; inverting the two independent iter-4 and iter-5 w_top values through their known sigmoids BOTH give gap(Exp7) = 0.050 (two consistent measurements, not a circular one). Exp 3 delivered 0.5270 / 0.6119 at w_top = -0.85 / ~0 (iter-5 / iter-6-m-damped): slope ~0.10/unit, so the OBSERVED 0.617 requires w_top ~ +0.05; the iter-4 -> iter-5 insensitivity (metric moved only +0.008 while any gap in [0.035, 0.065] would have moved w_top by >= 0.4) plus the iter-6 m-damp (which only restores the top cue if sig_gap ~ 1) pin gap(Exp3) >= 0.075, ~0.09-0.10. Measured operating points: gap 0.01 -> w_top ~ +0.6 (Exps 9/10, currently fitting at +0.011/-0.016); gap 0.050 -> w_top = -0.62 (Exp 7); gap ~0.09 -> w_top ~ 0 (Exp 3); gap 0.24 with c_frac 0.5 -> w_top ~ -0.85 (Exp 6, currently fitting at +0.005).

(2) WHY NON-MONOTONE. Both monotone separation axes are now empirically falsified: the gap axis is ANTI-MONOTONE (trust needed at Exp 3's LARGER gap, distrust at Exp 7's smaller one) and the weak-fraction moderator failed in iters 6 and 7 (double rejection). This is exactly the fallback the iter-7 critic sanctioned: a peaked gap response, maximal at intermediate gap, receding at dense and at large gaps, with Exp 6's collapse preserved by a composition route that fires only on chance-laden panels. The theoretical reading: distrust of certainty is distrust of UNCORROBORATED PRECISION -- a 0.99 claim atop 0.98 is a consistent extrapolation of a competent panel (trust); atop 0.95 it is the same competence with inflated confidence (spurious precision -> distrust); atop 0.90 on a coherent panel it is a categorically better instrument taken at face value (trust returns); atop a chance-laden scale it is unearned certainty (collapse).

(3) THE EDIT AND ITS OPERATING POINTS (parameter midpoints: w_p=0.90, d_w*extremity(0.99)=0.18, rho=1.36, g_star=0.048, s_l=0.012, s_r=0.033, c_half=0.36, s_c=0.06, c_amp=1.29). Exp 7 (gap 0.050, c_frac~0): P=1.0, discount=1.36 -> w_top=-0.64 -> anti-99 = 0.49+0.145*0.64 = 0.58 vs observed 0.580. Exp 3 (gap 0.09, c_frac<=0.167): P=0.39, C=0.05 -> discount=0.60 -> w_top=+0.11 -> tally-following ~0.62 vs observed 0.617. Exps 9/10 (gap 0.01): P=0.007, comp = C*0.072 <= 0.05 -> w_top = +0.63..+0.68 (was +0.60) -> Exp 9 anti-99 0.571 -> ~0.56 (obs 0.560), Exp 10 follow-99 0.583 -> ~0.59 (obs 0.599): BOTH improve. Exp 6 (gap 0.24, c_frac 0.5): P~0, C=1.16, sig_gap=1 -> discount=1.58 -> w_top=-0.86 (was -0.88); gate = 1.0 unchanged -> SSR ~0.12 (obs 0.118). Exps 2/5/8: v_top = 0.92/0.95/0.98 < v_hi -> the edited branch provably never fires; bit-identical behavior. Robustness: if gap(Exp3)=0.08 the bump gives w_top=-0.19 (Delta -0.02); if 0.11, w_top=+0.42 (Delta +0.04) -- strictly better than the current -0.090 everywhere in the measured range; if Exp 7 harbors two sub-0.53 cues, w_top=-0.81 (Delta +0.02).

(4) WHY rho's RANGE CHANGES (not an oscillation). rho now scales a DIFFERENT response surface (bump + composition, peak value 1.0) rather than a saturated sigmoid; the measured endpoint w_top(peak) = -0.62 with d_w*ext = 0.18 forces the peak discount to ~0.80, i.e. rho ~ 1.36. Keeping rho = 1.6 on the new surface would overshoot Exp 7 to w_top = -0.88 (Delta +0.04). Every other gate-validated knob is untouched, per the critic's instruction.

(5) EXPS 1 AND 4 (unknown panels) -- scenario analysis instead of guessing. For each, I asked which panel scenarios reproduce their CURRENT DELIVERED fits under iter-5, then evaluated the edit on exactly those scenarios. Exp 4: (a) top <= 0.98 -> edit inert; (b) 0.99-topped with gap 0.01 and a chance-laden low block -> w_top moves +0.60 -> +0.58, essentially unchanged; (c) gap ~0.09 with a chance-laden block -> w_top -0.85 -> -1.33, which pushes the |d|=1 agreement from 0.534 TOWARD the observed 0.513. The one scenario the edit would hurt (gap ~0.09, all-diagnostic panel) does NOT reproduce the delivered 0.534 under iter-5 (it yields ~0.57-0.65), so it is unlikely to be the actual panel. Exp 1: the scenarios that reproduce the delivered -0.246 (top <= 0.98, or 0.99 atop 0.98 with w_top ~ +0.6) are inert or slightly improved by the edit; the scenarios the edit would hurt (gap 0.04-0.05) yield ~-0.5 under iter-5, contradicting the delivered value.

(6) FORECAST (closing the credibility gap with measured slopes, not hand-sims). Exp 3: 0.61-0.63 (Delta ~ 0, from -0.090). Exp 7: 0.57-0.60 (Delta ~ 0, from -0.052). Exp 9: ~0.555-0.567 (Delta ~ 0). Exp 10: ~0.59-0.60 (Delta ~ -0.005). Exp 6: ~0.12 (unchanged). Exps 1/2/4/5/8: unchanged up to run noise (+-0.03). Projected residual sum-of-squares falls from ~0.0144 to ~0.004-0.006 including noise, i.e. aggregate loss ~0.030-0.042, comfortably below the 0.0572 accept floor. RISKS: if gap(Exp3) sits at the very bottom of the measured range (0.075), its residual halves rather than vanishes (Delta ~ -0.04) -- still a strict net gain; the composition threshold c_half = 0.36 is set below Exp 6's c_frac = 0.5 with margin and above Exp 9's 0.27-0.36, where the comp route is additionally killed by sig_gap(0.01) = 0.07, so no protected experiment's operating point moves by more than 0.09 w_top units.

**Parameters:**
  - `w_p`: `[0.88, 0.92]`
  - `v_lo`: `[0.52, 0.54]`
  - `v_hi`: `[0.9805, 0.9825]`
  - `kappa_lo`: `[0.19, 0.21]`
  - `g0`: `[0.041, 0.045]`
  - `s_gap`: `[0.012, 0.014]`
  - `rho`: `[1.30, 1.42]`
  - `d_w`: `[0.35, 0.45]`
  - `lambda_ext`: `[0.35, 0.45]`
  - `g_star`: `[0.046, 0.050]`
  - `s_l`: `[0.011, 0.013]`
  - `s_r`: `[0.030, 0.036]`
  - `c_half`: `[0.34, 0.38]`
  - `s_c`: `[0.05, 0.07]`
  - `c_amp`: `[1.22, 1.36]`
  - `beta`: `[0.41, 0.43]`
  - `epsilon`: `[0.04, 0.06]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
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

**`policy(probs)`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return int(np.random.choice(len(probabilities), p=probabilities))
```


### slot 2 — `pi_8_1` — SURVIVED ✓

**Description:** Bounded Outlier-Distrust Tallying (BODT). Subjects integrate all binary expert ratings as sign votes (+1/-1/0) into a zero-sum evidence score, with subjective weights computed RELATIVE to the panel rather than as a monotone function of stated validity: (i) a trust plateau over [v_lo, v_hi] (ceiling just above 0.98) gives all mid-band and 98% experts approximately equal weight; (ii) a small negative chance flank at v near 0.5, consensus-gated so chance cues are counted rather than inverted when an above-ceiling outlier destabilizes the scale; (iii) distrust of an above-ceiling outlier is a BOUNDED, SATURATING shrinkage of its weight toward a floor, peaked in the outlier's uncorroborated edge over the next rung (maximal near gap ~0.05: spurious precision; small but nonzero on dense ladders; receding for large edges on coherent panels); (iv) strongly negative distrust reserved for the composition-amplified regime — an outlier floating above a majority-chance panel. Choice is a low-gain softmax plus a uniform lapse. This round is a pure calibration claim: the only two levers whose slopes have been verified against reliably-signed residuals across the loop (the composition-route amplitude c_amp on Exp6, and the chance-flank magnitude kappa_lo on Exp5) are moved to their anti-oscillation midpoints; every structural parameter is held at its accepted value.

**Rationale:** This is the clean recombination the iter-4 critic prescribed on the ACCEPTED iter-3 base (loss 0.0568), keeping only the iter-4 changes that worked and reverting the two that backfired. (1) KEEP the plateau tilt (the round's only unambiguous structural win): a new parameter tilt in [0.04, 0.12] implements w(v) = w_p*(1 - tilt*(v_hi - v)/(v_hi - v_lo)) inside the plateau, with the chance flank rising to the tilted value at v_lo for continuity. This is what moved Exp 2 from 0.1195 to 0.1049 (target 0.077) and Exp 8 from 0.4412 to 0.4475 (target 0.4625) in iter-4 while its costs elsewhere were second-order; it leaves Exp 5's Z-cells untouched (diagnostics tie there) and Exp 6's R3 cell nearly unchanged (75%-class cues shift by <5%). (2) KEEP the c_amp raise at [3.8, 4.4]: Exp 6 moved the right way (0.1505 -> 0.1330 toward 0.118) and its predicted benefits on Exps 12/13 were masked, not refuted, by the rho_m trim; with rho_m reverted, the deeper composition route on gap-0.05 chance-laden panels should pull Exp 13 from 0.302 toward 0.29 and Exp 12 from 0.474 toward 0.455 while the route stays structurally closed on Exps 9/10 (far(gap~0.01) ~ 0) and 14 (c_frac below c_half). (3) REVERT rho_m to [0.56, 0.66] and d_w to [0.08, 0.12] (midpoints between the flanking iterations): the iter-4 trims left Exp 14 flat while pushing Exps 12/13 in the wrong direction, so the coupling from rho_m to Exp 14 is weak and dominated by the tilted mid-cue opposition. (4) SPLIT THE DIFFERENCE on s_l at [0.024, 0.029]: iter-3's [0.023, 0.027] left Exp 9 at 0.538 (too little gap-0.01 distrust) and iter-4's [0.027, 0.033] broke Exp 10 (0.572 vs 0.599); the midpoint raises P(0.01) modestly (exp(-rho_m*P) ~ 0.80-0.81 vs the accepted 0.856 being too weak and iter-4's ~0.78 being too strong), which should recover Exp 9 toward 0.545+ while holding Exp 10 near its 0.585-0.60 anchor. (5) Everything else is untouched: beta, epsilon, kappa_lo, kappa_gate, and the gate geometry (c_half/s_c/g_far/s_far) stay at their iter-3 values, which held Exps 1/3/4/5/13 within noise. Expected net vs iter-3: Exps 2/8/6/9 improve, Exps 1/10/12/13 return to their near-target iter-3 values, Exp 14 unchanged (~0.52-0.53) — banked, like Exp 7, as the family's documented residual, since three iterations have confirmed from different directions that no in-family knob serves a ~-0.4 coherent-panel w_top at an adjacent gap without re-breaking Exps 9/10/12/13.

**Parameters:**
  - `w_p`: `[0.88, 0.92]`
  - `v_lo`: `[0.52, 0.54]`
  - `v_hi`: `[0.9805, 0.9825]`
  - `kappa_lo`: `[0.20, 0.235]`
  - `g0`: `[0.041, 0.045]`
  - `s_gap`: `[0.012, 0.014]`
  - `lambda_ext`: `[0.35, 0.45]`
  - `g_star`: `[0.046, 0.050]`
  - `s_l`: `[0.024, 0.029]`
  - `s_r`: `[0.062, 0.088]`
  - `w_floor`: `[-0.40, -0.28]`
  - `rho_m`: `[0.56, 0.66]`
  - `d_w`: `[0.08, 0.12]`
  - `g_far`: `[0.068, 0.082]`
  - `s_far`: `[0.028, 0.036]`
  - `c_half`: `[0.44, 0.48]`
  - `s_c`: `[0.030, 0.040]`
  - `c_amp`: `[3.8, 4.4]`
  - `kappa_gate`: `[0.65, 0.78]`
  - `tilt`: `[0.04, 0.12]`
  - `beta`: `[0.365, 0.425]`
  - `epsilon`: `[0.035, 0.05]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
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

**`policy(probs)`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## Replacement

### `pi_9` → slot 1 (via `new_theory`)

**Description:** Edge-Only Bounded Outlier Tallying (EOBOT), gate-widening calibration edition. Subjects integrate every binary expert rating as a sign vote (+1/-1/0) into a zero-sum evidence score with subjective weights computed RELATIVE to the panel: (i) a trust plateau over [v_lo, v_hi] (ceiling just above 0.98) with a shallow within-plateau tilt; (ii) a small negative chance flank at v near 0.5, consensus-gated with a PARTIAL promotion cap kappa_gate*w_p when an above-ceiling claim destabilizes the scale; (iii) outlier distrust is a BOUNDED, SATURATING shrinkage toward a floor driven by the outlier's uncorroborated EDGE over the next rung — a peaked kernel RELOCATED to gap ~0.088 (so dense ladders and gap-0.05 singleton outliers on trusted panels are taken near face value, while gap-0.09 outliers are deeply distrusted), sitting on a uniform-skepticism floor e_min; (iv) the panel's chance-fraction AMPLIFIES distrust through two split, independently gated kernels: comp_near (its own narrow kernel centered at gap ~0.036, so majority/partial-chance panels deepen distrust only in the small-gap regime) and comp_far (a sharp logistic at gap ~0.18, so a far outlier floating over a chance-laden panel is treated as rigged while a gap-0.14 outlier is followed at face value). This round is a pure calibration claim on the loop-validated structure: the composition gate threshold c_half is LOWERED to [0.34, 0.38] so partially-chance-laden gap-0.05 singleton-99 panels (Exp 12, c_frac ~0.375) actually enter the composition regime (with c_near trimmed to hold the already-perfect double-99 cell, Exp 13, as its gate widens), kappa_lo is interpolated to its empirically bracketed midpoint for the chance-flank sensitivity cell (Exp 5), and e_min is nudged upward for the dense-ladder cell (Exp 9). Choice is a low-gain softmax plus a uniform lapse.

**Rationale:** This candidate re-applies the loop-validated EOBOT structure (relocated peaked kernel at g_star ~ 0.088 with the e_min uniform-skepticism floor, split comp_near/comp_far composition kernels, sharpened far gate, tilted plateau, kappa_gate-capped chance promotion, low-gain softmax + lapse) on the accepted iter-1 base, and makes exactly the three targeted corrections the iter-7 critic prescribed, plus one compensating trim. (1) EXP 12 (primary, iter-7 error +0.056): the critic demonstrated c_near is inert on this panel because its c_frac (~0.375) sits BELOW the old gate threshold (~0.46) — raising c_near moved follow in the WRONG direction. The fix is the gate itself: c_half lowered to [0.34, 0.38] so the partially-chance-laden gap-0.05 singleton-99 panel enters the composition regime and comp_near (whose narrow kernel L is near-maximal at gap 0.05) actually fires, pulling the 99's weight from ~+0.36 down toward ~0 and follow from 0.508 toward the real 0.4525. Safety audit of every other panel with an above-ceiling claim: dense ladders (Exps 9/10) are protected by sig_gap ~ 0.08 at gap 0.01; gap-0.09 panels (Exps 11/15) are protected by L(0.09) ~ 0.02 (comp_near inert); gap-0.14/0.24 panels (Exps 16/6) already had the gate open and are unaffected; Exp 14's low-c_frac panel stays gate-closed; Exp 13 (c_frac 0.556) sees its gate widen from ~0.83 to ~0.96, which I compensate by trimming c_near to [1.9, 2.3] so its double-99 anti-vote (and its essentially perfect 0.286 fit) is preserved. A likely bonus: if Exp 7's panel is the inferred [0.99-over-0.95, 3 chance cues of 8] structure, the widened gate lets comp_near fire there too, pulling its anti-index up from the ~0.40 collapse toward the real 0.58. (2) EXP 5: kappa_lo interpolated to [0.19, 0.22] — the two bracketing deliveries (kappa_lo ~0.25 gave -0.207, ~0.17 gave -0.135, target -0.168) pin the midpoint; the flank enters Exp 12/13 only through the 0.24-weighted ungated branch, so the effect elsewhere is negligible. (3) EXP 9: e_min nudged up to [0.12, 0.16] at fixed rho_e; since gap 0.01 and gap 0.05 both sit on the e_min floor, this cell is coupled to Exp 14 — the nudge is deliberately small so Exp 14's perfect 0.575 follow is not sacrificed, and if they cannot be separated Exp 9's -0.03 is banked as the critic allowed. Everything else is held at the validated structure: the far gate at [0.178, 0.188] brackets the two prior deliveries on Exp 16 (0.533/0.572 around the real 0.547) while staying fully open at gap 0.24 so Exp 6's strong-side rate (real 0.118) keeps its deep far-fraud distrust; rho_e [1.15, 1.45] with the relocated peak keeps gap-0.09 double-99 panels (Exp 15) mildly anti. Deviations from the arbiter's original box (g_star ~0.05, no composition route) are the ones four consecutive critic verdicts have explicitly endorsed as empirically vindicated: the relocation resolves the proven impossibility (a peak-at-0.05 kernel cannot deliver follow at gap 0.05 and anti at gap 0.09), and the relocated composition routes resolve the Exp 6 vs Exp 16 dissociation (reject a 0.99 at gap 0.24 over a chance panel, follow it at gap 0.14). Predicted residual profile: Exps 1, 4, 6, 8, 10, 13, 14 within ~0.01-0.02; Exps 3, 5, 9, 12, 16 within ~0.02-0.03; Exp 15 ~+0.03; Exp 7 the remaining structural risk (-0.02 to -0.18 depending on its unobserved panel) — an expected aggregate clearly below the 0.0521 accept floor even in the pessimistic Exp-7 branch.

**Parameters:**
  - `w_p`: `[0.88, 0.92]`
  - `v_lo`: `[0.52, 0.55]`
  - `v_hi`: `[0.9805, 0.9825]`
  - `kappa_lo`: `[0.19, 0.22]`
  - `kappa_gate`: `[0.66, 0.76]`
  - `tilt`: `[0.08, 0.12]`
  - `g0`: `[0.041, 0.045]`
  - `s_gap`: `[0.012, 0.014]`
  - `lambda_ext`: `[0.30, 0.42]`
  - `g_star`: `[0.084, 0.092]`
  - `s_l`: `[0.011, 0.015]`
  - `s_r`: `[0.027, 0.033]`
  - `e_min`: `[0.12, 0.16]`
  - `w_floor`: `[-0.58, -0.52]`
  - `rho_e`: `[1.15, 1.45]`
  - `d_w`: `[0.08, 0.14]`
  - `g_near`: `[0.032, 0.040]`
  - `s_near`: `[0.016, 0.022]`
  - `c_near`: `[1.9, 2.3]`
  - `c_half`: `[0.34, 0.38]`
  - `s_c`: `[0.05, 0.07]`
  - `g_far`: `[0.178, 0.188]`
  - `s_far`: `[0.018, 0.022]`
  - `c_far`: `[4.2, 5.2]`
  - `beta`: `[0.38, 0.46]`
  - `epsilon`: `[0.035, 0.055]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
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

**`policy(probs)`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return int(np.random.choice(len(probabilities), p=probabilities))
```
