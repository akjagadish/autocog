# Round 8 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_9` — SURVIVED ✓

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


### slot 2 — `pi_8_1` — KILLED ✗

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

### `pi_10` → slot 2 (via `new_theory`)

**Description:** SG-CGD (Small-Gap Composition-Gated Outlier Distrust), half-depth uniform-skepticism edition. Subjects integrate every binary expert rating as a sign vote into a zero-sum evidence score with subjective weights computed relative to the panel: a tilted trust plateau over [v_lo, v_hi] (ceiling just above 0.98), a consensus-gated chance flank with partial-ceiling promotion, and an outlier branch for above-ceiling claims consisting of (a) a bounded, saturating base distrust peaked at gap ~0.088 sitting on a uniform-skepticism floor, (b) a near-kernel composition amplification (narrow Gaussian at gap ~0.035, chance-fraction gate, sig_gap-throttled) that deepens distrust only of small-edge outliers on chance-laden scales, (c) a far-fraud route at gap ~0.185, and (d) a composition-gated edge-credibility ramp that promotes a well-separated 99% expert back toward plateau trust when the surrounding scale is unreliable. The single theoretical claim updated this round: the uniform-skepticism floor e_min — the residual distrust applied to a panel-max expert even when the gap kernel is silent (dense ladders, gap ~0.05 cells) — sits at a MODERATE depth (~0.36), roughly halfway between the near-zero floor of the accepted calibration and the deep floor of the rejected overshoot. This floor is what generates the empirically required mild anti-follow of 99% experts on dense validity ladders (Exp 7) without unbounded sign flips. Choice is a low-gain softmax plus uniform lapse.

**Rationale:** This is the critic-prescribed half-depth re-edit of the single e_min lever on the ACCEPTED iter-3 base. The mechanism code is byte-identical to the running-best candidate; exactly ONE parameter box moves (e_min: [0.13, 0.17] -> [0.32, 0.40], midpoint ~0.36, inside the critic's 0.30-0.45 window and roughly half the rejected iter-4 depth of ~0.59). The rejected iter-4 candidate's w_floor dip is NOT present here — w_floor stays at the accepted iter-3 box [-0.24, -0.17] — and every other parameter (near-route gate, credibility ramp, kernel placement, shared machinery, far-fraud route) is pinned to the accepted values, exactly as the iter-4 feedback instructs after two failed attempts established those are calibrated.

Pre-submission panel audit with the actual panel numbers, as requested (using box midpoints: w_p=0.90, w_floor=-0.205, rho_e=1.175, e_min=0.36):

- Dense ladders (gap ~0.01, kernel ~0, E = e_min = 0.36): w_base = -0.205 + 1.105*exp(-1.175*0.36) = -0.205 + 1.105*0.655 = +0.52. At the iter-3 floor (e_min=0.15) the same arithmetic gives +0.74, so the edit lowers dense-ladder w_top by ~0.22 per 99% expert. Exp 7 (anti-follow, real 0.580, iter-3 0.369): the two 99% experts' combined vote drops by ~0.44 trusted-cue units on the discriminating trials, moving anti-follow toward ~0.42-0.46 — roughly half of iter-4's gain (0.448) retained, as the endpoint interpolation predicts. Exp 9 (real 0.560, iter-3 0.529): the small extra distrust moves it to ~0.55-0.57, i.e. ON target rather than iter-4's 0.640 overshoot. Exp 10 (real 0.599, iter-3 0.603): follow drops to ~0.56-0.57, inside the >=0.56 guardrail with margin.

- Gap-0.05 cells (kernel ~0, E = e_min): w_base moves +0.74 -> +0.52. Exp 13 (c_frac 0.556, G~0.77, comp_near ~1.15): w_top -0.50 -> -0.71, follow 0.320 -> ~0.26 (real 0.290) — the endpoint interpolation's landing zone. Exp 12 (c_frac 0.375, G~0.19, comp_near ~0.29): w_top ~0.38 -> ~0.17, follow 0.480 -> ~0.44 (real 0.4525). Exp 14 (coherent, c_frac 0.30, G~0.16): w_top 0.43 -> 0.23, follow 0.558 -> ~0.52-0.53, at the >=0.52 guardrail edge — the accepted cost. Exp 15 (c_frac ~0.60, gate fully open, comp_near ~1.4): the extra ~0.22 of floor distrust adds anti-follow on a cell that has been stuck at 0.42 vs real 0.483 — a small favorable drift, not chased.

- Kernel-dominated cells (gap ~0.09, E = kernel ~ 1): e_min is mechanically inert. Exps 17/18 are unchanged from the accepted base (0.619/0.423), as are Exps 16 (gap 0.14, kernel 0.51 > 0.36) and 6 (far-route dominated). No-outlier experiments (1-5, 8) are structurally untouched.

Where my audit and the critic's endpoint interpolation could disagree (Exps 13/14, where I project slightly past the interpolation's landing zones), I follow the critic's explicit instruction to trust the interpolation and choose the shallower side: e_min's box TOP is 0.40, not 0.45, so the deepest sampled subject stays at the interpolation's upper edge. Expected movement vs the accepted base: Exp 7 0.369 -> ~0.42-0.45, Exp 9 0.529 -> ~0.56, Exp 12 0.480 -> ~0.44, Exp 13 0.320 -> ~0.26-0.28, with Exps 3-6/8/10/11/16/17/18 held within ~0.01-0.04 — a projected aggregate loss near ~0.050-0.055, comfortably below the 0.0622 gate floor. Exps 15 and 17 remain accepted structural residuals this round per the feedback.

**Parameters:**
  - `w_p`: `[0.88, 0.92]`
  - `v_lo`: `[0.52, 0.55]`
  - `v_hi`: `[0.9805, 0.9825]`
  - `tilt`: `[0.09, 0.13]`
  - `kappa_lo`: `[0.19, 0.22]`
  - `kappa_gate`: `[0.66, 0.74]`
  - `lambda_ext`: `[0.30, 0.42]`
  - `g0`: `[0.041, 0.045]`
  - `s_gap`: `[0.012, 0.014]`
  - `g_star`: `[0.084, 0.092]`
  - `s_l`: `[0.013, 0.017]`
  - `s_r`: `[0.040, 0.050]`
  - `e_min`: `[0.32, 0.40]`
  - `w_floor`: `[-0.24, -0.17]`
  - `rho_e`: `[1.05, 1.30]`
  - `d_w`: `[0.08, 0.12]`
  - `g_near`: `[0.032, 0.038]`
  - `s_near`: `[0.014, 0.018]`
  - `c_near`: `[3.5, 4.0]`
  - `c_half`: `[0.44, 0.48]`
  - `s_c`: `[0.085, 0.105]`
  - `g_far`: `[0.180, 0.190]`
  - `s_far`: `[0.018, 0.022]`
  - `c_far`: `[20, 26]`
  - `c_half_far`: `[0.57, 0.63]`
  - `s_c_far`: `[0.06, 0.08]`
  - `c_cred`: `[1.15, 1.40]`
  - `g_edge`: `[0.070, 0.080]`
  - `s_edge`: `[0.018, 0.022]`
  - `c_cred_half`: `[0.57, 0.63]`
  - `s_cred`: `[0.05, 0.06]`
  - `beta`: `[0.38, 0.46]`
  - `epsilon`: `[0.035, 0.055]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
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

**`policy(probs)`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return int(np.random.choice(len(probabilities), p=probabilities))
```
