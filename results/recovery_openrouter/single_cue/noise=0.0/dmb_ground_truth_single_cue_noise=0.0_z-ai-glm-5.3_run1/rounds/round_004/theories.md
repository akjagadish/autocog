# Round 4 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_6` — KILLED ✗

**Description:** Broad-Spectrum Skepticism Tallying (BSST), low-gain/held-cliff calibration. People integrate ALL binary expert ratings as sign votes (+1/-1/0) into a zero-sum evidence score, where the subjective weight w(v) is a wide inverted-U over STATED validity with skepticism on both flanks: (i) chance experts (v = 0.5) carry a small negative weight (~0.23 trusted-cue units) — a coin-flipper's endorsement is mildly anti-diagnostic; (ii) the entire band v ~ [0.55, 0.95] is a flat trust plateau (~0.9), making mid-band experts nearly exchangeable and generating moderate unweighted-tally behavior; (iii) distrust of claimed certainty is CONCENTRATED in a very sharp cliff between 0.98 and 0.99: w(0.98) stays mildly positive while w(0.99) plunges to ~ -1.1 trusted-cue units. The calibration claim added this round is about the GAIN-FLANK DECOUPLING: the softmax gain beta can be lowered (0.34) while the chance-flank magnitude kappa_lo is raised (0.23) so that the product beta*kappa_lo (~0.077) — the quantity that pins the chance-cue signature — is held invariant, but the product beta*|w(0.99)| (~0.37) is allowed to FALL because kappa_hi is held at its iter-2 value rather than deepened. This is the one combination never tried in nine iterations: every prior beta reduction (iters 5, 9) was bundled with a kappa_hi increase that exactly cancelled the gain on the 0.99-driven cells.

**Rationale:** Minimal-diff edit of the ACCEPTED iter-2 base: the code is byte-identical; only three parameter ranges move, implementing the critic's iter-9 recommendation — the ONE combination never tried in nine iterations. Every previous beta reduction (iters 5 and 9) was bundled with a kappa_hi INCREASE, so the product beta*|w(0.99)| never actually fell; here kappa_hi is HELD at iter-2's 1.70 while beta drops from ~0.40 to ~0.34 and kappa_lo rises from ~0.19 to ~0.2275 so the product beta*kappa_lo stays at ~0.077 (the iter-2 value that produced Exp 5's -0.155). Net effect: beta*|w(0.99)| falls from ~0.44 to ~0.37 while every other driving quantity is approximately invariant.

Expected movement, with the invariance structure stated explicitly: (1) Exp 7 (0.685 vs real 0.580): the lone-99 anti-follow logit 2*beta*|w(0.99)| shrinks from ~0.88 to ~0.75, and the mixed cells soften proportionally -> ~0.63-0.66 (squared-error gain ~0.004). (2) Exp 8 (0.436 vs real 0.4625): the D-cell contrast logit beta*(w(0.98)-w(0.90)) ~ 0.25*beta shrinks toward zero -> follow ~0.45-0.47. (3) Exp 2 (0.119 vs real 0.077): both regression slopes scale with beta -> ~0.10. (4) Exp 1 (-0.254 vs real -0.218): iter-9 already confirmed beta-down is this metric's friend -> ~-0.23. (5) Exps 3/4: the flank-flipped small-margin trials soften toward 0.5 from below, and unlike iter-9 the 0.99 lobe is NOT deepened (kappa_hi frozen), removing the channel that dragged iter-9's Exp 3 to 0.448. (6) Exp 5: held by the invariant product -> ~-0.15. (7) Exp 6: the K-cells shallow slightly (K2 logit ~-0.29 -> ~-0.22, strong-side +0.01-0.02) while the R3 cell loosens (logit ~0.95 -> ~0.80), netting roughly neutral at ~0.42-0.43; I accept this bounded cost and do NOT chase 0.118 — the K-cell logit remains an algebraic sum of the two products pinned by Exps 5 and 7, and R3 is pinned whenever w(0.75) >= 0 > w(0.5).

Range discipline: the three moved parameters get tight ranges (beta +/-0.01, kappa_lo +/-0.0075, kappa_hi +/-0.02) so the realized products beta*kappa_lo (0.073-0.082) and beta*|w(0.99)| cannot drift — the one lesson from iters 6-9 that is unambiguously vindicated (wide ranges destroyed the Exp 5/7 signatures in iter-5). The frozen parameters keep their verbatim iter-2 ranges, since those ranges produced the accepted 0.1811 operating point. All values sit inside the arbiter's prescribed BSST box (beta in [0.2, 0.45], kappa_lo in [0.08, 0.3], kappa_hi in [1.2, 2.8]). Per the critic's stop condition: if this candidate also fails to beat 0.1811, iter-2 is the ceiling of the linear sign-vote BSST family and the loop should terminate with iter-2 shipping.

**Parameters:**
  - `v_rise`: `[0.515, 0.54]`
  - `w_plateau`: `[0.88, 0.93]`
  - `v_c`: `[0.9855, 0.9875]`
  - `s_hi`: `[0.0025, 0.0033]`
  - `kappa_lo`: `[0.22, 0.235]`
  - `kappa_hi`: `[1.68, 1.72]`
  - `beta`: `[0.33, 0.35]`
  - `epsilon`: `[0.05, 0.07]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # Broad-Spectrum Skepticism Tallying (BSST), sharp-late-cliff edition.
    # Stimulus: array of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    # Each feature j casts a sign vote sign(a_j - b_j); the vote is
    # weighted by a wide inverted-U 'skepticism' function of the
    # expert's STATED validity v:
    #   v <= v_rise :  w = -kappa_lo * (v_rise - v)/(v_rise - 0.5)
    #                  (chance experts: linear lobe, w(0.5) = -kappa_lo;
    #                   a coin-flipper's endorsement is mildly
    #                   anti-diagnostic noise)
    #   v >  v_rise :  w = w_plateau - (w_plateau + kappa_hi)
    #                        * sigmoid((v - v_c)/s_hi)
    #                  (flat trust plateau over the mid-band -- the
    #                   logistic is ~0 for every v <= 0.95, so all
    #                   mid-band experts get the full plateau weight;
    #                   then an extremely SHARP distrust cliff packed
    #                   between 0.98 and 0.99: w(0.98) stays mildly
    #                   positive, w(0.99) plunges to ~ -1.1 trusted-cue
    #                   units, saturating at -kappa_hi as v -> 1)
    # Votes are summed into a zero-sum evidence score; choice is a
    # low-gain softmax with inverse temperature beta plus a uniform
    # lapse epsilon.
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

    v_rise = float(parameters["v_rise"])
    w_plateau = float(parameters["w_plateau"])
    v_c = float(parameters["v_c"])
    s_hi = float(parameters["s_hi"])
    kappa_lo = float(parameters["kappa_lo"])
    kappa_hi = float(parameters["kappa_hi"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    a, b = stim[0], stim[1]

    # Feature-wise votes: +1 where A wins the feature, -1 where B
    # wins, 0 on ties (a tied cue is uninformative for the pair).
    votes = np.sign(a - b)

    v = np.clip(val, 0.5, 1.0)
    w = np.empty(n_features, dtype=float)

    # (i) Low flank: chance experts carry a small negative weight,
    # linearly reaching -kappa_lo at v = 0.5 and 0 at v_rise.
    lo = v < v_rise
    denom_lo = max(v_rise - 0.5, 1e-6)
    w[lo] = -kappa_lo * np.clip(v_rise - v[lo], 0.0, None) / denom_lo

    # (ii)+(iii) Flat trust plateau followed by an extremely sharp
    # distrust cliff concentrated between v = 0.98 and v = 0.99.
    # In the mid-band the logistic term is ~0, so w ~ w_plateau
    # (nearly equal weights -> tally-like counting); at 0.98 the
    # weight is only mildly discounted, at 0.99 it is strongly
    # anti-diagnostic, and it saturates at -kappa_hi as v -> 1.
    hi = ~lo
    # Numerically stable logistic.
    x = (v[hi] - v_c) / s_hi
    sig = np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)),
                   np.exp(x) / (1.0 + np.exp(x)))
    w[hi] = w_plateau - (w_plateau + kappa_hi) * sig

    # Weighted evidence score. B's score is the mirror of A's
    # because every vote is zero-sum across the two options.
    s_a = float(np.dot(w, votes))
    scores = np.array([s_a, -s_a])

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


### slot 2 — `pi_4` — SURVIVED ✓

**Description:** Flattened-Diagnosticity Evidence Integration (the extreme-compression corner of the Bayesian cue-counting family): people integrate ALL binary expert ratings into one evidence score, where each discriminating cue casts a sign vote weighted by w_j = max(log(v_j/(1-v_j)) - tau, 0)^delta — chance cues (v=0.5) are annihilated exactly, but among genuinely diagnostic cues the subjective weight is a HEAVILY flattened (delta ~ 0.15) version of the normative log-odds, and choice is a low-gain softmax (2*beta ~ 0.6) plus a small uniform lapse. The new theoretical claim is that the compression exponent delta is itself the structural margin-compression device: shrinking delta equalizes weights, which GROWS count-dominated conflict margins (where cues split across options) while SHRINKING alignment-dominated cumulative margins (where several strong cues pile onto one side). This differential action — not any global gain knob, and not trial-level normalization — is what lets a single small beta simultaneously produce moderate TTB-leaning conflict-trial behavior (Exp 1), a slightly positive tally-vs-top-cue regression slope (Exp 2), graded ~0.62 tally-following (Exp 3), and near-chance |d|=1 agreement driven by the cancellation between super-expert matches and super-expert reversals (Exp 4).

**Rationale:** MINIMAL-DIFF EDIT: the code is the accepted iter-2 base VERBATIM; the only change is the parameter box (delta, tau, beta, epsilon). I accept the critic's iter-3 DIAGNOSIS in full (Exps 3/4 are saturated by large cumulative margins; Exps 1/2 need more effective gain; no global beta/epsilon can serve both) but I reject its prescribed FIX (kappa normalization) on quantitative grounds, and I reject the delta floor of ~0.35 as well. (1) WHY KAPPA HAS THE WRONG SIGN. Normalization divides the margin by (sum of discriminating weights)^kappa. Exp 1's conflict trials have 3-4 discriminators splitting across options (net weighted margin ~0.6-0.9, total weight ~3.6-4.2), so kappa=1 compresses their effective signal to ~0.17-0.25 — exactly the trials that need MORE gain. Exp 4's single-super-expert |d|=1 types have one discriminator, so m_norm = w0/w0^kappa = w0^(1-kappa), i.e. exactly 1 at kappa=1 — unsaturable only at beta ~ 0.1, which collapses Exp 1 to ~-0.01. Kappa compresses the trials that need gain and spares the trials that need compression: the opposite of the required differential action. (2) THE CORRECT IN-FAMILY STRUCTURAL KNOB IS DELTA ITSELF. Shrinking delta from 0.5 to 0.15 equalizes weights: count-dominated conflict margins GROW (Exp 1 type-1: -0.62 -> -0.91, because the tally side's cue count reasserts itself) while alignment-dominated cumulative margins SHRINK (Exp 4's super-expert sum W = w0+w1+w2: 4.8 -> 3.4). The ratio W/conflict drops from ~5 to ~2.6, so ONE small gain (2*beta ~ 0.6) puts Exp 1 conflicts at tanh ~ 0.27-0.49, Exp 4 matches at sigma ~ 0.65-0.82, and — critically — Exp 4's reversed types (three super-experts opposing the raw tally) at 1-sigma(2*beta*W) ~ 0.10-0.16, whose cancellation against the matches is what holds the |d|=1 metric near 0.51 instead of the saturated 0.58-0.60 of iter 2. (3) VALIDATION OF THE ALGEBRA. Applying the same margin-level computation at the ACCEPTED iter-2 box center (delta 0.4, beta 0.65) reproduces that candidate's actual outputs almost exactly: predicted Exp 1 = -0.288 vs actual -0.2795; predicted Exp 4 = 0.583 vs actual 0.5796. The algebra is therefore trustworthy, and it identifies (delta ~ 0.15, beta ~ 0.3) as the simultaneous optimum: Exp 1 ~ -0.21 to -0.22 (target -0.218), Exp 2 ~ +0.06 (target +0.077; near-equal weights keep b_d positive while the small cue-0 tilt keeps b_s small), Exp 3 ~ 0.61-0.63 (target 0.617), Exp 4 ~ 0.52 (target 0.513). (4) WHY THIS IS NOT OSCILLATION. The delta box [0.10, 0.20] extends the ACCEPTED base's [0.20, 0.6] downward — it is not a return to the critic's rejected delta~1 regime — and the annihilation property (llr(0.5)=0 <= tau for every tau in the box) is preserved everywhere, so the validated Exp 4 mechanism survives intact. The beta box [0.27, 0.37] is the critic's requested restoration of gain, rescaled to the ~2.5x smaller margins that delta-compression produces (equivalent effective gain to the critic's [0.35, 1.0] at delta 0.5). Expected residuals: |Exp1| ~ 0.01, |Exp2| ~ 0.02, |Exp3| ~ 0.01-0.03, |Exp4| ~ 0.01 — every one at or below the accepted base's residuals (0.062, 0.026, 0.050, 0.066), with the two dominant errors (Exp 1 and Exp 4) essentially eliminated.

**Parameters:**
  - `delta`: `[0.1, 0.2]`
  - `tau`: `[0.0, 0.04]`
  - `beta`: `[0.27, 0.37]`
  - `epsilon`: `[0.0, 0.05]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # Diagnosticity-Weighted Evidence Integration (Bayesian cue-counting).
    # Stimulus: array of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are binary expert ratings (0/1).
    # Each feature j casts a sign vote sign(a_j - b_j); the vote is
    # weighted by the cue's log-likelihood-ratio diagnosticity
    # llr_j = log(v_j / (1 - v_j)), rectified by an attention floor
    # tau and compressed by an exponent delta:
    #   w_j = (max(llr_j - tau, 0))^delta
    # A v = 0.5 expert has llr = 0 -> exactly zero evidence.
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

    delta = float(parameters["delta"])
    tau = float(parameters["tau"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    a, b = stim[0], stim[1]

    # Feature-wise votes: +1 where A wins the feature, -1 where B
    # wins, 0 on ties (a tied cue is uninformative for the pair).
    votes = np.sign(a - b)

    # Normative diagnosticity: log-likelihood-ratio of each expert.
    # Clip away from v = 1.0 to keep the log finite; v = 0.5 maps to 0.
    v = np.clip(val, 0.5, 1.0 - 1e-9)
    llr = np.log(v / (1.0 - v))

    # Limited-attention floor: diagnosticity below tau is discarded;
    # the remainder is compressed by the exponent delta. delta = 1 is
    # the purely normative Bayesian weight; delta < 1 flattens the
    # diagnosticity gradient toward tallying over informative cues
    # (while v = 0.5 cues stay annihilated at any delta, since
    # llr(0.5) = 0 <= tau); delta > 1 sharpens it toward
    # lexicographic/TTB-like use of the best experts.
    w = np.power(np.maximum(llr - tau, 0.0), delta)

    # Weighted evidence score. B's score is the mirror of A's because
    # every vote is zero-sum across the two options.
    s_a = float(np.dot(w, votes))
    scores = np.array([s_a, -s_a])

    # Numerically stable low-temperature softmax. When all attended
    # cues tie (s_a == 0) the softmax is exactly uniform, which is the
    # correct behavior for undiscriminating evidence.
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

### `pi_7` → slot 1 (via `new_theory`)

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
