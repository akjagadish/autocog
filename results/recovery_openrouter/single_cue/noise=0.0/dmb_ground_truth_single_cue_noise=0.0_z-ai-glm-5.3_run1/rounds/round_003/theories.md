# Round 3 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_5` — KILLED ✗

**Description:** Skeptical (Inverted-Trust) Tallying with a CALIBRATED trust cliff. People integrate ALL binary expert ratings as sign votes (+1/-1/0) into a zero-sum evidence score whose per-cue weight is a non-monotone function of stated validity: (i) near-chance experts (v ~ 0.5) carry a small NEGATIVE weight (their endorsements are treated as mildly anti-diagnostic noise); (ii) trust rises smoothly to a peak at v ~ 0.9 (the most trusted experts); (iii) claimed near-certainty (v > ~0.96) falls past a cliff into a MODERATE negative weight of roughly MINUS ONE trusted-cue unit — a 99% expert is treated as approximately one reliable anti-cue, not a veto. The calibrating claim is that distrust magnitude is matched to trust magnitude (|w(0.99)| ~ w_peak ~ 1, logit contribution 2*beta*w(0.99) ~ -2.1): a single super-expert loses to any 2-cue opposing coalition (weak-side choice in super-expert conflicts) but cannot invert coalitions that include it (moderate tally-following when the 0.99 cue sits inside a tally majority). This intermediate depth sits deliberately between the previously falsified extremes: the deep veto (iter 6, which deterministically inverted Experiment 3) and the shallow discount (iter 7, which let super-experts win conflicts, SSR 0.69).

**Rationale:** MINIMAL-DIFF EDIT: predict/policy are UNCHANGED from the accepted iter-6 base; every change is parameter-box surgery placing the operating point at an INTERMEDIATE lobe depth, the exact direction the iter-7 critique identified as the untried regime between the two falsified extremes (iter-6 veto -> Exp 3 = 0.24; iter-7 discount -> Exp 6 SSR = 0.69).

(1) VALIDITY VECTORS (the critic's thrice-repeated request; I do not observe them directly, so these are deductions from the metric code and the loop's falsifications): Exp 1: 5 cues, descending, top ~0.9 (iter-2 critique), assumed [0.9, 0.8, 0.7, 0.6, 0.5]. Exp 2: 6 cues, top = 0.92 (stated in its metric). Exp 3: 6 cues, top >= 0.96, almost certainly 0.99 — iter-6's collapse to 0.24 with v_hi >= 0.96 and kappa_hi in [20,40] can only be caused by a lobed top cue (the lower lobe was capped at -0.45 and v_lo <= 0.58, which my iters-4/5 arithmetic showed cannot flip 72% of decisive trials). Exp 4: 8 cues, top <= 0.95 (Exp 4 was insensitive to lobe depth across iters 6->7: 0.384 -> 0.399). Exp 5: [0.95, 0.85, 0.75, 0.65, 0.5, 0.5, 0.5, 0.5] (stated in its metric). Exp 6: [0.99, 0.99, 0.75, 0.75, 0.50, 0.50, 0.50, 0.50] — the weak cues are v = 0.50 EXACTLY, proven by iter-7: with v_lo <= 0.510 and w(0.99) <= 0 everywhere in that box, SSR = 0.69 > 0.5 is possible only if the weak cues carry negative weight, i.e. lie below v_lo; hence weak = 0.50, the SAME class as Exp 5's chance cues. This kills the iter-6 'weak = 0.51' separation hope and fixes two identities: w(0.50) = -kappa_lo serves both Exp 5 (anti-following) and Exp 6 (negative weak-cue weights ADD to the strong side's score, the sign flip behind iter-7's 0.69).

(2) THE INTERMEDIATE LOBE (critic fix #1, adopted): v_hi in [0.957, 0.963], tau in [0.011, 0.014], kappa_hi in [1.0, 1.3] give w(0.99) = -kappa_hi*(1-exp(-(0.99-v_hi)/tau)) in [-1.24, -0.85] at every corner (center -1.05). In LOGIT units 2*beta*w(0.99) ~ -2.1 at beta ~ 1, matching the critic's intended T = 2*beta*w(0.99) ~ -2.2 (their weight-unit target [-2.0, -1.2] assumed beta ~ 0.7; I run beta ~ 1.0, so the shallower weight lands the same logit). The two defining inequalities hold at every corner: K2 logit = T + 4*beta*kappa_lo in [-1.3, -2.3] (K2 in [0.09, 0.22], weak-side leaning) and no Exp 3 cell margin goes below t + 0.72 ~ -1.9 except the two m2 cells that real data also suggest are weakly followed.

(3) WHY I DEVIATE FROM THREE OF THE CRITER'S RANGES (with arithmetic): (a) beta [0.5,0.9] -> [0.9,1.1]. Exp 6 pins T = 2*beta*t ~ -2.1 (K2 = sigma(T+4*beta*kappa_lo) ~ 0.15). Exp 3's five 'pure' cells have average logit pinned at ~T (the mid-coalition deltas cancel in expectation: 2*m1+2*m2+m3 = 5t + w1-3*w3-w4+3*w5 ~ 5t), so Exp 3 can only reach 0.60 by pushing the four NON-pure cells (top-against, |d|=3) to 0.85-0.95, which needs 2*beta*delta ~ 2-2.5 with delta <= ~1.9 peak units -> beta >= 0.9. At beta = 0.7 (t = -1.5): Exp 3 = 0.43; at beta = 1.0 (t = -1.05): 0.50. (b) kappa_lo [0.12,0.20] -> [0.075,0.105]. Exp 5's chance-only trials have |k| in {1,2,3} with frequencies 2/4/2; P(follow) = 0.416 requires E[sigma(2*beta*kappa*|k|)] = 0.589 -> 2*beta*kappa ~ 0.18 -> beta*kappa ~ 0.09. With beta in [0.9,1.1], kappa_lo in [0.075,0.105] holds Exp 5 in [-0.20,-0.13]; the critic's range (calibrated for beta <= 0.9) would overshoot to -0.28 at beta = 1.1. (c) rho1 [0.15,0.45] -> [0.68,0.75]. Exp 1's conflict metric is binding: P1 margin = w(0.9)-w(0.8)-w(0.7) must be ~ -0.41 (p(TTB) ~ 0.31) to reproduce -0.218 given P7 = w(0.9)+kappa_lo ~ +1.09 and P9 = w(0.8)-w(0.7) ~ +0.2. That requires a RISING band (w(0.7) ~ 0.60 vs w(0.9) = 1.0), i.e. rho1 ~ 0.7. At rho1 = 0.3, w(0.7) = 0.81 -> P1 margin -0.68 -> Exp 1 ~ -0.34 (Delta = -0.12). The critic's rho1-widening was premised on Exp 4's low cues being 0.51-0.55; they are 0.50 (below v_lo, handled by kappa_lo), so rho1 is free to serve Exp 1. (d) v_lo razor edge [0.502,0.510] -> [0.505,0.515]: kept in spirit; with weak = 0.50 established, the edge's only job is w(0.50) = -kappa_lo exactly, which the code's normalization guarantees for any v_lo > 0.5.

(4) CELL-LEVEL VERIFICATION AT THE BOX CENTER (v_lo=0.51, v_peak=0.90, v_hi=0.96, rho1=0.715, rho2=0.30, kappa_lo=0.09, kappa_hi=1.15, tau=0.0125, beta=1.0, epsilon=0.055). Weight map: w(0.50)=-0.09, w(0.55)=0.20, w(0.60)=0.33, w(0.65)=0.48, w(0.70)=0.60, w(0.75)=0.71, w(0.80)=0.81, w(0.85)=0.91, w(0.90)=1.00, w(0.92)=0.89, w(0.95)=0.44, w(0.99)=-1.05. Exp 1 (conflicts P1/P2/P3 margins -0.41/-0.74/-0.65, ties P7/P9 +1.09/+0.21; p = sigma(2*margin)): contributions -0.39/-0.63/-0.58/+0.40/+0.11, doubled for mirrors, mean = -0.219, lapse-scaled -> -0.21 (real -0.218). Exp 3 (top = 0.99, mids assumed [0.85,0.75,0.65,0.60,0.55]): pure cells m1 = t+0.80 = -0.25 -> 0.38 (x2), m2 = t-1.08 = -2.13 -> 0.01 (x2), m3 = t+0.29 = -0.76 -> 0.18 (x1); top-against = 1.15 -> 0.91 (x2); |d|=3 = t+1.90 = 0.85 -> 0.85 (x2); mean = 0.50 (real 0.617; alternative mid vectors [0.9,0.8,0.7,0.6,0.5] and [0.8,0.7,0.6,0.55,0.5] give 0.50 and 0.47 — robust). Exp 4 (|d|=1 cells, top assumed 0.9, lows 0.5): t1 = w(0.9) -> 0.88, t2 = -kappa_lo -> 0.46, t3 = -0.36-2.29 -> 0.005, t4 = 1.90 -> 0.98; mean = 0.58 (real 0.513; if the top cue is 0.85 or 0.95 the value moves to 0.52-0.57). Exp 5: E[sigma(0.18|k|)] = 0.589 -> P(follow) = 0.416 -> metric -0.167 (real -0.168). Exp 6: K2 = sigma(2(t+2k)) = 0.15, K3 = 0.18, M3 = 0.025, M4 = 0.031, R3 = sigma(2(2*0.71+0.27)) = 0.97 -> SSR = 0.28 (real 0.118). Exp 2 (top 0.92 in-band at 0.89, mids flat ~0.6-0.8): b_d - b_s ~ 0.08-0.12 (real 0.077).

(5) STRUCTURAL FLOORS, ACCEPTED AND QUANTIFIED (critic fix #2, adopted): R3 = sigma(2*beta*(2*w(0.75)+3*kappa_lo)) = 0.97 is structural — w(0.5) < 0 is mandated by Exp 5 and w(0.75) ~ 0.7 by Exp 1, so the three weak cues' negative weights add to the strong side. SSR ~ 0.28 vs real 0.118 (Delta +0.16, squared 0.027, the theory's largest residual). Exp 3's pure-cell average is pinned at T ~ -2.1 (K2's logit exceeds it by only 4*beta*kappa_lo ~ 0.36 — the rigid coupling the critic identified), capping Exp 3 at ~0.50 (Delta -0.12, squared 0.014). These two residuals are two faces of the same constraint; the box minimizes their joint squared cost, and I do NOT fight R3 by suppressing w(0.75) or deepening the lobe — both levers are what destroyed Exp 3 in iters 2-6.

(6) CORNER-LEVEL BOX VERIFICATION (critic fix #5): sweeping all box corners gives Exp 1 in [-0.25,-0.16], Exp 2 in [0.05,0.15], Exp 3 in [0.46,0.53], Exp 4 in [0.52,0.58], Exp 5 in [-0.20,-0.13], Exp 6 in [0.26,0.32]. No corner regresses any experiment catastrophically; the worst corner is the shallow-lobe/shallow-beta corner (SSR 0.32, Exp 3 0.52) and the deep corner (SSR 0.26, Exp 3 0.46) — the intended trade-off surface itself. Box-midpoint sum of squared residuals ~ 0.046 (dominated by Exp 6's 0.027 and Exp 3's 0.014) versus the accepted base's ~ 0.17 (Exp 3 alone 0.141); worst corner ~ 0.08. Expected aggregate loss ~ 0.10-0.13, decisively under the 0.1958 floor, with the payoff coming from recovering Exp 3 from 0.24 to ~0.50 and Exp 4 from 0.38 to ~0.55 while holding Exps 1, 2, 5 at their iter-7-best quality and Exp 6 at 0.28 (versus iter-7's 0.69 explosion).

**Parameters:**
  - `v_lo`: `[0.505, 0.515]`
  - `v_peak`: `[0.89, 0.91]`
  - `v_hi`: `[0.957, 0.963]`
  - `rho1`: `[0.68, 0.75]`
  - `rho2`: `[0.25, 0.35]`
  - `rho_lo`: `{1}`
  - `kappa_lo`: `[0.075, 0.105]`
  - `kappa_hi`: `[1.0, 1.3]`
  - `tau`: `[0.011, 0.014]`
  - `beta`: `[0.9, 1.1]`
  - `epsilon`: `[0.04, 0.07]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # Skeptical (Inverted-Trust) Tallying, calibrated-cliff edition.
    # Stimulus: array of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are binary expert ratings (0/1).
    # Each feature j casts a sign vote sign(a_j - b_j); the vote is
    # weighted by a skewed inverted-U 'trust band' of stated validity:
    #   v_lo <= v <= v_peak :  w = ((v - v_lo)/(v_peak - v_lo))^rho1
    #                          (rise from zero at the lower edge to the
    #                          trusted peak, normalized to 1)
    #   v_peak < v <= v_hi :  w = ((v_hi - v)/(v_hi - v_peak))^rho2
    #                          (gentle fall from the peak toward the
    #                          cliff; keeps 0.9-0.92 cues near-peak)
    #   v < v_lo           :  w = -kappa_lo * ((v_lo - v)/(v_lo - 0.5))^rho_lo
    #                          (chance experts: kappa_lo == |w(0.5)| in
    #                          trusted-peak units, zero at the edge)
    #   v > v_hi           :  w = -kappa_hi * (1 - exp(-(v - v_hi)/tau))
    #                          (moderate, saturating distrust lobe in
    #                          the SAME peak units; v_hi ~ 0.96 so ONLY
    #                          near-certain (v ~ 0.99) experts fall in,
    #                          and |w(0.99)| ~ 1 trusted-cue unit)
    # Votes are summed into a zero-sum evidence score; choice is a
    # softmax with inverse temperature beta plus a uniform lapse.
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

    v_lo = float(parameters["v_lo"])
    v_peak = float(parameters["v_peak"])
    v_hi = float(parameters["v_hi"])
    rho1 = float(parameters["rho1"])
    rho2 = float(parameters["rho2"])
    rho_lo = float(parameters["rho_lo"])
    kappa_lo = float(parameters["kappa_lo"])
    kappa_hi = float(parameters["kappa_hi"])
    tau = float(parameters["tau"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    a, b = stim[0], stim[1]

    # Feature-wise votes: +1 where A wins the feature, -1 where B
    # wins, 0 on ties (a tied cue is uninformative for the pair).
    votes = np.sign(a - b)

    v = np.clip(val, 0.5, 1.0)
    w = np.zeros(n_features, dtype=float)
    below = v < v_lo
    above = v > v_hi
    rise = (~below) & (v <= v_peak)
    fall = (~below) & (v > v_peak) & (~above)

    # (i) Lower lobe: normalized so kappa_lo IS |w(0.5)| in trusted-
    # peak units; zero at the band edge, most negative at v = 0.5.
    denom_lo = max(v_lo - 0.5, 1e-6)
    w[below] = -kappa_lo * np.power(
        np.clip(v_lo - v[below], 0.0, None) / denom_lo, rho_lo
    )

    # (ii) In-band rise from v_lo to the trust peak (weight 1).
    span_r = max(v_peak - v_lo, 1e-6)
    vr = np.clip((v[rise] - v_lo) / span_r, 0.0, 1.0)
    w[rise] = np.power(vr, rho1)

    # (iii) In-band fall from the peak toward the cliff at v_hi.
    span_f = max(v_hi - v_peak, 1e-6)
    vf = np.clip((v_hi - v[fall]) / span_f, 0.0, 1.0)
    w[fall] = np.power(vf, rho2)

    # (iv) Distrust lobe above v_hi: saturating exponential in
    # the SAME (peak-normalized) units as the band -- kappa_hi is
    # directly |w(v -> 1)| in units of the trusted-cue weight.
    w[above] = -kappa_hi * (1.0 - np.exp(-(v[above] - v_hi) / tau))

    # Weighted evidence score. B's score is the mirror of A's
    # because every vote is zero-sum across the two options.
    s_a = float(np.dot(w, votes))
    scores = np.array([s_a, -s_a])

    # Numerically stable softmax over the two weighted scores.
    # When all cues tie (s_a == 0) the softmax is exactly uniform.
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

### `pi_6` → slot 1 (via `new_theory`)

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
