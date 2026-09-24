# feedback_iter_04

## System Prompt

You are a renowned cognitive scientist critiquing a freshly proposed candidate theory and model in the Decision Making (Binary Features) domain.

The candidate has been simulated on every previously run experiment. For each experiment you are shown the design, the metric, the value the metric takes on real data, and the value it takes on the candidate's simulated data.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the feedback is to SURFACE theories that are EXPERIMENT-INVARIANT: that is,theories that explain data across multiple experiments. 
Your task is to determine whether the candidate captures the human/real behavior well enough across these experiments. Return a verdict:
  * "continue"   — the candidate is good enough; carry on.
  * "regenerate" — the candidate fails to capture the empirical pattern; the proposing agent must produce a new candidate, taking your rationale into account.

Justify the verdict with a concrete diagnosis (which experiments fail, in what direction, what mechanism is likely missing or miscalibrated).

## SCOPE OF YOUR CRITIQUE — STAY INSIDE THE ARBITER'S MECHANISM FAMILY
When an "## ARBITER RECOMMENDATION" block is present below, the proposer was explicitly instructed to implement the mechanism family the arbiter prescribed. Your job is to grade FIT QUALITY *within that prescribed family*, not to relitigate which family should be used — that is the arbiter's call, made one level above this loop.

Concretely:
  * If the candidate misses the data, you may push for MINOR ADJUSTMENTS that keep the prescribed mechanism intact: tightening / widening parameter ranges, adding a temperature, swapping a normalization scheme, fixing a softmax / distance metric, re-balancing attention weights, fixing a learning-rate sign, correcting a bug in the gating or recurrence, etc.
  * You MUST NOT recommend switching to a different mechanism family. Such a switch is the arbiter's prerogative; recommending it here will mislead the proposer into oscillating between families across iterations.
  * Also grade FAITHFULNESS to the recommendation explicitly: if the candidate has clearly drifted into a different family than the one prescribed, say so in the rationale and ask for a return to the prescribed family — again, with minor adjustments, not a re-design.

## ACCEPT GATE — HOW THE LOOP DECIDES WHAT TO BUILD ON NEXT
This propose-loop has a programmatic accept gate. After every iteration the candidate's `aggregate_loss` is compared against the running-best loss (`accepted_loss`):
  * `loss < accepted_loss` → ACCEPTED. The candidate becomes the new running-best base; the next iteration's proposer will build on THIS candidate.
  * `loss >= accepted_loss` → REJECTED. The base is unchanged; the next iteration's proposer will build on the SAME `accepted` candidate again, with your new feedback on top. Rejected candidates are discarded — the loop guarantees the base never regresses, so you do NOT need to ask the proposer to "revert" anything; that already happens for free.

Two consequences for your verdict:
  * If the candidate you are grading was REJECTED by the gate, returning `"continue"` is silently downgraded to `"regenerate"` (returning a worse candidate would defeat the gate). Spend your rationale on a NEW direction the proposer should try on top of the unchanged accepted base, not on defending the rejected attempt.
  * If the candidate was ACCEPTED, you can return `"continue"` to stop the loop and ship this candidate, or `"regenerate"` to keep tuning further.

## LEARN FROM YOUR OWN PAST ADVICE
When a "## YOUR PRIOR CRITIQUES" block is present below, each prior iteration ends with an "Outcome of your advice" line that says whether the next candidate the proposer produced was ACCEPTED (your advice helped — its loss strictly beat the running best) or REJECTED (your advice didn't help — the proposer discarded the result and reset to the previous accepted base). This is the loop's ground-truth signal on whether *your own previous critique was good*. Use it explicitly:
  * If a previous piece of advice was ACCEPTED, it is OK to repeat / extend it. Reinforce in the same direction.
  * If a previous piece of advice was REJECTED, do NOT repeat the same recommendation; in your new rationale, briefly acknowledge that the previous push in that direction was rejected by the gate and try a different in-family knob (or a smaller step in the same direction) instead.
  * If you find yourself oscillating (e.g. iter 1 said "increase α", iter 2 said "decrease α", iter 3 about to say "increase α" again), STOP and recommend a value between the two flanking iterations instead.
  * The "## LOSS TRAJECTORY" block at the top of the user prompt summarises the same information at the loop level — consult it before issuing a new regenerate-with-direction recommendation.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## ARBITER RECOMMENDATION (mechanism family the proposer was told to implement)
The arbiter labelled this round's two theories in its recommendation as follows:
- THEORY 1 = `pi_7`
- THEORY 2 = `pi_8_1`
- The recommendation below acts on THEORY 1 (= `pi_7`).

Replace pi_7 (CREDT) with a new theory — propose 'Edge-Only Bounded Outlier Tallying' (EOBOT): (1) Keep the shared, well-supported architecture: sign votes (+1/-1/0) into a zero-sum evidence score, trust plateau over [v_lo, v_hi≈0.981] with BODT's shallow within-plateau tilt (validated on Exps 4/5/10), consensus-gated chance flank with the PARTIAL promotion cap kappa_gate*w_p (validated on Exps 5/6/7/2's X- and Z-ladders), low-gain softmax plus uniform lapse. (2) Replace the outlier branch with a SINGLE mechanism: w_top = w_floor + (w_p - w_floor) * exp(-rho_e * E(gap)) - d_w*(v_top - v_hi)/(1 - v_hi), where E(gap) is a peaked kernel maximal at gap ≈ 0.05 with a WIDE right flank (s_r ≈ 0.045-0.065, so distrust is near zero by gap ≈ 0.09 and the outlier is taken at (slightly above) face value by gap ≥ 0.12) and a moderate left flank (near zero for dense ladders gap ≤ 0.02). Calibrate the peak depth so that w_top ≈ -0.35 to -0.5 at gap 0.05 (about half a trusted cue AGAINST — matches Exps 9/14). (3) CRUCIALLY, DELETE the composition-amplified route entirely (c_amp = 0): panel chance-fraction must NOT deepen outlier distrust — Exp 2 is decisive (subjects FOLLOW a 0.99 outlier floating over a majority-chance panel at gap 0.14, follow index 0.547 vs BODT's 0.180 and CREDT's 0.465). (4) All experts tied at the panel maximum receive the same w_top, so double-99 panels double the anti-vote (explains Exp 15's 0.29 follow rate without any composition mechanism, and Exp 1's near-zero net anti-follow at gap 0.09). (5) Parameter ranges: w_p ∈ [0.88, 0.92], v_lo ∈ [0.52, 0.56], kappa_lo ∈ [0.15, 0.25], kappa_gate ∈ [0.60, 0.75], tilt ∈ [0.04, 0.12], g_star ∈ [0.045, 0.055], s_l ∈ [0.015, 0.025], s_r ∈ [0.045, 0.065], w_floor ∈ [-0.55, -0.30], rho_e ∈ [0.5, 1.2], d_w ∈ [0.05, 0.20], beta ∈ [0.35, 0.50], epsilon ∈ [0.03, 0.06]. This theory is a direct, falsifiable competitor to BODT: it predicts BODT's successes (Exps 3-8, 11, 14, 15) while reversing BODT's composition-route prediction on gap-0.14 majority-chance panels (Exp 2) and predicting moderate (not near-zero) distrust on gap-0.05 single-outlier panels (Exp 9) — exactly the two cells where the incumbent front-runner fails.


## CANDIDATE THEORY
Edge-Only Bounded Outlier Tallying (EOBOT), relocated-peak edition. Subjects integrate every binary expert rating as a sign vote (+1/-1/0) into a zero-sum evidence score with subjective weights computed RELATIVE to the panel: (i) a trust plateau over [v_lo, v_hi] (ceiling just above 0.98) with a shallow within-plateau tilt; (ii) a small negative chance flank at v near 0.5, consensus-gated with a partial promotion cap kappa_gate*w_p; (iii) outlier distrust is a bounded, saturating shrinkage toward a floor driven by the outlier's uncorroborated EDGE over the next rung, with the kernel peak RELOCATED to gap ~0.088 — the 'implausible-jump' zone where a 99% claim floating ~0.09 above its next rung is maximally distrusted (Exp 7) — plus a mild uniform skepticism floor e_min (~25% discount) applying to ANY above-ceiling claim (dense ladders and small-gap outliers keep ~70% of face value: Exps 9/10/14), a steep left wall (distrust ~0 below gap 0.05), and a right shoulder receding by gap 0.14 (Exps 15/16 followed); (iv) composition amplification is split into a NEAR branch — chance-laden panels deepen distrust only in the spurious-precision window gap ~0.05 via its own narrow kernel L (Exps 12/13), closed by gap 0.09 to protect Exp 11 — and a FAR branch (gap >= ~0.19 over a majority-chance panel = fraud signal, Exp 6). Choice is a low-gain softmax plus a uniform lapse.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Edge-Only Bounded Outlier Tallying (EOBOT), relocated-peak edition.
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
    #         w_top_mod = w_floor + (w_p - w_floor)*exp(-rho_e*E(gap)),
    #       where E(gap) = e_min + (1-e_min)*B(gap) and B is a PEAKED
    #       kernel now centered at g_star ~ 0.088 -- the
    #       'implausible-jump' zone (deepest distrust at gap ~0.09),
    #       with a STEEP left wall s_l (distrust ~0 below gap 0.05)
    #       and a right shoulder s_r receding by gap ~0.14. The NEW
    #       floor e_min carries a mild uniform skepticism (~25%
    #       discount) toward ANY above-ceiling claim, so dense
    #       ladders (gap ~0.01) and small-edge outliers (gap ~0.05)
    #       keep ~70% of face value.
    #  (iv) COMPOSITION AMPLIFICATION, SPLIT INTO TWO REGIMES, both
    #       gated by comp_gate(c_frac):
    #         comp_near = c_near * sig_gap * L(gap) * comp_gate
    #       with L its OWN narrow kernel peaked at g_near ~ 0.05
    #       (the spurious-precision window: chance-laden panels
    #       deepen distrust ONLY near gap 0.05; L is essentially
    #       closed by gap 0.09, protecting gap-0.09 panels), and
    #         comp_far  = c_far * sig_gap * far(gap) * comp_gate
    #       with far() a SHARP logistic centered at g_far ~ 0.19:
    #       closed at gap 0.14 (a 0.99 outlier over a 0.85 rung is
    #       followed even on a majority-chance panel) and open at
    #       gap >= ~0.20 (far-fraud signal).
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
    g_far = float(parameters["g_far"])
    s_far = float(parameters["s_far"])
    c_far = float(parameters["c_far"])
    c_half = float(parameters["c_half"])
    s_c = float(parameters["s_c"])
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
        w[plateau] = w_p * (1.0 - tilt * (v_hi - v[plateau]) / span)
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

    # ---- (iii) + (iv): relocated-peak bounded distrust with the
    # split composition amplification. ----
    if outlier:
        # (a) RELOCATED-PEAK base kernel: B is maximal at gap ~
        # g_star ~ 0.088 (the implausible-jump zone); steep left
        # wall s_l keeps B ~ 0 below gap 0.05; right shoulder s_r
        # recedes by gap ~0.14. The floor e_min carries a mild
        # uniform skepticism toward ANY above-ceiling claim.
        width = s_l if gap < g_star else s_r
        B = float(np.exp(-0.5 * ((gap - g_star) / width) ** 2))
        E = e_min + (1.0 - e_min) * B

        # (b) BOUNDED, SATURATING shrinkage toward the floor:
        # w_top_mod in [w_floor, w_p] for every E.
        w_top_mod = w_floor + (w_p - w_floor) * float(np.exp(-rho_e * E))

        # (c) Composition amplification, SPLIT. comp_near uses its
        # OWN narrow kernel L peaked at g_near ~ 0.05 (the
        # spurious-precision window), essentially closed by gap
        # 0.09 so it cannot deepen distrust on gap-0.09 panels.
        # comp_far is a sharp logistic centered at g_far ~ 0.19,
        # closed at gap 0.14 and open at gap >= ~0.20.
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

`policy(probs) -> int`:
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return int(np.random.choice(len(probabilities), p=probabilities))

`parameters`:
- w_p: [0.88, 0.92]
- v_lo: [0.52, 0.55]
- v_hi: [0.9805, 0.9825]
- kappa_lo: [0.21, 0.26]
- kappa_gate: [0.66, 0.76]
- tilt: [0.04, 0.10]
- g0: [0.041, 0.045]
- s_gap: [0.012, 0.014]
- lambda_ext: [0.35, 0.45]
- g_star: [0.084, 0.092]
- s_l: [0.011, 0.015]
- s_r: [0.027, 0.033]
- e_min: [0.05, 0.07]
- w_floor: [-0.58, -0.52]
- rho_e: [2.3, 2.7]
- d_w: [0.10, 0.15]
- g_near: [0.047, 0.053]
- s_near: [0.014, 0.018]
- c_near: [1.7, 2.1]
- g_far: [0.186, 0.196]
- s_far: [0.018, 0.022]
- c_far: [3.9, 4.7]
- c_half: [0.43, 0.49]
- s_c: [0.05, 0.07]
- beta: [0.38, 0.46]
- epsilon: [0.035, 0.055]
- validities: validities

`rationale`:
I built the mandatory panel-level accounting table (see analysis above) before touching any knob, as the critic demanded after two iterations failed on guessed panel structure. The table confirms the critic's core structural diagnosis and extends it in one respect the critic's own box cannot accommodate. THE STRUCTURAL FIX: Exp 14 (gap 0.05, c_frac=0, comp gate closed) requires w_top ~ +0.60 (from iter-3's sensitivity: w_top 0.28 -> follow 0.5024; needed 0.5716; slope ~0.15/unit -> w_top ~ 0.74; conservative target +0.60) while Exp 7 (gap 0.09, c_frac 0.125, comp gate closed) requires w_top ~ -0.50 (trial-level reconstruction: w_top=+0.18 gives anti 0.504 = iter-1's actual; w=-0.50 gives anti 0.578 = real 0.5796). This 1.1-unit separation within the composition-FREE branch is impossible under any peak-at-0.05 kernel, exactly as the critic proved. I therefore relocate the kernel peak to g_star ~ 0.088 -- slightly beyond even the critic's [0.065, 0.080] box, justified because the deepest required distrust sits AT gap 0.09, and the required left-wall steepness (B(0.05) ~ 0.014, s_l ~ 0.013) is what actually delivers Exp 14's near-face trust. THE NEW FLOOR: the table exposes a second constraint the critic's box misses: Exps 9/10 (gap 0.01) and Exp 14 (gap 0.05) BOTH demand a mild ~25% discount (w_top ~ 0.60-0.65 vs face 0.84). A Gaussian left wall that is near-zero at 0.05 is even nearer-zero at 0.01, restoring FULL trust on dense ladders and breaking Exps 9/10 (follow would rise to ~0.65, anti-99 would fall to ~0.51). The additive floor E = e_min + (1-e_min)*B resolves this: uniform mild skepticism for any above-ceiling claim, edge-driven distrust on top. SPLITTING comp_near: Exps 12/13 (gap 0.05, c_frac 0.5/0.556) still need deep anti-votes (w_top -0.20/-0.37) while Exp 14 at the same gap needs +0.60; the ~0.8-1.0 of extra distrust is now carried entirely by comp_near with its OWN narrow kernel L peaked at g_near ~ 0.05 and closed by 0.09 (L(0.09) ~ 0.04). This is essential to protect Exp 11 (gap 0.09, c_frac 0.5, comp gate OPEN), whose discriminant index must stay below ~0.06: comp_near(Exp 11) = 0.05, so its w_top = -0.56 comes only from the base peak that Exp 7 demands; the predicted index drift is 0.033 -> ~0.045 (error +0.024, within the critic's guard). FAR GATE: bracketed between iter-1's leak and iter-4's overshoot -- g_far ~ 0.191, s_far ~ 0.020, c_far trimmed to 4.3 so comp_far(0.14) ~ 0.15 on Exp 16's c_frac=0.556 panel (w_top(0.16) ~ -0.02, follow ~ 0.525 vs real 0.547) and ~ 0.05 on Exp 15's c_frac=0.4 panel (w_top ~ +0.09, anti ~ 0.482 vs real 0.483), while remaining fully open at gap 0.24 (w_top(Exp 6) ~ -1.97, SSR ~ 0.118, unchanged). Noise-free midpoint predictions vs iter-1: Exp 7 error -0.076 -> -0.002; Exp 14 -0.107 -> -0.02; Exp 15 -0.051 -> -0.001; Exp 16 -0.040 -> -0.022; Exp 12 -0.025 -> -0.015; costs: Exp 9 -0.016 -> -0.023, Exp 10 +0.010 -> +0.016, Exp 11 +0.013 -> +0.024, Exp 6 ~ +0.007. Predicted sum of squared errors drops from ~0.025 to ~0.007, far above the demonstrated ~0.01-0.03 per-cell noise floor on the targeted cells. All edits are kernel-shape and gating-shape changes inside the prescribed EOBOT architecture (sign votes, plateau/tilt, kappa_gate cap, bounded saturating shrinkage, softmax+lapse); no falsified edit (s_l narrowing at fixed g_star=0.05, lambda_ext cuts, beta nudges, unaccompanied rho_e shallowing) is repeated, and the validated calibrations (Exp 5 chance flank, Exp 6 far distrust, Exps 9/13 peak cells, softmax/lapse) are preserved.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.0521 -> ACCEPTED
- iter 2: loss=0.0582 -> REJECTED
- iter 3: loss=0.0652 -> REJECTED
- iter 4: loss=0.0569 -> REJECTED
- iter 5 (current candidate you are grading): loss=0.0630 -> REJECTED
Running-best (last accepted) base: iter 1 at loss=0.0521.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    rows_a = data["option_a_ratings"].tolist()
    rows_b = data["option_b_ratings"].tolist()
    resp = data["response"].tolist()
    n_rows = len(rows_a)
    contribs = []
    for i in range(n_rows):
        ca = rows_a[i]
        cb = rows_b[i]
        if isinstance(ca, str):
            ca = ca.replace("[", " ").replace("]", " ").replace(",", " ").split()
        if isinstance(cb, str):
            cb = cb.replace("[", " ").replace("]", " ").replace(",", " ").split()
        a = [int(float(v)) for v in ca]
        b = [int(float(v)) for v in cb]
        r = int(resp[i])
        n = len(a)
        if n == 0 or n != len(b):
            continue
        ttb = -1
        for j in range(n):
            if a[j] > b[j]:
                ttb = 0
                break
            if b[j] > a[j]:
                ttb = 1
                break
        if ttb < 0:
            continue
        aw = 0
        bw = 0
        for j in range(n):
            if a[j] > b[j]:
                aw = aw + 1
            elif b[j] > a[j]:
                bw = bw + 1
        if aw == bw:
            if r == ttb:
                contribs.append(0.5)
            else:
                contribs.append(-0.5)
        else:
            if bw > aw:
                tal = 1
            else:
                tal = 0
            if tal == ttb:
                continue
            if r == ttb:
                contribs.append(1.0)
            else:
                contribs.append(-1.0)
    if len(contribs) == 0:
        return 0.0
    total = 0.0
    for c in contribs:
        total = total + c
    return float(total / len(contribs))
```

**Observed (real) value:** -0.2180 (var=0.0140)
**Candidate trajectory (this loop):**
  - iter 1: -0.2377 (var=0.0059) (Δ vs real -0.0197)
  - iter 2: -0.2490 (var=0.0087) (Δ vs real -0.0310)
  - iter 3: -0.2390 (var=0.0066) (Δ vs real -0.0210)
  - iter 4: -0.2417 (var=0.0090) (Δ vs real -0.0237)
  - iter 5 (current): -0.2298 (var=0.0078) (Δ vs real -0.0118)
**Other theories' values on this metric (for reference):**
- pi_1: 0.6130 (var=0.0149)
- pi_2: -0.4425 (var=0.0124)
- pi_3: -0.2182 (var=0.0156)
- pi_4: -0.2188 (var=0.0064)
- pi_5: -0.4125 (var=0.0044)
- pi_6: -0.1973 (var=0.0076)
- pi_7: -0.2530 (var=0.0065)
- pi_8: -0.2522 (var=0.0047)
- pi_8_1: -0.2380 (var=0.0059)

### Experiment 2
**Design**
  A=[1, 1, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0, 1]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Dual-dissociation regression metric.
    # Per trial, compute:
    #   d = tally margin  = (#features A wins) - (#features B wins)
    #   s = top-cue direction = sign(A[0] - B[0])   (feature 0 has the
    #       highest validity 0.92, and every designed pair discriminates on it)
    # Regress chose-A on [1, d, s] via OLS and return b_d - b_s.
    # Tallying: choice probability depends only on d  -> b_d > 0, b_s = 0 -> metric > 0.
    # TTB: choice probability depends only on s       -> b_d = 0, b_s > 0 -> metric < 0.
    # The two theories push this scalar in opposite directions, and the
    # partial regression cleanly removes the design's d-s correlation.
    df = data.reset_index(drop=True)

    A = np.stack([np.asarray(list(r), dtype=float) for r in df["option_a_ratings"].values])
    B = np.stack([np.asarray(list(r), dtype=float) for r in df["option_b_ratings"].values])

    # Tally margin: strict feature-wise wins; ties count for neither side.
    d = (A > B).sum(axis=1).astype(float) - (B > A).sum(axis=1).astype(float)

    # Direction of the highest-validity (top) cue, which always discriminates
    # in this design: +1 if it favors A, -1 if it favors B.
    s = np.sign(A[:, 0] - B[:, 0])

    # Chose-A indicator (response: 0 = A, 1 = B).
    y = 1.0 - df["response"].to_numpy(dtype=float)

    X = np.column_stack([np.ones(len(y)), d, s])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    b_d = float(coef[1])
    b_s = float(coef[2])

    # Tallying's signature (slope in d) minus TTB's signature (slope in s).
    return b_d - b_s
```

**Observed (real) value:** 0.0771 (var=0.0031)
**Candidate trajectory (this loop):**
  - iter 1: 0.1039 (var=0.0017) (Δ vs real +0.0268)
  - iter 2: 0.1141 (var=0.0020) (Δ vs real +0.0371)
  - iter 3: 0.1124 (var=0.0025) (Δ vs real +0.0354)
  - iter 4: 0.1174 (var=0.0015) (Δ vs real +0.0403)
  - iter 5 (current): 0.1232 (var=0.0016) (Δ vs real +0.0461)
**Other theories' values on this metric (for reference):**
- pi_2: 0.1196 (var=0.0023)
- pi_1: -0.3725 (var=0.0076)
- pi_3: 0.0861 (var=0.0026)
- pi_4: 0.0881 (var=0.0019)
- pi_5: 0.0789 (var=0.0021)
- pi_6: 0.0956 (var=0.0023)
- pi_7: 0.1225 (var=0.0025)
- pi_8: 0.1105 (var=0.0025)
- pi_8_1: 0.1174 (var=0.0015)

### Experiment 3
**Design**
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Tally-following rate on decisive trials.

    For every trial, compute the raw (unweighted) tally margin
    d = (# cues where A wins) - (# cues where B wins).
    On trials with d != 0, check whether the subject's choice agrees
    with the option favored by the raw cue count (A if d > 0, B if
    d < 0). The metric is the fraction of such agreement over all
    decisive trials (d = 0 trials are excluded because the tallying
    heuristic itself is indifferent there).
    """
    df = data.reset_index(drop=True)
    n = len(df)
    if n == 0:
        return 0.5

    a_lists = df['option_a_ratings'].tolist()
    b_lists = df['option_b_ratings'].tolist()
    resp = df['response'].to_numpy(dtype=float)

    d = np.empty(n, dtype=float)
    for i in range(n):
        a = np.asarray(a_lists[i], dtype=float)
        b = np.asarray(b_lists[i], dtype=float)
        d[i] = float(np.sum(a > b) - np.sum(b > a))

    decisive = d != 0.0
    if not np.any(decisive):
        return 0.5

    dd = d[decisive]
    rr = resp[decisive]
    # Option favored by the raw tally: A (response 0) if d > 0,
    # otherwise B (response 1).
    agree = np.where(dd > 0.0, rr == 0.0, rr == 1.0)
    return float(np.mean(agree))
```

**Observed (real) value:** 0.6170 (var=0.0034)
**Candidate trajectory (this loop):**
  - iter 1: 0.6496 (var=0.0039) (Δ vs real +0.0326)
  - iter 2: 0.6296 (var=0.0038) (Δ vs real +0.0126)
  - iter 3: 0.6437 (var=0.0039) (Δ vs real +0.0267)
  - iter 4: 0.6237 (var=0.0038) (Δ vs real +0.0067)
  - iter 5 (current): 0.5722 (var=0.0037) (Δ vs real -0.0448)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6456 (var=0.0096)
- pi_2: 0.8389 (var=0.0095)
- pi_1: 0.6944 (var=0.0071)
- pi_4: 0.6485 (var=0.0027)
- pi_5: 0.4996 (var=0.0026)
- pi_6: 0.4811 (var=0.0035)
- pi_7: 0.6267 (var=0.0043)
- pi_8: 0.6393 (var=0.0026)
- pi_8_1: 0.6463 (var=0.0039)

### Experiment 4
**Design**
  A=[1, 1, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 0, 0, 0]  B=[1, 1, 0, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 0, 1, 0, 0, 1]  B=[1, 1, 1, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0, 1, 1]  B=[1, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Hard-ceiling test on the |d|=1 family (FACT 1 of the design).

    Metric = proportion of choices that agree with the UNWEIGHTED tally
    count-winner, restricted to trials whose raw tally margin is exactly
    +/-1 (the 8 trial types t1-t8; 48 of the 96 trials per subject).

    Tallying (any beta >= 1, any epsilon) predicts ~1 - epsilon/2 here,
    i.e. >= ~0.85 for any moderate lapse. The validity-weighted
    competitor is structurally capped: its choice probability on a |d|=1
    trial is (1-eps)*sigmoid(2*beta*m) with m = sum_j v_j^gamma*sign diff;
    the mean of |m| over the 8 |d|=1 trial types is <= 1 for every
    gamma in [0.05, 0.7] (equality only at gamma=0), and beta <= 0.55,
    so by concavity of tanh its mean agreement on this family is
    <= 0.5 + tanh(0.55)/2 = 0.750 over its ENTIRE parameter box.
    """
    def _mat(col):
        vals = []
        for v in col:
            if isinstance(v, np.ndarray):
                vals.append(v)
            else:
                vals.append(list(v))
        return np.asarray(vals, dtype=float)

    A = _mat(data["option_a_ratings"])
    B = _mat(data["option_b_ratings"])
    resp = np.asarray(data["response"], dtype=float)

    if A.ndim != 2 or A.shape[0] == 0 or A.shape != B.shape:
        return 0.5

    # Raw (unweighted) tally margin per trial.
    d = (A > B).sum(axis=1) - (B > A).sum(axis=1)

    # The |d| = 1 family: 8 unique pairs x 6 reps = 48 trials/subject.
    mask = np.abs(d) == 1
    if mask.sum() < 10:
        # Defensive fallback: use all non-tied trials.
        mask = d != 0
    if mask.sum() == 0:
        return 0.5

    dm = d[mask]
    rm = resp[mask]
    # Count-winner is A when d>0 (response 0), B when d<0 (response 1).
    consistent = np.where(dm > 0, rm == 0, rm == 1)
    return float(np.mean(consistent))
```

**Observed (real) value:** 0.5133 (var=0.0042)
**Candidate trajectory (this loop):**
  - iter 1: 0.5275 (var=0.0035) (Δ vs real +0.0142)
  - iter 2: 0.4988 (var=0.0050) (Δ vs real -0.0146)
  - iter 3: 0.5379 (var=0.0032) (Δ vs real +0.0246)
  - iter 4: 0.5350 (var=0.0038) (Δ vs real +0.0217)
  - iter 5 (current): 0.5312 (var=0.0038) (Δ vs real +0.0179)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8387 (var=0.0140)
- pi_3: 0.6062 (var=0.0085)
- pi_1: 0.6825 (var=0.0042)
- pi_4: 0.5192 (var=0.0023)
- pi_5: 0.3950 (var=0.0023)
- pi_6: 0.3850 (var=0.0041)
- pi_7: 0.5425 (var=0.0051)
- pi_8: 0.5350 (var=0.0045)
- pi_8_1: 0.5279 (var=0.0036)

### Experiment 5
**Design**
  A=[1, 0, 1, 0, 1, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 1, 1, 1]  B=[0, 1, 0, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1, 0, 1]  B=[1, 1, 0, 1, 1, 0, 0, 1]
  A=[0, 0, 1, 0, 1, 1, 1, 0]  B=[1, 0, 1, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1, 0, 1, 0]  B=[0, 0, 0, 1, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 0, 0, 1]  B=[0, 1, 0, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 1, 1, 0, 1]  B=[1, 1, 1, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 1, 1, 0]  B=[0, 1, 0, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 1]  B=[1, 1, 1, 1, 0, 1, 0, 1]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    # Chance-cue sensitivity on "chance-only" (Z-family) trials.
    # Design validities: [0.95, 0.85, 0.75, 0.65, 0.5, 0.5, 0.5, 0.5]
    # -> features 0-3 are diagnostic, features 4-7 are the four v=0.5 experts.
    # A Z-family trial is one where ALL diagnostic cues tie, so the pair is
    # discriminated ONLY by the 50%-accurate experts. On such a trial let
    # k = net number of chance cues won by A (signed). The competing theory
    # (llr-rectified weights) annihilates v=0.5 cues exactly, so p(A)=0.5
    # regardless of k -> subjects match the k-implied side at chance (metric 0).
    # My theory gives v=0.5 cues weight 0.5^gamma, so choices should follow
    # sign(k) above chance (metric > 0).
    DIAG = [0, 1, 2, 3]
    CHANCE = [4, 5, 6, 7]

    a_ratings = data["option_a_ratings"].apply(lambda x: np.asarray(list(x), dtype=int))
    b_ratings = data["option_b_ratings"].apply(lambda x: np.asarray(list(x), dtype=int))
    responses = data["response"].to_numpy()

    correct = []
    for av, bv, r in zip(a_ratings, b_ratings, responses):
        if av.shape[0] != 8 or bv.shape[0] != 8:
            continue
        # Keep only trials where every diagnostic cue ties (chance-only trials)
        if not np.array_equal(av[DIAG], bv[DIAG]):
            continue
        # Net chance-cue advantage for A
        k = int(np.sign(av[CHANCE] - bv[CHANCE]).sum())
        if k == 0:
            continue
        chose_a = (int(r) == 0)  # response: 0 = A, 1 = B
        followed = (k > 0 and chose_a) or (k < 0 and not chose_a)
        correct.append(1.0 if followed else 0.0)

    if len(correct) == 0:
        return 0.0
    # 2 * (accuracy - 0.5): 0 = chance-level following of chance cues,
    # positive = subjects systematically follow the 50%-accurate experts.
    return 2.0 * (float(np.mean(correct)) - 0.5)
```

**Observed (real) value:** -0.1683 (var=0.0124)
**Candidate trajectory (this loop):**
  - iter 1: -0.1975 (var=0.0146) (Δ vs real -0.0292)
  - iter 2: -0.1633 (var=0.0198) (Δ vs real +0.0050)
  - iter 3: -0.1850 (var=0.0202) (Δ vs real -0.0167)
  - iter 4: -0.1750 (var=0.0174) (Δ vs real -0.0067)
  - iter 5 (current): -0.1892 (var=0.0271) (Δ vs real -0.0208)
**Other theories' values on this metric (for reference):**
- pi_3: 0.2958 (var=0.0606)
- pi_4: 0.0308 (var=0.0233)
- pi_1: 0.1625 (var=0.0145)
- pi_2: 0.7617 (var=0.0356)
- pi_5: -0.1167 (var=0.0195)
- pi_6: -0.1350 (var=0.0257)
- pi_7: -0.1425 (var=0.0147)
- pi_8: -0.1750 (var=0.0140)
- pi_8_1: -0.1600 (var=0.0245)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Strong-Side Choice Rate on Multi-Weak Conflict trials (SSR).

    Selects the five conflict cells in which a high-validity side (99%
    experts, or 75% experts for R3) faces TWO OR MORE opposing weak
    experts: K2 (1x99 vs 2 weak), K3 (1x99 vs 3 weak), M3 (2x99 vs 3
    weak), M4 (2x99 vs 4 weak), R3 (2x75 vs 3 weak) -- 5 unique pairs
    x 2 mirrors x 4 reps = 40 trials per subject. Returns the
    proportion of trials on which the subject chose the strong side.
    """
    hits = 0
    n = 0

    a_col = data['option_a_ratings']
    b_col = data['option_b_ratings']
    r_col = data['response']

    for a, b, r in zip(a_col, b_col, r_col):
        a = [int(x) for x in a]
        b = [int(x) for x in b]
        if len(a) < 8 or len(b) < 8:
            continue

        # Validity-class vote balances (features 0-1 = 0.99 experts,
        # 2-3 = 0.75 experts, 4-7 = weak experts).
        d_hi = (a[0] + a[1]) - (b[0] + b[1])
        d_mid = (a[2] + a[3]) - (b[2] + b[3])
        d_lo = (a[4] + a[5] + a[6] + a[7]) - (b[4] + b[5] + b[6] + b[7])

        # K2/K3/M3/M4: super experts alone on one side (no 75s involved)
        # opposing >= 2 weak experts.
        if d_mid == 0 and d_hi != 0 and abs(d_lo) >= 2:
            target_is_a = d_hi > 0
        # R3: two 75% experts vs three weak experts.
        elif d_hi == 0 and abs(d_mid) == 2 and abs(d_lo) == 3:
            target_is_a = d_mid > 0
        else:
            # Excludes K1 (|d_lo|=1), R1, W2/W3 (no strong side),
            # W4a (|d_mid|=1), G1 (mixed 99+75 side), X1 (all tied).
            continue

        n += 1
        chose_a = (int(r) == 0)
        if chose_a == target_is_a:
            hits += 1

    if n == 0:
        return 0.5
    return hits / n

```

**Observed (real) value:** 0.1180 (var=0.0074)
**Candidate trajectory (this loop):**
  - iter 1: 0.1175 (var=0.0020) (Δ vs real -0.0005)
  - iter 2: 0.1335 (var=0.0018) (Δ vs real +0.0155)
  - iter 3: 0.1355 (var=0.0025) (Δ vs real +0.0175)
  - iter 4: 0.1280 (var=0.0022) (Δ vs real +0.0100)
  - iter 5 (current): 0.1315 (var=0.0020) (Δ vs real +0.0135)
**Other theories' values on this metric (for reference):**
- pi_4: 0.7670 (var=0.0039)
- pi_3: 0.4190 (var=0.0107)
- pi_1: 0.8475 (var=0.0076)
- pi_2: 0.1180 (var=0.0091)
- pi_5: 0.2870 (var=0.0027)
- pi_6: 0.4360 (var=0.0065)
- pi_7: 0.1235 (var=0.0028)
- pi_8: 0.1115 (var=0.0018)
- pi_8_1: 0.1280 (var=0.0020)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Super-expert contrarianism index.

    Over all trials in which the feature-0 expert (the 99%-validity
    expert) discriminates between the two options (a[0] != b[0]),
    compute the proportion of choices that go AGAINST the option
    endorsed by that expert.

    Advocated theory (inverted trust): the 99% expert's endorsement is
    anti-diagnostic (w(0.99) ~ -1 trusted-cue unit), so subjects
    predominantly choose AGAINST it -> index ~0.8-0.9.

    Competing theory (flattened diagnosticity): the 99% expert gets the
    LARGEST positive weight, so subjects predominantly FOLLOW it ->
    index ~0.2-0.35.
    """
    n_anti = 0
    n_sel = 0
    for ra, rb, resp in zip(
        data["option_a_ratings"], data["option_b_ratings"], data["response"]
    ):
        a = np.asarray(ra, dtype=int).ravel()
        b = np.asarray(rb, dtype=int).ravel()
        if a.size == 0 or b.size == 0 or a[0] == b[0]:
            continue  # expert 0 does not discriminate on this trial
        # Option index endorsed by the 99%-validity (feature-0) expert.
        endorsed = 0 if a[0] == 1 else 1
        r = int(resp)
        n_sel += 1
        if r != endorsed:
            n_anti += 1
    if n_sel == 0:
        return 0.5
    return float(n_anti) / float(n_sel)

```

**Observed (real) value:** 0.5796 (var=0.0066)
**Candidate trajectory (this loop):**
  - iter 1: 0.5036 (var=0.0071) (Δ vs real -0.0760)
  - iter 2: 0.4947 (var=0.0053) (Δ vs real -0.0849)
  - iter 3: 0.4618 (var=0.0056) (Δ vs real -0.1178)
  - iter 4: 0.5049 (var=0.0036) (Δ vs real -0.0747)
  - iter 5 (current): 0.4040 (var=0.0058) (Δ vs real -0.1756)
**Other theories' values on this metric (for reference):**
- pi_5: 0.7902 (var=0.0033)
- pi_4: 0.3191 (var=0.0050)
- pi_1: 0.1662 (var=0.0114)
- pi_2: 0.2569 (var=0.0106)
- pi_3: 0.3831 (var=0.0085)
- pi_6: 0.6756 (var=0.0042)
- pi_7: 0.5787 (var=0.0053)
- pi_8: 0.5133 (var=0.0036)
- pi_8_1: 0.4542 (var=0.0043)

### Experiment 8
**Design**
  A=[1, 1, 0, 0, 1, 0, 0, 1, 0]  B=[1, 0, 0, 1, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 1, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 1, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 1, 1, 1, 1, 0, 1]
  A=[1, 0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 0, 1, 0, 0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 0, 1, 0, 0]  B=[1, 0, 0, 1, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 1, 0, 0, 1, 0]  B=[1, 0, 0, 1, 1, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 0, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0, 1, 0, 0, 1]
  A=[1, 0, 0, 1, 1, 1, 0, 0, 1]  B=[0, 1, 0, 1, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 0, 1]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 0, 0, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 1, 0, 0, 1]  B=[0, 1, 0, 1, 0, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """P(follow the 98% expert | 98%-vs-90% conflict trials).

    A 'D-cell' trial is one where the top expert (v=0.98, feature 0) and the
    second expert (v=0.90, feature 1) endorse OPPOSITE options, while the
    third expert (v=0.75, feature 2) is tied. On these trials the metric is
    the fraction of choices that follow the option endorsed by the 98%
    expert. FDEI (compressed diagnosticity) treats 0.98 and 0.90 as nearly
    exchangeable, so this should be ~0.50 (coin flip). SITT (inverted trust)
    assigns the 98% expert a large NEGATIVE weight, so subjects should
    systematically choose AGAINST the 98% endorsement (~0.05-0.10).
    """
    if data is None or len(data) == 0:
        return 0.5

    follow_flags = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.ndim != 1 or a.shape[0] < 3 or a.shape != b.shape:
            continue

        d0 = a[0] - b[0]  # 98% expert's relative endorsement of A
        d1 = a[1] - b[1]  # 90% expert's relative endorsement of A
        d2 = a[2] - b[2]  # 75% expert must be tied (no interference)

        # Need a genuine 98-vs-90 conflict with the 75% expert tied.
        if d0 == 0 or d1 == 0 or d2 != 0:
            continue
        if np.sign(d0) == np.sign(d1):
            continue

        endorsed_a = d0 > 0          # 98% expert endorses option A
        chose_a = (int(row["response"]) == 0)
        follow_flags.append(1.0 if (endorsed_a == chose_a) else 0.0)

    if len(follow_flags) == 0:
        return 0.5
    return float(np.mean(follow_flags))
```

**Observed (real) value:** 0.4625 (var=0.0103)
**Candidate trajectory (this loop):**
  - iter 1: 0.4738 (var=0.0156) (Δ vs real +0.0112)
  - iter 2: 0.4525 (var=0.0138) (Δ vs real -0.0100)
  - iter 3: 0.4475 (var=0.0163) (Δ vs real -0.0150)
  - iter 4: 0.4400 (var=0.0137) (Δ vs real -0.0225)
  - iter 5 (current): 0.4688 (var=0.0196) (Δ vs real +0.0062)
**Other theories' values on this metric (for reference):**
- pi_4: 0.5400 (var=0.0192)
- pi_5: 0.0550 (var=0.0029)
- pi_1: 0.8688 (var=0.0163)
- pi_2: 0.6587 (var=0.0158)
- pi_3: 0.6062 (var=0.0141)
- pi_6: 0.4213 (var=0.0156)
- pi_7: 0.4813 (var=0.0186)
- pi_8: 0.4550 (var=0.0133)
- pi_8_1: 0.4813 (var=0.0124)

### Experiment 9
**Design**
  A=[1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0]  B=[0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1]  B=[0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1]
  A=[0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
  A=[0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0]  B=[0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0]  B=[1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1]  B=[0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0]  B=[0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1]
  A=[1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1]  B=[1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1]
  A=[1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """
    Anti-super-expert choice rate on the 10 pre-registered 'cliff conflict'
    trials (S1/S1m, S2/S2m, D1/D1m, D2/D2m, V1/V1m).

    On each of these trials exactly one option is endorsed by the
    99%-validity experts (features 0 and 1) while the rival option is
    endorsed only by experts of validity <= 0.98, and the 99%-endorsed
    side has no 0.90-class support (this excludes the G1/G1m agreement
    anchors).  The metric is the fraction of these trials on which the
    subject chooses the option NOT endorsed by the 99% experts.

    BSST (sharp distrust cliff between 0.98 and 0.99, w(0.99) ~ -1.1
    trusted-cue units, w(0.98) ~ +0.6, w(0.90) ~ +0.9, w(0.50) ~ -0.23):
    the 99%-endorsed side is disfavored on every one of these cells, with
    predicted anti-99% choice rates of roughly S1 .86-.90, S2 .90-.95,
    D1 .72-.79, D2 .85-.94, V1 .70-.76  ->  overall ~0.84.

    FDEI (delta ~ 0.15 compression makes 0.98 and 0.99 experts nearly
    exchangeable, w99 - w98 in [-0.21, +0.15]): it is locked near 0.50 on
    the D cells, mildly PRO-99% on the S cells (the 99-side also carries
    a tiny validity edge), and strongly pro-99% on V1 (three chance cues
    are annihilated exactly)  ->  overall anti-99% rate ~0.30-0.53,
    center ~0.44.  Sign-disjoint from BSST with a ~0.40 mean gap.
    """

    def as_tuple(cell):
        try:
            return tuple([int(x) for x in cell])
        except Exception:
            return None

    # key: (A ratings, B ratings) -> favored option (0 = A, 1 = B),
    # where 'favored' is the side NOT endorsed by the 99% experts.
    TRIALS = {
        # S1 / S1m  (A or B wins both 0.99; rival wins both 0.90)
        ((1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0), (0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0)): 1,
        ((0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0), (1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0)): 0,
        # S2 / S2m  (one side wins 0.99+0.99+0.98; rival 0.90+0.90+0.55)
        ((1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1), (0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1)): 1,
        ((0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1), (1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1)): 0,
        # D1 / D1m  (one side wins a 0.99; rival wins a 0.98)
        ((0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1), (1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1)): 0,
        ((1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1), (0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1)): 1,
        # D2 / D2m  (one side wins both 0.99; rival wins both 0.98)
        ((0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0), (1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0)): 0,
        ((1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0), (0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0)): 1,
        # V1 / V1m  (one side wins a 0.99; rival wins all three 0.50)
        ((1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0), (0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1)): 1,
        ((0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1), (1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0)): 0,
    }

    hits = []
    a_col = data['option_a_ratings']
    b_col = data['option_b_ratings']
    r_col = data['response']

    for a, b, r in zip(a_col, b_col, r_col):
        ka = as_tuple(a)
        kb = as_tuple(b)
        if ka is None or kb is None:
            continue
        fav = TRIALS.get((ka, kb))
        if fav is None:
            continue
        try:
            ri = int(r)
        except Exception:
            continue
        hits.append(1.0 if ri == fav else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.5600 (var=0.0052)
**Candidate trajectory (this loop):**
  - iter 1: 0.5440 (var=0.0067) (Δ vs real -0.0160)
  - iter 2: 0.4665 (var=0.0063) (Δ vs real -0.0935)
  - iter 3: 0.5270 (var=0.0050) (Δ vs real -0.0330)
  - iter 4: 0.5715 (var=0.0050) (Δ vs real +0.0115)
  - iter 5 (current): 0.5310 (var=0.0048) (Δ vs real -0.0290)
**Other theories' values on this metric (for reference):**
- pi_6: 0.8115 (var=0.0019)
- pi_4: 0.3985 (var=0.0055)
- pi_1: 0.1815 (var=0.0131)
- pi_2: 0.5685 (var=0.0047)
- pi_3: 0.5220 (var=0.0036)
- pi_5: 0.7830 (var=0.0038)
- pi_7: 0.5715 (var=0.0054)
- pi_8: 0.5455 (var=0.0066)
- pi_8_1: 0.5560 (var=0.0071)

### Experiment 10
**Design**
  A=[0, 0, 1, 1, 0, 1, 0, 0, 1, 1]  B=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 0, 1, 0, 0, 1, 1]  B=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1]  B=[0, 1, 0, 1, 0, 1, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 0, 1, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 0, 1, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1]  B=[0, 1, 0, 1, 0, 1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 0, 1, 0, 0, 1, 1]  B=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1]  B=[1, 0, 0, 1, 0, 1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0, 1, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 0, 0, 1, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 1, 0, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 1, 0, 1, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 1, 0, 1, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 1, 0, 1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 0, 1, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1, 0, 1, 1, 0]
  A=[1, 0, 1, 1, 0, 1, 0, 1, 0, 1]  B=[0, 0, 0, 0, 1, 1, 1, 0, 0, 1]
  A=[1, 0, 1, 1, 0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 0, 1, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 0, 1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 1, 0, 0]  B=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # "Follow-the-99%-expert rate": across all trials where the single
    # 0.99-validity expert (feature index 0 in the fixed validity vector
    # [0.99, 0.98, 0.95, 0.90, 0.85, 0.75, 0.65, ., ., .]) discriminates
    # between A and B, compute the proportion of choices that go to the
    # option the 0.99 expert rated 1 (i.e., the option the top expert
    # "endorsed").
    #
    # FDEI: w(0.99) is the LARGEST positive weight in the whole vector
    # (its compression never inverts the validity ordering), so the
    # 0.99-favored option is followed above chance on essentially every
    # such trial -> metric ~ 0.60-0.65.
    # BSST: w(0.99) is strongly NEGATIVE (~ -1.1 trusted-cue units, the
    # distrust cliff), so the 0.99-favored option is chosen BELOW chance
    # on every such trial (the background cues are balanced/weak in this
    # design) -> metric ~ 0.15-0.35.
    # The sign of (metric - 0.5) alone separates the two theories on
    # every trial in the 0.99-discriminating set.

    TOP = 0  # index of the 0.99-validity expert, fixed by the design

    resp = data['response'].to_numpy()
    follows = []

    for i in range(len(data)):
        a_raw = data['option_a_ratings'].iloc[i]
        b_raw = data['option_b_ratings'].iloc[i]
        try:
            av = np.asarray(list(a_raw), dtype=int).ravel()
            bv = np.asarray(list(b_raw), dtype=int).ravel()
        except Exception:
            continue
        if av.size <= TOP or bv.size <= TOP:
            continue

        d = int(av[TOP]) - int(bv[TOP])
        if d == 0:
            continue  # 0.99 expert tied on this trial -> uninformative

        favored_a = d > 0          # option the 0.99 expert endorsed
        chose_a = (int(resp[i]) == 0)
        follows.append(1.0 if (chose_a == favored_a) else 0.0)

    if len(follows) == 0:
        return 0.5  # neutral fallback (should not occur: 8 trials x 4 reps)

    return float(np.mean(follows))
```

**Observed (real) value:** 0.5988 (var=0.0026)
**Candidate trajectory (this loop):**
  - iter 1: 0.6088 (var=0.0104) (Δ vs real +0.0100)
  - iter 2: 0.6156 (var=0.0107) (Δ vs real +0.0169)
  - iter 3: 0.6238 (var=0.0083) (Δ vs real +0.0250)
  - iter 4: 0.6156 (var=0.0061) (Δ vs real +0.0169)
  - iter 5 (current): 0.6200 (var=0.0083) (Δ vs real +0.0212)
**Other theories' values on this metric (for reference):**
- pi_4: 0.6719 (var=0.0085)
- pi_6: 0.2988 (var=0.0058)
- pi_1: 0.8525 (var=0.0127)
- pi_2: 0.6687 (var=0.0078)
- pi_3: 0.6206 (var=0.0104)
- pi_5: 0.1444 (var=0.0034)
- pi_7: 0.6094 (var=0.0057)
- pi_8: 0.5800 (var=0.0079)
- pi_8_1: 0.5725 (var=0.0067)

### Experiment 11
**Design**
  A=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1, 1, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0, 1, 1, 1, 0, 1]
  A=[1, 0, 0, 0, 1, 0, 1, 0, 1, 0]  B=[1, 1, 0, 0, 1, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 0, 1, 1, 1, 0, 1]  B=[0, 1, 1, 0, 0, 1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 1, 0, 0, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 1, 0, 0, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 1, 0, 0, 1]  B=[0, 0, 1, 0, 0, 1, 1, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0, 0, 1, 0, 0, 1]
  A=[1, 0, 0, 1, 0, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 1, 1, 1, 0, 1]  B=[0, 1, 0, 0, 1, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 1, 0, 1, 1, 0, 1]  B=[0, 1, 0, 0, 1, 1, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 1, 0, 1, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 0, 1, 0, 0, 1]  B=[0, 1, 0, 0, 1, 0, 0, 1, 0, 1]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    """
    Theory-discriminant index for the 10-expert panel
    validities = [0.99, 0.90, 0.80, 0.70, 0.60, 0.50, 0.50, 0.50, 0.50, 0.50].

    Both theories reduce to a FIXED weight vector over the 10 experts on
    this panel. At each theory's parameter-box midpoint we compute its
    predicted p(choose A) for the observed trial, take the difference
    d_t = p_CREDT(t) - p_FDEI(t), and average d_t * y_t over all trials,
    where y_t = +1 if the subject chose A and -1 if B.

    E[metric] = (1/T) * sum_t d_t * (2*p_true(t) - 1).
    If CREDT generated the data this is strongly POSITIVE (~ +0.045);
    if FDEI generated the data it is NEGATIVE (~ -0.022). The contrast
    is driven by the cells where the boxes are disjoint: the Z-family
    (chance-cue ladders: CREDT 0.59-0.74 vs FDEI exactly 0.50), the
    0.99-expert trials (CREDT ~0.28-0.36 vs FDEI ~0.55-0.60), the
    X-family sign flips, and K1 (four trusted experts vs four
    coin-flippers: CREDT ~0.50 vs FDEI ~0.78).
    """
    validities = np.array([0.99, 0.90, 0.80, 0.70, 0.60, 0.50, 0.50, 0.50, 0.50, 0.50])

    # ---------- CREDT weights at parameter-box midpoints ----------
    w_p, v_lo, v_hi, kappa_lo = 0.90, 0.53, 0.9815, 0.20
    g0, s_gap = 0.043, 0.013
    rho, d_w, lambda_ext = 1.36, 0.40, 0.40
    g_star, s_l, s_r = 0.048, 0.012, 0.033
    c_half, s_c, c_amp = 0.36, 0.06, 1.29
    beta_c, eps_c = 0.42, 0.05

    v = np.clip(validities, 0.5, 1.0)
    v_top = float(v.max())
    below = v[v < v_top - 1e-9]
    v_second = float(below.max()) if below.size > 0 else 0.5
    gap = v_top - v_second
    outlier = v_top > v_hi
    if outlier:
        sig_gap = 0.5 * (1.0 + np.tanh(0.5 * ((gap - g0) / s_gap)))
    else:
        sig_gap = 0.0
    gate = (lambda_ext + (1.0 - lambda_ext) * sig_gap) if outlier else 0.0

    w_c = np.full(10, w_p, dtype=float)
    flank = v < v_lo
    if np.any(flank):
        w_fl = -kappa_lo + (w_p + kappa_lo) * np.clip(v[flank] - 0.5, 0.0, None) / (v_lo - 0.5)
        w_c[flank] = (1.0 - gate) * w_fl + gate * w_p
    if outlier:
        c_frac = float(flank.mean())
        C = c_amp * 0.5 * (1.0 + np.tanh(0.5 * (c_frac - c_half) / s_c))
        width = s_l if gap < g_star else s_r
        P = float(np.exp(-0.5 * ((gap - g_star) / width) ** 2))
        w_top = w_p - d_w * (v_top - v_hi) / (1.0 - v_hi) - rho * (P + C * sig_gap)
        w_c[v >= v_top - 1e-9] = w_top

    # ---------- FDEI weights at parameter-box midpoints ----------
    delta, tau = 0.15, 0.02
    beta_f, eps_f = 0.32, 0.025
    vv = np.clip(validities, 0.5, 1.0 - 1e-9)
    llr = np.log(vv / (1.0 - vv))
    w_f = np.power(np.maximum(llr - tau, 0.0), delta)

    def _p_choose_a(s, beta, eps):
        # low-gain softmax over [s, -s] plus uniform lapse
        z = beta * s
        p_core = 1.0 / (1.0 + np.exp(-z))
        return (1.0 - eps) * p_core + eps * 0.5

    total = 0.0
    n = 0
    for a_cell, b_cell, resp in zip(data['option_a_ratings'],
                                    data['option_b_ratings'],
                                    data['response']):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        votes = np.sign(a - b)
        s_c = float(np.dot(w_c, votes))
        s_f = float(np.dot(w_f, votes))
        p_c = _p_choose_a(s_c, beta_c, eps_c)
        p_f = _p_choose_a(s_f, beta_f, eps_f)
        d_t = p_c - p_f
        y = 1.0 if int(resp) == 0 else -1.0
        total += d_t * y
        n += 1

    if n == 0:
        return 0.0
    return float(total / n)
```

**Observed (real) value:** 0.0206 (var=0.0003)
**Candidate trajectory (this loop):**
  - iter 1: 0.0333 (var=0.0004) (Δ vs real +0.0127)
  - iter 2: 0.0325 (var=0.0003) (Δ vs real +0.0120)
  - iter 3: 0.0372 (var=0.0003) (Δ vs real +0.0166)
  - iter 4: 0.0399 (var=0.0004) (Δ vs real +0.0193)
  - iter 5 (current): 0.0398 (var=0.0003) (Δ vs real +0.0192)
**Other theories' values on this metric (for reference):**
- pi_7: 0.0758 (var=0.0002)
- pi_4: -0.0398 (var=0.0003)
- pi_1: -0.0446 (var=0.0003)
- pi_2: 0.0431 (var=0.0003)
- pi_3: 0.0148 (var=0.0004)
- pi_5: -0.0288 (var=0.0001)
- pi_6: -0.0227 (var=0.0002)
- pi_8: 0.0576 (var=0.0002)
- pi_8_1: 0.0464 (var=0.0004)

### Experiment 12
**Design**
  A=[1, 1, 0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 1, 0, 0]  B=[1, 1, 0, 1, 0, 1, 0, 0]
  A=[1, 0, 1, 1, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 0, 0, 1, 0]  B=[1, 0, 1, 1, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 1, 0, 1, 1]  B=[0, 1, 0, 1, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 1, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 0, 1, 1, 1]  B=[0, 1, 0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 1, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0, 0, 1]
  A=[1, 0, 1, 1, 0, 1, 1, 0]  B=[1, 0, 1, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 1, 0, 1, 1]  B=[0, 1, 0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 1, 1, 0, 1]
  A=[1, 1, 0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 1, 0, 0]
  A=[1, 1, 0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 1, 0, 1]
  A=[1, 0, 0, 1, 1, 1, 0, 1]  B=[1, 1, 0, 1, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 1, 0]
  A=[1, 0, 1, 1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 1, 1, 1, 0]  B=[1, 0, 1, 1, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """
    Super-expert deference rate.

    Among all trials on which the highest-validity expert (feature 0, the
    99% expert) discriminates between option A and option B, return the
    fraction of choices that FOLLOW that expert's recommendation (i.e.
    chose the option the 99% expert rated 1).

    FDEI (flattened-diagnosticity integration) gives the 0.99 expert its
    largest positive weight, so it predicts deference well above chance
    (~0.57 overall, ~0.69 on isolated super-expert trials, but BELOW 0.5
    when a 0.94+0.65+0.50 coalition opposes it).

    CREDT, on this panel (0.99 sitting 0.05 above the 0.94 rung with a
    partially chance-laden scale), assigns the panel-max expert a large
    NEGATIVE weight, so it predicts systematic ANTI-deference (~0.21).
    """
    df = data.reset_index(drop=True)

    def _first_rating(cell):
        # Cells may be list / tuple / np.ndarray of 0/1 ints, or a string
        # serialization like "[1, 0, 1, ...]". Robustly extract element 0.
        if isinstance(cell, str):
            s = cell.strip().strip("[]")
            parts = s.replace(",", " ").split()
            return int(float(parts[0]))
        seq = list(cell)
        return int(seq[0])

    a_col = df["option_a_ratings"]
    b_col = df["option_b_ratings"]
    r_col = df["response"]

    n = 0
    follow = 0
    for i in range(len(df)):
        try:
            a0 = _first_rating(a_col.iloc[i])
            b0 = _first_rating(b_col.iloc[i])
        except Exception:
            continue
        # Skip trials where the 99% expert does not discriminate.
        if a0 not in (0, 1) or b0 not in (0, 1) or a0 == b0:
            continue
        # Option endorsed by the 99% expert: 0 = A, 1 = B.
        winner = 0 if a0 == 1 else 1
        resp = int(r_col.iloc[i])
        n += 1
        if resp == winner:
            follow += 1

    if n == 0:
        return 0.5
    return float(follow) / float(n)
```

**Observed (real) value:** 0.4525 (var=0.0028)
**Candidate trajectory (this loop):**
  - iter 1: 0.4271 (var=0.0047) (Δ vs real -0.0254)
  - iter 2: 0.4238 (var=0.0045) (Δ vs real -0.0287)
  - iter 3: 0.4617 (var=0.0034) (Δ vs real +0.0092)
  - iter 4: 0.4004 (var=0.0053) (Δ vs real -0.0521)
  - iter 5 (current): 0.5029 (var=0.0043) (Δ vs real +0.0504)
**Other theories' values on this metric (for reference):**
- pi_4: 0.6017 (var=0.0039)
- pi_7: 0.2367 (var=0.0031)
- pi_1: 0.8358 (var=0.0153)
- pi_2: 0.6479 (var=0.0042)
- pi_3: 0.5867 (var=0.0071)
- pi_5: 0.1121 (var=0.0017)
- pi_6: 0.2667 (var=0.0041)
- pi_8: 0.4387 (var=0.0066)
- pi_8_1: 0.4658 (var=0.0052)

### Experiment 13
**Design**
  A=[1, 1, 1, 1, 0, 0, 1, 1, 1]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 0, 0, 1, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 1, 1, 1, 0, 1]
  A=[0, 0, 0, 1, 1, 1, 1, 0, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 0, 1, 0, 1]  B=[0, 0, 0, 0, 1, 0, 1, 0, 1]
  A=[0, 0, 0, 0, 1, 0, 1, 0, 1]  B=[1, 1, 1, 1, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 1, 0, 1, 0, 1]  B=[0, 0, 1, 1, 1, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 1, 0, 1, 0, 1]  B=[0, 0, 1, 1, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 1, 1, 0, 1, 0]  B=[0, 0, 1, 1, 0, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 1, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 1, 1, 1, 1, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 1, 0]  B=[1, 1, 0, 0, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 1, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 0, 1, 0]  B=[1, 1, 0, 1, 0, 0, 0, 1, 0]
  A=[1, 1, 0, 1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 1, 1, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 1, 1, 0, 1, 0]  B=[1, 1, 0, 1, 0, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 1, 1, 0, 0]  B=[1, 0, 1, 0, 0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1, 0, 1]
  A=[0, 0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # "Double-99 endorsement avoidance" metric.
    #
    # On every trial, identify whether BOTH top-validity experts (the two
    # tied 0.99 experts, features 0 and 1 of the fixed panel
    # [0.99, 0.99, 0.94, 0.75, 0.5, 0.5, 0.5, 0.5, 0.5]) cast the SAME
    # vote (both favor A, or both favor B). On those trials, record
    # whether the subject chose the option the 0.99 pair endorsed.
    # The metric is the proportion of such choices.
    #
    # This isolates the single structural disagreement between CREDT and
    # BODT on this panel: the DEPTH of distrust of a 0.99 claim sitting
    # gap=0.05 above its next rung on a chance-laden panel.
    #   CREDT: w_top ~ -1.7 per 99% expert  -> the endorsement is worth
    #          roughly two trusted cues AGAINST -> p(pick endorsed) ~ 0.16
    #          (box ~[0.12, 0.19]).
    #   BODT:  w_top ~ -0.14 per 99% expert -> the endorsement is worth
    #          roughly nothing -> p(pick endorsed) ~ 0.40 (box ~[0.35, 0.47]).
    # The two predicted boxes are disjoint by ~0.2 in probability units.
    # Mirror trials (e.g., trials 1/2, 3/4, 6/7, 8/9, 16/17) put the 99-pair
    # on opposite sides, so pure A/B side bias cancels.
    # Works identically on the pooled DataFrame and on a single subject's
    # slice (each subject has 14 qualifying trial types x 4 reps = 56 rows).
    validities = [0.99, 0.99, 0.94, 0.75, 0.5, 0.5, 0.5, 0.5, 0.5]
    val = np.asarray(validities, dtype=float)
    v_top = float(val.max())
    top_idx = np.where(val >= v_top - 1e-9)[0]

    picks = []
    a_col = data['option_a_ratings']
    b_col = data['option_b_ratings']
    r_col = data['response']

    for a_cell, b_cell, r in zip(a_col, b_col, r_col):
        a = np.asarray(a_cell, dtype=float).ravel()
        b = np.asarray(b_cell, dtype=float).ravel()
        if a.shape[0] != val.shape[0] or b.shape[0] != val.shape[0]:
            continue
        diffs = a[top_idx] - b[top_idx]
        # Both top experts must vote, and vote together.
        if np.all(diffs > 0):
            endorsed = 0  # both 0.99 experts favor A
        elif np.all(diffs < 0):
            endorsed = 1  # both 0.99 experts favor B
        else:
            # split votes or ties among the top experts -> not a
            # double-endorsement trial
            continue
        resp = int(r)
        picks.append(1.0 if resp == endorsed else 0.0)

    if len(picks) == 0:
        return 0.5
    return float(np.mean(picks))
```

**Observed (real) value:** 0.2900 (var=0.0036)
**Candidate trajectory (this loop):**
  - iter 1: 0.2854 (var=0.0026) (Δ vs real -0.0046)
  - iter 2: 0.3282 (var=0.0039) (Δ vs real +0.0382)
  - iter 3: 0.3121 (var=0.0038) (Δ vs real +0.0221)
  - iter 4: 0.3064 (var=0.0037) (Δ vs real +0.0164)
  - iter 5 (current): 0.2957 (var=0.0034) (Δ vs real +0.0057)
**Other theories' values on this metric (for reference):**
- pi_7: 0.0639 (var=0.0012)
- pi_8: 0.3475 (var=0.0040)
- pi_1: 0.8289 (var=0.0104)
- pi_2: 0.5486 (var=0.0043)
- pi_3: 0.6114 (var=0.0059)
- pi_4: 0.7611 (var=0.0029)
- pi_5: 0.1286 (var=0.0030)
- pi_6: 0.3032 (var=0.0042)
- pi_8_1: 0.2904 (var=0.0028)

### Experiment 14
**Design**
  A=[1, 1, 1, 0, 1, 0, 0, 1]  B=[0, 1, 1, 1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 1, 1, 0, 1]  B=[0, 1, 0, 1, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 1, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0, 0, 0, 1]  B=[1, 1, 1, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 1, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 1, 1, 0]  B=[0, 1, 1, 0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 1, 1, 1]  B=[0, 1, 1, 0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 1, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0, 1, 0]  B=[1, 1, 1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 1, 1, 0]  B=[1, 0, 1, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 1, 0, 1, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    def _to_vec(x):
        if isinstance(x, str):
            import ast
            x = ast.literal_eval(x)
        return np.asarray(x, dtype=float).ravel()

    follows = []
    for a_raw, b_raw, resp in zip(
        data["option_a_ratings"], data["option_b_ratings"], data["response"]
    ):
        try:
            a = _to_vec(a_raw)
            b = _to_vec(b_raw)
        except Exception:
            continue
        if a.size == 0 or a.size != b.size:
            continue
        # The 0.99-validity expert is feature index 0 (displayed first).
        d0 = float(a[0]) - float(b[0])
        if d0 == 0.0:
            # Top expert tied between options: endorsement undefined.
            continue
        endorsed_a = d0 > 0.0
        try:
            chose_a = (int(resp) == 0)
        except Exception:
            continue
        follows.append(1.0 if (endorsed_a == chose_a) else 0.0)

    if len(follows) == 0:
        return 0.5
    return float(np.mean(follows))
```

**Observed (real) value:** 0.5716 (var=0.0029)
**Candidate trajectory (this loop):**
  - iter 1: 0.4650 (var=0.0034) (Δ vs real -0.1066)
  - iter 2: 0.4642 (var=0.0036) (Δ vs real -0.1074)
  - iter 3: 0.5024 (var=0.0020) (Δ vs real -0.0692)
  - iter 4: 0.4689 (var=0.0030) (Δ vs real -0.1026)
  - iter 5 (current): 0.5682 (var=0.0031) (Δ vs real -0.0034)
**Other theories' values on this metric (for reference):**
- pi_8: 0.4855 (var=0.0025)
- pi_7: 0.3326 (var=0.0039)
- pi_1: 0.8437 (var=0.0114)
- pi_2: 0.7321 (var=0.0085)
- pi_3: 0.6150 (var=0.0059)
- pi_4: 0.6305 (var=0.0031)
- pi_5: 0.1179 (var=0.0015)
- pi_6: 0.2871 (var=0.0027)
- pi_8_1: 0.5024 (var=0.0027)

### Experiment 15
**Design**
  A=[1, 1, 0, 0, 1, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 1, 0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 1, 0, 0, 1, 0, 0]
  A=[1, 1, 0, 1, 0, 0, 1, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 1, 0, 0, 1]
  A=[0, 0, 0, 1, 0, 0, 1, 0, 0, 1]  B=[1, 1, 0, 1, 0, 0, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 0, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 1, 0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0, 0, 0, 1, 0]
  A=[1, 0, 1, 1, 0, 0, 1, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 1, 0, 0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 0, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 1, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0, 1, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 1, 0, 0, 1]  B=[0, 0, 0, 1, 1, 1, 1, 0, 0, 1]
  A=[0, 0, 0, 1, 1, 1, 1, 0, 0, 1]  B=[1, 1, 1, 0, 0, 0, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 1, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 1, 1, 0]  B=[1, 0, 1, 0, 1, 0, 0, 0, 1, 0]
  A=[1, 0, 1, 1, 0, 0, 1, 1, 1, 0]  B=[0, 0, 0, 1, 0, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 0, 1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1, 0, 0, 1, 1, 0]  B=[1, 0, 1, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 0, 0, 0, 1, 0, 1]  B=[1, 0, 0, 1, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 1, 0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 1, 0, 1, 0, 1]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 1, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Anti-follow rate of the 99%-expert endorsements.

    On every trial where the two top-validity (0.99) experts cast a
    net nonzero vote (i.e., they jointly endorse one option), compute
    the probability that the subject chooses the NON-endorsed option.
    Under CREDT (negative subjective weight for the uncorroborated
    0.99 outlier on this chance-laden panel) this rate is well above
    0.5 (predicted ~0.56-0.65); under BODT (bounded distrust, w_top
    near zero) it is below 0.5 (predicted ~0.38-0.48). The two
    predicted boxes are disjoint.
    """

    def _to_vec(cell):
        if isinstance(cell, np.ndarray):
            return cell
        if isinstance(cell, (list, tuple)):
            return np.asarray(cell, dtype=float)
        # Fallback: parse string-encoded sequences like "[1, 0, ...]"
        s = str(cell).strip().replace("[", "").replace("]", "")
        return np.asarray([float(v) for v in s.split(",") if v.strip() != ""], dtype=float)

    a = np.vstack([_to_vec(c) for c in data["option_a_ratings"]])
    b = np.vstack([_to_vec(c) for c in data["option_b_ratings"]])
    resp = np.asarray(data["response"], dtype=float)

    # Net vote of the two 0.99-validity experts (features 0 and 1,
    # fixed by this experiment's validity vector).
    net = np.sign(a[:, 0] - b[:, 0]) + np.sign(a[:, 1] - b[:, 1])

    mask = net != 0
    if not np.any(mask):
        return 0.5

    net_m = net[mask]
    r = np.rint(resp[mask]).astype(int)
    endorsed_a = net_m > 0  # True -> 0.99 experts endorse A

    # 'against' = subject picked the option the 0.99 experts did NOT endorse
    against = np.where(endorsed_a, r == 1, r == 0)

    return float(np.mean(against))
```

**Observed (real) value:** 0.4830 (var=0.0041)
**Candidate trajectory (this loop):**
  - iter 1: 0.4319 (var=0.0043) (Δ vs real -0.0511)
  - iter 2: 0.4489 (var=0.0043) (Δ vs real -0.0341)
  - iter 3: 0.3822 (var=0.0041) (Δ vs real -0.1007)
  - iter 4: 0.4359 (var=0.0058) (Δ vs real -0.0470)
  - iter 5 (current): 0.5396 (var=0.0055) (Δ vs real +0.0567)
**Other theories' values on this metric (for reference):**
- pi_7: 0.6411 (var=0.0036)
- pi_8_1: 0.4474 (var=0.0052)
- pi_1: 0.1615 (var=0.0132)
- pi_2: 0.2174 (var=0.0047)
- pi_3: 0.3044 (var=0.0057)
- pi_4: 0.2544 (var=0.0028)
- pi_5: 0.7807 (var=0.0039)
- pi_6: 0.6574 (var=0.0055)
- pi_8: 0.5137 (var=0.0034)

### Experiment 16
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 1, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 1, 1, 1, 0]  B=[1, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """
    Global outlier-endorsement follow index, side-balanced.

    Uses every trial on which the 0.99-validity outlier expert (feature
    index 0 in this design) actually discriminates the two options (his
    rating is 1 for exactly one of A/B). On those trials we ask: did the
    subject choose the side the outlier endorsed? The index averages the
    follow rate over outlier-endorses-A trials and outlier-endorses-B
    trials separately, so pure display/side bias cancels.

    These are trials 1-8 and 20-24 of the design (R0-R5, R3m, R5m, V1,
    V2, V1m, K1, K2), 13 trial types x 3 reps = 39 trials per subject.

    BODT (deep, composition-amplified outlier distrust: w(0.99) ~ -1.8
    to -3.5 in chance-cue units) predicts subjects systematically REJECT
    the outlier-endorsed side: index ~ 0.32 (centers .20/.24/.29/.34/.39/
    .45 on R0-R5, .34/.46 on the mirrors, ~.21 on the V cells, .42/.36 on
    K1/K2). CREDT (shallow distrust: w(0.99) ~ -0.7 to -1.3, coin-flippers
    near-exchangeable with experts) predicts the outlier's endorsement
    still carries the day on balance: index ~ 0.51 (R-ladder crosses 0.5
    at k~2). The two predicted values are separated by ~0.19 in mean
    against a per-subject SD of ~0.09-0.11 (binomial noise over 39 trials
    plus parameter heterogeneity), giving Welch t of order 6-8 at N=25.
    """
    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()

    n_follow_outlier_a = 0
    n_outlier_a = 0
    n_follow_outlier_b = 0
    n_outlier_b = 0

    for a_cell, b_cell, resp in zip(a_col, b_col, r_col):
        try:
            a = [int(v) for v in a_cell]
            b = [int(v) for v in b_cell]
        except (TypeError, ValueError):
            continue
        if len(a) == 0 or len(a) != len(b):
            continue
        # The 0.99-validity outlier expert occupies feature index 0 in
        # this experiment's panel [0.99, 0.85, 0.80, 0.75, 0.50 x5].
        if a[0] == b[0]:
            # Outlier's endorsement does not discriminate this pair.
            continue
        r = int(resp)
        if a[0] == 1:
            # Outlier endorses A.
            n_outlier_a += 1
            if r == 0:
                n_follow_outlier_a += 1
        else:
            # Outlier endorses B.
            n_outlier_b += 1
            if r == 1:
                n_follow_outlier_b += 1

    if n_outlier_a == 0 and n_outlier_b == 0:
        return 0.5

    rate_a = (float(n_follow_outlier_a) / float(n_outlier_a)) if n_outlier_a > 0 else 0.5
    rate_b = (float(n_follow_outlier_b) / float(n_outlier_b)) if n_outlier_b > 0 else 0.5
    # Side-balanced average: cancels any constant A-vs-B display bias.
    return 0.5 * (rate_a + rate_b)
```

**Observed (real) value:** 0.5474 (var=0.0024)
**Candidate trajectory (this loop):**
  - iter 1: 0.5071 (var=0.0060) (Δ vs real -0.0403)
  - iter 2: 0.5627 (var=0.0052) (Δ vs real +0.0153)
  - iter 3: 0.5829 (var=0.0051) (Δ vs real +0.0354)
  - iter 4: 0.5897 (var=0.0056) (Δ vs real +0.0422)
  - iter 5 (current): 0.5468 (var=0.0076) (Δ vs real -0.0006)
**Other theories' values on this metric (for reference):**
- pi_8_1: 0.1801 (var=0.0041)
- pi_7: 0.4646 (var=0.0065)
- pi_1: 0.8708 (var=0.0110)
- pi_2: 0.7154 (var=0.0058)
- pi_3: 0.6538 (var=0.0058)
- pi_4: 0.6016 (var=0.0074)
- pi_5: 0.1555 (var=0.0017)
- pi_6: 0.2763 (var=0.0042)
- pi_8: 0.4048 (var=0.0041)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is a genuine improvement over the incumbent family and was correctly accepted by the gate. It nails the two decisive cells that motivated EOBOT: Exp 13 (double-99, gap 0.05, chance-laden panel: 0.2854 vs real 0.2900) and Exp 16 (gap 0.14, majority-chance panel: 0.5071 vs real 0.5474, versus pi_8_1's catastrophic 0.1801). It also preserves the validated components: Exp 6 SSR is essentially perfect (0.1175 vs 0.1180), Exps 1, 4, 8, 9, 10, 11 are all within ~0.02, and Exps 3, 5, 12 are within ~0.03. The peak-of-kernel calibration (gap 0.05) should not be touched — Exps 9/12/13 are all well fit.

However, four residual misses form a coherent diagnostic pattern, and two of them are large:

1. **Exp 14 is the worst miss (0.4650 vs real 0.5716, error 0.107).** This is the dense-ladder / small-gap outlier cell. The candidate distrusts a gap≈0.01 outlier far too much. Two culprits are identifiable in the code: (a) the left flank s_l ∈ [0.020, 0.026] is too wide — at gap 0.01 with g_star≈0.05, E ≈ 0.2, which via the saturating shrinkage still pulls w_top well below face value; the arbiter's own box had s_l down to 0.015. (b) The chance-flank promotion leaks at small gaps: sig_gap ≈ 0.08 at gap 0.01, but with lambda_ext ≈ 0.4 the gate is already ~0.45 open, giving chance cues a net POSITIVE weight (~+0.15) that fights the outlier's side. Both are in-family knobs.

2. **Exp 16 under-follows (0.5071 vs 0.5474) because the far-fraud logistic leaks at gap 0.14.** With c_far ∈ [4.2, 5.8] and s_far up to 0.038, comp_far at gap 0.14 (g_far ≈ 0.19) is sigmoid(-1.6) ≈ 0.17, subtracting ~0.8 from w_top on a majority-chance panel. The relocation concept is right (Exp 6 at gap 0.24 needs deep distrust, SSR 0.118), but the gate must be essentially closed at 0.14 and fully open only by ~0.22-0.24: raise g_far toward 0.20-0.205 and/or cut s_far to ~0.02.

3. **Exp 7 under-predicts contrarianism (0.5036 vs real 0.5796).** The 99-expert's endorsement is treated as nearly neutral when it should be mildly anti-diagnostic relative to the next rung. Either the right flank at gap ≈ 0.09 retains slightly too little distrust, or the within-plateau tilt is too shallow to let the 0.90-class experts outrank the discounted outlier on conflict trials. pi_7's 0.5787 shows this cell is reachable within this architecture.

4. **Exp 15 over-follows the 99-pair (anti-follow 0.4319 vs real 0.4830).** Note this pulls in the opposite direction from Exp 14 if both panels are dense ladders — so before moving s_l, the proposer must check the actual gap/chance-fraction structure of the Exp 14 and Exp 15 panels. If Exp 15 is also gap≈0.01, its residual over-trust likely traces to the chance-flank promotion (kappa_gate / lambda_ext) rather than the kernel, and should be fixed there so the two experiments separate.

Minor: Exp 2 (+0.027) and Exp 3 (+0.033) are slightly over-tallying; a small beta or tilt nudge, low priority.

On faithfulness: the candidate openly deviated from the arbiter's 'delete the composition route' prescription, reinstating comp_near and comp_far. I judge this acceptable — the arbiter's own parameter box was internally inconsistent (rho_e ∈ [0.5, 1.2] cannot reach the demanded w_top ≈ -0.35 to -0.5 at gap 0.05), and the relocation is empirically vindicated by the Exp 6 vs Exp 16 dissociation (both majority-chance panels: reject at gap 0.24, follow at gap 0.14). The deviation stays inside the prescribed architecture (sign votes, plateau, kappa_gate cap, bounded saturating distrust). But the far gate as parameterized is now the single thing hurting the arbiter's decisive cell (Exp 16), so it must be tightened, not removed.
Rationale: Regenerate — the candidate is the new accepted base and the right architecture, but there is identifiable, separable headroom. Keep the peak-of-kernel calibration (gap 0.05) and the Exp 6 far-distrust exactly as they are, and make four targeted in-family adjustments: (1) Sharpen the far-fraud gate: g_far toward 0.20-0.205 and s_far down to ~0.020-0.025, so comp_far ≈ 0 at gap 0.14 (Exp 16 follow should rise from 0.507 toward 0.547) while remaining fully open by gap 0.24 (Exp 6 SSR must stay ≈ 0.118). (2) Narrow the left flank s_l toward 0.014-0.018 so gap-0.01 distrust is near zero (Exp 14 follow should rise from 0.465 toward 0.57) — but first verify the Exp 15 panel's gap; if Exp 15 is also a dense ladder and needs MORE anti-follow, fix its over-trust via the chance-flank promotion (lower lambda_ext and/or kappa_gate at small sig_gap) rather than via s_l, so Exps 14 and 15 separate. (3) Recover Exp 7's contrarianism (0.504 → ~0.58) by either slightly deepening the right-flank distrust at gap ≈ 0.09 or increasing the within-plateau tilt so mid-band experts outrank the discounted outlier on conflict trials. (4) Optionally nudge beta/tilt to trim the small over-tallying on Exps 2/3. Do NOT redesign the outlier branch or switch mechanism families — these are all parameter-range and gating-shape corrections within the prescribed EOBOT architecture.

**Outcome of your advice:** iter 1 candidate loss=0.0521 -> iter 2 candidate loss=0.0582 -> the gate marked it REJECTED.

### Iteration 2
Verdict: regenerate
Interpretation: The iter-2 candidate was REJECTED by the gate (0.0582 vs accepted 0.0521), so my verdict is regenerate regardless; the useful work here is diagnosing WHY my iter-1 directives misfired and charting a different in-family direction on the unchanged iter-1 base. Per-experiment accounting of the iter-1 → iter-2 movement is revealing: the four parameter edits decomposed into one clear win, one null result on the primary target, and two self-inflicted regressions. WIN: the far-fraud gate sharpening (g_far → ~0.20, s_far → ~0.021) fixed Exp 16 exactly as intended (Δ vs real: -0.0403 → +0.0153) while Exp 6's far-gap distrust held (SSR 0.1335 vs 0.1180, acceptable). It also contributed to Exp 15's improvement (Δ -0.0511 → -0.0341) and Exp 5 landed near-perfect (-0.1633 vs -0.1683). NULL RESULT: Exp 14 — the worst miss and the explicit target of BOTH the s_l narrowing and the lambda_ext cut — did not move at all (0.4650 → 0.4642, real 0.5716). The proposer's own rationale predicted error 0.107 → ~0.02; it stayed at 0.107. This falsifies the iter-1 hypothesis that Exp 14's under-follow is driven by small-gap kernel leakage plus promotion leakage: the two knobs that most affect exactly those terms were turned hard and the metric was inert. Either Exp 14's panel gap is not small (most likely it sits at/near the kernel peak ~0.05, where s_l and lambda_ext are irrelevant), or the two edits cancelled each other through the chance-flank gate. REGRESSIONS: (a) the s_l narrowing broke Exp 9 (0.5440 → 0.4665 vs real 0.5600) — Exp 9's cliff cells are dense-ladder (0.99 over 0.98, gap 0.01), and cutting E(0.01) from ~0.22 to ~0.04 over-trusted the 99s there; iter-1's s_l was already correct. (b) The lambda_ext cut broke Exp 7 further (0.5036 → 0.4947 vs real 0.5796): with the gate floor lowered, chance cues on Exp 7's trials fall back to their negative flank weight, and chance cues opposing the 99-endorsed side with negative weight effectively PUSH toward the 99's side — the opposite of the contrarianism we need. The same mechanism contributed to Exp 9's V-cells. (c) The s_r widening, aimed at Exp 7, did not recover it (it got worse) because the lambda_ext cut counteracted it; it did help Exp 15 slightly, so it is not purely harmful. (d) The beta nudge did not trim Exp 2 (0.1039 → 0.1141 vs real 0.0771 — worse). Net: the far-gate fix should be KEPT; the s_l/lambda_ext/beta edits should be REVERTED; s_r should sit between the two iterations. The remaining big misses after reverts will be Exp 14 (~-0.107), Exp 7 (~-0.076), Exp 15 (~-0.05 residual), Exp 2 (~+0.027), Exp 12 (~-0.025). The new insight available from this iteration: Exp 14's flat response to small-gap knobs, combined with the accepted fits at the peak on chance-laden double-99 panels (Exp 13: 0.285 vs 0.290), points to a structural tension the current parameterization resolves wrongly — a SINGLE 99 near the peak on a (probably) non-chance-laden panel should be mildly FOLLOWED (0.5716) while a DOUBLE 99 on a chance-laden panel is avoided (0.29). A shared composition-independent peak depth rho_e cannot easily deliver both; but the architecture already contains the right segregating variable: the composition gate. The kernel's base peak (rho_e) can be shallowed so a singleton outlier at gap ~0.05 lands near face value, while the deep anti-vote on chance-laden panels is carried by comp_near (raise c_near), which only opens when c_frac is high. This single rebalance predicts improvements on Exp 14 (singleton, low chance-fraction → trusted), Exp 7 (gap ~0.09, chance-laden → comp_near deepens distrust of the 99 → anti-follow rises toward 0.58), Exp 15 (double-99, chance-laden → more anti-follow toward 0.483), and preserves Exp 13 (double-99, chance-laden, comp_near keeps the deep avoid). CRITICAL CAVEAT: this direction is only valid if Exp 14's panel is indeed low-chance-fraction; the proposer must FIRST extract Exp 14's actual validity vector, compute its gap, c_frac, and the current w_top and chance-cue contributions there before committing — the iter-2 failure came precisely from editing knobs without verifying the target panel's structure.
Rationale: REGENERATE (candidate was gate-rejected; build on the iter-1 accepted base). My iter-1 advice was partially wrong and the gate confirmed it — do not repeat it. Concretely: (1) KEEP the one validated edit: the far-fraud gate at g_far [0.198, 0.206], s_far [0.018, 0.024]. This fixed the arbiter's decisive cell (Exp 16: 0.507 → 0.563 vs real 0.547) at negligible cost to Exp 6. (2) REVERT s_l to the iter-1 range [0.020, 0.026] and lambda_ext to [0.30, 0.42]: the narrowing/cut produced ZERO movement on Exp 14 (0.4650 → 0.4642) while breaking Exp 9 (0.544 → 0.467 vs real 0.560) and worsening Exp 7 (0.504 → 0.495 vs real 0.580). The small-gap distrust hypothesis for Exp 14 is falsified. (3) REVERT beta to [0.38, 0.46]; the nudge made Exp 2 worse. (4) Set s_r to an intermediate [0.052, 0.062] (between iter-1's 0.046-0.058 and iter-2's 0.060-0.072): the widening helped Exp 15 but not Exp 7, so take half the step. (5) NEW DIRECTION for the three residual misses (Exp 14, 7, 15), which form a coherent pattern: rebalance peak distrust from the composition-INDEPENDENT kernel to the composition-GATED term — shallow the base peak (rho_e down toward [0.55, 0.80], or raise w_floor toward [-0.38, -0.30]) so a singleton 0.99 at gap ~0.05 on a low-chance panel is taken at roughly face value (Exp 14 follow should rise from 0.464 toward 0.57), and RAISE c_near toward [0.95, 1.25] so chance-laden panels (Exps 13, 15, and Exp 7's gap-0.09 panel) retain or recover deep outlier distrust through comp_near. This keeps every prescribed EOBOT element intact (sign votes, plateau/tilt, kappa_gate cap, bounded saturating shrinkage, softmax+lapse) — it is a re-weighting between two existing branches, not a redesign. (6) MANDATORY before committing: verify Exp 14's actual validity vector — its gap, chance fraction, and which side the chance/mid-band cues fall on relative to the 99's endorsement. If Exp 14 turns out to be chance-laden (c_frac above c_half), the rho_e↓/c_near↑ swap will backfire there and the miss must instead be attacked via kappa_gate (chance cues under-counted, dragging the endorsed side) or the within-plateau tilt. (7) Optionally, for Exp 7 alone, the untried tilt route (raise tilt toward [0.10, 0.12] so 0.90-class experts outrank the discounted outlier on conflict trials) remains available as a separate, single-knob step — but change one knob per iteration so the next gate decision is interpretable. Do NOT touch the far-gap/Exp 6 calibration, the Exp 5 chance-flank fit, or the peak-of-kernel cells that are currently good (Exps 9's D-cells, 10, 13 at iter-1 settings).

**Outcome of your advice:** iter 2 candidate loss=0.0582 -> iter 3 candidate loss=0.0652 -> the gate marked it REJECTED.

### Iteration 3
Verdict: regenerate
Interpretation: The iter-3 candidate was REJECTED by the gate (0.0652 vs accepted 0.0521), so the verdict is regenerate regardless of merits; the base reverts to iter-1. This is the second consecutive rejection of my advice, and the per-experiment accounting shows my iter-2 'rebalance' prescription (shallow the composition-independent kernel, carry peak depth on comp_near) failed in a diagnostic way. What actually moved, iter-1 → iter-3: WINS: Exp 14 improved (0.4650 → 0.5024, still -0.069 short of 0.5716) and Exp 12 improved to near-perfect (0.4271 → 0.4617 vs real 0.4525) — confirming that shallowing the base kernel helps singleton-99, gap≈0.05 panels. LOSSES: Exp 7 collapsed further (0.5036 → 0.4618 vs real 0.5796), Exp 15 collapsed (0.4319 → 0.3822 vs real 0.4830), Exp 16 overshot (0.563 → 0.583 vs real 0.5474), and small regressions on Exps 2, 4, 6, 9. The critical anomaly: the raised c_near and sharpened comp_gate were supposed to DEEPEN distrust on Exps 7 and 15 (chance-laden panels) and instead anti-follow DROPPED on both — the compensation did not materialize. The most parsimonious explanation is that on those panels comp_gate is effectively closed (chance fraction below c_half ≈ 0.46) or E is small at their gaps, so the ONLY active edit was the base shallowing — which is exactly the wrong direction for cells that need MORE outlier distrust. In other words, the rebalance was not a rebalance; it was an unopposed shallowing applied to panels the composition gate never reaches. Two iterations have now been burned tuning knobs against INFERRED panel structures: the iter-2 null result on Exp 14 already falsified one guess, and iter-3's pattern (helps 12/14, hurts 7/15/16) falsifies the assumption that Exps 7/15 sit in the comp-gate-open regime. On faithfulness: the candidate stays inside the prescribed EOBOT architecture (sign votes, plateau/tilt, kappa_gate cap, bounded saturating distrust, softmax+lapse); the retained composition branches remain the empirically vindicated deviation (Exp 6 vs Exp 16 dissociation), and I do not ask for their removal.
Rationale: REGENERATE — build on the unchanged iter-1 accepted base. My iter-2 rebalance advice was wrong and the gate confirmed it; do NOT repeat the rho_e↓/w_floor↑/c_near↑ direction, and do not over-correct back beyond iter-1 either (that would be oscillation). Concretely: (1) REVERT rho_e to iter-1's [0.80, 1.05], w_floor to [-0.50, -0.38], c_near to [0.65, 0.95], s_c to [0.05, 0.07], kappa_lo to [0.15, 0.25]. (2) KEEP the single validated edit from iter-2: the sharpened far-fraud gate (g_far [0.198, 0.206], s_far [0.018, 0.024]) — it fixed Exp 16 (-0.040 → +0.015) at negligible cost to Exp 6 — and set s_r to the intermediate [0.052, 0.062]. This yields the minimal candidate 'iter-1 + far-gate fix + s_r midpoint', a combination whose components have each been individually validated but never tested together without the confounding s_l/lambda_ext/beta edits that sank iter-2. Expected: Exp 16 Δ ≈ +0.015, Exp 15 Δ ≈ -0.034, Exp 5 Δ ≈ +0.005, everything else at iter-1's already-good values — this should land clearly below the 0.0521 accept floor. Bank that win first. (3) MANDATORY before any further attack on the two dominant residual errors (Exp 14 -0.107, Exp 7 -0.076): produce an explicit panel-level accounting table — for EACH of Exps 7, 12, 13, 14, 15, 16, write out the actual validity vector and compute gap, c_frac, sig_gap, E(gap), gate, comp_gate, and the full subjective weight vector at the parameter midpoints. Two iterations have failed because knobs were turned against guessed gap/c_frac values; the iter-3 result itself is the evidence (Exps 7/15 did not respond to comp_near as a gate-open panel would). Only after that table exists, choose the next single knob. (4) If the table shows Exp 14 and Exp 7 share a low-c_frac, similar-gap regime yet need opposite follow signs (0.572 vs 0.580 anti), then NO kernel-depth edit can separate them and the difference must be sought in the mid-band/flank treatment (within-plateau tilt, kappa_gate promotion, or the flank slope), since the two panels differ in their non-outlier composition — but make that determination from the table, not from another inference. (5) If a smaller step of the iter-3 direction is ever retried for Exp 12/14 (it did help both), take at most half the step (e.g., rho_e [0.70, 0.85] with w_floor unchanged) and only after confirming from the table that Exps 7/15/16 are protected by something other than base kernel depth. Do NOT touch the Exp 6 far-gap calibration, the Exp 5/13 peak cells, or the softmax/lapse settings.

**Outcome of your advice:** iter 3 candidate loss=0.0652 -> iter 4 candidate loss=0.0569 -> the gate marked it REJECTED.

### Iteration 4 (most recent)
Verdict: regenerate
Interpretation: The iter-4 candidate (iter-1 base + sharpened far-gate + s_r midpoint) was REJECTED (0.0569 vs 0.0521), so the verdict is regenerate regardless. The per-experiment accounting of iter-1 → iter-4 is informative in three ways. (1) The far-gate edit worked directionally but overshot: Exp 16 moved from 0.507 (under-follow, comp_far leaking ~0.85) to 0.590 (over-follow, comp_far ~0.25) against a real 0.547 — and critically, Exp 15 (the same gap-0.14 majority-chance regime, double-99) still over-follows (anti 0.436 vs real 0.483). The two gap-0.14 cells now err in the SAME direction (too little distrust), whereas at iter-1 they pulled oppositely. The gap-0.14 target is now bracketed: comp_far(0.14) needs to sit between iter-1's and iter-4's values. (2) Several 'regressions' are noise, not mechanism: Exps 2, 4, 8, 10 moved 0.01-0.034 between iter-1 and iter-4 even though the edited parameters (far-gate, s_r) are structurally inert on those panels (no above-ceiling claim, or gap below every edited regime). Run-to-run variance is ~0.01-0.03 per cell — the same magnitude as the effects being chased. This is why 'bank the minimal combination' failed: its true gain (~0.02 on one cell) is below the noise floor. (3) The s_r widening leaks into the peak cells whenever the sampled g_star falls below 0.05 (gap-0.05 panels then use s_r as their width), so s_r is not a clean '0.09-0.14 only' knob. The stubborn residuals after four iterations are Exp 14 (-0.103, essentially unmoved by s_l, lambda_ext, or rho_e edits), Exp 7 (-0.075, unmoved), Exp 15 (-0.047), Exp 16 (+0.042), Exp 2 (+0.040). The central diagnostic finding: Exp 14 (gap ~0.05, low chance fraction, singleton 99, comp gate closed) needs a POSITIVE w_top while Exp 7 (gap 0.09, c_frac ~0.125, singleton 99, comp gate closed) needs a NEGATIVE w_top. With the kernel peaked at g_star = 0.05 and distrust monotone in E(gap), w_top(0.05) <= w_top(0.09) ALWAYS — no parameterization within the current kernel shape can deliver the required ordering. This is not a depth problem; it is a shape problem, and it explains why three consecutive depth-oriented knob attempts (s_l/lambda_ext, rho_e/w_floor/c_near, far-gate/s_r) each failed to move these two cells. The candidate remains faithful to the prescribed EOBOT architecture (sign votes, plateau/tilt, kappa_gate cap, bounded saturating shrinkage, softmax+lapse); the retained composition branches remain the empirically vindicated deviation (Exp 6 vs Exp 16 dissociation) and I do not ask for their removal.
Rationale: REGENERATE — build on the unchanged iter-1 accepted base. My iter-3 'bank the minimal combination' advice was rejected by the gate; do not re-attempt micro-combinations, which are below the demonstrated noise floor. Instead make one coordinated, structurally motivated move. (1) MANDATORY FIRST: produce the full panel table for Exps 7, 11, 12, 13, 14, 15, 16 — validity vector, gap, c_frac, sig_gap, E(gap), gate, comp_gate, the complete subjective weight vector at parameter midpoints, AND which side the mid-band/chance cues fall on in the metric-relevant trials. The iter-4 rationale's table omitted Exps 7 and 14, the two dominant residuals; every failed iteration has edited knobs against guessed panel structure. (2) PRIMARY STRUCTURAL FIX (if the table confirms Exp 14 at gap ~0.05 with low c_frac): shift the kernel peak rightward — g_star in [0.065, 0.080], steep left flank s_l in [0.018, 0.026] (keeps E(0.01) ~ 0, protecting the dense-ladder cells Exps 9/10), steep right flank s_r in [0.030, 0.042] (distrust falls by gap 0.14), and rho_e in [1.3, 1.8] so gap-0.09 distrust is deep. This is the only in-family resolution of the proven contradiction: Exp 7 (gap 0.09) sits at/near the new peak and gets the anti-follow it needs (w_top ~ w_floor < 0), Exp 14 (gap 0.05) sits on the shallow left flank and is followed, dense ladders are untouched. Note this moves g_star outside the arbiter's stated [0.045, 0.055] box — justify it explicitly: three gate-falsified attempts within the box plus the mathematical impossibility argument (a peak-at-0.05 kernel gives w_top(0.05) <= w_top(0.09) for any depths) warrant the test; the mechanism (peaked spurious-precision kernel) is unchanged. (3) COMPENSATION: with E(0.05) shrinking, the gap-0.05 high-c_frac anti-vote (Exps 13/12) must be carried by comp_near — raise c_near to [1.0, 1.5], or give comp_near its own wider left flank so it stays near-maximal at gap 0.05 while the base kernel does not (a minor in-family gating change). Verify from the table that Exp 13 stays ~0.29 and Exp 12 ~0.45. (4) FAR-GATE: set the bracketed intermediate — g_far in [0.190, 0.198], s_far in [0.022, 0.028], targeting comp_far(0.14) ~ 0.4-0.6 (between iter-1's ~0.85 and iter-4's ~0.25) so Exps 15/16 land near 0.48/0.55, while remaining fully open at gap 0.24 (Exp 6 SSR must stay ~0.118). (5) MONITOR the risk cells before submitting: Exp 11 (gap 0.09, c_frac 0.5, comp gate OPEN) will receive deepened 99-distrust from the peak shift plus raised c_near — its discriminant index (real 0.0206) must not drift past ~0.06 toward pi_7's 0.076; if the midpoint computation shows that, cap the shift (g_star <= 0.070) or lower rho_e and lean on the comp branch. (6) NOISE CONTROL: compute noise-free midpoint predictions of every metric before submitting, and if the harness allows, increase simulated subjects/parameter draws; do not submit candidates whose predicted true gain is under ~0.02 aggregate. (7) Do NOT re-attempt: s_l narrowing at fixed g_star=0.05, lambda_ext cuts, beta nudges, or unaccompanied rho_e/w_floor shallowing — all falsified by the gate. (8) DEFERRED to a later single-knob iteration: Exp 2's over-tallying (+0.040, no-outlier panel) plus Exp 8's slight under-follow are jointly addressable by raising the within-plateau tilt toward [0.10, 0.14] (top of the arbiter's box) — but only after the structural fix is banked, since tilt touches nearly every cell. Do not switch mechanism families; everything above is parameter-range and kernel-shape adjustment within the prescribed EOBOT architecture.

**Outcome of your advice:** iter 4 candidate loss=0.0569 -> CURRENT candidate loss=0.0630 -> the gate marked it REJECTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The gate rejected iter-5 (0.0630 vs accepted 0.0521), so the verdict is regenerate regardless of merits; the base reverts to iter-1. The per-experiment accounting shows my iter-4 structural advice was substantially VINDICATED on its primary targets and simultaneously sabotaged by a calibration error on one cell. WINS (keep these mechanisms): Exp 14 \u2014 the dominant residual for four iterations \u2014 was fixed exactly as designed (0.4650 \u2192 0.5682 vs real 0.5716); Exp 16 was fixed (0.5468 vs real 0.5474); Exp 13 held (0.2957 vs 0.2900); Exp 1 improved. The peak relocation (g_star \u2248 0.088), the e_min uniform-skepticism floor, the split comp_near kernel, and the bracketed far gate are now empirically validated components. COLLATERAL: Exp 7 collapsed (0.5036 \u2192 0.4040 vs real 0.5796; at -0.176 this is the worst single-cell error in the entire loop), Exp 15 flipped from -0.051 to +0.057 over-anti, Exp 12 flipped from -0.025 to +0.050 over-follow, Exp 3 regressed to -0.045, Exp 2 drifted to +0.046, Exp 11 sits at +0.019 (nearing the 0.06 contamination guard). The net loss ROSE because this collateral outweighs the two big fixes. The decisive diagnostic: the proposer's own 'noise-free midpoint predictions' missed the actual simulated values by 0.05-0.18 on exactly the targeted cells (Exp 7 predicted error -0.002, actual -0.176; Exp 15 predicted -0.001, actual +0.057; Exp 12 predicted -0.015, actual +0.050). Hand reconstruction of metrics has now failed twice. On Exp 7 specifically, the arithmetic is damning: at the proposer's claimed panel (gap 0.09) and the parameter midpoints, the code yields w_top \u2248 -0.49, and direct trial-level computation over Exp 7's ten 99-discriminating trial families gives an anti index \u2248 0.55 \u2014 not the observed 0.404. The observed collapse is only consistent with the 99 sitting at a SMALL gap (~0.04-0.05, e.g. a 0.99-over-0.95 rung), where the relocated kernel's steep left wall (s_l \u2248 0.013) delivers near-ZERO distrust and the outlier is followed at ~face value (w_top \u2248 +0.6 \u2192 anti \u2248 0.42-0.46, matching the simulation). Conclusion: the mandatory panel table's Exp-7 row is WRONG \u2014 the gap (and possibly c_frac) was mis-stated, the relocation was aimed at a nonexistent gap-0.09 cell, and this also retroactively explains why three consecutive depth-knob iterations never moved Exp 7: the knobs being turned (s_r, rho_e, w_floor at gap 0.09) are inert if the true gap sits on the left wall. The candidate remains faithful to the prescribed EOBOT architecture throughout; the failure is calibration and verification, not mechanism.",
  "verdict": "regenerate",
  "rationale": "REGENERATE \u2014 build on the unchanged iter-1 accepted base, re-applying iter-5's validated structure with the Exp-7 miscalibration corrected. (1) MANDATORY FIRST, NON-NEGOTIABLE: extract the ACTUAL validity vectors from the experiment configurations for Exps 2, 3, 7, 12, 14, 15 \u2014 every cell that moved unexpectedly \u2014 and recompute gap, c_frac, and the full weight vector. The iter-5 Exp-7 result is arithmetically inconsistent with the table's claimed gap 0.09 (which predicts anti \u2248 0.55 at the delivered w_top \u2248 -0.49) and consistent with a small gap ~0.04-0.05. Four of five rejections trace to guessed or mis-transcribed panel structure; do not touch another knob until the true panels are in hand. (2) RE-APPLY the validated iter-5 edits on the iter-1 base, unchanged: g_star [0.084, 0.092], s_l [0.011, 0.015], s_r [0.027, 0.033], e_min [0.05, 0.07], rho_e [2.3, 2.7], w_floor [-0.58, -0.52], the split comp_near with its own narrow kernel at g_near \u2248 0.05, and the far gate at g_far [0.186, 0.196], s_far [0.018, 0.022]. These fixed Exps 14, 16, and 13 simultaneously \u2014 that combination is the loop's biggest structural win and must be preserved exactly. (3) EXP 7: if the true panel shows small gap with HIGH chance fraction, carry its anti-follow through comp_near \u2014 widen s_near leftward and/or raise c_near so L \u2248 1 at Exp 7's actual gap, and position c_half between Exp 14's c_frac (gate closed, 99 followed) and Exp 7's c_frac (gate open, 99 distrusted). This is the in-family resolution: the base kernel stays shallow at small gaps (protecting Exp 14), and composition carries Exp 7's distrust. If instead the true panel shows small gap with LOW c_frac, then Exps 7 and 14 are structurally near-identical singleton-99 panels demanding opposite behavior \u2014 no weight-only knob can separate them \u2014 and the proposer must report the actual panel back rather than guess at a lever. (4) EXP 12 (+0.050 over-follow, gap 0.05, c_frac 0.5) needs MORE distrust at gap 0.05 under an open comp gate \u2014 the same comp_near strengthening as (3); these two fixes reinforce each other, verify Exp 13 (~0.29) does not over-shoot. (5) EXP 15 (+0.057 over-anti, double-99): with each 99 at positive w_top, the excess anti likely comes from promoted chance cues opposing the pair; prefer c_frac-separating levers (raise c_half so Exp 15's c_frac \u2248 0.4 gate closes) over a global kappa_gate cut, and verify Exp 16 (currently perfect at 0.5468, c_frac 0.556) does not move. (6) Exps 2/3 moved despite supposedly no-outlier panels \u2014 verify their panels; if they contain 0.99s the relocation now touches them and they must be added to the monitor list; if not, treat the drift as run-to-run noise (demonstrated 0.01-0.03/cell) and do not chase it. (7) PROCESS RULE, binding going forward: before submitting, run the actual predict\u2192simulate\u2192metric pipeline at parameter midpoints AND at the range corners; submit only if the SIMULATED (not hand-reconstructed) values land within ~0.02 on every targeted cell. Iter-5's submission missed its own predictions by up to 0.18 \u2014 that gap between analytic reconstruction and simulation is the single largest source of wasted iterations in this loop. (8) Do NOT repeat the gate-falsified moves: s_l narrowing at g_star = 0.05, lambda_ext cuts, beta nudges, unaccompanied rho_e/w_floor shallowing, and sub-noise micro-combinations. Everything requested is parameter-range, kernel-shape, and gating adjustment inside the prescribed EOBOT architecture; no mechanism-family change."
}
```

## Usage

```json
{
  "input_tokens": 50292,
  "output_tokens": 25292,
  "reasoning_tokens": 23565,
  "provider": "Sail Research"
}
```
