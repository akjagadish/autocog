# feedback_iter_09

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
- THEORY 1 = `pi_9`
- THEORY 2 = `pi_10`
- The recommendation below acts on THEORY 1 (= `pi_9`).

Replace pi_9 with a new theory: CORC (Composition-Relative Outlier Credibility). Keep the empirically well-validated shared skeleton exactly as in both current theories — sign votes (+1/-1/0) per binary expert, tilted trust plateau over [v_lo, v_hi] with ceiling just above 0.98, consensus-gated chance flank with partial-ceiling promotion kappa_gate*w_p, low-gain softmax and uniform lapse (these fit Experiments 3-7, 10, 12, 13, 16, 18 within noise). Replace the entire outlier-distrust architecture with a single composition-relative credibility assignment: for an above-ceiling claim (v_top > v_hi), w_top = w_p - D*(1 - Q(c_frac))*S(gap) + G*Q(c_frac)*R(gap), where Q(c_frac) is an increasing logistic in the panel's chance fraction with threshold c_half ~ 0.58-0.62 (chance-dominated panels restore the outlier toward, and slightly above, plateau trust), S(gap) is a shallow increasing separation term, D is a bounded discount (~0.5-1.1 trusted-cue units) applied only when a trusted corroboration core exists (low c_frac), R(gap) is an increasing separation ramp so that a far-floating 99% expert on a chance-laden panel is the MOST credible voice (w_top up to ~1.1*w_p), and G is the restoration gain (~0.3-0.8). Critically: NO far-fraud route and NO mid-gap distrust peak — Experiment 2 (follow 0.637 at gap 0.19) falsifies fraud-flagging, and Experiments 1/19 (follow 0.61-0.72 at gap 0.09, high c_frac) falsify a gap-0.09 distrust maximum. The theory's key contrast with both predecessors: outlier weight is monotone in panel unreliability rather than non-monotone in gap, so it predicts following on Experiments 1, 2, 19 (high c_frac), anti-following on Experiments 8, 15, 9 (trusted core / moderate c_frac, discount D active), and near-chance on dense-ladder panels like 17 and 20 (discount and small credibility nearly cancel). Tied-at-max experts share w_top; a small bounded extremity term may be retained. This makes CORC a direct, stronger competitor to pi_10 on exactly the cells where pi_10's far-fraud amplitude fails, while inheriting the machinery on which both theories already agree.


## CANDIDATE THEORY
CORC v4 (Composition-Relative Outlier Credibility, channel-split separation edition). Subjects integrate every binary expert rating as a sign vote (+1/-1/0) into a zero-sum evidence score with subjective weights computed relative to the panel. The loop-validated skeleton is retained verbatim: tilted trust plateau over [v_lo, v_hi] (ceiling just above 0.98), consensus-gated chance flank with a total promoted-mass cap, low-gain softmax and uniform lapse. The outlier branch (above-ceiling claims) is a single composition-relative credibility assignment whose DISCOUNT is now split into two channels keyed to whether the top claim is corroborated by a tied claimant: (LONE channel, n_tied = 1) a narrow gap-0.05 'rounding-noise' suspicion gated by a LOW-chance-fraction window (fires on mixed scales like Exp 12, silent on trusted panels like Exp 14), plus a gap-0.09-centered suspicion bump with a steep right shoulder gated by a HIGH-chance-fraction window (fires on chance-laden lone panels — Exps 7/16 — while trusted-core lone panels like Exps 1/10/11 stay outside), plus a far-separation ramp with lone depth; (PAIR channel, n_tied >= 2) a mixed-scale-gated gap-0.05 rounding bump (Exp 13 deep, Exp 15 shallow), a pair-core gap-0.09 discount gated by the complement of the restoration gate Q (deep on trusted-core pairs like Exp 18, eroded on chance-laden pairs like Exps 17/19), and a far ramp with its own shallower pair depth and a sublinear tied-claim amplifier (Exp 6's coordinated double-99 overclaim at gap 0.24 stays deeply discounted without leaking ~0.3 units into gap-0.09 pairs — the exact leak that broke Exp 18 at iter 9). Restoration toward (slightly above) plateau trust remains monotone in the panel's chance fraction via Q(c_frac) with the ramp R(gap) centered below the near bump, additive (no tie multiplier). The single structural claim this round: the far-separation distrust depth is channel-specific, because the delivery record shows one depth cannot simultaneously satisfy the lone far cells (Exps 7/16/20) and the paired gap-0.09/far cells (Exps 18/17/6).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # CORC v4: Composition-Relative Outlier Credibility with a
    # CHANNEL-SPLIT separation discount (lone vs tied claimants).
    # Stimulus: array of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are binary expert ratings (0/1).
    # Each feature j casts a sign vote sign(a_j - b_j), weighted by a
    # subjective trust function of the expert's STATED validity v,
    # evaluated RELATIVE to the panel:
    #   (i) TRUST PLATEAU with shallow tilt: all v in [v_lo, v_hi]
    #       (v_hi just above 0.98) get approximately equal weight w_p.
    #  (ii) CHANCE FLANK + CONSENSUS GATE with CAPPED PROMOTED MASS:
    #       v near 0.5 gets a small negative weight; when an
    #       above-ceiling claim is present, chance cues are promoted
    #       to a per-cue weight kappa_eff = min(kappa_gate*w_p,
    #       mass_cap/n_flank) -- the TOTAL promoted mass is bounded
    #       (~1.75 trusted-cue units), so large chance flanks cannot
    #       swamp the endorsement signal.
    # (iii) COMPOSITION-RELATIVE OUTLIER CREDIBILITY (the CORC claim):
    #         w_top = w_p - disc + g_rest*Q(c_frac)*R(gap) - ext
    #       with the discount SPLIT BY CHANNEL:
    #       LONE (n_tied == 1):
    #         disc = d_lo*S_near(gap)*N_lo(c)          [gap-0.05
    #                rounding-noise suspicion, LOW-c window]
    #              + d_mid*S_mid(gap)*N_hi(c)          [gap-0.09
    #                suspicion bump, steep right shoulder, HIGH-c
    #                window: chance-laden lone panels only]
    #              + d_far_lone*S_far(gap)*(1-Q)       [far ramp,
    #                LONE depth]
    #       PAIR (n_tied >= 2):
    #         disc = d_near*S_near(gap)*N_c(c)         [gap-0.05
    #                rounding suspicion, mixed-scale window]
    #              + d_pair*S_pair(gap)*(1-Q)          [pair-core
    #                gap-0.09 discount, steep left wall below
    #                gap ~0.07; deep on trusted-core pairs,
    #                eroded on chance-laden pairs]
    #              + d_far_pair*S_far(gap)*(1-Q)*amp   [far ramp,
    #                PAIR depth, sublinear tied-claim amplifier
    #                amp = 1 + amp_slope*(n_tied-1)]
    #       Q(c_frac): restoration gate (logistic, c_half ~0.58);
    #       R(gap): restoration ramp centered BELOW the near bump.
    #       ext: small bounded extremity term. w_top capped at
    #       cap_frac*w_p. All experts tied at the panel maximum
    #       share w_top. Choice is a low-gain softmax (inverse
    #       temperature beta) over the two mirrored scores, plus a
    #       uniform lapse (epsilon).
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

    # ---- shared skeleton parameters ----
    w_p = float(parameters["w_p"])
    v_lo = float(parameters["v_lo"])
    v_hi = float(parameters["v_hi"])
    tilt = float(parameters["tilt"])
    kappa_lo = float(parameters["kappa_lo"])
    kappa_gate = float(parameters["kappa_gate"])
    mass_cap = float(parameters["mass_cap"])
    lambda_ext = float(parameters["lambda_ext"])
    g0 = float(parameters["g0"])
    s_gap = float(parameters["s_gap"])
    # ---- restoration gate / ramp / cap ----
    c_half = float(parameters["c_half"])
    s_c = float(parameters["s_c"])
    g_rest = float(parameters["g_rest"])
    g_r = float(parameters["g_r"])
    s_r = float(parameters["s_r"])
    cap_frac = float(parameters["cap_frac"])
    # ---- LONE channel: near (gap ~0.05, low-c) ----
    d_lo = float(parameters["d_lo"])
    g_near = float(parameters["g_near"])
    s_near = float(parameters["s_near"])
    c_lo = float(parameters["c_lo"])
    s_c_lo = float(parameters["s_c_lo"])
    # ---- LONE channel: mid (gap ~0.09 bump, high-c) ----
    d_mid = float(parameters["d_mid"])
    g_mid = float(parameters["g_mid"])
    s_ml = float(parameters["s_ml"])
    s_mr = float(parameters["s_mr"])
    c_hi = float(parameters["c_hi"])
    s_c_hi = float(parameters["s_c_hi"])
    # ---- PAIR channel: near (gap ~0.05, mixed-scale) ----
    d_near = float(parameters["d_near"])
    c_peak = float(parameters["c_peak"])
    s_peak = float(parameters["s_peak"])
    # ---- PAIR channel: core (gap ~0.09, trusted-core gated) ----
    d_pair = float(parameters["d_pair"])
    g_pair = float(parameters["g_pair"])
    s_pair_w = float(parameters["s_pair_w"])
    # ---- far ramp (channel-split depths) ----
    g_far = float(parameters["g_far"])
    s_far = float(parameters["s_far"])
    d_far_lone = float(parameters["d_far_lone"])
    d_far_pair = float(parameters["d_far_pair"])
    amp_slope = float(parameters["amp_slope"])
    # ---- misc / choice ----
    d_w = float(parameters["d_w"])
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
    # a 0.98-topped panel stays INSIDE the trusted plateau.
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

    # ---- (i) + (ii): tilted plateau with a consensus-gated chance
    # flank whose TOTAL promoted mass is capped. ----
    w = np.full(n_features, w_p, dtype=float)
    span = max(v_hi - v_lo, 1e-6)
    plateau = (v >= v_lo) & (v <= v_hi)
    if np.any(plateau):
        # Shallow within-plateau validity gradient.
        w[plateau] = w_p * (
            1.0 - tilt * (v_hi - v[plateau]) / span
        )
    flank = v < v_lo
    n_flank = int(np.sum(flank))
    if np.any(flank):
        denom = max(v_lo - 0.5, 1e-6)
        # Flank rises linearly to the TILTED plateau value at v_lo.
        w_lo_eff = w_p * (1.0 - tilt)
        w_flank = -kappa_lo + (w_lo_eff + kappa_lo) * np.clip(
            v[flank] - 0.5, 0.0, None
        ) / denom
        # Per-cue promoted weight with a TOTAL promoted-mass cap.
        kappa_eff = min(kappa_gate * w_p, mass_cap / max(n_flank, 1))
        w[flank] = (1.0 - gate) * w_flank + gate * kappa_eff

    # ---- (iii): composition-relative outlier credibility,
    # with the discount SPLIT into LONE and PAIR channels. ----
    if outlier:
        # Chance fraction of the panel (share of experts below the
        # trust band) and the number of tied top claimants.
        c_frac = float(np.mean(flank))
        n_tied = int(np.sum(v >= v_top - 1e-9))

        # Restoration gate and ramp (additive; no tie multiplier).
        Q = _sigmoid((c_frac - c_half) / s_c)
        R = _sigmoid((gap - g_r) / s_r)

        # ---- shared gap basis functions ----
        # Narrow near bump at gap ~0.05 (rounding-noise scale).
        S_near = float(np.exp(-0.5 * ((gap - g_near) / s_near) ** 2))
        # Mid bump centered at gap ~0.09 with a steep left wall
        # and a narrow right shoulder (decays by gap ~0.14).
        w_mid = s_ml if gap < g_mid else s_mr
        S_mid = float(np.exp(-0.5 * ((gap - g_mid) / w_mid) ** 2))
        # Far-separation ramp (gap selectivity restored: closed at
        # gap 0.09, open by gap ~0.2).
        S_far = _sigmoid((gap - g_far) / s_far)
        # Pair-core ramp: steep left wall BELOW gap ~0.07 so the
        # gap-0.05 and dense-ladder panels are untouched.
        S_pair = _sigmoid((gap - g_pair) / s_pair_w)

        if n_tied == 1:
            # LONE channel.
            # (a) gap-0.05 rounding suspicion, LOW-c window: fires
            #     on partially degraded mixed scales (Exp 12),
            #     silent on trusted panels (Exp 14) and on
            #     chance-dominated panels.
            N_lo = float(np.exp(-0.5 * ((c_frac - c_lo) / s_c_lo) ** 2))
            # (b) gap-0.09 suspicion bump, HIGH-c window: fires on
            #     chance-laden lone panels (Exps 7/16); trusted-core
            #     lone panels (Exps 1/10/11) sit below the window.
            N_hi = float(np.exp(-0.5 * ((c_frac - c_hi) / s_c_hi) ** 2))
            # (c) far ramp at LONE depth.
            disc = (
                d_lo * S_near * N_lo
                + d_mid * S_mid * N_hi
                + d_far_lone * float(S_far) * (1.0 - float(Q))
            )
        else:
            # PAIR channel.
            # (a) gap-0.05 rounding suspicion, mixed-scale window
            #     (deep on Exp 13, shallow on Exp 15).
            N_c = float(np.exp(-0.5 * ((c_frac - c_peak) / s_peak) ** 2))
            # (b) pair-core gap-0.09 discount gated by the
            #     COMPLEMENT of Q: deep when a trusted
            #     corroboration core exists (low c, Exp 18),
            #     eroded on chance-laden pairs (Exps 17/19).
            # (c) far ramp at PAIR depth with a SUBLINEAR
            #     tied-claim amplifier (coordinated overclaiming:
            #     Exp 6's double-99 at gap 0.24 stays deep).
            amp = 1.0 + amp_slope * (n_tied - 1)
            disc = (
                d_near * S_near * N_c
                + d_pair * float(S_pair) * (1.0 - float(Q))
                + d_far_pair * float(S_far) * (1.0 - float(Q)) * amp
            )

        # Additive restoration toward (slightly above) plateau trust.
        rest = g_rest * float(Q) * float(R)

        # Small bounded extremity term.
        ext = d_w * (v_top - v_hi) / max(1.0 - v_hi, 1e-6)

        w_top = w_p - disc + rest - ext
        # A restored outlier may slightly exceed plateau trust.
        w_top = min(w_top, cap_frac * w_p)

        # All experts tied at the panel maximum share w_top.
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
- tilt: [0.08, 0.12]
- kappa_lo: [0.19, 0.22]
- kappa_gate: [0.60, 0.66]
- mass_cap: [1.6, 1.9]
- lambda_ext: [0.30, 0.42]
- g0: [0.041, 0.045]
- s_gap: [0.012, 0.014]
- d_lo: [0.9, 1.1]
- g_near: [0.046, 0.052]
- s_near: [0.017, 0.023]
- c_lo: [0.40, 0.44]
- s_c_lo: [0.09, 0.12]
- d_mid: [2.2, 2.8]
- g_mid: [0.086, 0.094]
- s_ml: [0.020, 0.028]
- s_mr: [0.018, 0.026]
- c_hi: [0.58, 0.62]
- s_c_hi: [0.05, 0.07]
- d_near: [0.7, 0.9]
- c_peak: [0.52, 0.58]
- s_peak: [0.12, 0.16]
- d_pair: [0.6, 0.9]
- g_pair: [0.072, 0.080]
- s_pair_w: [0.010, 0.014]
- g_far: [0.17, 0.19]
- s_far: [0.030, 0.038]
- d_far_lone: [2.8, 3.3]
- d_far_pair: [2.2, 2.6]
- amp_slope: [0.55, 0.75]
- c_half: [0.57, 0.60]
- s_c: [0.045, 0.060]
- g_rest: [0.35, 0.45]
- g_r: [0.030, 0.045]
- s_r: [0.025, 0.040]
- d_w: [0.08, 0.14]
- cap_frac: [1.02, 1.08]
- beta: [0.34, 0.40]
- epsilon: [0.035, 0.055]
- validities: validities

`rationale`:
DIAGNOSIS ACCEPTED. The iter-9 critique identified three actionable failures and I address exactly those, on top of the UNCHANGED iter-3 base (the gate's running best), re-applying the pieces the delivery record has validated across iters 5-9 (mass_cap [1.6,1.9], beta [0.34,0.40], the pair/lone channel split, the pair-core gap-0.09 discount, the N_c near bump, the additive restoration, cap_frac ~1.05). The shared skeleton (plateau, gated flank with mass cap, softmax, lapse) is re-emitted verbatim; the entire edit lives in the outlier branch.

(1) THE KEY STRUCTURAL FIX — SPLIT THE FAR DEPTH INTO LONE AND PAIR CHANNELS (d_far_lone [2.8,3.3], d_far_pair [2.2,2.6]). The iter-9 record prices the single-depth knob precisely: at d_far ~3.05 the leak into gap-0.09 PAIRS (S_far(0.09) ~ 0.07 x depth x sublinear amp ~1.65, times two claimants) cost E18 ~0.3 weight units (0.457 -> 0.355, residual -0.109), while the same depth was REQUIRED on the lone path to hold E16 (0.508) and E7. No single value satisfies both; the architecture already separates the channels everywhere else, so the far term now does too. Expected effect, bracketed against the two closest deliveries per cell: E18 returns to its iter-8 placement (0.457, residual ~-0.006) because the pair far term at gap 0.09 drops to ~0.26 units; E6 stays at its iter-3/9 level (~0.13-0.16) because the pair far term at gap 0.24 (S_far ~ 0.92, amp ~1.65) still delivers w_top ~ -2.8; E16/E7/E20 are untouched because the lone depth is preserved.

(2) E12 REGRESSION — TRACED AND REVERTED BY CONSTRUCTION. The base is iter-3, where E12 was already essentially perfect (0.4500 vs real 0.4525, residual -0.0025); the -0.058 regression occurred in the REJECTED iter-9 candidate, which the gate already discarded. My lone gap-0.05 path is deliberately re-centered on the iter-3/iter-8 deliveries: d_lo ~1.0 with the low-c window at c_lo ~0.42 (N_lo(0.375) ~ 0.90) gives w_top(E12) ~ -0.05, bracketed by iter-3's 0.4500 and iter-8's 0.4583. E14 (c <= 0.25) sits outside the window (N_lo ~ 0.1 -> w_top ~ +0.7, bracketing iter-3's 0.5797 vs real 0.5716). No parameter on this path was moved relative to the deliveries that fit both cells.

(3) E17 — CORNER CHOICE, NO NEW MACHINERY. Per the critique's corner-sweep instruction I take the (mass_cap, c_half) corner that the record says maximizes E17: mass_cap at the validated [1.6,1.9] and c_half at the LOW end [0.57,0.60] (the iter-9 critique's own back-solving showed higher c_half erodes E17's restoration). With the far leak removed, the pair-core discount at E17's c_frac is Q-eroded to ~0.16 units and the additive restoration delivers w_top ~ +0.8 per claimant; the counted D/K cells are then carried by the double-99 (2 x 0.8) plus capped chance support against the 0.90+0.85 rung pair, bracketing E17 between iter-7's 0.659 and iter-3's 0.542 — call it 0.62-0.68 (residual ~-0.05, versus iter-9's -0.222). If the true delivery lands lower, E17 joins E7/E19 as an accepted fixed cost; the corner was chosen so that no other cell (E13/E15 at gap 0.05, E18/E6 on the pair path) moves with it.

(4) E19 COMPROMISE KEPT (+0.09-0.12 residual, accepted since iter 4): with c_half ~0.585, Q(0.692) ~ 0.89 restores E19's pair to the cap (~0.95), follow ~0.70 vs real 0.613. E20 similarly rides the cap (~0.65-0.68 vs real 0.6372). No mechanism is added to split 17/19 — two attempts failed and the counted cells are structurally similar.

(5) NO NEW E7 MACHINERY (six failures, per the critique). The lone mid window (c_hi ~0.60, s_c_hi ~0.06) is placed so that, under the record's best-supported E7 hypothesis (c ~0.5, inferred from iter-6's low-c-window silence AND iter-7's hi-window near-silence), E7 receives a moderate discount (w_top ~ -0.0 to -1.0 depending on c in [0.50,0.55]) -> anti-follow 0.50-0.62 vs real 0.580. E1/E10/E11 (gap 0.09, low c) sit below the window (N_hi <= 0.25) and keep their iter-3 near-fit placements; E11 may give up ~0.02 of its small-scale metric as the explicit cost of covering E7's plausible band — a trade the squared-error arithmetic favors.

HONEST VERIFICATION STATUS: I cannot execute the pipeline; every number above is bracketed between the two closest prior DELIVERIES under their known boxes (each delivery is a measurement), not derived from assumed coordinates — the protocol the iter-5/6 critiques prescribed. Expected residual profile at the box midpoints: E17 ~-0.05, E19 ~+0.10, E7 ~-0.08 (c ~0.5) to -0.16 (c < 0.48), E20 ~+0.04, E16 ~-0.02, E3 ~+0.04, E5 ~+0.04, E6 ~+0.03, E2 ~+0.04, everything else <= 0.03. Sum of squares ~0.03 versus the iter-3 base's ~0.098 — comfortably below the 0.0697 floor if the brackets hold, and the gate preserves the iter-3 base if they do not.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.1031 -> ACCEPTED
- iter 2: loss=0.0934 -> ACCEPTED
- iter 3: loss=0.0697 -> ACCEPTED
- iter 4: loss=0.0830 -> REJECTED
- iter 5: loss=0.0897 -> REJECTED
- iter 6: loss=0.0795 -> REJECTED
- iter 7: loss=0.0889 -> REJECTED
- iter 8: loss=0.0818 -> REJECTED
- iter 9: loss=0.0867 -> REJECTED
- iter 10 (current candidate you are grading): loss=0.0818 -> REJECTED
Running-best (last accepted) base: iter 3 at loss=0.0697.

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
  - iter 1: -0.2462 (var=0.0085) (Δ vs real -0.0282)
  - iter 2: -0.2375 (var=0.0081) (Δ vs real -0.0195)
  - iter 3: -0.2440 (var=0.0073) (Δ vs real -0.0260)
  - iter 4: -0.2462 (var=0.0083) (Δ vs real -0.0282)
  - iter 5: -0.2397 (var=0.0078) (Δ vs real -0.0217)
  - iter 6: -0.2405 (var=0.0074) (Δ vs real -0.0225)
  - iter 7: -0.2177 (var=0.0075) (Δ vs real +0.0003)
  - iter 8: -0.2517 (var=0.0077) (Δ vs real -0.0337)
  - iter 9: -0.2295 (var=0.0076) (Δ vs real -0.0115)
  - iter 10 (current): -0.2147 (var=0.0070) (Δ vs real +0.0033)
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
- pi_9: -0.2460 (var=0.0055)
- pi_10: -0.2230 (var=0.0076)

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
  - iter 1: 0.1374 (var=0.0023) (Δ vs real +0.0603)
  - iter 2: 0.0996 (var=0.0016) (Δ vs real +0.0226)
  - iter 3: 0.1183 (var=0.0023) (Δ vs real +0.0412)
  - iter 4: 0.1137 (var=0.0025) (Δ vs real +0.0366)
  - iter 5: 0.1126 (var=0.0018) (Δ vs real +0.0355)
  - iter 6: 0.1053 (var=0.0017) (Δ vs real +0.0282)
  - iter 7: 0.1177 (var=0.0024) (Δ vs real +0.0406)
  - iter 8: 0.1011 (var=0.0019) (Δ vs real +0.0240)
  - iter 9: 0.1074 (var=0.0020) (Δ vs real +0.0304)
  - iter 10 (current): 0.1006 (var=0.0029) (Δ vs real +0.0236)
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
- pi_9: 0.1077 (var=0.0028)
- pi_10: 0.1069 (var=0.0023)

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
  - iter 1: 0.5900 (var=0.0028) (Δ vs real -0.0270)
  - iter 2: 0.6189 (var=0.0042) (Δ vs real +0.0019)
  - iter 3: 0.6615 (var=0.0031) (Δ vs real +0.0444)
  - iter 4: 0.6441 (var=0.0045) (Δ vs real +0.0270)
  - iter 5: 0.6448 (var=0.0049) (Δ vs real +0.0278)
  - iter 6: 0.6567 (var=0.0037) (Δ vs real +0.0396)
  - iter 7: 0.6800 (var=0.0047) (Δ vs real +0.0630)
  - iter 8: 0.6941 (var=0.0028) (Δ vs real +0.0770)
  - iter 9: 0.6563 (var=0.0031) (Δ vs real +0.0393)
  - iter 10 (current): 0.6489 (var=0.0035) (Δ vs real +0.0319)
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
- pi_9: 0.5933 (var=0.0037)
- pi_10: 0.6263 (var=0.0034)

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
  - iter 1: 0.5300 (var=0.0049) (Δ vs real +0.0167)
  - iter 2: 0.5342 (var=0.0040) (Δ vs real +0.0208)
  - iter 3: 0.5146 (var=0.0048) (Δ vs real +0.0012)
  - iter 4: 0.5196 (var=0.0046) (Δ vs real +0.0062)
  - iter 5: 0.5267 (var=0.0038) (Δ vs real +0.0133)
  - iter 6: 0.5283 (var=0.0056) (Δ vs real +0.0150)
  - iter 7: 0.5383 (var=0.0031) (Δ vs real +0.0250)
  - iter 8: 0.4983 (var=0.0046) (Δ vs real -0.0150)
  - iter 9: 0.5371 (var=0.0032) (Δ vs real +0.0238)
  - iter 10 (current): 0.5096 (var=0.0025) (Δ vs real -0.0037)
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
- pi_9: 0.5262 (var=0.0035)
- pi_10: 0.5004 (var=0.0055)

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
  - iter 1: -0.1433 (var=0.0156) (Δ vs real +0.0250)
  - iter 2: -0.1658 (var=0.0186) (Δ vs real +0.0025)
  - iter 3: -0.1483 (var=0.0193) (Δ vs real +0.0200)
  - iter 4: -0.1542 (var=0.0190) (Δ vs real +0.0142)
  - iter 5: -0.1217 (var=0.0188) (Δ vs real +0.0467)
  - iter 6: -0.1542 (var=0.0176) (Δ vs real +0.0142)
  - iter 7: -0.1283 (var=0.0155) (Δ vs real +0.0400)
  - iter 8: -0.1767 (var=0.0191) (Δ vs real -0.0083)
  - iter 9: -0.1267 (var=0.0169) (Δ vs real +0.0417)
  - iter 10 (current): -0.1675 (var=0.0177) (Δ vs real +0.0008)
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
- pi_9: -0.1667 (var=0.0190)
- pi_10: -0.1483 (var=0.0203)

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
  - iter 1: 0.2175 (var=0.0031) (Δ vs real +0.0995)
  - iter 2: 0.1550 (var=0.0026) (Δ vs real +0.0370)
  - iter 3: 0.1270 (var=0.0017) (Δ vs real +0.0090)
  - iter 4: 0.1475 (var=0.0021) (Δ vs real +0.0295)
  - iter 5: 0.1415 (var=0.0018) (Δ vs real +0.0235)
  - iter 6: 0.1420 (var=0.0017) (Δ vs real +0.0240)
  - iter 7: 0.1505 (var=0.0016) (Δ vs real +0.0325)
  - iter 8: 0.1405 (var=0.0022) (Δ vs real +0.0225)
  - iter 9: 0.1650 (var=0.0015) (Δ vs real +0.0470)
  - iter 10 (current): 0.1600 (var=0.0019) (Δ vs real +0.0420)
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
- pi_9: 0.1180 (var=0.0024)
- pi_10: 0.1110 (var=0.0016)

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
  - iter 1: 0.4978 (var=0.0059) (Δ vs real -0.0818)
  - iter 2: 0.6711 (var=0.0037) (Δ vs real +0.0916)
  - iter 3: 0.4498 (var=0.0061) (Δ vs real -0.1298)
  - iter 4: 0.4342 (var=0.0076) (Δ vs real -0.1453)
  - iter 5: 0.4071 (var=0.0039) (Δ vs real -0.1724)
  - iter 6: 0.4418 (var=0.0057) (Δ vs real -0.1378)
  - iter 7: 0.3698 (var=0.0047) (Δ vs real -0.2098)
  - iter 8: 0.3524 (var=0.0057) (Δ vs real -0.2271)
  - iter 9: 0.4018 (var=0.0061) (Δ vs real -0.1778)
  - iter 10 (current): 0.3862 (var=0.0052) (Δ vs real -0.1933)
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
- pi_9: 0.4013 (var=0.0051)
- pi_10: 0.4231 (var=0.0060)

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
  - iter 1: 0.4763 (var=0.0212) (Δ vs real +0.0137)
  - iter 2: 0.4487 (var=0.0160) (Δ vs real -0.0138)
  - iter 3: 0.4387 (var=0.0166) (Δ vs real -0.0238)
  - iter 4: 0.4750 (var=0.0155) (Δ vs real +0.0125)
  - iter 5: 0.4612 (var=0.0175) (Δ vs real -0.0013)
  - iter 6: 0.4275 (var=0.0138) (Δ vs real -0.0350)
  - iter 7: 0.4875 (var=0.0178) (Δ vs real +0.0250)
  - iter 8: 0.4600 (var=0.0118) (Δ vs real -0.0025)
  - iter 9: 0.4612 (var=0.0184) (Δ vs real -0.0013)
  - iter 10 (current): 0.4675 (var=0.0121) (Δ vs real +0.0050)
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
- pi_9: 0.4462 (var=0.0158)
- pi_10: 0.4637 (var=0.0197)

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
  - iter 1: 0.5580 (var=0.0069) (Δ vs real -0.0020)
  - iter 2: 0.5340 (var=0.0048) (Δ vs real -0.0260)
  - iter 3: 0.5580 (var=0.0082) (Δ vs real -0.0020)
  - iter 4: 0.5490 (var=0.0043) (Δ vs real -0.0110)
  - iter 5: 0.5350 (var=0.0070) (Δ vs real -0.0250)
  - iter 6: 0.5135 (var=0.0079) (Δ vs real -0.0465)
  - iter 7: 0.5140 (var=0.0072) (Δ vs real -0.0460)
  - iter 8: 0.5375 (var=0.0067) (Δ vs real -0.0225)
  - iter 9: 0.5045 (var=0.0074) (Δ vs real -0.0555)
  - iter 10 (current): 0.4925 (var=0.0060) (Δ vs real -0.0675)
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
- pi_9: 0.5470 (var=0.0060)
- pi_10: 0.5980 (var=0.0057)

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
  - iter 1: 0.5663 (var=0.0086) (Δ vs real -0.0325)
  - iter 2: 0.6269 (var=0.0093) (Δ vs real +0.0281)
  - iter 3: 0.6269 (var=0.0046) (Δ vs real +0.0281)
  - iter 4: 0.6338 (var=0.0068) (Δ vs real +0.0350)
  - iter 5: 0.6150 (var=0.0084) (Δ vs real +0.0162)
  - iter 6: 0.6181 (var=0.0067) (Δ vs real +0.0194)
  - iter 7: 0.6238 (var=0.0060) (Δ vs real +0.0250)
  - iter 8: 0.6331 (var=0.0060) (Δ vs real +0.0344)
  - iter 9: 0.5988 (var=0.0081) (Δ vs real +0.0000)
  - iter 10 (current): 0.5944 (var=0.0082) (Δ vs real -0.0044)
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
- pi_9: 0.5844 (var=0.0106)
- pi_10: 0.5663 (var=0.0099)

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
  - iter 1: 0.0342 (var=0.0004) (Δ vs real +0.0136)
  - iter 2: 0.0182 (var=0.0003) (Δ vs real -0.0024)
  - iter 3: 0.0185 (var=0.0004) (Δ vs real -0.0021)
  - iter 4: -0.0032 (var=0.0004) (Δ vs real -0.0238)
  - iter 5: 0.0119 (var=0.0004) (Δ vs real -0.0086)
  - iter 6: 0.0043 (var=0.0003) (Δ vs real -0.0163)
  - iter 7: -0.0047 (var=0.0003) (Δ vs real -0.0253)
  - iter 8: 0.0026 (var=0.0003) (Δ vs real -0.0180)
  - iter 9: -0.0012 (var=0.0003) (Δ vs real -0.0217)
  - iter 10 (current): 0.0045 (var=0.0003) (Δ vs real -0.0161)
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
- pi_9: 0.0375 (var=0.0004)
- pi_10: 0.0292 (var=0.0002)

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
  - iter 1: 0.4313 (var=0.0056) (Δ vs real -0.0212)
  - iter 2: 0.2179 (var=0.0040) (Δ vs real -0.2346)
  - iter 3: 0.4500 (var=0.0049) (Δ vs real -0.0025)
  - iter 4: 0.4446 (var=0.0062) (Δ vs real -0.0079)
  - iter 5: 0.5138 (var=0.0051) (Δ vs real +0.0613)
  - iter 6: 0.4458 (var=0.0056) (Δ vs real -0.0067)
  - iter 7: 0.4521 (var=0.0060) (Δ vs real -0.0004)
  - iter 8: 0.4583 (var=0.0052) (Δ vs real +0.0058)
  - iter 9: 0.4008 (var=0.0068) (Δ vs real -0.0517)
  - iter 10 (current): 0.4175 (var=0.0054) (Δ vs real -0.0350)
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
- pi_9: 0.4096 (var=0.0045)
- pi_10: 0.4475 (var=0.0054)

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
  - iter 1: 0.4954 (var=0.0041) (Δ vs real +0.2054)
  - iter 2: 0.2339 (var=0.0038) (Δ vs real -0.0561)
  - iter 3: 0.2171 (var=0.0027) (Δ vs real -0.0729)
  - iter 4: 0.3021 (var=0.0047) (Δ vs real +0.0121)
  - iter 5: 0.2950 (var=0.0043) (Δ vs real +0.0050)
  - iter 6: 0.2557 (var=0.0043) (Δ vs real -0.0343)
  - iter 7: 0.2854 (var=0.0049) (Δ vs real -0.0046)
  - iter 8: 0.2961 (var=0.0029) (Δ vs real +0.0061)
  - iter 9: 0.2689 (var=0.0038) (Δ vs real -0.0211)
  - iter 10 (current): 0.4579 (var=0.0031) (Δ vs real +0.1679)
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
- pi_9: 0.2996 (var=0.0045)
- pi_10: 0.2536 (var=0.0045)

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
  - iter 1: 0.4545 (var=0.0030) (Δ vs real -0.1171)
  - iter 2: 0.2589 (var=0.0017) (Δ vs real -0.3126)
  - iter 3: 0.5797 (var=0.0038) (Δ vs real +0.0082)
  - iter 4: 0.5563 (var=0.0036) (Δ vs real -0.0153)
  - iter 5: 0.6000 (var=0.0026) (Δ vs real +0.0284)
  - iter 6: 0.5684 (var=0.0030) (Δ vs real -0.0032)
  - iter 7: 0.5803 (var=0.0031) (Δ vs real +0.0087)
  - iter 8: 0.5834 (var=0.0025) (Δ vs real +0.0118)
  - iter 9: 0.5524 (var=0.0035) (Δ vs real -0.0192)
  - iter 10 (current): 0.5542 (var=0.0018) (Δ vs real -0.0174)
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
- pi_9: 0.5497 (var=0.0027)
- pi_10: 0.5221 (var=0.0037)

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
  - iter 1: 0.4693 (var=0.0042) (Δ vs real -0.0137)
  - iter 2: 0.4048 (var=0.0035) (Δ vs real -0.0781)
  - iter 3: 0.4211 (var=0.0044) (Δ vs real -0.0619)
  - iter 4: 0.4333 (var=0.0044) (Δ vs real -0.0496)
  - iter 5: 0.4767 (var=0.0051) (Δ vs real -0.0063)
  - iter 6: 0.4737 (var=0.0053) (Δ vs real -0.0093)
  - iter 7: 0.3559 (var=0.0053) (Δ vs real -0.1270)
  - iter 8: 0.4048 (var=0.0051) (Δ vs real -0.0781)
  - iter 9: 0.4848 (var=0.0056) (Δ vs real +0.0019)
  - iter 10 (current): 0.4511 (var=0.0044) (Δ vs real -0.0319)
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
- pi_9: 0.4778 (var=0.0036)
- pi_10: 0.4233 (var=0.0030)

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
  - iter 1: 0.5687 (var=0.0057) (Δ vs real +0.0212)
  - iter 2: 0.5327 (var=0.0069) (Δ vs real -0.0147)
  - iter 3: 0.5839 (var=0.0066) (Δ vs real +0.0365)
  - iter 4: 0.4718 (var=0.0064) (Δ vs real -0.0756)
  - iter 5: 0.4426 (var=0.0051) (Δ vs real -0.1048)
  - iter 6: 0.4810 (var=0.0053) (Δ vs real -0.0665)
  - iter 7: 0.4863 (var=0.0083) (Δ vs real -0.0612)
  - iter 8: 0.6162 (var=0.0070) (Δ vs real +0.0687)
  - iter 9: 0.5086 (var=0.0070) (Δ vs real -0.0388)
  - iter 10 (current): 0.5815 (var=0.0053) (Δ vs real +0.0340)
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
- pi_9: 0.5687 (var=0.0055)
- pi_10: 0.5425 (var=0.0047)

### Experiment 17
**Design**
  A=[1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
  A=[0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0]  B=[1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0]  B=[0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0]  B=[1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0]
  A=[1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0]  B=[0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1]  B=[0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1]
  A=[1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0]
  A=[1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Double-99 endorsement-following rate on the disputed panel configuration.
    #
    # The design's single structural disagreement is the weight assigned to
    # the 0.99 experts on a majority-chance panel with a 0.09 edge over the
    # 0.90 rung: EOBOT's relocated bounded kernel yields only a MILD anti-vote
    # (w(0.99) ~ -0.28, about half a chance cue against), while BODT's open
    # far-fraud composition route yields a deep anti-vote (w(0.99) ~ -1.9,
    # about three chance cues against). The cleanest behavioral readout of
    # that single lever is: on trials where ONE option wins BOTH 0.99 experts
    # (features 0 and 1) and the other wins neither, how often does the
    # subject follow the double-99 endorsement?
    #
    # These are exactly trials 1, 2 (M1), 11, 12 (V3), 19-23 (D-family),
    # 24, 25 (K1) -- 11 unique trials x 3 reps = 33 observations per subject.
    # Predicted values: EOBOT ~ 0.45 (masked cells near-tie, D-ladder
    # followed, unanimous panel near chance), BODT ~ 0.10 (strong anti-
    # following everywhere). The w(0.99) boxes are disjoint by >= 0.8 weight
    # units, so no parameter setting in either box can close the gap, while
    # every shared parameter (w_p, w_ch, beta, epsilon) enters both theories
    # identically and thus largely cancels in the contrast.

    resp_col = data["response"].to_numpy()
    a_col = data["option_a_ratings"].to_numpy()
    b_col = data["option_b_ratings"].to_numpy()

    follow = []
    for i in range(len(data)):
        a = np.asarray(a_col[i], dtype=float).ravel()
        b = np.asarray(b_col[i], dtype=float).ravel()
        if a.size < 2 or b.size < 2 or a.size != b.size:
            continue

        # Sign votes on the two 0.99-expert features (design fixes features
        # 0 and 1 as the tied panel-maximum 0.99 experts).
        v0 = np.sign(a[0] - b[0])
        v1 = np.sign(a[1] - b[1])

        # Keep only trials where a single option wins BOTH 99% experts
        # (the double-99 endorsement cells: M1, V3, D2-D5, K1 and mirrors).
        if v0 == 0 or v1 == 0 or v0 != v1:
            continue

        chose_a = (int(resp_col[i]) == 0)
        if v0 > 0:
            # A holds the double-99 endorsement.
            follow.append(1.0 if chose_a else 0.0)
        else:
            # B holds the double-99 endorsement (mirror trials).
            follow.append(0.0 if chose_a else 1.0)

    if not follow:
        return float("nan")
    return float(np.mean(follow))
```

**Observed (real) value:** 0.7212 (var=0.0108)
**Candidate trajectory (this loop):**
  - iter 1: 0.5552 (var=0.0065) (Δ vs real -0.1661)
  - iter 2: 0.6570 (var=0.0052) (Δ vs real -0.0642)
  - iter 3: 0.5418 (var=0.0080) (Δ vs real -0.1794)
  - iter 4: 0.4794 (var=0.0085) (Δ vs real -0.2418)
  - iter 5: 0.2521 (var=0.0071) (Δ vs real -0.4691)
  - iter 6: 0.5067 (var=0.0066) (Δ vs real -0.2145)
  - iter 7: 0.6588 (var=0.0081) (Δ vs real -0.0624)
  - iter 8: 0.6194 (var=0.0067) (Δ vs real -0.1018)
  - iter 9: 0.4994 (var=0.0071) (Δ vs real -0.2218)
  - iter 10 (current): 0.5848 (var=0.0071) (Δ vs real -0.1364)
**Other theories' values on this metric (for reference):**
- pi_9: 0.4618 (var=0.0063)
- pi_8_1: 0.0976 (var=0.0019)
- pi_1: 0.8418 (var=0.0119)
- pi_2: 0.7291 (var=0.0086)
- pi_3: 0.6915 (var=0.0149)
- pi_4: 0.7200 (var=0.0069)
- pi_5: 0.1764 (var=0.0018)
- pi_6: 0.2176 (var=0.0030)
- pi_7: 0.2091 (var=0.0039)
- pi_8: 0.3964 (var=0.0050)
- pi_10: 0.6127 (var=0.0049)

### Experiment 18
**Design**
  A=[1, 1, 1, 1, 0, 0, 1, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 1, 1, 0, 0]  B=[1, 1, 1, 0, 1, 1, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
  A=[1, 0, 1, 1, 0, 0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1, 1, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 0, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 1, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 0, 0, 1, 1, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 1, 0, 0, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 1, 0, 0]  B=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 1, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 0, 0, 1, 0, 1]
  A=[1, 1, 1, 1, 0, 0, 1, 0, 0, 0]  B=[1, 1, 1, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    def _to_mat(col):
        vals = list(col)
        rows = [np.asarray(v, dtype=float).ravel() for v in vals]
        return np.array(rows, dtype=float)

    a = _to_mat(data['option_a_ratings'])
    b = _to_mat(data['option_b_ratings'])
    resp = np.asarray(data['response'], dtype=float).ravel()

    # The two 99%-validity experts occupy features 0 and 1 of the fixed
    # validity vector [0.99, 0.99, 0.90, 0.85, 0.80, 0.75, 0.70, 0.5, 0.5, 0.5].
    # Net vote of the double-99 endorsement on this trial:
    #   +2 / +1 : the 99(s) favor A,  -2 / -1 : favor B,  0 : tied/split.
    e = (a[:, 0] - b[:, 0]) + (a[:, 1] - b[:, 1])

    mask = e != 0
    if not np.any(mask):
        return 0.0

    e_m = e[mask]
    r_m = resp[mask]  # 0 = chose A, 1 = chose B

    # "Follow" = the chosen option is the one the 99% experts endorse.
    follow = np.where(e_m > 0, 1.0 - r_m, r_m)

    return float(np.mean(follow))
```

**Observed (real) value:** 0.4635 (var=0.0041)
**Candidate trajectory (this loop):**
  - iter 1: 0.2909 (var=0.0032) (Δ vs real -0.1725)
  - iter 2: 0.4781 (var=0.0047) (Δ vs real +0.0147)
  - iter 3: 0.4515 (var=0.0049) (Δ vs real -0.0120)
  - iter 4: 0.4437 (var=0.0048) (Δ vs real -0.0197)
  - iter 5: 0.4397 (var=0.0038) (Δ vs real -0.0237)
  - iter 6: 0.3725 (var=0.0056) (Δ vs real -0.0909)
  - iter 7: 0.5603 (var=0.0033) (Δ vs real +0.0968)
  - iter 8: 0.4576 (var=0.0055) (Δ vs real -0.0059)
  - iter 9: 0.3549 (var=0.0032) (Δ vs real -0.1085)
  - iter 10 (current): 0.4091 (var=0.0042) (Δ vs real -0.0544)
**Other theories' values on this metric (for reference):**
- pi_8_1: 0.4960 (var=0.0032)
- pi_9: 0.3261 (var=0.0028)
- pi_1: 0.8240 (var=0.0131)
- pi_2: 0.7483 (var=0.0039)
- pi_3: 0.6723 (var=0.0044)
- pi_4: 0.7141 (var=0.0034)
- pi_5: 0.0501 (var=0.0007)
- pi_6: 0.1819 (var=0.0018)
- pi_7: 0.3120 (var=0.0025)
- pi_8: 0.3811 (var=0.0037)
- pi_10: 0.4312 (var=0.0032)

### Experiment 19
**Design**
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0]  B=[1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1]  B=[0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0]
  A=[1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1]
  A=[0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1]  B=[1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]
  A=[1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1]
  A=[1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0]  B=[0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0]  B=[1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1]  B=[0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0]  B=[0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1]  B=[0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1]
  A=[0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0]  B=[1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0]  B=[0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1]  B=[0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0]  B=[1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    # This experiment fixes two 99%-validity experts at feature positions 0 and 1
    # (validities = [0.99, 0.99, 0.90, 0.85, 0.50 x 9]). The single structural
    # disagreement between EOBOT and SG-CGD on this panel is the SIGN of the
    # subjective weight assigned to those 99% experts:
    #   EOBOT:     w(0.99) in [-0.57, -0.11]  (deep distrust at gap 0.09, c_frac 0.692)
    #   SG-CGD:    w(0.99) in [+0.10, +0.92]  (credibility ramp promotes it back)
    # => sign-disjoint by construction.
    #
    # Metric: the probability that the subject FOLLOWS a UNANIMOUS double-99
    # endorsement. On every trial where both 99% experts cast the same sign vote
    # (both favor A or both favor B), score 1 if the subject chose the
    # 99-endorsed option, 0 otherwise; average over all such trials. Under
    # EOBOT (anti-cue) this sits well below 0.5 (~0.37); under SG-CGD
    # (promoted cue) well above 0.5 (~0.71). The common-currency cues (0.90,
    # 0.85, gated chance cues, lapse, temperature) are shared between the two
    # theories, so they shift both predictions equally and do not attenuate the
    # gap; mirrors in the design balance side orientation. 18 of 32 unique
    # trials x 3 reps = 54 observations per subject, so the per-subject value
    # is a stable proportion with small between-subject variance.

    TOP_IDX = (0, 1)  # positions of the two 0.99-validity experts

    vals = []
    for ra, rb, resp in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"],
    ):
        a = np.asarray(ra, dtype=float).ravel()
        b = np.asarray(rb, dtype=float).ravel()
        if a.shape[0] < 2 or a.shape != b.shape:
            continue

        # Sign votes of the two 99% experts (+1 A, -1 B, 0 tie).
        d = 0
        for j in TOP_IDX:
            diff = a[j] - b[j]
            if diff > 0:
                d += 1
            elif diff < 0:
                d -= 1

        # Only trials where BOTH 99% experts unanimously endorse the same option.
        if d != 2 and d != -2:
            continue

        endorsed_A = d > 0
        r = int(resp)  # 0 = chose A, 1 = chose B
        followed = (r == 0) if endorsed_A else (r == 1)
        vals.append(1.0 if followed else 0.0)

    if not vals:
        return 0.5
    return float(np.mean(vals))
```

**Observed (real) value:** 0.6133 (var=0.0039)
**Candidate trajectory (this loop):**
  - iter 1: 0.7563 (var=0.0035) (Δ vs real +0.1430)
  - iter 2: 0.7578 (var=0.0036) (Δ vs real +0.1444)
  - iter 3: 0.7519 (var=0.0037) (Δ vs real +0.1385)
  - iter 4: 0.7889 (var=0.0033) (Δ vs real +0.1756)
  - iter 5: 0.5370 (var=0.0040) (Δ vs real -0.0763)
  - iter 6: 0.7441 (var=0.0024) (Δ vs real +0.1307)
  - iter 7: 0.7433 (var=0.0027) (Δ vs real +0.1300)
  - iter 8: 0.7411 (var=0.0027) (Δ vs real +0.1278)
  - iter 9: 0.7352 (var=0.0031) (Δ vs real +0.1219)
  - iter 10 (current): 0.7400 (var=0.0039) (Δ vs real +0.1267)
**Other theories' values on this metric (for reference):**
- pi_9: 0.3578 (var=0.0047)
- pi_10: 0.6974 (var=0.0057)
- pi_1: 0.8570 (var=0.0113)
- pi_2: 0.7915 (var=0.0049)
- pi_3: 0.6856 (var=0.0110)
- pi_4: 0.7526 (var=0.0040)
- pi_5: 0.1163 (var=0.0011)
- pi_6: 0.1870 (var=0.0026)
- pi_7: 0.1022 (var=0.0018)
- pi_8: 0.2815 (var=0.0041)
- pi_8_1: 0.0574 (var=0.0012)

### Experiment 20
**Design**
  A=[1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0]
  A=[1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0]  B=[1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0]  B=[1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]  B=[0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0]  B=[0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1]  B=[0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0]  B=[1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1]  B=[1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1]
  A=[1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0]
  A=[1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0]
  A=[1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0]  B=[1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0]  B=[1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1]  B=[1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Fraction of choices that FOLLOW the 99%-validity expert's endorsement,
    # computed only on trials where that expert discriminates between A and B.
    #
    # The 0.99 expert is feature index 0 (validities = [0.99, 0.80, 0.75, 0.5 x 8],
    # displayed first). On trials 1-24 of this design the 0.99 expert's rating
    # differs between A and B; on trials 25-32 it is tied and carries no vote.
    #
    # The two theories make sharply disjoint predictions for this quantity:
    #  - SG-CGD: w(0.99) in [-14.1, -5.8] on this panel, so the far-fraud route
    #    swamps every other contribution; the 99-endorsed side loses on every
    #    discriminating trial and the follow rate sits at the lapse floor
    #    (~0.02-0.03) for ANY admissible parameter setting.
    #  - EOBOT: w(0.99) in [-2.92, -0.48]; the anti-vote is comparable in size
    #    to a few chance cues, so backing coin-flippers / plateau experts can
    #    override it. Follow rates range from ~0.09 (lone 99 vs 0.80) up to
    #    ~0.75+ (99 + 6-8 coin-flippers vs 0.80), averaging roughly 0.4-0.5
    #    across the 24 discriminating cells.
    # The gap (~0.45) is far larger than the per-subject sampling noise
    # (72 discriminating observations per subject), so the metric is tight
    # within each theory's simulated population.

    a_col = data['option_a_ratings']
    b_col = data['option_b_ratings']
    resp = data['response'].values

    follows = []
    for i in range(len(data)):
        ra = a_col.iloc[i]
        rb = b_col.iloc[i]
        # Rating of the 99% expert (feature index 0) for each option.
        a0 = int(ra[0])
        b0 = int(rb[0])
        if a0 == b0:
            # The 99% expert is tied on this trial: no endorsement to follow.
            continue
        r = int(resp[i])
        if a0 == 1 and b0 == 0:
            # A carries the 99% endorsement.
            follows.append(1.0 if r == 0 else 0.0)
        else:
            # B carries the 99% endorsement.
            follows.append(1.0 if r == 1 else 0.0)

    if len(follows) == 0:
        return 0.5
    return float(np.mean(follows))
```

**Observed (real) value:** 0.6372 (var=0.0022)
**Candidate trajectory (this loop):**
  - iter 1: 0.7978 (var=0.0017) (Δ vs real +0.1606)
  - iter 2: 0.7800 (var=0.0024) (Δ vs real +0.1428)
  - iter 3: 0.7497 (var=0.0024) (Δ vs real +0.1125)
  - iter 4: 0.7172 (var=0.0033) (Δ vs real +0.0800)
  - iter 5: 0.5911 (var=0.0043) (Δ vs real -0.0461)
  - iter 6: 0.7056 (var=0.0030) (Δ vs real +0.0683)
  - iter 7: 0.6272 (var=0.0026) (Δ vs real -0.0100)
  - iter 8: 0.7247 (var=0.0016) (Δ vs real +0.0875)
  - iter 9: 0.6881 (var=0.0023) (Δ vs real +0.0508)
  - iter 10 (current): 0.6903 (var=0.0033) (Δ vs real +0.0531)
**Other theories' values on this metric (for reference):**
- pi_10: 0.0222 (var=0.0004)
- pi_9: 0.3742 (var=0.0039)
- pi_1: 0.8383 (var=0.0102)
- pi_2: 0.8008 (var=0.0047)
- pi_3: 0.7411 (var=0.0083)
- pi_4: 0.6267 (var=0.0034)
- pi_5: 0.2058 (var=0.0017)
- pi_6: 0.2881 (var=0.0025)
- pi_7: 0.5717 (var=0.0022)
- pi_8: 0.4964 (var=0.0021)
- pi_8_1: 0.2425 (var=0.0029)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is a clear improvement over pi_9 and was rightly accepted by the gate: the inherited skeleton continues to fit the no-outlier and skeleton-driven cells well (Exps 1, 3, 4, 5, 8, 9, 10, 11, 12, 15, 16 all within ~0.03), and the CORC restoration branch fixed pi_9's catastrophic misses on Exps 19 (0.358 → 0.756 vs real 0.613) and 20 (0.374 → 0.798 vs real 0.637). However, the residual errors are not noise — they are systematic and concentrated exactly on the cells the CORC claim was built to explain, and they point at a specific miscalibration of the composition gate and the gap basis functions. The failing cells are: Exp 13 (+0.21, worst: 0.495 vs real 0.29), Exp 17 (-0.17: 0.555 vs 0.721), Exp 18 (-0.17: 0.291 vs 0.464), Exp 20 (+0.16: 0.798 vs 0.637), Exp 19 (+0.14: 0.756 vs 0.613), Exp 14 (-0.12: 0.455 vs 0.572, a regression vs pi_9's 0.550), Exp 6 (+0.10: 0.218 vs 0.118), Exp 7 (-0.08: 0.498 vs 0.580). Diagnosis of the mechanism behind these: (a) At Exp 13 (gap 0.05, c_frac 0.556), Q ≈ 0.27 already erodes the bounded discount by ~27%, S(0.05) ≈ 0.67, and the five chance cues promoted to kappa_gate*w_p ≈ 0.6 each swamp the anti-vote — the model lands at w_top ≈ +0.2 where the data (follow 0.29, with 0.94+0.75 rungs and 4-vs-1 chance opposition on the canonical trial) demand w_top ≈ -0.5 to -0.7 per 99 expert. (b) The 17/19 pair is inverted: the model produces 0.555 vs 0.756 while the data show 0.721 vs 0.613 — the composition gate splits these two gap-0.09 panels in the wrong direction (17 gets discounted, 19 gets over-restored). (c) The 19/20 overshoot is compounded by the consensus-gated chance promotion: on Exp 19's panel nine chance cues at ~0.6 units each add ~5+ units to whichever side they favor, pushing following to 0.76-0.80 where real behavior is 0.61-0.64. (d) A structural tension: with S(gap) monotone increasing and nearly flat over [0.05, 0.09] (S ≈ 0.67 vs 0.92), the shared discount depth D cannot simultaneously be deep at gap 0.05 (Exp 13) and shallow at gap 0.09 (Exps 17, 18); the empirical discount surface is non-monotone in gap (deep at 0.05, weak at 0.09-0.14, deep again at 0.24 for Exp 6), which the single increasing ramp cannot express. Note also the rationale's arithmetic on the extremity term is loose (it claims ext ≈ 0.11 but d_w ∈ [0.08, 0.14] with (v_top-v_hi)/(1-v_hi) ≈ 0.46 gives ext ≈ 0.04-0.06), so the per-cell w_top targets were never actually verified.
Rationale: Regenerate, staying strictly inside the CORC family (sign votes, tilted plateau, consensus-gated flank, Q/D/S/R/G composition-relative outlier credibility, low-gain softmax + lapse — all retained). The adjustments should be to the basis functions and gate placement, not the architecture. (1) FIX EXP 13 FIRST (worst residual, +0.21): raise c_half to ~0.65-0.70 so that c_frac ≈ 0.556 no longer erodes the discount (Q should be ≈ 0 there), and re-center/steepen the separation term so it is near-saturated at gap 0.05 (g_s ≈ 0.02, s_s ≈ 0.015); allow d_disc up to ~1.6-1.8. Target w_top ≈ -0.5 to -0.7 at (gap 0.05, c_frac 0.556). (2) RESHAPE S TO BE NON-MONOTONE IN GAP — this is the key structural fix, and it does NOT reintroduce pi_9's falsified machinery: keep NO far-fraud route and NO composition-split kernels, but decompose the discount's separation term into (i) a narrow 'near-rung suspicion' bump centered at gap ≈ 0.04-0.06 that decays by gap ≈ 0.09 (a 0.99 claim floating barely above a 0.94 rung looks like rounding noise — this is what Exp 13 demands and it is NOT falsified by the gap-0.09 following in Exps 1/17/19), plus (ii) the existing slow increasing ramp that saturates only by gap ~0.2 (this carries Exp 6's deep distrust at gap 0.24, c_frac 0.5, where the current candidate is still 0.10 too shallow). This single change resolves the D-conflict between Exps 13/6 (deep) and 17/18 (shallow) that a monotone S cannot. (3) TEMPER THE RESTORATION AND THE CHANCE PROMOTION to cure the 19/20 overshoot: lower G toward 0.3-0.45, cap_frac toward 1.0-1.05, and reduce kappa_gate (or make the promotion per-cue decay with the number of chance cues) so that a nine-cue chance flank does not contribute ~5 units; real following on the most chance-dominated panels is 0.61-0.64, not 0.76-0.80. (4) EXPLICITLY RE-DIAGNOSE PANELS 17 AND 14 before re-fitting: report their exact (gap, c_frac) and the per-cell w_top the metric implies — the 17/19 inversion (real 0.72 vs 0.61 while the model gives 0.55 vs 0.76) must be traced to whether it is a composition-gate placement error (fixable by moving c_half below 17's chance fraction) or a chance-flank context effect (fixable via kappa_gate), since 19's endorsement faces nine promoted chance cues that 17's tie-heavy trials do not. (5) Verify Exps 14 and 7 fall in line once the discount is reshaped (pi_9 fit 14 at 0.550 with a near-rung-sensitive discount; the current flat-S discount over-penalizes it). (6) Recompute the per-cell w_top targets numerically rather than by the rationale's back-of-envelope (the ext miscalculation shows the current box was set on miscomputed quantities). Minor: Exp 2 (+0.06, over-tallying) may improve with a small trim to tilt or beta, but it is low priority. If the next iteration lands even half of these corrections, the aggregate loss should drop well below 0.1031; the loop's gate preserves this accepted base if it does not.

**Outcome of your advice:** iter 1 candidate loss=0.1031 -> iter 2 candidate loss=0.0934 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The iter-2 candidate was accepted by the gate (0.1031 -> 0.0934), and my iter-1 advice was partially vindicated: the non-monotone separation basis (near-rung bump + far ramp) fixed or greatly improved exactly the cells it was aimed at — Exp 13 (+0.21 -> -0.056), Exp 17 (-0.166 -> -0.064), Exp 18 (-0.173 -> +0.015), Exp 6 (+0.100 -> +0.037) — while the shared skeleton cells (1, 2, 3, 4, 5, 8, 9, 10, 11, 16) all remain within ~0.03. However, the candidate is NOT shippable: it traded old errors for new, larger ones, and the rationale's own quantitative predictions failed on four cells, which means the proposer's per-cell w_top bookkeeping no longer tracks what the implemented model actually does. The current residual profile: Exp 14: 0.259 vs real 0.572 (Δ -0.313, a regression from iter-1's 0.455 and pi_9's 0.550); Exp 12: 0.218 vs real 0.4525 (Δ -0.235, regression from 0.431); Exp 19: 0.758 vs 0.613 (Δ +0.144, essentially unchanged from iter-1 despite the rationale claiming the kappa_gate reduction would bring it to 0.60-0.65); Exp 20: 0.780 vs 0.637 (Δ +0.143, same story — predicted 0.62, delivered 0.78); Exp 7: 0.671 vs 0.580 (Δ +0.092, overshot from -0.082 in the opposite direction); Exp 15: 0.405 vs 0.483 (Δ -0.078, regression). Diagnosis of the mechanisms behind these: (a) The 19/20 over-following persists because the chance-flank promotion is still far too strong: kappa_gate ~0.585 leaves 8-9 promoted chance cues contributing ~4-5 trusted-cue units, which swamp everything else on chance-dominated panels. The trim from ~0.65 to ~0.585 was an order of magnitude too small — the rationale itself admits 'nine cues now contribute ~4.8 units', which is still enormous. (b) The 12/13 contrast collapsed: both gap-0.05 panels now get nearly identical anti-following (0.218 vs 0.234) while real behavior differs by 0.16 (0.4525 vs 0.29). The steepened Q gate (c_half ~0.59) is correctly positioned BETWEEN c_frac 0.556 (Exp 13) and 0.625 (Exp 12), but the restoration ramp R is centered too high (g_r 0.075-0.105 gives R(0.05) ≈ 0.14-0.27), so even Q ≈ 0.83 cannot lift w_top at gap 0.05 — the composition split exists in the gate but is neutered by the ramp placement. (c) Exp 14's panel was mis-diagnosed: the rationale claimed a dense ladder at gap ~0.01 where the bump is 'silent' and predicted follow 0.57, but the simulation delivers 0.259 — deep anti-following — meaning either the gap is actually ~0.04-0.09 (so S_near or S_far fires) or the flank/ discount interaction was never computed correctly. This must be traced numerically before any further fitting. (d) Exp 7 flipped sign: the far ramp at center ~0.15 with width ~0.04 is too open at moderate gaps (if Exp 7's gap is ~0.14, S_far ≈ 0.44, giving w_top ≈ -0.5 where real behavior demands only mild distrust, anti-follow 0.58).
Rationale: Regenerate, staying strictly inside the CORC family (sign votes, tilted plateau, consensus-gated flank, single composition-relative credibility with Q/disc/restoration, softmax + lapse — all retained). My iter-1 structural advice (non-monotone separation basis) was accepted and worked; the remaining errors are calibration and verification failures, not architecture failures. Five concrete in-family adjustments on top of the accepted iter-2 base: (1) KILL THE 19/20 OVERSHOOT FOR REAL: kappa_gate ~0.585 still gives 8-9 chance cues ~4-5 units. Either drop kappa_gate to ~0.35-0.45, or (better, and still in-family) make the per-cue promotion decay with the number of chance cues (e.g., effective per-cue weight kappa_gate*w_p / sqrt(n_chance) or a capped total flank mass of ~1.5-2 units). Note the gate only opens when an outlier is present, so Exps 1-5, 8-11 are untouched by this knob; verify Exp 17 (real 0.721) stays high — its following should be carried by the double-99 at near-plateau credibility vs a single 0.90 rung, not by the flank. (2) RE-CENTER THE RESTORATION RAMP BELOW THE NEAR BUMP: lower g_r to ~0.03-0.045 (possibly with a slightly wider s_r) so that R(gap 0.05) ≈ 0.6-0.8. Combined with the existing steep Q at c_half ~0.58-0.60, this restores the 12/13 split: Exp 12 (c_frac 0.625, Q ≈ 0.8+) gets w_top near the cap (follow ~0.45-0.5) while Exp 13 (c_frac 0.556, Q ≈ 0.1-0.2) keeps the deep near-rung discount (follow ~0.29). Keep g_rest ~0.5-0.6 and the cap_frac ~0.98 so 19/20 don't re-inflate. (3) NUMERICALLY RE-DIAGNOSE EXP 14 BEFORE TOUCHING ANYTHING ELSE: report its exact (gap, c_frac) and the w_top the metric implies. The rationale predicted 0.57 and the simulation delivered 0.259 — a 0.31 prediction failure means the panel's coordinates were wrong or an unmodeled interaction (flank promotion on the 99-opposing side) dominates. If gap is ~0.04, narrow S_near (s_near ~0.010-0.012) so it is silent by 0.04; if gap is ~0.09-0.14, shift g_far up. (4) TIGHTEN THE FAR RAMP: move g_far to ~0.18-0.20 and shrink s_far to ~0.025-0.030 so S_far(0.14) ≈ 0.1 (Exp 7 back to mild distrust, anti-follow ~0.58) while S_far(0.24) ≈ 0.9 (Exp 6 keeps its deep discount, SSR ~0.12). (5) MANDATORY VERIFICATION STEP: the iter-2 rationale mispredicted four cells by 0.14-0.31 (12, 14, 19, 20). Before proposing parameter boxes, compute the model's actual per-cell w_top and predicted metric numerically at the box midpoint AND at both box corners for every outlier experiment (6, 7, 12-20), and include those numbers in the rationale. Back-of-envelope arithmetic on this architecture has now failed twice. If these five points land, the aggregate loss should drop well below 0.0934; the gate preserves the accepted iter-2 base if they do not.

**Outcome of your advice:** iter 2 candidate loss=0.0934 -> iter 3 candidate loss=0.0697 -> the gate marked it ACCEPTED.

### Iteration 3
Verdict: regenerate
Interpretation: The iter-3 candidate is a substantial, gate-accepted improvement (0.0934 -> 0.0697) and my iter-2 advice was largely vindicated: the mass cap, the N_c mixed-scale gate, and the restoration re-centering fixed exactly the cells they targeted — Exp 14 (-0.313 -> +0.008), Exp 12 (-0.235 -> -0.002), Exp 6 (+0.037 -> +0.009), Exp 18 (-0.015 -> -0.012), and the skeleton cells (1, 4, 5, 8, 9, 10, 11, 16) remain within ~0.03. The candidate is NOT yet shippable, however: four cells still miss by 0.11-0.18 (Exp 17: 0.542 vs real 0.721; Exp 19: 0.752 vs 0.613; Exp 20: 0.750 vs 0.637; Exp 7: 0.450 vs 0.580), two more miss by ~0.06-0.07 (Exp 13: 0.217 vs 0.290; Exp 15: 0.421 vs 0.483), and two previously-clean cells regressed (Exp 3: +0.002 -> +0.044; Exp 2: +0.023 -> +0.041). Critically, the rationale's own midpoint predictions failed AGAIN on four cells (17 predicted 0.67, delivered 0.542; 19 predicted 0.63, delivered 0.752; 20 predicted 0.62, delivered 0.750; 7 predicted 0.60, delivered 0.450) — the third consecutive iteration in which back-of-envelope w_top bookkeeping missed by >0.10, so the proposer's analytic model of their own implementation is still untrustworthy. Mechanistic diagnosis of the remaining errors: (a) THE PROMOTED CHANCE FLANK IS STILL THE DOMINANT ERROR SOURCE ON HIGH-c_frac OUTLIER PANELS, BUT ITS SIGN OF EFFECT DEPENDS ON WHICH SIDE THE CHANCE CUES FALL: in Exp 19/20 the counted trials put promoted chance cues ON the 99-endorsed side (inflating follow to 0.75), while in Exps 17/13's counted cells they OPPOSE the 99s (deflating follow to 0.54/0.22). A single reduction of the total promoted mass (mass_cap ~2.9 is still roughly 2x too large; target ~1.2-1.8, or kappa_gate ~0.45-0.55) moves ALL FOUR cells in the right direction simultaneously — this is the highest-leverage knob. (b) EXP 17 IS RUNG-DRIVEN, NOT FLANK-DRIVEN: its counted trials pit the restored double-99 against the 0.90+0.85 rung pair (~1.75 trusted units), and real subjects still follow at 0.72, so 2*w_top must clearly dominate 1.75 + chance opposition — the current cap (cap_frac ~1.02, w_top ~0.92) leaves the double-99 barely at parity. The restored-outlier cap needs to rise (cap_frac toward 1.15-1.30 for high-Q panels). This is separable from the 19/20 over-following (flank-driven), so the 17-vs-19 trade-off that has blocked this inversion for three iterations can finally be broken by moving BOTH knobs: mass_cap down, cap_frac up. (c) EXP 7 OSCILLATED: my iter-2 advice (tighten the far ramp) was implemented instead as a WIDER ramp (s_far 0.055) plus the n_tied amplifier, and the cell flipped from +0.09 to -0.13 — the lone 99 at gap ~0.19 now receives only ~1.3 discount units where ~2 is needed, while the paired panels (6, 18) are correctly placed. The amplifier is too aggressive a protection for n_tied=2: make it sublinear (e.g., 1 + 0.4-0.6*(n_tied-1)) or raise d_far to ~3.5-4.0 with a compensating cap, and verify Exps 6 (n=2, gap 0.24) and 18 (n=2, gap 0.09) do not move. (d) Exps 13/15 are both slightly over-discounted at gap 0.05 (0.217 vs 0.290; 0.421 vs 0.483): a modest d_near trim (toward 1.1-1.3) or a slightly narrower s_peak helps both without disturbing 12 (which sits at -0.002). (e) Exp 3's +0.044 regression is a leak: if its panel has no above-ceiling validity the outlier branch cannot touch it, so either it does (and the widened s_far is the suspect) or a shared parameter drifted — trace it numerically.
Rationale: Regenerate, staying strictly inside the CORC family (sign votes, tilted plateau, consensus-gated flank with mass cap, Q/N_c/S_near/S_far/R credibility, softmax + lapse — all retained). Five concrete in-family adjustments on top of the accepted iter-3 base, in priority order: (1) CUT THE TOTAL PROMOTED FLANK MASS ROUGHLY IN HALF: mass_cap from ~2.9 to ~1.2-1.8 (and/or kappa_gate to ~0.45-0.55). This is the single knob that moves 19, 20, 17, and 13 all in the correct direction, because the flank's sign of contribution flips across those designs (supporting the endorsed side in 19/20, opposing the 99s in 17/13). Verify numerically that Exps 6, 14, 18 (whose fit currently depends on weak-cue weights) tolerate the reduction; if Exp 6's SSR rises above ~0.15, compensate with the far-discount depth (point 3). (2) RAISE THE RESTORED-OUTLIER CAP: cap_frac toward 1.15-1.30 so that a fully restored double-99 (2*w_top ~ 2.1-2.3) clearly dominates the 0.90+0.85 rung pair (~1.75 units) that opposes it in Exp 17's counted trials — this exceeds the arbiter's sketched ~1.1*w_p but is a bounded calibration parameter inside the prescribed restoration branch, and Exp 17's real 0.72 follow-against-rungs leaves no alternative. The 19/20 over-following must NOT be re-fixed by lowering this cap — it is flank-driven and handled by (1). (3) FIX EXP 7 WITHOUT RE-BREAKING 6/18: make the tied-claim amplification sublinear (1 + 0.4-0.6*(n_tied-1)) or raise d_far to ~3.5-4.0 with the amplifier attenuated, so the LONE far 99 gets ~1.7-2.0 discount units (anti-follow ~0.58) while the paired far panels stay at their current, correct depths. Do not simply re-narrow s_far — that oscillation has already happened once (iter2 -> iter3). (4) TRIM d_near slightly (1.1-1.3) or narrow s_peak to lift Exps 13 and 15 by their residual ~0.06-0.07, checking Exp 12 (currently -0.002) does not move. (5) MANDATORY NUMERIC VERIFICATION, FOR THE THIRD TIME: the rationale's midpoint predictions missed by 0.10-0.13 on four cells this iteration (17, 19, 20, 7) after missing by 0.14-0.31 on four cells last iteration. Before proposing the parameter box, run the actual simulate-and-metric pipeline at the box midpoint AND both corners for every outlier experiment (6, 7, 12-20) and report those numbers; also report which parameter change leaked into Exp 3 (+0.044 regression on a cell the outlier branch should not touch, if its panel has no above-ceiling validity). If the flank-mass and cap_frac adjustments land even approximately, the aggregate loss should drop well below 0.0697; the gate preserves the accepted iter-3 base if they do not.

**Outcome of your advice:** iter 3 candidate loss=0.0697 -> iter 4 candidate loss=0.0830 -> the gate marked it REJECTED.

### Iteration 4
Verdict: regenerate
Interpretation: The gate REJECTED this candidate (0.0830 vs the accepted iter-3 base at 0.0697), so regenerate is the only coherent verdict. The iter-4 edit was a faithful implementation of my iter-3 priority list, and the simulation record now gives valuable ground truth about which of my recommendations were right and wrong. What worked: the mass_cap halving fixed Exp 13 (+0.073 -> +0.012), improved Exp 15 (-0.062 -> -0.050) and Exp 20 (+0.113 -> +0.080), and left 12, 14, 18 within noise. What backfired: (a) the cap_frac raise pushed Exp 19 FURTHER from the data (0.752 -> 0.789 vs real 0.613) instead of splitting 17/19 — because the far-discount at gap 0.09 (S_far(0.09) ~ 0.12 x amp 1.5 x d_far 3.9 ~ 0.7 units) keeps w_top well below the cap, so the raised cap never binds at Exp 17 while it does inflate the already-over-followed 19; (b) my iter-3 claim that the promoted flank OPPOSES the 99s in Exp 17's counted cells was wrong — the D-family cells put the chance cues on the 99-side, so the mass cut LOWERED Exp 17 (0.542 -> 0.479 vs real 0.721); (c) the d_far raise over-discounted Exp 16 (0.584 -> 0.472 vs real 0.547) and still did not deliver Exp 7 (predicted 0.57, delivered 0.434 vs real 0.580). The 17/19 inversion is now worse than at any point in the loop (17: -0.24, 19: +0.18), and Exp 7 has oscillated around its target for four consecutive iterations without landing (0.498, 0.671, 0.450, 0.434 vs real 0.580). Critically, the 'numeric verification table' was again NOT a simulation — the rationale admits it was 'reverse-engineered from the loop's own simulation record' — and it missed by 0.13-0.21 on Exps 17, 19, and 7. That is the fourth consecutive iteration in which the proposer's analytic model of their own implementation has mispredicted decisive cells, and this time the miss was on exactly the two knobs (mass_cap sign in 17, cap_frac effect in 19) that the edit was built around. The remaining error budget is now concentrated in five cells (17, 19, 7, 20, 16); everything else sits within ~0.035.
Rationale: Regenerate on top of the UNCHANGED iter-3 base, staying strictly inside the CORC family. My iter-3 advice was partially wrong and I am correcting it explicitly rather than repeating it. (1) MANDATORY REAL VERIFICATION — FIFTH AND FINAL REQUEST: before proposing any parameter box, run the actual simulate-and-metric pipeline per experiment at the box midpoint and both corners, and report the exact panel coordinates (v_top, v_second, gap, c_frac, n_tied) computed from the true validity vectors for every outlier experiment (6, 7, 12-20). The analytic bookkeeping has now failed four times, and this iteration it failed on the very cells the edit targeted; in particular the claimed 'Exp 7 gap ~0.19' is suspect, since if S_far(0.19) ~ 0.44 with d_far 3.9 and Q ~ 0 the model should deliver anti-follow ~0.55-0.58, yet it delivered 0.434 — either 7's gap is smaller or its c_frac puts Q high enough to shut the discount off, and the fix differs completely between those cases. (2) REVERT cap_frac to [1.00, 1.10]: the simulation showed the raise inflates 19/20 without helping 17, because the far term leaks ~0.7 discount units into gap 0.09 and keeps w_top below the cap there. (3) FIX THE 17/19 INVERSION THROUGH THE GAP-0.09 DISCOUNT STRUCTURE, NOT THE FLANK OR THE CAP: the empirical w_top surface required at high c_frac is ~0 at gap 0.05 (Exp 12, follow 0.4525) rising to ~+0.9 at gap 0.09 (Exp 17, follow 0.721), while at low c_frac gap 0.09 stays mild (Exp 18, +0.1). The current S_far x amp leaks into gap 0.09 for tied pairs; steepen the gap selectivity of the far term (or add an explicit small-gap floor) so paired panels at gap 0.09 carry near-zero far discount, and let the restoration ramp R carry the 0.05 -> 0.09 rise. Exp 19's remaining over-following should then be handled by the flank dilution (nine chance cues adding noise), NOT by suppressing w_top, since 17 and 19 share (gap 0.09, double-99) coordinates and differ mainly in flank size. (4) KEEP the mass_cap cut — it fixed 13/15/20 and must not be reverted; compensate Exp 17 via point (3) only. (5) REBALANCE d_far/amp for the 7-vs-16 tension: both regressed this iteration, and they moved together, suggesting their (gap, c_frac) coordinates are closer than the rationale assumes — resolve with the coordinate report from point (1) before choosing between depth (d_far) and placement (g_far/s_far); do not re-narrow s_far blindly, that oscillation has already occurred once. (6) Exp 3 (+0.027) is within parameter-draw noise as the rationale argues; leave it. If points (1)-(3) land, the 17/19/20 residuals alone should compress enough to beat 0.0697; the gate preserves the iter-3 base if they do not.

**Outcome of your advice:** iter 4 candidate loss=0.0830 -> iter 5 candidate loss=0.0897 -> the gate marked it REJECTED.

### Iteration 5
Verdict: regenerate
Interpretation: The gate REJECTED this candidate (0.0897 vs the accepted iter-3 base at 0.0697), so regenerate is the only coherent verdict, and the base reverts to iter-3. The edit was a mixed bag that lost more than it won. What worked (and should be retained next iteration): the N_c mixed-scale gate plus re-centered near bump fixed Exp 13 (-0.073 → +0.005) and Exp 15 (-0.062 → -0.006); the Q-threshold direction moved Exp 19 (+0.139 → -0.076) and Exp 20 (+0.113 → -0.046) to the correct side; Exps 3 and 10 also improved. What broke: Exp 17 collapsed catastrophically (0.542 → 0.252 vs real 0.721, residual -0.469 — the worst single-cell miss of the entire loop), Exp 7 got worse (-0.130 → -0.172), Exp 16 regressed (+0.037 → -0.105), Exp 12 regressed (-0.003 → +0.061), Exp 5 regressed (+0.020 → +0.047). The decisive failure is that the threshold-split premise itself is falsified by the loop's own record: back-solving the delivered Exp 17 value against the iter-5 box shows Q(17) ≈ 0 in the actual simulation. Under the claimed coordinates (c_frac 0.727), ANY point in the c_half ∈ [0.715, 0.725], s_c ∈ [0.010, 0.014] box gives Q ∈ [0.54, 0.85], w_top ≈ 0.9–1.04 (cap-bound), and follow ≈ 0.6–0.7 — not 0.252. The delivered value is only consistent with a panel whose c_frac sits at or below ~0.69, most plausibly [0.99, 0.99, 0.90, 0.85, 0.5×7] (c_frac ≈ 0.636, 11 features). The same back-solving on Exp 7: the claimed (gap 0.09, c_frac 0.50, n_tied 1) coordinates give a near-disc of ~1.2 plus far ~0.4, i.e., w_top ≈ -0.7 and anti-follow ≈ 0.58 — the target — yet the simulation delivered 0.407, implying w_top ≈ +0.2, i.e., N_c ≈ 0.15 and minimal far leakage, consistent with an actual c_frac of ~0.37–0.45 (three or four chance cues of eight). The near-total insensitivity of Exp 7 across iters 3→5 (0.450, 0.434, 0.407) despite drastic surgery on the near bump and d_far confirms the discount machinery barely fires at 7's true coordinates. This is the FIFTH consecutive iteration in which the rationale's analytic predictions missed decisive cells (17 predicted ~0.62–0.66, delivered 0.252; 7 predicted ~0.57, delivered 0.407; 16 predicted ~0.54, delivered 0.443; 12 predicted 0.45, delivered 0.514), and this time the miss was on the exact mechanism the edit was built around. Two of my own prior recommendations must also be explicitly retired: the iter-3 'raise cap_frac' push backfired via iter-4, and the iter-4 'split 17/19 by c_frac threshold' suggestion — which this proposer implemented faithfully — is now falsified: 17 and 19's coordinates are too close (and 17's c_frac evidently LOWER than 19's, while 17 needs MORE w_top), so no monotone-in-c_frac restoration can split them in the required direction. The correct reading of the 17/19/18/7 quartet at gap 0.09: 7 (lone 99) needs w_top ≈ -0.6; 18 (pair, c 0.30) needs ≈ +0.1; 17/19 (pairs, c ≈ 0.64–0.69) need ≈ +0.6 to +1.0 — the required surface is strongly n_tied-dependent in the PRO-trust direction at small-moderate gap, which the current amp_tie (which amplifies DISTRUST with n_tied) has backwards for gap ≲ 0.12, while it is correct for far gaps (Exp 6, gap 0.24, n=2, needs the deep amplified discount).
Rationale: Regenerate on top of the UNCHANGED iter-3 base, staying strictly inside the CORC family. Six concrete adjustments, in priority order. (1) BAN KNIFE-EDGE THRESHOLDS: revert c_half to a gentle, low placement (~0.62–0.66 with s_c ~0.04–0.06, per the arbiter's original sketch). A step function with s_c ~0.01 placed between panels whose c_frac you cannot verify is unidentifiable, and the simulation record now proves it landed on the wrong side of Exp 17. Accept a COMPROMISE w_top ≈ 0.75–0.9 at (gap 0.09, n_tied 2, c_frac 0.64–0.69): by my enumeration of both metrics' counted cells (similar rung-opposition mix), this puts 17 near ~0.65–0.70 (residual ~-0.05) and 19 near ~0.70 (residual ~+0.09) — a large net improvement over iter-3's (-0.18, +0.14) without the catastrophic 17 failure. Do NOT try to split 17 from 19; their counted cells are structurally too similar. (2) CORRECT THE COORDINATES FROM THE RECORD, NOT FROM ASSUMPTIONS: use docstring-confirmed validities where they exist (Exps 11, 12, 13, 18, 19, 20 all state them explicitly); for the rest, back-solve from delivered values as I did above — Exp 7's c_frac is ~0.37–0.45, not 0.50, and Exp 17's is ~0.64 or a near-tie with 19's 0.692, not 0.727. Report these corrected coordinates and which are confirmed vs inferred. (3) SPLIT THE TIED-CLAIM EFFECT BY GAP — the key structural fix: at gap ≲ 0.12, tied top claimants should CORROBORATE (reduce the discount, e.g., divide the near/mid-gap discount by 1 + corr*(n_tied-1) or add a bounded corroboration term to the restoration), while the anti-trust amplifier amp_tie applies ONLY to the far term (gap ≳ 0.15, keeping Exp 6's deep paired discount and Exp 20's lone-99 moderate treatment). This is the only in-family lever that separates Exp 7 (lone 99, gap 0.09, needs w_top ≈ -0.6) from Exps 17/18/19 (tied pairs, gap 0.09, need +0.1 to +1.0). (4) RE-CENTER THE LONE-OUTLIER SUSPICION AT GAP ~0.09 AND WIDEN ITS c-WINDOW: the near bump must fire at 7's true c_frac (~0.4), so either drop the N_c gate for n_tied = 1 or re-center c_peak ~0.45 with s_peak ~0.10; keep the pair-applicable rounding-noise bump at gap ~0.05 (that is what fixed 13, and 12 at gap 0.05/n=1 must stay outside its depth — verify 12 does not regress again). Exp 16 (gap 0.14, n=1, needs w_top ≈ -0.15) then sits on the bump's decay shoulder plus a small far term — tune d_near and g_far so the total discount at (0.14, n=1) is ~1.0 unit, not the current ~1.35. (5) KEEP the validated iter-5 pieces verbatim: mass_cap [1.4, 1.9], the gap-selective far term (g_far ~0.17–0.19, s_far ~0.03–0.035), the N_c gate as applied to PAIRS at gap ~0.05 (13/15 are now essentially perfect), and cap_frac at a modest [1.05, 1.12]. (6) VERIFICATION PROTOCOL, adapted to the fact that the proposer cannot execute the pipeline: for every outlier experiment, derive the predicted metric from the DELIVERED simulation record of iters 1–5 (each iteration is a measurement at known parameter boxes — use them as the coordinate oracle, as this critique did), state the inferred (v_top, gap, c_frac, n_tied) with its evidentiary basis, and bracket the prediction between the two closest prior deliveries. Back-of-envelope arithmetic from assumed coordinates has now failed five times; the record itself is the only trustworthy instrument available. If points (1)–(4) land even approximately, the 17/7/16 residuals alone should compress enough to beat 0.0697; the gate preserves the iter-3 base if they do not.

**Outcome of your advice:** iter 5 candidate loss=0.0897 -> iter 6 candidate loss=0.0795 -> the gate marked it REJECTED.

### Iteration 6
Verdict: regenerate
Interpretation: The gate REJECTED iter-6 (0.0795 vs the iter-3 base at 0.0697), so regenerate is the only coherent verdict, and the base reverts to iter-3. The edit was a largely faithful implementation of my iter-5 advice, and the simulation record now cleanly separates what worked from what didn't. What worked and must be retained next iteration: the gentle Q at c_half 0.62-0.66 (no knife edge), the tie-corroboration split in DIRECTION (near discount divided by tie_fac, far term amplified per claimant), the asymmetric lone bump, and the kept mass_cap — these landed Exps 12 (-0.007), 13 (-0.034), 14 (-0.003), 15 (-0.009) essentially on target and improved Exp 20 (+0.113 → +0.068). What failed has two identifiable causes. CAUSE 1 — the far-term deviation: the proposer moved g_far DOWN to 0.14 (against my iter-5 point 5, which said keep 0.17-0.19), justifying it via Exp 16. At gap 0.09 this gives S_far ≈ 0.19, i.e. ~0.9 discount units leaked into every gap-0.09 tied panel: this is directly why Exp 18 regressed (0.452 → 0.3725 vs real 0.4635), why Exp 17 stayed 0.21 low (0.5067 vs 0.7212) despite the corroboration machinery, why Exp 16 ended over-discounted (0.481 vs 0.547), and it contributed to Exp 9's drift (-0.047). Restoring g_far to ~0.18 drops the gap-0.09 leak to ~0.065 (S_far), which by my arithmetic lifts E17 to w_top ≈ 0.85 (follow ~0.63-0.67), returns E18 to w_top ≈ +0.5 (follow ~0.46), and lets E16 be carried by the lone shoulder instead. CAUSE 2 — Exp 7's coordinates are provably wrong, and I can now demonstrate it: if E7 were at the claimed (gap 0.09, c_frac 0.375, n_tied 1), then at EVERY point of the iter-6 box the lone bump fires (d_lone ≥ 1.0, S_lone ≥ 0.99 since |gap - g_lone| ≤ 0.008 with width ≥ 0.048, N_lone ≥ 0.88 since |c - c_lone| ≤ 0.025 with s ≥ 0.05) plus far ≥ 0.39, giving w_top ≤ -0.33 (midpoint ≈ -0.7) and anti-follow ≥ ~0.53. The simulation delivered 0.4418 — i.e. w_top mildly POSITIVE. Back-solving the delivery, E7 behaves as a high-c_frac (~0.6-0.65) lone panel at gap 0.09, where the 0.385-centered c-window is silent and Q partially restores. This is the sixth consecutive iteration in which the rationale's predictions missed the decisive cells (E7 predicted 0.60, delivered 0.4418; E17 predicted 0.65, delivered 0.5067; E18 predicted 0.46, delivered 0.3725). The full lone-outlier cell set now forms a coherent surface that the current single c-window cannot express: E1 (0.09, c ~0.4) needs w_top ≈ +0.7; E11 (0.09, c 0.50) ≈ +0.3; E7 (0.09, c ~0.625) ≈ -0.6; E12 (0.05, c 0.375) ≈ 0; E14 (0.05, c ≤ 0.25) ≈ +0.65; E16 (0.14, c 0.556) ≈ -0.15; E20 (0.19, c 0.727) ≈ +0.45. Read vertically: the gap-0.05 rounding suspicion is a LOW-c phenomenon (E12 fires, E14 doesn't), while the gap-0.09+ separation distrust is a HIGH-c phenomenon (E7/E16 fire, E11/E1 don't) — the discount's c-dependence rotates with gap, which no single Gaussian c-window can produce. Finally, the E9 regression (-0.047) is traceable to the restoration now being MULTIPLIED by tie_fac: a 0.99-pair sitting 0.01 above a 0.98 rung is pure rounding noise and iter-3 treated it correctly; the corroboration bonus over-restores it.
Rationale: Regenerate on top of the UNCHANGED iter-3 base, staying strictly inside the CORC family (sign votes, tilted plateau, consensus-gated flank with mass cap, Q/discount/restoration credibility, softmax + lapse — all retained). Re-apply iter-6's structural additions (they are validated) with the following corrections, in priority order. (1) RESTORE THE FAR TERM'S GAP SELECTIVITY: g_far back to [0.17, 0.19], s_far [0.030, 0.038]. The deviation to 0.14 is the single identified cause of the E17/E18 failures and E16's over-discount; the proposer's justification for it rested on unverified coordinates and the delivery record refutes it. Verify E6 (gap 0.24, pair) returns to ~0.12 (it regressed to 0.142 under the leaky ramp). (2) RE-PLACE THE LONE DISCOUNT AS TWO c-REGIMES, not one: keep the existing gap-0.05 'rounding-noise' bump gated by the LOW-c window at ~0.385 (this is exactly what fits E12 at disc ≈ 0.85 and keeps E14 at ≈ 0.2), and add/shift a second lone component — the asymmetric gap-0.09 bump's right shoulder plus the far term — gated by a HIGH-c window centered ~0.58-0.63 (width ~0.07-0.10). Targets: E7 (0.09, c ~0.625) total discount ≈ 1.5 → w_top ≈ -0.6 (anti ≈ 0.58); E16 (0.14, c 0.556) ≈ 1.0-1.2 → w_top ≈ -0.15; E11 (0.09, c 0.50) must stay OUTSIDE the high-c window (discount ≈ 0.5, w_top ≈ +0.3 — its current fit is near-perfect and must not move); E1 (0.09, c ~0.4) likewise outside. This 'discount migrates to larger gaps as the panel gets more chance-laden' reading is monotone in panel unreliability within each gap band and fully consistent with the arbiter's CORC sketch (no far-fraud route, no single mid-gap distrust peak — the two bumps are c-disjoint). (3) DROP THE ×tie_fac MULTIPLIER ON THE RESTORATION (keep the ÷tie_fac on the near discount and the ×n_tied on the far term): it is what leaked into E9 (gap-0.01 pair over-restored, -0.047), and E17 does not need it once the far leak is removed — without the bonus E17 lands at w_top ≈ 0.85, inside its target band. (4) ACCEPT THE E19 COMPROMISE, as established in iter-5: with the far leak removed, E17 rises to ~0.63-0.67 (residual ~-0.06) while E19 stays ~0.72-0.75 (residual ~+0.11); no monotone-in-c mechanism can split them and attempts to do so have now failed twice (iters 4 and 5). If any budget remains, a small kappa_gate trim (within the kept mass_cap framework) trims E19's chance-cue support of the 99-side without touching E17's rung-driven cells — verify before committing. (5) COORDINATE VALIDATION PROTOCOL, now with teeth: before building any edit on a panel hypothesis, check it against at least THREE prior iterations' delivered values under their known parameter boxes (each delivery is a measurement). E7's (0.09, 0.375) hypothesis is falsified by iter-6 alone; the (0.09, c ≈ 0.625) hypothesis reproduces iter-6 — confirm it also reproduces iters 3-5 (delivered 0.450, 0.434, 0.407) before re-centering the high-c window on it, and state in the rationale which hypothesis is assumed and which deliveries it reproduces. If the hypothesis fails the multi-iteration check, enumerate the design-consistent panels instead of assuming. If points (1)-(3) land, the E17/E18/E16/E9 residuals alone should compress enough to beat 0.0697 comfortably; the gate preserves the iter-3 base if they do not.

**Outcome of your advice:** iter 6 candidate loss=0.0795 -> iter 7 candidate loss=0.0889 -> the gate marked it REJECTED.

### Iteration 7
Verdict: regenerate
Interpretation: The gate REJECTED iter-7 (0.0889 vs the iter-3 base at 0.0697), so regenerate is the only coherent verdict, and the base reverts to iter-3. The edit was a faithful implementation of my iter-6 advice, and the delivery record cleanly separates what worked from what backfired. VALIDATED (must be re-applied next iteration, since the gate discards this candidate): the restored far-term gap selectivity (g_far 0.17-0.19) fixed Exp 17 (-0.179 -> -0.062) and Exp 20 (+0.113 -> -0.010, the best E20 of the entire loop); the two-regime lone discount kept Exps 12 (-0.000), 13 (-0.005), 14 (+0.009) essentially perfect; Exp 1 landed exactly (+0.0003). FAILED, with three identifiable causes. CAUSE 1 — Exp 18 FLIPPED (-0.012 -> +0.097): removing the far leak at gap 0.09 over-corrected LOW-c paired panels upward. The required w_top surface for gap-0.09 PAIRS rises steeply in c_frac (E18, c 0.30, needs ~ +0.1; E17, c ~0.64, needs ~ +0.85), but v4 carries that slope only through the weak (1-Q)-eroded far term plus restoration, leaving the low-c end ~0.4 units too high. Iter-3's leaky far ramp was accidentally supplying E18's discount. CAUSE 2 — Exp 15 regressed (-0.062 -> -0.127): the near bump's LOW-c window (center 0.48) under-fires at c ~0.63 for pairs (W_near ~ 0.49), and the un-eroded restoration adds ~+0.17 there, leaving w_top clearly positive where the data (anti-follow 0.483) demand ~0. CAUSE 3 — Exp 7 got WORSE (-0.130 -> -0.210, now the loop's worst cell) and the (0.09, c 0.625, lone) hypothesis is falsified by v4's own delivery: under the v4 box those coordinates give disc_hi ~ 1.7, w_top ~ -0.6, anti-follow ~0.58 — the target — yet the simulation delivered 0.370, implying the hi-window barely fired, i.e. E7's true c is ~0.5 or lower. This is the seventh consecutive iteration in which the rationale's coordinate assumptions mispredicted a decisive cell (E7 predicted -0.6 to -0.7 w_top, delivered mildly positive; E15 predicted +0.22, delivered ~+0.3+). Critically, the E7 trajectory now ANTI-correlates with flank strength across iterations (iter-3's larger mass_cap delivered 0.450; v4's smaller cap delivered 0.370), which means E7's residual is substantially FLANK-driven: in E7's counted trials the promoted chance cues evidently OPPOSE the 99-endorsed side, and the mass_cap cut removed that opposition. The same mechanism explains the E9 regression (-0.002 -> -0.046): E9's V1 cell needs the chance cues to oppose the lone 99, and the weakened flank removed it. So E7 and E9 now pull the mass_cap in the OPPOSITE direction from E13/E20 — a genuine three-way tension that must be named, not ignored. Remaining residuals: E19 +0.130 (the accepted compromise, per iters 4-6 — no monotone-in-c mechanism splits it from E17 and two attempts have failed), E3 +0.063 and E5 +0.040 (small drifts, possibly shared-parameter leak or draw noise), E6 +0.033, E11 -0.025.
Rationale: Regenerate on top of the UNCHANGED iter-3 base, staying strictly inside the CORC family, but RE-APPLY v4's validated pieces (they are net wins the gate discarded along with the candidate): far-term gap selectivity at g_far 0.17-0.19 / s_far 0.030-0.038, the two-regime lone discount, the untied additive restoration, and the mass cap. Then make four targeted corrections, in priority order. (1) ADD A PAIR-APPLICABLE, (1-Q)-GATED GAP-0.09 DISCOUNT — the key structural fix for E18: mirror the lone hi-c window with a component for tied top claimants at gap ~0.08-0.14 whose c-gate is the COMPLEMENT of Q (deep at low c, eroded at high c), with a steep left wall below gap ~0.07 so E13 (gap 0.05), E9 (gap 0.01) and E12 are untouched. Targets: total pair discount ~0.8 units at (0.09, c 0.30) [E18, w_top ~ +0.1] falling to ~0.05 at (0.09, c 0.64) [E17 keeps its current near-plateau fit]. This is exactly the arbiter's original 'discount D applies only when a trusted corroboration core exists (low c_frac)' clause, which v4 implemented for LONE experts but not for pairs. (2) LIFT E15 WITHOUT MOVING E12/E13 (both currently perfect): either shift c_near up to ~0.52-0.55 or widen s_c_near so W_near(0.63) rises to ~0.6-0.7 for pairs, OR (cleaner, since E15's Q ~ 0.55 vs E13's ~ 0.29) trim g_rest to ~0.35-0.45 so the restoration discriminates between them; verify E12 (lone, c 0.375, disc ~ 0.67 currently correct) does not lose its discount. (3) TREAT E7 AND E9 JOINTLY AS FLANK-SENSITIVE, AND STOP CENTERING WINDOWS ON UNVERIFIED E7 COORDINATES: the (0.09, 0.625) hypothesis is falsified by v4's own delivery. Before touching E7's w_top, test a moderate mass_cap raise to ~2.0-2.4, which simultaneously restores E9's V1 opposition and E7's counted-trial opposition; check the cost on E13 (currently perfect at cap ~1.65, will drift ~+0.02 toward over-following) and E20 (currently perfect, will drift up) — if the net across E7+E9+E13+E20 is positive, take it; if not, keep the cap and instead deepen the LONE mid-gap discount (gap 0.14-0.19, c ~ 0.5) by ~0.5 units via d_far or the hi-window's right shoulder, verifying E6 (pair, gap 0.24) and E16 (lone, gap 0.14, c 0.556, currently -0.061) do not over-discount. (4) KEEP THE E19 COMPROMISE (+0.13 residual, accepted since iter 4) and the E17 NEAR-PLATEAU placement — do not re-open the 17/19 split; spend no budget there. (5) VERIFICATION PROTOCOL, WITH THE RECORD AS THE ONLY ORACLE: the proposer's analytic w_top bookkeeping has now missed decisive cells in seven consecutive iterations (this time E7 by ~0.9 weight units and E15 by ~0.15). For every outlier experiment, derive the implied w_top from the DELIVERED values of iters 1-7 under their known boxes (each delivery is a measurement), state confirmed vs inferred coordinates, and bracket the next prediction between the two closest prior deliveries — in particular, demonstrate which (gap, c_frac, n_tied) hypothesis for E7 reproduces BOTH iter-6's 0.4418 AND iter-7's 0.3698 before building anything on it. If points (1)-(3) land, the E18/E15/E7 residuals alone (~0.05 of squared error) should compress enough to beat 0.0697; the gate preserves the iter-3 base if they do not.

**Outcome of your advice:** iter 7 candidate loss=0.0889 -> iter 8 candidate loss=0.0818 -> the gate marked it REJECTED.

### Iteration 8
Verdict: regenerate
Interpretation: The gate REJECTED iter-8 (0.0818 vs the iter-3 base at 0.0697), so regenerate is the only coherent verdict and the base reverts to iter-3. The edit was a faithful implementation of my iter-7 advice, and the delivery record now cleanly prices each recommendation. VALIDATED (must be re-applied next iteration, since the gate discarded them along with the candidate): (a) the pair-core gap-0.09 discount with the low-c complement gate fixed exactly what it targeted — E18 is now essentially perfect (-0.006, vs -0.097 at iter-7 and -0.012 at iter-3's leaky-ramp accident); (b) the N_c-carried, non-Q-eroded near bump landed E13 at +0.006 (best of the loop) without moving E12 (+0.006) or E14 (+0.012); (c) E17 improved to -0.102 (from iter-3's -0.179); (d) E2, E5, E8 all improved. FAILED, with identifiable causes: (1) THE MASS_CAP RAISE WAS NET-NEGATIVE — my iter-7 point-3 tested branch (cap 1.65→2.2) came back with a clear answer: E7 regressed further (0.370→0.352, now the loop's worst cell at -0.227), E20 regressed badly (0.627→0.725, destroying iter-7's best-in-loop E20 fit), E6 drifted, while only E9 gained (+0.024). I explicitly retire that recommendation. The direction of the E7 move also falsifies my iter-7 flank hypothesis: raising the promoted-cue mass made E7 MORE pro-99, so in E7's counted trials the promoted chance cues net SUPPORT the 99-endorsed side (the 99-allied chance cues in the 4/7/8/16/17 trial families dominate the opposed ones in 3/15), and the residual is not recoverable via the flank. (2) E16 REGRESSED (+0.037→+0.069): the d_far trim (2.7→2.0) plus the cap raise under-discounts the lone 99 at gap 0.14 by ~0.3-0.5 units. (3) E3 HAS DRIFTED MONOTONICALLY UPWARD ACROSS ALL EIGHT ITERATIONS (0.590→0.694 vs real 0.617, now +0.077) — since E3 is structurally untouched by the outlier branch, this is a shared-skeleton creep (beta/epsilon/tilt/kappa_lo) that has never been re-centered; it is now a top-3 residual. (4) E15 regressed slightly (-0.062→-0.078) and the rationale's prediction missed again (predicted 0.48-0.52, delivered 0.405) — the eighth consecutive iteration in which the proposer's bookkeeping mispredicted decisive cells (this time E7 by ~0.08, E20 by ~0.04, E15 by ~0.09). (5) E19 (+0.128) remains the accepted compromise, per iters 4-7. The remaining error budget is concentrated in E7 (-0.227), E19 (+0.128), E17 (-0.102), E20 (+0.088 after the cap damage), E3 (+0.077), E15 (-0.078), E16 (+0.069); everything else sits within ~0.035.
Rationale: Regenerate on top of the UNCHANGED iter-3 base, staying strictly inside the CORC family (sign votes, tilted plateau, consensus-gated flank with mass cap, Q/N_c/near/far/pair discount + restoration credibility, softmax + lapse — all retained). Re-apply iter-8's validated pieces and make five targeted corrections, in priority order. (1) REVERT mass_cap TO [1.6, 1.9] (kappa_gate ~0.60-0.66): the iter-8 test of my iter-7 branch is now answered and it is net-negative (E7 -0.083, E20 -0.078, vs E9 +0.024). This single reversion should restore E20 to ~0.63 (iter-7 level, residual ~-0.01), E7 to ~0.45 (residual ~-0.13), and E6 to ~0.13, while the pair-core and N_c pieces — which are NOT cap-dependent — keep E13/E15-machinery/E18/E17 at their iter-8 placements. Verify E13 (which stayed perfect at cap 2.2) does not drift when the cap drops; iter-7's delivery at cap 1.65 says it will not. (2) RESTORE LONE-PATH FAR DEPTH FOR E16: raise the far discount for n_tied=1 back to ~2.6-3.2 (iter-8 trimmed it to 1.8-2.2, which is the identified cause of E16's over-following), while ATTENUATING the tied-claim amplifier to sublinear (1 + ~0.5*(n_tied-1)) so paired panels at gap 0.24 (E6, currently slightly under-discounted at 0.1405 vs 0.118) do not over-discount. Check: E20 (lone, gap 0.19, c 0.727) is protected because Q ≈ 0.95 erodes the far term there; E11/E1 (lone, gap 0.09, low c) are protected because S_far(0.09) ≈ 0.06; E18/E17 (pairs, gap 0.09) are protected by the pair/lone separation and the G_pair gate. (3) RE-CENTER THE SHARED SKELETON ON E3: E3 has drifted +0.077 over eight iterations with no structural cause — trim beta toward [0.34, 0.40] (or the equivalent epsilon/tilt adjustment), verifying numerically that E4 stays ~0.513, E1 stays ~-0.22, and E10 does not regress past +0.03; this is ~0.006 of squared error available for free. (4) DO NOT BUILD ANY NEW E7 MACHINERY: six attempts at E7 w_top surgery have failed, and the iter-8 record shows the cell is flank-sensitive in the WRONG direction for the cap fix. After the cap revert, accept E7 at ~-0.13 as this theory's largest fixed cost (as the proposer themselves concluded in iter-8 — that judgment was correct and should stand). The only E7-compatible lever left is the lone mid-gap deepening in point (2), which helps E7 only if its true gap is 0.14-0.19; do not center any window on unverified E7 coordinates. (5) KEEP the E19 compromise (+0.13) and the E17 near-plateau placement — do not re-open the 17/19 split; two attempts have failed and E17's remaining -0.10 is partially locked by E19's shared coordinates. On E15 (-0.078): the delivered 0.405 contradicts the box arithmetic (the near bump plus N_c(0.63) ≈ 0.89 should give w_top ≈ -0.2 and anti ≈ 0.55), so before touching parameters, back-solve which (gap, c_frac) E15 actually occupies from the iter-5/6 deliveries (0.477/0.474, the best fits) under their known boxes — the current coordinates are evidently wrong, and a coordinate error here has poisoned four consecutive iterations. (6) VERIFICATION: continue using the delivery record as the only oracle, but bracket the CAP-DEPENDENT cells (E7, E9, E13, E20, E6) explicitly at both cap values before committing — the iter-8 table missed E7 and E20 precisely because the cap interaction was never bracketed. If points (1)-(3) land, the residual profile should be roughly E7 -0.13, E19 +0.13, E17 -0.10, E15 -0.06, everything else ≤0.04 — comfortably below 0.0697; the gate preserves the iter-3 base if they do not.

**Outcome of your advice:** iter 8 candidate loss=0.0818 -> iter 9 candidate loss=0.0867 -> the gate marked it REJECTED.

### Iteration 9 (most recent)
Verdict: regenerate
Interpretation: The gate REJECTED iter-9 (0.0867 vs the iter-3 base at 0.0697), so regenerate is the only coherent verdict and the base reverts to iter-3. The edit was a faithful implementation of my iter-8 advice, and the delivery record now prices each of my five recommendations. WHAT WORKED (validated, must be retained): (a) the mass_cap revert to [1.6,1.9] — E20 improved to +0.051 (from +0.088) and E13 stayed fine (-0.021); (b) the lone mid-gap window (disc_hi) fixed E16 exactly as designed (+0.069 → -0.039); (c) c_half at 0.58-0.62 delivered the best E15 of the entire loop (+0.002, previously never better than -0.006); (d) the beta trim helped more cells than it hurt in squared-error terms (E3 +0.077→+0.039, E10 →0.000, E1 →-0.012, E7 →-0.178) though it cost E4/E5/E9 — roughly a wash, keep it. WHAT FAILED, with two identified causes and one unexplained: (1) THE d_far RAISE LEAKED INTO GAP-0.09 PAIRS AND RE-BROKE E18: iter-8's pair-core fix had landed E18 at -0.006; the far-depth raise (2.0→~3.05) adds ~0.1-0.15 discount units per 99 expert at gap 0.09 (S_far(0.09)≈0.07 × d_far 3.05 × amp 1.5), and with two tied experts the endorsed side lost ~0.3 units — E18 fell to -0.109. The sublinear amplifier did not prevent this because the leak is through d_far itself, not the amplifier. (2) E17 REGRESSED (-0.102 → -0.222) and my own box arithmetic CANNOT explain it: relative to iter-8, the c_half move (0.64→0.60) should RAISE E17's w_top (less pair-core discount via higher Q, more restoration), the far change is tiny for pairs, and the cap cut is the only candidate left — yet the iter-7→iter-8 transition shows the cap moving E17 in the OPPOSITE direction (cap 1.65→2.2, E17 0.659→0.619). E17's driver under this architecture is now genuinely unidentified after seven attempts, and it is the loop's largest single error (0.049 of squared error). (3) E12 REGRESSED -0.058 (0.458→0.401) with no stated cause: the lone gap-0.05 path parameters (d_lone, c_lo, g_near, s_near) are unchanged from iter-8, Q≈0 there either way, disc_hi is silent at its coordinates, and the beta trim should compress E12 toward 0.5 (i.e., UP, not down) — this regression is unexplained and must be traced before anything else is touched. Net: iter-9 traded a fixed E18 and a stable E12 for an already-fixed E16 and a perfect E15 — a net loss. The remaining error budget: E17 (-0.222), E7 (-0.178), E19 (+0.122), E18 (-0.109), E20 (+0.051), E12 (-0.052), E9 (-0.056), E3 (+0.039), E5 (+0.042), E6 (+0.047); everything else within ~0.03.
Rationale: Regenerate on top of the UNCHANGED iter-3 base, staying strictly inside the CORC family, re-applying iter-9's validated pieces (mass_cap [1.6,1.9], disc_hi window, c_half [0.58,0.62], beta [0.34,0.40], N_c near bump, pair-core discount, sublinear amplifier, additive restoration, cap_frac [1.05,1.12]) with three targeted corrections, in priority order. (1) SPLIT THE FAR DEPTH INTO LONE AND PAIR CHANNELS — the key structural fix: the architecture already separates lone/pair paths everywhere else (disc_near has lone/pair c-gates, disc_pair is pair-only, disc_hi is lone-only), so give the far term two depths: d_far_lone [2.8,3.3] (keeps E7 at ~0.40, E16 at ~0.51, E1/E11 protected by S_far(0.09)≈0.07, E20 protected by Q≈0.93) and d_far_pair [1.9,2.3] (returns E18 to iter-8's ~0.46 and E6 to ~0.14; E17's pair far-term is small either way). This resolves the E7-vs-E18 tension that the single d_far knob has been flipping back and forth: deep helps E7/E16, shallow helps E18/E17, and no single value is right. (2) TRACE AND REVERT THE E12 REGRESSION BEFORE ANY OTHER EDIT: iter-8 delivered 0.4583 and iter-9 delivered 0.4008 with nominally identical lone gap-0.05 machinery — diff the two boxes parameter by parameter along the E12 path (d_lone, c_lo, s_lo, g_near, s_near, and any interaction with beta/g_r), identify which drift moved it, and restore it to ~0.45 without disturbing E14 (currently fine at -0.019) or E13. An unexplained -0.058 move on a previously-perfect cell means some parameter interaction is unmodeled in the proposer's bookkeeping — the ninth consecutive iteration in which analytic predictions missed decisive cells. (3) BRACKET E17 AT THE CORNERS OF (mass_cap, c_half) AND ACCEPT THE BEST CORNER — build NO new E17 machinery: with corrections (1)-(2) in place, run the config at cap ∈ {1.6, 1.9, 2.2} × c_half ∈ {0.58, 0.62} and report E17, E20, E13, E15, E7, E6 at each corner. The iter-7/8/9 record shows E17 moving 0.659→0.619→0.499 under cap 1.65→2.2→1.75 with confounded co-changes; a clean corner sweep is the only way to isolate whether the flank or the pair-core gate is the driver. If the best corner gives E17 ≥ 0.60 (residual ≤ -0.12) without breaking E20/E13/E15, take it; if E17 stays near 0.50 everywhere, accept it as this theory's third fixed cost alongside E7 (-0.18) and E19 (+0.12) — the arithmetic in this critique shows the rest of the profile can still beat 0.0697 with E17 at -0.15 (estimated sum-of-squares ≈ 0.089 → loss ≈ 0.061) but not with E17 at -0.22 AND E18/E12 unfixed (≈ 0.075 loss). (4) DO NOT re-open the 17/19 split (two failures, iters 4-5), do NOT build E7 machinery (six failures), do NOT touch the E13/E15/E16/E20 placements — they are the best of the loop. (5) Keep beta at the trimmed [0.34,0.40]: the corner analysis above shows it was net-neutral-to-positive (E3/E10/E1/E7 gains outweigh E4/E5/E9 losses in squared error), and it is partially responsible for E7's improvement — but verify E5 does not drift past +0.05; if it does, split the difference to [0.36,0.42] rather than reverting. If points (1)-(3) land even at the conservative E17 corner, the aggregate loss should come in at ~0.055-0.065, below the 0.0697 floor; the gate preserves the iter-3 base if they do not.

**Outcome of your advice:** iter 9 candidate loss=0.0867 -> CURRENT candidate loss=0.0818 -> the gate marked it REJECTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The gate REJECTED iter-10 (0.0818 vs the iter-3 base at 0.0697), so regenerate is the only coherent verdict and the base reverts to iter-3. The edit was a faithful implementation of my iter-9 advice on points 1-3, and the delivery record prices it as a near-miss that lost on one self-inflicted wound. WHAT WORKED (validated, must be re-applied on top of the iter-3 base next iteration): (a) the channel-split far depths did what I predicted \u2014 E17 improved (-0.222 -> -0.136), E18 recovered most of the way (-0.109 -> -0.054), E6's leak concern did not materialize catastrophically; (b) the lone two-window structure delivered the best E1 of the entire loop (+0.003) and the best E5 (+0.001); (c) E2, E8, E10, E15, E20 all sit at or near their best-of-loop placements. WHAT BROKE, with one dominant cause: (1) EXP 13 COLLAPSED (+0.168, now tied with E7 as the loop's worst cell): 0.458 vs real 0.290, a +0.19 swing from iter-9's 0.269. The pair gap-0.05 near-bump was re-parameterized to d_near [0.7, 0.9] with c_peak [0.52, 0.58] \u2014 roughly half the effective discount depth that delivered E13 \u2248 0.27-0.30 at iters 8/9 \u2014 and the rationale's expected-residual table did not even list E13 as a risk. This directly violated my iter-9 point 4 ('do NOT touch the E13/E15/E16/E20 placements \u2014 they are the best of the loop'). Back-solving the delivery: E13 (pair, gap 0.05, c_frac 0.556) received net discount \u2248 0.5-0.6 units where the data (follow 0.29 against a 0.94+0.75 rung pair) demand \u2248 1.3-1.5. (2) E9 regressed further (-0.056 -> -0.068): the pair near-bump's shallow symmetric Gaussian still leaks ~0.15 x d_near at gap 0.01 in the wrong direction \u2014 no wait, the leak is pro-discount; the regression is from the cumulative beta trim plus the weakened pair-core discount at gap 0.01; either way the cell has drifted monotonically from -0.002 (iter-3) across six iterations of otherwise-good edits. (3) E6 regressed (+0.009 -> +0.042): d_far_pair [2.2, 2.6] is shallower than the iter-3 single depth at gap 0.24. (4) E12 remains slightly low (-0.035 vs iter-8's -0.006). Net arithmetic: iter-10's gains over iter-3 (E1/E2/E3/E5/E8/E10/E15/E17/E20, \u2248 0.031 squared units) were outweighed by E13 (+0.023), E7 (+0.020), E9 (+0.005), E18/E6/E12 (+0.006). This is the TENTH consecutive iteration in which the proposer's bracketing missed a decisive cell \u2014 this time by omitting from the bracket table the one cell (E13) that shared the moved machinery.",
  "verdict": "regenerate",
  "rationale": "Regenerate on top of the UNCHANGED iter-3 base, staying strictly inside the CORC family, re-applying iter-10's validated pieces (channel-split far depths d_far_lone [2.8,3.3] / d_far_pair [2.2,2.6], the two-window lone discount, beta [0.34,0.40], mass_cap [1.6,1.9], c_half [0.57,0.60], additive restoration, cap_frac ~1.05), with corrections in priority order. (1) RESTORE THE PAIR GAP-0.05 NEAR-BUMP DEPTH \u2014 the single decisive fix: raise d_near from [0.7,0.9] to [1.3,1.6], keep c_peak [0.52,0.56] and s_peak [0.12,0.16]. Arithmetic at box midpoint: E13 (gap 0.05, c 0.556, N_c \u2248 1.0, Q \u2248 0.33 eroding only the small restoration) gets net discount \u2248 1.45 - 0.08 \u2248 1.35 \u2192 w_top \u2248 -0.5 \u2192 follow \u2248 0.30 (target 0.29); E15 (gap 0.05, c ~0.63, N_c \u2248 0.81, Q \u2248 0.68, restoration \u2248 0.16) gets net \u2248 1.0 \u2192 w_top \u2248 0 \u2192 anti-follow \u2248 0.48 (target 0.483); E9 (gap 0.01, S_near \u2248 0.15, N_c \u2248 0.7) gets \u2248 0.15 extra discount, pushing its anti-99 rate UP toward the real 0.56 \u2014 i.e., this ONE knob moves E13, E15, and E9 all in the correct direction simultaneously. Verify E12 (lone channel, different path) and E14 (c \u2264 0.25, outside N_c) do not move. (2) MODEST PAIR-FAR DEEPENING FOR E6: raise d_far_pair to [2.5,2.9] or amp_slope to [0.65,0.85]; the cost at E18 is bounded (S_far(0.09) \u2248 0.07, so +0.3 depth adds only ~0.035 units at gap 0.09) \u2014 bracket E18 at both d_far_pair values before committing and take the corner that keeps E18 within -0.07. (3) E12 TRIM: d_lo from [0.9,1.1] to [0.8,1.0] (or shift c_lo up 0.02) to lift the lone gap-0.05 cell ~0.03; E14 is protected by the c-window and must be bracket-checked. (4) ACCEPT THE FIXED COSTS \u2014 do NOT build E7 machinery (seven failures; it drifted marginally worse to -0.193, leave the lone mid window exactly where it is), do NOT re-open the 17/19 split (E17 at -0.136 and E19 at +0.127 are the established compromise; at most take c_half at the very low end 0.57 and mass_cap at 1.6, the corner the iter-10 record already partially validates). (5) VERIFICATION PROTOCOL WITH A NEW RULE: every cell whose computed path touches a moved parameter must appear in the bracket table \u2014 not just the targeted cell. Iter-10 bracketed E12/E17/E18/E6/E7 and omitted E13, which shared the N_c machinery with the very bump that was re-parameterized; that omission cost the loop its best candidate. Bracket each prediction between the two closest prior DELIVERIES under known boxes as before, and explicitly enumerate which experiments' w_top paths each parameter change enters. Feasibility check: fixing E13 alone removes ~0.027 of squared error from iter-10's profile (~0.118 \u2192 ~0.091), and the E9/E6/E12 trims bring it to ~0.086 \u2014 below iter-3's ~0.098 equivalent, i.e., the next candidate should beat 0.0697 if points (1)-(2) land even approximately; the gate preserves the iter-3 base if they do not."
}
```

## Usage

```json
{
  "input_tokens": 78551,
  "output_tokens": 11550,
  "reasoning_tokens": 9786,
  "provider": "Sail Research"
}
```
