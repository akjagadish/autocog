# feedback_iter_07

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
- THEORY 1 = `pi_6`
- THEORY 2 = `pi_4`
- The recommendation below acts on THEORY 1 (= `pi_6`).

Replace pi_6 with 'Consensus-Relative Edge-Distrust Tallying' (CREDT). Keep the shared, well-supported architecture (sign votes × validity weights × low-gain softmax × lapse) — that scaffold is endorsed by every experiment. Replace the weight function with three components: (1) MID-BAND PLATEAU WITHOUT A CLIFF: all cues with stated validity in [v_lo, v_hi] (v_lo ≈ 0.55, v_hi ≈ 0.96) receive approximately equal positive weight w_p ≈ 1 — pure tally behavior over diagnostic cues (recovers Exps 2-6, 10). There is NO absolute 0.98/0.99 discontinuity; adjacent high validities are near-exchangeable. (2) CHANCE-FLANK: a small negative weight for v near 0.5, w(0.5) = -kappa_lo with kappa_lo in [0.15, 0.25] — the one BSST component the data confirm (Exp 7: observed -0.168). (3) THE NOVEL PIECE — OUTLIER-GAP DISTRUST: the subjective weight of the top expert(s) is modulated by the gap between the highest stated validity and the next-highest validity in the panel (computable from parameters['validities']). Small gap (dense ladder, e.g., 0.99 atop 0.98/0.95/0.90 as in Exp 2) → the top cue keeps ~full plateau weight (predicts follow-99 ≈ 0.58-0.62, matching observed 0.599). Large gap (isolated outlier, e.g., 0.99 atop 0.75 as in Exps 8/9) → its weight collapses toward zero or slightly negative via w_top = w_p - rho * sigmoid((gap - g0)/s), with g0 ≈ 0.05-0.10, rho allowing the asymptote to reach [-0.6, +0.1] (predicts Exp 8 strong-side rate ≈ 0.08-0.20 near the lapse floor, and Exp 9 anti-99 ≈ 0.52-0.60, matching observed 0.580). Suggested parameters: w_p ∈ [0.8, 1.2], v_lo ∈ [0.52, 0.58], v_hi ∈ [0.95, 0.97], kappa_lo ∈ [0.15, 0.25], g0 ∈ [0.04, 0.12], s ∈ [0.02, 0.06], rho ∈ [0.9, 1.7], beta ∈ [0.30, 0.50], epsilon ∈ [0.03, 0.08]. This theory differs from BSST by making distrust of certainty RELATIVE to the panel consensus rather than an absolute cliff, and differs from FDEI by (i) keeping chance cues anti-diagnostic rather than annihilated and (ii) capping the influence of isolated super-experts rather than giving them the largest positive weight. It should simultaneously recover the tally core (Exps 2-6), the chance-flank negativity (Exp 7), the isolated-outlier distrust (Exps 1, 8, 9), and 98/90 exchangeability (Exp 10), making it a genuine task-invariant competitor to FDEI.


## CANDIDATE THEORY
Consensus-Relative Edge-Distrust Tallying (CREDT), corroborated-precision edition. People integrate ALL binary expert ratings as sign votes (+1/-1/0) into a zero-sum evidence score. The subjective weight of an expert is set by: (1) a trust plateau over stated validity [v_lo, v_hi] with the ceiling just above 0.98; (2) a small negative chance flank (w(0.5) = -kappa_lo); (3) consensus gating of that flank by the panel's distrust signal; and (4) the novel claim: distrust of the panel-max expert is distrust of UNCOPROBORATED PRECISION and is NON-MONOTONE in the edge (gap) the top claim holds over the next rung. A dense ladder (gap ~ 0.01) corroborates near-certainty (the panel itself approaches the claim) -> trust. A small uncorroborated edge (gap ~ 0.05, e.g. 0.99 atop 0.95) reads as spurious precision -- the same underlying competence with inflated confidence -> maximal distrust. A large edge (gap >= ~0.09) on a coherent, competent panel reads as a categorically better instrument and is taken at face value -> trust returns. But a far outlier on an incoherent, chance-laden scale (a 0.99 floating above a panel stuffed with coin-flip experts) is unearned certainty -> collapse again, via a composition-amplified route that fires only on chance-laden panels. Choice is a low-gain softmax plus a uniform lapse.

`predict(parameters, state, history) -> np.ndarray`:
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

`policy(probs) -> int`:
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return int(np.random.choice(len(probabilities), p=probabilities))

`parameters`:
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

`rationale`:
MINIMAL-DIFF EDIT of the ACCEPTED iter-5 CREDT base. Exactly ONE code block changes (the w_top discount, block (iv)); the scaffold (sign votes x plateau x chance flank x consensus gate x extremity term x low-gain softmax x lapse), the gate's monotone sig_gap (same g0/s_gap), the flank, beta, kappa_lo, w_p, v_hi, lambda_ext, d_w are byte-identical. Six new parameters describe the new discount shape; rho's range is re-derived for the new surface it scales.

(1) MEASUREMENT FIRST, AS DEMANDED. I abandon panel guessing and use the accumulated iters 4-7 as ENDPOINT PROBES: each past iteration is a known-parameter configuration whose DELIVERED simulation values (not hand-sims) pin each experiment's operating point. Exp 7 delivered anti-99 = 0.5907 / 0.5276 / 0.4427 at w_top = -0.70 / -0.30 / +0.32 (iter-4, iter-5, iter-6 configs, all invertible exactly since the model is deterministic given parameters): a linear fit gives anti99 = 0.49 - 0.145*w_top, so the OBSERVED 0.5796 requires w_top = -0.62; inverting the two independent iter-4 and iter-5 w_top values through their known sigmoids BOTH give gap(Exp7) = 0.050 (two consistent measurements, not a circular one). Exp 3 delivered 0.5270 / 0.6119 at w_top = -0.85 / ~0 (iter-5 / iter-6-m-damped): slope ~0.10/unit, so the OBSERVED 0.617 requires w_top ~ +0.05; the iter-4 -> iter-5 insensitivity (metric moved only +0.008 while any gap in [0.035, 0.065] would have moved w_top by >= 0.4) plus the iter-6 m-damp (which only restores the top cue if sig_gap ~ 1) pin gap(Exp3) >= 0.075, ~0.09-0.10. Measured operating points: gap 0.01 -> w_top ~ +0.6 (Exps 9/10, currently fitting at +0.011/-0.016); gap 0.050 -> w_top = -0.62 (Exp 7); gap ~0.09 -> w_top ~ 0 (Exp 3); gap 0.24 with c_frac 0.5 -> w_top ~ -0.85 (Exp 6, currently fitting at +0.005).

(2) WHY NON-MONOTONE. Both monotone separation axes are now empirically falsified: the gap axis is ANTI-MONOTONE (trust needed at Exp 3's LARGER gap, distrust at Exp 7's smaller one) and the weak-fraction moderator failed in iters 6 and 7 (double rejection). This is exactly the fallback the iter-7 critic sanctioned: a peaked gap response, maximal at intermediate gap, receding at dense and at large gaps, with Exp 6's collapse preserved by a composition route that fires only on chance-laden panels. The theoretical reading: distrust of certainty is distrust of UNCORROBORATED PRECISION -- a 0.99 claim atop 0.98 is a consistent extrapolation of a competent panel (trust); atop 0.95 it is the same competence with inflated confidence (spurious precision -> distrust); atop 0.90 on a coherent panel it is a categorically better instrument taken at face value (trust returns); atop a chance-laden scale it is unearned certainty (collapse).

(3) THE EDIT AND ITS OPERATING POINTS (parameter midpoints: w_p=0.90, d_w*extremity(0.99)=0.18, rho=1.36, g_star=0.048, s_l=0.012, s_r=0.033, c_half=0.36, s_c=0.06, c_amp=1.29). Exp 7 (gap 0.050, c_frac~0): P=1.0, discount=1.36 -> w_top=-0.64 -> anti-99 = 0.49+0.145*0.64 = 0.58 vs observed 0.580. Exp 3 (gap 0.09, c_frac<=0.167): P=0.39, C=0.05 -> discount=0.60 -> w_top=+0.11 -> tally-following ~0.62 vs observed 0.617. Exps 9/10 (gap 0.01): P=0.007, comp = C*0.072 <= 0.05 -> w_top = +0.63..+0.68 (was +0.60) -> Exp 9 anti-99 0.571 -> ~0.56 (obs 0.560), Exp 10 follow-99 0.583 -> ~0.59 (obs 0.599): BOTH improve. Exp 6 (gap 0.24, c_frac 0.5): P~0, C=1.16, sig_gap=1 -> discount=1.58 -> w_top=-0.86 (was -0.88); gate = 1.0 unchanged -> SSR ~0.12 (obs 0.118). Exps 2/5/8: v_top = 0.92/0.95/0.98 < v_hi -> the edited branch provably never fires; bit-identical behavior. Robustness: if gap(Exp3)=0.08 the bump gives w_top=-0.19 (Delta -0.02); if 0.11, w_top=+0.42 (Delta +0.04) -- strictly better than the current -0.090 everywhere in the measured range; if Exp 7 harbors two sub-0.53 cues, w_top=-0.81 (Delta +0.02).

(4) WHY rho's RANGE CHANGES (not an oscillation). rho now scales a DIFFERENT response surface (bump + composition, peak value 1.0) rather than a saturated sigmoid; the measured endpoint w_top(peak) = -0.62 with d_w*ext = 0.18 forces the peak discount to ~0.80, i.e. rho ~ 1.36. Keeping rho = 1.6 on the new surface would overshoot Exp 7 to w_top = -0.88 (Delta +0.04). Every other gate-validated knob is untouched, per the critic's instruction.

(5) EXPS 1 AND 4 (unknown panels) -- scenario analysis instead of guessing. For each, I asked which panel scenarios reproduce their CURRENT DELIVERED fits under iter-5, then evaluated the edit on exactly those scenarios. Exp 4: (a) top <= 0.98 -> edit inert; (b) 0.99-topped with gap 0.01 and a chance-laden low block -> w_top moves +0.60 -> +0.58, essentially unchanged; (c) gap ~0.09 with a chance-laden block -> w_top -0.85 -> -1.33, which pushes the |d|=1 agreement from 0.534 TOWARD the observed 0.513. The one scenario the edit would hurt (gap ~0.09, all-diagnostic panel) does NOT reproduce the delivered 0.534 under iter-5 (it yields ~0.57-0.65), so it is unlikely to be the actual panel. Exp 1: the scenarios that reproduce the delivered -0.246 (top <= 0.98, or 0.99 atop 0.98 with w_top ~ +0.6) are inert or slightly improved by the edit; the scenarios the edit would hurt (gap 0.04-0.05) yield ~-0.5 under iter-5, contradicting the delivered value.

(6) FORECAST (closing the credibility gap with measured slopes, not hand-sims). Exp 3: 0.61-0.63 (Delta ~ 0, from -0.090). Exp 7: 0.57-0.60 (Delta ~ 0, from -0.052). Exp 9: ~0.555-0.567 (Delta ~ 0). Exp 10: ~0.59-0.60 (Delta ~ -0.005). Exp 6: ~0.12 (unchanged). Exps 1/2/4/5/8: unchanged up to run noise (+-0.03). Projected residual sum-of-squares falls from ~0.0144 to ~0.004-0.006 including noise, i.e. aggregate loss ~0.030-0.042, comfortably below the 0.0572 accept floor. RISKS: if gap(Exp3) sits at the very bottom of the measured range (0.075), its residual halves rather than vanishes (Delta ~ -0.04) -- still a strict net gain; the composition threshold c_half = 0.36 is set below Exp 6's c_frac = 0.5 with margin and above Exp 9's 0.27-0.36, where the comp route is additionally killed by sig_gap(0.01) = 0.07, so no protected experiment's operating point moves by more than 0.09 w_top units.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.7165 -> ACCEPTED
- iter 2: loss=0.1676 -> ACCEPTED
- iter 3: loss=0.0827 -> ACCEPTED
- iter 4: loss=0.0573 -> ACCEPTED
- iter 5: loss=0.0572 -> ACCEPTED
- iter 6: loss=0.0602 -> REJECTED
- iter 7: loss=0.0697 -> REJECTED
- iter 8 (current candidate you are grading): loss=0.0392 -> ACCEPTED
Running-best (last accepted) base: iter 8 at loss=0.0392.

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
  - iter 1: -0.2672 (var=0.0083) (Δ vs real -0.0492)
  - iter 2: -0.2757 (var=0.0056) (Δ vs real -0.0577)
  - iter 3: -0.2607 (var=0.0049) (Δ vs real -0.0427)
  - iter 4: -0.2532 (var=0.0064) (Δ vs real -0.0352)
  - iter 5: -0.2462 (var=0.0059) (Δ vs real -0.0282)
  - iter 6: -0.2785 (var=0.0066) (Δ vs real -0.0605)
  - iter 7: -0.2565 (var=0.0058) (Δ vs real -0.0385)
  - iter 8 (current): -0.2555 (var=0.0047) (Δ vs real -0.0375)
**Other theories' values on this metric (for reference):**
- pi_1: 0.6130 (var=0.0149)
- pi_2: -0.4425 (var=0.0124)
- pi_3: -0.2182 (var=0.0156)
- pi_4: -0.2188 (var=0.0064)
- pi_5: -0.4125 (var=0.0044)
- pi_6: -0.1973 (var=0.0076)

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
  - iter 1: 0.1107 (var=0.0021) (Δ vs real +0.0336)
  - iter 2: 0.1304 (var=0.0016) (Δ vs real +0.0533)
  - iter 3: 0.1144 (var=0.0013) (Δ vs real +0.0373)
  - iter 4: 0.1203 (var=0.0013) (Δ vs real +0.0432)
  - iter 5: 0.1155 (var=0.0014) (Δ vs real +0.0384)
  - iter 6: 0.1133 (var=0.0016) (Δ vs real +0.0362)
  - iter 7: 0.1218 (var=0.0021) (Δ vs real +0.0447)
  - iter 8 (current): 0.1264 (var=0.0022) (Δ vs real +0.0493)
**Other theories' values on this metric (for reference):**
- pi_2: 0.1196 (var=0.0023)
- pi_1: -0.3725 (var=0.0076)
- pi_3: 0.0861 (var=0.0026)
- pi_4: 0.0881 (var=0.0019)
- pi_5: 0.0789 (var=0.0021)
- pi_6: 0.0956 (var=0.0023)

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
  - iter 1: n/a (var=n/a)
  - iter 2: 0.6007 (var=0.0025) (Δ vs real -0.0163)
  - iter 3: 0.5796 (var=0.0045) (Δ vs real -0.0374)
  - iter 4: 0.5193 (var=0.0056) (Δ vs real -0.0978)
  - iter 5: 0.5270 (var=0.0048) (Δ vs real -0.0900)
  - iter 6: 0.6119 (var=0.0025) (Δ vs real -0.0052)
  - iter 7: 0.5315 (var=0.0065) (Δ vs real -0.0856)
  - iter 8 (current): 0.6456 (var=0.0040) (Δ vs real +0.0285)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6456 (var=0.0096)
- pi_2: 0.8389 (var=0.0095)
- pi_1: 0.6944 (var=0.0071)
- pi_4: 0.6485 (var=0.0027)
- pi_5: 0.4996 (var=0.0026)
- pi_6: 0.4811 (var=0.0035)

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
  - iter 1: n/a (var=n/a)
  - iter 2: 0.5058 (var=0.0033) (Δ vs real -0.0075)
  - iter 3: 0.5046 (var=0.0033) (Δ vs real -0.0087)
  - iter 4: 0.5454 (var=0.0039) (Δ vs real +0.0321)
  - iter 5: 0.5342 (var=0.0036) (Δ vs real +0.0208)
  - iter 6: 0.5479 (var=0.0033) (Δ vs real +0.0346)
  - iter 7: 0.5162 (var=0.0042) (Δ vs real +0.0029)
  - iter 8 (current): 0.5304 (var=0.0045) (Δ vs real +0.0171)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8387 (var=0.0140)
- pi_3: 0.6062 (var=0.0085)
- pi_1: 0.6825 (var=0.0042)
- pi_4: 0.5192 (var=0.0023)
- pi_5: 0.3950 (var=0.0023)
- pi_6: 0.3850 (var=0.0041)

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
  - iter 1: -0.1217 (var=0.0218) (Δ vs real +0.0467)
  - iter 2: -0.1983 (var=0.0180) (Δ vs real -0.0300)
  - iter 3: -0.1733 (var=0.0202) (Δ vs real -0.0050)
  - iter 4: -0.1525 (var=0.0204) (Δ vs real +0.0158)
  - iter 5: -0.1558 (var=0.0213) (Δ vs real +0.0125)
  - iter 6: -0.1150 (var=0.0201) (Δ vs real +0.0533)
  - iter 7: -0.1675 (var=0.0196) (Δ vs real +0.0008)
  - iter 8 (current): -0.1467 (var=0.0240) (Δ vs real +0.0217)
**Other theories' values on this metric (for reference):**
- pi_3: 0.2958 (var=0.0606)
- pi_4: 0.0308 (var=0.0233)
- pi_1: 0.1625 (var=0.0145)
- pi_2: 0.7617 (var=0.0356)
- pi_5: -0.1167 (var=0.0195)
- pi_6: -0.1350 (var=0.0257)

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
  - iter 1: n/a (var=n/a)
  - iter 2: 0.6350 (var=0.0077) (Δ vs real +0.5170)
  - iter 3: 0.1210 (var=0.0018) (Δ vs real +0.0030)
  - iter 4: 0.1200 (var=0.0031) (Δ vs real +0.0020)
  - iter 5: 0.1230 (var=0.0026) (Δ vs real +0.0050)
  - iter 6: 0.1000 (var=0.0013) (Δ vs real -0.0180)
  - iter 7: 0.1035 (var=0.0020) (Δ vs real -0.0145)
  - iter 8 (current): 0.1265 (var=0.0023) (Δ vs real +0.0085)
**Other theories' values on this metric (for reference):**
- pi_4: 0.7670 (var=0.0039)
- pi_3: 0.4190 (var=0.0107)
- pi_1: 0.8475 (var=0.0076)
- pi_2: 0.1180 (var=0.0091)
- pi_5: 0.2870 (var=0.0027)
- pi_6: 0.4360 (var=0.0065)

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
  - iter 1: n/a (var=n/a)
  - iter 2: 0.3462 (var=0.0035) (Δ vs real -0.2333)
  - iter 3: 0.3880 (var=0.0031) (Δ vs real -0.1916)
  - iter 4: 0.5907 (var=0.0049) (Δ vs real +0.0111)
  - iter 5: 0.5276 (var=0.0050) (Δ vs real -0.0520)
  - iter 6: 0.4427 (var=0.0046) (Δ vs real -0.1369)
  - iter 7: 0.4342 (var=0.0051) (Δ vs real -0.1453)
  - iter 8 (current): 0.5889 (var=0.0087) (Δ vs real +0.0093)
**Other theories' values on this metric (for reference):**
- pi_5: 0.7902 (var=0.0033)
- pi_4: 0.3191 (var=0.0050)
- pi_1: 0.1662 (var=0.0114)
- pi_2: 0.2569 (var=0.0106)
- pi_3: 0.3831 (var=0.0085)
- pi_6: 0.6756 (var=0.0042)

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
  - iter 1: n/a (var=n/a)
  - iter 2: 0.4088 (var=0.0125) (Δ vs real -0.0538)
  - iter 3: 0.3900 (var=0.0142) (Δ vs real -0.0725)
  - iter 4: 0.4550 (var=0.0138) (Δ vs real -0.0075)
  - iter 5: 0.4412 (var=0.0127) (Δ vs real -0.0213)
  - iter 6: 0.4975 (var=0.0167) (Δ vs real +0.0350)
  - iter 7: 0.4562 (var=0.0135) (Δ vs real -0.0063)
  - iter 8 (current): 0.4275 (var=0.0105) (Δ vs real -0.0350)
**Other theories' values on this metric (for reference):**
- pi_4: 0.5400 (var=0.0192)
- pi_5: 0.0550 (var=0.0029)
- pi_1: 0.8688 (var=0.0163)
- pi_2: 0.6587 (var=0.0158)
- pi_3: 0.6062 (var=0.0141)
- pi_6: 0.4213 (var=0.0156)

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
  - iter 1: n/a (var=n/a)
  - iter 2: 0.4155 (var=0.0059) (Δ vs real -0.1445)
  - iter 3: 0.4905 (var=0.0041) (Δ vs real -0.0695)
  - iter 4: 0.5725 (var=0.0090) (Δ vs real +0.0125)
  - iter 5: 0.5710 (var=0.0052) (Δ vs real +0.0110)
  - iter 6: 0.5810 (var=0.0051) (Δ vs real +0.0210)
  - iter 7: 0.5890 (var=0.0075) (Δ vs real +0.0290)
  - iter 8 (current): 0.5775 (var=0.0039) (Δ vs real +0.0175)
**Other theories' values on this metric (for reference):**
- pi_6: 0.8115 (var=0.0019)
- pi_4: 0.3985 (var=0.0055)
- pi_1: 0.1815 (var=0.0131)
- pi_2: 0.5685 (var=0.0047)
- pi_3: 0.5220 (var=0.0036)
- pi_5: 0.7830 (var=0.0038)

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
  - iter 1: n/a (var=n/a)
  - iter 2: 0.6300 (var=0.0062) (Δ vs real +0.0312)
  - iter 3: 0.5487 (var=0.0067) (Δ vs real -0.0500)
  - iter 4: 0.5694 (var=0.0070) (Δ vs real -0.0294)
  - iter 5: 0.5831 (var=0.0067) (Δ vs real -0.0156)
  - iter 6: 0.5962 (var=0.0091) (Δ vs real -0.0025)
  - iter 7: 0.5863 (var=0.0084) (Δ vs real -0.0125)
  - iter 8 (current): 0.6125 (var=0.0063) (Δ vs real +0.0138)
**Other theories' values on this metric (for reference):**
- pi_4: 0.6719 (var=0.0085)
- pi_6: 0.2988 (var=0.0058)
- pi_1: 0.8525 (var=0.0127)
- pi_2: 0.6687 (var=0.0078)
- pi_3: 0.6206 (var=0.0104)
- pi_5: 0.1444 (var=0.0034)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate's simulation results reveal a critical implementation failure rather than a theoretical one. The candidate produced scorable values on only 3 of 10 experiments (Exp 1: -0.2672 vs -0.2180; Exp 2: 0.1107 vs 0.0771; Exp 5: -0.1217 vs -0.1683 — all reasonable fits), and 'n/a' on Experiments 3, 4, 6, 7, 8, 9, 10. The pattern of which experiments failed is diagnostic: the three experiments that succeeded are exactly those whose panel-maximum validity sits at or below the plateau's upper edge (Exp 2's top validity is 0.92; Exp 5's is 0.95; Exp 1's top is evidently ≤ v_hi), while every experiment whose top validity is ≥ 0.98 — precisely the experiments that exercise the OUTLIER-GAP DISTRUST branch — returned n/a. The mechanism is clear: the outlier-gap branch calls math.exp(), but `math` is never imported in the predict function (only numpy is used elsewhere). Whenever v_top > v_hi, the code hits `math.exp(-x)` / `math.exp(x)` and raises NameError, crashing the simulation. The three 'successful' experiments never execute the novel branch, so what was scored is just the shared tally/plateau/chance-flank scaffold — essentially pi_3-like behavior. The accept-gate loss (0.7165) therefore reflects a candidate whose defining component (the relative distrust of isolated super-experts, the entire point of CREDT per the arbiter's recommendation) has never been tested against the data. The fits on Exps 1, 2, 5 are encouraging for the scaffold, and Exp 5's negative chance-cue sensitivity (-0.1217 vs observed -0.1683) confirms the chance-flank works directionally (slightly under-negative; kappa_lo at the upper end of its range would help), but none of this validates the theory's novel claim.
Rationale: Regenerate — but this is a bug fix plus recalibration, not a redesign; the CREDT mechanism family is correct per the arbiter's prescription and must be kept intact. (1) CRITICAL BUG: replace `math.exp` with `np.exp` (or add `import math`, or use a fully numpy-vectorized stable sigmoid) in the outlier-gap branch. This single fix will make the model scorable on Experiments 3, 4, 6, 7, 8, 9, 10 — the seven experiments that actually test the theory's novel component and on which it currently cannot be evaluated. (2) After the fix, verify the gap-sigmoid calibration against the four anchor points: gap ≈ 0.01 (Exp 10: follow-99 must be ≈ 0.60, so the discount at gap 0.01 must be near zero — this argues for s_gap at the low end of [0.02, 0.06] and g0 ≈ 0.05-0.08 so the sigmoid is still in its foot at gap 0.01); gap ≈ 0.08 (Exp 8: follow-98 ≈ 0.46, i.e., only a mild discount of the 0.98 expert relative to the 0.90 expert — check that w_top - w(0.90) combined with beta yields ~0.46, not a strong anti-follow); gap ≈ 0.24 (Exp 6: strong-side rate ≈ 0.118, requiring near-full collapse of the 0.99 weight — rho at the upper end, ~1.3-1.7, so the asymptote reaches ≤ 0); and Exp 7 (anti-99 ≈ 0.58) and Exp 9 (anti-99 ≈ 0.56) — inspect the actual next-highest validity in those panels and confirm the fitted sigmoid delivers the required discount at those specific gaps; if Exp 7's gap is small (~0.09) while its required discount is large, the tension with Exp 8 (gap 0.08, mild discount) must be resolved within the family, e.g., by sharpening s_gap and shifting g0 so the transition sits between the two experiments' gaps, or by noting Exp 7's second rung is lower than assumed. (3) Minor: Exp 5 came out at -0.12 vs observed -0.168 — once re-run, push kappa_lo toward 0.22-0.25 and/or beta toward 0.45-0.50 to deepen the chance-cue negativity. Do not change the architecture: sign votes × plateau weights × chance flank × gap-driven top-cue distrust × low-gain softmax × lapse is exactly what the arbiter prescribed and the three scored experiments support it; the candidate simply has never actually run its core mechanism on the data.

**Outcome of your advice:** iter 1 candidate loss=0.7165 -> iter 2 candidate loss=0.1676 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The bug fix worked: all ten experiments are now scorable and the loss dropped from 0.7165 to 0.1676. Seven of ten experiments fit well — Exps 1 (-0.276 vs -0.218), 2 (0.130 vs 0.077), 3 (0.601 vs 0.617), 4 (0.506 vs 0.513), 5 (-0.198 vs -0.168), 8 (0.409 vs 0.463), 10 (0.630 vs 0.599) are all within |Δ| ≤ 0.06. But now that the outlier-gap branch actually runs, it fails precisely on the experiments that test it hardest. (A) EXP 6 is catastrophic: 0.635 vs observed 0.118 (Δ +0.52). The diagnosis is identifiable from the arithmetic: the model can only produce a strong-side rate this high if the 'weak' experts (v ≈ 0.50-0.52) are receiving NEGATIVE chance-flank weights, so their endorsements are inverted — 2-4 weak cues endorsing B then push the score toward A, and in the R3 cell two plateau-weighted 0.75 cues (w ≈ +1) crush three inverted chance cues, giving p(strong) ≈ 0.9. Real subjects follow the raw endorsement count (pi_2, pure tallying, hits 0.118 exactly). Note the proposer's own rationale predicted 0.08-0.15 here assuming the weak cues were plateau cues — the simulation contradicts that, confirming the weak validities sit below the effective v_lo and land in the negative flank. (B) EXP 7: 0.346 vs 0.580 (Δ -0.23) — the model net-FOLLOWS the 99% expert; the gap discount (rho ≤ 1.4, w_p up to 1.1, so w_top ≥ -0.5 at best, likely ≈ -0.2) is too weak against background plateau cues aligned with the 99. (C) EXP 9: 0.416 vs 0.560 (Δ -0.14) — the dense ladder (0.99 atop 0.98, gap 0.01) correctly leaves the distrust off, but then the D-cells sit at 0.5 and the V1 cell is strongly PRO-99 because the negative chance flank adds ~+0.66 to the 99-side, dragging the overall anti-99 rate below 0.5. (D) Minor regressions vs iter 1 on Exps 1 and 2 (over-negative / over-tally) and a slight overshoot on Exp 5 (-0.198 vs -0.168) from the kappa_lo/beta increase I recommended last round — that advice was accepted and helped overall, but the flank depth now needs to settle back toward the middle rather than push further.
Rationale: Regenerate — the architecture is right and gate-accepted, but three experiments (6, 7, 9) mis-fit in a way that points at specific, fixable miscalibrations inside the CREDT family. Do NOT change the family (sign votes × plateau × chance flank × gap distrust × low-gain softmax × lapse); make these in-family repairs: (1) FIX EXP 6 FIRST (Δ +0.52, the dominant loss term). Push v_lo to the bottom of its range (0.52) so that any weak-but-diagnostic cue above 0.50 joins the plateau; if Exp 6's weak experts are exactly v = 0.50 (so no v_lo ≥ 0.52 can include them), then GATE the chance flank by the outlier-distrust signal: when the gap-sigmoid is near 1 (isolated super-expert), distrust generalizes to the whole panel and chance cues should be counted (weight → ~0 or a positive fraction of w_p) rather than inverted — e.g., effective kappa = kappa_lo · (1 − sig_gap). Exp 5's panel (top 0.95, no outlier, branch off) retains the full negative flank, preserving its -0.168; Exp 6's panel reverts to endorsement counting, which is what its data (and pi_2's exact 0.118) show. This is a gating/conditioning fix squarely within the 'consensus-relative' brief, not a new mechanism. (2) SHARPEN AND STRENGTHEN the gap discount: s_gap → ~0.02, g0 → 0.11-0.12, rho → 1.5-1.7 (top of the arbiter box), w_p → 0.9. Then: gap 0.01 → essentially no discount (Exp 10 stays ≈ 0.60); gap 0.08 → w(0.98) ≈ 0.65-0.7 vs w(0.90) = 0.9, giving follow-98 ≈ 0.45-0.47 (Exp 8); gap 0.24 → w(0.99) ≈ -0.7 to -0.8, which — once the chance cues stop being inverted — drives Exp 6's K/M cells toward the lapse floor and lifts Exp 7's anti-99 from 0.35 toward the observed 0.58. (3) EXP 9 (Δ -0.14): after (1)-(2), re-check; if the D-cells still sit at 0.5, add a small SMOOTH, gap-independent discount of the panel-max cue whenever v_top > v_hi (Δw ≈ 0.2-0.3, no cliff — adjacent validities remain near-exchangeable), which moves the D-cells toward 0.56 while spending Exp 10's current +0.03 slack. (4) Settle kappa_lo ≈ 0.19-0.21 and beta ≈ 0.40-0.45 — between the iter-1 and iter-2 settings — to re-center Exps 1, 2, 5 (my iter-1 push to raise them was accepted and helped, but it has now slightly overshot; take an intermediate value rather than reversing direction). (5) Re-verify Exps 3 and 4 after retuning, since they are currently excellent (Δ ≤ 0.016) and the rho/v_lo changes touch their panels too; do not sacrifice them for Exp 9.

**Outcome of your advice:** iter 2 candidate loss=0.1676 -> iter 3 candidate loss=0.0827 -> the gate marked it ACCEPTED.

### Iteration 3
Verdict: regenerate
Interpretation: The consensus-gating repair worked exactly as intended on its primary target: Exp 6 went from 0.635 to 0.121 vs observed 0.118 (Δ +0.003), and the aggregate loss halved from 0.1676 to 0.0827 — my iter-2 advice was accepted and paid off. Seven of ten experiments now fit within |Δ| ≤ 0.05 (Exps 1-6 all good, several essentially exact: Exp 4 Δ -0.009, Exp 5 Δ -0.005, Exp 6 Δ +0.003). The theory's core architecture — sign votes × plateau × chance flank × relative top-cue distrust × low-gain softmax × lapse — is now validated across the tally core (Exps 2-6), and the consensus-gating mechanism is confirmed as the right resolution of the Exp 5/Exp 6 chance-cue tension. However, the four remaining residuals are not noise; they form a coherent pattern that localizes to the top-cue discount calibration: (A) EXP 7: 0.388 vs 0.580 (Δ -0.19, by far the dominant remaining error). The model still net-FOLLOWS the 99% expert in exactly the experiment designed to test the theory's signature claim (distrust of an isolated super-expert). Notably, the proposer's own hand-simulation predicted 0.55-0.60 for this parameter setting but the simulation delivered 0.388 — a 0.17 discrepancy between hand-calc and code that strongly suggests the assumed panel structure (gap ≈ 0.14) is wrong; if Exp 7's true second rung sits at ~0.90 (gap ≈ 0.09), the current sigmoid (g0 ≈ 0.115, s ≈ 0.022) gives sig_gap ≈ 0.25, a discount of only ~0.4, leaving w(0.99) ≈ +0.25 — i.e., a followed, not distrusted, super-expert. (B) EXP 8: 0.390 vs 0.4625 (Δ -0.073) — the 0.98 expert (panel max, gap 0.08) is now over-discounted; follow-98 needs to rise. (C) EXP 10: 0.549 vs 0.599 (Δ -0.050) — the new d_w extremity discount overshot downward here (iter 2 was +0.031 without it). (D) EXP 9: 0.491 vs 0.560 (Δ -0.070) — improved but still short. The good news is that (A) and (B) point in OPPOSITE directions along the gap axis (Exp 7's gap needs a bigger discount, Exp 8's gap 0.08 needs a smaller one), so a sharper gap sigmoid can fix both simultaneously; (C) is fixed by trimming d_w. The genuine remaining tension is Exp 9 vs Exp 10 (both 0.99-atop-0.98, gap 0.01, needing anti-99 = 0.56 vs follow-99 = 0.60 respectively), which the gap term cannot separate — only the extremity term and the V1 cell's chance-cue treatment can, and these currently pull against each other via d_w.
Rationale: Regenerate — the candidate was gate-accepted (loss 0.1676 → 0.0827) and the architecture is confirmed, but shipping now would ship the theory with its centerpiece experiment (Exp 7, isolated-super-expert distrust) misfit by 0.19. All fixes are in-family calibrations of the existing four components; do NOT touch the scaffold. (1) FIX EXP 7 FIRST, AND START BY VERIFYING THE PANEL: the proposer's hand-simulated 0.55-0.60 vs simulated 0.388 discrepancy means the assumed gap (0.14) is almost certainly wrong. Extract Exp 7's actual validity vector and recompute. If the true gap is ~0.09-0.10 (dangerously close to Exp 8's 0.08), the separation between Exps 7 and 8 CANNOT come from the gap term — it must come from the extremity term, since Exp 7's top is 0.99 (extremity (0.99-v_hi)/(1-v_hi) = 0.75) while Exp 8's top is 0.98 (0.5): in that case scale d_w UP toward 0.45-0.55 so 0.99-panels get a large smooth extremity discount while 0.98-panels get a moderate one, and compensate Exp 10 (see 3). If the gap really is ~0.14, instead sharpen the transition: s_gap → 0.012-0.018 with g0 ≈ 0.10-0.115 and rho → 1.6-1.7, so gap 0.14 → sig ≈ 0.95+ (near-full collapse, w_top ≈ -0.8) while gap 0.08 → sig ≤ 0.1 (minimal discount). This latter setting simultaneously lifts Exp 8's follow-98 from 0.39 toward 0.46 — check that it lands there. (2) TRIM d_w to ~0.25-0.30 (from 0.30-0.40) to recover Exp 10's follow-99 from 0.549 toward 0.58-0.60 — unless step 1's diagnosis forces d_w up, in which case find Exp 10's compensation elsewhere (e.g., slightly higher w_p or beta). (3) EXP 9: after (1)-(2), decompose the residual by cell (S/D/V1). The V1 cell (one 0.99 vs three 0.50 cues) is likely the drag: with gap 0.01 the consensus gating is off, so the three chance cues stay INVERTED (-0.2 each), which pushes toward the 99-side — the wrong direction. Consider extending the gating so a mild gating signal fires on the extremity indicator itself (v_top > v_hi) even when the gap is dense, e.g., effective gating = max(sig_gap, lambda_ext * 1[v_top > v_hi]) with lambda_ext ≈ 0.4-0.6 — this is still 'consensus-relative distrust generalizing to the scale,' squarely in-family, and it leaves Exp 5 (top 0.95 < v_hi) untouched; but verify Exp 10's chance-cue backgrounds don't flip against the 99 there. (4) PROTECT THE CORE: Exps 1-6 are now all within |Δ| ≤ 0.05 with three essentially exact; the knobs being tuned (s_gap, g0, rho, d_w) only fire when v_top > v_hi, so they should not touch Exp 5, but re-verify Exps 3 and 4 (Exp 3 slipped from 0.601 to 0.580, Δ -0.037) after retuning, and re-center kappa_lo/beta only if the retune shifts them. (5) Do a code-level audit of the Exp 7 arithmetic before re-running: hand-simulate two or three actual Exp 7 trials with the verified validity vector and confirm the predicted anti-99 rate matches what the simulator produces — the iter-3 prediction-vs-simulation gap of 0.17 on this experiment means either the panel assumption or the hand-sim itself is unreliable, and that must be resolved before trusting any recalibration.

**Outcome of your advice:** iter 3 candidate loss=0.0827 -> iter 4 candidate loss=0.0573 -> the gate marked it ACCEPTED.

### Iteration 4
Verdict: regenerate
Interpretation: The iter-3 repairs landed almost perfectly on their targets: Exp 7 went 0.388 → 0.5907 (observed 0.5796, Δ +0.011), Exp 8 went 0.390 → 0.4550 (observed 0.4625, Δ -0.008), Exp 9 went 0.491 → 0.5725 (observed 0.560, Δ +0.013), and Exps 5/6 remain essentially exact (Δ +0.016, +0.002). The relocated ceiling (v_hi ≈ 0.981), the sharp rung-corroboration transition, and the extremity gating of the chance flank together resolved the three residuals I flagged, and the loss dropped 0.0827 → 0.0573. The theory's centerpiece — distrust of certainty that is relative to the panel ladder — is now validated on the experiments designed to test it. However, the iteration was not free: the fix was paid for out of the tally core. EXP 3 is now the dominant residual and has degraded monotonically across iterations (0.6007 → 0.5796 → 0.5193 vs observed 0.6170; Δ now -0.098, ~1.7 between-subject SDs). EXP 4 also moved away (0.5046 → 0.5454 vs observed 0.5133; Δ +0.032), in the opposite direction. This is exactly the failure mode the proposer flagged in their own residual-risk analysis ('if Exp 1's or Exp 4's top is 0.99 with gap >= 0.03 those experiments will shift') — the relocated ceiling plus the sharpened gap sigmoid (g0 ≈ 0.032, s_gap ≈ 0.009) now catches panels it was never meant to catch. The most likely mechanism: if Exp 3's panel is 0.99-topped with a mid gap (~0.09), sig_gap ≈ 1 and w_top collapses to ~-0.9, inverting the top cue on precisely the trials where it sits on the majority side — which crushes tally-following exactly as observed in the simulation. But the observed 0.617 requires Exp 3's top cue to remain trusted. Note the structural problem this exposes: if Exp 3's gap (~0.09) is LARGER than Exp 7's audited gap (0.01-0.04) yet Exp 3 needs trust and Exp 7 needs distrust, a single monotone gap-sigmoid is sign-wrong for one of them — the separation cannot come from the gap axis alone. Either the Exp 7 audit (which inferred the second rung circularly, by matching simulated anti-99 rates) is wrong, or the distrust signal must key on something richer than the single next-rung gap (e.g., corroboration count/density — how many cues sit within a tolerance of the top — or the top's isolation from the panel median rather than from the next rung). Remaining residuals elsewhere are acceptable: Exp 1 (Δ -0.035), Exp 2 (Δ +0.043), Exp 10 (Δ -0.029).
Rationale: Regenerate — but this is a surgical repair of one experiment, not a redesign; the architecture is confirmed and the 7/8/9 fixes must be preserved exactly. (1) FIX EXP 3 FIRST (Δ -0.098, the dominant loss term, and monotonically worsening). Apply the same discipline that fixed Exp 7: extract Exp 3's actual validity vector, hand-simulate 3-4 of its decisive trials under the current parameters, and identify which branch fires. Two cases: (a) If Exp 3's top is <= 0.98 (no outlier), the degradation came from the beta/kappa_lo/w_p re-centering — restore the iter-2-era core settings (which gave 0.6007) without touching the outlier-branch knobs, since those knobs provably do not fire on this panel. (b) If Exp 3's top is 0.99 with a mid-large gap, the current sharp sigmoid fully inverts its top cue — but the data demand trust there. In that case the gap axis is being asked to be anti-monotone (trust at Exp 3's larger gap, distrust at Exp 7's smaller gap), which no single sigmoid can deliver. Resolve it IN-FAMILY by enriching the corroboration signal: e.g., distrust magnitude keyed to the COUNT of cues within a tolerance of the top (a dense ladder of near-top cues corroborates; a single 0.99 floating above a mid-validity panel does not), or to the top's isolation from the panel's median/mean validity rather than only the next rung. Before building this, RE-AUDIT EXP 7's panel with a non-circular method: the iter-4 audit inferred the second rung by matching simulated rates, which is circular; if Exp 7's true gap is actually ~0.09-0.14 (as the iter-3 hand-sim assumed), the original monotone gap story may be recoverable and Exp 3/Exp 7 are then on the same side of the transition. (2) Check Exp 4 in the same audit (Δ +0.032, moved the wrong way this iteration); it is minor but its direction confirms something in the ceiling relocation is firing on its panel. (3) PROTECT THE WINS: Exps 5, 6, 7, 8, 9 are now within |Δ| <= 0.016 — whatever fixes Exp 3 must be verified not to fire on those panels; re-run them explicitly after the change. (4) Do NOT reverse the ceiling relocation or un-sharpen the gap transition wholesale — those changes are gate-validated; make the Exp 3 fix conditional/additive (a new gating condition or a re-derived corroboration signal), not a retuning of the accepted knobs. (5) Secondary: Exp 2's over-tallying (Δ +0.043) and Exp 1 (Δ -0.035) can absorb a small beta increase (0.43-0.45) if it helps Exp 3's case (a); leave them otherwise. The theory is one clean repair away from fitting all ten experiments within ~0.04; shipping now would ship it with its tally core (Exp 3) — the component every other experiment endorses — misfit by 0.10.

**Outcome of your advice:** iter 4 candidate loss=0.0573 -> iter 5 candidate loss=0.0572 -> the gate marked it ACCEPTED.

### Iteration 5
Verdict: regenerate
Interpretation: The iter-5 edit is a near-noise event: aggregate loss moved 0.0573 → 0.0572 (0.02% gain), and its actual outcome contradicts its own forecast. The edit targeted Exp 3 (predicted 0.56–0.57; delivered 0.527, Δ still −0.090 — the dominant residual, unchanged in substance) and paid for it by breaking Exp 7, which regressed from an essentially exact +0.011 to −0.052. Exps 1, 4, 10 improved slightly; everything else held. Nine of ten experiments now sit within |Δ| ≤ 0.052, and the theory's core claims (tally plateau, chance flank, consensus gating, 98/90 exchangeability, dense-ladder trust) are validated. But the edit's failure mode is diagnostic, not just disappointing — and it exposes a calculable inconsistency in the proposer's model of Exp 3. Arithmetic: under the stated parameter midpoints, moving g0 from 0.032 to 0.043 (s_gap 0.009 → 0.013) shifts w(0.99) at a gap of 0.05 from ≈ −0.70 to ≈ −0.30. Per the proposer's OWN audit of Exp 3, six of its eight decisive pairs are d=±1 trials on which the choice score reduces to w_top alone — so a +0.40 shift in w_top should have moved Exp 3's tally-following rate by roughly +0.03–0.04. It moved +0.008. Two hypotheses explain this, and they demand opposite repairs: (a) Exp 3's gap is actually LARGE (≳ 0.07), the sigmoid saturated at near-full collapse in BOTH calibrations, and the inversion that inferred gap ≈ 0.048 is invalid; or (b) the top cue is tied or on the minority side in most decisive trials, w_top is nearly irrelevant to Exp 3's metric, and the misfit lives in the core parameters (beta/w_p), not the distrust branch. The proposer cannot choose between these by more inversion arithmetic — the inversions are what failed. Note also that if (a) is true, the gap axis is now empirically ANTI-MONOTONE between the two experiments (trust at Exp 3's larger gap, distrust at Exp 7's smaller ~0.05 gap), which no single sigmoid can deliver — but the panels differ on a second, well-identified dimension the proposer's audit never tested: composition. Exp 7's design contains chance-cue blocks ([0,0,0,0,0,1,1,1]-type trials); Exp 3's panel is plateau-dense. A super-expert floating above a panel that itself contains coin-flip experts is an uncorroborated outlier; one atop a panel of solid experts is corroborated by the panel's competence. This is squarely within the 'consensus-relative' brief and is the one separator that does not gamble on ±0.005 of gap noise.
Rationale: Regenerate — not because the architecture is wrong (it is gate-validated across nine experiments) but because the current iteration bought nothing, broke Exp 7, and left the theory's stated tally core (Exp 3) misfit by 0.09 while its own predictions failed. The next iteration must be diagnostic-first, in-family, and must not oscillate. (1) RUN A SENSITIVITY SWEEP BEFORE TOUCHING ANY PARAMETER: on Exp 3's panel, override w_top across [−1, +1] with everything else fixed, and separately sweep beta, and plot the metric. If the metric is flat in w_top, hypothesis (b) holds — repair the core (restore beta/w_p toward the iter-2-era settings that gave 0.6007) without touching the validated distrust branch. If the metric responds strongly but the current operating point sits deep in a saturated region, hypothesis (a) holds — Exp 3's gap is large and the sigmoid has been pinned at collapse; no g0/s_gap retune can fix it. (2) IF (a): resolve the Exp 3/Exp 7 anti-monotonicity with the PANEL-COMPOSITION moderator, not the gap: modulate the top-cue discount by the panel's chance-cue fraction (e.g., multiply rho's effective magnitude, or gate the gap discount, by the fraction of cues below v_lo). Verify against every other panel before shipping: Exp 6 (chance-heavy, huge gap — collapse unchanged), Exp 9 (contains 0.55-class cues — confirm the extra distrust does not push its current +0.011 overshoot past ~0.03), Exp 10 (no chance cues — trust preserved), Exps 2/5/8 (branch off, untouched), Exp 1 (panel unknown — check its Δ −0.028 does not worsen). This is the refined version of the corroboration enrichment I suggested at iter 4; the proposer correctly showed count-within-tolerance and median-isolation are unidentified between the two panels, but chance-cue fraction differs by construction between them and is testable. (3) DO NOT simply move g0/s_gap back toward iter-4 values — that is the oscillation trap (iter 4 → 5 already moved them once for near-zero gain, and it would re-break Exp 7's direction). Any re-centering of the Exp 3/Exp 7 compromise must follow from the sweep, not precede it. (4) PROTECT THE WINS: Exps 1, 2, 4, 5, 6, 8, 9, 10 are all within |Δ| ≤ 0.052; every new gating condition must be verified not to fire on their panels. (5) Acknowledge explicitly in the next rationale which hypothesis the sweep supported and why the forecast now matches the simulation — the iter-5 gap between predicted (0.56–0.57) and delivered (0.527) on Exp 3 must close before any recalibration is credible. The theory is one correct diagnosis away from fitting all ten experiments within ~0.05; shipping now would ship it with its tally core misfit by 0.09 and a centerpiece experiment (Exp 7) freshly degraded.

**Outcome of your advice:** iter 5 candidate loss=0.0572 -> iter 6 candidate loss=0.0602 -> the gate marked it REJECTED.

### Iteration 6
Verdict: regenerate
Interpretation: The iter-6 edit (panel-composition moderator m) was the right diagnosis and half the right cure — but the gate rejected it (0.0602 vs 0.0572), and the per-experiment pattern shows exactly why. WHAT WORKED: Exp 3, the dominant residual for three iterations, collapsed from Δ -0.090 to Δ -0.005 (0.6119 vs 0.617). This confirms my iter-5 hypothesis (a): Exp 3's gap is large, the rung sigmoid was saturated at full collapse, and damping the discount via m_min ≈ 0.45 restored the top cue to w_top ≈ 0. The composition axis is real. WHAT FAILED: Exp 7 regressed from Δ -0.052 to Δ -0.137 (0.4427 vs 0.5796) — the exact 'worst case (ii)' the proposer flagged as the gamble. The arithmetic is diagnostic: with m = m_min ≈ 0.45 and Exp 7's gap ≈ 0.05 (sig_gap ≈ 0.63), w_top ≈ 0.9 - 0.21 - 1.6·0.63·0.45 ≈ +0.24 — a FOLLOWED super-expert, which is precisely what 0.4427 implies. The moderator's gamble was that Exp 7's weak-feature blocks sit below v_lo ≈ 0.53; the simulation proves they do not — Exp 7's weak cues are evidently in the 0.55-0.60 band, so c_frac ≈ 0 under the current threshold and the distrust SHRANK exactly where it must AMPLIFY. The forecast-vs-simulation gap recurred on exactly this experiment (predicted 0.58, delivered 0.443). Secondary: Exp 5 (Δ +0.053) and Exp 1 (Δ -0.061) both moved on a branch the edit provably cannot touch (tops 0.95, outlier off) — these are run-to-run noise/parameter-drift of ±0.03-0.04, visible throughout their trajectories, and should not be chased. Exp 4 worsened mildly (Δ +0.035); Exp 9 drifted +0.021; Exps 2, 8, 10 improved or held. Net: the iteration traded a 0.09 residual (Exp 3) for a 0.14 residual (Exp 7) plus noise — a net loss, correctly rejected.
Rationale: Regenerate. The base reverts to iter 5 (no moderator), so the next candidate must RE-INTRODUCE the composition moderator — it is the validated separator — but with its threshold corrected. The single failure is identifiable and fixable in-family: (1) DECOUPLE THE MODERATOR'S THRESHOLD FROM v_lo. The moderator currently counts c_frac as cues below v_lo ≈ 0.53, but Exp 7's weak block sits just ABOVE that (empirically pinned: c_frac ≈ 0 in Exp 7, else the delivered value would have been ~0.58-0.60, not 0.443). Introduce a separate composition threshold v_c ≈ 0.58-0.62 (or, equivalently robust, key m on the panel's mean/median validity being low) so that a 0.55-0.60 weak block registers as 'chance-laden scale' while genuinely diagnostic mid-band cues (0.65+) do not. Leave v_lo, the flank, and the gate untouched — Exp 5's -0.168 depends on them. (2) TARGET OPERATING POINTS: Exp 7 (three weak of eight → c_frac ≈ 0.375 → m ≈ 1.3): w_top at gap 0.05 ≈ 0.9 - 0.21 - 1.6·0.63·1.3 ≈ -0.62, the iter-4 operating point that fit Exp 7 at +0.011. Exp 3 (plateau-dense, gap ≥ 0.08): if its cues are all ≥ v_c, m = m_min and w_top ≈ 0 — the fix is preserved. Exp 6 (cues at 0.50, gap 0.24): m ≈ 1.33, collapse unchanged, K/M cells stay at the lapse floor. (3) HEDGE THE EXP 3 RISK EXPLICITLY: if Exp 3's panel contains 1-2 cues in the [0.53, 0.62] band, the new threshold would raise its m and re-break it. Keep the moderator GRADED (c_half ≈ 0.28-0.32, s_c ≈ 0.04) so a small weak fraction (≤ 0.2) yields m ≈ 0.5-0.6 (w_top ≈ -0.1 to -0.2, Exp 3 still ≈ 0.60) while a genuine weak block (≥ 0.3) amplifies. Verify Exp 9 (three 0.50 + one 0.55 → c ≈ 0.36 under v_c = 0.60) does not push its current +0.021 past ~0.04 — its sig_gap ≈ 0.07 sits in the sigmoid foot, so the extra distrust is small, but check the V1 cell. (4) DO NOT touch g0, s_gap, rho, d_w, beta, kappa_lo, w_p, v_hi — every one is gate-validated at current values, and iter 4→5 already demonstrated that retuning them buys nothing. (5) DO NOT chase Exps 1 and 5 this round: both moved on an untouched branch (±0.04 run noise); re-running will likely re-center them, and Exp 5's iter-3 value (-0.173) shows the flank is correctly calibrated. (6) Close the forecast-credibility gap: before shipping, hand-simulate two Exp 7 trials (e.g., [1,0,...] vs [0,...,1,0] and [1,0,1,0,...] vs [0,0,0,1,...]) under the new threshold and confirm the predicted anti-99 rate matches the simulator — the iter-6 miss of 0.14 on the forecast must not recur. The theory is one threshold correction away from fitting all ten experiments within ~0.04: the architecture (sign votes × plateau × chance flank × consensus-gated, composition-moderated relative distrust × low-gain softmax × lapse) is now validated on every component; only the moderator's read of 'weak' is miscalibrated.

**Outcome of your advice:** iter 6 candidate loss=0.0602 -> iter 7 candidate loss=0.0697 -> the gate marked it REJECTED.

### Iteration 7 (most recent)
Verdict: regenerate
Interpretation: The iter-7 edit implemented my iter-6 recommendation faithfully (decoupled composition threshold v_c ≈ 0.60, everything else untouched), but the recommendation's empirical premise was wrong, and the simulation functions as a clean natural experiment that falsifies it. Raising the threshold from v_lo ≈ 0.53 to v_c ≈ 0.60 moved Exp 3 DOWN (0.6119 → 0.5315, Δ back to -0.086) while leaving Exp 7 FLAT (0.4427 → 0.4342, Δ still -0.145). Reading this jointly with iter 6: (i) Exp 3's c_frac rose when the threshold rose, so Exp 3's panel contains cues in (0.53, 0.60) — the moderator fired there and re-collapsed its top cue; (ii) Exp 7 gained no moderator leverage, so either Exp 7 has NO cues below 0.60 (most likely — the iter-4→5 movement of Exp 7 under pure g0/s_gap changes shows its gap ≈ 0.03-0.06 sits in the sigmoid's sensitive region, ruling out the alternative that sig_gap ≈ 0 makes m irrelevant), or its gap is dense and the entire gap axis is dead for it. Either way the conclusion stands: the composition axis as 'fraction of weak cues' is ANTI-MONOTONE between Exps 3 and 7 — the experiment needing trust (Exp 3) has MORE weak cues and (per the accumulated inversions) the LARGER gap, while the experiment needing distrust (Exp 7) has the cleaner, denser panel. Both candidate separation axes now point the wrong way. This also marks the THIRD consecutive iteration in which the proposer's Exp 7 forecast missed the simulation by ~0.14 (predicted 0.56-0.62, delivered 0.4342; same magnitude of failure in iters 5 and 6) — the panel-inference-by-circular-inversion methodology has now failed often enough that it must be abandoned, not iterated. Meanwhile, the rest of the theory is in excellent shape: 8 of 10 experiments fit within |Δ| ≤ 0.045 (Exps 4, 5, 8 essentially exact at +0.003, +0.001, -0.006), confirming the scaffold — sign votes × plateau × consensus-gated chance flank × relative top-cue distrust × low-gain softmax × lapse. The candidate was correctly rejected (0.0697 vs 0.0572): it is worse than the base on BOTH stuck experiments simultaneously — a clean double loss, with nothing learned except the falsification above, which is nonetheless valuable.
Rationale: Regenerate. The base reverts to iter 5 automatically; the next candidate must change METHODOLOGY, not just parameters — three consecutive speculative panel assumptions have failed identically. (1) WITHDRAW THE WEAK-FRACTION MODERATOR. It is empirically falsified between Exps 3 and 7: do NOT re-submit it in any 'fraction below threshold' form, and do NOT raise v_c further (raising it 0.53 → 0.60 re-broke Exp 3 and bought nothing on Exp 7). (2) MEASURE BEFORE MODELING — run identification probes on the simulator using the accepted iter-5 base, exploiting that predict() receives parameters['validities']: (a) ENDPOINT PROBE: force w_top = +w_p (rho = 0, d_w = 0) and record Exp 3's and Exp 7's metrics; then force full collapse (rho large, g0 tiny) and record again. Interpolating each experiment's observed value between its two measured endpoints yields the REQUIRED w_top for each experiment directly from data — no panel assumptions. (b) GAP-LINEARIZATION PROBE: set s_gap very large so the sigmoid is ≈ linear in gap with moderate rho; the delivered Exp 3/Exp 7 metrics become smooth functions of the true gap; build a small lookup from synthetic panels spanning gap × weak-fraction × second-rung and read off each real experiment's actual panel statistics. This converts the guessing that has burned iters 5-7 into measurement. (3) ONLY THEN choose the distrust moderator, keyed to whatever statistic the probes show separates the panels in the RIGHT direction. Candidates the probes should explicitly test: v_second itself (distrust of a top cue that barely edges a high second rung — a 'spurious precision' reading), the count of cues within a tolerance of the top, and the top's isolation from the panel mean/median. If NO monotone statistic separates them, the in-family fallback is a NON-MONOTONE (peaked) gap response: discount maximal at intermediate gap (≈ 0.04-0.05, Exp 7), receding at larger gaps (≈ 0.07, Exp 3) and near zero at dense-ladder gaps (≈ 0.01, Exps 9/10), with Exp 6's full collapse preserved by the composition amplification firing on its 0.50-cue panel — this is still consensus-relative distrust of the panel-max expert, just non-monotone in the gap, and it is a calibration of the prescribed family, not a new family. (4) PROTECT THE WINS: any new term must be verified not to fire on Exps 1, 2, 4, 5, 8, 9, 10 — all currently within |Δ| ≤ 0.045 — and the gate-validated knobs (w_p, v_lo, v_hi, kappa_lo, g0, s_gap, rho, d_w, lambda_ext, beta, epsilon) must not be retuned; iter 4→5 already demonstrated retuning them buys nothing. (5) Do not chase Exps 1 and 2 (±0.04 movements on branches the edit cannot touch are run noise). (6) Close the forecast-credibility gap with MEASURED endpoints, not hand-simulations on assumed panels — the hand-sim vs simulator discrepancy on Exp 7 has now recurred three times and is the single most reliable predictor of a rejected candidate.

**Outcome of your advice:** iter 7 candidate loss=0.0697 -> CURRENT candidate loss=0.0392 -> the gate marked it ACCEPTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The iter-8 candidate is the payoff of the entire diagnostic arc. The proposer finally did what iters 5-7 demanded \u2014 measured operating points from delivered simulations rather than circular panel inversions \u2014 and then implemented exactly the in-family fallback my iter-7 critique sanctioned: a NON-MONOTONE (peaked) gap response for the panel-max expert's distrust, with a composition-amplified route that preserves Exp 6's collapse on chance-laden panels. The results validate it decisively. The two chronically stuck experiments are both fixed simultaneously: Exp 3 went from \u0394 -0.090/-0.086 (iters 4-7) to \u0394 +0.028 (0.6456 vs 0.6170), and Exp 7 went from \u0394 -0.145 (iter 7) to \u0394 +0.009 (0.5889 vs 0.5796) \u2014 the first iteration in the loop where BOTH are simultaneously within 0.03. Every other protected experiment held or improved: Exp 6 \u0394 +0.009, Exp 9 \u0394 +0.018, Exp 10 \u0394 +0.014, Exp 4 \u0394 +0.017, Exp 5 \u0394 +0.022, Exp 8 \u0394 -0.035, Exp 1 \u0394 -0.038, Exp 2 \u0394 +0.049. All ten experiments now fit within |\u0394| \u2264 0.05, seven within 0.03, and the aggregate loss dropped from 0.0572 to 0.0392 \u2014 a 31% improvement over the previous accepted base and by far the best in the loop. Critically, the forecast-vs-simulation credibility gap closed: the proposer's measured-endpoint method (linear fits through iters 4-7 delivered values) predicted Exp 7 \u2248 0.58 (delivered 0.5889) and Exp 3 \u2248 0.61-0.63 (delivered 0.6456) \u2014 the three-iteration streak of ~0.14 forecast misses is broken. The mechanism story is also now coherent rather than ad hoc: distrust of uncorroborated precision is maximal at intermediate gaps (spurious precision), recedes at dense ladders (corroboration) and at large coherent edges (categorically better instrument), and re-imposes on chance-laden panels (unearned certainty). This is still squarely the arbiter's CREDT family \u2014 sign votes \u00d7 mid-band plateau \u00d7 chance flank \u00d7 consensus gating \u00d7 relative top-cue distrust \u00d7 low-gain softmax \u00d7 lapse \u2014 with the distrust response shaped in-family, as prescribed.",
  "verdict": "continue",
  "rationale": "Ship it. The remaining residuals do not justify further tuning, for three reasons. (1) They are small and stable: Exp 2's over-tallying (+0.049) has sat between +0.034 and +0.053 across ALL eight iterations, untouched by every edit \u2014 it is a property of the shared scaffold at this parameterization, not of the distrust branch, and every attempt to move core knobs (iters 4\u21925) bought nothing while risking the eight experiments that now fit. Exp 8's \u0394 -0.035 has oscillated between -0.072 and +0.035 across iterations without any branch edit touching its 0.98-topped panel (v_hi \u2248 0.981 keeps it inside the plateau), i.e., it is run-to-run noise plus softmax-gain calibration, well within the between-subject variance (0.0103). Exp 1's \u0394 -0.038 likewise moved \u00b10.03 on branches that provably cannot fire on its panel. (2) The calibration is now delicately balanced on a non-monotone surface with six interacting shape parameters (g_star, s_l, s_r, c_half, s_c, c_amp) pinned by measured operating points; the loop's own history shows that touching validated knobs after a large gain (iters 5\u21926\u21927) reliably produces rejections, and the accept gate guarantees the base only regresses if I invite unnecessary edits. (3) The theory now satisfies the experiment-invariance standard: the same parameterized mechanism recovers the tally core (Exps 2-6), the chance-flank negativity (Exp 5), the isolated-outlier distrust (Exps 1, 6, 7), 98/90 exchangeability (Exp 8), dense-ladder trust (Exps 9, 10), and the corroborated-precision distinction (Exp 3 vs 7) \u2014 with no per-experiment hand-tuning, since predict() sees only the panel's validities. The candidate was gate-ACCEPTED at the loop's best loss; further iterations face diminishing returns against real risk of regression. Stop here."
}
```

## Usage

```json
{
  "input_tokens": 37943,
  "output_tokens": 1752,
  "reasoning_tokens": 701,
  "provider": "Sail Research"
}
```
