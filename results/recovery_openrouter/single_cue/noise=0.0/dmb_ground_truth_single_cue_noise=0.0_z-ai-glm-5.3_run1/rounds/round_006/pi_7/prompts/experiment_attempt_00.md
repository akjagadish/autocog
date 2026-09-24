# experiment_attempt_00

## System Prompt

You are a renowned cognitive scientist designing an experiment in the Decision Making (Binary Features) domain.

Your goal is to be an adversarial collaborator: propose a design whose outcomes would be predicted by your advocated theory but NOT by the competing theory. Both are provided below.

A useful proposal targets a *quantitative* dissociation between the two theories — how they respond differently to specific stimuli in addition to differences in overall performance.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

A multi-attribute decision-making experiment. On each trial the subject sees two options (A, B), each described by `n_features` integer expert ratings (`n_features` is set by the length of `validities` you propose). Choose `validities` — one per feature, each in [0.5, 1.0], order free — to fix each expert's advertised accuracy; subjects are told these values up front. Then choose `trial_a_ratings/trial_b_ratings` (each rating value in [0, 1]) so that the intended decision-making theories make distinguishable predictions: avoid degenerate pairs where every theory agrees, and prefer pairs that maximally dissociate the candidate theories. Validities and the trial ratings together define the design; they are fixed across all trials in this experiment. No trial-by-trial correctness feedback. The total number of trials per subject is held at roughly 96: each unique pair is repeated K = max(1, 96 // n_unique_pairs) times in an independently-randomized order per subject.

Subjects see the following instructions:
In this experiment you will repeatedly choose between two fictitious products, A and B. On every trial you will see `n_features` expert ratings for each product (the number of experts is fixed across all trials and is set by the length of `validities`).

Each rating is an integer in [0, 1]. The ratings are displayed as a horizontal filled bar with the numeric value (e.g. "0/1") shown next to it. Higher = more positive.

The same experts (in the same order) provide ratings for both products on every trial. Each expert's accuracy (their validity expressed as a percentage, e.g. "Expert 1 (80%)") is shown next to their rating on every trial AND is also listed up front in an "Expert accuracies" panel.

On each trial, decide which product is of higher quality and press A for product A or B for product B. There is no time limit and no feedback. Note that for the first ~`min_rt_ms` of each trial the answer prompt is hidden and the keys are locked, so subjects first see the full ratings and can answer once the A / B prompt appears — design pairs that actually require comparing the ratings.

Total trials per subject is roughly `MAX_TRIALS`: each unique pair you propose is repeated `K = max(1, MAX_TRIALS // n_unique_pairs)` times in an independently-randomized order per subject.

## ADVOCATED THEORY
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


## COMPETING THEORY
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


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This design creates two orthogonal quantitative dissociations between Take-The-Best (TTB) and Tallying, using 5 cues with well-spread validities (0.95, 0.85, 0.75, 0.65, 0.55). DISSOCIATION 1 (graded vs. flat dose-response curve): Trials C1-C3 hold the TTB-relevant structure constant (the top cue v=0.95 favors A, so TTB always picks A with probability (1-eps)*sigmoid(beta)+eps/2) while systematically increasing the number of lower-validity cues favoring B (tally margins 1-vs-2, 1-vs-3, 1-vs-4). TTB predicts a FLAT p(A) across these three trial types — the number of opposing low-validity cues is irrelevant to a one-reason rule. Tallying predicts a strictly MONOTONE DECREASING p(A) = sigmoid(-k*beta) for k=1,2,3 — a graded dose-response in tally margin. Trials C5-C7 mirror this with B favored by the top cue (TTB flat-high p(B); Tallying increasingly favors A). This is a pure quantitative dissociation: the slope of p(choice) against tally margin is exactly 0 for TTB and strictly negative for Tallying at any beta > 0. DISSOCIATION 2 (parameter-free point prediction): Trials T1-T4 have equal tallies (1 win each) but a clear validity-ordered discriminator (e.g., T1: A wins the v=0.95 cue, B wins only the v=0.55 cue). Tallying MUST predict exactly p(A)=0.50 on these trials for ALL parameter values (the softmax over tied scores is uniform, and the lapse also injects uniform). TTB predicts p(A) = (1-eps)*sigmoid(beta)+eps/2, which is strictly above 0.50 for any beta>0. Thus, if subjects choose systematically (not at chance) on T1/T3 (toward A) or T2/T4 (toward B), Tallying is falsified outright with no free parameters to rescue it — an adversarial knife. Trials G1-G2 are agreement controls (both theories pick the same option) that anchor the noise parameters. 12 unique pairs x K=8 = 96 trials per subject. Note the design also avoids any all-tie (fully degenerate) trials.

[1] This design implements a fully crossed TWO-FACTOR double dissociation between Tallying and Take-The-Best (TTB), which is structurally stronger than a one-dimensional dose-response: each theory predicts a main effect of exactly one factor and a PARAMETER-FREE ZERO effect of the other. Factor 1 is the tally margin d = (A's cue wins) - (B's cue wins), the only thing Tallying's prediction depends on: p_Tally(A) = (1-eps)*sigmoid(beta*d) + eps/2. Factor 2 is the direction s of the highest-validity discriminating cue, the only thing TTB's prediction depends on: p_TTB(A) = (1-eps)*sigmoid(beta*s) + eps/2. The 16 pairs cross d in {-4,-3,-2,-1,0,+1,+2,+3,+4} with s in {+1,-1}. KEY QUANTITATIVE PREDICTIONS: (1) WITHIN-COLUMN FLATNESS falsifies TTB with no free parameters: trials P1-P6, P13, P15 all have s=+1 (top cue v=0.92 favors A), so TTB predicts the IDENTICAL p(A) on all eight for every (beta, eps). Yet d ranges from +4 (P15: A wins 5 of 6 cues) to -4 (P6: A wins ONLY the top cue, B wins the other five). Tallying predicts p(A) falling from ~(1-eps) to ~eps/2 across these same trials. If subjects' p(A) on P6 is any lower than on P15, TTB is falsified outright -- no parameters can rescue a flat function that is observed to slope. The s=-1 column (P7-P12, P14, P16) mirrors this. (2) WITHIN-ROW FLATNESS falsifies Tallying with no free parameters: matched pairs (P2,P11), (P3,P10), (P4,P9), (P5,P8) share the same d with s flipped, so Tallying predicts EXACTLY equal p(A) within each pair for all (beta, eps), while TTB predicts a swing of (1-eps)*(2*sigmoid(beta)-1) -- up to nearly 1.0. The d=0 knife pair (P13, P14) is even sharper: Tallying must predict exactly 0.50 on both (tied tallies give a uniform softmax for any beta, and the lapse also injects uniform), while TTB predicts symmetric deviations toward the top-cue winner. (3) GRADED DOSE-RESPONSE: within each column, Tallying predicts a strictly monotone p(A) in d (sigmoid(beta*d) for d = -4..+4), giving a 9-point psychometric curve at fixed TTB output; TTB predicts a horizontal line. Conversely, across columns at fixed d, TTB predicts the full sigmoid swing while Tallying predicts zero. (4) EVEN THE AGREEMENT TRIALS SEPARATE QUANTITATIVELY: on P15 (d=+4, s=+1) both theories choose A, but Tallying predicts sigmoid(4*beta) vs TTB's sigmoid(beta) -- e.g., at beta=1, 0.982 vs 0.731 -- so confidence on unambiguous trials discriminates the theories too. As an adversarial touch against my own advocated theory, cues are displayed in descending-validity order (0.92 first), which maximally favors TTB's serial search (display order matches its search order) while being irrelevant to Tallying; if Tallying still wins under TTB's best-case display, the evidence is stronger. 16 unique pairs x K=6 = 96 trials; A/B sides are balanced (8 trials TTB-favors-A, 8 TTB-favors-B; 7 tally-favors-each-side, 2 tied), P15/P16 and P6/P12 are label-swapped mirrors that control side bias, no all-tie trials, and every pair requires genuine cue comparison.

[2] CORE LOGIC: The competitor (unweighted Tallying) predicts p(A) as a function of the raw tally margin d = (#A cue wins) - (#B cue wins) ONLY. My advocated theory (validity-weighted tallying) predicts p(A) as a function of the validity-weighted margin m = sum_j v_j^gamma * sign(a_j - b_j) ONLY. The design therefore decorrelates d and m: it holds d fixed while varying m, which yields two parameter-free knife predictions for the competitor and a graded dose-response for my theory. Cues: 6 experts, validities [0.99, 0.90, 0.80, 0.70, 0.60, 0.51] (maximal spread to maximize the achievable m-range; weights v_j^gamma live in [0.61, 1] over gamma in [0.05, 0.7]). DISSOCIATION 1 (parameter-free knife at d=0): Trials Z1-Z4, Z1', Z4', ZC all have exactly tied tallies (3 wins vs 3, or 2 vs 2 with ties). The competitor must predict p(A) = 0.50 EXACTLY on all seven, for every (beta, epsilon) -- a tied softmax is uniform and the lapse injects uniform. My theory predicts a graded ladder in m: at gamma=0.4, m = +0.42 (Z1), +0.33 (Z2), +0.24 (Z3), +0.12 (Z4), giving p(A) = sigmoid(2*beta*m) strictly increasing (e.g., beta=0.25: 0.553, 0.541, 0.530, 0.516; at gamma=0.7, beta=0.55: 0.680, 0.645, 0.605, 0.555). Z1' and Z4' mirror the ladder endpoints (m = -0.42, -0.12) testing sign symmetry, and ZC is a near-zero-m control (A wins cues 1 and 6, B wins cues 2 and 5; v1+v6 = v2+v5 = 1.50, so m ~ -0.01 for all gamma) where BOTH theories predict ~0.50 -- this rules out a generic 'follow the highest-validity cue' strategy (which would pick A on ZC) and generic tied-trial side bias, isolating pure validity weighting. Any systematic deviation from 0.50 on Z1/Z1' falsifies the competitor outright with no free parameters; any nonzero slope of p(A) against m across Z4->Z1 falsifies it as a dose-response. DISSOCIATION 2 (parameter-free within-d flatness): Trials P1, P3, P4 all have d = +1 (3 cue wins vs 2, one tie) and trials N1-N4 all have d = -1. The competitor must predict the IDENTICAL p(A) on P1/P3/P4 (and on N1-N4) for every parameter setting -- only the count matters. My theory predicts a strong within-d ladder: at gamma=0.4, m = 1.19 (P1: A wins the three best experts), 0.70 (P3), 0.49 (P4: A wins the three WORST experts, B wins the two best) -> p(A) = 0.644, 0.587, 0.561 at beta=0.25, spreading to 0.81 vs 0.55 at gamma=0.7, beta=0.55. A 15-25 point spread where the competitor demands exact equality is the sharpest quantitative signature of validity weighting. Note P4 is doubly diagnostic: A wins the majority of cues but only the low-validity ones, so the two theories still agree on the choice while disagreeing sharply on confidence -- pure quantitative dissociation. DISSOCIATION 3 (cross-family m-matching): Z1 (d=0, m~0.42-0.69) vs P4 (d=+1, m~0.16-0.49) nearly match m while mismatching d; the competitor predicts a large gap (sigmoid(2*beta) vs 0.50), my theory predicts a small or even reversed gap -- an honest adversarial test of my own m-only claim. ANCHORS: G1/G2 (d = +/-3, all four best experts vs one worst) let both theories agree at high confidence and pin down beta and epsilon; notably, at my claimed regime beta~0.25, G1 predicts only ~0.81-0.89, while the competitor (beta up to 20) can go to ~1.0, so even anchor confidence separates the families quantitatively. ADVERSARIAL HONESTY: if subjects sit at exactly 0.50 across the whole d=0 family and show zero within-d m-slope, my theory's central claim (gamma > 0) is falsified and the competitor wins; the design is a clean gamma-estimation experiment because the competitor is exactly the gamma -> 0 nest of my family. 16 unique pairs x K=6 = 96 trials; sides balanced (8 A-tilted, 7 B-tilted, 1 neutral control), no all-tie trials, every trial requires comparing ratings across experts, and the descending-validity display is irrelevant to both theories (neither uses display order).

[3] ADVOCATED THEORY: unweighted Tallying (choice depends ONLY on the raw tally margin d = #A-cue-wins - #B-cue-wins, with unbounded beta up to 20). COMPETITOR: validity-weighted tallying whose own theoretical claim pins it to a bounded box: gamma in [0.05, 0.7], beta in [0.03, 0.55], epsilon in [0, 0.08] ('unsaturated regime'). The design exploits two structural facts about that box. FACT 1 (HARD CEILINGS): the competitor's maximum achievable confidence on a trial is (1-eps)*sigmoid(2*beta*m) with m = sum_j v_j^gamma*sign(a_j-b_j). Because beta <= 0.55 and any single cue's weight is <= 1, on ANY trial where exactly one more cue favors A than B and only one cue discriminates (S1, S2), m <= 0.9995, so the competitor's absolute ceiling is p(A) = 0.750 -- no parameter setting in its box can exceed it. On minimal-m d=+2 (A wins only the two lowest-validity cues, L2) the ceiling is 0.894, and on minimal-m d=+3 (L3) it is 0.961. Tallying, with beta up to 20, can predict ~0.99 on all of these. So if subjects choose the count-winner on >80% of S1/S2 trials or >92% of L2 trials, the competitor is falsified outright with zero free parameters -- the exact mirror-image knife of Experiment [2], but aimed at the bounded regime rather than at gamma. FACT 2 (MINIMAL-m LADDER): on the ladder S2 -> L2 -> L3 every A-win is on a low-validity cue, so the competitor's evidence per tally step is only ~0.62-0.97 of a raw count unit; its steepest possible psychometric slope in d is 2*0.55*0.97 ~ 1.07 logits per step, whereas Tallying's is 2*beta, unbounded. A subject psychometric function in d that is steeper than ~1.07 logits/unit (e.g., 0.75 at d=1 rising to ~0.99 at d=3) exceeds the competitor's box. DISSOCIATION FAMILY d=+/-1 (S1, S2, S3, S4 and mirrors): all four A-tilted trials have d = +1, so Tallying makes the PARAMETER-FREE prediction p(S1)=p(S2)=p(S3)=p(S4) exactly, for every (beta, epsilon). The competitor predicts a large within-d spread because m differs sharply across compositions: at gamma=0.7, beta=0.55 its predictions are S1=0.749, S2=0.665, S4=0.816, and -- the sharpest cell -- S3=0.389, i.e., BELOW 0.50. S3 is the sign-flip trial: A wins the four weakest experts (v=0.54,0.53,0.52,0.51) while B wins the three super-experts (v=0.99,0.98,0.97); the raw count favors A (4 vs 3) but the validity-weighted margin at high gamma is m = -0.41, so the competitor predicts a B preference while Tallying predicts the same strong A preference as on S1. Subject behavior on S3 is a direct gamma-map: p(A) near p(S1) kills the competitor twice (ceiling violation + zero validity weighting); p(A) < 0.50 falsifies Tallying on a choice (not just confidence) measure. The validity vector [0.99, 0.98, 0.97, 0.55, 0.54, 0.53, 0.52, 0.51] is engineered for this: the top-two/top-three weights (0.993+0.986+0.979 = 2.958 at gamma=0.7) exceed any three or four low-cue weights (e.g., 2.548 for the four lowest), which is what makes the sign flip achievable at all with binary cues. HONEST ADVERSARIAL KNIVES AGAINST MY OWN THEORY: (i) T0 (d=0: A wins the three super-experts, B wins the three worst): Tallying MUST predict exactly 0.50 (tied softmax is uniform and the lapse injects uniform), while the competitor predicts p(A) up to 0.76 -- any systematic A-bias on T0 falsifies Tallying with no parameters; (ii) the within-d=1 spread S3<S2<S1<S4 and the L2-vs-M2 comparison (both d=+2; Tallying demands exact equality, competitor predicts L2 < M2 by up to ~10 points since M2's wins are on the two super-experts, m ~ 2.0 vs 1.26-1.93) are the competitor's best cells, included deliberately; (iii) displaying the three ~99% experts first maximizes validity salience, favoring the competitor's processing assumptions. ANCHORS: G+/G- (d = +/-6) are near-agreement trials (both theories ~1) that pin lapse rates; even here the competitor's epsilon <= 0.08 forces p in [0.958, 0.998] on G+, so an anchor error rate above ~5% strains it while Tallying's epsilon up to 0.5 absorbs any sloppiness. PREDICTED PATTERN IF MY THEORY IS RIGHT: flat near-ceiling performance across the entire d=1 family (all four compositions equal, ~0.95+), steep sigmoid in d reaching ~0.99 by d=3, exactly 0.50 on T0, L2 = M2. PREDICTED PATTERN IF THE COMPETITOR IS RIGHT: graded 0.65-0.75 on d=1 with a 15-40 point within-d spread, S3 at or below 0.50, shallow d-slope, A-bias on T0. The two patterns are disjoint almost everywhere: 16 unique pairs x K=6 = 96 trials; sides balanced (8 count-favors-A, 7 count-favors-B, 1 tied); no all-tie trials; every trial requires scanning all 8 experts to locate discriminators.

[4] CORE LOGIC. The two theories share the same architecture (sign votes x weights x softmax) and differ ONLY in the weight function. The one validity where they are structurally disjoint is v = 0.5: the competitor's weight is max(llr - tau, 0)^delta with llr(0.5) = 0, so a 50%-accurate expert is annihilated EXACTLY (weight 0) for every (delta, tau, beta, epsilon) in its box. My advocated theory gives it weight 0.5^gamma in [0.62, 0.97] -- i.e., 62-97% of a super-expert's weight. The design therefore builds everything around three chance experts (v = 0.5) set against one near-chance diagnostic expert (v = 0.55) and one super-expert (v = 0.95), using 8 features: [0.95, 0.85, 0.75, 0.65, 0.55, 0.50, 0.50, 0.50]. DISSOCIATION 1 (parameter-free knife: chance-only trials). Trials Z1-Z3 and mirror Z1m have ONLY v=0.5 cues discriminating. The competitor must predict p(A) = 0.50 EXACTLY on all four (all weights zero -> uniform softmax, and the lapse also injects uniform) for every parameter setting. My theory predicts a strictly graded ladder in the net chance-cue count k: p(A) = sigmoid(2*beta*0.5^gamma*k); at representative (beta=0.25, gamma=0.4): Z3 = 0.59, Z2 = 0.68, Z1 = 0.76 (range across my box: 0.51-0.96, always > 0.5 for k > 0). Any systematic deviation from 0.50 on Z trials, or any nonzero slope of p(A) against k, falsifies the competitor outright with no free parameters. DISSOCIATION 2 (parameter-free flatness + robust sign flip: super-expert vs chance-cue dose). Trials X0-X3 hold the diagnostic structure fixed (B wins the v=0.95 expert; for the competitor the chance cues contribute exactly nothing) while the net chance-cue count for A varies 0, 1, 2, 3. The competitor must predict the IDENTICAL p(A) on X0, X3, X2, X1 for all parameters -- its entire box spans only p(A) in [0.29, 0.36]. My theory predicts a strictly monotone ladder crossing 0.5: at (beta=0.25, gamma=0.4): X0 = 0.38, X3 = 0.47, X2 = 0.57, X1 = 0.66. Critically, on X1 and X2 the SIGN of preference disagrees robustly across BOTH entire parameter boxes: my theory predicts p(A) > 0.5 for every admissible (gamma, beta) (minimum 0.52), the competitor predicts p(A) < 0.5 for every admissible (delta, tau, beta, epsilon) (maximum 0.36). Mirrors Xm1, Xm2 flip sides and control for side bias. DISSOCIATION 3 (threshold structure of the 0.55 expert -- includes an honest knife against my own theory). The competitor treats v=0.55 as a genuinely diagnostic cue with weight 0.72-0.85 (about 60-75% of the super-expert's weight), while my theory treats 0.55 and 0.5 as nearly identical (weight ratio (0.5/0.55)^gamma in [0.94, 1.0]). W1 (A wins the 0.55 cue, B wins one 0.5 cue): my theory predicts p(A) <= 0.52 across its ENTIRE box (the validity difference is at most 0.042 in weighted units, so even beta = 0.55 yields only 0.523); the competitor predicts 0.59-0.65. This cell alone can falsify MY theory -- included deliberately. W2/W3 (A wins 2 or 3 chance cues, B wins the 0.55 cue): my theory predicts an increasing A-ladder (0.59 -> 0.68 at representative params; >= 0.52 across the box); the competitor predicts p(A) = 0.36-0.41, and moreover demands W2 = W3 EXACTLY (both equal -w(0.55) in evidence) -- a second parameter-free flatness knife. Wm2 mirrors W2. So the W family pits the competitor's sharp 0.5/0.55 threshold against my theory's near-equal weighting: the competitor demands W1 = 1 - W2 = 1 - W3 with W2 = W3; my theory demands W1 ~ 0.50 with W2 < W3 strictly. ANCHORS. G1/G1m (four best experts vs the 0.55 expert) are agreement trials pinning beta and epsilon (my theory ~0.81/~0.19; competitor ~0.90/~0.10 at representative params -- even anchor confidence mildly separates them). KEY QUANTITATIVE METRICS: (i) slope of p(A) on net chance-cue wins k at fixed diagnostic opposition (Z family and X family): exactly 0 for the competitor, ~0.08-0.10 per cue for my theory at the claimed unsaturated regime (beta ~ 0.25); (ii) the sign of p(A)-0.5 on X1/X2/W2: positive for all my parameters, negative for all competitor parameters; (iii) p(W1): mine <= 0.52, competitor >= 0.59 -- disjoint intervals, a two-sided knife. PREDICTED PATTERN IF MY THEORY IS RIGHT: graded chance-cue dose-response everywhere (Z ladder 0.59/0.68/0.76; X ladder 0.38/0.47/0.57/0.66; W ladder 0.50/0.59/0.68 at representative params), i.e., subjects treat 50%-accurate experts as carrying most of a super-expert's weight. IF THE COMPETITOR IS RIGHT: exact 0.50 on all Z trials, flat 0.32 across the X family, W1 = 0.62 with W2 = W3 = 0.38, i.e., subjects exactly annihilate chance experts and sharply distinguish 0.55 from 0.50. The patterns are disjoint almost everywhere. 16 unique pairs x K = 6 = 96 trials; sides balanced (8 A-tilted, 7 B-tilted, 1 near-neutral); no all-tie trials; every trial requires scanning all 8 experts to locate discriminators; the three 50% experts are announced up front, so the design is a fair test of whether explicitly chance information is integrated or annihilated.

[5] CORE LOGIC. Both theories share the identical architecture (sign votes x weights x softmax x lapse) and differ ONLY in the weight function: competitor w_j = v_j^gamma (gamma in [0.05, 0.7]); mine w_j = max(log(v_j/(1-v_j)) - tau, 0)^delta (delta in [0.1, 0.2], tau in [0, 0.04]). With no chance-level cues, the only identifiable quantity is the EXCHANGE RATE ER(v_hi, v_lo) = w(v_hi)/w(v_lo) between validity classes, measurable behaviorally from conflict trials. The design engineers validity classes so that the two theories' ER boxes are DISJOINT. Validities: [0.99, 0.99, 0.75, 0.75, 0.51, 0.51, 0.51, 0.51] -- two super-experts, two mid experts, four barely-diagnostic 51% experts (all cues are genuinely diagnostic; NO v=0.5 cues, so this is a pure test of the weighting FORM among diagnostic cues, unlike Exp [4]'s chance-annihilation design, and unlike [2]/[3]'s count-vs-weight decorrelations). The key arithmetic: llr(0.99)=4.595 vs llr(0.51)=0.040, a 115x log-odds gap, so even the heaviest delta-compression (delta=0.1, tau=0) leaves my theory a minimum super-premium ER(0.99,0.51) = (4.595/0.040)^0.1 = 1.61, while the competitor's maximum is (0.99/0.51)^0.7 = 1.59 at its steepest corner (gamma=0.7). DISJOINT INTERVALS: competitor ER(99,51) in [1.03, 1.59], mine in [1.61, ~2.6+] (center delta=0.15, tau=0.01: 2.13). Likewise ER(0.75, 0.51): competitor [1.02, 1.31], mine [1.39, 1.94] -- disjoint with a wider margin. Psychologically: does one 99% expert outweigh TWO 51% experts (my theory: yes, ~2 weak experts per super) or barely one (competitor: at most 1.59, i.e., two weak experts always beat one super)?

BETA-FREE RATIO STATISTICS. Because both theories share the softmax, the logit RATIO between trial types cancels beta and (to first order) epsilon: ER_hat = 1 + logit(K1)/logit(W2), where K1 (A wins one 99% expert, B wins one 51% expert) has margin w99 - w51 and W2 (A wins two 51% experts, B wins one) has margin w51. Competitor box forces ER_hat <= 1.59; my box forces ER_hat >= 1.61 (center: 2.13). The parallel mid-level statistic ER_M = 1 + logit(R1)/logit(W2) (R1: one 75% expert vs one 51% expert) is disjoint with a bigger gap: competitor <= 1.31, mine >= 1.39 (centers 1.15 vs 1.71). These ratios remain disjoint even at the two theories' closest corners (delta=0.1, tau=0 vs gamma=0.7), where single-cell probabilities nearly converge -- the ratio metrics are the robust separator.

ONE-SIDED PARAMETER-FREE KNIVES against the competitor. (i) K2 (one 99% for A vs two 51% for B) and M4 (two 99% vs four 51%): the competitor's margin 0.99^gamma - 2*0.51^gamma (or doubled) is NEGATIVE for every (gamma, beta, epsilon) in its box, so it must predict p(A) < 0.50 strictly (box range roughly [0.26, 0.50) at K2). My center predicts p(A) = 0.51 (K2) and 0.52 (M4), rising to 0.60-0.70 at higher delta/tau. Any reliable p(A) > 0.55 on these cells falsifies the competitor outright. (ii) M3 (two 99% vs three 51%): competitor box max p(A) = 0.536 (at gamma=0.7, beta=0.55); my box min p(A) = 0.521 (at delta=0.1, tau=0, beta=0.27) -- near-disjoint intervals, with centers far apart (0.44 vs 0.62). p(M3) >= 0.60 is essentially impossible for the competitor. (iii) K1: competitor p(A) <= 0.60 across its entire box (center 0.53); mine >= 0.56 (center 0.61). (iv) R3 (two 75% vs three 51%): competitor in [0.40, 0.48], mine in [0.48, 0.69], centers 0.42 vs 0.54.

CROSSOVER DISSOCIATION. In the K family (one super expert for A vs k weak experts for B, k = 1, 2, 3) the competitor's psychometric crosses p = 0.5 strictly between k=1 and k=2 (one super is worth less than two weak experts for EVERY gamma), whereas my theory's crossover sits at k* = ER in [2, 2.6] for the bulk of my box (center: p(K1)=0.605, p(K2)=0.512, p(K3)=0.418 -- crossing between k=2 and k=3). Competitor center: 0.531, 0.413, 0.305. The location of the crossover is a direct, interpretable estimate of the super-expert premium.

HONEST ADVERSARIAL CELLS AGAINST MY OWN THEORY. The competitor's weak-cue weights (0.51^gamma in [0.62, 0.97]) are LARGER than mine at high tau (down to ~0.09 if tau annihilates the 0.51 cues), so on weak-cue-dominated trials the competitor can be far more confident than I can: W3 (three 51% for A vs one for B): my box ceiling ~0.75 (center 0.680), competitor up to 0.89 -- observed p(W3) > 0.78 falsifies my box. W4a (four 51% for A vs one 75% for B): my ceiling ~0.79 (center 0.70), competitor up to 0.92 -- p(W4a) > 0.85 falsifies my theory with no parameters to rescue it. G1 anchors (two supers + one 75% vs one 51%): I predict >= 0.79 across my box (center 0.87; my beta floor 0.27 forces high confidence), the competitor can go as low as 0.53. X1/X1m (the two 99% experts split between options) are tied-evidence controls where BOTH theories must predict exactly 0.50 for every parameter -- they isolate side/display bias that would otherwise contaminate every knife cell.

PREDICTED PATTERN IF MY THEORY IS RIGHT (center params delta=0.15, tau=0.01, beta=0.32): K1 0.605, K2 0.512, K3 0.418, M3 0.616, M4 0.524, R1 0.567, R3 0.540, W2 0.593, W3 0.680, W4a 0.703, G1 0.868, X1 0.500 -- a super-premium of ~2x, near-0.5 choices at the 1-super-vs-2-weak and 2-super-vs-4-weak ratios, moderate (never extreme) confidence on weak-aligned trials. IF THE COMPETITOR IS RIGHT (center gamma=0.35, beta=0.30): K1 0.531, K2 0.413, K3 0.305, M3 0.444, M4 0.332, R1 0.517, R3 0.417, W2 0.616, W3 0.721, W4a 0.795, G1 0.780 -- a super-premium of ~1.26x, consistent weak-side majorities at every 2:1 weak:super ratio, and higher confidence on weak-cue stacks. The patterns diverge most where it matters: every super/mid-vs-weak conflict cell separates the centers by 7-19 points, and the ER ratio statistics separate the entire boxes.

STRUCTURE. 24 unique pairs x K = 96//24 = 4 reps = 96 trials; every family has an exact A/B mirror (12 tilted each way) controlling side bias; no all-tie trials; every trial has 2-5 discriminators among 8 features, requiring genuine scanning; tied features are shown as 0/0 for both options. Display order (descending validity) is irrelevant to both theories, which re-sort internally. Differentiation from prior experiments: no v=0.5 cues and no chance-annihilation logic ([4]); no tally-margin decorrelation or d=0 tied-count knives ([1], [2]); no bounded-box ceiling attack on an unbounded-beta tallying model ([3]) -- here both models are bounded, weighted, and compared on the functional FORM of diagnosticity weighting via exchange rates.

[6] CORE LOGIC. Both theories share the identical architecture (sign votes x validity weights x softmax x lapse) and differ ONLY in the weight function's shape across validity. My advocated theory makes three sign-structured claims the competitor cannot mimic: (i) claimed near-certainty is ANTI-diagnostic: w(0.99) is negative (~ -1 trusted-cue unit, box |w(0.99)| in [0.86, 1.24]) while the competitor gives 0.99 its LARGEST positive weight (w in [1.17, 1.38]); (ii) trust peaks at v~0.90 and FALLS by 0.95 (my w(0.95)/w(0.90) ~ 0.5-0.7) while the competitor is monotone (though its delta~0.15 flattening makes 0.95 vs 0.90 nearly a tie, ratio 1.05); (iii) near-chance experts carry a small NEGATIVE weight (w(0.50) = -0.075 to -0.105) while the competitor annihilates v=0.5 EXACTLY (llr=0 for every parameter) and gives v=0.51 up to +0.53 cue units where my theory gives ~0. Validities: [0.99, 0.95, 0.90, 0.80, 0.70, 0.51, 0.50, 0.50] -- one expert in each regime. 19 unique pairs x K = 96//19 = 5 = 95 trials.

FAMILY S/X (super-expert inversion + parameter-free flatness knife). S1 (A wins only the 0.99 expert): my box p(A) in [0.08, 0.20] (center 0.13); competitor box [0.62, 0.73] (center 0.68) -- DISJOINT with a 40-point gap, the flagship cell: I predict subjects choose AGAINST the 99% expert's endorsement, the competitor predicts they follow it. S1m mirrors. X1 (A wins 0.99, B wins one 0.50 cue) and X3 (A wins 0.99 plus one 0.50 cue): the competitor's w(0.50)=0 exactly, so it must predict p(S1) = p(X1) = p(X3) EXACTLY for every (delta, tau, beta, epsilon) -- a parameter-free triple equality. My theory predicts a strict ordering X3 (0.12) < S1 (0.13) < X1 (0.15) at center, with box gaps of 0.05-0.13 per step, because a chance cue won by B is anti-diagnostic evidence FOR A. Observed equality across S1/X1/X3 falsifies my chance-cue anti-weight; any reliable deviation from the triple equality falsifies the competitor outright.

FAMILY P (peak-at-0.90 non-monotonicity). P1 (A wins 0.95, B wins 0.90): my box [0.23, 0.38] (center 0.31 -- the 0.90 expert beats the 0.95 expert); competitor box [0.47, 0.55] (center 0.51 -- its flattened weights make the two nearly exchangeable). Disjoint. P1m mirrors. Note p(P1) > 0.56 would falsify BOTH theories, an honest boundary.

FAMILY D (calibrated cliff depth vs super-expert coalitions). D4 (A wins 0.99+0.90, B wins 0.80): my box [0.10, 0.25] (center 0.17 -- the super-expert's anti-vote nearly cancels the 0.90 endorsement, so a single 0.80 expert wins); competitor box [0.62, 0.87] (center 0.69). Sign-flip cell. D4m mirrors. D1m (A wins 0.90, B wins 0.99): mine [0.93, 0.97] vs competitor [0.42, 0.50] -- the competitor can never exceed chance when a lone 0.99 expert opposes a 0.90 expert, I predict near-certainty. D5 (A wins 0.99+0.90+0.80, B wins 0.70) tests the 'cannot invert coalitions that include it' claim: I follow the 3-vs-1 tally only MODERATELY (center 0.58, box [0.44, 0.71]) while the competitor is confident (center 0.81, box [0.74, 0.94]) -- a confidence dissociation even where choices agree; p(D5) >= 0.85 strains my calibrated cliff depth. D2 (A wins 0.99+0.51, B wins 0.90) vs D1-structure: the competitor predicts D2 > D1 by up to +0.15 (the 0.51 cue is worth up to 0.53 units to it); I predict |D2 - D1| <= 0.02 (w(0.51) ~ 0), and my box on D2 is [0.04, 0.07] vs competitor [0.48, 0.65] -- disjoint.

FAMILY C (chance-cue anti-weight). C2 (A wins both 0.50 cues, nothing else discriminates): the competitor must predict EXACTLY 0.50 (all weights zero -> uniform softmax, lapse uniform) -- parameter-free; I predict 0.42 (box [0.38, 0.43]), i.e., a majority of 50%-accurate endorsements pushes choice BELOW chance. C3 (A wins 0.51+0.50+0.50, a 3-0 cue sweep): my box [0.35, 0.41]; competitor box [0.50, 0.60] -- disjoint, and at center a 20-point sign-direction flip (I choose B despite A winning every discriminating cue). C3m mirrors. Any p > 0.50 on C2/C3 falsifies my anti-diagnostic sign; any reliable p != 0.50 on C2 falsifies the competitor with no free parameters.

FAMILY F/T (delta-flattening vs v_lo annihilation). F1 (A wins 0.80; B wins 0.70+0.51+0.50): the competitor's flattened weights make the 0.51 cue worth over half a 0.80 cue, so its box predicts B (center 0.41, box [0.42, 0.52] for p(A)); my w(0.51)~0 and w(0.50)<0, so I predict A (center 0.64, box [0.57, 0.69]) -- disjoint, center sign flip despite a 1-vs-3 cue count. F1m mirrors. T1 (A wins 0.90; B wins 0.51+0.50+0.50): my box [0.85, 0.92] (the weak stack is anti-diagnostic, amplifying A) vs competitor [0.55, 0.70] -- disjoint; p(T1) < 0.80 is a knife against MY theory's chance-cue distrust magnitude, included deliberately.

ANCHORS. G1/G1m (0.95+0.90+0.80+0.70 vs 0.51+0.50+0.50): both theories choose the trusted stack at high confidence (mine ~0.95, competitor ~0.89), pinning lapse/temperature. PREDICTED PATTERN IF MY THEORY IS RIGHT (center params): S1 .13, X3 .12, S1 .13, X1 .15 (ordered), P1 .31, D4 .17, D1m .93, D5 .58, D2 .04, C2 .42, C3 .38, F1 .64, T1 .88, G1 .95 -- super-expert endorsements followed inverts, chance-cue majorities inverted, 0.90 > 0.95. IF THE COMPETITOR IS RIGHT: S1 = X1 = X3 = .68 exactly, P1 .51, D4 .69, D1m .47, D5 .81, D2 .60, C2 exactly .50, C3 .58, F1 .41, T1 .57, G1 .89 -- super-experts followed, chance cues invisible, near-flat validity gradient. The boxes are disjoint on 11 of 19 cells and every family contains a parameter-free equality or exact-0.50 prediction for the competitor that my theory breaks in a signed, graded, pre-registered direction. Sides are balanced (9 A-tilted / 10 B-tilted under my theory; every family has an A/B mirror); no all-tie trials; every trial requires scanning all 8 experts.

[7] CORE LOGIC. Both theories share the identical architecture (sign votes x validity weights x softmax x lapse) and differ ONLY in the weight function. My advocated theory (FDEI) compresses diagnosticity heavily (delta in [0.1,0.2]), so among genuinely diagnostic cues its weights are nearly FLAT: w(0.98) in [1.14,1.31], w(0.90) in [1.08,1.17], w(0.75) ~ 1.01, w(0.55) in [0.69,0.85]. The competitor (SITT) has a steep inverted-U: w(0.55) ~ 0.18-0.23, w(0.75) ~ 0.68-0.74, w(0.90) ~ 0.95-1.0 (peak), w(0.98) NEGATIVE in [-1.14,-0.70]. The single most informative behavioral quantity is therefore the EXCHANGE RATE between a strong expert and weak (55%) experts, measurable as the crossover point k* in a weak-cue dose-response ladder. Validities: [0.98, 0.90, 0.75, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55] -- one expert per regime plus SIX 55% experts, enough to bracket both theories' crossovers. This is a design no prior experiment used: no v=0.5 annihilation logic ([4]), no tally-margin decorrelation ([1],[2]), no bounded-box ceiling attack ([3]), no validity-ratio ER between diagnosticity classes ([5]), and unlike [6] the flagship is not a standalone 0.99 inversion but graded crossover ladders (the 0.98 cliff cells here appear only inside coalitions).

FAMILY K (flagship crossover ladder). K1-K6: A wins the 90% expert, B wins k of the 55% experts, k=1..6. The crossover k* = w(0.90)/w(0.55) is boxed DISJOINTLY: FDEI k* in [1.27, 1.69] (crosses 0.5 between k=1 and k=2); SITT k* in [4.1, 5.6] (crosses between k=4 and k=5). Every single cell is box-disjoint. FDEI center (delta=0.15, tau=0.01, beta=0.32): p(A) = .55, .43, .32, .22, .15, .10 for k=1..6. SITT center (v_lo=.51, v_peak=.90, v_hi=.96, rho1=.7, rho2=.3, kappa_hi=1.15, beta=1.0, eps=.055): p(A) = .81, .75, .68, .59, .49, .40. Box-level: at k=1, FDEI [.53,.59] vs SITT [.75,.84]; k=2, [.42,.46] vs [.69,.79]; k=3, [.31,.42] vs [.61,.72]; k=4, [.22,.31] vs [.51,.64]; k=5, [.15,.24] vs [.42,.55]; k=6, [.10,.17] vs [.33,.45]. Two psychometric curves that cross each other in opposite directions: the observed crossover location is a direct, interpretable estimate of the strong:weak exchange rate, separated by a factor of ~3 across the two boxes. Km2/Km4/Km6 mirror the ladder for side balance.

FAMILY R (second crossover at a mid-strength reference). R1-R4: A wins the 75% expert vs k 55% experts (k=1..4); Rm3 mirrors. FDEI crossover k* = w(0.75)/w(0.55) in [1.18, 1.46]; SITT in [3.0, 4.1] -- again disjoint. Centers: FDEI .54/.42/.31/.22 vs SITT .72/.64/.55/.45. Two independent exchange-rate estimates, both disjoint, test whether the dissociation scales with reference strength as each theory's weight function demands (FDEI: ER roughly constant ~1.3; SITT: ER shrinks 4.9 -> 3.5 as reference drops from 0.90 to 0.75).

FAMILY D/C/H (cliff cells: 0.98 as anti-cue vs near-equal-cue). D1 (A wins 0.98, B wins 0.90): FDEI box [.50,.53] -- a coin flip, because compressed weights make 98% and 90% nearly exchangeable; SITT box [.04,.06] -- subjects should choose AGAINST the 98% endorsement. D1m mirrors. C1 (A wins 0.90+0.75, B wins 0.98+one 0.55): FDEI [.49,.55] (net evidence ~0); SITT [.93,.97] (the 98% endorsement is worth about one trusted ANTI-cue, so the two trusted experts plus the inversion overwhelm it). C1m mirrors. H2 (A wins 0.98 + two 55%s, B wins 0.90): FDEI [.66,.80] (weak majority plus super-expert beats the peak expert); SITT [.06,.13] (the super-expert's anti-vote sinks the coalition). H2m mirrors. These cells separate a near-tie from a 90-point swing.

HONEST KNIVES AGAINST MY OWN THEORY. K1 and R1 are the sharpest: FDEI's delta<=0.2 compression caps the 90-vs-55 exchange rate at 1.69, so it cannot predict p(A) > 0.59 on K1 or > 0.56 on R1 for ANY parameter setting. If subjects follow the strong expert at p >= 0.70 on K1, my flattening claim is falsified outright and the design doubles as a direct delta-estimation experiment (observed k* maps monotonically onto delta). Conversely K3/K4 and R2/R3 falsify SITT if subjects favor the weak stacks at all (SITT demands p(A) >= 0.51 at K4 for every parameter).

CONTROLS. X1/X2 (one 55% expert won by each side): BOTH theories must predict exactly 0.50 for every parameter -- a parameter-free side/display-bias assay. G1/G1m are agreement anchors (both theories > 0.85 at center: FDEI .94, SITT .96) pinning lapse/temperature. Tied features are a MIX of 0/0 and 1/1 ties so that 'count the 1s' is not a viable heuristic (e.g., K6 has A with three 1s vs B with eight 1s while the theories disagree only about the discriminators). No all-tie trials; every trial requires scanning all 9 experts. 24 unique pairs x K = 96//24 = 4 reps = 96 trials; sides balanced (10 A-tilted, 12 B-tilted, 2 neutral under FDEI centers; every non-anchor family contains an A/B mirror). PREDICTED PATTERN IF MY THEORY IS RIGHT: shallow FDEI psychometrics crossing 0.5 between k=1 and 2 in both ladders, coin flips on D1/C1, A-choice ~0.74 on H2, exactly 0.50 on X. IF THE COMPETITOR IS RIGHT: steep SITT psychometrics crossing between k=4 and 5 (K) and k=3 and 4 (R), near-ceiling strong-expert following at low k, ~0.05 on D1, ~0.95 on C1, ~0.07 on H2. The boxes are disjoint on 20 of 24 cells.

[8] CORE LOGIC. Both theories share the identical architecture (sign votes x validity weights x softmax x lapse) and differ ONLY in the weight function. The weight functions are disjoint at three validity regimes, and each regime admits a PARAMETER-FREE knife against exactly one theory: (1) CHANCE FLANK (v=0.50): FDEI annihilates chance experts exactly (llr=0 for every delta, tau), so any trial whose discriminators are only 50% experts forces FDEI to predict EXACTLY p=0.50; BSST gives them weight -kappa_lo in [-0.235,-0.22], predicting a small graded ANTI-diagnostic ladder (this is the held-cliff calibration signature: beta*kappa_lo ~ 0.077 logits per chance cue, deliberately modest). (2) MID-BAND PLATEAU (v in [0.55,0.95]): BSST's cliff logistic is ~1e-13 at 0.90 and 0.55, so w(0.90)=w(0.55)=w_plateau EXACTLY -- any trial with tied plateau-cue counts forces BSST to predict EXACTLY p=0.50; FDEI's compressed-but-graded weights (w(0.90) in [1.08,1.17] vs w(0.55) in [0.69,0.85]) predict a systematic tilt toward the higher-validity side. (3) CLIFF FLANK (0.98 vs 0.99): BSST has w(0.98) in [0.47,0.81] (mildly trusted) but w(0.99) in [-1.35,-0.85] (strongly ANTI-diagnostic); FDEI has w(0.98) in [1.14,1.31] and w(0.99) in [1.16,1.35] -- its delta-flattening makes 98% and 99% experts nearly EXCHANGEABLE (w99-w98 in [0.02,0.04], i.e., FDEI is locked within 1 point of 0.50 on any pure 0.98-vs-0.99 contrast). Validities (11 experts): [0.99, 0.99, 0.98, 0.98, 0.90, 0.90, 0.55, 0.55, 0.50, 0.50, 0.50]. 24 unique pairs x K=4 = 96 trials.

FAMILY S (flagship cliff sign-flip). S1 (A wins both 0.99 experts, B wins both 0.90 experts): BSST box p(A) in [0.18, 0.27] (center 0.22 -- the two 99% endorsements are worth -2.2 anti-cue units against +1.81 trusted units); FDEI box [0.50, 0.55] (center 0.52 -- flattened weights make 99% barely beat 90%). DISJOINT with a ~30-point gap and a sign disagreement about whether following super-experts is above or below chance. S2 (A wins 0.99+0.99+0.98 vs B wins 0.90+0.90+0.55): BSST [0.16, 0.27] vs FDEI [0.54, 0.60] -- FDEI says A, BSST says strongly B. Mirrors S1m/S2m control side bias.

FAMILY D (the 0.98-vs-0.99 cliff that FDEI cannot see). D1 (A wins the 0.98 expert, B wins one 0.99 expert): BSST box [0.59, 0.68] (center 0.64 -- one cliff step flips the preference); FDEI locked at [0.494, 0.497] -- essentially exactly 0.50 for EVERY parameter, because w99-w98 is at most 0.04. D2 (A wins both 0.98s, B wins both 0.99s): BSST [0.68, 0.81] (center 0.75) vs FDEI [0.49, 0.50]. A 25-point dissociation on a stimulus difference (0.98 vs 0.99) that FDEI's delta-compression renders invisible. Mirrors included.

FAMILY V (cliff specificity control -- the sharpest single contrast). V1 (A wins one 0.99 expert; B wins all three 0.50 experts): BSST box [0.435, 0.499] (center 0.467: the 99% anti-vote plus three chance endorsements won by B both push toward A losing... net s = w99 + 3*kappa_lo = -0.42); FDEI box [0.57, 0.62] (center 0.594: the most diagnostic cue vs invisible chance cues). SIGN-DISJOINT. V0 (identical but A wins the 0.98 expert instead): BSST [0.58, 0.63] (center 0.605), FDEI [0.57, 0.61] (center 0.588) -- AGREEMENT by design. The V0-vs-V1 difference isolates the cliff: FDEI must predict V1 > V0 by ~0-1 points (w99 >= w98), BSST predicts V1 - V0 = -14 points crossing 0.50. A subject-level V0-V1 difference near zero falsifies BSST's cliff; near -14 falsifies FDEI's flattening. Mirrors included.

FAMILY Z (chance-flank ladder; FDEI parameter-free knife). Z1/Z2/Z3 (A wins 1/2/3 chance experts, nothing else discriminates): FDEI must predict EXACTLY 0.50 on all three for every (delta, tau, beta, epsilon). BSST predicts a monotone anti-diagnostic ladder: 0.481 / 0.463 / 0.446 (box half-widths < 0.015 -- the tightest cells in the design, since beta is pinned in [0.33,0.35] and kappa_lo in [0.22,0.235]). Any reliable negative slope of p(A) on chance-cue count, or any reliable deviation below 0.50 on Z3, falsifies FDEI with zero free parameters; a flat 0.50 falsifies the BSST chance-flank magnitude. Z3m mirrors. Note the effect is deliberately small (~2 points/cue): the held-cliff calibration pins beta*kappa_lo ~ 0.077, so the Z family is a trend/dose-response test, while the S/D/V families carry the large-effect dissociations -- recovering BOTH signatures at their predicted relative sizes is the quantitative calibration test unique to this round.

FAMILY X (chance x diagnostic interaction; mutual knife). X0 (A wins one 0.90 expert, B wins one 0.55 expert): BSST must predict EXACTLY 0.50 (plateau weights identical to machine precision); FDEI predicts [0.52, 0.54] (center 0.527). X2/X3 add 1/3 net chance cues won by A on top of the identical diagnostic background: FDEI must predict X0 = X2 = X3 EXACTLY (chance weight is identically zero -- a parameter-free flatness knife); BSST predicts a monotone ladder 0.50 -> 0.481 -> 0.446, i.e., X3 falls BELOW 0.50 while FDEI is locked ABOVE 0.52 -- a sign-disjoint cell. The X family is a double bind: X0 alone can falsify BSST (any A-tilt), the X0=X2=X3 equality alone can falsify FDEI (any slope). Xm3 mirrors. BSST additionally predicts the cross-family equalities X2=Z1 and X3=Z3 exactly (the plateau background contributes exactly 0), an internal consistency check FDEI violates (it predicts Z*=0.50 but X*=0.527).

FAMILY P (mid-band composition; BSST parameter-free knife). P1 (A wins both 0.90 experts, B wins both 0.55 experts): tied plateau counts force BSST to EXACTLY 0.50; FDEI predicts [0.53, 0.58] (center 0.553) because its graded weights give the 0.90 side a net +0.71 evidence margin. P1m mirrors (FDEI [0.42, 0.47]). Any systematic tilt on P1/P1m falsifies BSST with no parameters; exactly 0.50 falsifies FDEI's delta-flattening claim (delta would have to be ~0).

CONTROLS. T1 (the two 0.99 experts split): BOTH theories must predict exactly 0.50 -- a parameter-free side/display-bias assay. G1/G1m (all four trusted experts vs the three chance experts) are agreement anchors (BSST ~0.77/0.23, FDEI ~0.80/0.20) pinning beta and epsilon. Tie patterns mix 0/0 and 1/1 so 'count the 1s' is not viable, and every trial has 2-7 discriminators requiring genuine scanning. Sides balanced (9 A-tilted / 11 B-tilted / 4 exactly neutral under BSST centers; near-mirror under FDEI); no all-tie trials.

DIFFERENTIATION FROM PRIOR EXPERIMENTS. Unlike [1]-[3] (count-vs-weight decorrelations) and [4] (chance annihilation vs full weighting), this design's two theories are BOTH weighted and BOTH bounded, and the dissociation is carried by the SHAPE discontinuities: the 0.98/0.99 cliff (D, V families) was never contrasted directly before ([6] used 0.99 inversions but no 0.98-vs-0.99 within-pair contrast and no V0/V1 specificity pair), and the mutual parameter-free knife architecture (BSST exact-0.50 at X0/P1 vs FDEI exact-0.50 at Z/X-flatness) is new. PREDICTED PATTERN IF BSST IS RIGHT (center params): S1 .22, S2 .21, D1 .64, D2 .75, V1 .47, V0 .61, Z1/Z2/Z3 .48/.46/.45, X0 .50, X2 .48, X3 .45, P1 .50, T1 .50, G1 .77. IF FDEI IS RIGHT: S1 .52, S2 .55, D1 .50, D2 .50, V1 .59, V0 .59, Z1=Z2=Z3 .50 exactly, X0=X2=X3 .527 exactly, P1 .55, T1 .50, G1 .80. The boxes are disjoint on 18 of 24 cells; every family contains at least one parameter-free equality or exact-0.50 prediction for one theory that the other breaks in a signed, pre-registered direction.

[9] CORE LOGIC. Both theories share the identical architecture (sign votes x validity weights x low-gain softmax x lapse) and differ ONLY in the weight function, and every validity in this design is >= 0.55, so BSST's low flank is inert and the entire dissociation is carried by two structural features never tested together: (a) BSST's FLAT PLATEAU over v in [0.55, 0.95] (all these weights equal w_plateau to within 5e-5) vs FDEI's compressed-but-strictly-MONOTONE gradient (w(0.55)=0.77 < w(0.95)=1.18 at center; the gradient is what delta > 0 exists to produce); (b) BSST's sharp 0.98/0.99 distrust cliff (w(0.98) in [0.47, 0.81], w(0.99) in [-1.35, -0.85]) vs FDEI's delta-flattened near-equality of 0.98 and 0.99 (w99 - w98 in [0.020, 0.044] across its ENTIRE box, so FDEI is locked within 0.8 probability points of invariance to a 0.98-vs-0.99 swap on any trial). Validities (10 experts): [0.99, 0.98, 0.95, 0.90, 0.85, 0.75, 0.65, 0.55, 0.55, 0.55]. 24 unique pairs x K=4 = 96 trials. Differentiation from prior experiments: no v=0.5 chance-flank logic ([4], [8]); no tally-margin decorrelation ([1]-[3]); no exchange-rate ladders against a trusted 0.90 reference ([7]); unlike [8], the cliff is tested MASKED inside balanced mid-band coalitions (where FDEI's near-invariance is parameter-free) rather than in isolated 0.98-vs-0.99 pairs, and the reciprocal PLATEAU-swap equality (invisible to BSST, visible to FDEI) is entirely new -- giving a double dissociation of SENSITIVITY ORDERING.

FAMILY V (plateau validity psychometric -- delta estimation). V1/V2/V3: A wins {0.95, v} with v = 0.65/0.75/0.85, B wins two 0.55s; V4: A wins {0.95, 0.55}, B wins two 0.55s. BSST must predict EXACTLY p(A)=0.50 on all four (every discriminator sits on the plateau, counts are 2-vs-2 tied, uniform softmax + uniform lapse) -- a parameter-free flatness knife. FDEI predicts a strictly monotone ladder in v (its w is strictly increasing for any delta > 0): centers .564 (V4), .588 (V1), .601 (V2), .612 (V3); box ranges e.g. V1 [.553,.626], V3 [.567,.662], every rung strictly above .50 across the whole box. The ladder slope is a direct delta estimate: log-odds = 2*beta*(llr(v)-tau)^delta regressed on log(llr-tau). Flat 0.50 across the ladder falsifies FDEI's delta in [0.1,0.2] outright; any single rung above ~.55 falsifies BSST with zero free parameters. Mirrors V1m/V3m control side bias.

FAMILY R (reference-validity crossover ladder -- sign dissociation and reference-ordering reversal). A wins one reference cue r in {0.95, 0.98, 0.99}, B wins k in {1,2} cues of 0.55. R1 (r=.95, k=1): BSST exactly 0.50 (plateau tie); FDEI .564 [.538,.594]. R2 (r=.98, k=1): FDEI .573 [.542,.607] vs BSST .457 [.420,.488] -- SIGN-DISJOINT around 0.50: FDEI says one 98% expert beats one 55% expert, BSST says it loses. R5 (r=.99, k=1): FDEI .577 [.545,.615] vs BSST .204 [.192,.265 with lapse] -- a 30+ point sign flip. R3/R4 (k=2): FDEI [.399,.472]/[.405,.482] vs BSST [.342,.359]/[.274,.348] -- disjoint. R6 (r=.99, k=2): FDEI [.408,.488] vs BSST [.096,.152] -- disjoint. Crossover structure: FDEI crosses 0.5 between k=1 and k=2 for EVERY reference; BSST never exceeds 0.50 for ANY reference at ANY k (its 0.98/0.99 weights are below the plateau). REFERENCE-ORDERING REVERSAL (quantitative knife both ways): R2 vs R1 and R4 vs R3 compare 0.98 vs 0.95 references at fixed k. FDEI predicts R2 > R1 and R4 > R3 STRICTLY for every parameter (llr(0.98) > llr(0.95) always), but by at most ~1 point (w98-w95 <= 0.079). BSST predicts R2 < R1 and R4 < R3 STRICTLY for every parameter (w98 <= 0.81 < plateau) -- by 2-8 points at k=1 and 4-9 points at k=2. The SIGN of the 0.95-vs-0.98 comparison separates the entire boxes of both theories. Mirrors R2m/R4m/R5m balance sides.

FAMILY M/P (masked cliff -- the sensitivity-ordering double dissociation, flagship). All cells share the balanced mid-band background A/B = {0.75, 0.55} vs {0.90, 0.85, 0.65} and embed one super cue in A's coalition. M98 (A adds 0.98): FDEI .482 (box [.475,.487] -- tightly pinned); BSST .457. M99 (A adds 0.99): FDEI .486, box [.482,.489]; BSST .204 [.192,.265]. CLIFF SWAP (M98 -> M99, a 1-point validity change): FDEI predicts |delta p| <= 0.8 points for EVERY (delta, tau, beta, epsilon) in its box (w99-w98 <= 0.044, 2*beta*that <= 0.033 logits) -- a parameter-free near-equality; BSST predicts a drop of 16-31 points. PLATEAU SWAPS: P95 (A adds 0.99+0.95+0.55... i.e., coalition {0.99, 0.95, 0.55}) and P55 (coalition {0.99, 0.55, 0.55}) differ by swapping a 0.95 for a 0.55 -- a 40-point validity change. BSST must predict P95 = P55 = M99 EXACTLY (all reduce to margin w99 - w_plateau; plateau swaps are invisible) -- parameter-free triple equality. FDEI predicts P95 (.512, box [.504,.524]) > M99 (.486) > P55 (.448, box [.429,.466]), a 4-9 point swing per swap, with P95 and P55 on OPPOSITE sides of 0.50... P55 box entirely below .466 vs BSST entirely below .265 (disjoint), P95 box entirely above .504 vs BSST entirely below .265 (disjoint, sign flip). The headline quantitative signature is the 2x2 of swap sensitivity: cliff swap (0.98->0.99): FDEI ~0 pts, BSST -16 to -31 pts; plateau swap (0.95->0.55): FDEI -4 to -9 pts, BSST exactly 0 pts. Each theory is parameter-free-insensitive to exactly the manipulation the other finds huge. Mirrors M99m/P55m balance sides.

HONEST KNIVES AGAINST MY OWN THEORY (FDEI). (i) R2: FDEI's box ceiling is .607 -- if subjects follow a lone 98% expert over a 55% expert at p >= 0.61, my flattening claim (delta <= 0.2 caps the 98:55 exchange rate at 1.81) is falsified outright. (ii) The V ladder: flat 0.50 kills delta > 0. (iii) M99 vs M98: any reliable cliff-sized drop inside balanced coalitions falsifies w99-w98 <= 0.044. (iv) G1 anchor (A wins 0.95/0.90/0.85/0.75/0.65 vs three 0.55s): FDEI box [.811,.918] core, BSST [.761,.786] -- p(G1) <= 0.77 strains FDEI, p(G1) >= 0.80 exceeds BSST's ceiling. CONTROLS: X1 (one 0.55 won by each side, everything else tied): BOTH theories must predict exactly 0.50 for every parameter -- a side/display-bias assay. R1 vs V4 is a surface-invariance check both theories demand (identical evidence w95-w55, different surface: 1-vs-1 vs 2-vs-2 discriminators) -- inequality would indicate count-based heuristics outside both models. Tie patterns mix 0/0 and 1/1 so 'count the 1s' is unviable (e.g., M98 shows A with five 1s vs B with three 1s while the theories nearly tie; G1m shows B with six 1s vs A with three). No all-tie trials; every trial has 2-7 discriminators among 10 experts requiring genuine scanning. Sides balanced (10 A-tilted / 9 B-tilted / 5 near-neutral under FDEI centers; every non-anchor family contains an A/B mirror). PREDICTED PATTERN IF FDEI IS RIGHT (centers): R1 .564, R2 .573, R3 .441, R4 .450, R5 .577, R6 .454, V4<V1<V2<V3 = .564/.588/.601/.612, M98 .482 = M99 .486, P95 .512 > P55 .448, G1 .872, X1 .500 -- a rising plateau psychometric, near-zero cliff sensitivity, several-points plateau sensitivity, crossovers between k=1 and k=2. IF BSST IS RIGHT: R1/V1-V4 exactly .500, R2 .457, R3 .351, R4 .313, R5 .204, R6 .122, M98 .457 >> M99 = P95 = P55 = .204 exactly, G1 .774, X1 .500 -- flat plateau psychometric, zero plateau sensitivity, massive cliff sensitivity, never following a 0.98/0.99 expert. The boxes are disjoint or parameter-free-separated on 16 of 24 cells, and the two sensitivity orderings are mutually exclusive.

[10] CORE LOGIC. Validities are fixed for the whole experiment, so both theories reduce to a FIXED weight vector over the 10 experts, and the design's job is to make the two weight vectors maximally anti-correlated. The panel [0.99, 0.90, 0.80, 0.70, 0.60, 0.51, 0.50, 0.50, 0.50, 0.50] is engineered to push CREDT into its most distinctive regime, never instantiated in any prior experiment: (a) the 0.99 claim is an above-ceiling outlier with gap = 0.09 over the 0.90 second rung (sig_gap ~ 0.96-0.98, so the consensus gate is ~open); (b) the panel is CHANCE-LADEN: 5 of 10 experts sit below v_lo, so c_frac = 0.5 > c_half for the entire box, firing the composition amplifier (C ~ 1.0-1.3). Result: CREDT's weight on the 0.99 expert is w_top in [-1.93, -0.96] across its ENTIRE box (composition-amplified collapse), while the gate simultaneously pushes the chance flank to POSITIVE near-plateau weight: w(0.50) in [0.86, 0.91], w(0.51) ~ [0.87, 0.91], and the plateau experts (0.60-0.90) all get exactly w_p in [0.88, 0.92]. So on this panel CREDT is approximately 'count everybody's endorsement equally, EXCEPT the 99% expert, whose endorsement counts strongly AGAINST'. FDEI is the exact structural opposite: monotone graded weights with w(0.50) = 0 EXACTLY for every parameter, w(0.51) in [0, 0.73], and w(0.99) the LARGEST weight, in [1.17, 1.36]. Critically, FDEI is exactly invariant to the validities of tied cues and to panel composition, so the entire panel-structure manipulation is invisible to it except through which cues vote -- the structural source of every dissociation below. 24 unique pairs x K = 96//24 = 4 = 96 trials.

FAMILY Z (chance-only ladder; FDEI parameter-free knife). Z1/Z2/Z3: only 50% experts discriminate (A wins 1/2/3 of them); Z2m mirrors. FDEI must predict EXACTLY p = 0.50 on all four for every (delta, tau, beta, epsilon) -- all attending weights are identically zero. CREDT predicts a steep positive ladder: Z1 in [0.66, 0.68], Z2 in [0.79, 0.81], Z3 in [0.87, 0.90] across its whole box. Any reliable deviation from 0.50 on any Z trial falsifies FDEI outright with zero free parameters; a flat 0.50 falsifies CREDT's consensus-gating claim (which here flips chance cues from anti-diagnostic to near-fully trusted -- the OPPOSITE sign from the chance flank tested in [8], whose panel had c_frac = 0.27, below the amplifier threshold).

FAMILY X (90% expert vs k chance cues; FDEI flatness knife + sign flip). X1/X2/X3: B wins the 0.90 expert, A wins 1/2/3 chance cues; X3m mirrors. FDEI must predict X1 = X2 = X3 EXACTLY (chance weight is identically zero), with common value in [0.28, 0.37]. CREDT predicts a monotone ladder crossing 0.5: X1 in [0.49, 0.51], X2 in [0.65, 0.68], X3 in [0.78, 0.82]. X2 and X3 are box-disjoint SIGN FLIPS: 'does a 90% expert beat two coin-flippers?' CREDT: no (the coin-flippers win); FDEI: yes, strongly. Double bind: any X-ladder slope falsifies FDEI parameter-free; any X2 < 0.5 falsifies CREDT.

FAMILY S/V (0.99 inversion). S1 (A wins only the 0.99 expert): CREDT [0.19, 0.32] vs FDEI [0.64, 0.73] -- sign-disjoint. S1m mirrors. S2 (0.99 for A vs 0.90 for B): CREDT [0.11, 0.20] vs FDEI [0.51, 0.54]. V1 (0.99 for A vs one 50% for B): CREDT [0.11, 0.20] vs FDEI [0.64, 0.73] -- the single sharpest sign flip: FDEI says the 99% expert crushes a coin-flipper; CREDT says both point the same way, toward B.

FAMILY K (flagship: four trusted vs four coin-flip). K1: A wins the 0.90/0.80/0.70/0.60 experts, B wins all four 50% experts. CREDT: s = 4*w_p - 4*w(0.5) ~ 0, p in [0.48, 0.56] -- a COIN FLIP, because gated chance cues carry the same weight as plateau experts. FDEI: p in [0.88, 0.95]. A 35-45 point dissociation on a unanimous-expertise trial. K1m mirrors. p(K1) >= 0.60 falsifies CREDT outright; p(K1) <= 0.70 falsifies FDEI outright.

FAMILY P (plateau flatness; CREDT parameter-free knife). P2/P2m: A wins {0.90, 0.80} vs B wins {0.70, 0.60} (and mirror). All four discriminators sit on CREDT's plateau with tied counts, so CREDT must predict EXACTLY 0.50 for every parameter. FDEI's graded weights predict a tilt: P2 in [0.53, 0.58], P2m in [0.42, 0.47]. Any systematic tilt falsifies CREDT's plateau with no parameters; exact 0.50 falsifies FDEI's delta > 0. P4/P4m (0.90 vs 0.51 head-to-head): CREDT [0.49, 0.51] (gated 0.51 ~ plateau weight) vs FDEI [0.55, 0.69] / [0.31, 0.45] -- disjoint: CREDT says a 51% expert is exchangeable with a 90% expert on this panel; FDEI says the 90% expert clearly wins.

FAMILY W (honest knives against CREDT). W1 (0.51 for A vs 0.50 for B): CREDT is capped at 0.513 across its entire box; FDEI allows up to 0.63. If subjects distinguish 51% from 50% at p > 0.52, CREDT's gated-equalization claim is falsified. W2 (0.51 for A vs two 0.50 for B): CREDT [0.32, 0.35] vs FDEI [0.50, 0.63] -- disjoint; W2m mirrors.

ANCHORS/CONTROLS. G3/G3m (five sub-ceity experts vs one coin-flipper) are agreement anchors (CREDT [0.92, 0.95], FDEI [0.90, 0.97]) pinning lapse/temperature. N1 (one 50% expert won by each side): BOTH theories must predict exactly 0.50 for every parameter -- a side/display-bias assay; note N1 also defeats a 'count the 1s' heuristic (B has more 1s), as do V1 and W2 where raw-1 counts favor A while both theories' evidence favors B. Tie patterns mix 0/0 and 1/1 throughout so count-based heuristics are unviable; no all-tie trials; every trial requires scanning all 10 experts; sides are balanced (7 A-tilted / 6 B-tilted / near-neutral remainder under CREDT centers; 9/10 under FDEI).

DIFFERENTIATION FROM PRIOR EXPERIMENTS. No prior design instantiated the composition-amplifier regime (c_frac = 0.5 > c_half; [6] had 2/8, [8] 3/11, [9] 0/10 chance cues), and no prior design tested the CONJUNCTION it produces: 0.99 anti-diagnostic AND chance cues at near-full POSITIVE weight AND a flat plateau, on one fixed panel. [4] tested chance-cue weighting vs annihilation but with a monotone advocated theory and no outlier inversion; [6]-[9] tested 0.99 cliffs with chance cues negative or absent. The K1 cell (unanimous trusted panel vs equal-count coin-flip panel -> CREDT coin flip, FDEI 0.90) and the X-family sign crossover are new quantitative signatures. PREDICTED PATTERN IF CREDT IS RIGHT (centers): Z .67/.80/.87, X .50/.66/.79, S1 .23, S1m .77, S2 .14, V1 .14, P2 = P2m = 0.50 exact, P4 ~ .50, W1 ~ .51, W2 .32, K1 .51, G3 .92, N1 0.50. IF FDEI IS RIGHT: Z = 0.50 exactly (all four), X flat at .33, S1 .69, S1m .31, S2 .52, V1 .69, P2 .55, P4 .58, W1 .58, W2 .58, K1 .90, G3 .93, N1 0.50. The boxes are disjoint on 14 of 24 cells and every family contains a parameter-free exact-equality prediction for one theory that the other breaks in a signed, graded, pre-registered direction.

[11] CORE LOGIC. Both theories share the identical architecture (sign votes x validity weights x low-gain softmax x lapse) and differ ONLY in the weight function, but the weight functions are near-orthogonal on this panel. The panel [0.99, 0.94, 0.85, 0.75, 0.65, 0.55, 0.50, 0.50] is engineered to place CREDT in its PEAKED EDGE-DISTRUST regime, never instantiated in any prior experiment: the 0.99 claim sits exactly 0.05 above the 0.94 second rung -- the gap that CREDT's non-monotone edge-distrust bump P treats as MAXIMAL spurious precision (g_star ~ 0.048) -- while the panel is coherent (c_frac = 2/8 = 0.25 < c_half, so the composition amplifier contributes only C ~ 0.09-0.26 rather than firing). Resulting CREDT weights across its ENTIRE box: w(0.99) in [-1.04, -0.59] (strongly ANTI-diagnostic), w(0.94)...w(0.55) all exactly w_p in [0.88, 0.92] (flat plateau), and the two 0.50 experts GATED to positive weight (gate = lambda_ext + (1-lambda_ext)*sig_gap ~ 0.73-0.82 open, since the outlier's sig_gap ~ 0.59-0.68), giving w(0.50) in [0.62, 0.69]. FDEI weights across its entire box: w(0.99) in [1.16, 1.36] (the LARGEST weight), w(0.94) in [1.11, 1.23], w(0.85) in [1.06, 1.12], w(0.75) in [1.01, 1.02], w(0.65) in [0.91, 0.95], w(0.55) in [0.69, 0.85], w(0.50) = 0 EXACTLY for every (delta, tau, beta, epsilon). The two weight vectors are almost perfectly anti-correlated: FDEI's best cue is CREDT's worst cue; FDEI's invisible cues are worth ~70% of a plateau cue to CREDT; CREDT's plateau is flat where FDEI is (weakly) graded. 24 unique pairs x K = 96//24 = 4 = 96 trials.

FAMILY S/W (flagship: spurious-precision inversion + parameter-free chance-cue flatness ladder). S1 (A wins only the 0.99 expert): FDEI box p(A) in [0.64, 0.73] (center 0.69); CREDT box [0.30, 0.39] (center 0.35) -- SIGN-DISJOINT with a 25-point gap: FDEI says follow the 99% expert, CREDT says a 0.99-atop-0.94 claim is spurious precision and its endorsement counts AGAINST. S1m mirrors. S2 (A wins 0.99, B wins 0.94): FDEI's delta-flattening locks w99-w94 in [0.05, 0.14], so its ENTIRE box spans only p(A) in [0.509, 0.524] -- a near-parameter-free coin flip; CREDT box [0.17, 0.25] (center 0.21). A 28-point dissociation on a cell where FDEI is pinned within 1.5 points of 0.50. THE LADDER: S4 (B also wins one 0.50) / S2 / S3 (A also wins one 0.50) / W2 (A also wins both 0.50s) holds the 0.99-vs-0.94 diagnostic background fixed while the net chance-cue count for A runs -1, 0, +1, +2. FDEI must predict S4 = S2 = S3 = W2 EXACTLY for every parameter (chance weight is identically zero) -- a parameter-free flatness knife. CREDT predicts a steep monotone ladder: 0.141 / 0.209 / 0.305 / 0.425 at center (boxes [0.11,0.17] / [0.17,0.25] / [0.26,0.35] / [0.36,0.48]) -- a ~30-point climb that directly measures the gated chance-cue weight (~0.65) against the outlier distrust (~-0.79) in one psychometric. W2m mirrors. Any reliable slope across the ladder falsifies FDEI with zero free parameters; flatness at 0.51 falsifies CREDT's consensus-gating claim.

FAMILY V (outlier vs chance, and the S1=V1 flatness knife). V1 (A wins 0.99, B wins one 0.50): FDEI [0.64, 0.73] (identical to S1 -- FDEI must predict V1 = S1 EXACTLY, since chance cues are invisible); CREDT [0.20, 0.28] (center 0.24: BOTH the 99% anti-vote and the gated coin-flipper point toward B). V1m mirrors. CREDT predicts V1 < S1 by ~10 points (the chance cue helps B); FDEI demands exact equality -- a second parameter-free knife, and V1 is a sign-disjoint cell in its own right.

FAMILY Z (chance-only ladder; FDEI parameter-free knife). Z1/Z2 (A wins 1/2 of the 0.50 experts, nothing else discriminates): FDEI must predict EXACTLY 0.50 on both (all weights zero -> uniform softmax, lapse uniform) for every parameter. CREDT predicts a steep positive ladder: Z1 in [0.62, 0.74] (center 0.63), Z2 in [0.72, 0.76] (center 0.74). Z2m mirrors. Any reliable deviation from 0.50 falsifies FDEI outright; a flat 0.50 falsifies CREDT's gated-trust claim. Note the sign here is POSITIVE (chance cues trusted), the OPPOSITE of the ungated negative flank -- on this panel the gate is partially open, which is CREDT's distinctive conjunction: distrust the 0.99 outlier AND simultaneously count the coin-flippers.

FAMILY X (plateau expert vs gated chance cues; mutual sign flip and mutual knife). X1 (A wins one 0.50, B wins the 0.55 expert): FDEI [0.35, 0.41] (center 0.38); CREDT [0.435, 0.465] (center 0.45). X2 (A wins both 0.50s, B wins 0.55): FDEI must predict X2 = X1 EXACTLY (parameter-free flatness; center 0.38); CREDT predicts a SIGN FLIP to [0.56, 0.60] (center 0.58) -- two coin-flippers beat a 55% expert under gating. X2m mirrors. X3 (A wins both 0.50s, B wins the 0.75 expert): CREDT must predict X3 = X2 EXACTLY (plateau weights identical to machine precision) -- its parameter-free flatness knife; FDEI predicts X3 < X2 (0.35 vs 0.38, its graded 0.75-vs-0.55 weights). The X family is a double bind: the X1=X2 equality falsifies FDEI if broken; the X2=X3 equality falsifies CREDT if broken; and the X2/X3 sign (FDEI [0.33,0.41] vs CREDT [0.56,0.60]) is box-disjoint in the opposite direction from X1.

FAMILY Q/P (FDEI's within-plateau gradient vs CREDT's flat plateau; CREDT parameter-free knife). Q1 (A wins 0.94, B wins 0.55; 1-vs-1): CREDT must predict EXACTLY 0.50 (both on the plateau, tied counts) for every parameter; FDEI predicts [0.535, 0.597] (center 0.56) because its compressed-but-graded weights give the 0.94 side a net +0.27 to +0.53 evidence margin. Q1m mirrors. P1 (A wins {0.94, 0.85}, B wins {0.65, 0.55}; 2-vs-2): CREDT exactly 0.50; FDEI [0.549, 0.635] (center 0.58) -- its strongest plateau-tilt cell. P1m mirrors. Any systematic tilt on Q1/P1 falsifies CREDT's plateau with zero free parameters; exact 0.50 across the family falsifies FDEI's delta > 0 gradient. The Q/P family is deliberately modest in size (5-8 points) -- recovering BOTH this small graded tilt AND the huge S/Z effects at their predicted relative sizes is the quantitative calibration test.

FAMILY H (masked inversion inside a balanced 3-vs-3 coalition). H3 (A wins {0.99, 0.85, 0.75}, B wins {0.94, 0.65, 0.55}; raw cue count tied 3-3, raw 1-count tied): FDEI margin is remarkably delta-stable (0.48-0.51 across its whole box), locking p(A) in [0.575, 0.594] (center 0.58); CREDT margin = w_top - w_p, giving [0.17, 0.25] (center 0.21) -- a 35-point SIGN FLIP on a count-balanced trial. H3m mirrors. This cell defeats any count-based heuristic (counts are exactly tied) and isolates pure weighting.

ANCHOR. G1 (A wins all five plateau experts, B wins both 0.50s): both theories choose A at high confidence (FDEI [0.92, 0.97], center 0.95; CREDT [0.90, 0.93], center 0.91), pinning lapse/temperature; G1m-equivalent structure is carried by the mirrors.

DIFFERENTIATION FROM PRIOR EXPERIMENTS. No prior design instantiated the g_star PEAK regime of CREDT's edge-distrust bump: [10] used a 0.99-atop-0.90 panel (gap 0.09, large-edge branch) that was CHANCE-LADEN (c_frac = 0.5, amplifier firing, chance cues at 0.86-0.91); here gap = 0.05 with a coherent panel (c_frac = 0.25, amplifier sub-threshold, chance cues at 0.62-0.69, w_top shallower at -0.6 to -1.0). The S4->S2->S3->W2 ladder measuring gated chance weight against peaked outlier distrust in one psychometric is new, as is the FDEI-locked near-tie on 0.99-vs-0.94 (its entire box within 1.5 points of 0.50) and the S1=V1 and X1=X2 flatness knives. [6]-[9] tested FDEI against other competitors (SITT/BSST), not CREDT. PREDICTED PATTERN IF FDEI IS RIGHT (center delta=0.15, tau=0.01, beta=0.32, eps=0.025): S1 .69, S2=S3=S4=W2 .51 exactly, V1 .69 (=S1), Z1=Z2 .50 exactly, X1=X2 .38, X3 .35, Q1 .56, P1 .58, H3 .58, G1 .95. IF CREDT IS RIGHT (center w_p=.90, rho=1.36, beta=.42, eps=.05): S1 .35, S4 .14 < S2 .21 < S3 .31 < W2 .43, V1 .24, Z1 .63, Z2 .74, X1 .45, X2=X3 .58 exactly, Q1=P1 .50 exactly, H3 .21, G1 .91. The boxes are disjoint on 14 of 24 cells, sign-disjoint on 6 (S1, S2, V1, X2, X3, H3), and every family contains a parameter-free exact-equality or exact-0.50 prediction for one theory that the other breaks in a signed, graded, pre-registered direction. Sides are balanced (7 A-tilted / 7 B-tilted / 10 near-neutral under FDEI centers; 8/11/5 under CREDT centers; every family contains an A/B mirror); no all-tie trials; every trial requires scanning all 8 experts; tie patterns mix 0/0 and 1/1 so 'count the 1s' is unviable (e.g., H3 has exactly tied 1-counts while the theories disagree by 35 points, and X2/X3 have MORE 1s for A while FDEI favors B).

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "CORE LOGIC. CREDT (advocated) and BODT (competing) share the identical architecture (sign votes x panel-relative weights x low-gain softmax x lapse) and agree on the plateau (w_p), on the trust ceiling (~0.981), and on the consensus-gated chance flank (identical g0/s_gap/lambda_ext machinery). They differ structurally in exactly one place: the DEPTH of distrust of an above-ceiling outlier. CREDT's discount is an UNBOUNDED additive term rho*(P + C*sig_gap); BODT's is a BOUNDED saturating shrinkage toward w_floor (~-0.28) plus a composition route that is multiplied by far(gap), a logistic gate its own description commits to being 'essentially closed at gap ~0.05'. The design therefore engineers the single panel configuration that maximally separates these two formulas: a 0.99 claim sitting exactly 0.05 above its next rung (gap = g_star, CREDT's PEAKED spurious-precision bump P ~ 1) on a CHANCE-LADEN panel (c_frac = 5/9 = 0.556 > both theories' c_half), with TWO tied 0.99 experts so the lever is applied twice. Panel: [0.99, 0.99, 0.94, 0.75, 0.50, 0.50, 0.50, 0.50, 0.50]. On this panel CREDT's composition amplifier fires fully at the peak (C ~ 1.24, sig_gap ~ 0.63, no far-gate), giving w_top in [-2.03, -1.36] across its ENTIRE parameter box -- each 99% expert is worth roughly -1.7, i.e., about two trusted cues AGAINST. BODT's bounded shrinkage caps w_top_mod at w_floor ~ -0.28 and its composition route is strangled by far(0.05) in [0.06, 0.19], giving w_top in [-0.40, +0.09] across its ENTIRE box. THE TWO w_top BOXES ARE DISJOINT with a gap of ~1.0-1.6 weight units per 99% expert, while every other weight is common currency: w(0.94) = w(0.75) = w_p exactly in both, and the gated chance weight is 0.657 (CREDT) vs 0.641 (BODT) -- the kappa_lo difference (0.20 vs 0.27) is suppressed to ~0.016 per cue by the ~0.78-open consensus gate, which is itself nearly identical across boxes. The design thus isolates ONE structural disagreement -- how much a 99%-atop-94% claim on a coin-flip-laden panel is distrusted -- and measures it psychometrically. 24 unique pairs x K = 96//24 = 4 = 96 trials.\n\nFAMILY S (flagship: the 0.99 PAIR vs varying counterweights). S1 (A wins both 99%s + 0.94 + 0.75; B wins two coin-flippers): CREDT p(A) = .24 (box [.19, .30]); BODT p(A) = .55 (box [.47, .58]) -- SIGN-DISJOINT. S6 (same A-stack vs ONE coin-flipper): CREDT .29 [.23, .35] vs BODT .57 [.52, .63] -- sign-disjoint. S2 (both 99%s vs 0.94+0.75): CREDT .12 [.09, .19] vs BODT .34 [.26, .35]. S3, S4, K1 extend the counterweight ladder; K1 is the unanimity cell (A wins ALL five real experts incl. both 99%s; B wins four coin-flippers): CREDT .20 [.14, .24] vs BODT .47 [.39, .51] -- even BODT's near-tie here is diagnostic, since its bounded floor cannot push 2*w_top below -0.80, so a subject p(A) <= 0.35 on K1 falsifies BODT outright, while p(A) >= 0.30 falsifies CREDT outright. Mirrors S1m/S2m/S4m/S6m control side bias.\n\nFAMILY R (single-0.99 crossover psychometric -- the key quantitative signature). A wins one 99% expert plus j coin-flippers; B wins the 0.94 expert; j = 0..4 (R0-R4). CREDT centers: .26/.31/.38/.44/.50; BODT centers: .41/.47/.55/.57/.62. The psychometric CROSSES p=0.5 at j* ~ 3.9 for CREDT but j* ~ 1.6 for BODT -- a ~2.3-coin-flip separation in crossover location, equivalent to ~1.3 evidence units, that no parameter setting in either box can close (CREDT's box at R3 is [.38, .47], BODT's is [.53, .62] -- disjoint; at R2 they are edge-disjoint [.32, .41] vs [.48, .55]). R3 is the sharpest single cell: one 99% + three coin-flippers vs one 94% expert -- CREDT says B, BODT says A, boxes disjoint. R3m mirrors.\n\nFAMILY D/V (0.99 vs 0.94 / coin-flippers head-to-head). D2 (both 99%s vs 0.94 + two coin-flippers): CREDT .11 [.07, .12] vs BODT .31 [.22, .34]. V1 (one 99% vs two coin-flippers): CREDT .21 [.19, .25] vs BODT .38 [.33, .41]. Mirrors included. These are the honest ceiling probes: if subjects follow a double-99% endorsement over a 94% + 75% panel at p >= 0.27 (S2) or p(A) >= 0.30 on K1, CREDT is falsified with zero free parameters (its w_top cannot exceed -1.36); conversely if subjects abandon the 99%-endorsed option at p >= 0.47 wherever BODT demands >= 0.50 (S1) or at p <= 0.42 on R2/R3, BODT's bounded-floor claim is falsified outright.\n\nCONTROLS. T1 (the two 99% experts split between A and B): BOTH theories must predict EXACTLY 0.50 for every parameter (tied softmax is uniform, lapse injects uniform) -- a parameter-free side/display-bias assay. Z1 (three coin-flippers vs two, nothing else discriminates): both theories agree (.57 vs .55), anchoring the common gated-chance currency -- a deviation here indicts the SHARED gating machinery, not the disputed distrust depth. G1/G1m (both trusted experts + three coin-flippers vs both 99%s + two coin-flippers): agreement anchors on choice whose CONFIDENCE also dissociates -- CREDT [.90, .95] vs BODT [.68, .76], disjoint boxes even on the anchor, because the 99%s on B's side actively help A under CREDT but are merely discounted under BODT. Tie patterns mix 0/0 and 1/1 throughout so 'count the 1s' is unviable (e.g., S1 has MORE 1s for A while CREDT predicts A loses by 4 evidence units; K1 has 5-vs-4 ones for A while both theories favor B; R0/V1/T1 have exactly tied 1-counts). No all-tie trials; every trial has 3-8 discriminators among 9 experts requiring genuine scanning.\n\nKEY QUANTITATIVE METRICS: (i) crossover location j* in the R-ladder (CREDT ~3.9 vs BODT ~1.6); (ii) p(A) on the sign-flip cells S1/S6/R2/R3 (CREDT .24/.29/.38/.44 vs BODT .55/.57/.55/.57); (iii) the S-family counterweight slope (how much opposition the 99% endorsement needs before subjects abandon it); (iv) anchor confidence on G1 (0.90 vs 0.71 at centers, disjoint boxes); (v) the K1 unanimity cell. PREDICTED PATTERN IF CREDT IS RIGHT (centers): S1 .24, S2 .12, S3 .18, S4 .15, S6 .29, K1 .20, R0-R4 .26/.31/.38/.44/.50, D2 .11, V1 .21, Z1 .57, T1 .50, G1 .92 -- subjects treat each 99%-atop-94% endorsement on this coin-flip-laden panel as ~2 trusted cues AGAINST, crossing over only when ~4 coin-flippers back it. IF BODT IS RIGHT: S1 .55, S2 .34, S3 .43, S4 .39, S6 .57, K1 .47, R0-R4 .41/.47/.55/.57/.62, D2 .31, V1 .38, Z1 .55, T1 .50, G1 .71 -- the same endorsement is worth only ~0.14 cues against (near-zero), crossovers at ~1.6 coin-flippers, and near-tie on the unanimity cell. The w_top boxes are disjoint by construction, so the two predicted patterns are mutually exclusive essentially everywhere. DIFFERENTIATION FROM PRIOR EXPERIMENTS: [10] instantiated the large-edge (gap 0.09) chance-laden regime against a monotone competitor and [11] the gap-0.05 COHERENT regime (c_frac 0.25, amplifier sub-threshold); no prior design instantiated the CONJUNCTION of the peaked bump (gap = g_star) with a composition-amplified chance-laden panel -- CREDT's absolute maximum-distrust configuration, where its additive discount (-1.7 per expert) and BODT's far-gated bounded floor (-0.14 per expert) are separated by more than an entire plateau weight -- and no prior round pitted CREDT against BODT at all. The tied 0.99 PAIR (doubling the lever), the R-ladder crossover psychometric, and the disjoint anchor-confidence boxes are new quantitative signatures. Sides are balanced via mirrors in every family (8 A-tilted / 13 B-tilted / 3 near-neutral under CREDT centers; near-balanced under BODT centers, reflecting that the B-tilt is itself CREDT's signature prediction).",
  "validities": [
    0.99,
    0.99,
    0.94,
    0.75,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5
  ],
  "trial_a_ratings": [
    [
      1,
      1,
      1,
      1,
      0,
      0,
      1,
      1,
      1
    ],
    [
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1
    ],
    [
      1,
      1,
      0,
      0,
      1,
      1,
      1,
      0,
      0
    ],
    [
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      0,
      0
    ],
    [
      1,
      1,
      0,
      0,
      1,
      1,
      1,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      1
    ],
    [
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      0,
      1
    ],
    [
      1,
      1,
      1,
      1,
      0,
      0,
      1,
      0,
      1
    ],
    [
      0,
      0,
      0,
      0,
      1,
      0,
      1,
      0,
      1
    ],
    [
      1,
      0,
      0,
      1,
      1,
      0,
      1,
      0,
      1
    ],
    [
      1,
      0,
      0,
      1,
      1,
      0,
      1,
      0,
      1
    ],
    [
      1,
      0,
      0,
      1,
      1,
      1,
      0,
      1,
      0
    ],
    [
      1,
      1,
      0,
      0,
      1,
      1,
      1,
      1,
      0
    ],
    [
      1,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      0
    ],
    [
      0,
      1,
      1,
      0,
      0,
      0,
      0,
      1,
      0
    ],
    [
      1,
      1,
      0,
      1,
      0,
      0,
      0,
      1,
      0
    ],
    [
      0,
      0,
      1,
      1,
      1,
      1,
      0,
      1,
      0
    ],
    [
      1,
      1,
      0,
      1,
      0,
      0,
      0,
      1,
      0
    ],
    [
      0,
      1,
      0,
      1,
      1,
      1,
      0,
      1,
      0
    ],
    [
      1,
      0,
      1,
      0,
      1,
      1,
      1,
      0,
      0
    ],
    [
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1
    ],
    [
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      0,
      0
    ],
    [
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      1,
      1
    ],
    [
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1
    ],
    [
      1,
      1,
      1,
      1,
      0,
      0,
      1,
      1,
      1
    ],
    [
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      0,
      0
    ],
    [
      1,
      1,
      0,
      0,
      1,
      1,
      1,
      0,
      0
    ],
    [
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      1,
      0
    ],
    [
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      0,
      1
    ],
    [
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      1
    ],
    [
      0,
      0,
      0,
      0,
      1,
      0,
      1,
      0,
      1
    ],
    [
      1,
      1,
      1,
      1,
      0,
      0,
      1,
      0,
      1
    ],
    [
      0,
      0,
      1,
      1,
      1,
      0,
      1,
      0,
      1
    ],
    [
      0,
      0,
      1,
      1,
      0,
      0,
      1,
      0,
      1
    ],
    [
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      1,
      0
    ],
    [
      0,
      1,
      1,
      0,
      0,
      0,
      0,
      1,
      0
    ],
    [
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      0,
      0,
      1,
      1,
      1,
      1,
      0
    ],
    [
      0,
      0,
      1,
      1,
      1,
      1,
      0,
      1,
      0
    ],
    [
      1,
      1,
      0,
      1,
      0,
      0,
      0,
      1,
      0
    ],
    [
      0,
      1,
      0,
      1,
      1,
      1,
      0,
      1,
      0
    ],
    [
      1,
      1,
      0,
      1,
      0,
      0,
      0,
      1,
      0
    ],
    [
      1,
      0,
      1,
      0,
      0,
      0,
      0,
      1,
      1
    ],
    [
      0,
      1,
      1,
      0,
      1,
      0,
      1,
      0,
      1
    ],
    [
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      1,
      1
    ],
    [
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 28587,
  "output_tokens": 50669,
  "reasoning_tokens": 47229,
  "provider": "Sail Research"
}
```
