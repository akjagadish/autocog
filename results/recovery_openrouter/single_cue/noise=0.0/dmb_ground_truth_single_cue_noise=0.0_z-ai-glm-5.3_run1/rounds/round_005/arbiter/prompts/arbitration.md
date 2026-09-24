# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_7" and "pi_4") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_7" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_4" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_7
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

## THEORY 2 — pi_4
Flattened-Diagnosticity Evidence Integration (the extreme-compression corner of the Bayesian cue-counting family): people integrate ALL binary expert ratings into one evidence score, where each discriminating cue casts a sign vote weighted by w_j = max(log(v_j/(1-v_j)) - tau, 0)^delta — chance cues (v=0.5) are annihilated exactly, but among genuinely diagnostic cues the subjective weight is a HEAVILY flattened (delta ~ 0.15) version of the normative log-odds, and choice is a low-gain softmax (2*beta ~ 0.6) plus a small uniform lapse. The new theoretical claim is that the compression exponent delta is itself the structural margin-compression device: shrinking delta equalizes weights, which GROWS count-dominated conflict margins (where cues split across options) while SHRINKING alignment-dominated cumulative margins (where several strong cues pile onto one side). This differential action — not any global gain knob, and not trial-level normalization — is what lets a single small beta simultaneously produce moderate TTB-leaning conflict-trial behavior (Exp 1), a slightly positive tally-vs-top-cue regression slope (Exp 2), graded ~0.62 tally-following (Exp 3), and near-chance |d|=1 agreement driven by the cancellation between super-expert matches and super-expert reversals (Exp 4).

`predict(parameters, state, history) -> np.ndarray`:
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

`policy(probs) -> int`:
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return int(np.random.choice(len(probabilities), p=probabilities))

## EXPERIMENT 1 (proposed by pi_7)

### DESIGN
**Validities (n_features=10):** [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5]

**Trial pairs (n=24):**
  trial 1: A=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0, 0, 0, 1, 0]
  trial 2: A=[0, 1, 0, 1, 0, 1, 1, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1, 0, 0, 0, 1]
  trial 3: A=[1, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  trial 4: A=[0, 1, 0, 1, 0, 1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0, 1, 1, 1, 0, 1]
  trial 5: A=[1, 0, 0, 0, 1, 0, 1, 0, 1, 0]  B=[1, 1, 0, 0, 1, 0, 0, 0, 1, 0]
  trial 6: A=[0, 0, 1, 0, 0, 1, 1, 1, 0, 1]  B=[0, 1, 1, 0, 0, 1, 0, 0, 0, 1]
  trial 7: A=[1, 0, 0, 1, 0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
  trial 8: A=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0, 1, 1, 1, 0]
  trial 9: A=[1, 0, 1, 0, 1, 0, 0, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0, 0, 1, 0, 1]
  trial 10: A=[0, 0, 1, 0, 1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 0, 1, 0, 1]
  trial 11: A=[1, 0, 0, 1, 0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0, 1, 0, 1, 0]
  trial 12: A=[1, 1, 0, 0, 1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
  trial 13: A=[1, 1, 1, 0, 0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1, 0, 0, 1, 0, 0]
  trial 14: A=[1, 0, 0, 1, 1, 0, 0, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 1, 0, 0]
  trial 15: A=[0, 1, 1, 0, 0, 0, 1, 0, 0, 1]  B=[0, 0, 1, 0, 0, 1, 1, 0, 0, 1]
  trial 16: A=[0, 0, 1, 0, 0, 1, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0, 0, 1, 0, 0, 1]
  trial 17: A=[1, 0, 0, 1, 0, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 0, 0, 0, 1, 0]
  trial 18: A=[0, 1, 0, 0, 1, 1, 1, 1, 0, 1]  B=[0, 1, 0, 0, 1, 0, 0, 0, 0, 1]
  trial 19: A=[0, 1, 0, 0, 1, 0, 1, 1, 0, 1]  B=[0, 1, 0, 0, 1, 1, 0, 0, 0, 1]
  trial 20: A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  trial 21: A=[1, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 22: A=[1, 1, 1, 1, 1, 1, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0, 1, 0, 1, 0]
  trial 23: A=[1, 0, 0, 0, 0, 0, 1, 0, 1, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 1, 0]
  trial 24: A=[0, 1, 0, 0, 1, 0, 1, 0, 0, 1]  B=[0, 1, 0, 0, 1, 0, 0, 1, 0, 1]

**Rationale:** CORE LOGIC. Validities are fixed for the whole experiment, so both theories reduce to a FIXED weight vector over the 10 experts, and the design's job is to make the two weight vectors maximally anti-correlated. The panel [0.99, 0.90, 0.80, 0.70, 0.60, 0.51, 0.50, 0.50, 0.50, 0.50] is engineered to push CREDT into its most distinctive regime, never instantiated in any prior experiment: (a) the 0.99 claim is an above-ceiling outlier with gap = 0.09 over the 0.90 second rung (sig_gap ~ 0.96-0.98, so the consensus gate is ~open); (b) the panel is CHANCE-LADEN: 5 of 10 experts sit below v_lo, so c_frac = 0.5 > c_half for the entire box, firing the composition amplifier (C ~ 1.0-1.3). Result: CREDT's weight on the 0.99 expert is w_top in [-1.93, -0.96] across its ENTIRE box (composition-amplified collapse), while the gate simultaneously pushes the chance flank to POSITIVE near-plateau weight: w(0.50) in [0.86, 0.91], w(0.51) ~ [0.87, 0.91], and the plateau experts (0.60-0.90) all get exactly w_p in [0.88, 0.92]. So on this panel CREDT is approximately 'count everybody's endorsement equally, EXCEPT the 99% expert, whose endorsement counts strongly AGAINST'. FDEI is the exact structural opposite: monotone graded weights with w(0.50) = 0 EXACTLY for every parameter, w(0.51) in [0, 0.73], and w(0.99) the LARGEST weight, in [1.17, 1.36]. Critically, FDEI is exactly invariant to the validities of tied cues and to panel composition, so the entire panel-structure manipulation is invisible to it except through which cues vote -- the structural source of every dissociation below. 24 unique pairs x K = 96//24 = 4 = 96 trials.

FAMILY Z (chance-only ladder; FDEI parameter-free knife). Z1/Z2/Z3: only 50% experts discriminate (A wins 1/2/3 of them); Z2m mirrors. FDEI must predict EXACTLY p = 0.50 on all four for every (delta, tau, beta, epsilon) -- all attending weights are identically zero. CREDT predicts a steep positive ladder: Z1 in [0.66, 0.68], Z2 in [0.79, 0.81], Z3 in [0.87, 0.90] across its whole box. Any reliable deviation from 0.50 on any Z trial falsifies FDEI outright with zero free parameters; a flat 0.50 falsifies CREDT's consensus-gating claim (which here flips chance cues from anti-diagnostic to near-fully trusted -- the OPPOSITE sign from the chance flank tested in [8], whose panel had c_frac = 0.27, below the amplifier threshold).

FAMILY X (90% expert vs k chance cues; FDEI flatness knife + sign flip). X1/X2/X3: B wins the 0.90 expert, A wins 1/2/3 chance cues; X3m mirrors. FDEI must predict X1 = X2 = X3 EXACTLY (chance weight is identically zero), with common value in [0.28, 0.37]. CREDT predicts a monotone ladder crossing 0.5: X1 in [0.49, 0.51], X2 in [0.65, 0.68], X3 in [0.78, 0.82]. X2 and X3 are box-disjoint SIGN FLIPS: 'does a 90% expert beat two coin-flippers?' CREDT: no (the coin-flippers win); FDEI: yes, strongly. Double bind: any X-ladder slope falsifies FDEI parameter-free; any X2 < 0.5 falsifies CREDT.

FAMILY S/V (0.99 inversion). S1 (A wins only the 0.99 expert): CREDT [0.19, 0.32] vs FDEI [0.64, 0.73] -- sign-disjoint. S1m mirrors. S2 (0.99 for A vs 0.90 for B): CREDT [0.11, 0.20] vs FDEI [0.51, 0.54]. V1 (0.99 for A vs one 50% for B): CREDT [0.11, 0.20] vs FDEI [0.64, 0.73] -- the single sharpest sign flip: FDEI says the 99% expert crushes a coin-flipper; CREDT says both point the same way, toward B.

FAMILY K (flagship: four trusted vs four coin-flip). K1: A wins the 0.90/0.80/0.70/0.60 experts, B wins all four 50% experts. CREDT: s = 4*w_p - 4*w(0.5) ~ 0, p in [0.48, 0.56] -- a COIN FLIP, because gated chance cues carry the same weight as plateau experts. FDEI: p in [0.88, 0.95]. A 35-45 point dissociation on a unanimous-expertise trial. K1m mirrors. p(K1) >= 0.60 falsifies CREDT outright; p(K1) <= 0.70 falsifies FDEI outright.

FAMILY P (plateau flatness; CREDT parameter-free knife). P2/P2m: A wins {0.90, 0.80} vs B wins {0.70, 0.60} (and mirror). All four discriminators sit on CREDT's plateau with tied counts, so CREDT must predict EXACTLY 0.50 for every parameter. FDEI's graded weights predict a tilt: P2 in [0.53, 0.58], P2m in [0.42, 0.47]. Any systematic tilt falsifies CREDT's plateau with no parameters; exact 0.50 falsifies FDEI's delta > 0. P4/P4m (0.90 vs 0.51 head-to-head): CREDT [0.49, 0.51] (gated 0.51 ~ plateau weight) vs FDEI [0.55, 0.69] / [0.31, 0.45] -- disjoint: CREDT says a 51% expert is exchangeable with a 90% expert on this panel; FDEI says the 90% expert clearly wins.

FAMILY W (honest knives against CREDT). W1 (0.51 for A vs 0.50 for B): CREDT is capped at 0.513 across its entire box; FDEI allows up to 0.63. If subjects distinguish 51% from 50% at p > 0.52, CREDT's gated-equalization claim is falsified. W2 (0.51 for A vs two 0.50 for B): CREDT [0.32, 0.35] vs FDEI [0.50, 0.63] -- disjoint; W2m mirrors.

ANCHORS/CONTROLS. G3/G3m (five sub-ceity experts vs one coin-flipper) are agreement anchors (CREDT [0.92, 0.95], FDEI [0.90, 0.97]) pinning lapse/temperature. N1 (one 50% expert won by each side): BOTH theories must predict exactly 0.50 for every parameter -- a side/display-bias assay; note N1 also defeats a 'count the 1s' heuristic (B has more 1s), as do V1 and W2 where raw-1 counts favor A while both theories' evidence favors B. Tie patterns mix 0/0 and 1/1 throughout so count-based heuristics are unviable; no all-tie trials; every trial requires scanning all 10 experts; sides are balanced (7 A-tilted / 6 B-tilted / near-neutral remainder under CREDT centers; 9/10 under FDEI).

DIFFERENTIATION FROM PRIOR EXPERIMENTS. No prior design instantiated the composition-amplifier regime (c_frac = 0.5 > c_half; [6] had 2/8, [8] 3/11, [9] 0/10 chance cues), and no prior design tested the CONJUNCTION it produces: 0.99 anti-diagnostic AND chance cues at near-full POSITIVE weight AND a flat plateau, on one fixed panel. [4] tested chance-cue weighting vs annihilation but with a monotone advocated theory and no outlier inversion; [6]-[9] tested 0.99 cliffs with chance cues negative or absent. The K1 cell (unanimous trusted panel vs equal-count coin-flip panel -> CREDT coin flip, FDEI 0.90) and the X-family sign crossover are new quantitative signatures. PREDICTED PATTERN IF CREDT IS RIGHT (centers): Z .67/.80/.87, X .50/.66/.79, S1 .23, S1m .77, S2 .14, V1 .14, P2 = P2m = 0.50 exact, P4 ~ .50, W1 ~ .51, W2 .32, K1 .51, G3 .92, N1 0.50. IF FDEI IS RIGHT: Z = 0.50 exactly (all four), X flat at .33, S1 .69, S1m .31, S2 .52, V1 .69, P2 .55, P4 .58, W1 .58, W2 .58, K1 .90, G3 .93, N1 0.50. The boxes are disjoint on 14 of 24 cells and every family contains a parameter-free exact-equality prediction for one theory that the other breaks in a signed, graded, pre-registered direction.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



### METRIC
Rationale:
Both theories collapse to a fixed 10-dimensional weight vector on this panel, so the optimal discriminating statistic is the linear contrast whose per-trial coefficient is the difference of the two theories' predicted p(A). Metric = mean_t [p_CREDT(t) - p_FDEI(t)] * y_t, with y_t = +1 for choosing A, -1 for B; the predictions are computed inside the metric from each theory's own source code at its parameter-box midpoints (CREDT weights ~ [-1.45, .90, .90, .90, .90, .883x5]; FDEI weights ~ [1.256, 1.124, 1.048, 0.972, 0.867, 0...0]). Expected value: if CREDT generated the data, E[metric] = (1/24) sum_t d_t (2 p_CREDT - 1) ~ +0.045 (positive contributions from the Z-family chance-cue ladders where CREDT says 0.59-0.74 and FDEI says exactly 0.50, from the 0.99-expert trials where CREDT says 0.28-0.36 and FDEI says 0.55-0.60, and from the X-family sign flips); if FDEI generated the data, E[metric] ~ -0.022 (negative contributions dominated by K1/K1m, where FDEI's 0.78/0.22 collides with CREDT's ~0.50, and by the 0.99 and X trials). The between-theory gap (~0.067) is large relative to per-subject sampling noise: each subject contributes 96 trials (4 reps of each of the 24 cells), giving a per-subject standard deviation of roughly 0.018-0.022 (binomial noise on the contrast, plus modest parameter-box heterogeneity whose sign structure is box-stable: CREDT's Z predictions stay in [0.66, 0.75], its 0.99 trials stay below 0.5, and FDEI's chance weight is identically zero for every parameter, so the contrast signs never flip). Welch's t at N=25 on (mean ~ +0.045, var ~ 4e-4) vs (mean ~ -0.022, var ~ 4e-4) gives |t| ~ 10+, far below alpha = 0.01. The metric works identically on the pooled DataFrame and on any single subject's slice (it is a per-trial average), satisfying the between-subject variance requirement. Trials where the theories agree (G3 anchors, N1, P-family) receive near-zero coefficients automatically, so they add no variance; the contrast concentrates its weight on the 14 box-disjoint cells.

Source:
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

### RESULTS
- Predicted under pi_7 (simulated): 0.0758 (var=0.0002)
- Predicted under pi_4 (simulated): -0.0398 (var=0.0003)
- Observed on real data: 0.0206 (var=0.0003)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=8):** [0.99, 0.94, 0.85, 0.75, 0.65, 0.5, 0.5, 0.5]

**Trial pairs (n=24):**
  trial 1: A=[1, 1, 0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 0]
  trial 2: A=[0, 1, 0, 1, 0, 1, 0, 0]  B=[1, 1, 0, 1, 0, 1, 0, 0]
  trial 3: A=[1, 0, 1, 1, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0, 0, 1, 0]
  trial 4: A=[0, 1, 1, 1, 0, 0, 1, 0]  B=[1, 0, 1, 1, 0, 0, 1, 0]
  trial 5: A=[1, 0, 0, 1, 1, 0, 1, 1]  B=[0, 1, 0, 1, 1, 0, 0, 1]
  trial 6: A=[1, 0, 1, 0, 1, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1, 1, 0]
  trial 7: A=[1, 0, 0, 1, 0, 1, 1, 1]  B=[0, 1, 0, 1, 0, 1, 0, 0]
  trial 8: A=[0, 1, 0, 1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 0, 1, 1, 1]
  trial 9: A=[1, 1, 1, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 1, 0, 0, 1]
  trial 10: A=[0, 1, 1, 0, 1, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0, 0, 1]
  trial 11: A=[1, 0, 1, 1, 0, 1, 1, 0]  B=[1, 0, 1, 1, 0, 1, 0, 0]
  trial 12: A=[0, 1, 0, 1, 1, 0, 1, 1]  B=[0, 1, 0, 1, 1, 0, 0, 0]
  trial 13: A=[0, 1, 0, 1, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0, 1, 1]
  trial 14: A=[1, 0, 1, 0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 1, 1, 0, 1]
  trial 15: A=[1, 1, 0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 0, 1, 1, 0, 0]
  trial 16: A=[1, 1, 0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 1, 0, 1, 1]
  trial 17: A=[0, 1, 1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 1, 0, 0]
  trial 18: A=[1, 1, 0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 1, 0, 1]
  trial 19: A=[1, 0, 0, 1, 1, 1, 0, 1]  B=[1, 1, 0, 1, 1, 0, 0, 1]
  trial 20: A=[1, 1, 1, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 1, 1, 1, 0]
  trial 21: A=[1, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 1, 0]
  trial 22: A=[1, 0, 1, 1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1, 1, 0]
  trial 23: A=[0, 1, 0, 0, 1, 1, 1, 0]  B=[1, 0, 1, 1, 0, 0, 1, 0]
  trial 24: A=[1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1]

**Rationale:** CORE LOGIC. Both theories share the identical architecture (sign votes x validity weights x low-gain softmax x lapse) and differ ONLY in the weight function, but the weight functions are near-orthogonal on this panel. The panel [0.99, 0.94, 0.85, 0.75, 0.65, 0.55, 0.50, 0.50] is engineered to place CREDT in its PEAKED EDGE-DISTRUST regime, never instantiated in any prior experiment: the 0.99 claim sits exactly 0.05 above the 0.94 second rung -- the gap that CREDT's non-monotone edge-distrust bump P treats as MAXIMAL spurious precision (g_star ~ 0.048) -- while the panel is coherent (c_frac = 2/8 = 0.25 < c_half, so the composition amplifier contributes only C ~ 0.09-0.26 rather than firing). Resulting CREDT weights across its ENTIRE box: w(0.99) in [-1.04, -0.59] (strongly ANTI-diagnostic), w(0.94)...w(0.55) all exactly w_p in [0.88, 0.92] (flat plateau), and the two 0.50 experts GATED to positive weight (gate = lambda_ext + (1-lambda_ext)*sig_gap ~ 0.73-0.82 open, since the outlier's sig_gap ~ 0.59-0.68), giving w(0.50) in [0.62, 0.69]. FDEI weights across its entire box: w(0.99) in [1.16, 1.36] (the LARGEST weight), w(0.94) in [1.11, 1.23], w(0.85) in [1.06, 1.12], w(0.75) in [1.01, 1.02], w(0.65) in [0.91, 0.95], w(0.55) in [0.69, 0.85], w(0.50) = 0 EXACTLY for every (delta, tau, beta, epsilon). The two weight vectors are almost perfectly anti-correlated: FDEI's best cue is CREDT's worst cue; FDEI's invisible cues are worth ~70% of a plateau cue to CREDT; CREDT's plateau is flat where FDEI is (weakly) graded. 24 unique pairs x K = 96//24 = 4 = 96 trials.

FAMILY S/W (flagship: spurious-precision inversion + parameter-free chance-cue flatness ladder). S1 (A wins only the 0.99 expert): FDEI box p(A) in [0.64, 0.73] (center 0.69); CREDT box [0.30, 0.39] (center 0.35) -- SIGN-DISJOINT with a 25-point gap: FDEI says follow the 99% expert, CREDT says a 0.99-atop-0.94 claim is spurious precision and its endorsement counts AGAINST. S1m mirrors. S2 (A wins 0.99, B wins 0.94): FDEI's delta-flattening locks w99-w94 in [0.05, 0.14], so its ENTIRE box spans only p(A) in [0.509, 0.524] -- a near-parameter-free coin flip; CREDT box [0.17, 0.25] (center 0.21). A 28-point dissociation on a cell where FDEI is pinned within 1.5 points of 0.50. THE LADDER: S4 (B also wins one 0.50) / S2 / S3 (A also wins one 0.50) / W2 (A also wins both 0.50s) holds the 0.99-vs-0.94 diagnostic background fixed while the net chance-cue count for A runs -1, 0, +1, +2. FDEI must predict S4 = S2 = S3 = W2 EXACTLY for every parameter (chance weight is identically zero) -- a parameter-free flatness knife. CREDT predicts a steep monotone ladder: 0.141 / 0.209 / 0.305 / 0.425 at center (boxes [0.11,0.17] / [0.17,0.25] / [0.26,0.35] / [0.36,0.48]) -- a ~30-point climb that directly measures the gated chance-cue weight (~0.65) against the outlier distrust (~-0.79) in one psychometric. W2m mirrors. Any reliable slope across the ladder falsifies FDEI with zero free parameters; flatness at 0.51 falsifies CREDT's consensus-gating claim.

FAMILY V (outlier vs chance, and the S1=V1 flatness knife). V1 (A wins 0.99, B wins one 0.50): FDEI [0.64, 0.73] (identical to S1 -- FDEI must predict V1 = S1 EXACTLY, since chance cues are invisible); CREDT [0.20, 0.28] (center 0.24: BOTH the 99% anti-vote and the gated coin-flipper point toward B). V1m mirrors. CREDT predicts V1 < S1 by ~10 points (the chance cue helps B); FDEI demands exact equality -- a second parameter-free knife, and V1 is a sign-disjoint cell in its own right.

FAMILY Z (chance-only ladder; FDEI parameter-free knife). Z1/Z2 (A wins 1/2 of the 0.50 experts, nothing else discriminates): FDEI must predict EXACTLY 0.50 on both (all weights zero -> uniform softmax, lapse uniform) for every parameter. CREDT predicts a steep positive ladder: Z1 in [0.62, 0.74] (center 0.63), Z2 in [0.72, 0.76] (center 0.74). Z2m mirrors. Any reliable deviation from 0.50 falsifies FDEI outright; a flat 0.50 falsifies CREDT's gated-trust claim. Note the sign here is POSITIVE (chance cues trusted), the OPPOSITE of the ungated negative flank -- on this panel the gate is partially open, which is CREDT's distinctive conjunction: distrust the 0.99 outlier AND simultaneously count the coin-flippers.

FAMILY X (plateau expert vs gated chance cues; mutual sign flip and mutual knife). X1 (A wins one 0.50, B wins the 0.55 expert): FDEI [0.35, 0.41] (center 0.38); CREDT [0.435, 0.465] (center 0.45). X2 (A wins both 0.50s, B wins 0.55): FDEI must predict X2 = X1 EXACTLY (parameter-free flatness; center 0.38); CREDT predicts a SIGN FLIP to [0.56, 0.60] (center 0.58) -- two coin-flippers beat a 55% expert under gating. X2m mirrors. X3 (A wins both 0.50s, B wins the 0.75 expert): CREDT must predict X3 = X2 EXACTLY (plateau weights identical to machine precision) -- its parameter-free flatness knife; FDEI predicts X3 < X2 (0.35 vs 0.38, its graded 0.75-vs-0.55 weights). The X family is a double bind: the X1=X2 equality falsifies FDEI if broken; the X2=X3 equality falsifies CREDT if broken; and the X2/X3 sign (FDEI [0.33,0.41] vs CREDT [0.56,0.60]) is box-disjoint in the opposite direction from X1.

FAMILY Q/P (FDEI's within-plateau gradient vs CREDT's flat plateau; CREDT parameter-free knife). Q1 (A wins 0.94, B wins 0.55; 1-vs-1): CREDT must predict EXACTLY 0.50 (both on the plateau, tied counts) for every parameter; FDEI predicts [0.535, 0.597] (center 0.56) because its compressed-but-graded weights give the 0.94 side a net +0.27 to +0.53 evidence margin. Q1m mirrors. P1 (A wins {0.94, 0.85}, B wins {0.65, 0.55}; 2-vs-2): CREDT exactly 0.50; FDEI [0.549, 0.635] (center 0.58) -- its strongest plateau-tilt cell. P1m mirrors. Any systematic tilt on Q1/P1 falsifies CREDT's plateau with zero free parameters; exact 0.50 across the family falsifies FDEI's delta > 0 gradient. The Q/P family is deliberately modest in size (5-8 points) -- recovering BOTH this small graded tilt AND the huge S/Z effects at their predicted relative sizes is the quantitative calibration test.

FAMILY H (masked inversion inside a balanced 3-vs-3 coalition). H3 (A wins {0.99, 0.85, 0.75}, B wins {0.94, 0.65, 0.55}; raw cue count tied 3-3, raw 1-count tied): FDEI margin is remarkably delta-stable (0.48-0.51 across its whole box), locking p(A) in [0.575, 0.594] (center 0.58); CREDT margin = w_top - w_p, giving [0.17, 0.25] (center 0.21) -- a 35-point SIGN FLIP on a count-balanced trial. H3m mirrors. This cell defeats any count-based heuristic (counts are exactly tied) and isolates pure weighting.

ANCHOR. G1 (A wins all five plateau experts, B wins both 0.50s): both theories choose A at high confidence (FDEI [0.92, 0.97], center 0.95; CREDT [0.90, 0.93], center 0.91), pinning lapse/temperature; G1m-equivalent structure is carried by the mirrors.

DIFFERENTIATION FROM PRIOR EXPERIMENTS. No prior design instantiated the g_star PEAK regime of CREDT's edge-distrust bump: [10] used a 0.99-atop-0.90 panel (gap 0.09, large-edge branch) that was CHANCE-LADEN (c_frac = 0.5, amplifier firing, chance cues at 0.86-0.91); here gap = 0.05 with a coherent panel (c_frac = 0.25, amplifier sub-threshold, chance cues at 0.62-0.69, w_top shallower at -0.6 to -1.0). The S4->S2->S3->W2 ladder measuring gated chance weight against peaked outlier distrust in one psychometric is new, as is the FDEI-locked near-tie on 0.99-vs-0.94 (its entire box within 1.5 points of 0.50) and the S1=V1 and X1=X2 flatness knives. [6]-[9] tested FDEI against other competitors (SITT/BSST), not CREDT. PREDICTED PATTERN IF FDEI IS RIGHT (center delta=0.15, tau=0.01, beta=0.32, eps=0.025): S1 .69, S2=S3=S4=W2 .51 exactly, V1 .69 (=S1), Z1=Z2 .50 exactly, X1=X2 .38, X3 .35, Q1 .56, P1 .58, H3 .58, G1 .95. IF CREDT IS RIGHT (center w_p=.90, rho=1.36, beta=.42, eps=.05): S1 .35, S4 .14 < S2 .21 < S3 .31 < W2 .43, V1 .24, Z1 .63, Z2 .74, X1 .45, X2=X3 .58 exactly, Q1=P1 .50 exactly, H3 .21, G1 .91. The boxes are disjoint on 14 of 24 cells, sign-disjoint on 6 (S1, S2, V1, X2, X3, H3), and every family contains a parameter-free exact-equality or exact-0.50 prediction for one theory that the other breaks in a signed, graded, pre-registered direction. Sides are balanced (7 A-tilted / 7 B-tilted / 10 near-neutral under FDEI centers; 8/11/5 under CREDT centers; every family contains an A/B mirror); no all-tie trials; every trial requires scanning all 8 experts; tie patterns mix 0/0 and 1/1 so 'count the 1s' is unviable (e.g., H3 has exactly tied 1-counts while the theories disagree by 35 points, and X2/X3 have MORE 1s for A while FDEI favors B).

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



### METRIC
Rationale:
METRIC: 'Super-expert deference rate' — on every trial where feature 0 (the 0.99-validity expert) discriminates between A and B, score 1 if the subject chose the option that expert endorsed, 0 otherwise; the metric is the mean. In this design 12 of the 24 unique trial types have the 0.99 expert discriminating (trials 1-10, 22, 23), giving 48 scored trials per subject, so the metric is well-defined both on the pooled DataFrame and on every individual subject's slice.

WHY IT MAXIMALLY SEPARATES THE TWO THEORIES. The single deepest structural disagreement on this panel is the sign of the weight on the 0.99 expert. FDEI (any delta in [0.1,0.2], tau in [0,0.04]) gives the 0.99 expert its LARGEST weight (w ~ 1.16-1.36 vs 1.10-1.22 for the 0.94 expert), so it predicts deference: p(follow) ~ 0.67-0.70 on trials 1/2/9/10 (super-expert alone or with a chance cue), ~0.51 on trials 3-8 (0.99 vs 0.94, chance-invariant), and honestly BELOW 0.5 on trials 22/23 (0.99 vs a 0.94+0.65+0.50 coalition), averaging ~0.57 across the whole box. CREDT, by contrast, reads a 0.99 claim sitting 0.05 above the 0.94 rung on a panel that is 3/8 chance-laden as spurious precision amplified by composition: across its ENTIRE parameter box the panel-max weight is strongly negative (w_top ~ -1.0 to -1.6, since the peaked bump P ~ 1 and the composition term C ~ 0.6-1.0 both fire), so CREDT predicts ANTI-deference: p(follow) ~ 0.20-0.29 on trials 1/2, ~0.13-0.19 on trials 3/4, ~0.05-0.10 on trial 6, ~0.05-0.08 on trials 22/23, averaging ~0.19-0.27. The two predicted values are sign-opposite around 0.5 with a mean gap of ~0.35 (0.57 vs 0.21 at the centers).

WHY IT PASSES THE ACCEPTANCE TEST AT N=25. Per subject the metric is a mean of 48 Bernoulli draws whose per-trial success probabilities lie in [0.06, 0.70]. The predicted between-subject standard deviation is ~0.07 under FDEI (var ~ 0.0049) and ~0.06 under CREDT (var ~ 0.0033). Welch's t on (0.57, 0.0049, 25) vs (0.21, 0.0033, 25) gives t ~ 19, p << 0.001 — orders of magnitude below alpha = 0.01, with huge slack against parameter heterogeneity (the boxes never overlap: FDEI's whole box stays in [0.55, 0.60], CREDT's in [0.18, 0.27]).

WHY IT IS NOT TRIVIALLY TRUE FOR FDEI. The metric is a raw behavioral signature — 'do people obey a 99% expert?' — not a restatement of the model. FDEI does NOT predict a value near 1: it predicts below-chance deference on the coalition trials 22/23 (0.36/0.64 for the endorsed side) and only ~0.51 on the 0.99-vs-0.94 trials, so the pooled 0.57 is a genuine, falsifiable quantitative prediction. A human value near 0.21 (systematic anti-deference) would decisively falsify FDEI, and a value near 0.57 would decisively falsify CREDT's spurious-precision/edge-distrust claim; the two theories' predictions differ by 0.35, so the human number will clearly adjudicate between them.

Source:
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

### RESULTS
- Predicted under pi_7 (simulated): 0.2367 (var=0.0031)
- Predicted under pi_4 (simulated): 0.6017 (var=0.0039)
- Observed on real data: 0.4525 (var=0.0028)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
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
**Predicted under pi_7:** -0.2530 (var=0.0065)
**Predicted under pi_4:** -0.2188 (var=0.0064)

### Experiment 4
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
**Predicted under pi_7:** 0.1225 (var=0.0025)
**Predicted under pi_4:** 0.0881 (var=0.0019)

### Experiment 5
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
**Predicted under pi_7:** 0.6267 (var=0.0043)
**Predicted under pi_4:** 0.6485 (var=0.0027)

### Experiment 6
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
**Predicted under pi_7:** 0.5425 (var=0.0051)
**Predicted under pi_4:** 0.5192 (var=0.0023)

### Experiment 7
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
**Predicted under pi_7:** -0.1425 (var=0.0147)
**Predicted under pi_4:** 0.0308 (var=0.0233)

### Experiment 8
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
**Predicted under pi_7:** 0.1235 (var=0.0028)
**Predicted under pi_4:** 0.7670 (var=0.0039)

### Experiment 9
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
**Predicted under pi_7:** 0.5787 (var=0.0053)
**Predicted under pi_4:** 0.3191 (var=0.0050)

### Experiment 10
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
**Predicted under pi_7:** 0.4813 (var=0.0186)
**Predicted under pi_4:** 0.5400 (var=0.0192)

### Experiment 11
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
**Predicted under pi_7:** 0.5715 (var=0.0054)
**Predicted under pi_4:** 0.3985 (var=0.0055)

### Experiment 12
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
**Predicted under pi_7:** 0.6094 (var=0.0057)
**Predicted under pi_4:** 0.6719 (var=0.0085)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all 12 experiments, pi_7 (CREDT) is clearly the stronger theory, but it is not flawless, and pi_4 (FDEI) is degenerate on a large portion of the domain. Per-experiment absolute errors: Exp1: pi_7 0.055 vs pi_4 0.060 (slight edge pi_7, correct sign); Exp2: 0.216 vs 0.149 (edge pi_4); Exp3: 0.035 vs 0.001 (pi_4); Exp4: 0.045 vs 0.011 (pi_4); Exp5: 0.010 vs 0.032 (pi_7); Exp6: 0.029 vs 0.006 (pi_4); Exp7: 0.026 vs 0.199 (pi_7); Exp8: 0.006 vs 0.649 (pi_7, catastrophic miss for pi_4); Exp9: 0.001 vs 0.261 (pi_7); Exp10: 0.019 vs 0.078 (pi_7); Exp11: 0.012 vs 0.162 (pi_7); Exp12: 0.011 vs 0.073 (pi_7). pi_7 wins 8 of 12 experiments with a mean absolute error of ~0.039 versus ~0.140 for pi_4, and \u2014 critically \u2014 pi_7 gets the SIGN of every observed effect right in all 12 experiments, including the hardest ones: strong-side avoidance on chance-laden panels (Exp8: 0.124 vs 0.118), mild anti-deference to a 99% expert (Exp9: 0.579 vs 0.580), anti-99% cliff conflicts (Exp11: 0.572 vs 0.560), and pro-99% deference on a dense corroboration ladder (Exp12: 0.609 vs 0.599). pi_4 fails qualitatively wherever a 99% expert must be distrusted (Exp8, 9, 11) and wherever chance cues carry weight (Exp7), because its monotone flattened-diagnosticity weighting structurally cannot invert or gate cues. However, pi_7 systematically OVER-polarizes in its outlier-distrust regimes: in Exp2 (gap=0.05, coherent panel) it predicted deference 0.237 where the observed value is 0.453 \u2014 the right direction (below 0.5) but roughly twice too strong; in Exp1 its contrast signature came in at about half its predicted magnitude (0.021 vs 0.076). The data thus say: pi_7's qualitative architecture (non-monotone edge-distrust, consensus gating, panel-composition sensitivity) is right, but its distrust magnitudes are unbounded and overshoot; human distrust of uncorroborated precision is real but attenuated, landing choices nearer 0.45-0.58 rather than 0.12-0.24 in the moderate regimes while still collapsing hard in the composition-amplified regime (Exp8). Since pi_4 is degenerate (it cannot explain Exps 7, 8, 9, 11 at any parameter setting \u2014 these are structural, parameter-free failures of monotone diagnosticity weighting), the correct action is to replace pi_4 with a new theory that is a genuine competitor to pi_7: one that keeps the task-invariant insight CREDT got right (validity weights are relative to panel structure, not monotone in stated validity) but replaces CREDT's unbounded sign-flipping distrust with bounded, saturating distrust.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace pi_4 with a new theory: Bounded Outlier-Distrust Tallying (BODT). Core claims: (1) Subjects integrate all binary ratings as sign votes with subjective weights that are NOT a monotone function of stated validity \u2014 the weight of an expert is computed relative to panel structure (this is the insight pi_7 got right and FDEI structurally cannot express). (2) Experts with validity inside a trusted band [v_lo, v_hi] receive approximately equal plateau weight; chance experts (v near 0.5) receive a small weight whose sign and magnitude are set by a consensus gate (near zero on coherent panels, mildly positive when an above-ceiling outlier destabilizes the scale). (3) THE KEY DEPARTURE FROM CREDT: distrust of an above-ceiling outlier is a bounded, saturating SHRINKAGE of its weight toward a floor, not an unbounded sign flip. The discount grows with the outlier's uncorroborated edge (peaked near a small gap ~0.05, receding for dense ladders and for large edges on coherent panels) and is amplified on chance-laden panels, but it saturates so that w_top never falls below roughly -0.3 to -0.5 plateau units in moderate regimes, and reaches strongly negative values only in the composition-amplified regime (a far outlier floating above a majority-chance panel, as in Exp8). This single change predicts: deference ~0.45-0.50 (not 0.24) in the gap=0.05 spurious-precision regime (Exp2), an attenuated contrast signature in Exp1, mild anti-deference ~0.55-0.60 in Exp9/Exp11, pro-99% deference ~0.60 on dense ladders (Exp12), and near-chance tally following ~0.52 on |d|=1 trials (Exp6), while retaining the strong-side collapse of Exp8. (4) Choice remains a low-gain softmax plus a uniform lapse, with a moderately elevated lapse rate to absorb the residual over-polarization pi_7 exhibited. BODT differs from pi_4 (FDEI) in abandoning monotone diagnosticity weighting entirely (fixing Exps 7, 8, 9, 11), and differs from pi_7 (CREDT) in replacing unbounded peaked distrust with saturating bounded distrust (fixing the magnitude overshoots in Exps 1 and 2 while preserving every sign prediction pi_7 got right)."
}
```

## Usage

```json
{
  "input_tokens": 33766,
  "output_tokens": 5288,
  "reasoning_tokens": 3907,
  "provider": "Sail Research"
}
```
