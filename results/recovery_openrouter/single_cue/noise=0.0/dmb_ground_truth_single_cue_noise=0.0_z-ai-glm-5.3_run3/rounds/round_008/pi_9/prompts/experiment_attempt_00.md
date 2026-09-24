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
**Description:** POSITION-BRIDGED DISTRUST TALLYING (PBDT), un-bundled-gate edition. Subjects scan expert ratings in presentation order through a steep floored convex primacy gate g_j = delta + (1-delta)*((n-j)/(n-1))^eta, with the gate restored to the steep end of its box (eta high, delta low) so late low-validity helper cues are crushed relative to early ones — the structural lever separating primacy-rescue trials from validity-position-confounded layouts. Stated validities are distrust-inverted (treated as error rates) via inv_j = ((1-v_j) - c*min_k(1-v_k))^gamma / Z, kept at the steep bridged range [2.6, 2.9] with a high distrust share tau, so the anti-validity gradient in confounded layouts is carried by the distrust kernel alone rather than by the gate. Trust in high-validity experts ramps with position from a small-but-nonzero floor t_j = t0 + (t1-t0)*(j/(n-1))^kappa, with a slightly concave-leaning ramp (kappa below 1) so late trusted experts recover nearly the full flat share without raising the early floor. Weights combine multiplicatively, w_j proportional to g_j * [t_j*(1-tau)/n + tau*inv_j]; evidence d = sum_j w_j*sign(A_j - B_j) is integrated additively over discriminating cues; choice is logistic in beta*d with a uniform lapse epsilon, with noise set between the two previous iterations so slope/contrast metrics that were over-attenuated recover while anchor allegiances keep their margin.

**Parameters:**
- tau: [0.91, 0.93]
- eta: [6.5, 8.0]
- delta: [0.15, 0.25]
- gamma: [2.6, 2.9]
- c: [0.05, 0.25]
- t0: [0.25, 0.35]
- t1: [0.95, 1.00]
- kappa: [0.8, 1.5]
- beta: [3.6, 4.4]
- epsilon: [0.11, 0.14]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # PBDT: Position-Bridged Distrust Tallying, UN-BUNDLED-GATE edition
    # (iter-3). Minimal-diff edit of the ACCEPTED iter-2 base per the
    # critic's iter-2 directive, which diagnosed that the iter-2 move (b)
    # had bundled an efficacious half (distrust strengthening: the Exp 3
    # win) with a counterproductive half (gate flattening: the Exp 14
    # and Exp 13 regressions). This iteration UN-BUNDLES them:
    #   (A) REVERT THE GATE FLATTENING, KEEP THE DISTRUST STRENGTHENING:
    #       eta [5.5,6.0] -> [6.5, 8.0] (steep end), delta [0.25,0.30]
    #       -> [0.15, 0.25] (low floor), while gamma stays [2.6, 2.9]
    #       and tau stays [0.91, 0.93] exactly as in iter-2. The steep
    #       gate crushes the late 50%-validity helpers on Exp 14's
    #       trusted-early side (0.350 -> toward 0.132) and restores the
    #       M/C position families on Exp 13 (1.35 -> toward 1.84);
    #       the anti-validity gradient in confounded layouts (Exp 3)
    #       is carried by the distrust kernel, which iter 2 verified.
    #   (B) PARTIALLY ROLL BACK THE NOISE: beta [3.2,4.0] -> [3.6, 4.4]
    #       and epsilon [0.13,0.16] -> [0.11, 0.14] -- between the
    #       iter-1 and iter-2 settings. Lifts the over-attenuated slope
    #       metrics (Exps 1, 11, 12, 13, 15) at a small cost on Exps
    #       4/7, which currently have margin. Exp 2 is held as a
    #       monitored constraint (structural, noise-insensitive).
    #   (C) WIDEN THE RAMP CONVEXITY DOWNWARD: kappa [1.0,1.5] ->
    #       [0.8, 1.5], a concave-leaning ramp so the late trusted
    #       expert recovers more of the flat share (Exp 8 FEDI
    #       0.294 -> toward 0.396) without touching t0, tau, or the
    #       gate. t0/t1 unchanged from iter-2.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"PBDT expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n = int(stim.shape[1])

    v = np.asarray(parameters["validities"], dtype=float).ravel()
    if v.shape[0] != n:
        # Graceful fallback: strictly decreasing validities.
        v = np.linspace(0.95, 0.55, n)
    v = np.clip(v, 0.0, 1.0)

    tau = float(parameters["tau"])        # distrust share of the bracket (HIGH, kept)
    eta = float(parameters["eta"])        # convexity of the primacy gate (STEEP, reverted)
    delta = float(parameters["delta"])    # floor of the primacy gate (LOW, reverted)
    gamma = float(parameters["gamma"])    # steepness of distrust inversion (kept)
    c = float(parameters["c"])           # partial centering of error rates
    t0 = float(parameters["t0"])          # trust floor at position 1 (LOW, kept)
    t1 = float(parameters["t1"])          # trust residue at the last position (HIGH, kept)
    kappa = float(parameters["kappa"])   # convexity of the trust ramp (WIDENED DOWN)
    beta = float(parameters["beta"])     # choice consistency (PARTIALLY RESTORED)
    eps = float(parameters["epsilon"])   # uniform lapse (PARTIALLY RESTORED)

    # (1) DISTRUST INVERSION with partial centering: treat the stated
    # validity as an error rate; subtract a fraction c of the smallest
    # error rate before raising to the gamma power, then normalize.
    # gamma in [2.6, 2.9] and tau in [0.91, 0.93] are preserved verbatim
    # from iter-2 -- the verified carriers of the anti-validity gradient
    # in validity-position-confounded layouts (Exp 3).
    err = 1.0 - v
    centered = np.maximum(err - c * float(err.min()), 0.0)
    inv = np.power(centered, gamma)
    Z_v = float(inv.sum())
    if Z_v > 1e-12:
        w_inv = inv / Z_v
    else:
        # Degenerate design (all validities tied at the extreme): the
        # distrust kernel is flat and position alone carries structure.
        w_inv = np.full(n, 1.0 / n)

    if n > 1:
        pos = np.arange(n, dtype=float)
        # (2) TRUST RAMP with widened-down convexity: low floor t0,
        # high t1, and kappa now allowed below 1.0 so the ramp is
        # concave-leaning -- the late trusted expert recovers more of
        # the flat share earlier in the tail (the Exp 8 FEDI lever)
        # without raising the early floor that Exp 14 needs kept low.
        t = t0 + (t1 - t0) * np.power(pos / (n - 1.0), kappa)
        # (3) PRIMACY GATE REVERTED TO THE STEEP END: eta in [6.5, 8.0]
        # with delta in [0.15, 0.25]. The steep, low-floored gate
        # concentrates attention on the first one or two experts and
        # crushes the late tail -- the structural fix for Exp 14's
        # late-helper over-weighting and Exp 13's M/C position
        # families. The anti-validity regression in confounded layouts
        # is carried by the (preserved) steep distrust kernel instead.
        g = delta + (1.0 - delta) * np.power((n - pos) / (n - 1.0), eta)
    else:
        t = np.array([1.0])
        g = np.array([1.0])

    # (4) MULTIPLICATIVE combination: the gate scales the bracket
    # cue-by-cue. tau in [0.91, 0.93] keeps the distrust channel
    # dominant; the trusted channel is carried by the ramp (small
    # early, near-full and now earlier-recovering late).
    bracket = t * (1.0 - tau) / n + tau * w_inv
    raw = g * bracket
    Z_t = float(raw.sum())
    if Z_t > 1e-12:
        w = raw / Z_t
    else:
        w = np.full(n, 1.0 / n)

    # (5) Deterministic attention: every discriminating cue is attended;
    # no capacity interference, beta is design-independent.
    s = np.sign(a - b)

    # Additive integration of all discriminating cues.
    d = float(np.dot(w, s))

    # Numerically stable logistic choice rule -- beta in [3.6, 4.4],
    # between the iter-1 and iter-2 calibrations, so the slope metrics
    # over-attenuated by iter-2's noise (Exps 1, 11, 12, 13, 15)
    # recover while Exps 4/7 keep their remaining margin.
    p_a = 0.5 * (1.0 + np.tanh(0.5 * beta * d))
    probs = np.array([p_a, 1.0 - p_a], dtype=float)

    # (6) Uniform lapse, epsilon in [0.11, 0.14]: partially rolled back
    # from iter-2's [0.13, 0.16] per the critic's split-the-difference
    # directive. Exp 2 is NOT chased with further noise -- it is held
    # as a monitored structural constraint of the distrust family.
    probs = (1.0 - eps) * probs + eps * 0.5
    return probs
```

**`policy source code`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # guard against float drift
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** PRIMACY-GATED DISTRUST TALLYING (PGDT), floored-gate edition with a rebalanced distrust share. Subjects scan the expert ratings in presentation order and attention is deployed through a steep convex primacy gate with a substantial nonzero floor: g_j = delta + (1-delta)*((n-j)/(n-1))^eta, so the first one or two experts carry the dominant share of attention but late experts retain a meaningful attention residue. Independently, the communicated validities are distrust-inverted exactly as in the shared kernel: subjects act as if each stated validity were the expert's error rate, inv_j = ((1-v_j) - c*min_k(1-v_k))^gamma / Z. The two kernels combine MULTIPLICATIVELY: w_j proportional to g_j * [(1-tau)/n + tau*inv_j], normalized. The refinement this iteration is a rebalanced distrust share: tau is lowered to [0.82, 0.92] so the flat component (1-tau)/n of the multiplicative bracket grows from ~1.5% to ~4-5% per cue. This directly feeds trusted high-validity LATE experts (whose bracket value is structurally pinned at the flat share regardless of gate shape), softens the distrust-driven allegiance contrasts that overshoot their targets, and pulls the effective weights off pure anti-validity where that overshoots. A modestly raised gate floor delta in [0.20, 0.45] compensates on decreasing-validity layouts by further concentrating the product weights onto the back-loaded low-validity experts, protecting the anti-validity regression. Evidence is integrated additively over all discriminating cues, d = sum_j w_j * sign(A_j - B_j), and choice is logistic in beta*d with a small uniform lapse epsilon.

**Parameters:**
- tau: [0.82, 0.92]
- eta: [4.0, 6.0]
- delta: [0.20, 0.45]
- gamma: [2.8, 3.5]
- c: [0.10, 0.25]
- beta: [2.5, 6.0]
- epsilon: [0.06, 0.15]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # PGDT: Primacy-Gated Distrust Tallying, FLOORED-GATE edition,
    # REBALANCED-DISTRUST-SHARE variant.
    # iter-4 minimal-diff edit per critic's iter-3 directive:
    #   (1) PRIMARY KNOB: LOWER TAU. Range moved from [0.90, 0.97] to
    #       [0.82, 0.92] (still inside the arbiter's prescribed
    #       [0.80, 0.97]). This grows the flat share (1-tau)/n of the
    #       multiplicative bracket from ~1.5% to ~4-5% per cue, which is
    #       the ONLY lever that can move Exp8's final 0.90-validity
    #       expert (its weight is structurally pinned at the flat share
    #       regardless of gate shape), and simultaneously softens the
    #       over-committed distrust allegiances on Exp2/Exp4/Exp7.
    #   (2) COMPENSATING KNOB: RAISE DELTA CEILING. Range moved from
    #       [0.15, 0.40] to [0.20, 0.45] so the iter-2->3 trajectory's
    #       verified leverage on Exp3's back-loading (stronger
    #       anti-validity as the early-gate flattens) compensates the
    #       Exp3 weakening that the tau drop would otherwise cause.
    #   (3) Everything else frozen: eta, gamma, c, beta, epsilon.
    # Mechanism structure unchanged from the accepted iter-3 base.
    # Stimulus: array of shape (2, n_features); row 0 = option A, row 1 = B.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"PGDT expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n = int(stim.shape[1])

    v = np.asarray(parameters["validities"], dtype=float).ravel()
    if v.shape[0] != n:
        # Graceful fallback: strictly decreasing validities.
        v = np.linspace(0.95, 0.55, n)
    v = np.clip(v, 0.0, 1.0)

    tau = float(parameters["tau"])      # trust in the distrust kernel
    eta = float(parameters["eta"])       # convexity of the primacy gate
    delta = float(parameters["delta"])  # floor of the primacy gate
    gamma = float(parameters["gamma"])   # steepness of distrust inversion
    c = float(parameters["c"])           # partial centering of error rates
    beta = float(parameters["beta"])     # choice consistency
    eps = float(parameters["epsilon"])  # uniform lapse

    # (1) DISTRUST INVERSION with partial centering: treat the stated
    # validity as an error rate; subtract a fraction c of the smallest
    # error rate before raising to the gamma power, then normalize.
    err = 1.0 - v
    centered = np.maximum(err - c * float(err.min()), 0.0)
    inv = np.power(centered, gamma)
    Z_v = float(inv.sum())
    if Z_v > 1e-12:
        w_inv = inv / Z_v
    else:
        # Degenerate design (all validities tied at the extreme): the
        # distrust kernel is flat and the gate alone carries position.
        w_inv = np.full(n, 1.0 / n)

    # (2) STEEP CONVEX PRIMACY GATE WITH RAISED FLOOR over presentation
    # positions. Cue j (0-indexed) gets gate mass
    #     g_j = delta + (1-delta) * ((n - j)/(n - 1))^eta
    # with delta now in [0.20, 0.45]: the FIRST position still carries
    # the dominant share, but every late position retains a substantial
    # attention residue, so the multiplicative product never approaches
    # a pure primacy gradient and late low-validity (maximum-distrust)
    # cues keep enough weight to carry the anti-validity signature.
    # The raised ceiling further back-loads the product on decreasing-
    # validity layouts, compensating the tau drop for Exp3/Exp5.
    if n > 1:
        pos = np.arange(n, dtype=float)
        g = delta + (1.0 - delta) * np.power((n - pos) / (n - 1.0), eta)
    else:
        g = np.ones(1)

    # (3) MULTIPLICATIVE combination: the gate scales the distrust
    # bracket cue-by-cue. Where the bracket is flat (tied validities)
    # this reduces to the (floored) primacy gradient; where the bracket
    # nearly vanishes (trusted high-validity early experts) the gate
    # multiplies almost nothing and the distrust kernel decides where
    # the remaining weight mass sits. With tau lowered, the flat share
    # (1-tau)/n is large enough that trusted high-validity LATE experts
    # retain meaningful weight through the gate's floor residue -- the
    # structural fix for the under-weighted late-cue cluster (Exp6/8)
    # that no amount of gate flattening could rescue at tau >= 0.90.
    bracket = (1.0 - tau) / n + tau * w_inv
    raw = g * bracket
    Z_t = float(raw.sum())
    if Z_t > 1e-12:
        w = raw / Z_t
    else:
        w = np.full(n, 1.0 / n)

    # (4) Deterministic attention: every discriminating cue is always
    # attended; no capacity interference, beta is design-independent.
    s = np.sign(a - b)

    # (5) Additive integration of all discriminating cues.
    d = float(np.dot(w, s))

    # Numerically stable logistic choice rule.
    p_a = 0.5 * (1.0 + np.tanh(0.5 * beta * d))
    probs = np.array([p_a, 1.0 - p_a], dtype=float)

    # (6) Small uniform lapse; kept below IVC's range because the gate
    # itself supplies response dilution on late-cue trials.
    probs = (1.0 - eps) * probs + eps * 0.5
    return probs
```

**`policy source code`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # guard against float drift
    return int(np.random.choice(len(probs), p=probs))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This design exploits the core structural difference between Take-The-Best (TTB) and Tallying: TTB's choice (and its predicted choice probability, sigmoid(beta)) is determined solely by the single highest-validity discriminating cue and is therefore INVARIANT to how many lower-validity cues oppose it, whereas Tallying's predicted probability is a softmax over win counts and is therefore GRADED in the tally margin. I use 5 features with a wide, strictly ordered validity spread [0.95, 0.85, 0.75, 0.65, 0.55] so the cue hierarchy is unambiguous. The critical trials (T1-T4 and their mirrors T5-T8) hold the TTB-relevant structure fixed — the top cue (validity 0.95) always discriminates in favor of the same option — while the tally margin against that option grows from 0 (tie) to 1, 2, and 3. TTB predicts a FLAT probability of choosing the top-cue winner across all four margin levels (identical sigmoid(beta) - epsilon mixture), while Tallying predicts a monotonically DECLINING curve: 0.5, sigmoid(-beta), sigmoid(-2beta), sigmoid(-3beta). This is a quantitative, parametric signature: the slope of P(top-cue winner) vs. tally margin is exactly zero under TTB and strongly negative under Tallying. Trials T9-T12 repeat the same logic with the top cue tied and the second cue (0.85) discriminating, and T15-T16 with the third cue (0.75) discriminating, testing whether the dissociation persists deeper in the cue hierarchy (TTB predicts the same flat sigmoid(beta) at every cue level; Tallying's predictions depend only on counts, not on which cue discriminates). Trials T13-T14 are agreement trials (both heuristics favor the same option) serving as a baseline. Every conflict type appears in both A-favored and B-favored versions to absorb side/position bias. With 16 unique pairs, K = 96 // 16 = 6 repetitions yields exactly 96 trials per subject. Predicted dissociation summary: (1) on tally-tie trials with a discriminating top cue (T1, T5, T9, T11), TTB predicts above-chance preference for the cue winner while Tallying predicts exactly 50/50 regardless of beta; (2) across the margin gradient (T1->T4), TTB predicts zero slope, Tallying predicts a steep negative slope; (3) TTB is indifferent to which cue in the hierarchy discriminates (same probability at f0, f1, f2 levels for matched tally structure), while any validity-sensitive weighting would break this invariance.

[1] This design's core novelty is a cue-allegiance flip at FIXED tally counts — the mirror image of the classic margin-gradient design. Tallying's predicted probability depends ONLY on the win counts through a softmax, which is shift-invariant (softmax over [3,2] equals softmax over [1,0]); it is therefore EXACTLY invariant to which features each option wins on and to cue depth. TTB's choice, in contrast, is determined solely by the allegiance of the highest-validity discriminating cue. The factorial core (trials 1-8) crosses tally margin {1, 2} x top-cue allegiance {pro-tally (agree), anti-tally (conflict)}: (i) C+ vs C- (and H+ vs H-) hold the win counts and even the tie placement identical (f0 tied, counts 3-2) while permuting cue allegiances so the top discriminating cue flips from favoring the tally winner to opposing it. Tallying predicts a precisely zero change in P(A) (identical sigma(beta)); TTB predicts a full flip from sigma(beta) to sigma(-beta). (ii) C vs H holds allegiance fixed and doubles the margin: TTB predicts exactly zero change (its probability is sigma(beta) for the top-cue winner regardless of how many cues oppose it), while Tallying predicts sigma(beta) -> sigma(2beta). The two theories thus occupy orthogonal cells of the 2x2: TTB shows a main effect of allegiance only; Tallying a main effect of margin only. The depth series (trials 9-14) places the anti-tally win at f0 (v=.95), f2 (v=.85), or f3 (v=.65), all at margin 1 with counts (3,2) or (2,1): both theories predict FLAT behavior here (TTB's probability is cue-depth invariant; Tallying's softmax is shift-invariant across (3,2)/(2,1)), so any validity-weighted hybrid would break this double invariance — a sharp signature that neither target theory but only mixtures would produce. The extreme anchors (trials 15-16) are 1-vs-5 splits where the top cue alone opposes five cues: TTB predicts P(cue winner) = sigma(beta) — numerically IDENTICAL to its prediction on the margin-1 conflict trials (invariance to 4 extra opposing cues), while Tallying predicts sigma(4beta) for the tally winner, near-deterministic at any reasonable beta. Quantitative dissociation summary: (1) on the allegiance flip (C+/C-, H+/H-), Tallying's predicted slope is exactly 0 and TTB's is -tanh(beta/2); (2) on the margin manipulation (C vs H), TTB's predicted slope is exactly 0 and Tallying's is positive; (3) across the depth series both are flat, jointly excluding weighted hybrids; (4) E1 vs C- isolates TTB's cue-count invariance against Tallying's steep margin gradient. Mirrors of every trial type absorb side/position bias. With 16 unique pairs, K = 96 // 16 = 6, yielding exactly 96 trials per subject.

[2] This design dissociates NVW (validity-weighted, capacity-limited integration with one-reason lapses) from Tallying by exploiting the fact that Tallying's predicted P(A) depends ONLY on the win-count margin: for any beta and epsilon, Tallying assigns exactly ONE probability to every trial with the same (a_wins, b_wins), and exactly 0.5 to every tie trial. NVW instead assigns a graded probability ordered by the validity-weighted evidence difference d = sum(v^gamma over A-won cues) - sum(v^gamma over B-won cues), plus a small one-reason tilt toward the highest-validity discriminating cue. I use n=5 (one cue beyond the working-memory span, so the interference factor delta is active but integration remains informative: beta_eff = beta*delta in [0.9, 3.3]) with a wide validity spread [0.95, 0.85, 0.75, 0.65, 0.55]. The design contains four dissociation axes, each with mirrored (A/B-swapped) versions to absorb side bias: (1) MARGIN-0 VALIDITY GRADIENT (types 1-5): 2-2 tie trials where the validity allocation varies from strongly pro-A (d=+0.60: A wins the two most valid cues) through d=+0.40, +0.20, 0.00, to a top-cue-conflict probe (d=+0.10 with the top cue favoring B). Tallying predicts EXACTLY 0.5 on all five types for every parameter value; NVW predicts a monotone deviation from 0.5 whose sign is parameter-free (because v^gamma preserves validity order for any gamma>0, A winning {f0,f1} vs B winning {f3,f4} yields d>0 at any gamma in [0.7,1.3]), with magnitude sigma(beta_eff*d) roughly 0.61-0.93 at d=0.60. (2) WITHIN-MARGIN-1 SPREAD (types 6-14): nine 3-2 trials spanning d from 1.35 down to 0.15. Tallying predicts an identical probability sigma(beta) on ALL nine; NVW predicts a monotone spread of at least ~0.12 (and up to ~0.5) across the aligned subseries (types 6-10, where the top cue also favors A, so both the integration and one-reason components agree). (3) ORDER REVERSAL: NVW predicts P(type 1; margin 0, d=0.60) > P(type 14; margin 1, d=0.15) strictly for every admissible parameter setting (the d-gap is positive at all gamma, and the one-reason lapse favors A on type 1 but B on type 14), whereas Tallying predicts the strict opposite ordering (0.5 < sigma(beta)). The sign of P(margin-0, high-d) - P(margin-1, low-d) is therefore a decisive, parameter-free test. (4) MATCHED-d ALLEGIANCE FLIP: types 9 vs 11 (both d=0.75) and 10 vs 12 (both d=0.55) hold counts AND the validity-weighted evidence identical while flipping which option wins the top cue; Tallying predicts exactly zero difference, NVW predicts a small positive difference (the one-reason lapse signature, magnitude up to ~0.07). Types 15-16 are margin-2 anchors probing the steepness of the margin gradient (Tallying: sigma(2beta); NVW: sigma(beta_eff*1.90) vs sigma(beta_eff*1.00), itself a within-margin spread Tallying cannot produce). In total Tallying predicts only THREE distinct choice probabilities across all 32 pairs (margin 0, 1, 2 and their complements), while NVW predicts a graded profile ordered by d with allegiance modulations - a sharp structural signature. With 32 unique pairs, K = 96 // 32 = 3, giving exactly 96 trials per subject.

[3] This design attacks NVW's central and most falsifiable structural claim: capacity interference. NVW asserts that beyond a working-memory span of ~4 cues, each additional pooled cue multiplies the effective inverse temperature by delta < 1, so choice consistency collapses as feature count grows. Tallying has no capacity limit: its softmax inverse temperature beta is design-independent, and its predicted probability depends only on win counts. I therefore set n_features = 8 (four cues beyond the span), which drives NVW's effective sensitivity to beta_eff = beta * delta^4 <= 11 * 0.30^4 = 0.089. Since the maximum possible validity-weighted evidence difference across all trials is |d| <= sum(v^gamma) <= 6.34 (at gamma = 0.7), NVW's predicted P(A) is HARD-BOUNDED to roughly [0.37, 0.64] on every single trial, for every admissible parameter setting (the one-reason lapse uses the same tiny beta_eff, and epsilon/omega only shrink deviations further; attention lapses at alpha <= 1 compress it even more). Tallying, in contrast, predicts a steep, margin-graded ladder: sigmoid(beta), sigmoid(2*beta), sigmoid(4*beta), sigmoid(6*beta) across the margin gradient G1->G4, and near-deterministic choice on the anchor trials E1/E2 (e.g., sigmoid(8*beta) >= 0.98 for beta >= 0.5). The design thus yields three decisive, parameter-free signatures: (1) CAPACITY CLIFF: observed choice proportions on unambiguous anchors (E1: 8-0 split; E2: 7-1 split) exceeding ~0.7 are impossible for NVW at ANY admissible parameter combination, while trivially accommodated by Tallying; likewise, a spread across the margin ladder (G1 vs G4) exceeding ~0.12 is impossible for NVW (its maximum ladder spread is sigmoid(0.089*4.73) - sigmoid(0.089*0.98) ~ 0.08) but natural for Tallying. (2) FIXED-COUNT VALIDITY GRADIENT: trials S1, S2, S3 all have identical win counts (4,3) but validity-weighted evidence d = +1.63, +0.99, -0.37 (order preserved for all gamma in [0.7, 1.3], verified at both endpoints, since v^gamma preserves validity order). Tallying predicts EXACTLY identical probabilities on all three; NVW predicts a strictly decreasing series with S3 below 0.5. Similarly, the tie trials Z1 (d = +1.11), Z2 (d = +0.25), Z3 (d = -0.19) all have counts (4,4): Tallying predicts exactly 0.5 on all three; NVW predicts a strictly ordered series crossing 0.5 between Z2 and Z3. (3) MATCHED-COUNT ALLEGIANCE: G4 and E2 both have counts (7,1) — Tallying predicts identical probabilities; NVW predicts P(G4) > P(E2) strictly, because both the integrated evidence (d = 4.73 vs 3.81) and the one-reason component (top cue favors the tally winner in G4, opposes it in E2) align. Conversely, if NVW is correct, subjects should be near-chance everywhere with only a small d-ordered ripple — a profile Tallying can only mimic by taking beta near 0, which would then destroy the margin ladder Tallying needs elsewhere; the joint pattern (near-chance + d-ordering at fixed counts) is uniquely NVW, while (steep ladder + exact flatness at fixed counts) is uniquely Tallying. Every trial type appears in A/B-mirrored pairs to absorb side and position bias. With 24 unique pairs, K = 96 // 24 = 4, giving exactly 96 trials per subject.

[4] STRUCTURAL LEVER. With validities fixed per feature, both theories reduce to a fixed position-indexed weight profile applied to the cue-allegiance pattern s_j = sign(A_j - B_j): NVW uses w_j ∝ v_j^gamma (validity-VALUE based), while SWC uses w_j = (1-lambda)/n + lambda[(1-rho)vnorm_j + rho*(j/(n-1))^eta] (position-RECENCY based, with recency necessarily dominating: lam*rho >= 0.64 versus a validity coefficient <= 0.20 and flat <= 0.20 across SWC's entire box). I therefore present validities in STRICTLY DECREASING order [0.95, 0.90, 0.85, 0.62, 0.56, 0.52] — SWC's own assumed presentation format, i.e., its home turf — with a sharp validity gap between positions 3 and 4 creating a 'high-validity early block' {0.95, 0.90, 0.85} and a 'low-validity late block' {0.62, 0.56, 0.52}. This makes the two weight profiles strongly ANTI-CORRELATED: NVW's weights fall with position, SWC's rise (convexly) with position. n_features = 6 also places two cues beyond NVW's span-4, engaging its capacity-interference signature (beta_eff = beta*delta^2 in [0.16, 0.99], i.e., NVW itself predicts compressed confidence), while SWC's beta is design-independent.

DISSOCIATION AXES (all predicted boxes verified at the corners of BOTH parameter spaces; NVW's d-sign is invariant over gamma in [0.7, 1.3] because v^gamma is monotone in v, and SWC's recency dominance holds over its whole box):

(1) SIGN-FLIP FAMILY (S1, S2, S3, E1, E3, T3, T6, T8 + mirrors; 16 of 24 trials). The early high-validity block favors one option and the late low-validity block the other. NVW predicts the early-block (validity) winner; SWC predicts the late-block (recency) winner — PARAMETER-FREE OPPOSITE SIGNS. Headline boxes for P(A): S1 (A wins only expert 6 vs expert 1): NVW 0.39-0.51, SWC 0.60-0.85. S2 (A wins experts 5-6 vs 1-2): NVW 0.28-0.48, SWC 0.64-0.91. S3 (A wins experts 4-6 vs 1-3): NVW 0.25-0.47, SWC 0.63-0.91. E1 — the most damning trial for SWC: B wins the FOUR MOST VALID experts and A only the two least valid, yet SWC still predicts A (P(A) in [0.55, 0.88]) because recency weight on the last two positions outweighs everything; NVW predicts B (P(A) in [0.10, 0.41]). E3: A wins the two top experts vs B's single bottom expert — SWC predicts B (P(A) in [0.18, 0.41]) because one late cue outweighs two top cues; NVW predicts A (0.55-0.80). T6/T8 extend the flip to mixed patterns (boxes: NVW 0.52-0.71 vs SWC 0.09-0.37; NVW 0.44-0.50 vs SWC 0.62-0.89).

(2) VALIDITY-TIE / RECENCY-ONLY (T1, T1m). A wins experts 1 and 6, B wins experts 3 and 4: the validity-weighted evidence is exactly balanced (|d| <= 0.04 for all gamma), so NVW predicts chance (P(A) in [0.50, 0.55], the tiny upward tilt being its one-reason lapse toward the top cue), while SWC's recency gradient (expert 6 vs experts 3-4) predicts P(A) in [0.58, 0.82]. Non-overlapping boxes: observed P(A) <= 0.55 falsifies SWC's entire parameter space.

(3) FIVE-VS-ONE (E2, E2m). The five most valid experts versus the single least valid expert: NVW predicts the validity winner with P in [0.62, 0.97]; SWC's flat + vnorm components nearly cancel its recency component, so its box STRADDLES chance (P(validity winner) in [0.42, 0.58]). Observed P > 0.6 for the five-expert side is impossible for SWC at any admissible parameter combination.

(4) CUE-MIGRATION PAIR (S3 vs S4). Moving expert 3 (v = 0.85) from the early side to the late side flips NVW's sign (d: -1.00 to +0.70; predicted P(A) crosses 0.5 between adjacent trial types) while leaving SWC's prediction on the SAME side (d approx 1.0 to 1.3, both strongly pro-late-block). A sign-crossing between S3 and S4 is a qualitative signature only NVW produces.

(5) AGREEMENT ANCHORS (A1/A1m total splits; S4/S4m). Both theories agree; they establish that subjects can perform the task and calibrate overall consistency. Note the confidence dissociation: because of capacity interference, NVW's maximum predicted probability on any non-total trial is approx 0.8-0.9, whereas SWC predicts up to 0.93-0.97 on recency-unbalanced trials — the joint profile 'validity-ordered choices at moderate consistency' is uniquely NVW; 'recency-ordered choices at high consistency' is uniquely SWC.

Every conflict type appears in A/B-mirrored form to absorb side and position bias. With 24 unique pairs, K = 96 // 24 = 4 repetitions in randomized order = exactly 96 trials. Summary of decisive, parameter-free signatures: (i) on 16 sign-flip trials the theories predict opposite choices with non-overlapping probability boxes; (ii) on T1 NVW predicts exactly chance where SWC's box excludes chance; (iii) on E2 SWC's box is confined to near-chance where NVW predicts high confidence; (iv) the S3->S4 sign crossing is NVW-only; (v) regressing P(A) on validity-weighted evidence (sum v_j*s_j) versus recency-weighted evidence (sum (j/5)^3*s_j) — anti-correlated by construction — identifies which profile subjects use, with opposite regression signs under the two theories.

[5] GOAL: dissociate SWC (convex cue-order recency, deterministic attention, no capacity limit) from NVW (validity^gamma weighting, capacity interference, span ~4). Prior design [4] used strictly decreasing validities, making recency and validity ANTI-correlated everywhere. This design takes the opposite, novel route: a NON-MONOTONE validity profile [0.60, 0.96, 0.60, 0.60, 0.60, 0.60, 0.90] with two high-validity peaks at positions 2 and 7 and five equal low-validity cues. This decorrelates validity from recency (rather than anti-correlating it), so each axis can be isolated in pure form, and it lets the CONVEXITY of SWC's recency gradient (eta in [2.5,4.5]: the last cue's weight is ~2x the second-to-last's and ~10x the third-to-last's) fight NVW's validity MASS head-to-head. n_features = 7 also puts three cues beyond NVW's span-4, engaging its capacity interference (beta_eff = beta*delta^3 <= 0.30), while SWC's beta is design-independent. All boxes below were verified at the corners of BOTH parameter spaces.

AXIS 1 - EQUAL-VALIDITY RECENCY LADDER (T4/T5/T6 + mirrors): A wins expert 1 (60%), B wins exactly one equally-valid (60%) expert at position 4, 5, or 6. NVW's evidence is EXACTLY zero for every parameter value (equal validities => equal weights v^gamma, symmetric attention/lapses), so NVW predicts exactly 50/50 on all six trials - parameter-free. SWC's recency gradient predicts a monotone ladder below 0.5 for the early-cue option: P(A) boxes [0.42,0.49] (vs p4), [0.30,0.43] (vs p5), [0.17,0.31] (vs p6). Observed P(later expert's product) > 0.60 on T6 is impossible for NVW at ANY admissible parameter combination, and required by SWC. This is the single cleanest trial family: pure recency with validity perfectly controlled.

AXIS 2 - HEAD-TO-HEAD DUELS (D1/D2 + mirrors): D1: A wins expert 2 (96%), B wins expert 7 (90%). NVW follows validity: P(A) in [0.50, 0.53]. SWC's convex recency makes the LAST cue dominate: P(A) in [0.14, 0.38]. Opposite signs, non-overlapping boxes - the subject prefers the product endorsed by the slightly LESS accurate but LATER expert iff SWC is true. D2 sharpens it: 96% expert 2 vs 60% expert 6 - a huge validity gap that NVW respects (P(A) in [0.50,0.54]) but SWC's recency overrides (P(A) in [0.31,0.46]). These flips survive even if NVW's capacity interference were removed (delta=1): NVW would just become more confident in the validity winner; the sign disagreement is structural.

AXIS 3 - CONVEXITY vs VALIDITY MASS (C1, L1-L4, E2 + mirrors): C1: A wins expert 7 (90%) alone; B wins experts 2+4 (96%+60%). NVW: validity mass favors B, P(A) in [0.43,0.50]; SWC: the last cue's convex weight beats both, P(A) in [0.59,0.80]. The L-ladder measures the crossover count: A wins expert 7 vs B wins k of the mid cues {3,4,5,6}. NVW's validity arithmetic says the 90% cue beats exactly ONE 60% cue (d=+0.23 at k=1, d=-0.47 at k=2), so NVW crosses 0.5 between L1 and L2. SWC's convex gradient says the last cue outweighs three mid cues (recency sum at k=3 is 0.40-0.60 < 1.0), crossing only at k=4 - a measured crossover-count dissociation of 2-3 opposing cues. E2 is the block-level flip: A wins the 3 late cues (incl. only the 90%), B wins 4 early cues (incl. the 96%): NVW says B (P(A) in [0.42,0.50]), SWC says A (P(A) in [0.71,0.94]).

AXIS 4 - CAPACITY CLIFF / CONFIDENCE (E1 + mirror): A wins experts 4-7, B wins 1-3. Both theories favor A in direction, but NVW's beta_eff <= 0.30 caps its P(A) at ~[0.50,0.56] on EVERY trial in this design (max |beta_eff*d| ~ 0.55), while SWC predicts [0.75,0.95]. Observed accuracy > 0.70 on E1 falsifies NVW's entire parameter space; near-chance performance everywhere with a residual validity-ordering falsifies SWC.

Aggregate signature: regressing P(A) on recency-weighted evidence sum((j/6)^3.5 * s_j) versus validity-weighted evidence sum(v^gamma * s_j) - decorrelated here by construction (expert 2: high validity, negligible recency; expert 6: low validity, high recency) - yields opposite-loading profiles under the two theories. Every trial type appears in A/B-mirrored form to absorb side/position bias. 24 unique pairs, K = 96//24 = 4 repetitions in randomized order = exactly 96 trials per subject.

[6] STRUCTURAL LEVER. IVC's weights are a function of the validity VALUES ONLY (w_j proportional to (1-v_j)^kappa: the LESS valid the expert, the MORE weight), with zero position sensitivity. SWC's weights are dominated by POSITION (convex recency, coefficient lam*rho >= 0.64, plus a weak POSITIVE validity component <= 0.20). I therefore use a non-monotone, duplicated validity profile v = [0.55, 0.90, 0.60, 0.95, 0.55, 0.60] (n=6): the two lowest validities sit at positions 1 and 5, duplicated 0.55s at positions 1/5 and duplicated 0.60s at positions 3/6, and the two highest validities at positions 2 and 4. The duplicates create validity-MATCHED comparisons at different positions (killing the validity axis and isolating pure recency), and the low-early/high-late arrangement makes IVC's weight profile (early-heavy) anti-aligned with SWC's (late-heavy). All boxes below were verified at the corners of BOTH parameter spaces.

AXIS 1 - VALIDITY-MATCHED RECENCY BATTERY (EQ1-EQ4 + mirrors). EQ1: A wins expert 1 vs B wins expert 5 (0.55 vs 0.55). EQ2: expert 3 vs expert 6 (0.60 vs 0.60). EQ3: A wins {1,3} vs B wins {5,6} (validity multiset {0.55,0.60} on BOTH sides). EQ4: A wins {1,6} vs B wins {3,5} (same matched multiset). Because IVC's weights depend only on validity values, d = 0 EXACTLY on all four -> IVC predicts P(A) = 0.50 PARAMETER-FREE. SWC's recency gradient predicts a wide spread: P(A) boxes [0.28,0.45] (EQ1), [0.12,0.39] (EQ2), [0.08,0.34] (EQ3), [0.61,0.69] (EQ4). Observed P(EQ3) <= 0.40 or P(EQ4) >= 0.55 is impossible for IVC at ANY parameter combination; observed P(EQ3) >= 0.45 is impossible for SWC. SWC requires an EQ3-vs-EQ4 spread >= 0.27 where IVC requires exactly 0.

AXIS 2 - POSITION-TRANSFER PAIRS (IVC invariance vs SWC flip). TP1: A wins expert 1 (0.55) vs B wins expert 6 (0.60). TP2: A wins expert 5 (0.55) vs B wins expert 3 (0.60). Identical validity allegiances, opposite position allegiances: IVC predicts EXACTLY the same P(A) on both, in [0.52, 0.55]; SWC predicts P(TP1) in [0.12, 0.37] (expert 6's recency dominates) and P(TP2) in [0.54, 0.69] (expert 5 beats expert 3 on recency) - a required pair difference >= 0.17 where IVC requires exactly 0. The same logic at the extreme validity gap: TP3 (A wins expert 1 vs B wins expert 4, 0.55 vs 0.95), TP4 (A wins expert 5 vs B wins expert 4), and M1 (A wins {1,3} vs B wins {4,6}, where the two 0.60 cues cancel) all have d = w(0.55) - w(0.95) under IVC -> P(A) IDENTICAL in [0.59, 0.66] on all three; SWC predicts [0.32,0.47], [0.50,0.57], [0.08,0.33] respectively - a required spread >= 0.17 (TP4 vs M1). TP3'/TP4' repeat this with the 0.90 expert: IVC-equal [0.59,0.66]; SWC [0.42,0.50] vs [0.52,0.69]. Any observed spread within a transfer family (e.g., P(TP1) != P(TP2), or P(TP3) != P(M1)) falsifies IVC's position-invariance; flatness within families falsifies SWC's recency gradient.

AXIS 3 - SIGN-FLIP DUELS. On TP1, TP3, TP3', S3 (A wins expert 3 (0.60) vs B wins expert 4 (0.95) - an ADJACENT-position duel where the later cue is also more valid), M1 and M2 (A wins {1,3,5}, the three least valid experts, vs B wins {2,4,6}), IVC favors the early/low-validity option and SWC favors the late/high-validity option with NON-OVERLAPPING boxes: IVC [0.52,0.55] vs SWC [0.12,0.37] (TP1); IVC [0.59,0.66] vs SWC [0.32,0.47] (TP3); IVC [0.59,0.66] vs SWC [0.42,0.50] (TP3'); IVC [0.57,0.61] vs SWC [0.38,0.48] (S3); IVC [0.59,0.66] vs SWC [0.08,0.33] (M1); IVC [0.63,0.73] vs SWC [0.12,0.37] (M2). M2 is the headline block flip: the option endorsed by the three WORST experts (55/60/55) is chosen confidently only by IVC; SWC confidently picks the option endorsed by the later, more valid experts.

AXIS 4 - CUE MIGRATION (AN2 -> M2). AN2: A wins {1,3,5} vs B wins {2,4}. Adding expert 6 (a mere 60%-valid cue) to B's side yields M2. IVC predicts only a modest drop (AN2 [0.70,0.78] -> M2 [0.63,0.73], still confidently A, since expert 6's inverted weight is only ~0.21 and expert 3's cancels it); SWC predicts a catastrophic flip (AN2 [0.43,0.66] -> M2 [0.12,0.37]) because expert 6 carries the maximal recency weight. A drop exceeding ~0.3 on this single-cue migration is impossible for IVC and required by SWC.

AXIS 5 - REVERSE CONFIDENCE DISSOCIATION. AN2 (A wins {1,3,5} vs B wins {2,4}): IVC [0.70,0.78] vs SWC [0.43,0.66] - IVC more confident (and SWC's box straddles chance). AN3 (A wins {5,6} vs B wins {1,2}): IVC [0.56,0.60] vs SWC [0.66,0.86] - SWC MORE confident. The confidence ordering between AN2 and AN3 reverses between theories: IVC says P(AN2) >> P(AN3); SWC says P(AN3) > P(AN2). TOT (total split) is the agreement anchor both theories accommodate.

PARAMETRIC SIGNATURE. IVC predicts only ~7 distinct probability levels across all 32 pairs (0.50 on all EQ; one level per validity-allegiance multiset for the TP/S/M/AN families; near-1 on TOT), with EXACT equalities within families; SWC predicts a wide, position-ordered spread that breaks every equality. Regressing P(A) on inverted-validity evidence sum((1-v_j)^2.7 * s_j) versus recency evidence sum((j/5)^3.5 * s_j) yields opposite-loading profiles: the two evidence axes are decorrelated by construction (expert 1: maximal inverted weight, zero recency; expert 6: maximal recency, middling inverted weight; experts 2/4: high validity, i.e., low weight for IVC but positive validity weight for SWC). Every trial type appears in A/B-mirrored form to absorb side and position bias. 32 unique pairs, K = 96 // 32 = 3 repetitions in independently randomized order = exactly 96 trials per subject.

[7] STRUCTURAL LEVER. SWC's weights are dominated by POSITION (convex recency, coefficient lam*rho >= 0.64, plus a weak POSITIVE validity component <= 0.20); IVC's weights are a function of the validity VALUES ONLY through the distrust kernel (1-v_j)^kappa, with zero position sensitivity. Prior designs anti-correlated the two profiles via strictly DECREASING validities ([4], vs NVW) or broke them apart with non-monotone duplicated validities ([6]). This design takes the unexplored route: STRICTLY INCREASING validities v = [0.55, 0.60, 0.68, 0.80, 0.90, 0.96]. Now SWC's recency gradient and its weak validity component ALIGN (both favor late cues), so SWC's weight profile is late-heavy at EVERY corner of its box and its predictions are maximally extreme; IVC's inverted kernel is early-heavy (the 55% expert alone carries ~42% of total weight at kappa=2.4). The two weight profiles are therefore near-perfectly ANTI-ALIGNED on every conflict trial. All boxes below were verified at the corners of BOTH parameter spaces; the sign arguments are parameter-free (IVC's d for any 1-vs-1 duel is tau*(inv_A - inv_B)/Z, strictly decreasing in the early cue's validity since the kernel is monotone and flats cancel; SWC's duel d = w_j - w_6 is strictly increasing in j because BOTH vnorm_j and pos_j increase with j for every admissible parameter combination).

AXIS 1 - VALIDITY-GAP LADDER, OPPOSITE-SIGNED SLOPES (L1-L5 + mirrors). A wins exactly one early cue (c1..c5) against B's win of the last cue c6 (96%). IVC predicts a monotone DECREASING series in P(A): 0.64-0.75 (c1, 55%), 0.59-0.68 (c2), 0.55-0.60 (c3), 0.52-0.54 (c4), 0.50-0.51 (c5) - as the early opponent becomes more valid, its inverted weight shrinks toward c6's. SWC predicts a monotone INCREASING series: 0.15-0.35, 0.15-0.34, 0.15-0.36, 0.18-0.40, 0.20-0.44 - as the early opponent becomes later, its recency weight grows. Every adjacent rung has non-overlapping boxes, and the two theories demand opposite-signed slopes across the whole ladder - a parameter-free quantitative signature. L5 is the sharpest single trial: the two BEST experts (90% vs 96%) duel; IVC predicts exactly chance (both inverted weights are tiny), SWC predicts P(later expert's product) in [0.56, 0.80].

AXIS 2 - CROSSOVER COUNT AGAINST THE LAST CUE (L1, M3, M4 + mirrors). How many early-cue wins does it take to flip choice toward the early option? IVC's answer: k* = 1 (P(A) already >= 0.64 with the single 55% cue, L1). SWC's answer: k* >= 5 or never - with FOUR early cues (M3: A wins c1-c4 vs c6) SWC still predicts P(A) in [0.18, 0.44] while IVC predicts [0.82, 0.90]; with ALL FIVE early cues (M4) SWC's box only reaches [0.29, 0.61] (it straddles chance because the last cue's convex recency weight rivals five early cues combined) while IVC predicts [0.82, 0.92]. The measured crossover point separates the theories by 3-4 cues.

AXIS 3 - DOUBLE DISSOCIATION IN CONFIDENCE (W1 vs M4 + mirrors). W1: the product endorsed by the five LATER, MORE valid experts (incl. 90% and 96%) vs the single WORST expert (55%, first position). M4: the product endorsed by the five EARLIER, LESS valid experts vs the single BEST expert (96%, last position). IVC: W1 in [0.52, 0.60] (the lone 55% expert soaks up ~42% of the weight) but M4 in [0.82, 0.92] -> P(W1) - P(M4) is NEGATIVE everywhere (<= -0.21). SWC: W1 in [0.79, 0.97] but M4 in [0.29, 0.61] -> the difference is POSITIVE everywhere (>= +0.18). Opposite-signed differences with disjoint boxes on both trials; note M4 is deliberately SWC-risky (observed P(A) >= 0.65 on M4 falsifies SWC's entire space), making the design genuinely adversarial rather than one-sided.

AXIS 4 - BLOCK FLIP (B1 + mirror). A wins the three early low-validity experts, B the three late high-validity experts. IVC: [0.77, 0.90] for the early block; SWC: [0.05, 0.28]. Non-overlapping.

AXIS 5 - ADJACENT MICRO-DUELS (A1, A2 + mirrors). c3(68%) vs c4(80%) and c4(80%) vs c5(90%): small but sign-clean opposite tilts. IVC: P(earlier/lower-validity product) in [0.55, 0.58] (A1) and [0.50, 0.53] (A2); SWC: [0.42, 0.48] and [0.39, 0.45]. Parameter-free opposite signs on every adjacent duel - a micro-sign series that neither theory can reverse anywhere in its box.

E1/E1m are total-split agreement anchors establishing that subjects can perform the task. Aggregate signature: regressing P(A) on inverted-validity evidence sum((1-v_j)^2.7 * s_j) versus recency evidence sum((j/5)^3.5 * s_j) yields opposite-loading profiles, since the two evidence axes are anti-correlated by construction here. IVC predicts a validity-allegiance-determined probability profile (~8 distinct levels, invariant to which positions carry which allegiances); SWC predicts a position-ordered profile that breaks every such invariance. Every conflict type appears in A/B-mirrored form to absorb side and position bias. 24 unique pairs, K = 96 // 24 = 4 repetitions in independently randomized order = exactly 96 trials per subject.

[8] STRUCTURAL LEVER. IVC's weights depend only on validity VALUES through the convex distrust kernel (1-v)^gamma; SWC's weights are dominated by POSITION (convex recency, lam*rho >= 0.64) with only a weak, min-max-normalized validity component. Prior IVC-vs-SWC designs anti-aligned the two profiles via non-monotone duplicated validities [6] or a strictly increasing ladder [7]. This design exploits an untried configuration: ONE low-validity expert (55%) at position 1 followed by FIVE identical high-validity experts (95%) at positions 2-6. The kernel ratio between the worst and each good expert is (0.45/0.05)^gamma in [~58, ~1130] across IVC's box, so expert 1 alone absorbs 97-99.8% of IVC's kernel mass (w1 in [0.83, 1.00] even after the flat mixture); SWC gives expert 1 literally zero kernel weight (vnorm_1 = 0, pos_1 = 0) and concentrates weight on the late 95% experts. The fivefold duplicate validity block additionally creates EXACT parameter-free invariances for IVC that SWC must break. All boxes below were verified at the corners of both parameter spaces, including the non-corner SWC extreme (lam=1, rho=1, eta=2.5) where mid-position recency gaps are largest.

AXIS 1 - WORST-EXPERT VETO (V0, V1, V3, V5 + mirrors). Expert 1 (55%) versus one, three, or all five of the 95% experts. IVC: d = w1 - (opposing good weights) in [0.67, 1.00] at every corner -> P(55%-expert's product) in [0.71, 0.95], nearly FLAT across the ladder. SWC: d < 0 strictly (g1 = 0, any good expert's g > 0) -> P in [0.41, 0.50) on V0, [0.15, 0.35] on V1, [0.03, 0.23] on V3, [0.01, 0.29] on V5. Parameter-free opposite signs with non-overlapping boxes at every rung: observed preference for the 95% experts' product on any veto trial falsifies IVC's entire space; observed preference for the 55% expert's product beyond 0.50 falsifies SWC's entire space.

AXIS 2 - EXACT-EQUALITY INVARIANCES (V0 = V1 = G2). Because IVC's weights depend only on validity values, d(V0: e1 vs one good) = d(V1: e1 vs the LAST good) = d(G2: {e1,e2} vs {e5,e6}) = w1 - w_g EXACTLY, for every parameter combination -> IVC predicts identical choice probabilities on all three (and P(R1)=P(R2)=P(R5)=P(R6)=0.50 exactly). SWC's recency gradient REQUIRES spreads: P(V0) - P(V1) >= 0.06 (up to 0.32), P(V0) - P(G2) >= 0.06 (up to 0.40). Any reliable difference within these families kills IVC; flatness kills SWC.

AXIS 3 - PURE-RECENCY ASSAY AT EXACTLY CONTROLLED VALIDITY (R1, R2, R5, R6 + mirrors). Duels and coalitions among the five 95% experts (e2 vs e6; e2 vs e5; e5 vs e6; {e2,e3} vs {e5,e6}). IVC assigns exactly equal weights to all five -> d = 0 -> P = 0.50 EXACTLY, parameter-free (the lapse preserves 0.5). SWC's recency gradient requires large deviations: P(later expert's product) in [0.62, 0.85] on R1, [0.55, 0.72] on R2, [0.55, 0.75] on R5, [0.67, 0.92] on R6. Any within-block position gradient above ~0.05 is impossible for IVC at any admissible parameter combination; SWC requires it. This is SWC's home-turf cell, making the design genuinely two-sided.

AXIS 4 - COUNT-VS-RECENCY KNIFE (M2', M1 + mirrors). M2' ({e2,e3} vs {e6}): two equal-validity experts versus the single last expert. IVC: d = w_g > 0 strictly -> P(majority) in [0.50, 0.54]. SWC: w2+w3 < w6 at every corner -> d < 0 strictly -> P(majority) in [0.16, 0.42]. Parameter-free opposite signs. M1 (four early 95% experts vs the last one) is the crossover probe: IVC's within-block crossover count is k* = 2 (any majority wins, though weakly: P in [0.50, 0.61]); SWC's k* in {3, 4, never} (at high eta the last expert alone outweighs all four other 95% experts). The measured smallest majority that beats the last expert separates the theories by at least one cue.

AXIS 5 - ANCHOR (TOT). Total split: both theories confidently agree (IVC [0.78, 0.95], SWC [0.84, 0.99]), establishing task comprehension.

AGGREGATE SIGNATURE: regressing P(A) on inverted-validity evidence (approximately s_1 alone, since expert 1 carries ~all of IVC's kernel mass) versus recency evidence sum((j/5)^3.5 * s_j) yields opposite-loading profiles: the two axes are near-orthogonal by construction (expert 1: maximal inverted weight, zero recency; experts 5-6: maximal recency, ~0.5% of inverted mass). Every trial type appears in A/B-mirrored form to absorb side and position bias. 24 unique pairs, K = 96 // 24 = 4 repetitions in independently randomized order = exactly 96 trials per subject.

[9] STRUCTURAL LEVER. IVC's weights are a function of validity VALUES ONLY (distrust kernel (1-v)^gamma), with zero position sensitivity; SWC's weights are dominated by POSITION (convex recency, lam*rho >= 0.64) with only a weak min-max validity component (<= 0.20). Prior IVC-vs-SWC designs used non-monotone duplicated validities [6], a strictly increasing ladder [7], or a single worst expert [8]. This design uses an untried PALINDROMIC validity profile v = [0.55, 0.95, 0.75, 0.75, 0.95, 0.55] (n=6). Because IVC's weights depend only on validity values, its weight profile is exactly POSITION-SYMMETRIC (w1 = w6, w2 = w5, w3 = w4), while SWC's is strictly position-graded and late-heavy. The palindrome therefore creates validity-MATCHED comparisons at every position separation, pinning IVC to exact, parameter-free point predictions that SWC's recency gradient must break. All boxes below were verified at the corners of both parameter spaces (SWC's eta corners 2.5/4.5 with lam*rho in [0.64, 1]; IVC's gamma/tau/c/beta/eps corners).

AXIS 1 - VALIDITY-MATCHED RECENCY LADDER (R1, R2, R3 + mirrors). Duels between validity-equal experts at maximal (e1 vs e6, both 55%), large (e2 vs e5, both 95%), and minimal (e3 vs e4, both 75%) position separation. IVC: d = 0 EXACTLY for every parameter combination -> P(A) = 0.50 parameter-free (the uniform lapse preserves 0.5). SWC: d = -lam*rho*(1 - (k/5)^eta-differences), giving P(A) boxes [0.15, 0.38] (R1), [0.28, 0.45] (R2), [0.42, 0.49] (R3). SWC REQUIRES a monotone ladder with within-subject spread P(R3) - P(R1) in [0.10, 0.32] (the ratio |d_R1|/|d_R3| >= 5.6 at every corner); IVC requires exactly 0 on all three. Observed P(later expert's product) > 0.55 on R1 is impossible for IVC at ANY admissible parameter combination and required by SWC.

AXIS 2 - VALIDITY-MATCHED COALITIONS (K1-K5, D4 + mirrors; 12 of 32 trials). Coalitions whose validity multisets match exactly (K1: {0.55,0.95} vs {0.55,0.95}; K2: {0.55,0.75} both; K3: {0.95,0.75} both; K4: {0.55,0.95,0.75} both; K5: {0.55,0.75} both; D4: interleaved {0.55,0.75,0.95} both). IVC predicts EXACTLY 0.50 on all twelve, parameter-free. SWC's recency gradient requires late-block preference with boxes P(A) in [0.07, 0.40] (K1), [0.12, 0.37] (K2), [0.22, 0.45] (K3), [0.05, 0.33] (K4 - the half-block flip, the headline trial), [0.17, 0.40] (K5), [0.23, 0.42] (D4). Any reliable deviation from 0.50 in the late direction on this family falsifies IVC's entire parameter space; flatness at 0.50 falsifies SWC's.

AXIS 3 - POSITION-TRANSFER PAIRS (T1a/T1b, T2a/T2b + mirrors). Same validity allegiance, opposite position allegiance. T1a: A wins the 55% expert at position 1 vs B wins the 75% expert at position 3; T1b: A wins the 55% expert at position 6 vs B wins the 75% expert at position 4. IVC: d = w(0.55) - w(0.75) on BOTH -> IDENTICAL predictions P(A) in [0.58, 0.84] on the pair. SWC: P(T1a) in [0.42, 0.50] (early low-validity cue loses to a mid cue) but P(T1b) in [0.65, 0.83] (the same low-validity cue, placed last, beats the mid cue) - a REQUIRED pair difference >= 0.15 where IVC requires exactly 0. T1a alone is also a sign dissociation: IVC > 0.58, SWC < 0.50. T2a/T2b repeat the logic with the 95% vs 75% experts (positions 2 vs 4, and 5 vs 3): IVC identical [0.40, 0.48] on both; SWC [0.47, 0.49] vs [0.57, 0.70], required spread >= 0.08. Any within-family difference kills IVC's position-invariance; flatness within families kills SWC's recency gradient.

AXIS 4 - IVC-DIAGNOSTIC CELLS (D1, D2 + mirrors; deliberately adversarial, two-sided design). D1: the 55% expert at position 1 vs the 95% expert at position 2. IVC's distrust inversion predicts subjects prefer the WORSE expert's product: P(A) in [0.63, 0.81]; SWC predicts [0.44, 0.50] (non-overlapping). D2 (worst-expert veto): the lone 55% first expert against FOUR better experts (95/75/75/95). IVC: P(55%-expert's product) in [0.52, 0.80]; SWC: [0.09, 0.41] (non-overlapping). Observed preference for the 55% expert beyond 0.50 on D1/D2 falsifies SWC's entire space - these cells make the design genuinely risky for the advocated theory rather than one-sided.

AXIS 5 - ANCHOR (TOT + mirror). Total split: both theories predict >= 0.69, establishing task comprehension and overall responsiveness.

PARAMETRIC SIGNATURE. IVC predicts only ~6 distinct probability levels across all 32 pairs (exactly 0.50 on the 18 validity-matched trials; one level per validity-allegation multiset on the transfer/duel trials; high confidence on TOT), with EXACT equalities within families. SWC predicts a position-ordered profile breaking every equality. Regressing P(A) on recency evidence sum((j/5)^3.5 * s_j) versus inverted-validity evidence sum((1-v_j)^2.7 * s_j) yields opposite-loading profiles: the two axes are decorrelated by the palindrome (expert 1: high inverted weight, zero recency; expert 6: equally high inverted weight, maximal recency - the transfer pairs exploit exactly this). Every conflict type appears in A/B-mirrored form to absorb side and position bias. 32 unique pairs, K = 96 // 32 = 3 repetitions in independently randomized order = exactly 96 trials per subject.

[10] STRUCTURAL LEVER. IVC and PW-IVC share the distrust-inversion kernel; they differ ONLY in that PW-IVC adds a convex primacy gradient concentrated on the first one or two positions (mixture share tau*rho in [0.14, 0.29]; positions 1-2 carry ~82-94% of the primacy kernel, and position 1 alone carries 0.53-0.72). IVC's weights are a function of validity VALUES ONLY -- strictly position-invariant. The design therefore manipulates POSITION while holding validity allegiances fixed, using the layout v = [0.95, 0.60, 0.60, 0.95, 0.60, 0.95], which (i) decorrelates validity from position (three 0.60 experts at positions 2,3,5; three 0.95 experts at positions 1,4,6), (ii) places a 0.95 expert at POSITION 1 -- the maximal-primacy slot carrying ~0.1% of IVC's kernel mass -- creating head-to-head primacy-vs-distrust conflicts, and (iii) triplicates each validity level, enabling validity-matched cells where IVC is pinned to exact point predictions. Because (0.40/0.05)^gamma is ~125-340, the three 0.60 experts absorb ~99.8% of the distrust kernel: every 0.60-vs-0.95 allegiance imbalance moves d by ~tau*(1/3) regardless of which positions are involved. All boxes below were verified at the corners of both parameter spaces, respecting the coupling between the kernel coefficient tau*(1-rho) and the primacy coefficient tau*rho.

AXIS 1 -- VALIDITY-MATCHED PRIMACY DOSE LADDER (trials 1-14). Duels and coalitions whose validity multisets match exactly (e.g., A wins the 0.95 expert at position 1 vs B the 0.95 expert at position 6; A wins {0.95@1, 0.60@2} vs B {0.60@5, 0.95@6}). IVC: d = 0 EXACTLY for every parameter combination -> P = 0.500 parameter-free on ALL 14 matched cells. PW-IVC: d = tau*rho*(primacy contrast) > 0 strictly, growing with position contrast: near-zero dose D3 (positions 4 vs 6) P in [0.50, 0.52]; low doses (0.60-duels, positions 2v3, 3v5, 2v5) P in [0.50, 0.63]; high dose D1 (position 1 vs 6) P in [0.53, 0.82]; max dose C1 ({1,2} vs {5,6}) P in [0.55, 0.86]. Observed preference >= 0.55 for the earlier expert's product on the D1/C1 family is impossible for IVC at ANY admissible parameter combination; flat 0.500 across the ladder falsifies PW-IVC. The slope of P versus primacy contrast is EXACTLY 0 under IVC and strictly positive under PW-IVC -- a parametric signature.

AXIS 2 -- FIVE-WAY EXACT EQUALITY (T1a/T1b/T1c/M7a/M7b + mirrors; trials 15-20, 25-28). All five hold validity allegiances fixed (T1: A wins one 0.60 vs B one 0.95; M7: A wins two 0.60 + one 0.95 vs B one 0.60 + two 0.95) while permuting positions, so d_IVC = tau*(w60 - w95) IDENTICALLY: IVC predicts ONE probability, box [0.59, 0.81], on all TEN cells. PW-IVC predicts a strict ordering for every admissible parameter set: M7a (A's wins at positions 1-3) [0.65, 0.91] > T1b (0.60@2 vs 0.95@6) [0.57, 0.90] > T1c (0.60@3 vs 0.95@4) [0.53, 0.87] > T1a (0.60@5 vs 0.95@1) [0.50, 0.79] > M7b (A's wins at 3,5,6 vs B's at 1,2,4) [0.44, 0.77]. On M7b PW-IVC's box dips BELOW CHANCE on the very allegiance pattern where it predicts 0.65-0.91 when positions are permuted, while IVC requires the same probability. Any reliable spread across the family (or a sub-0.5 proportion on M7b) kills IVC; equality kills PW-IVC.

AXIS 3 -- 2v2 TRANSFER (trials 21-24). A wins both 0.60s, B both 0.95s, positions permuted (A@{2,3} vs B@{4,6} vs A@{3,5} vs B@{1,4}). IVC: identical [0.68, 0.93] on all four cells. PW-IVC: T2a [0.75, 0.92] vs T2b [0.59, 0.89] with a pointwise strict ordering -- a required spread where IVC requires exactly 0.

AXIS 4 -- PRIMACY-RESCUE (trials 31-32). The product endorsed by the three 95% experts (positions 1, 4, 6) vs the product endorsed by a single 60% expert (position 5). IVC: the worst-expert veto, P(60%-product) in [0.57, 0.81]. PW-IVC: position-1 primacy (0.53-0.72 of the primacy kernel) fights the distrust kernel, d in [-0.03, +0.17], P in [0.42, 0.78] with sub-chance values at strong-primacy corners -- a confidence dilution (and possible sign reversal) IVC cannot produce anywhere in its box.

AXIS 5 -- SHARED-CORE ANCHOR AND NULL CONTROL (trials 29-30; trial 3-4). E2: the three 60% experts' product vs the three 95% experts' product at a 3-3 count tie: both theories confidently favor the 60% product (IVC [0.74, 0.95], PW [0.71, 0.97]) -- verifying that subjects read and distrust-invert the communicated validities (and screening out plain tallying, which predicts exactly 0.5). D3 (0.95@4 vs 0.95@6) is the zero-dose rung: both theories ~0.5, assaying generic side/late-position bias.

PARAMETRIC SIGNATURE. IVC predicts only FIVE distinct probability levels across all 32 pairs (exactly 0.500 on the 14 matched cells; one level on the ten T1/M7 cells; one on the four T2 cells; one on E2; one on V1; mirrors give complements), with EXACT equalities within families. PW-IVC predicts a position-graded profile that breaks every equality in a parameter-free direction. Regressing P(A) on primacy evidence (position-gradient-weighted cue signs, e.g., eta=5) controlling for inverted-validity evidence yields a primacy coefficient of exactly 0 under IVC and strictly positive under PW-IVC. 32 unique pairs, K = 96 // 32 = 3 repetitions in independently randomized order = exactly 96 trials; every cell appears in A/B-mirrored form (16 complement pairs) to absorb side bias.

[11] STRUCTURAL LEVER. PW-IVC and IVC share the identical distrust-inversion kernel ((1-v)-centered, gamma-steep); they differ ONLY in that PW-IVC adds a convex primacy gradient concentrated on positions 1-2 (mixture share tau*rho in [0.14, 0.29]; position 1 alone carries 53-72% of the primacy kernel). IVC's weights are a function of validity VALUES ONLY -- strictly position-invariant. The design therefore manipulates POSITION while holding validity allegiances exactly fixed, using the layout v = [0.90, 0.60, 0.60, 0.60, 0.60, 0.90]: the two 0.90 experts occupy the EXTREME positions 1 and 6 (maximal position contrast at TIED validity), and the four 0.60 experts form an interior equal-validity block (positions 2-5) enabling a pure within-block position-gradient assay. Because (0.40/0.10)^gamma is ~23-256 across both boxes, the 0.60 experts absorb ~97-99% of the distrust kernel, so every 0.90-vs-0.60 allegiance imbalance moves d by ~-0.245 (normalized) under BOTH theories -- the distrust signal is held constant while primacy varies. All boxes below were verified at the corners of both parameter spaces.

AXIS 1 -- MATCHED-VALIDITY PRIMACY DOSE LADDER (20 of 32 trials: D0, Z1, D1, W1, W2, D2, D2b, D3, D4, D5 + mirrors). Every cell has IDENTICAL validity multisets on the two sides (e.g., D5: A wins {0.90@1, 0.60@2, 0.60@3} vs B wins {0.60@4, 0.60@5, 0.90@6}), so the distrust kernel and the flat kernel cancel EXACTLY: IVC predicts P(A) = 0.500 PARAMETER-FREE on all twenty (d = 0 for every tau, gamma, c; the lapse preserves 0.5). PW-IVC predicts d = tau*rho*C, where C is the primacy contrast (normalized position-gradient evidence), spanning C ~ 0.04 (D0: 0.60@3 vs 0.60@4) to C ~ 0.98 (D5). Verified boxes: D0 [0.50, 0.53] (near-null control rung), Z1 [0.51, 0.59], D1 [0.51, 0.64], W1 [0.52, 0.68], W2 [0.52, 0.75], D2 (0.90@1 vs 0.90@6 duel) [0.53, 0.82], D2b [0.51, 0.73], D3 [0.54, 0.84], D4 [0.55, 0.88], D5 [0.55, 0.88]. Headline cells: on D5 and D4, observed P(A) >= 0.55 is IMPOSSIBLE for IVC at any admissible parameter combination and REQUIRED by PW-IVC; observed P(A) <= 0.52 falsifies PW-IVC's entire space. The regression slope of P(A) on C is EXACTLY 0 under IVC and strictly positive under PW-IVC -- a parametric signature, with the ladder's monotone ordering (D0 < Z1 < D1 < W1/W2 < D2/D2b < D3 < D4 < D5) providing 10 ordered rungs.

AXIS 2 -- POSITION-PERMUTATION FAMILY AT FIXED ALLEGIANCE (X2a, X2b, X2d + mirrors). All three hold the allegiance fixed at A wins {one 0.90, one 0.60}, B wins {two 0.60s}, permuting only positions. IVC: d = w(0.90) - w(0.60) IDENTICALLY on all three -> one probability, box [0.25, 0.43] (confident preference for the B product, endorsed by two 60% experts). PW-IVC: graded by primacy -- X2a (A's wins at positions 1,2 vs B's at 4,5) box [0.37, 0.69], STRADDLING/ABOVE chance because position-1 primacy (up to tau*rho*0.93) rivals the distrust deficit ((1-rho)*0.245); X2b (A at 1,5 vs B at 2,4) box [0.44, 0.50], near chance; X2d (A at 5,6 vs B at 2,3) box [0.12, 0.39], confidently B. Pointwise, PW-IVC REQUIRES a spread P(X2a) - P(X2d) >= ~0.07 (up to 0.55 at high-rho/high-beta corners) where IVC requires EXACTLY 0. Any reliable within-family spread falsifies IVC's position-invariance; exact equality falsifies PW-IVC. X2a is the design's two-sided adversarial cell: if subjects robustly distrust-follow here (P(A) <= 0.43), PW-IVC's primacy share is bounded small -- but then the Axis-1 ladder must be flat at 0.500, which is itself PW-falsifying; the joint pattern is winnowing in both directions.

AXIS 3 -- PRIMACY-DILUTED WORST-EXPERT VETO (V1 + mirror). A wins the single 0.60@2 expert vs B winning BOTH 0.90 experts (positions 1 and 6). Both theories' distrust kernels favor A (the less valid expert's endorsement counts more), but PW-IVC's primacy simultaneously favors B (positions 1 and 6 carry the maximal primacy mass): IVC box [0.55, 0.75], PW-IVC box [0.53, 0.66] -- a confidence dissociation (observed P(A) >= 0.70 is impossible for PW-IVC), and V1 also screens out plain tallying (which would pick B, 2 cues > 1).

AXIS 4 -- ANCHORS. E1 (all four 0.60 experts vs both 0.90 experts): both theories confidently favor the 0.60 product (IVC [0.75, 0.95], PW [0.76, 0.96]), verifying that subjects read and distrust-invert the communicated validities. TOT (total split): comprehension anchor, both ~1.

AGGREGATE SIGNATURE. IVC predicts exactly 0.500 on all 20 matched cells plus ONE identical probability on the six X2-family cells -- i.e., only ~4 distinct probability levels across all 32 pairs. PW-IVC predicts a position-graded profile breaking every equality in a parameter-free direction. Regressing P(A) on primacy contrast C (computed at a reference eta, e.g., 5) with inverted-validity evidence controlled (it is exactly zero on the matched cells by construction) yields a primacy coefficient of exactly 0 under IVC and strictly positive under PW-IVC. Every cell appears in A/B-mirrored form (16 complement pairs) to absorb side and position bias. 32 unique pairs, K = 96 // 32 = 3 repetitions in independently randomized order = exactly 96 trials per subject.

[12] STRUCTURAL LEVER. PGDT and PW-IVC share the distrust-inversion kernel; they differ ONLY in how position and distrust combine. PGDT MULTIPLIES: w_j proportional to g_j*[(1-tau)/n + tau*inv_j], so a cue's position effect on its weight is PROPORTIONAL to its distrust bracket. For a 95% expert the bracket collapses to the flat share (1-tau)/n = 0.013-0.030, so position contrasts among trusted experts move d by at most ~0.08; for a 55% expert the bracket is ~0.30, amplifying the SAME gate contrast by a factor bl/bh >= 9.9 (verified at every corner of PGDT's box: tau, eta, delta, gamma, c). PW-IVC ADDS: w_j = (1-tau)/n + tau*[(1-rho)*inv_j + rho*prim_j], so for any EQUAL-VALIDITY duel the distrust terms cancel EXACTLY and d = tau*rho*(prim_i - prim_j) -- INDEPENDENT of the validity value. Position effects are validity-INVARIANT under PW-IVC and validity-AMPLIFIED under PGDT. LAYOUT: v = [0.95, 0.55, 0.95, 0.55, 0.95, 0.55] (n=6; high-validity experts at odd positions 1/3/5, low at even 2/4/6). This places PW's weapon (a 95% expert at position 1, carrying 0.60-0.79 of the primacy kernel) head-to-head with PGDT's weapon (a 55% expert at position 2, where the floored gate retains 41-68% of its mass), and provides equal-validity duels at three position separations within each validity level. All boxes below were verified at the corners of both parameter spaces.

AXIS 1 -- VALIDITY-SCALED POSITION CROSSOVER (H1: A wins expert 1 vs B wins expert 5; H2: 1 vs 3; both 95% duels; L1: A wins expert 2 vs B wins expert 6; L2: 2 vs 4; both 55% duels; H3/L3 near-null rungs). PGDT: d(low duel)/d(high duel) = [gate-contrast ratio]*[bracket ratio] >= 0.262*9.9 = 2.6, so the LOW-validity position ladder is >= 2.6x steeper: P(early product | low duel) > P(early product | high duel) at EVERY corner. PW-IVC: d(high duel)/d(low duel) = primacy-contrast ratio >= 2.18, so the HIGH-validity ladder is >= 2.2x steeper: the OPPOSITE ordering at every corner. Parameter-free double crossover. Falsification caps: PGDT confines P(early|H1) <= 0.60 and P(early|H2) <= 0.59 (observed >= 0.65 on the high-duel family kills PGDT's entire space); PW-IVC confines P(early|L1) <= 0.65, P(early|L2) <= 0.63, P(early|L3) <= 0.51 (observed >= 0.70 on L1/L2 or >= 0.55 on L3 kills PW's entire space). Boxes: H1 PGDT [0.50,0.60] vs PW [0.53,0.86]; H2 PGDT [0.50,0.59] vs PW [0.53,0.84]; L1 PGDT [0.55,0.85] vs PW [0.51,0.65]; L2 PGDT [0.55,0.83] vs PW [0.51,0.63]; L3 PGDT [0.50,0.53] vs PW [0.50,0.51].

AXIS 2 -- SIGN-FLIP FAMILY (M1: A wins the 55% expert at position 2 vs B winning the 95% expert at position 1 PLUS the 55% expert at position 6; M1c: same with B's late 55% expert at position 4; M5: A wins {55%@2, 95%@3} vs B winning {95%@1, 55%@6}). These trials are distrust-NEUTRALIZED: the two 55% endorsements cancel in the distrust kernel of BOTH theories, so choice is decided by position architecture alone. PGDT: the gate difference between positions 2 and 6 (or 2 and 4), multiplied by the large low-validity bracket (raw product >= 0.043), beats the position-1 expert's flat-share weight (<= 0.031) -- d > 0 at every corner (margin >= 41%) -> P(A) in [0.51, 0.87] across M1/M1c/M5. PW-IVC: distrust cancels exactly, and the additive primacy kernel hands the win to position 1 (0.60-0.79 of primacy mass): d = -(1-tau)/6 - tau*rho*(prim_1 - prim_2) < 0 at every corner -> P(A) in [0.25, 0.47]. DISJOINT BOXES, OPPOSITE SIGNS on all three cells. Observed P(A) >= 0.50 on any M-cell falsifies PW-IVC everywhere in its box; observed P(A) <= 0.50 falsifies PGDT. (These cells also exclude naive tallying, which picks B: 2 cues beat 1.)

AXIS 3 -- MIGRATION CROSSINGS (M1 -> M2; M1 -> M8). Moving A's low expert from position 2 to position 4 (M2: A wins 55%@4 vs B winning {95%@1, 55%@6}): PGDT CROSSES 0.5 (P in [0.51,0.86] -> [0.44,0.49]; sign change guaranteed since (g_4-g_6)*bl < bh at every corner) while PW stays below 0.5 on both. Replacing B's late low expert with the 95%@3 (M8: A wins 55%@2 vs B winning {95%@1, 95%@3}): PW FLIPS from B to A (distrust no longer cancels: d in [0.02,0.20] > 0) while PGDT stays A (d in [0.13,0.73]). The joint pattern {M1: A, M2: B, M8: A} is PGDT-only; PW requires {M1: B, M2: B, M8: A}.

AXIS 4 -- COUNT- AND VALIDITY-BALANCED COALITION POSITION CELLS (L5a: A wins {55%@2, 95%@3} vs B winning {55%@4, 95%@5}; L5b: vs B winning {95%@5, 55%@6}). Both sides carry one low and one high expert, so distrust and counting cancel; only position varies. PGDT d in [0.08, 0.39] (P(A) up to 0.85-0.89); PW d in [0.03, 0.11] (P(A) capped at 0.67-0.69). Observed P >= 0.72 on L5b is impossible for PW at any admissible parameter combination.

AXIS 5 -- ANCHORS (D1, D3, TOT). D1: all three 55% experts vs all three 95% experts at a 3-3 count tie -- both theories confidently prefer the LOW-validity product (PGDT [0.90,0.96], PW [0.68,0.97]), verifying that subjects read and distrust-invert the stated validities and excluding plain tallying (exactly 0.5). D3: two 55% experts vs three 95% experts -- both prefer the 2-cue low-validity product (anti-count and anti-validity). TOT: total-split comprehension anchor (both >= 0.85).

PARAMETRIC SIGNATURE. PGDT predicts choice tracks the PRODUCT evidence sum_j g_j*bracket_j*s_j; PW-IVC predicts choice tracks the MIXTURE evidence sum_j [(1-rho)*inv_j + rho*prim_j]*s_j. The two axes are ANTI-CORRELATED on the M-family (PGDT evidence > 0 exactly where PW evidence < 0) and differently scaled on the duel ladders (ratio >= 2.6 in opposite directions). The single-number crossover contrast C = mean[P(L1),P(L2)] - mean[P(H1),P(H2)] is strictly positive under PGDT (underlying d-ratio >= 2.6) and strictly negative under PW-IVC (ratio >= 2.2). Every cell appears in A/B-mirrored form to absorb side and position bias. 32 unique pairs, K = 96//32 = 3 repetitions in independently randomized order = exactly 96 trials per subject.

[13] STRUCTURAL LEVER. PW-IVC and PGDT share the distrust-inversion kernel; they differ in how primacy and distrust COMBINE. PW-IVC ADDS: w_j = (1-tau)/n + tau*[(1-rho)*inv_j + rho*prim_j], so the primacy bonus for an early cue is validity-INDEPENDENT -- an early TRUSTED (95%) expert receives the full rho*prim_1 boost on top of its (tiny) distrust weight. PGDT MULTIPLIES: w_j proportional to g_j*[(1-tau)/n + tau*inv_j], so the gate boost scales with the bracket: for a 95% expert the bracket collapses to the flat share (1-tau)/n = 0.013-0.030, and the maximal gate (g_1 = 1) multiplies almost nothing; for a 55% expert the bracket is ~0.22-0.26 and the gate amplifies it. Consequence, verified at all 8 corners of each parameter box on the layout below: the position-1 95% expert carries PW weight 0.10-0.24 but PGDT weight only 0.03-0.11 (a 2-4x relative rescue), while the position-2 55% expert carries PGDT weight 0.29-0.41 vs PW 0.22-0.25. Conversely, the position gradient among 55% experts is 3-4x steeper under PGDT (gate x big bracket) than under PW, and the position gradient among 95% experts exists ONLY under PW (PGDT's is pinned at the flat share). Prior design [12] attacked this with alternating validities and position-duel ladders plus M-migrations; this design uses a different layout (2 trusted experts at positions 1 and 4, FOUR distrusted experts at 2, 3, 5, 6) and centers on a NOVEL axis: a COUNT-CROSSOVER / VOTE-EQUIVALENCE LADDER measuring how many 55% endorsements one early 95% expert is worth, plus a trusted-block assay and gradient-ownership duels.

LAYOUT: v = [0.95, 0.55, 0.55, 0.95, 0.55, 0.55] (n=6). The kernel ratio (0.44/0.05)^gamma >= 600 concentrates ~99.9% of BOTH distrust kernels on the four 55% experts (each ~0.25 of inv mass), so the distrust signal is essentially 'count of 55% endorsements per side' and is held nearly constant across cells; position architecture and the trusted-expert weight are what vary. All boxes below are P(A) for the listed A-set, verified at the corners of both parameter spaces (PW tau/rho/eta corners; PGDT tau/delta/eta corners, with beta/epsilon at their worst-case ends).

AXIS 1 -- VOTE-EQUIVALENCE LADDER (L0, L1, L2, L3; the headline). A wins the 95% expert at position 1 plus j late 55% experts (position 6, then 5, then 3); B wins the single 55% expert at position 2. PW: the early trusted expert is worth 0.6-1.1 late 55% experts, so the ladder crosses chance at rung j*=1: L1 (A={1,6} vs B={2}) has d in [+0.06, +0.19] at EVERY PW corner -> P(A) in [0.52, 0.79]. PGDT: the early trusted expert is worth only 0.1-0.3 of one 55% expert, so the ladder does not cross until rung 2-3: L1 has d in [-0.21, -0.01] at EVERY PGDT corner -> P(A) in [0.24, 0.50]. DISJOINT BOXES on L1: observed P(A) >= 0.55 is impossible for PGDT at any admissible parameter combination; observed <= 0.48 is impossible for PW. The measured crossing rung j* separates the theories by one to two rungs (PW j*=1; PGDT j* in {2,3}). L2: PW [0.71, 0.94] vs PGDT [0.47, 0.75]; L3: both A (agreement high end). L1e (helper = the position-3 55% expert) replicates the rung with an early helper: PW d in [+0.07, +0.19] (all positive) vs PGDT d in [-0.13, +0.01] (nearly all negative).

AXIS 2 -- TRUSTED-BLOCK ASSAY (TB, TB+, F3). TB: A wins BOTH 95% experts (positions 1 and 4) vs B wins the single position-2 55% expert. PGDT: d in [-0.36, -0.19] -> P(55%-product) in [0.60, 0.87]; the two BEST experts lose to one distrusted expert. PW: d in [-0.15, +0.03] -> P(A) in [0.28, 0.53], i.e., chance. Observed P(95%-product) >= 0.50 falsifies PGDT's entire space. TB+ (both 95% + the position-3 55% vs the position-2 55%): PW [0.54, 0.83] vs PGDT [0.36, 0.54]. F3 (both 95% + the late 55% at position 6 vs the two early 55% experts at 2 and 3): PGDT P(A) in [0.10, 0.28] -- the two early lows dominate a 3-cue coalition containing both best experts; PW in [0.27, 0.53]. TB also excludes naive tallying (2 cues beat 1, yet both theories pick the 1-cue 55% product).

AXIS 3 -- GRADIENT OWNERSHIP (HD vs LDf/LDa/LDm/LB). HD: duel between the two 95% experts (position 1 vs 4). PW: d in [+0.08, +0.23] -> P(early) in [0.53, 0.85]; PGDT: d in [+0.01, +0.08] -> P(early) in [0.51, 0.62]. LDf (55%@2 vs 55%@6): PW d in [+0.01, +0.08] -> P(early) in [0.50, 0.58]; PGDT d in [+0.07, +0.26] -> [0.54, 0.81]. LDa (55%@2 vs 55%@3): PW [0.49, 0.57] vs PGDT [0.52, 0.72]. LB (early-low pair {2,3} vs late-low pair {5,6}): PW [0.52, 0.68] vs PGDT [0.52, 0.86]. The aggregate contrast C = P(early | HD) - P(early | LDf) is POSITIVE at every PW corner (HD effect 1.5-4.5x the low-duel effect) and NEGATIVE at every PGDT corner (low-duel effect 1.9-7.7x the HD effect) -- a parameter-free sign flip. Observed P(early|HD) >= 0.68 kills PGDT; observed P(early|LDf) >= 0.65 or P(early|LB) >= 0.75 kills PW -- the design is genuinely two-sided, with the low-validity gradient cells as PGDT's home turf.

AXIS 4 -- DISTRUST-NEUTRAL SIGN-FLIP CELLS (L1, FL). On L1 and FL the 55% endorsements cancel exactly in both theories' distrust kernels, so choice is decided by position architecture alone. PW's additive primacy hands the win to the position-1 expert's product; PGDT's multiplicative gate x bracket hands it to the position-2 expert's product. FL (A = {95%@1, 55%@5} vs B = {55%@2, 95%@4}): PW d in [+0.05, +0.18] (all corners positive) -> P(A) in [0.52, 0.78]; PGDT d in [-0.22, -0.04] (all corners negative) -> P(A) in [0.22, 0.48]. DISJOINT, opposite signs.

AXIS 5 -- ANCHORS. DINV (four 55% experts vs the two 95% experts): both theories confidently prefer the 55% product (PW [0.72, 0.97], PGDT [0.85, 0.99]) -- verifies that subjects read and distrust-invert the communicated validities. TOT (total split): comprehension anchor, both ~1. LDm (55%@3 vs 55%@5) is a near-null rung assaying generic side bias.

PARAMETRIC SIGNATURE. Regressing P(A) on primacy-only evidence sum(prim_j * s_j) versus product evidence sum(g_j * bracket_j * s_j) yields opposite-loading profiles: the two axes are anti-correlated on the L1/FL/F3 family by construction. The single-number readouts: (i) the ladder crossing rung j* (PW: 1; PGDT: 2-3); (ii) the gradient-ownership contrast C (PW: positive everywhere; PGDT: negative everywhere); (iii) the trusted-block veto (PGDT uniquely predicts P(55%@2-product) >= 0.60 against both 95% experts). Every cell appears in A/B-mirrored form to absorb side and position bias. 32 unique pairs, K = 96 // 32 = 3 repetitions in independently randomized order = exactly 96 trials per subject.

[14] STRUCTURAL LEVER. PGDT and PRDT share the distrust-inversion kernel and a floored convex primacy gate; they differ in exactly one structural respect: how the TRUSTED (flat-share) channel treats POSITION. PGDT's bracket is (1-tau)/n + tau*inv_j -- position-INDEPENDENT -- so a 99%-validity expert's weight is g_j*(1-tau)/n at every position, and among trusted experts the GATE alone orders weights (early > late). PRDT multiplies the flat share by a trust ramp t_j with t0 pinned at EXACTLY 0 and t1 in [0.85, 1.0]: an EARLY trusted expert's bracket collapses to tau*inv_trusted ~ 0 (with v=0.99, inv_trusted <= (0.01(1-c))^gamma <= 6.3e-4 pre-normalization), while a LATE trusted expert recovers nearly the full flat share. Among trusted experts PRDT's weights are therefore ordered LATE > EARLY -- the opposite of PGDT. LAYOUT: four 99% experts at positions 1, 2, 6, 7 (maximal position contrast at tied validity), one 55% expert at position 4, two 78% experts at positions 3 and 5 (n=7). All trusted-axis cells hold the three distrusted experts TIED, so the distrust kernel cancels exactly and choice is decided by the trusted channel x gate alone.

AXIS 1 -- TRUSTED-POSITION SIGN FLIP (parameter-free, the headline). Cells D17 (A wins expert 1 vs B wins expert 7), D16, D27, D26, D12, BLK (A wins {1,2} vs B wins {6,7}), BLKa (A wins {1} vs B wins {6,7}), T21 (A wins {1,2} vs B wins {7}). PGDT: d = flat*Delta(gate) + nonnegative leakage, with Delta(gate) > 0 STRUCTURALLY (gate strictly decreasing, flat share position-independent, inv leakage from the 99% experts <= ~2e-4 raw at gamma>=2.8) -> P(early-trusted product) > 0.5 at EVERY parameter corner. PRDT: d = flat*Delta(t*gate) + leakage, with Delta(t*gate) < 0 STRUCTURALLY (t0 = 0 kills positions 1-2; t1 >= 0.85 restores positions 6-7); the leakage term tau*w_inv_tr*Delta(gate) was bounded corner-by-corner and never exceeds ~50% of the flat term (e.g., BLK worst corner: leakage 1.7e-3 vs flat term -0.0022; D17: 9.6e-4 vs -0.0021; BLKa: 6.4e-4 vs -0.0034) -> P(early-trusted product) < 0.5 at EVERY corner. Verified boxes: BLK PGDT [0.51, 0.62] vs PRDT [0.476, 0.492]; D17 PGDT [0.507, 0.572] vs PRDT [0.489, 0.494]; BLKa PGDT [0.502, 0.56] vs PRDT [0.481, 0.493]; D16 PGDT [0.503, 0.565] vs PRDT [0.492, 0.498]; D27/D26 weaker but sign-identical. D12 and T21 are near-chance under PRDT (|P-0.5| < 0.01) but strictly > 0.5 under PGDT. Headline readout: C = mirror-corrected mean P(early-trusted product) over the family is POSITIVE at every PGDT corner and NEGATIVE at every PRDT corner. Falsification: pooled P(early) <= 0.50 on D17/BLK/BLKa/D16 kills PGDT's entire space; >= 0.50 kills PRDT's. These cells are also anti-tallying (BLKa: 1 cue beats 2) and anti-TTB (the top expert does not always win).

AXIS 2 -- KERNEL STEEPNESS (disjoint ratio boxes / estimated gamma). PGDT gamma in [2.8, 3.5]; PRDT gamma in [1.6, 2.2] -- DISJOINT. With one 55% and two 78% experts, the effective low:mid bracket ratio (including the flat share, verified at corner pairs) is >= 5.9 under PGDT and <= 4.9 under PRDT -- disjoint. Consequence: on ML1 (two 78% experts vs one 55%), ML2, and TL (99% expert 1 vs 55% expert 4), both theories prefer the LOW-validity-endorsed product, but PGDT does so more confidently at matched corners (e.g., ML1: PGDT P(low product) up to ~0.90 vs PRDT up to ~0.85 at high beta; TL: 0.89 vs 0.955 crosses the other way only via beta, so these are fitting cells); a kernel-exponent fit over this family must land in [2.8, 3.5] (PGDT) or [1.6, 2.2] (PRDT) -- disjoint parameter boxes, a quantitative signature no single-cell analysis could deliver given the wide beta ranges.

AXIS 3 -- SCREENING AND ANCHORS. ANCH: the product endorsed by the single 55% expert against the product endorsed by ALL SIX other experts (including all four 99% experts): both theories pick the 55% product (P >= 0.62 at every corner) -- a dramatic anti-validity anchor that screens out normative followers, TTB (which follows expert 1), and tallying (6 cues beat 1). ML1 also screens tallying (2 cues lose to 1). MM (78% expert 3 vs 78% expert 5) is an agreement cell assaying the position gradient among equal-validity distrusted experts (both predict the earlier expert's product, P ~ 0.54-0.56). MX1/MX2 are mixed fitting cells constraining the gate and ramp jointly (both theories agree on sign there, with PGDT more extreme on MX1). TOT is the comprehension anchor.

AGGREGATE SIGNATURE. Regress mirror-corrected P(A) on trusted-GATE evidence (sum of g_j*s_j over trusted positions, reference eta=5, delta=0.3) versus trusted-RAMP evidence (sum of t_j*s_j, reference kappa=1.5, t1=0.925): PGDT loads on the gate contrast only (ramp coefficient exactly 0 -- it has no t-ramp); PRDT loads on the ramp contrast, with early-trusted cells carrying ~zero weight (t0 = 0). Opposite-loading profiles on anti-correlated regressors. Every cell type appears in A/B-mirrored form to absorb side and position bias. 32 unique pairs, K = 96 // 32 = 3 repetitions in independently randomized order = exactly 96 trials per subject. Effect sizes on the trusted axis are modest (0.48-0.62) but sign-robust at every corner; the family-level contrast C and the within-subject repetition of all 8 trusted cell types provide the statistical power for the parameter-free sign test.

[15] STRUCTURAL LEVER. PRDT and PGDT share the distrust-inversion kernel and a floored primacy gate; they differ in two quantitatively disjoint respects. (1) KERNEL EXPONENT: PRDT gamma in [1.6, 2.2] (flattened inversion -- mid-validity experts retain large distrust weight), PGDT gamma in [2.8, 3.5] (steep inversion -- the lowest-validity expert monopolizes the kernel). For the 0.55-vs-0.75 pair the full-weight ratio w55/w75 is ~[2.8, 4.0] under PRDT and ~[4.6, 8.8] under PGDT -- DISJOINT. (2) TRUSTED CHANNEL: PRDT multiplies the flat share by a trust ramp with t0 pinned at EXACTLY 0, so an early trusted expert's weight collapses to the (tiny) kernel leakage, and PRDT's tau is pinned high ([0.90, 0.93], flat share <= 0.0125/cue); PGDT's flat share (1-tau)/n in [0.010, 0.0225] is position-INDEPENDENT and its tau can drop to 0.82. Unlike prior design [14] (which isolated the trusted-position contrast with tied distrusted cues at 99% validity), this design embeds the trusted channel in a kernel-ACTIVE architecture and centers on a novel axis: the VOTE-EQUIVALENCE LADDER measuring how many 0.75-experts one 0.55-expert outweighs, plus a kernel-neutral trusted-rescue cell. LAYOUT (n=8): v = [0.99, 0.99, 0.99, 0.75, 0.55, 0.75, 0.75, 0.75]. The three 0.99 experts soak up the super-primacy slots 1-3 (gates 1.57-1.98, 1.0, 0.55-0.67): under PRDT their combined normalized weight is <= 0.027 (t0=0 kills the early flat share; inv(0.99) ~ 0), under PGDT it is 0.068-0.218 (flat share x gate). The 0.55 expert sits at position 5 and the four 0.75 experts at positions 4, 6, 7, 8 -- the gate-FLOOR region where gates ~ delta (position-independent to within ~15%), so cue weights are proportional to brackets and the kernel ratio expresses itself directly, unconfounded by the gate. All boxes below were verified at 8 corners of each parameter box (PRDT: gamma x c x tau x eta/delta x t1/kappa extremes; PGDT: gamma x c x tau x eta/delta extremes, including mixed corners), with beta/epsilon at their worst-case ends.

AXIS 1 -- VOTE-EQUIVALENCE LADDER (the headline, gamma assay). B wins the 0.55 expert (position 5); A wins k of the 0.75 experts. Crossing rung k*: PRDT 3-4, PGDT >= 5. The decisive rung is k=4 (L4: A wins 0.75@{4,6,7,8}): PRDT d in [+0.040, +0.214] at EVERY corner (min at gamma2.2/c0.3/tau0.93/t1 0.85/kappa3) -> P(A) in [0.535, 0.742]; PGDT d in [-0.278, -0.047] at EVERY corner -> P(A) in [0.346, 0.475]. DISJOINT BOXES (gap 0.06). L4b (A wins 0.75@{3,6,7,8}) replicates: PRDT d [+0.069, +0.239] -> P(A) [0.560, 0.766]; PGDT d [-0.238, -0.013] -> P(A) [0.345, 0.493]. Parameter-free sign flip: observed P(A) >= 0.52 pooled on L4/L4b is impossible for PGDT anywhere in its box; observed P(A) <= 0.49 is impossible for PRDT. The measured crossover rung separates the theories by one to two cues, and the implied weight ratio w55/w75 (fit from the ladder plus the D-duels) must land in [2.8, 4.0] (PRDT) or [4.6, 8.8] (PGDT) -- disjoint, i.e., a direct behavioral estimate of the kernel exponent gamma lands in disjoint boxes [1.6, 2.2] vs [2.8, 3.5].

AXIS 2 -- DUEL CONFIDENCE PROFILE (gamma assay, agreement cells). On D1 (0.55 vs 0.75@4), D2 (0.55 vs 0.75@6), D3 (0.55 vs 0.75@3), D4 (0.55 vs two floor 0.75s) both theories prefer the 0.55-endorsed product, but PGDT is uniformly MORE confident: d-ranges are disjoint on D1 (PRDT [0.200, 0.316] vs PGDT [0.342, 0.462]), D3 ([0.175, 0.289] vs [0.326, 0.422]), D4 ([0.113, 0.241] vs [0.259, 0.454]); the family-mean d is [0.184, 0.299] under PRDT vs [0.321, 0.488] under PGDT -- disjoint. D4 and D6 (0.55 vs 0.75@{4,6,7}: PGDT says the 0.55 wins, d in [+0.145, +0.336]; PRDT is split, d in [-0.075, +0.078]) also exclude plain tallying (1 cue beating 2-3 cues). L3 (three floor mids vs the 0.55: PGDT confidently B, d in [-0.396, -0.161]; PRDT split, d in [-0.123, +0.024]) and D6 jointly pin each theory's location inside its own box.

AXIS 3 -- TRUSTED-RESCUE SIGN FLIP (tau/t0 assay, kernel-neutral). T1c: A wins {0.99@1, 0.99@2, 0.99@3, 0.75@6} vs B wins {0.75@4}. Each side carries exactly ONE 0.75 expert, so the distrust kernels cancel exactly and choice is decided by the trusted channel x gate: PGDT's position-independent flat share (0.010-0.0225) times the huge position-1 gate (1.57-1.98) gives the 0.99 block weight 0.068-0.218 -> d in [+0.040, +0.167], P(A) in [0.521, 0.724] at every corner; PRDT's t0=0 ramp gives the same block <= 0.027 -> d in [-0.046, -0.021], P(A) in [0.443, 0.482] at every corner. DISJOINT, OPPOSITE SIGNS. T3 cap cell (0.99@1 vs 0.75@8 duel): PRDT confines P(0.99-product) <= 0.40 (d in [-0.137, -0.120]); PGDT allows up to 0.57. Observed P >= 0.45 on T3 falsifies PRDT's entire space.

AXIS 4 -- REVERSE CAPS (two-sided, PRDT-risky cells). VETO: A wins ALL SEVEN other experts (including all three 0.99s), B wins only the 0.55. PRDT: d = 1-2*w55 in [+0.042, +0.234] -> P(A) >= 0.535 at every corner; PGDT's steep-kernel corners REQUIRE near-chance or below (d down to -0.160, P(A) down to 0.32). Observed P(A) <= 0.50 on VETO kills PRDT. M2 (all four 0.75s vs all three 0.99s): PRDT requires P(A) >= 0.81 (d in [+0.519, +0.600]); PGDT allows 0.53-0.86. Observed P(A) <= 0.75 on M2 kills PRDT. These cells make the design genuinely adversarial for the advocated theory: a steep distrust kernel (PGDT) predicts the single 55% expert's product can beat the product endorsed by everyone else.

ANCHORS AND CONTROLS. AN1 (three 0.99s vs the single 0.55): BOTH theories confidently prefer the 0.55-endorsed product (P >= 0.64 at every corner) -- verifies subjects read and distrust-invert the stated validities; excludes normative weighting, TTB (follows expert 1), and tallying (3 cues beat 1). D4 excludes tallying in the other direction. M1 (0.75@6 vs 0.75@4) is a gate-agreement cell; C0 (0.75@7 vs 0.75@8) is a near-null side-bias control; TOT is the comprehension anchor.

AGGREGATE SIGNATURE. Mirror-corrected P(A) regressed on PRDT reference evidence (w-PRDT profile at gamma1.9/c0.15) versus PGDT reference evidence (gamma3.1/c0.17) loads with opposite signs: the two evidence axes are anti-correlated by construction on L4/L4b/VETO/T1c. Single-number readouts: (i) ladder crossing rung k* (PRDT 3-4, PGDT >= 5); (ii) implied w55/w75 ratio (disjoint [2.8, 4.0] vs [4.6, 8.8]); (iii) trusted-block rescue sign on T1c (PRDT B, PGDT A at every corner); (iv) VETO and M2 caps (PRDT requires P >= 0.51 / 0.81; PGDT allows 0.32 / 0.53). Every cell type appears in A/B-mirrored form to absorb side and position bias. 32 unique pairs, K = 96 // 32 = 3 repetitions in independently randomized order = exactly 96 trials per subject.

[16] STRUCTURAL LEVER. PGDT and PBDT share the distrust-inversion kernel and a floored convex primacy gate; they differ in exactly two quantitatively dissociable respects. (1) THE FLAT (TRUSTED) SHARE. PGDT's bracket is (1-tau)/n + tau*inv_j with tau in [0.82, 0.92]: a POSITION-INDEPENDENT flat share of 1.3-3.0% per cue (n=6). PBDT's bracket is t_j*(1-tau)/n + tau*inv_j with tau in [0.91, 0.93] and a trust ramp t_j rising from t0 in [0.25, 0.35] to t1 in [0.95, 1.0]. Consequence, verified at the corners of both boxes: the raw weight of an EARLY high-validity (95%) expert is DISJOINT across the theories -- PGDT [0.013, 0.031] vs PBDT [0.004, 0.007], a 2-7x gap -- because PBDT's t0 crushes the flat share exactly where the gate is maximal. For any cue whose bracket is flat-share-relevant (validity 0.85-0.95), PBDT's ramp CANCELS its gate, so PBDT's position effects on the trusted/mid channel collapse toward zero, while PGDT's remain gate-scaled. (2) GATE SHAPE. PGDT's floor is high (delta 0.20-0.45, eta 4-6); PBDT's gate is steeper with a lower floor (delta 0.15-0.25, eta 6.5-8). On tail-position contrasts among kernel cues (positions 3 vs 4), PGDT's gate contrast (1-delta)(0.6^eta-0.4^eta) reaches 2.5x PBDT's maximum. LAYOUT: palindromic v = [0.95, 0.85, 0.60, 0.60, 0.85, 0.95]. The palindrome creates VALIDITY-MATCHED position duels at three validity levels -- 95s at positions 1 vs 6, 85s at 2 vs 5, 60s at 3 vs 4 -- so the distrust kernel cancels exactly within each duel and choice is decided by gate x flat-share architecture alone. All boxes below are mirror-corrected P(A), verified at the corners of both parameter spaces (tau x eta x delta x gamma x c, with beta/epsilon at worst-case ends).

AXIS 1 -- VALIDITY-MATCHED POSITION-TRANSFER BATTERY (T16, M25, L34 + mirrors). T16 (95@1 vs 95@6): PGDT d = b_tr*(1-delta) in [0.0075, 0.025] raw -> P(A) in [0.51, 0.66]; PBDT d = flat'*(t0 - delta*t1) + leakage in [0.0004, 0.0033] -> P(A) in [0.50, 0.52]. PBDT is HARD-CAPPED at ~0.52: observed P(early-95 product) >= 0.55 is impossible for PBDT at any admissible parameter combination. M25 (85@2 vs 85@5): PGDT [0.51, 0.60]; PBDT [0.50, 0.55] (its ramp cancels its gate for the mid channel; the flat-share term straddles zero). L34 (60@3 vs 60@4): PGDT [0.52, 0.70]; PBDT [0.51, 0.62] -- observed >= 0.63 kills PBDT. PGDT requires a robustly nonzero transfer effect at ALL THREE validity levels; PBDT requires near-chance on the 95- and 85-level duels. Flatness at T16/M25 falsifies PGDT; any effect >= 0.55 at T16 or >= 0.57 at M25 falsifies PBDT's entire space.

AXIS 2 -- BLOCK-FLIP STAIRCASE (T16 -> BLK2 -> BLK3 + mirrors). BLK2 (A wins {95@1, 85@2} vs B wins {85@5, 95@6}, 60s tied): PGDT P in [0.51, 0.70]; PBDT [0.50, 0.55] -- raw-d boxes DISJOINT ([0.011, 0.044] vs [0.000, 0.011]). BLK3 (full 3v3 block flip): PGDT [0.53, 0.82]; PBDT [0.52, 0.66]. The within-subject staircase increments are the signature: PGDT's T16->BLK2 increment (adding the 85@2/85@5 pair) is robustly POSITIVE at every corner (raw [0.004, 0.019]) because the flat share x gate gives the early mid expert real weight; PBDT's increment STRADDLES ZERO ([-0.000, 0.008]) because its ramped flat share cancels the gate. Observed mirror-corrected P(BLK2) - P(T16) >= 0.04 is impossible for PBDT and required by PGDT's upper box; a flat or negative increment falsifies PGDT.

AXIS 3 -- EARLY-TRUSTED RESCUE AND POSITION TRANSFER (X1, X3 + mirrors). X1 (A wins {95@1, 85@5} vs B wins {85@2}): the two 85 endorsements cancel in the distrust kernel, so the 95@1 expert's flat-share weight decides. PGDT: d in [+0.002, +0.015] at every corner -> P(A) in [0.51, 0.60] (the early 95 RESCUES product A). PBDT: d in [-0.004, +0.007] -> straddles chance. X3 (A wins {85@5, 95@6} vs B wins {85@2}) is the position-transfer twin: the same 95 endorsement moved from position 1 to position 6. The transfer difference X1 - X3 equals b_tr*(1-delta) under PGDT -- robustly in [0.0075, 0.025] raw, P-difference >= 0.02 and up to 0.20 -- while under PBDT it equals flat'*(t0 - delta*t1) + leakage, confined to [0.0004, 0.0033] raw (P-difference <= 0.01). A reliable X1-vs-X3 difference >= 0.04 falsifies PBDT's entire space; a zero or negative difference falsifies PGDT.

AXIS 4 -- ANCHORS AND SCREENS. DINV (the product endorsed by all four 95/85 experts vs the product endorsed by the two 60 experts): BOTH theories confidently prefer the 60-product (PGDT d in [-0.39, -0.09]; PBDT [-0.24, -0.09]) -- verifies that subjects read and distrust-invert the stated validities, and excludes normative weighting, TTB (follows expert 1), and tallying (4 cues beat 2). VETO (95@1 vs 60@3): both prefer the 60-product ([-0.23, -0.07] / [-0.13, -0.06]) -- the single BEST expert's product loses to a mid-list distrusted expert's product. TOT: comprehension anchor (both ~0.95). R1 (95@6 vs 85@2) and X2 are agreement/fitting cells where both theories land on the same side (B, and straddle, respectively), constraining gate/ramp/beta jointly.

AGGREGATE SIGNATURE. Mirror-corrected P(A) regressed on PGDT reference evidence (g_j*[(1-tau)/n + tau*inv_j] at tau=0.87, eta=5, delta=0.3) versus PBDT reference evidence (g_j*[t_j*(1-tau)/n + tau*inv_j] at t0=0.30, kappa=1.0) loads with opposite profiles: the two evidence axes are anti-correlated on the X1/X3/BLK family by construction (position-1 trusted weight is maximal under the first, ~zero under the second). Single-number readouts: (i) family mean over {T16, M25, BLK2, X1} is hard-capped at 0.55 by PBDT (its maximum over all corners) while PGDT's central mass is 0.55-0.70; (ii) the transfer battery slope (effect size vs validity level: 95-duel, 85-duel, 60-duel) is strictly positive at every PGDT corner and confined to [~0, 0.05] under PBDT for the top two rungs; (iii) the X1-X3 transfer difference. Every cell type appears in A/B-mirrored form to absorb side and position bias. 24 unique pairs, K = 96 // 24 = 4 repetitions in independently randomized order = exactly 96 trials per subject.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "STRUCTURAL LEVER. PBDT and PGDT share the distrust-inversion kernel and a floored convex primacy gate; they differ in exactly one quantitatively decisive respect: the TRUSTED (flat-share) channel. PGDT's bracket is (1-tau)/n + tau*inv_j -- POSITION-INDEPENDENT -- so an early high-validity expert's raw weight is g_1*flat: the full amplified position-1 gate (n/(n-1))^eta times the FULL flat share. PBDT multiplies the flat share by a trust ramp t_j rising from t0 in [0.25, 0.35] at position 1 to t1 in [0.95, 1.0] at position n, so its position-1 trusted weight is g_1*t0*flat -- crushed by exactly the factor t0, a PARAMETER-FREE 2.9-4.0x reduction relative to PGDT (1/t0 at every corner), while late trusted experts (t1~1) are left nearly intact. PGDT's trusted profile is g_j (monotone, position-1 dominant); PBDT's is g_j*t_j. Prior design [16] probed this with a palindromic layout and pure trusted-expert duels. This design embeds the lever in a KERNEL-ACTIVE architecture and measures the early-trusted rescue behaviorally against a finely balanced mid-validity pair whose gate gap is calibrated to sit between the two theories' early-trusted weights.\n\nLAYOUT (n=6): v = [0.95, 0.95, 0.75, 0.75, 0.60, 0.95]. Trusted 95% experts sit at the three gate anchors: position 1 (amplified gate (6/5)^eta in [2.1, 4.3] over both boxes), position 2 (gate EXACTLY 1.0 in both theories), position 6 (gate floor delta). The two 75% experts at positions 3-4 form a finely balanced kernel pair whose weight gap (g_3-g_4)*bracket_75 is, at every corner of both boxes, SMALLER than PGDT's minimal position-1 trusted weight but comparable to PBDT's maximal one -- this calibration is what makes the rescue cells sign-diagnostic. The single 60% expert at position 5 is the dominant kernel cue anchoring the anti-validity screens. All boxes below are mirror-corrected P(A), verified at the corners and adversarial interior points of both parameter boxes (gate/ramp/kernel corners with beta and lapse at their worst-case ends).\n\nAXIS 1 -- EARLY-TRUSTED RESCUE (the headline). Base rung M34: A wins the 75% expert at position 3 vs B the 75% expert at position 4 -- both theories prefer A (PBDT [0.54, 0.60], PGDT [0.49, 0.63]). R1 adds the 95%@1 endorsement to B's side (A{75@3} vs B{75@4, 95@1}): PGDT's early-trusted weight (raw 0.028-0.057, always exceeding the 0.75-pair gap 0.011-0.028) flips choice to B at EVERY corner: P(A) in [0.39, 0.47]. PBDT's early-trusted weight (raw 0.011-0.021) usually does NOT close the gap: PBDT straddles chance, P(A) in [0.46, 0.55]. R1p (A{75@3} vs B{75@4, 95@1, 95@6}): PGDT robustly B, P(A) in [0.38, 0.44]; PBDT at chance, [0.48, 0.54] -- DISJOINT BOXES. Observed mirror-corrected P(A) >= 0.46 on R1p (or >= 0.50 on R1) is impossible for PGDT at any admissible parameter combination. Conversely R2 (A{75@3} vs B{75@4, 95@2}): PBDT's position-2 trusted weight (t2*flat, raw 0.005-0.009) never exceeds the 0.75-pair gap, so PBDT stays with A at EVERY corner: P(A) in [0.51, 0.64]; PGDT's position-2 trusted weight (the full flat share, raw 0.013-0.031) sometimes closes it: PGDT at chance, [0.47, 0.51]. Observed P(A) <= 0.50 on R2 kills PBDT's entire space. R2p (helper pair {95@2, 95@6}): PBDT [0.50, 0.61] vs PGDT [0.44, 0.50]. R3 (helper 95@6, the LATE trusted expert whose weight is nearly theory-invariant): BOTH theories stay with A (PBDT [0.54, 0.61], PGDT [0.51, 0.60]) -- the null rung proving the rescue effect is specific to the EARLY trusted channel, not a generic any-95%-helper effect. Note R1/R2 are 1-vs-2 count trials on which tallying picks B (2 cues beat 1) and TTB picks B (the top cue, 95%@1 or 95%@2, is on B's side): PBDT's A-leaning prediction on R2 is unique among all theories considered. The joint readout {R1p <= 0.45 AND R2 >= 0.55} is uniquely PBDT; {R1p <= 0.45 AND R2 <= 0.50} is uniquely PGDT.\n\nAXIS 2 -- TRUSTED-SWING BATTERY (quantitative core). The S-cells (S1: A{75@3, 95@1} vs B{75@4}, etc.) paired with the R-cells give within-subject double differences equal to twice the behavioral weight of a 95% endorsement at positions 1, 2, 6. PGDT's profile is w1:w2:w6 = g_1:1:delta; PBDT's is g_1*t0 : t2 : delta*t1. Boxes: S1 PBDT [0.61, 0.70] vs PGDT [0.57, 0.79]; S2 PBDT [0.58, 0.65] vs PGDT [0.55, 0.74]; S6 PBDT [0.57, 0.61] vs PGDT [0.53, 0.64]. Parametric signatures: (i) the raw position-1 trusted weight ratio PGDT/PBDT = 1/t0 in [2.9, 4.0] -- parameter-free; (ii) the estimated ratio w(95@2)/w(95@6) = 1/delta >= 2.2 at every PGDT corner but can fall to 1.25 under PBDT (t2/(delta*t1)) -- an observed swing ratio <= 2.0 excludes PGDT's entire box, while a ratio >= 3.8 excludes PBDT's; (iii) PGDT's swing@1/swing@6 = g_1/delta >= 4.1 everywhere, reachable by PBDT only above 2.9.\n\nAXIS 3 -- TRUSTED POSITION DUELS. T16 (95@1 vs 95@6): PBDT [0.52, 0.54] vs PGDT [0.52, 0.65] -- PGDT's box extends far higher because its early trusted expert carries the full flat share under the amplified gate. T26 (95@2 vs 95@6): PBDT [0.51, 0.55] vs PGDT [0.51, 0.64]. T12 (95@1 vs 95@2): PBDT [0.53, 0.57] vs PGDT [0.52, 0.60]. At matched (central) corners PGDT's trusted-position gradient is 2-3x PBDT's across the whole battery.\n\nAXIS 4 -- SCREENS AND ANCHORS. D60 (the single 60% expert at position 5 vs the 75% expert at position 3): BOTH theories prefer the 60%-endorsed product (PBDT [0.67, 0.80], PGDT [0.59, 0.87]) -- verifies that subjects read and distrust-invert the communicated validities. VETO (the three 95% experts vs the single 60% expert): both theories prefer the 60%-endorsed product (PBDT [0.32, 0.40], PGDT [0.36, 0.41]) -- the product endorsed by ALL THREE best experts loses to the worst expert's product; excludes normative weighting, TTB, and tallying (3 cues beat 1). DINV (the three 95s vs the 75/75/60 block): both strongly prefer the low-validity block (PBDT [0.11, 0.19], PGDT [0.03, 0.30]). TOT: comprehension anchor, both >= 0.85.\n\nAGGREGATE SIGNATURE. Mirror-corrected P(A) regressed on PGDT reference evidence (w_j proportional to g_j*[(1-tau)/n + tau*inv_j] at tau=0.87, eta=5, delta=0.3, gamma=3.15) versus PBDT reference evidence (w_j proportional to g_j*[t_j*(1-tau)/n + tau*inv_j] at t0=0.30, kappa=1.0, eta=7, delta=0.2, tau=0.92, gamma=2.75) loads with opposite profiles: the two evidence axes differ almost exclusively on the R1/R1p/S1/T16 family, where PGDT's early-trusted term is large and PBDT's is crushed by t0. Single-number readouts: (i) R1p sign (PGDT requires P(A) <= 0.45; PBDT allows >= 0.48 -- disjoint); (ii) R2 sign (PBDT requires P(A) >= 0.51; PGDT <= 0.51); (iii) the fitted trusted-channel position profile (w1, w2, w6) from the swing battery, whose w2/w6 ratio separates the theories at 2.2 (PGDT floor) vs 1.25 (PBDT floor). Every cell type appears in A/B-mirrored form to absorb side and position bias. 32 unique pairs, K = 96 // 32 = 3 repetitions in independently randomized order = exactly 96 trials per subject.",
  "validities": [
    0.95,
    0.95,
    0.75,
    0.75,
    0.6,
    0.95
  ],
  "trial_a_ratings": [
    [
      0,
      0,
      1,
      0,
      0,
      0
    ],
    [
      1,
      0,
      0,
      1,
      0,
      0
    ],
    [
      0,
      0,
      1,
      0,
      0,
      0
    ],
    [
      0,
      1,
      0,
      1,
      0,
      0
    ],
    [
      0,
      0,
      1,
      0,
      0,
      0
    ],
    [
      0,
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
      0,
      0,
      0
    ],
    [
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
      0,
      0,
      0
    ],
    [
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
      1,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      1,
      0,
      0
    ],
    [
      0,
      1,
      1,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      1,
      0,
      0
    ],
    [
      0,
      0,
      1,
      0,
      0,
      1
    ],
    [
      0,
      0,
      0,
      1,
      0,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      1
    ],
    [
      0,
      1,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      1
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      1,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      1,
      0,
      0
    ],
    [
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
      0,
      0,
      0
    ],
    [
      1,
      1,
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
      0
    ],
    [
      1,
      1,
      0,
      0,
      0,
      1
    ],
    [
      0,
      0,
      1,
      1,
      1,
      0
    ],
    [
      1,
      1,
      1,
      1,
      1,
      1
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0
    ]
  ],
  "trial_b_ratings": [
    [
      1,
      0,
      0,
      1,
      0,
      0
    ],
    [
      0,
      0,
      1,
      0,
      0,
      0
    ],
    [
      0,
      1,
      0,
      1,
      0,
      0
    ],
    [
      0,
      0,
      1,
      0,
      0,
      0
    ],
    [
      0,
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
      0,
      0,
      0
    ],
    [
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
      0,
      0,
      0
    ],
    [
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
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      1,
      0,
      0
    ],
    [
      1,
      0,
      1,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      1,
      0,
      0
    ],
    [
      0,
      1,
      1,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      1,
      0,
      0
    ],
    [
      0,
      0,
      1,
      0,
      0,
      1
    ],
    [
      0,
      0,
      0,
      0,
      0,
      1
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      1
    ],
    [
      0,
      1,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      1,
      0,
      0
    ],
    [
      0,
      0,
      1,
      0,
      0,
      0
    ],
    [
      0,
      0,
      1,
      0,
      0,
      0
    ],
    [
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
      0,
      0,
      1,
      0
    ],
    [
      1,
      1,
      0,
      0,
      0,
      1
    ],
    [
      0,
      0,
      1,
      1,
      1,
      0
    ],
    [
      1,
      1,
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
      0,
      0
    ],
    [
      1,
      1,
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
  "input_tokens": 31716,
  "output_tokens": 126637,
  "reasoning_tokens": 122931,
  "provider": "Sail Research"
}
```
