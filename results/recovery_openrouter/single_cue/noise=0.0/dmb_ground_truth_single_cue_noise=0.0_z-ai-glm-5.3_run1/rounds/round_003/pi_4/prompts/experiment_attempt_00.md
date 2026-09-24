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
**Description:** Flattened-Diagnosticity Evidence Integration (the extreme-compression corner of the Bayesian cue-counting family): people integrate ALL binary expert ratings into one evidence score, where each discriminating cue casts a sign vote weighted by w_j = max(log(v_j/(1-v_j)) - tau, 0)^delta — chance cues (v=0.5) are annihilated exactly, but among genuinely diagnostic cues the subjective weight is a HEAVILY flattened (delta ~ 0.15) version of the normative log-odds, and choice is a low-gain softmax (2*beta ~ 0.6) plus a small uniform lapse. The new theoretical claim is that the compression exponent delta is itself the structural margin-compression device: shrinking delta equalizes weights, which GROWS count-dominated conflict margins (where cues split across options) while SHRINKING alignment-dominated cumulative margins (where several strong cues pile onto one side). This differential action — not any global gain knob, and not trial-level normalization — is what lets a single small beta simultaneously produce moderate TTB-leaning conflict-trial behavior (Exp 1), a slightly positive tally-vs-top-cue regression slope (Exp 2), graded ~0.62 tally-following (Exp 3), and near-chance |d|=1 agreement driven by the cancellation between super-expert matches and super-expert reversals (Exp 4).

**Parameters:**
- delta: [0.1, 0.2]
- tau: [0.0, 0.04]
- beta: [0.27, 0.37]
- epsilon: [0.0, 0.05]
- validities: validities

**`predict source code`:**
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

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** Skeptical (Inverted-Trust) Tallying with a CALIBRATED trust cliff. People integrate ALL binary expert ratings as sign votes (+1/-1/0) into a zero-sum evidence score whose per-cue weight is a non-monotone function of stated validity: (i) near-chance experts (v ~ 0.5) carry a small NEGATIVE weight (their endorsements are treated as mildly anti-diagnostic noise); (ii) trust rises smoothly to a peak at v ~ 0.9 (the most trusted experts); (iii) claimed near-certainty (v > ~0.96) falls past a cliff into a MODERATE negative weight of roughly MINUS ONE trusted-cue unit — a 99% expert is treated as approximately one reliable anti-cue, not a veto. The calibrating claim is that distrust magnitude is matched to trust magnitude (|w(0.99)| ~ w_peak ~ 1, logit contribution 2*beta*w(0.99) ~ -2.1): a single super-expert loses to any 2-cue opposing coalition (weak-side choice in super-expert conflicts) but cannot invert coalitions that include it (moderate tally-following when the 0.99 cue sits inside a tally majority). This intermediate depth sits deliberately between the previously falsified extremes: the deep veto (iter 6, which deterministically inverted Experiment 3) and the shallow discount (iter 7, which let super-experts win conflicts, SSR 0.69).

**Parameters:**
- v_lo: [0.505, 0.515]
- v_peak: [0.89, 0.91]
- v_hi: [0.957, 0.963]
- rho1: [0.68, 0.75]
- rho2: [0.25, 0.35]
- rho_lo: {1}
- kappa_lo: [0.075, 0.105]
- kappa_hi: [1.0, 1.3]
- tau: [0.011, 0.014]
- beta: [0.9, 1.1]
- epsilon: [0.04, 0.07]
- validities: validities

**`predict source code`:**
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

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "CORE LOGIC. Both theories share the identical architecture (sign votes x validity weights x softmax x lapse) and differ ONLY in the weight function. My advocated theory (FDEI) compresses diagnosticity heavily (delta in [0.1,0.2]), so among genuinely diagnostic cues its weights are nearly FLAT: w(0.98) in [1.14,1.31], w(0.90) in [1.08,1.17], w(0.75) ~ 1.01, w(0.55) in [0.69,0.85]. The competitor (SITT) has a steep inverted-U: w(0.55) ~ 0.18-0.23, w(0.75) ~ 0.68-0.74, w(0.90) ~ 0.95-1.0 (peak), w(0.98) NEGATIVE in [-1.14,-0.70]. The single most informative behavioral quantity is therefore the EXCHANGE RATE between a strong expert and weak (55%) experts, measurable as the crossover point k* in a weak-cue dose-response ladder. Validities: [0.98, 0.90, 0.75, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55] -- one expert per regime plus SIX 55% experts, enough to bracket both theories' crossovers. This is a design no prior experiment used: no v=0.5 annihilation logic ([4]), no tally-margin decorrelation ([1],[2]), no bounded-box ceiling attack ([3]), no validity-ratio ER between diagnosticity classes ([5]), and unlike [6] the flagship is not a standalone 0.99 inversion but graded crossover ladders (the 0.98 cliff cells here appear only inside coalitions).\n\nFAMILY K (flagship crossover ladder). K1-K6: A wins the 90% expert, B wins k of the 55% experts, k=1..6. The crossover k* = w(0.90)/w(0.55) is boxed DISJOINTLY: FDEI k* in [1.27, 1.69] (crosses 0.5 between k=1 and k=2); SITT k* in [4.1, 5.6] (crosses between k=4 and k=5). Every single cell is box-disjoint. FDEI center (delta=0.15, tau=0.01, beta=0.32): p(A) = .55, .43, .32, .22, .15, .10 for k=1..6. SITT center (v_lo=.51, v_peak=.90, v_hi=.96, rho1=.7, rho2=.3, kappa_hi=1.15, beta=1.0, eps=.055): p(A) = .81, .75, .68, .59, .49, .40. Box-level: at k=1, FDEI [.53,.59] vs SITT [.75,.84]; k=2, [.42,.46] vs [.69,.79]; k=3, [.31,.42] vs [.61,.72]; k=4, [.22,.31] vs [.51,.64]; k=5, [.15,.24] vs [.42,.55]; k=6, [.10,.17] vs [.33,.45]. Two psychometric curves that cross each other in opposite directions: the observed crossover location is a direct, interpretable estimate of the strong:weak exchange rate, separated by a factor of ~3 across the two boxes. Km2/Km4/Km6 mirror the ladder for side balance.\n\nFAMILY R (second crossover at a mid-strength reference). R1-R4: A wins the 75% expert vs k 55% experts (k=1..4); Rm3 mirrors. FDEI crossover k* = w(0.75)/w(0.55) in [1.18, 1.46]; SITT in [3.0, 4.1] -- again disjoint. Centers: FDEI .54/.42/.31/.22 vs SITT .72/.64/.55/.45. Two independent exchange-rate estimates, both disjoint, test whether the dissociation scales with reference strength as each theory's weight function demands (FDEI: ER roughly constant ~1.3; SITT: ER shrinks 4.9 -> 3.5 as reference drops from 0.90 to 0.75).\n\nFAMILY D/C/H (cliff cells: 0.98 as anti-cue vs near-equal-cue). D1 (A wins 0.98, B wins 0.90): FDEI box [.50,.53] -- a coin flip, because compressed weights make 98% and 90% nearly exchangeable; SITT box [.04,.06] -- subjects should choose AGAINST the 98% endorsement. D1m mirrors. C1 (A wins 0.90+0.75, B wins 0.98+one 0.55): FDEI [.49,.55] (net evidence ~0); SITT [.93,.97] (the 98% endorsement is worth about one trusted ANTI-cue, so the two trusted experts plus the inversion overwhelm it). C1m mirrors. H2 (A wins 0.98 + two 55%s, B wins 0.90): FDEI [.66,.80] (weak majority plus super-expert beats the peak expert); SITT [.06,.13] (the super-expert's anti-vote sinks the coalition). H2m mirrors. These cells separate a near-tie from a 90-point swing.\n\nHONEST KNIVES AGAINST MY OWN THEORY. K1 and R1 are the sharpest: FDEI's delta<=0.2 compression caps the 90-vs-55 exchange rate at 1.69, so it cannot predict p(A) > 0.59 on K1 or > 0.56 on R1 for ANY parameter setting. If subjects follow the strong expert at p >= 0.70 on K1, my flattening claim is falsified outright and the design doubles as a direct delta-estimation experiment (observed k* maps monotonically onto delta). Conversely K3/K4 and R2/R3 falsify SITT if subjects favor the weak stacks at all (SITT demands p(A) >= 0.51 at K4 for every parameter).\n\nCONTROLS. X1/X2 (one 55% expert won by each side): BOTH theories must predict exactly 0.50 for every parameter -- a parameter-free side/display-bias assay. G1/G1m are agreement anchors (both theories > 0.85 at center: FDEI .94, SITT .96) pinning lapse/temperature. Tied features are a MIX of 0/0 and 1/1 ties so that 'count the 1s' is not a viable heuristic (e.g., K6 has A with three 1s vs B with eight 1s while the theories disagree only about the discriminators). No all-tie trials; every trial requires scanning all 9 experts. 24 unique pairs x K = 96//24 = 4 reps = 96 trials; sides balanced (10 A-tilted, 12 B-tilted, 2 neutral under FDEI centers; every non-anchor family contains an A/B mirror). PREDICTED PATTERN IF MY THEORY IS RIGHT: shallow FDEI psychometrics crossing 0.5 between k=1 and 2 in both ladders, coin flips on D1/C1, A-choice ~0.74 on H2, exactly 0.50 on X. IF THE COMPETITOR IS RIGHT: steep SITT psychometrics crossing between k=4 and 5 (K) and k=3 and 4 (R), near-ceiling strong-expert following at low k, ~0.05 on D1, ~0.95 on C1, ~0.07 on H2. The boxes are disjoint on 20 of 24 cells.",
  "validities": [
    0.98,
    0.9,
    0.75,
    0.5,
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
      0,
      0,
      1,
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
      1,
      0,
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
      0,
      0,
      0,
      1
    ],
    [
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0
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
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1
    ],
    [
      1,
      0,
      1,
      0,
      0,
      1,
      0,
      0,
      1
    ],
    [
      0,
      1,
      1,
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
      1,
      0,
      0,
      1
    ],
    [
      0,
      1,
      0,
      1,
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
      1,
      0,
      0,
      1,
      0
    ],
    [
      1,
      0,
      0,
      1,
      0,
      1,
      0,
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
      0,
      1
    ],
    [
      0,
      1,
      1,
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
      1,
      1,
      1,
      1,
      0,
      1
    ],
    [
      1,
      0,
      0,
      0,
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
      0,
      1,
      1,
      0,
      0,
      1
    ]
  ],
  "trial_b_ratings": [
    [
      1,
      0,
      0,
      1,
      1,
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
      1
    ],
    [
      1,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1
    ],
    [
      0,
      1,
      1,
      0,
      0,
      0,
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
      0,
      0
    ],
    [
      1,
      0,
      0,
      1,
      0,
      1,
      0,
      0,
      1
    ],
    [
      0,
      1,
      0,
      1,
      1,
      0,
      1,
      0,
      0
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
      0,
      1,
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
      0,
      1,
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
      1,
      0,
      0,
      1
    ],
    [
      1,
      0,
      0,
      1,
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
      1,
      1,
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
      1,
      0,
      0,
      1
    ],
    [
      0,
      1,
      0,
      1,
      0,
      0,
      0,
      0,
      1
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
      0,
      0,
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
      0,
      1,
      0
    ],
    [
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      0,
      1
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 13774,
  "output_tokens": 32760,
  "reasoning_tokens": 29952,
  "provider": "Sail Research"
}
```
