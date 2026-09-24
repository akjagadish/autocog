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
**Description:** Softened Validity-Distrust with Count Mixture (SVD). Every expert endorsement is a liability whose magnitude increases monotonically in the endorser's RAW claimed validity: d_j = c0 + c1*v_j^kappa, with no dampening near v=0.5 and no trust refund. Decision making is a dual-process mixture at the trial level: with probability w the subject is a 'skeptic' who integrates the accumulated-suspicion difference D = P(B) - P(A) through sigmoid(beta*D); with probability (1-w) the subject is a 'counter' who simply chooses the option with FEWER total endorsements (indifferent at matched counts). Two heterogeneity mechanisms soften the pooled extremes and capture between-subject variability: (i) the sigmoid slope beta is sampled from a WIDE per-subject range, so the population is a mixture of shallow and steep sigmoids — sensitive near D=0 (matched-count validity-composition trials) yet compressed at large |D| (unequal-count trials); (ii) the uniform lapse epsilon is also sampled from a wide per-subject range, which pulls every saturated large-D assay toward 0.5 while adding genuine between-subject dispersion in overall choice rates.

**Parameters:**
- c0: [0.05, 0.12]
- c1: [0.95, 1.3]
- kappa: [1.5, 1.8]
- beta: [1.2, 2.6]
- w: [0.80, 0.92]
- epsilon: [0.12, 0.24]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Softened Validity-Distrust with Count Mixture (SVD).
    # Stimulus: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    #
    # Branch 1 (skeptic, prob w): each endorsement is a LIABILITY whose
    # magnitude grows with the endorser's RAW claimed validity:
    #   d_j = c0 + c1 * v_j^kappa   (kappa >= 1.5; NO dampening near
    #                                v = 0.5, NO trust refund)
    #   accumulated suspicion: P(X) = sum_j x_j * d_j
    #   D = P(B) - P(A)  (positive favors A)
    #   P_skeptic(A) = sigmoid(beta * D)
    # Branch 2 (counter, prob 1-w): pure tallying on endorsement COUNTS;
    #   choose the option with FEWER total endorsements; exactly
    #   indifferent (0.5) at matched counts.
    # Mixture: P_core(A) = w * P_skeptic(A) + (1-w) * P_count(A),
    # mixed with a uniform lapse epsilon.
    # History is ignored: validities are in the instructions, so the
    # distrust weights are fixed for the whole block. beta AND epsilon
    # are sampled per subject from WIDE ranges: between-subject
    # heterogeneity in the sigmoid slope and in the lapse rate are
    # explicit parts of the theory.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"SVD expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n_features = stim.shape[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.shape[0] != n_features:
        raise ValueError(
            f"validities length {v.shape[0]} != n_features {n_features}."
        )
    v = np.clip(v, 0.5, 1.0)  # defensive; no logs taken, v=1 harmless.

    kappa = float(parameters["kappa"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    c0 = float(parameters["c0"])
    c1 = float(parameters["c1"])
    w_mix = float(parameters["w"])

    # --- Skeptic branch: raw validity-proportional distrust charge ---
    d = c0 + c1 * np.power(v, kappa)
    pen_a = float(np.dot(d, a))
    pen_b = float(np.dot(d, b))

    # D > 0 favors A (B carries more accumulated suspicion).
    D = pen_b - pen_a
    x = beta * D
    p_skeptic_a = 0.5 * (1.0 + np.tanh(0.5 * x))  # stable sigmoid

    # --- Counter branch: pure endorsement-count tallying ---
    cnt_a = float(np.sum(a))
    cnt_b = float(np.sum(b))
    if cnt_a < cnt_b:
        p_count_a = 1.0      # A has fewer endorsements -> prefer A
    elif cnt_a > cnt_b:
        p_count_a = 0.0
    else:
        p_count_a = 0.5      # matched counts -> indifferent

    # --- Dual-process mixture ---
    p_a = w_mix * p_skeptic_a + (1.0 - w_mix) * p_count_a
    p_core = np.array([p_a, 1.0 - p_a], dtype=float)

    # Independent lapse: with probability epsilon pick uniformly.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Graded Dual-Route Distrust with a Bimodal Strategy Population (GDRD-B). Subjects evaluate expert endorsements through two concurrent routes. ROUTE 1 (analytic distrust): every endorsement is a liability whose magnitude grows monotonically in the endorser's RAW claimed validity, d_j = c0 + c1*v_j^kappa; accumulated suspicion P(X) = sum_j x_j*d_j is compared across options through sigmoid(beta*(P(B)-P(A))), plus a small residual micro-trust edge alpha*sum_j clip(logit(v_j),0,3)*(a_j-b_j) (capped so perfect experts exert bounded, not unbounded, trust pull). ROUTE 2 (graded counting): a continuous preference for the fewer-endorsement option, sigmoid(gamma*(cnt_B-cnt_A)). The population is BIMODAL in the route weight w: a large majority (~88%) of 'skeptics' (w in [0.85, 0.95], moderate lapse) coexists with a minority (~12%) of 'counters' (w in [0.50, 0.62], low lapse, decisive count slope gamma >= 2). The mixture is implemented as a quantile transform of a single uniform per-subject population variable u_pop, which holds the population-MEAN route weight at its base value while exploding the between-subject behavioral variance — the empirically observed 3-12x super-binomial variance on the saturation assays (Exps 11, 12) that no uniform range-widening could reproduce.

**Parameters:**
- c0: [0.06, 0.13]
- c1: [1.00, 1.35]
- kappa: [1.6, 2.1]
- beta: [1.3, 2.5]
- alpha: [0.0, 0.05]
- u_pop: [0, 1]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Graded Dual-Route Distrust, bimodal-population variant (GDRD-B).
    #
    # Stimulus: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    #
    # ROUTE 1 (analytic distrust, weight w): each endorsement is a
    # LIABILITY whose magnitude grows with the endorser's RAW claimed
    # validity:
    #   d_j = c0 + c1 * v_j^kappa          (kappa >= 1, no dampening
    #                                        near v = 0.5)
    #   accumulated suspicion: P(X) = sum_j x_j * d_j
    #   D = (P(B) - P(A)) + alpha * sum_j clip(logit(v_j), 0, 3)*(a_j - b_j)
    #     (micro-trust term; the logit is CAPPED at 3 so perfect
    #      experts exert bounded, not unbounded, trust pull)
    #   P_skeptic(A) = sigmoid(beta * D)
    # ROUTE 2 (graded counting, weight 1-w): a CONTINUOUS preference
    # for the option with fewer total endorsements:
    #   P_count(A) = sigmoid(gamma * (cnt_B - cnt_A))
    #
    # BIMODAL POPULATION: instead of sampling w, gamma, epsilon from
    # independent uniform ranges (which shifts the population MEANS
    # whenever the ranges widen), a single uniform population quantile
    # u_pop in [0, 1] is deterministically mapped onto a two-component
    # mixture:
    #   u_pop < 0.12  -> 'counter' minority:  w in [0.50, 0.62],
    #                    epsilon in [0.05, 0.10], gamma in [2.0, 3.0]
    #   otherwise     -> 'skeptic' majority:   w in [0.85, 0.95],
    #                    epsilon in [0.08, 0.18], gamma in [1.5, 3.0]
    # This holds E[w] at the calibrated base value (~0.86) while
    # producing genuine between-subject bimodality — the mechanism the
    # real super-binomial variances on Exps 11/12 demand.
    #
    # History is ignored: validities are communicated in the
    # instructions, so the distrust weights are fixed for the whole
    # block. beta is sampled per subject from a wide range.
    if isinstance(state, dict):
        a_vec = np.asarray(state["option_a_ratings"], dtype=float).ravel()
        b_vec = np.asarray(state["option_b_ratings"], dtype=float).ravel()
        stim = np.vstack([a_vec, b_vec])
    else:
        stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"GDRD expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n_features = stim.shape[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.shape[0] != n_features:
        raise ValueError(
            f"validities length {v.shape[0]} != n_features {n_features}."
        )
    # Power route: v = 1 is harmless (no logs). Trust route: guard
    # the logit against v = 1 (perfect experts) and v < 0.5.
    v_pow = np.clip(v, 0.5, 1.0)
    v_tr = np.clip(v, 0.5, 1.0 - 1e-6)

    kappa = float(parameters["kappa"])
    beta = float(parameters["beta"])
    c0 = float(parameters["c0"])
    c1 = float(parameters["c1"])
    alpha = float(parameters["alpha"])
    u_pop = float(parameters["u_pop"])

    # --- Bimodal population mixture over (w, epsilon, gamma) ---
    # Quantile transform of one uniform variable: the population mean
    # route weight stays at the calibrated base (~0.86) while the
    # between-subject distribution becomes bimodal.
    f_counter = 0.12
    u = min(max(u_pop, 0.0), 1.0)
    if u < f_counter:
        # Counter minority: near-balanced routes, low lapse, decisive
        # graded count slope.
        t = u / f_counter
        w_mix = 0.50 + 0.12 * t          # [0.50, 0.62]
        epsilon = 0.05 + 0.05 * t        # [0.05, 0.10]
        gamma = 2.0 + 1.0 * t            # [2.0, 3.0]
    else:
        # Skeptic majority: analytic-distrust dominant, moderate lapse.
        t = (u - f_counter) / (1.0 - f_counter)
        w_mix = 0.85 + 0.10 * t          # [0.85, 0.95]
        epsilon = 0.08 + 0.10 * t        # [0.08, 0.18]
        gamma = 1.5 + 1.5 * t            # [1.5, 3.0]
    w_mix = float(min(max(w_mix, 0.5), 0.95))

    # --- Route 1: raw validity-proportional distrust charge ---
    d = c0 + c1 * np.power(v_pow, kappa)
    pen_a = float(np.dot(d, a))
    pen_b = float(np.dot(d, b))

    # Residual micro-trust: a small validity-weighted endorsement
    # edge. logit(0.5) = 0 (coin-flip experts contribute nothing) and
    # the logit is CAPPED at 3 so perfect experts (v = 1.0) exert a
    # bounded pull (validated twice on the Exp-12 saturation assay).
    lg = np.clip(np.log(v_tr / (1.0 - v_tr)), 0.0, 3.0)
    trust = float(np.dot(lg, a - b))

    # D > 0 favors A (B carries more accumulated suspicion, net of
    # the small trust edge).
    D = (pen_b - pen_a) + alpha * trust
    p_skeptic_a = 0.5 * (1.0 + np.tanh(0.5 * beta * D))  # stable sigmoid

    # --- Route 2: graded endorsement-count preference ---
    cnt_a = float(np.sum(a))
    cnt_b = float(np.sum(b))
    p_count_a = 0.5 * (1.0 + np.tanh(0.5 * gamma * (cnt_b - cnt_a)))

    # --- Dual-route mixture ---
    p_a = w_mix * p_skeptic_a + (1.0 - w_mix) * p_count_a
    p_a = float(min(max(p_a, 0.0), 1.0))
    p_core = np.array([p_a, 1.0 - p_a], dtype=float)

    # Independent lapse: with probability epsilon pick uniformly.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This design creates a quantitative double dissociation between Take-The-Best (TTB) and Tallying by exploiting their core structural difference: TTB's choice depends ONLY on the single highest-validity discriminating cue, while Tallying's choice depends ONLY on the total count of feature-wise wins. Validities are strictly spread and strictly ordered (0.90, 0.80, 0.70, 0.60, 0.55) so TTB's cue cascade is unambiguous. The design contains three families of trials, each mirrored across A/B to control for side bias. (1) Tally-tie trials (e.g., A=[1,0,0,0,0] vs B=[0,1,0,0,0]): the top cue favors one option but the tally is exactly 1-1. TTB predicts P(top-cue winner) = (1-eps)*sigmoid(beta) + eps/2, which rises above 50% as beta grows; Tallying predicts EXACTLY 50% regardless of beta (softmax over equal scores is uniform), so any systematic preference here falsifies Tallying. (2) The critical conflict-gradient family: the top cue favors A, but B wins on an increasing number of lower cues (tally margins 1-2, 1-3, 1-4 against A). TTB predicts P(A) is CONSTANT across these margins (always sigmoid(beta) for A, since only cue 1 is consulted), whereas Tallying predicts a monotone, graded shift toward B as the tally margin grows (sigmoid(beta), sigmoid(2*beta), sigmoid(3*beta)). This flat-vs-graded profile is the central quantitative signature: a subject whose P(A) declines with tally margin is following Tallying; a subject whose P(A) stays flat is following TTB. (3) A second-order family where cues 1-2 tie and discrimination first occurs at cue 3 (with tally conflicts), testing whether the cascade truly skips tied cues. All 16 unique pairs are non-degenerate (no all-tie pairs) and every pair places the two theories in disagreement or in a guess-vs-decide asymmetry. With K = 96 // 16 = 6 repetitions per pair, each subject contributes 6 samples per cell of the design, giving good power to estimate the flat-vs-graded choice gradient that separates the theories.

[1] This design is the complementary mirror of the already-explored experiment: instead of holding TTB's deciding cue fixed and varying the tally margin, it holds the TALLY MARGIN fixed and varies WHICH CUE TTB's cascade lands on. Six experts with validities [0.95, 0.90, 0.85, 0.65, 0.60, 0.55] create a sharp gap between high-validity cues (1-3) and low-validity cues (4-6), so the identity of TTB's first discriminating cue is unambiguous. Every trial is built around a designated tally-winner W with a fixed tally margin m in {1, 2, 3}, crossed with the DIRECTION of the top discriminating cue: CONGRUENT cells (top cue favors W, so TTB and Tallying agree in direction) and CONFLICT cells (top cue favors the loser, so TTB and Tallying point in opposite directions). Within conflict cells at m=1, the deciding cue's position d is also varied (d=1..4), testing whether choices track the validity of the deciding cue. The quantitative signature is a 2x3 factorial of predicted P(W): Tallying predicts P(W) = sigmoid(m*beta) + lapse, i.e., IDENTICAL across congruent/conflict cells at each margin and a monotone gradient across margins [sigma(beta) < sigma(2*beta) < sigma(3*beta)]. TTB predicts P(W) = sigmoid(beta) in every congruent cell and 1 - sigmoid(beta) in every conflict cell, i.e., IDENTICAL across margins (its noise model does not scale with the number of supporting cues) and independent of which cue d decides. Thus Tallying predicts a main effect of margin only; TTB predicts a main effect of top-cue direction only. A subject whose P(W) rises with m at fixed direction is Tallying; a subject whose P(W) is flat across m but flips with cue direction is TTB. Even the congruent cells dissociate at m>=2 (sigma(beta) vs sigma(2*beta)), and the m=1 congruent cells serve as agreement baselines for estimating beta. All 12 base pairs are mirrored (A/B swapped) to control side bias, giving 24 unique pairs with K = 96 // 24 = 4 repetitions each (96 trials total, 8 samples per base cell per subject). No pair is degenerate (every pair has discriminating cues, none is an all-tie), and no pair leaves both theories in pure-guess agreement. Secondary exploratory axis: both theories predict FLAT P(W) across the deciding-cue position d (TTB because its softmax does not scale with validity, Tallying because it ignores validity entirely), so any graded sensitivity to d would falsify both — but the primary dissociation rests on the margin x direction interaction.

[2] This design exploits a structural chasm between TTB and VWEI: TTB's score is binary (winner=1, loser=0), so its softmax NEVER sees cue validity, coalition size, or coalition composition. Consequence: TTB predicts ONE constant choice probability, P(pick TTB's winner) = (1-eps)*sigmoid(beta) + eps/2, on ALL 16 pairs — whether the deciding cue is the 90% expert or the 76% expert, and whether 0 or 7 lower cues oppose it. VWEI, with its bounded parameters (gamma in [2.8,3.2], beta in [0.18,0.23]), predicts a specific, monotone, non-flat profile across the same 16 pairs. The validity vector [0.90, 0.78, 0.76, 0.74, 0.72, 0.70, 0.68, 0.66, 0.64] is engineered so that (a) cue 1 (v=0.90, w=10.6) dominates every possible coalition, (b) cue 2 (v=0.78, w=2.03) is beaten by coalitions of 2+ of the next cues (w3+w4=2.68) but not by any single lower cue or by weak cues (w8+w9=0.48) — creating sign flips that are robust across the entire gamma range [2.8, 3.2] and beta range [0.18, 0.23] (verified: pair-7 E ranges -0.93..-0.95; pair-11 E ranges -3.35..-3.41; pair-13 E ranges +0.96..+1.14). Four dissociation signatures: (1) COALITION-SIZE GRADIENT (pairs 5-12): cue 1 ties, cue 2 picks the TTB winner, and the opposing coalition below cue 2 grows from 1 to 2 to 3 to 7 cues. TTB: P(cue-2 winner) flat at sigmoid(beta). VWEI: monotone decline 0.525 -> 0.468 -> 0.426 -> 0.354, crossing 50% between one and two opponents. (2) COMPOSITION AT FIXED COUNT (pairs 13/14 vs 7/8): exactly two opposing cues, but weak ones (experts 8-9) vs strong ones (experts 3-4). TTB: identical predictions (cue 2 decides in all four; validity-blind). VWEI: an ~11-point gap (0.577 vs 0.468) — this weight-magnitude sensitivity at fixed count falsifies TTB (and any count-based heuristic) while confirming VWEI's amplified log-odds weighting. (3) DECIDING-CUE VALIDITY FLATNESS (pairs 1-4 vs 5-14 vs 15-16): when the 90% expert decides (pairs 1-4, agreement baselines), VWEI predicts P(winner)=0.85-0.89; when the 78% expert decides, ~0.35-0.58; when the 76% expert decides against a 3-cue coalition (pairs 15-16, cascade-depth test), 0.447. TTB predicts the SAME probability in all three regimes — a large cross-family validity effect falsifies TTB, flatness falsifies VWEI. (4) SIGN CONFLICTS: on pairs 7-12 and 15-16 the theories point in OPPOSITE directions (TTB follows cue 2/3; VWEI's evidence sum favors the coalition), so a single trial type can flip a subject's majority choice. Every pair is non-degenerate (a discriminating cue always exists), the design is perfectly side-balanced (8 pairs with TTB winner = A, 8 with TTB winner = B; pairs are exact A/B mirrors), and with 16 unique pairs, K = 96//16 = 6 gives 96 trials and 12 samples per mirrored cell — enough per-subject power to estimate the flat-vs-graded profile that adjudicates the theories. A TTB subject is a horizontal line; a VWEI subject is a specific descending curve that no constant can fit (VWEI's spread 0.89->0.35 exceeds anything TTB's two noise parameters can produce, since TTB has no mechanism for any stimulus-driven variation at all).

[3] NOVEL AXIS (vs. Exps 0-2): a SHALLOW validity gradient with a NEAR-TIE at the top. All prior dissociation designs used a dominant top cue; here the maximum validity is 0.74 and the top two experts differ by only 0.01 (0.74 vs 0.73). This is the sharpest possible probe of TTB's ordinal cascade against VWEI's cardinal weights: TTB commits HARD to the 74% expert whenever it discriminates (a 1-point validity edge produces a full-strength decision), while VWEI's amplified weights for the top two differ by only w1-w2 = 0.16 (at gamma=3), so a single 70% expert (w=0.61) outweighs the entire top-two gap. Validities: [0.74, 0.73, 0.70, 0.69, 0.68, 0.67, 0.66, 0.65, 0.64] (9 experts; weights at gamma=3: 1.145, 0.984, 0.609, 0.512, 0.428, 0.356, 0.292, 0.237, 0.190; no cue can dominate: even all 8 lower weights sum to only 3.61 vs w1=1.145, so coalitions of 2-4 lower cues can override any single cue). 12 base pairs, each exactly A/B-mirrored (24 unique pairs, K=96//24=4, 96 trials, 8 samples per base cell per subject). In every base pair TTB's winner is A, so TTB predicts ONE constant P(A) = (1-eps)*sigmoid(beta)+eps/2 on all 12 base pairs (and its complement on mirrors) -- stimulus-invariant by construction. VWEI predicts a specific 25-point graded, anti-symmetric profile (verified robust across the full parameter box gamma in [2.8,3.2], beta in [0.18,0.23]; all evidence signs stable). FOUR QUANTITATIVE SIGNATURES: (1) NEAR-TIE ATTENUATION GRADIENT (pairs 1-8): cue 1 favors A, cue 2 favors B, and the net lower-cue support sweeps from +7 to -7. TTB: flat (cue 1 always decides for A). VWEI: monotone P(A) = 0.636, 0.585, 0.538, 0.508, 0.478, 0.452, 0.429, 0.388 (range across the parameter box: 0.617-0.670 down to 0.380-0.396), crossing 50% between zero and one opposing cue. Any nonzero slope of P(A) on coalition weight falsifies TTB; a flat profile falsifies VWEI. (2) CONTINUITY AT THE VALIDITY TIE (pair 4): the 0.74-vs-0.73 conflict alone. TTB commits fully to A at sigmoid(beta); VWEI predicts 0.508 (range 0.507-0.510) -- statistically a guess -- because a 1-point validity edge is worth only 0.16 evidence units. (3) CASCADE-INVARIANCE VIOLATION (pair 1 vs pair 11): pair 1 is decided by cue 1 with all lower cues congruent; pair 11 is decided by cue 2 (cue 1 ties) with all lower cues opposed. TTB predicts IDENTICAL P(A) on both (first discriminating cue favors A in both; its softmax never sees anything else). VWEI predicts 0.636 vs 0.420 -- a 22-point gap in opposite directions of deviation that no constant can fit. Similarly pair 8 vs 11 (0.388 vs 0.420) shows TTB-equivalent pairs differing by deciding-cue identity. (4) SIGN CONFLICTS: on 7 of 12 base pairs (5-8, 10-12) VWEI predicts a B-majority (P(A) = 0.388-0.478) where TTB predicts an A-majority; majority-B choices there directly falsify TTB at any beta (TTB's winner probability is always >= 0.5). Family C (pair 12) additionally tests deep-cascade skipping: cues 1-3 all tie, cue 4 decides for A against a 5-cue coalition; TTB says A at the same constant; VWEI says 0.451. Because TTB's prediction on this entire design is a single stimulus-invariant constant (plus its mirror), VWEI's 25-point stimulus-driven spread -- spanning both sides of 50% with a specific monotone shape -- is a profile no TTB parameterization can approximate, while every VWEI prediction is sign-robust across its entire parameter box. All pairs are non-degenerate (every pair has discriminating cues; no all-tie pairs; no pair where both theories guess), the design is perfectly side-balanced (12 TTB-winner-A, 12 TTB-winner-B), and every trial requires actually comparing the two rating vectors.

[4] NOVEL AXIS (vs. Exps 0-3): ZERO-WEIGHT EXPERTS. All prior designs dissociated theories that all read endorsements as positive evidence. SDI versus VWEI is a clash of SIGN and of WEIGHTING PROFILE: SDI treats every endorsement as a liability (penalty d_j = (1-v_j)^delta, nearly uniform at ~0.86-1.0 across the whole validity range for delta in [0, 0.05]) and prefers the option with FEWER endorsements; VWEI weights endorsements by amplified log-odds (w_j = logit(v_j)^gamma, enormous spread: 25.5 at v=.95 down to 0.61 at v=.70) and prefers the option with MORE validity-weighted wins. The design's keystone is four experts at validity EXACTLY 0.50: VWEI's weight for a 50% expert is logit(0.5)^gamma = 0 for EVERY gamma — a 50% expert's endorsement is structurally invisible to VWEI under any parameterization — while SDI's skepticism weight is (1-0.5)^delta in [0.966, 1], its NEAR-MAXIMUM penalty. A coin-flip expert's praise is, for SDI, a full unit of suspicion. Validities: [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.50, 0.50, 0.50, 0.50] (10 experts; VWEI weights at gamma=3: 25.5, 10.6, 5.2, 2.7, 1.3, 0.6, 0, 0, 0, 0; SDI penalty weights at delta=0.025: 0.93, 0.94, 0.95, 0.96, 0.97, 0.97, 0.98, 0.98, 0.98, 0.98). 16 base pairs, each exactly A/B-mirrored (32 unique pairs, K = 96//32 = 3, 96 trials, 6 samples per base cell per subject). In every base pair VWEI's winner is A, so the design is perfectly side-balanced. FOUR QUANTITATIVE SIGNATURES (P(A) at SDI mid-params delta=.025, beta=1, eps=.115, alpha=.025; VWEI mid-params gamma=3, beta=.205, eps=.015; all signs verified robust across the FULL parameter boxes): (1) JUNK-ENDORSEMENT GRADIENT, VWEI STRUCTURALLY FLAT (pairs 1-5): the 95% expert endorses A; k = 0..4 of the 50% experts also endorse A. VWEI: E = 25.5 regardless of k (50% weight is exactly 0 at any gamma), so P(A) is flat at 0.98-0.99. SDI: each junk endorsement adds ~0.98 penalty units to A, so P(A) declines monotonone 0.32 -> 0.18 -> 0.11 -> 0.08 -> 0.06 (~6-7 points per junk endorsement; box ranges 0.32-0.33 down to 0.05-0.08). ANY measurable effect of 50%-expert endorsements on choice falsifies VWEI outright — it has no mechanism, at any gamma, beta, or epsilon, for a zero-weight cue to move its prediction off a constant; flatness at ~0.99 with no decline falsifies SDI. Pairs 6-7 run the mirror logic (junk endorsements on B): VWEI flat 0.99; SDI rises 0.53 -> 0.72 as B accumulates 'suspicion'. (2) OPPOSITE-SLOPE DOUBLE DISSOCIATION (pairs 10-14): cue 1 (95%) endorses A while the opposing coalition among cues 2-6 grows from 1 to 5. VWEI: E = 25.5 - cumulative weight, so P(A) DECLINES 0.95 -> 0.87 -> 0.81 -> 0.76 -> 0.74. SDI: each opposing endorsement adds a penalty to B, so P(A) RISES 0.51 -> 0.69 -> 0.82 -> 0.89 -> 0.92. The two profiles are anti-correlated and CROSS between k=2 and k=3 — slope signs are parameter-independent (VWEI's E strictly decreases in k because every added cue has positive weight; SDI's D strictly increases because every added endorsement adds positive penalty), so the sign of the empirical slope across this 5-point gradient adjudicates the theories with no free parameters able to reverse it. (3) SIGN CONFLICT AT CEILING (pairs 8-9, plus mirrors 24-25): A endorsed by 3 (then 5) high-validity experts, B by none. VWEI: P(A) ~ 0.99. SDI: P(A) = 0.12 (3 endorsements) falling to 0.07 (5 endorsements) — majority choices OPPOSITE to VWEI on stimuli where VWEI is at ceiling, which is SDI's novel falsifiable prediction (high-validity option carrying MORE endorsements). SDI additionally predicts a within-family decline (0.12 -> 0.07) where VWEI is flat at ceiling. (4) VALIDITY-BLINDNESS AT MATCHED ENDORSEMENT COUNTS (pairs 15-16): exactly one endorsement per option, varying expert identity (95% vs 70%; 90% vs 70%). SDI: P(A) = 0.51-0.52 regardless — its penalty weights span only ~9% across the entire validity range at delta <= 0.05, so at matched counts SDI is essentially indifferent to WHO endorses (its only tilt, +1-2 points toward the higher-validity endorser, comes from delta and the alpha tie-break). VWEI: 0.99 and 0.88 — a 40+ point sensitivity to expert identity that SDI cannot produce under any delta in [0, 0.05]. No pair is degenerate: every pair has discriminating cues, no pair leaves both theories guessing (the closest SDI gets to 0.5 is on D-1/E pairs where VWEI is at 0.88-0.95), and every pair requires actually comparing the two rating vectors. The conjunction is un-fakeable in both directions: a VWEI subject is a flat line at ~0.99 across the entire junk gradient and a declining line across the coalition gradient; an SDI subject is a specific descending curve on junk endorsements, a specific ascending curve on opposing coalitions, near-50% at matched counts, and B-majorities on the ceiling-conflict pairs. Neither theory's parameter box can approximate the other's profile because the key contrasts (zero vs. near-maximal weight on 50% experts; fewer- vs. more-endorsements preference) are structural, not parametric.

[5] NOVEL AXIS (vs. Exps 0-4): a CARDINAL WEIGHT-MATCHING RIDGE built from duplicate moderate validities, with no zero-validity experts and no dominant top cue. Exp 4 dissociated VWEI from SDI structurally (50% experts are invisible to VWEI, penalized by SDI) and via 95%-anchored ceiling conflicts. This design instead engineers trials where VWEI's evidence sum is arithmetically pinned to ~0 by matching coalitions of moderate experts against single stronger experts, and dual trials where SDI's penalty difference is pinned to ~0 by matching endorsement counts. The validities [0.90, 0.85, 0.80, 0.80, 0.75, 0.75, 0.75, 0.75, 0.65, 0.60] are chosen so that VWEI's amplified log-odds weights satisfy 2*w(0.75) = w(0.80) exactly at gamma = ln2/ln(1.2616) = 2.98 and 4*w(0.75) = w(0.85) exactly at gamma = ln4/ln(1.579) = 3.03 -- both crossing points sit at the CENTER of VWEI's own gamma box [2.8, 3.2], so the weight-matching holds (|E| <= 0.53, hence P within +/-2.5 points of 0.50) across the ENTIRE parameter box, verified at gamma in {2.8, 3.0, 3.2} x beta in {0.18, 0.23}. 12 base pairs, each exactly A/B-mirrored (24 unique pairs, K = 96//24 = 4, 96 trials, 8 samples per base cell per subject), perfectly side-balanced. FAMILY 1 -- EVIDENCE-MATCHED COUNT GRADIENT (pairs 1-5): one option carries 1 (or 2) endorsements, the other carries a weight-matched coalition of 2-4 moderate experts (e.g., A = {85%} vs B = {80%, 75%, 75%}: E = 5.22 - 5.32 = -0.10). VWEI is pinned at chance: P(fewer-endorsement option) = 0.49-0.50 on all five pairs (box range 0.48-0.52) -- it has no mechanism, at any gamma/beta/epsilon in its box, for these stimuli to move it off a guess, because its cardinal weights cancel by construction. SDI climbs monotonically with endorsement-count difference: 0.69 (1v2), 0.70 (1v2, second realization), 0.83 (1v3), 0.83 (2v4), 0.89 (1v4) (box ranges 0.65-0.73 up to 0.86-0.92), since every net endorsement adds ~0.96 penalty units and its D strictly increases with count difference -- the slope sign is parameter-independent. Any systematic departure from 50% toward the fewer-endorsement option on pairs 1-5 falsifies VWEI; flatness at 50% falsifies SDI. FAMILY 2 -- COUNT-MATCHED VALIDITY-GAP GRADIENT (pairs 6-9): exactly one endorsement per side, with the endorser validity gap shrinking (.90 vs .60; .85 vs .65; .85 vs .80; .80 vs .75). SDI is pinned at 0.50-0.52 on all four (its penalty weights (1-v)^delta span <5% across the whole validity range at delta <= 0.05, and the alpha tie-break adds at most 3 points) -- structurally near-blind to WHO endorses at matched counts. VWEI sweeps 0.89 -> 0.73 -> 0.62 -> 0.56 (box ranges 0.84-0.95 down to 0.55-0.59), a 33-point monotone gradient in endorser validity that no SDI parameterization can produce. FAMILY 3 -- SIGN CONFLICT AT THE RIDGE CREST (pair 10): four 75% experts (total weight 5.30) vs one 80% expert (weight 2.66). VWEI: majority for the 4-endorsement coalition, P(A) = 0.63 (box 0.61-0.65). SDI: majority for the single endorsement, P(A) = 0.10 (box 0.08-0.12) -- opposite majorities, both far from chance, sign-robust across both boxes. BASELINE/SUB-RIDGE (pairs 11-12): pair 11 ({90%} vs {65%, 60%}) is the one agreement cell (VWEI 0.89, SDI 0.71 -- both prefer A, calibrating each theory's scale where they concur); pair 12 ({80%, 75%} vs {85%}) sits just BELOW the matching ridge (VWEI 0.43, SDI 0.30 -- both prefer B but at parametrically distinct depths, gap >= 9 points across boxes). THE DOUBLE DISSOCIATION IS UN-FAKEABLE: the two theories are placed in orthogonal indifference regimes -- VWEI is indifferent exactly when endorsement counts differ at matched evidence (Family 1), SDI is indifferent exactly when counts match but validity composition differs (Family 2) -- and no reparameterization of either theory can escape its own indifference condition (VWEI's weights cancel by arithmetic fixed at gamma ~ 3; SDI's penalty weights are count-driven and validity-flat by delta <= 0.05). A VWEI subject is a flat 50% line across the entire count gradient and a steep monotone curve across the gap gradient; an SDI subject is a steep monotone climb across the count gradient and a flat ~51% line across the gap gradient, plus a strong majority for the single endorser on pair 10 where VWEI favors the coalition. All pairs are non-degenerate (every pair has discriminating cues; no all-tie pairs; no pair leaves both theories guessing), and every trial requires actually comparing the two rating vectors.

[6] NOVEL AXIS (vs. Exps 0-5): the first head-to-head between the two DEFECT-INTEGRATION theories. SDI and VPS agree that every endorsement is a liability, but they disagree about the WEIGHT PROFILE attached to the endorser's validity: SDI's skepticism weight (1-v)^delta is nearly FLAT across the whole validity range (0.86-0.98 at delta <= 0.05, a <14% spread; at delta=0 exactly flat), so SDI is essentially a validity-blind endorsement COUNTER; VPS's weight c0 + c1*v^kappa INCREASES steeply in validity (a 0.95 expert's endorsement costs ~2.9x a 0.50 expert's at mid parameters), so VPS actively prefers options endorsed by WEAKER experts. The design places the two theories in orthogonal indifference regimes: SDI is pinned at chance exactly when endorsement counts match but validity composition differs; VPS is pinned at (near) chance exactly when counts differ but its validity-scaled penalties cancel. Validities (15 experts): [0.95, 0.90, 0.90, 0.90, 0.85, 0.80, 0.65, 0.60, 0.60, 0.60, 0.55, 0.50, 0.50, 0.50, 0.50] -- duplicates are required to build coalitions; the spread 0.50-0.95 spans the full range. 12 base pairs, each exactly A/B-mirrored (24 unique pairs, K = 96//24 = 4, 96 trials, 8 samples per base cell per subject), perfectly side-balanced. All predictions below are given as P(A) at SDI mid-params (delta=.025, beta=1, eps=.115, alpha=.025) and VPS mid-params (kappa=2, beta=2, eps=.10, c0=.1, c1=1), with full-box ranges verified sign-robust. FAMILY 1 -- VALIDITY-GAP LADDER AT MATCHED COUNTS (pairs 1-5): 1v1 pairs with shrinking endorser gaps (.95/.50, .90/.55, .85/.60, .80/.65) plus the doubled 2v2 max-gap pair (.95+.90 vs .55+.50). VPS predicts a monotone ladder BELOW 50%: 0.24, 0.29, 0.34, 0.40, 0.13 -- it always prefers the LOWER-validity endorser, structurally (its penalty is strictly increasing in v for any c1>0, so D<0 at matched counts for EVERY parameterization). SDI is pinned at 0.51 +/- 0.02 (box 0.50-0.56): its penalty weights vary <14% and its only tilt, the alpha<=0.05 tie-break, pushes TOWARD the high-validity endorser -- i.e., AWAY from VPS's prediction, so no SDI parameterization can mimic the ladder. A monotone descending P(A) in gap falsifies SDI; a flat ~0.51 line falsifies VPS. FAMILY 2 -- SIGN CONFLICTS, OPPOSITE MAJORITIES (pairs 6-8): (a) 3x0.50 vs 2x0.90: VPS 0.79 (box 0.60-0.90) vs SDI 0.26 (box 0.24-0.30); (b) 4x0.50 vs 3x0.90: VPS 0.89 (box 0.75-0.93) vs SDI 0.25 (box 0.26-0.31); (c) 2x0.90 vs 3x0.60: VPS 0.31 (box 0.17-0.45) vs SDI 0.72 (box 0.68-0.75). Every conflict is sign-robust across BOTH entire parameter boxes (e.g., pair 6: VPS D = -c0 + c1(2*0.9^kappa - 3*0.5^kappa) > 0 always since c0/c1 <= 0.214 < 0.700; pair 8: D < 0 always since c0/c1 <= 0.214 < 0.365). A single majority choice on any conflict cell adjudicates the theories with no free parameters able to reverse it. FAMILY 3 -- JUNK-COUNT GRADIENT VS FIXED 2x0.90 (pairs 9, 10, 6): k = 1, 2, 3 endorsements from 50% experts on A against the same two 90% endorsements on B. SDI declines 0.66 -> 0.45 -> 0.26, CROSSING 50% between k=1 and k=2 (each junk endorsement adds ~0.97 penalty units; crossing at count parity is structural). VPS declines only 0.91 -> 0.86 -> 0.79 and does NOT cross through k=3, because each 0.50 endorsement costs only ~0.35 penalty units vs ~0.91 for a 0.90 endorsement -- it takes ~2.5 junk endorsements to offset one 90% endorsement for VPS but only 1 for SDI. The LOCATION OF THE CROSSING POINT is the quantitative signature. FAMILY 4 -- INDIFFERENCE ASYMMETRY (pairs 11-12): pair 11 (one 0.90 vs two 0.60s) is engineered so 2*0.6^kappa ~= 0.9^kappa (within 0.04-0.19 across kappa in [1.6, 2.4]), with the c0 count term partially cancelling the residual: VPS's D ranges only [-0.20, +0.20] over its ENTIRE box, keeping P(A) in [0.40, 0.61] (mid 0.50) -- near chance -- while SDI delivers a clear majority, P(A) = 0.70 (box 0.65-0.71), because B carries one extra endorsement. Pair 12 (3x0.90 vs 2x0.50) is the floor test: both theories prefer B, but SDI only moderately (P(A) = 0.35, box 0.28-0.42) while VPS is pinned at the floor (P(A) = 0.07, box 0.04-0.10) -- a 25+-point magnitude gap where sign cannot adjudicate. THE DISSOCIATION IS UN-FAKEABLE IN BOTH DIRECTIONS: a VPS subject is a descending ladder in Family 1 (all rungs < 0.47), stays above 0.60 through k=3 in Family 3, and sits at chance on pair 11; an SDI subject is a flat ~0.51 line across the entire ladder (its alpha tie-break can only push it ABOVE 0.5, opposite to VPS), crosses 50% at k=2 in Family 3, and shows a 0.70 majority on pair 11. Because SDI's stimulus-driven variation at matched counts is bounded by alpha*delta-logit <= 0.25 penalty units while VPS's ladder spans D in [-1.6, -0.41], and because VPS's indifference on pair 11 is arithmetic (2*0.6^kappa ~= 0.9^kappa across its kappa box) while SDI's count signal there is a full penalty unit, neither theory's parameter box can approximate the other's profile. All pairs are non-degenerate (every pair has discriminating endorsements, no all-tie pairs, no pair where both theories guess -- the closest, pair 11, is precisely the cell where they maximally diverge), and every trial requires actually comparing the two rating vectors.

[7] NOVEL AXIS (vs. Exps 0-6): MATCHED-COUNT SIGNAL AMPLIFICATION plus the full COUNT-VALIDITY EXCHANGE ECONOMICS. Exp 6 probed VPS vs SDI at unit matched count (1v1 validity ladder) and against one fixed high side (k x 0.50 vs 2x0.90). This design instead manipulates the COUNT dimension itself, exploiting a structural fact neither prior design used: at matched endorsement counts the two theories' decision variables have OPPOSITE SIGNS and IDENTICAL linear scaling in count. VPS: D = pen_B - pen_A = k*c1*(v_hi^kappa - v_lo^kappa) > 0 (the flat c0 term cancels EXACTLY, so matched-count trials are a pure assay of the confidence-distrust term c1*v^kappa), growing linearly in k, favoring the LOW-validity coalition ever more strongly. SDI: D = k*[(w_hi - w_lo) + alpha*Delta-logit] < 0 (its skepticism weight (1-v)^delta DECREASES in v, and the alpha tie-break also favors the high-validity side), also growing linearly in k, favoring the HIGH-validity coalition. Matched count k is therefore a signal AMPLIFIER for a sign conflict: at k=1 the theories differ by ~20 points; at k=4 they are at floor-vs-ceiling in opposite directions. Validity pool (20 experts): [0.95 x3, 0.90 x4, 0.85 x2, 0.75 x1, 0.65 x3, 0.60 x4, 0.55 x3] -- duplicates are required to build matched coalitions. 12 base pairs, each exactly A/B-mirrored (24 unique pairs, K = 96//24 = 4, 96 trials, 8 samples per base cell per subject), perfectly side-balanced. All predictions below: P(A) at VPS mid-params (kappa=2, beta=2, eps=.10, c0=.1, c1=1) and SDI mid-params (delta=.025, beta=1, eps=.115, alpha=.025), with full-box ranges [min,max] verified at all box corners; CRITICALLY, the VPS and SDI prediction intervals are DISJOINT on every single pair. FAMILY 1 -- MATCHED-COUNT GAP AMPLIFICATION (pairs 1-6): k x 0.60 vs k x 0.90 at k=1,2,3,4, and k x 0.55 vs k x 0.95 at k=2,3. VPS climbs toward ceiling: 0.69, 0.82, 0.89, 0.93 (gap .90/.60) and 0.87, 0.93 (gap .95/.55); box ranges [0.60,0.80], [0.68,0.92], [0.76,0.96], [0.81,0.96], [0.73,0.96], [0.81,0.96]. SDI stays pinned at/below chance, drifting slightly TOWARD the high-validity side as k grows: 0.48, 0.47, 0.45, 0.43 and 0.45, 0.42; box ranges [0.46,0.50] down to [0.34,0.50]. The signature is a sign x count interaction: VPS's slope in k is positive and large (+24 points over k=1..4 at mid params, and >= +21 points even between worst-case corners); SDI's slope is NEGATIVE or zero (its matched-count D is bounded in [-k*(0.064+2.74*alpha), 0], so P(A) <= 0.50 always and >= 0.34 at k=4). No SDI parameterization can produce P(A) > 0.5 on ANY matched-count pair (its validity sensitivity has the wrong sign and is bounded by 12% + alpha<=0.05), while no VPS parameterization can stay below 0.60 at k=4 (its D >= 4*0.7*0.403 = 1.13 across the entire box). FAMILY 2 -- EXCHANGE-RATE LOCUS (pairs 7-9): 2x0.55 vs 1 x v_hi with v_hi = 0.95, 0.85, 0.75. SDI is a validity-blind endorsement COUNTER here: the single endorser always carries fewer penalties, and its D varies by < 0.1 across the whole v_hi range, so SDI is FLAT at 0.28 / 0.29 / 0.29 (box [0.25,0.31]). VPS's preference tracks its internal exchange rate (v_hi/v_lo)^kappa ~ 2: it PREFERS two 55% endorsements over one 95% endorsement (0.59, box [0.49,0.74] -- a sign conflict with SDI on every parameterization), is indifferent at 0.85 (0.51, box [0.45,0.62]), and leans single at 0.75 (0.44, box [0.42,0.49]). VPS's D is strictly increasing in v_hi (slope sign parameter-independent), producing a monotone 15-point descending curve that crosses chance exactly where the exchange rate predicts; SDI's flat line cannot cross chance at any v_hi. FAMILY 3 -- VPS INDIFFERENCE RIDGE vs SDI COUNT GRADIENT (pairs 10-12): unmatched-count cells engineered so k*(c0 + c1*v_lo^kappa) ~= m*(c0 + c1*v_hi^kappa) at kappa~2 (the box center), pinning VPS at chance by arithmetic while SDI delivers a count-difference-graded majority for the fewer-endorsement option. Pair 10 (3x0.65 vs 2x0.85, count diff 1): VPS 0.54 (box [0.46,0.67]) vs SDI 0.29 (box [0.25,0.30]). Pair 11 (4x0.60 vs 2x0.90, count diff 2, since 2*0.9^2 = 4*0.6^2 - 0.18 with the c0 term closing the gap): VPS 0.49 (box [0.38,0.70]) vs SDI 0.16 (box [0.14,0.17]) -- SDI's majority deepens with count difference while VPS stays at chance, and even at VPS's most extreme box corner (0.38) it exceeds SDI's ceiling (0.17) by 21 points. Pair 12 (3x0.55 vs 2x0.85) is the fixed-count-ratio validity probe: same 3-vs-2 structure as pair 10 but with a larger coalition validity gap, VPS jumps from 0.54 to 0.68 (box [0.54,0.84], sign-robust) while SDI stays at 0.28 (box [0.25,0.31]) -- at fixed count difference, VPS's preference grows with the validity gap and SDI is structurally flat. UN-FAKEABILITY IN BOTH DIRECTIONS: (i) SDI cannot mimic Family 1 because its per-endorsement matched-count signal is bounded by |w_hi - w_lo| <= 0.064 plus alpha*Delta-logit <= 0.14, all of the WRONG SIGN (toward the high-validity side), so its P(A) is <= 0.50 on every matched-count pair while VPS's is >= 0.60 at k>=1 (>= 0.68 at k>=2); (ii) VPS cannot mimic Families 2-3 because on those cells its D is pinned within [-0.36, +0.39] (pair 11) and [-0.25, +0.48] (pair 7) by the arithmetic of k*v_lo^kappa ~= m*v_hi^kappa across its whole kappa box, keeping P(A) in [0.38,0.70] where SDI sits at 0.14-0.31 -- VPS has no parameterization that produces a 0.16 majority for the 2-endorsement option on pair 11, and SDI has none that produces a 0.59 majority for the 2-endorsement option on pair 7. The design also isolates VPS's two components: Family 1 cancels c0 exactly (pure c1*v^kappa assay), while Families 2-3 force c0 (the count term) to trade off against c1*v^kappa, with the ridge cells showing where they cancel. All 24 pairs are non-degenerate (every pair has discriminating endorsements, both options carry >= 1 endorsement and >= 1 non-endorsement, no all-tie pairs, no pair where both theories guess -- the closest cells, pairs 8 and 11, are precisely where the boxes are maximally separated), and every trial requires actually comparing the two rating vectors.

[8] NOVEL AXIS (vs Exps 0-7): the first head-to-head between the two CONFIDENCE-DISTRUST theories, DCD (advocated) and VPS (competing). Both treat endorsements as liabilities increasing in the endorser's validity, so count-conflict cells and high-validity cells largely produce AGREEMENT; the dissociation lives in the SHAPE of the distrust weight near chance validity plus DCD's residual trust term. Formally, DCD's effective per-endorsement score s(v) = c0 + c1*(2v-1)^kappa - alpha*logit(v) is NON-MONOTONE in v (at mid parameters s(0.55)=0.268 < s(0.50)=0.280 < s(0.60)=0.271... < s(0.65)=0.288: the LEAST suspicious endorser is a 55-60% expert, because the (2v-1)^kappa dampening collapses the distrust charge near chance while alpha*logit(v) actively rewards validity), whereas VPS's penalty p(v) = c0 + c1*v^kappa is STRICTLY increasing (a 0.60 expert's endorsement costs c1*(0.6^kappa - 0.5^kappa) = 0.07-0.17 units more than a coin-flip's; DCD's corresponding charge is <= 0.043 units and is OPPOSED by alpha*logit(0.60) >= 0.032). At MATCHED endorsement counts VPS's flat term c0 cancels exactly, so VPS sign-robustly prefers the LOWER-validity endorser on every matched-count pair across its entire box (P(A) < 0.5 always); DCD prefers the 0.55/0.60 endorser (sign-robustly ABOVE 0.5 for v=0.55; within +/-0.06 of 0.5 for v=0.60). The design therefore concentrates on matched-count composition among near-chance experts and avoids the moderate/high-validity count cells where the two distrust theories agree. Validities (20 experts): 0.50 x4, 0.55 x4, 0.60 x4, 0.65 x3, 0.70 x3, 0.90 x2. 16 base pairs, each exactly A/B-mirrored (32 unique pairs, K = 96//32 = 3, 96 trials, 6 samples per base cell per subject), perfectly side-balanced. Predictions are P(A) (A = higher-validity-composition side) at DCD mid (c0=.28, c1=.78, kappa=2, alpha=.10, beta=2.4, eps=.10) and VPS mid (c0=.10, c1=1.0, kappa=2, beta=2.0, eps=.10); box ranges include the full parameter boxes AND the lapse. FAMILY 1 -- k-GRADIENT AT 0.55 vs 0.50 (pairs 1-4, k=1..4): DCD 0.51, 0.51, 0.52, 0.53 (flat-rising, ABOVE 0.5, sign-robust: -c1*0.1^kappa + alpha*0.2007 >= +0.004 per endorsement); VPS 0.48, 0.45, 0.43, 0.41 (descending; per-unit D = -c1*(0.55^kappa - 0.5^kappa) in [-0.070, -0.034], remarkably stable across kappa). OPPOSITE-SIGNED SLOPES in k; boxes disjoint including lapse (k=4: DCD [0.51, 0.55] vs VPS [0.36, 0.45]). FAMILY 2 -- k-GRADIENT AT 0.60 vs 0.50 (pairs 5-8): DCD 0.51, 0.51, 0.52, 0.52 (flat; per-unit D in [-0.011, +0.027], bounded by the dampening and opposed by the trust term); VPS 0.45, 0.40, 0.36, 0.31 (diving; per-unit in [-0.146, -0.074]). At k=4: DCD [0.48, 0.56] vs VPS [0.24, 0.39] -- disjoint by 9+ points. FAMILY 3 -- VALIDITY LADDER AT k=3 (pairs 3, 7, 9, 10: 3x{0.55, 0.60, 0.65, 0.70} vs 3x0.50): VPS is a monotone steep descent 0.43, 0.36, 0.29, 0.22 (box ranges [0.39,0.46], [0.29,0.42], [0.21,0.37], [0.15,0.33]); DCD is HUMP-SHAPED 0.52, 0.52, 0.49, 0.44 (boxes [0.52,0.53], [0.50,0.55], [0.44,0.54], [0.38,0.51]), crossing 0.5 between v=0.60 and v=0.65 -- DCD's trust-distrust equilibrium validity (~0.63) that VPS structurally cannot exhibit. Max mid separation 22 points at v=0.70; all four rungs box-disjoint. FAMILY 4 -- ONE-STEP GAP CELLS (pairs 13, 11, 14, 12: {2x0.60, 3x0.60, 2x0.65, 3x0.65} vs {2x0.55, 3x0.55}): a single 0.05-validity step among near-chance experts moves VPS 0.45, 0.42, 0.39, 0.35 (boxes down to [0.28,0.41]) but leaves DCD blind at 0.50, 0.50, 0.48, 0.47 (all boxes within [0.43, 0.51]); all four disjoint. FAMILY 5 -- HIGH-VALIDITY ANCHORS (pairs 15, 16): pair 15 (1x0.90 vs 1x0.50): both avoid the 0.90 endorser but DCD is systematically softer (DCD 0.35, box [0.31, 0.39] vs VPS 0.27, box [0.21, 0.34]) -- DCD's softening of VPS's extremes. Pair 16 (1x0.90 vs 2x0.50) is the c0-MAGNITUDE assay: DCD's large flat charge exactly offsets its dampened 0.90 distrust (2*s(0.50) = 0.560 = s(0.90) at mid; P(A) = 0.50), while VPS's small c0 (2*0.35 = 0.70 << 0.91) leaves the validity-distrust gap exposed (P(A) = 0.41) -- a 10-point mid separation testing whether the count floor can rescue the high-validity endorser. UNFAKEABILITY IN BOTH DIRECTIONS: (i) no VPS parameterization can produce P(A) >= 0.5 on ANY matched-count cell (c0 cancels, c1 > 0, v^kappa strictly increasing), so the above-chance preference for the 0.55 endorser (Family 1) or even chance-level performance at k=4 where VPS sits at 0.24-0.31 (Family 2) falsifies VPS outright; (ii) no DCD parameterization can reproduce VPS's low-validity ladder: DCD's matched-count decision variable between 0.60 and 0.50 experts is bounded per endorsement in [-0.011, +0.027] (dampened distrust <= 0.043, opposed by alpha*logit >= 0.032), so at any k <= 4 it cannot leave [0.44, 0.57], while VPS's is <= -0.073 per endorsement (P <= 0.39 at k=4 including lapse); between 0.55 and 0.50 DCD is bounded BELOW by +0.004 per endorsement while VPS is bounded ABOVE by -0.034. The two theories are thus placed in orthogonal sensitivity regimes exactly at the validity range where their weight profiles diverge maximally: VPS is most sensitive precisely where DCD is dampened. 14 of 16 base pairs have fully disjoint predicted intervals across the ENTIRE parameter boxes including lapse; the two high-validity anchors (pairs 15, 16) are secondary magnitude cells (10-11-point mid separations, same direction). All 32 pairs are non-degenerate (every pair has discriminating endorsements, both options carry >= 1 endorsement and >= 1 non-endorsement, no all-tie pairs, no pair where both theories guess -- every cell has at least one theory off-chance by >= 4 mid-points, and the profile statistics use all cells), and every trial requires actually comparing the two rating vectors. Both theories are history-free (validities in instructions), so the K=3 repetition is harmless.

[9] NOVEL AXIS (vs. Exps 0-8): COALITION COMPOSITION at matched endorsement count with arithmetically CANCELED trust, plus the COIN-FLIP EXCHANGE LOCUS. Exp 8 dissociated VPS from DCD using matched-count gradients among near-chance experts (0.55-0.70 vs 0.50) and two 0.90 anchors. It never pitted a MIXED coalition {v_hi, 0.50} against a HOMOGENEOUS coalition {v_mid, v_mid}, and never mapped how many coin-flip endorsements balance mid-validity endorsements. Both are direct assays of the one structural difference between the two confidence-distrust theories: the GEOMETRY of the distrust charge. VPS's charge c0 + c1*v^kappa uses RAW validity, so a 0.50 endorsement still costs c1*0.25-0.33 distrust units; DCD's charge c0 + c1*(2v-1)^kappa is DAMPENED to the bare floor c0 at v=0.50, and its residual trust term alpha*logit(v) partially refunds validity. Validities (20 experts): [0.92, 0.90x3, 0.85, 0.77x2, 0.75x6, 0.72x2, 0.50x5]. 12 base pairs, each exactly A/B-mirrored (24 unique pairs, K = 96//24 = 4, 96 trials, 8 samples per base cell per subject), perfectly side-balanced. FAMILY 1 -- COMPOSITION SIGN CONFLICT (pairs 1-5): A = {v_hi, 0.50} vs B = {v_mid, v_mid} where v_mid is the LOGIT-MEAN of v_hi and 0.50, so DCD's trust term cancels by arithmetic identity: logit(0.90) = 2*logit(0.75) exactly (ln 9 = 2 ln 3; rungs 0.92/0.77 and 0.85/0.72 have residuals of only +0.003 and -0.015 at mid alpha). With trust gone, the pure charge shapes make OPPOSITE predictions, sign-robustly across the ENTIRE parameter boxes: VPS satisfies 2*v_mid^kappa > v_hi^kappa + 0.5^kappa for every kappa in [1.6, 2.4] at every rung (e.g. 0.90/0.75: 2*0.75^kappa - 0.9^kappa - 0.5^kappa in [+0.037, +0.087]), so the MIXED coalition is LESS suspicious and P(A) > 0.5 always (D >= 0.7*0.037 > 0). DCD satisfies (2v_hi-1)^kappa > 2*(2v_mid-1)^kappa for every kappa in [1.85, 2.15] (e.g. 0.8^kappa - 2*0.5^kappa in [0.122, 0.181]), so the mixed coalition is MORE suspicious and P(A) < 0.5 always (D <= -0.068 with trust residual <= +0.003). Mid-parameter predictions (VPS: kappa=2, beta=2, eps=.10, c0=.1, c1=1; DCD: kappa=2, beta=2.4, eps=.10, c0=.28, c1=.78, alpha=.10), with full-box ranges including lapse: pair 1 {90,50}v{75,75}: VPS 0.53 [0.51,0.57] vs DCD 0.44 [0.44,0.48] -- DISJOINT. Pair 2 (amplified x2): VPS 0.56 [0.52,0.62] vs DCD 0.39 [0.37,0.43] -- DISJOINT. Pair 3 (amplified x3): VPS 0.59 [0.52,0.68] vs DCD 0.33 [0.31,0.35] -- DISJOINT with an 18-point gap between interval edges. The amplification gradient is itself a signature: opposite-signed slopes in k (VPS rises 0.53->0.56->0.59; DCD falls 0.44->0.39->0.33), because both decision variables scale linearly in the number of replicated coalitions. Pairs 4-5 replicate the composition conflict at v_hi = 0.85 (vs 0.72) and 0.92 (vs 0.77): VPS 0.53 [0.51,0.55] and 0.54 [0.52,0.58] vs DCD 0.45 [0.44,0.47] and 0.45 [0.45,0.47] -- both disjoint, showing the effect spans the whole upper-validity range. FAMILY 2 -- EXCHANGE LOCUS (pairs 6-10): A = {75,75} (or {75x3}) vs B = k coin-flip endorsements, k = 2..5. VPS's internal exchange rate is (c0+c1*0.75^kappa)/(c0+0.25c1) ~ 1.9 coin-flips per 0.75 (it must 'pay' the substantial raw-v distrust of a 0.50 endorsement); DCD's is (c0+c1*0.5^kappa-alpha*1.0986)/c0 ~ 1.3 (a 0.50 endorsement costs only the floor, and trust refunds part of the 0.75 charge). The indifference point therefore differs by a full rung: VPS crosses 50% between k=3 and k=4 per two 0.75s; DCD crosses between k=2 and k=3. Pair 6 (k=2): VPS 0.25 [0.22,0.35] vs DCD 0.41 [0.38,0.48] -- disjoint magnitude conflict. Pair 7 (k=3): SIGN CONFLICT, VPS 0.38 [0.26,0.48] vs DCD 0.56 [0.48,0.64]. Pair 8 (k=4): VPS 0.53 [0.41,0.67] vs DCD 0.70 [0.60,0.76] -- DCD robustly prefers TWO 0.75 endorsements over FOUR coin-flips; VPS calls it near-chance. Pair 9 (k=5): agreement baseline (VPS 0.68, DCD 0.80). Pair 10 ({75x3} vs {50x5}, ratio 1.67 -- squarely between the two exchange rates): the strongest cell in the design, SIGN CONFLICT with VPS 0.40 [0.26,0.55] vs DCD 0.66 [0.53,0.71]; DCD's D > 0 by >= 0.076 at every box corner while VPS's mid D = -0.24. FAMILY 3 -- CALIBRATION (pairs 11-12): pair 11 ({90} vs {50}) both avoid the 0.90 endorser with DCD systematically softer (VPS 0.27 [0.20,0.34] vs DCD 0.35 [0.32,0.45]); pair 12 ({50,50} vs {90,90}) is a ceiling agreement cell (VPS 0.86, DCD 0.76) that scales each theory's slope where they concur. UNFAKEABILITY IN BOTH DIRECTIONS: (i) on all five composition cells, no DCD parameterization can reach P(A) = 0.5 (its D is negative by >= 0.068 with the trust residual bounded in [-0.019, +0.003]), and no VPS parameterization can fall to 0.5 (its D is positive by >= 0.026) -- the sign of the majority choice on any composition cell adjudicates the theories with no free parameters able to reverse it, and the k-amplification makes the P3 cell a 26-point mid separation; (ii) on the exchange family, the only VPS corner that reaches a positive decision variable on pair 10 (c0=0.15, c1=0.7, kappa=2.4) simultaneously predicts pair 6 at 0.34 (vs DCD 0.43) and pair 3 at 0.53 (vs DCD 0.31) -- a joint profile across cells that neither theory's box can mimic, since the composition conflict and the exchange conflict are two faces of the same structural difference (raw-v vs dampened charge) and no single reparameterization flips both. As the VPS advocate, the predicted profile is: majority for the MIXED coalition on all composition cells, growing with amplification (0.53 -> 0.59), and coin-flip avoidance that persists through k=3 (0.38) with an indifference point near k=4. A DCD subject instead shows majority-HOMOGENEOUS on all composition cells (falling to 0.33 at k=3), near-chance already at k=3 on the exchange gradient (0.56), and a robust preference for two 0.75s over four coin-flips (0.70). All 24 pairs are non-degenerate (every pair has discriminating endorsements, both options carry >= 1 endorsement and >= 1 non-endorsement, no all-tie pairs, no pair where both theories guess), and every trial requires actually comparing the two rating vectors. Both theories are history-free (validities in the instructions), so K=4 repetition is harmless.

[10] NOVEL AXIS (vs. Exps 0-9): the first head-to-head between SVD (advocated) and VPS (competing) -- two theories that share the SAME distrust core. SVD's skeptic branch is functionally identical to VPS (penalty c0 + c1*v^kappa per endorsement, D = pen_B - pen_A, sigmoid(beta*D), uniform lapse), and SVD's parameter box is essentially NESTED inside VPS's (c0 [0.05,0.12] within [0.05,0.15]; c1 [0.95,1.3] within [0.7,1.3]; kappa [1.5,1.8] within [1.6,2.4] up to a 0.1 sliver). Two consequences drive the design. (1) A box-robust SIGN conflict is provably impossible: any penalty difference that is decisively signed across VPS's box is also signed on the shared region of SVD's box, so both cores always point the same way. The dissociation must therefore be QUANTITATIVE -- in CHANGES across cells and in saturation LEVELS. (2) Everything that distinguishes the theories flows from SVD's counter mixture (probability 1-w in [0.08,0.20]: a deterministic fewer-endorsement tally that is EXACTLY indifferent at matched counts) plus SVD's wider lapse. The design's keystone is a manipulation that is orthogonal to everything VPS computes: COUNT PARITY at fixed, saturated penalty evidence. Validities (20 experts): 0.95 x7, 0.75 x4, 0.50 x9. 12 base pairs, each exactly A/B-mirrored (24 unique pairs, K = 96//24 = 4, 96 trials, 8 samples per base cell per subject), perfectly side-balanced. Mid parameters used below: SVD (c0=.085, c1=1.125, kappa=1.65, beta=1.9, w=.86, eps=.18); VPS (c0=.10, c1=1.0, kappa=2, beta=2, eps=.10); box ranges verified at box corners. FAMILY 1 -- SATURATED COUNT-PARITY JUMP (pairs 1-4, primary): pair 1 is A = {50x7} vs B = {95x7}, a matched-count cell with a huge penalty gap (D = 7*c1*(0.95^k - 0.5^k) >= 2.90 on VPS's box, >= 3.81 on SVD's). Pair 2 adds ONE coin-flip endorsement to B (counts 7v8); pair 3 adds ONE coin-flip to A (8v7); pair 4 adds TWO to A (9v7). The added endorsements are 0.50 experts, whose penalty contribution (0.18-0.58 units) is trivial against the saturated D (3-6 units), so VPS's decision variable is essentially untouched: VPS's predicted P(A) is FLAT across all four pairs -- box-verified changes: Delta(pair2 - pair1) in [0, +0.06] points, Delta(pair3 - pair1) in [-0.7, 0], Delta(pair4 - pair1) in [-1.9, 0] -- while its LEVEL on pair 1 is [0.927, 0.965] (worst corner c1=.7, kappa=1.6, beta=1.6, eps=.13 still gives 0.927). SVD behaves completely differently: on pair 1 (matched counts) its counter branch contributes exactly 0.5, capping P(A) at (1-eps)(1+w)/2 + eps/2 <= 0.905 (box [0.798, 0.905], mid 0.853) -- DISJOINT from VPS's [0.927, 0.965]; the moment counts become unmatched, the counter branch flips to a deterministic vote: pair 2 (counter favors A) JUMPS to [0.876, 0.940] (mid 0.910), i.e., Delta in [+3.3, +8.8] points; pair 3 (counter favors B) DROPS to [0.717, 0.869] (mid 0.795), Delta in [-8.0, -3.5]; pair 4 drops further to [0.708, 0.869], Delta in [-8.9, -4.4]. THE CHANGES ARE BOX-DISJOINT: VPS moves at most 1.9 points where SVD moves at least 3.3 (pair 2) and at least 3.5 (pairs 3-4). A count-parity effect of >= 2 points at saturation falsifies VPS outright (no beta, kappa, c0, c1, or epsilon in its box can produce it); a flat profile (changes <= 2 points) falsifies SVD outright. The pair-1 LEVEL alone is nearly decisive: P(A) >= 0.92 falsifies SVD; <= 0.91 falsifies VPS. SVD additionally predicts the counter branch is a STEP at parity, not a gradient: pairs 3 and 4 are predicted nearly equal (mid 0.795 both). FAMILY 2 -- MATCHED-COUNT CEILING LADDER (pairs 10-12 + pair 1): k x 0.50 vs k x {0.75 or 0.95} at k = 2, 2, 4, 7. Both theories prefer the coin-flip side, but VPS climbs to its ceiling (mid: 0.750, 0.889, 0.945, 0.950; box at k=7: [0.927, 0.965]) while SVD is compressed toward 0.5 by the factor (1-eps)*w ~ 0.65 and plateaus (mid: 0.702, 0.803, 0.849, 0.853; box at k=7: [0.798, 0.905]). The asymptote LEVEL (>= 0.92 vs <= 0.91) and the ~10-point mid gap at every saturated rung is the signature; SVD's between-subject dispersion on the asymptote (wide eps in [0.12,0.24] and w in [0.80,0.92]) is a secondary heterogeneity signature VPS cannot produce (its eps box [0.07,0.13] spans 3 points of asymptote, SVD's spans 11). FAMILY 3 -- COUNTER BUFFERING ON CONFLICT CELLS (pairs 5-7): A = {95 x j} vs B = {50 x 2j}, j = 1, 2, 3. Here the skeptic core of BOTH theories prefers B (the coin-flip side: 2 coin-flips cost less suspicion than one 0.95 endorsement, sign-robustly: VPS's D = j*[c0 + c1*(2*0.5^k - 0.95^k)] < 0 at every box corner), but SVD's counter branch votes for A (fewer endorsements), dragging SVD back toward chance. VPS dives: mid P(A) = 0.368, 0.256, 0.176 (boxes [0.21,0.49], [0.09,0.48], [0.05,0.47]) -- a deepening B-majority, B-rate 0.63 -> 0.74 -> 0.82 at mid. SVD hovers: mid P(A) = 0.481, 0.411, 0.352 (boxes [0.44,0.51], [0.35,0.49], [0.30,0.47]) -- near chance at j=1 and never below 0.30. A B-rate >= 0.60 on pair 5 falsifies SVD; a B-rate <= 0.52 falsifies VPS; at j=3 the mid separation is 17.6 points. FAMILY 4 -- EXCHANGE-RIDGE CELLS (pairs 8-9): A = {75 x 2} vs B = {50 x 4}, and its x2 amplification A = {75 x 4} vs B = {50 x 8}. These sit exactly at the coin-flip/0.75 exchange ridge where VPS's penalty difference is pinned near zero (VPS mid D = +0.075 and +0.15, P(A) = 0.533 and 0.567; box [0.42,0.69] and [0.33,0.81] -- near chance), because 4 coin-flips cost roughly 2 x 0.75 in suspicion. SVD's exchange rate is slightly lower AND its counter branch votes for the 2-endorsement (or 4-vs-8) side, so SVD delivers a clear A-majority: mid 0.634 and 0.687 (boxes [0.59,0.72], [0.60,0.84]). SVD's P(A) rises with the amplification (+5 mid points, robust: SVD's D > 0 at every box corner since c0 + c1*(2*0.5^k - 0.75^k) >= 0.034 > 0 for kappa in [1.5,1.8], so doubling the coalition strictly increases its skeptic probability while the counter vote stays fixed). UNFAKEABILITY IN BOTH DIRECTIONS (joint profile): because the boxes are nested, single-cell mimicry is possible only at extreme corners, but the corners that let VPS imitate SVD on one family force it away on the others. The VPS corner that comes closest to SVD on Family 3 (c0=.15, c1=.7, kappa=1.6, the flattest corner, P(A)=0.489 on pair 5) simultaneously predicts P(A) = 0.927 on pair 1 with a count-jump of ~0 -- while SVD's box requires P(A) <= 0.905 AND a jump >= +3.3; conversely, the SVD corner that maximizes pair 1 (eps=.12, w=.92: 0.905, jump +3.5) predicts pair 5 at 0.51, far above VPS's ceiling there (0.49), and pair 7 at 0.47 vs VPS's 0.47 only at its flattest corner where pair 1 sits at 0.93. The conjunction {pair-1 level, count-jump size, Family-3 depth, ladder asymptote} is a profile neither box can approximate: VPS is a flat line at 0.93-0.97 across Family 1 with deep B-majorities in Family 3; SVD is a V-shaped step function on Family 1 (0.85 -> 0.91 -> 0.80 -> 0.80 at mid) with near-chance performance in Family 3 and an 0.85 asymptote. All 24 pairs are non-degenerate (every pair has discriminating endorsements, both options carry >= 1 endorsement and >= 11 non-endorsements, no all-tie pairs, and no pair leaves both theories guessing -- the closest cells, pairs 5 and 8, are exactly where the boxes maximally separate at mid), and every trial requires actually comparing the two rating vectors. Both theories are history-free (validities in the instructions), so K=4 repetition is harmless.

[11] NOVEL AXIS (vs. Exps 0-10): PERFECT EXPERTS (v = 1.0) combined with MINIMAL COUNT ASYMMETRY (+/-1, +/-2 endorsements). Exp 10, the only prior SVD-vs-VPS design, used 0.95/0.75/0.50 experts and 2:1 count ratios; its conflict cells left VPS's weak box corners unsaturated (its Family-3 VPS intervals reached up to [0.05, 0.47]), so VPS could mimic SVD's levels at extreme corners. The v = 1.0 expert closes that escape route STRUCTURALLY: its distrust charge is c0 + c1*1^kappa = c0 + c1 -- the maximum possible and KAPPA-INVARIANT -- so no parameterization of either theory can discount it, while the coin-flip expert's charge c0 + c1*0.5^kappa (in [0.18, 0.58] on VPS's box) is the cheapest suspicion unit; the charge ratio (c0+c1)/(c0+c1*0.5^kappa) is >= 2.23 at every VPS corner. On a 6-perfect-vs-7-coin cell VPS's decision variable is D = c0 + c1*(7*0.5^kappa - 6) <= -2.43 at EVERY corner of its box: VPS is pinned at its floor (or ceiling) by arithmetic, not by favorable parameters. Meanwhile a SINGLE extra coin-flip endorsement -- worth at most 0.58 penalty units against a 2-6 unit saturated gap -- flips SVD's deterministic counter branch from voting one way, to indifferent, to voting the other way. Validities (15 experts): 1.0 x 6, 0.50 x 9. 12 base stimuli, each exactly A/B-mirrored (24 unique pairs, K = 96//24 = 4, 96 trials, 8 samples per base cell per subject), perfectly side-balanced. Mid parameters quoted: VPS (kappa=2, beta=2, eps=.10, c0=.10, c1=1.0; charges: perfect 1.10, coin 0.35); SVD (c0=.085, c1=1.125, kappa=1.65, beta=1.9, w=.86, eps=.18; charges: perfect 1.21, coin 0.443); box ranges verified at box corners. FAMILY 1 -- SATURATED FLOOR (pairs 1-3): A carries j perfect endorsements (j=4,6) vs B's j+1 (pair 3: j+2) coin-flips. VPS: P(A) in [0.050, 0.138] (pair 1), [0.050, 0.082] (pair 2), [0.050, 0.096] (pair 3) -- near-certain avoidance of the perfect-expert side at every corner. SVD: its skeptic branch is equally saturated toward B, but its counter branch (prob 1-w >= 0.08) deterministically votes A (fewer endorsements): P(A) = (1-eps)(w*s + (1-w)) + eps/2 >= 0.130 on all three (boxes [0.130, 0.294], [0.130, 0.285], [0.130, 0.293]; mid 0.210, 0.205, 0.205). DISJOINT on pairs 2-3 (VPS max 0.082/0.096 < SVD min 0.130); pair 1 is a near-disjoint gradient rung (15.6 mid points, 8-point box overlap). An A-rate >= 0.12 on pairs 2-3 falsifies VPS outright; <= 0.11 falsifies SVD outright. This floor-level pinning is what Exp 10's 0.95-expert cells could not achieve. FAMILY 2 -- SATURATED CEILING, COUNTER-OPPOSED (pairs 14-15, mirrors of 2-3): A = 7 (or 8) coin endorsements vs B = 6 perfect. Both cores saturate toward A; SVD's counter votes B. VPS: P(A) >= 0.919 (pair 14), >= 0.904 (pair 15), up to 0.951. SVD: P(A) = (1-eps)*w*s + eps/2 <= 0.870 (boxes [0.715, 0.870], [0.710, 0.870]; mid 0.795 both). DISJOINT: an A-rate >= 0.91 falsifies SVD; <= 0.88 falsifies VPS. Unlike Exp 10's matched-count ceiling cells (SVD cap 0.905), the counter branch here actively drags SVD 3-8 points further from VPS. FAMILY 3 -- THE COUNT-PARITY STAIRCASE (pairs 2 -> 4 -> 5 floor side; 14 -> 16 -> 17 ceiling side): the perfect-expert side's endorsement count climbs 6 -> 7 -> 8 (one added coin-flip expert per rung) against a FIXED 7-coin opponent, so the skeptic evidence stays saturated (|D| >= 2.05 at every corner of both boxes) while count parity passes through fewer -> matched -> more. SVD is a STAIRCASE: mid P(A) = 0.205 -> 0.147 -> 0.090 (floor side) and 0.795 -> 0.853 -> 0.910 (ceiling side), with a step of exactly (1-w)(1-eps)/2 in [3.0, 8.8] points at EVERY parity crossing -- the step size is a direct read on SVD's mixture and lapse parameters. VPS is FLAT: a coin endorsement shifts D by at most 0.58 units against |D| >= 2.05, so P(A) moves <= 0.7 points box-wide (floor side [0.050,0.082] -> [0.050,0.075] -> [0.050,0.071]; ceiling side [0.919,0.951] -> [0.926,0.951] -> [0.929,0.951]). The change intervals are DISJOINT at every step (SVD >= 3.0 points where VPS <= 0.7). Any single coin-flip endorsement moving choice by >= 2 points on these saturated cells falsifies VPS (no beta, kappa, c0, c1, or epsilon in its box can produce it); flatness (<= 2 points) falsifies SVD outright. Pair 4's LEVEL is additionally disjoint by itself: VPS [0.050, 0.075] vs SVD [0.150, 0.156] -- SVD's matched-count indifference cap (1-eps)(1+w)/2 + eps/2 keeps it 7.5+ points above VPS's floor even though its skeptic branch, like VPS's, is fully saturated. FAMILY 4 -- THE EXCHANGE LOCUS (pairs 7-10): one perfect endorsement vs m = 1, 2, 3, 4 coin-flips. VPS mid: 0.214, 0.329, 0.478, 0.631 -- crossing 50% between m=3 and m=4 (internal exchange rate (c0+c1)/(c0+c1*0.5^kappa) = 3.14 at mid, in [2.23, 4.56] across the box). SVD mid: 0.280, 0.452, 0.598, 0.730 -- crossing between m=2 and m=3 (skeptic rate 2.73 at mid, in [2.59, 2.88], PLUS the counter branch voting for the single-endorsement side at m >= 2, pulling the crossing a full rung earlier). SIGN CONFLICT at m=3 (pair 8): VPS 0.478 (slight B-majority) vs SVD 0.598 (clear A-majority), 12 mid points; at m=2 (pair 7) VPS is at a clear B-majority (0.329) where SVD sits near chance (0.452). Boxes overlap here (VPS's exchange-rate range is wide), so this family is a SECONDARY crossing-rung signature: the observed indifference rung is m=3-4 for VPS vs m=2-3 for SVD, and the joint profile adjudicates corner ambiguity -- the VPS corner that crosses earliest (c0=0.15, c1=0.7, kappa=1.6, giving P(pair 8) = 0.598) is still pinned flat on the staircase and <= 0.138 on every floor cell, so a subject showing SVD's crossing rung TOGETHER WITH any staircase step >= 2 points or any floor-cell A-rate >= 0.12 lies outside VPS's entire box. FAMILY 5 -- MIXED-COALITION EXCHANGE (pairs 11-12): two perfect endorsements vs 5 (then 6) coin-flips: VPS 0.310, 0.455 vs SVD 0.489, 0.636 -- 18-point mid gaps in the same direction (SVD's counter votes the 2-endorsement side), secondary magnitude cells. FAMILY 6 -- MATCHED-COUNT CEILING BASELINE (pairs 6, 18): 6 perfect vs 6 coin, counts matched. VPS [0.926, 0.951] vs SVD [0.841, 0.905] (pair 6; mirror flipped) -- disjoint; the v=1.0 sharpening of Exp 10's pair-1 level assay (VPS's matched-count D = 6*c1*(1-0.5^kappa) >= 2.81 across its box). UNFAKEABILITY IN BOTH DIRECTIONS: (i) No VPS parameterization can produce the staircase: on staircase cells |D| >= 2.05 at every VPS corner while a coin endorsement is worth <= 0.58 units, bounding VPS's per-rung change at 0.7 points vs SVD's structural step (1-w)(1-eps)/2 >= 3.0. (ii) No VPS parameterization can reach SVD's floor levels on pairs 2-3 (box max 0.082/0.096 < SVD box min 0.130) or pair 4 (0.075 < 0.150). (iii) No SVD parameterization can reach VPS's ceiling on pairs 14-15 (counter-opposed cap (1-eps)w + eps/2 <= 0.870 < VPS min 0.904) or the matched-count cells 6/16/18 (indifference cap <= 0.905 < VPS min 0.926). (iv) The parameters trade off oppositely: SVD's most VPS-like corner (eps=0.12, w=0.92) simultaneously gives floor cells at 0.130 (still 3.4+ points above VPS's pairs-2/3 ceiling of 0.096), ceiling cells at 0.870 (still below VPS's 0.904 floor), and staircase steps of 3.5 points (still 2.8 points above VPS's 0.7 maximum) -- every SVD corner is disjoint from every VPS corner on at least two families, and the conjunction {floor level, ceiling level, staircase step, exchange crossing} is a profile neither box can approximate: a VPS subject is a flat line at 0.05/0.95 across the staircase with a 3-4-rung exchange crossing; an SVD subject is a 0.13-0.29 floor, a 0.71-0.87 counter-opposed ceiling, a 3-9-point staircase whose step size estimates (1-w)(1-eps)/2, and a 2-3-rung exchange crossing. All 24 pairs are non-degenerate (every pair has discriminating endorsements, both options carry >= 1 endorsement and >= 7 non-endorsements, no all-tie pairs, and no pair leaves both theories guessing -- the closest cells, pairs 5 and 17, are exactly the staircase rungs where the step signature lives), and every trial requires actually comparing the two rating vectors. Both theories are history-free (validities in the instructions), so K=4 repetition is harmless.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "NOVEL AXIS (vs. Exps 0-11): the first head-to-head between SVD (advocated) and GDRD-B (competing) \u2014 two theories that share the SAME distrust core (penalty c0 + c1*v^kappa, sigmoid comparison, lapse) AND, by construction, the same population-MEAN route weight (~0.86): GDRD-B's bimodal quantile transform explicitly holds E[w] at SVD's base value. A box-robust sign conflict is therefore provably impossible (the skeptic cores are near-identical and their boxes overlap), and pooled-mean gaps on any single cell are bounded at ~3 points. The dissociation must be fought on three quantitative fronts, all exploited here: (1) the SATURATION ASYMPTOTE LEVEL (a pure lapse assay: on cells where both branches saturate, P = 1 - E[eps]/2 \u2014 SVD .910 with every subject in [.874,.940]; GDRD-B .938 with NO subject below .899 and ~65% above .940); (2) a POOLED-MEAN DOUBLE CROSSING (GDRD-B is ABOVE SVD on saturated cells by its lower lapse, but BELOW SVD on single-vs-many exchange cells by its higher w and graded \u2014 not step \u2014 count route: an interaction no reparameterization can mimic, since eps and w push in opposite directions across the two families); (3) the PER-SUBJECT DISTRIBUTION (GDRD-B's 12% counter minority with w in [.50,.62] lands 8-20 points OUTSIDE SVD's box on count-conflict cells, producing the bimodality/super-binomial variance GDRD-B claims \u2014 turned here into its falsifiable liability: SVD bounds EVERY subject's conflict-cell rate >= .69 because w >= .80 and eps >= .12). VALIDITY ENGINEERING: 6 perfect experts (v=1.0) give the maximum, KAPPA-INVARIANT charge c0+c1 (closing every corner-escape route, as in Exp 11) and a capped logit (3) that bounds GDRD-B's micro-trust term at 0.15/endorsement vs a >= 0.6 per-pair charge gap \u2014 verified: saturation survives every corner of BOTH boxes including alpha=0.05; 8 coin experts (v=0.50) give the cheapest charge (c0+c1*0.5^k in [0.39,0.57]) and ZERO logit (alpha-inert); 2 experts at v=0.65 have a coin-charge ratio of 1.36-1.54 across BOTH entire boxes, making the 2-vs-3 penalty-matched cell residual |D| <= 0.14 (skeptic in [.44,.61]) by arithmetic. 8 base pairs, each exactly A/B-mirrored (16 unique pairs, K = 96//16 = 6, 96 trials, 12 samples per base cell per subject), perfectly side-balanced. Mid parameters quoted: SVD (c0=.085, c1=1.125, kappa=1.65, beta=1.9, w=.86, eps=.18); GDRD-B majority (c0=.095, c1=1.175, kappa=1.85, beta=1.9, w=.90, eps=.13, gamma=2.25, alpha=.025) and minority (w=.56, eps=.075, gamma=2.5); box ranges verified at box corners. FAMILY 1 \u2014 SATURATED CONFLICT / COUNTER-TAIL CELLS (pairs 1-2, primary): A = 8 (then 7) coin endorsements vs B = 6 (then 5) perfect endorsements. Skeptic evidence is saturated toward A at every corner of both boxes (|D| >= 2.16; SVD s_A >= .93, GDRD s_A >= .90 even after the worst alpha pull), while the count route saturates toward B (A carries MORE endorsements). SVD: P(A) = (1-eps)*w*s_A + eps/2, every subject in [.685,.870] (population ~.79). GDRD-B: skeptic majority [.72,.92], but the 12% counter minority lands in [.46,.62] \u2014 DISJOINT from SVD's box by >= 7 points on both pairs. FALSIFICATION: any subject with a pooled conflict-cell rate < .65 (24 trials) is outside SVD's entire box (w >= .80, eps >= .12, s >= .93 make it unreachable); conversely, ZERO such subjects in N >= 40 (P = .88^40 = 0.7% under GDRD-B) rejects the bimodal population outright. GDRD-B also predicts 2.2x super-binomial between-subject variance here (SD ~.10 vs SVD's ~.045). FAMILY 2 \u2014 ASYMPTOTE / LAPSE CELLS (pairs 3-4): A = 1 (then 2) coin endorsement vs B = 4 (then 5) perfect endorsements; both branches saturate toward A (|D| >= 3.6 at every corner), so P(A) = 1 - eps/2. SVD: every subject in [.874,.940], population .910. GDRD-B: skeptics >= .899, counters >= .937, population .938. A pooled asymptote <= .92 favors SVD; >= .925 favors GDRD-B (with 48 trials/subject and N >= 20 subjects the 2.8-point gap is > 4 SE). Additionally, > 40% of subjects with individual asymptote rates above .94 falsifies SVD (its eps >= .12 caps every subject at .940); a substantial fraction of subjects below .89 falsifies GDRD-B (its lowest possible subject is .899). FAMILY 3 \u2014 PARITY CAP CELL (pair 5): A = 5 perfects vs B = 5 coins; skeptic saturates toward B (the cheap side), counts matched so the count branch sits at exactly 0.5. P(B) = (1-eps)(1+w)/2 + eps/2: SVD [.79,.91] (pop .85); GDRD-B counters [.71,.80] \u2014 a second, independent location where the minority mode falls below SVD's floor. FAMILY 4 \u2014 PENALTY-MATCHED COUNT-STEP CELL (pair 6): A = 2x0.65 vs B = 3x0.50; the charge ratio pins the skeptic to s in [.44,.61] across both boxes while the count route votes A (fewer endorsements). SVD is squeezed into a remarkably tight window [.52,.65] (pop ~.585) because its eps and w effects offset; GDRD-B's counters rise to [.60,.77] (pop .567) \u2014 above SVD's ceiling \u2014 and its alpha term lets skeptics drift to .65 at extreme corners where SVD cannot exceed .61 in the opposite direction. FAMILY 5 \u2014 EXCHANGE CELLS (pairs 7-8): A = 1 perfect vs B = 2 (then 4) coins; these trace each theory's sigmoid for beta calibration and complete the mean double-crossing: SVD population P(A) = .45 and .72 vs GDRD-B .42 and .69 (GDRD-B's higher w and graded count route make it systematically LESS favorable to the single-endorsement side, ~3 points at both rungs; box-verified direction is stable at mid parameters, with corner overlap acknowledged \u2014 these are secondary cells). THE JOINT PROFILE IS UN-FAKEABLE: because GDRD-B holds E[w] at SVD's value, no single-cell reparameterization separates them \u2014 but the CONJUNCTION {asymptote level .91 vs .94, conflict-cell floor .69 vs a 12% mode at .49-.62, parity-cap floor .79 vs .71-.80, penalty-matched ceiling .65 vs .60-.77, exchange means .45/.72 vs .42/.69} is a profile neither box can approximate in the other's direction: SVD's eps >= .12 and w <= .92 bound every subject's asymptote <= .940 and conflict rate >= .685, while GDRD-B's skeptic eps <= .18 and counter w <= .62 bound its asymptote >= .899 and its minority's conflict rate <= .62; the two theories' per-subject envelopes are disjoint on Families 1-3 simultaneously, and the parameters trade off oppositely across families (the GDRD-B corner that best mimics SVD's asymptote, eps=.18, simultaneously deepens its conflict-cell divergence). POWER NOTE: the pooled-mean assays (Families 1-2) need only ~20 subjects; the distributional tail/bimodality assays need N >= 40-50 (detecting a 12% mode is inherently a large-N test \u2014 the design concentrates 24 conflict trials and 48 asymptote trials per subject to make per-subject rates estimable). PREDICTED OUTCOMES: an SVD population shows asymptote ~.91, zero subjects above .94, all conflict rates in [.69,.87], unimodal between-subject spread; a GDRD-B population shows asymptote ~.94, ~2/3 of subjects above .94, a distinct ~12% cluster at .49-.62 on conflict cells (and .60-.77 on pair 6, .44-.71 on pair 7), and super-binomial conflict-cell variance. All 16 pairs are non-degenerate (every pair has discriminating endorsements, both options carry >= 1 endorsement and >= 1 non-endorsement, no all-tie pairs, no pair where both theories guess \u2014 the closest cells, pairs 6-7, are exactly where the minority mode separates), and every trial requires actually comparing the two rating vectors. Both theories are history-free (validities in the instructions), so K=6 repetition is harmless.",
  "validities": [
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    0.65,
    0.65,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5
  ],
  "trial_a_ratings": [
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
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
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      0,
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
      0,
      0,
      0,
      0,
      0,
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
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      1,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
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
      0,
      0,
      0,
      0,
      0,
      1,
      1,
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
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      1,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
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
      0,
      0,
      0,
      0,
      0,
      1,
      1,
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
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
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
      0,
      0,
      0,
      1,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
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
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      0,
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
      0,
      0,
      0,
      0,
      0,
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
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 24176,
  "output_tokens": 57942,
  "reasoning_tokens": 54550,
  "provider": "Sail Research"
}
```
