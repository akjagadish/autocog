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
**Description:** Graded Sparse-Preference with Heterogeneous Attenuation (GSP), unnormalized-evidence variant. Subjects integrate ALL cues additively with near-flat validity weighting under an anti-endorsement (sparse-preferring) reading: the evidence for option A over B is E = sum_j v_j^gamma (b_j - a_j), where each endorsement (rating = 1) carried by an option counts AGAINST it, so the sparser option accumulates evidence. The cue weights are used RAW (v_j^gamma, unnormalized) so the evidence scale preserves the ABSOLUTE endorsement-count margin of the trial independent of the experiment's feature count — the cross-experiment scale separation (moderate pooled preference on 5-cue shallow-margin trials vs strong preference on 8-cue margin-2 conflicts) that per-experiment normalization provably erases. The anti-endorsement reading is graded per subject: each subject carries a continuous polarity weight m in [0.75, 1.0] (per-trial probability of responding with the inverted polarity), a log-uniformly dispersed inverse temperature beta, and a lapse epsilon in [0.02, 0.12]: p(A) = (1-eps)*[m*sigmoid(beta*E) + (1-m)*(1-sigmoid(beta*E))] + eps/2, with identical rating vectors yielding exactly 50%. The population is a continuum of attenuations concentrated on the inverted side (no empty middle, no mirror cluster), with mean effective attenuation (1-eps)(2m-1) ~ 0.70 matching the observed extremity anchors.

**Parameters:**
- gamma: [0, 1.5]
- log_beta: [0.095, 1.163]
- epsilon: [0.02, 0.12]
- m: [0.75, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Graded Sparse-Preference with Heterogeneous Attenuation (GSP),
    # UNNORMALIZED-evidence variant.
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are 0/1 cue values. History is ignored
    # (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"GSP expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    gamma = float(parameters["gamma"])
    # log-uniform inverse temperature: beta = exp(log_beta),
    # log_beta ~ U[ln 1.1, ln 3.2]  =>  beta log-uniform over [1.1, 3.2],
    # calibrated to the RAW (unnormalized) evidence scale below.
    # Lower bound raised from 0.8 to 1.1 (the interpolation lever);
    # upper bound capped at 3.2 (guardrail for Exp4/Exp6 overshoot).
    beta = float(np.exp(float(parameters["log_beta"])))
    epsilon = float(parameters["epsilon"])
    m = float(parameters["m"])

    # RAW validity weighting: w_j = v_j^gamma, NO normalization by
    # sum(w). This preserves the absolute endorsement-margin scale:
    # the evidence magnitude reflects the raw count margin of the
    # trial (a margin-2 conflict carries twice the evidence of a
    # margin-1 conflict) independent of n_features.
    w = np.power(val, gamma)

    a, b = stim[0], stim[1]
    # Anti-endorsement (sparse-preferring) evidence for A over B:
    # each endorsement carried by B counts FOR A, each endorsement
    # carried by A counts AGAINST A. The sparser option accumulates
    # evidence. E = 0 on identical rating vectors.
    E = float(np.sum(w * (b - a)))

    # Numerically stable logistic of the inverted evidence.
    # (both branches exponentiate a non-positive argument, so no
    # overflow is possible)
    x = beta * E
    if x >= 0.0:
        sig = 1.0 / (1.0 + np.exp(-x))
    else:
        ex = np.exp(x)
        sig = ex / (1.0 + ex)

    # Graded polarity: with per-trial probability m the subject responds
    # with the INVERTED polarity (sparse-preferring), with probability
    # (1 - m) with the conventional polarity. This is the expectation of
    # a per-trial Bernoulli polarity draw against a continuous,
    # strongly-inverted per-subject weight m in [0.75, 1.0].
    p_core = m * sig + (1.0 - m) * (1.0 - sig)

    # Independent lapse to uniform choice. Ties (E = 0) give exactly 50%.
    p_a = (1.0 - epsilon) * p_core + 0.5 * epsilon

    return np.array([p_a, 1.0 - p_a])
```

**`policy source code`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # Guard against float drift.
    return np.random.choice(len(probs), p=probs)
```


## COMPETING THEORY
**Description:** Raw-Evidence Sparse Dominance with Validity-Gated Precision (RSD-vgp). When choosing between two products described by binary expert ratings, subjects read an endorsement (rating = 1) as evidence AGAINST the option carrying it, so the count-sparser option is perceived as higher quality. Within a trial, ALL cues are integrated additively with RAW (unnormalized) validity weights: the evidence for A over B is E = sum_j v_j^gamma (b_j - a_j), so the absolute endorsement-count margin sets the evidence scale independent of feature count, and identical vectors give exactly 50%. The population is a continuum of graded polarity (no bimodal trait clusters): each subject carries a per-trial inverted-polarity probability m, a lapse epsilon, and — the new discriminative dimension — validity-gated precision: a per-subject validity exponent gamma dispersed over [0, 2] with a substantial steep-gamma subpopulation, and a steep log-uniformly dispersed raw inverse temperature beta over [1.8, 6.0], calibrated so that essentially every subject's effective normalized psychometric slope beta * sum(v^gamma) exceeds 9.5. The resulting subject family is 'steep slope x moderate asymptote': p(A) = (1-eps)*[m*sigmoid(beta*E) + (1-m)*(1-sigmoid(beta*E))] + eps/2, with effective attenuation A_eff = (1-eps)(2m-1) concentrated near 0.69. This decomposition is what the data jointly demand: the slope-sensitive diagnostics (psychometric slope posterior, evidence-usage slopes, mixture log-likelihood ratios) require steep normalized slopes, while the extremity/consistency diagnostics (saturated sparse-choice rates) require moderate asymptotic preference — a combination no single shallow-slope, high-extremity or steep-slope, high-extremity parameterization can produce.

**Parameters:**
- gamma_u: [0, 1]
- log_beta: [0.588, 1.792]
- epsilon: [0.03, 0.11]
- m: [0.74, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Raw-Evidence Sparse Dominance with Validity-Gated Precision (RSD-vgp).
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are 0/1 cue values. History is ignored
    # (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"RSD-vgp expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    # ---- Validity gating: per-subject gamma via a two-component mixture ----
    # gamma_u ~ U[0, 1]. With probability 0.65 the subject draws gamma
    # uniformly from [0, 1.3] (near-flat validity weighting); with
    # probability 0.35 from [1.3, 2.0] (the steep-gating subpopulation
    # that validity-conflict cells discriminate).
    u = float(parameters["gamma_u"])
    if u < 0.65:
        gamma = 1.3 * (u / 0.65)
    else:
        gamma = 1.3 + 0.7 * ((u - 0.65) / 0.35)

    # ---- Precision: log-uniform raw inverse temperature ----
    # beta = exp(log_beta), log_beta ~ U[ln 1.8, ln 6.0]  =>  beta
    # log-uniform over [1.8, 6.0]. Calibrated against the RAW evidence
    # scale below so that every subject's effective normalized slope
    # beta * sum(v^gamma) comfortably exceeds 9.5 on any plausible
    # validity vector (min ~ 1.8 * 7.4 ~ 13).
    beta = float(np.exp(float(parameters["log_beta"])))
    epsilon = float(parameters["epsilon"])
    m = float(parameters["m"])

    # RAW validity weighting: w_j = v_j^gamma, NO normalization.
    # This preserves the absolute endorsement-margin scale: a margin-2
    # conflict carries twice the evidence of a margin-1 conflict,
    # independent of n_features (the cross-experiment scale separation
    # that per-experiment normalization provably erases).
    w = np.power(val, gamma)

    a, b = stim[0], stim[1]
    # Anti-endorsement (sparse-preferring) evidence for A over B:
    # each endorsement carried by B counts FOR A, each endorsement
    # carried by A counts AGAINST A. E = 0 on identical rating vectors.
    E = float(np.sum(w * (b - a)))

    # Numerically stable logistic (both branches exponentiate a
    # non-positive argument, so no overflow is possible).
    x = beta * E
    if x >= 0.0:
        sig = 1.0 / (1.0 + np.exp(-x))
    else:
        ex = np.exp(x)
        sig = ex / (1.0 + ex)

    # Graded polarity: with per-trial probability m the subject responds
    # with the INVERTED (sparse-preferring) polarity, with probability
    # (1 - m) with the conventional polarity. Continuous, no empty
    # middle, no mirror cluster.
    p_core = m * sig + (1.0 - m) * (1.0 - sig)

    # Independent lapse to uniform choice. Ties (E = 0) give exactly 50%.
    p_a = (1.0 - epsilon) * p_core + 0.5 * epsilon

    return np.array([p_a, 1.0 - p_a])
```

**`policy source code`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # Guard against float drift.
    return np.random.choice(len(probs), p=probs)
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] The design exploits two orthogonal dissociations between Take-The-Best (TTB) and Tallying. (1) SIGN dissociation with graded tally margins: On conflict trials, the top-validity cue favors one option while the majority of cues favors the other. Critically, TTB's predicted choice probability is a constant sigmoid(beta) on every such trial (it only ever uses the single top discriminating cue), whereas Tallying's confidence scales with the tally margin: sigmoid(beta) for 2-vs-1, sigmoid(2*beta) for 3-vs-1, sigmoid(3*beta) for 4-vs-1. So TTB predicts FLAT confidence across trials 1/3/5/13 (margins 3, 1, 2, 1 respectively) while Tallying predicts systematically increasing confidence in the majority option — a quantitative, trial-level signature that model recovery cannot confuse. Mirrored versions (trials 2/4/6/14) control for side bias. (2) TALLY-TIE dissociation: Trials 7/8 have a 2-2 tally with one tied cue, so Tallying must guess (exactly 50%) while TTB is confident (sigmoid(beta)) because the top cue discriminates. Agreement/control trials (9-12, 15-16) anchor both models and check that subjects are not simply counting or simply using cue 1. Validities are spread (0.95 down to 0.55) so the validity ordering is unambiguous and communicated up front; the second and third cues are placed far enough below the top cue that TTB's cascade is well defined, and low-validity cues (0.65, 0.60, 0.55) are close together so Tallying treats them as a coherent majority block. With 16 unique pairs and K=6 repetitions (96 trials), each dissociating cell gets enough repetitions for reliable per-cell choice proportions. Predicted pattern: if TTB is correct, choice proportions on trials 1, 3, 5, 13 should be statistically indistinguishable from each other (all ~sigmoid(beta) favoring the top-cue option) and well above 50% on trials 7/8; if Tallying is correct, choices on conflict trials should track the tally margin (increasingly favoring the majority option from trial 3 -> 5 -> 1) and trials 7/8 should sit at exactly 50%.

[1] This design dissociates Tallying from Take-The-Best (TTB) with a 3x3 factorial grid plus controls, using 8 cues with strictly descending, well-separated validities (.95 down to .60) so TTB's cascade order is unambiguous. GRID: In the 9 core cells (G1-G9), the first cue that discriminates is cue d in {1, 3, 5} (cues above d are tied), and that cue always favors B, so TTB always picks B with p = sigmoid(beta) -- a completely FLAT prediction across the entire grid, independent of both the tally and the discriminating cue's validity. Orthogonally, the total tally margin M in {+2, 0, -2} (from A's perspective) is varied by arranging the cues below d. Tallying predicts p(B) = sigmoid(-M*beta): at M = -2 both theories pick B but with different confidence (Tallying sigmoid(2*beta) vs TTB sigmoid(beta) -- an agreement-direction CONFIDENCE dissociation); at M = 0 Tallying must guess at exactly 50% while TTB is confident at sigmoid(beta); at M = +2 the theories conflict in direction (Tallying strongly favors A at sigmoid(2*beta), TTB favors B at sigmoid(beta)). Thus the signature is a 2D response surface: TTB predicts a flat plane (no effect of margin, no effect of cue depth), Tallying predicts a monotone gradient in margin only, crossing exactly 50% at M = 0. The depth dimension additionally tests whether confidence depends on the discriminating cue's validity (.95 vs .85 vs .75) -- both theories predict no effect, which rules out validity-weighted alternatives and verifies TTB's order-based (not magnitude-based) cascade; if choices track cue validity both target theories fail. Mirrored versions (swapping A and B) control for side bias and must be analyzed by 'cue-winner side' rather than option label. EXTREME CONFLICT (T2/T2m): cue 1 alone favors one option against all 7 other cues -- TTB stays at sigmoid(beta) while Tallying is near-certain (sigmoid(6*beta)), the largest possible confidence gap. ANCHORS: T1/T1m (all cues agree) check that subjects follow the evidence at all; T3/T4 (identical vectors) verify that both models' 50% guessing prediction holds and catch key-bias artifacts. 24 unique pairs x K = 4 repetitions = 96 trials exactly, giving 8 observations per grid cell (4 per unmirrored cell + 4 mirrored) for stable per-cell choice proportions. Predicted outcomes: if Tallying is correct, p(cue-winner) declines monotonically with margin (high at M=-2, exactly 50% at M=0, low at M=+2) with no effect of depth; if TTB is correct, p(cue-winner) is statistically indistinguishable across all 18 grid trials and both extreme-conflict trials, at a common sigmoid(beta) < 1.

[2] This design targets the specific operating regime PWEI claims for humans (gamma in [35,50], i.e., a steep but finite power transform) and opposes it to TTB's two structural invariants: (i) constant confidence sigmoid(beta) on EVERY trial where any cue discriminates, regardless of which cue, its validity, or what lower cues say, and (ii) 50% only when all cues tie. The validity ladder is deliberately TIGHT at the top (0.90, 0.89, 0.88, 0.87) with a long tail (0.70, 0.60). At any gamma in [35,50], the three sub-top cues' combined power-weight EXCEEDS the top cue's weight (e.g., at gamma=40: 0.89^40+0.88^40+0.87^40 = 0.0193 > 0.90^40 = 0.0148), so PWEI must REVERSE against the top cue on the flagship 1-vs-3 trials (T2/T2m), predicting p(top-cue option) between 0.25 (gamma=35) and 0.45 (gamma=50) — never above 0.5 — while TTB predicts the top-cue option at its full constant confidence p*. This direction conflict is sign-robust across PWEI's entire parameter box. Second, trials T2, T3, T4, T5 all have cue 1 (0.90) as the first discriminating cue favoring the same option, but with graded weighted evidence D (gamma=40: -0.13, -0.02, +0.16, +0.43). TTB predicts a FLAT p* across all four; PWEI predicts a monotone psychometric sweep from ~0.31 to ~0.88 — a ~0.6-wide confidence gradient on trials TTB treats as identical. Third, depth invariance: on T6/T6m cues 1-4 tie and cue 5 (validity only 0.70) decides; TTB predicts the SAME constant p* as on cue-1 trials, while PWEI's weight for 0.70^gamma is ~1e-5 of the total, forcing ~50% choices. T8/T9 probe depth 2: TTB stays at p*, PWEI is at ~50% (T8, near-canceling block) or 0.82 (T9). T1/T1m anchor both models near ceiling; T10 (identical vectors) verifies both models' 50% guessing prediction and catches key bias. Mirrored versions of every dissociating trial control side bias; analysis should pool by cue-winner. With 16 unique pairs and K=6 (96 trials exactly), each cell yields 12 observations per subject (6 + 6 mirrored). Predicted outcomes: if TTB is correct, choice proportions favoring the first-discriminating-cue winner are statistically indistinguishable across ALL of T1-T9 and their mirrors (one common p*), including the 1-vs-3 reversal trials and the 0.70-validity depth trials, with T10 at 50%; if PWEI is correct, confidence tracks the weighted evidence margin D: below 50% for the top cue on T2, near 50% on T3/T6/T8, and graded from ~0.7 to ~0.9 on T4/T5/T9 — a pattern TTB's two-level step function cannot produce, and conversely a flat p* pattern that PWEI's steep-gamma regime cannot produce because the tight ladder forces it to vote against the top cue.

[3] This design attacks TTB's central structural invariant — a SINGLE constant choice probability p* = (1-eps)*sigmoid(beta) + eps/2 on EVERY trial where any cue discriminates — using a validity architecture opposite to previously explored tight-top-ladder designs: one DOMINANT top cue (0.95, whose power-weight exceeds the entire rest of the ladder combined at every gamma in [35,50]: 0.95^40=0.1286 vs 0.0886 for cues 2-6) followed by a CONTESTED second tier (0.92 vs a 0.91/0.90/0.89/0.88 block whose combined power-weight exceeds cue 2's weight at every gamma in [35,50], e.g., at gamma=40: 0.0303 vs 0.0228). Consequence 1 (direction dissociation at depth 2): on L4-type trials cue 1 is tied, cue 2 (first discriminator) favors one option, and all four lower cues favor the other. TTB picks the cue-2 winner at p*; PWEI REVERSES, predicting p(cue-2 winner) between 0.33 (gamma=35) and 0.45 (gamma=50) — never above 0.5, sign-robust across PWEI's entire parameter box. Consequence 2 (graded confidence ladder): trials L0/L1/L2/L3/L4 hold cue 1 tied and cue 2 as first discriminator while the opposing block grows from 0 to 4 cues. PWEI predicts a monotone sweep of p(cue-2 winner) of roughly 0.73 / 0.59 / 0.48 / 0.42 / 0.37 (gamma=40, beta=8, eps=0.2), crossing 50% between L1 and L2 — a psychometric function of the weighted evidence margin D that TTB's two-level step function (p* or 50%) cannot produce at any (beta, eps). Consequence 3 (depth gradient on 'clean' trials): T1 (cue 1 decides, rest tied), L0 (cue 2 decides, rest tied), and U2 (cue 3 decides, rest tied) are identical to TTB — first discriminator, no other signal — so TTB predicts one common p*; PWEI predicts declining confidence 0.89 > 0.73 > 0.66 because the deciding cue's share of total weight shrinks with depth. Consequence 4 (agreement anchors): T3 (all cues agree, PWEI ~0.90) and T2 (cue 1 against all five others, PWEI 0.61-0.78, direction agreeing with TTB) show that even TTB's best single p* must simultaneously miss ~0.90 (T3), ~0.73 (L0), ~0.37 (L4) — no constant fits. Controls: C1/C2 (identical vectors) verify both models' 50% guessing and catch key bias; every dissociating trial has a mirrored twin (A/B swapped) to cancel side bias, with analysis pooled by cue-winner side. 24 unique pairs x K=4 = exactly 96 trials, giving 8 observations per cell per subject (4 + 4 mirrored). Decisive quantitative signatures: (i) Spearman correlation between cell choice proportion and PWEI's weighted margin D (strongly positive if PWEI true, ~0 if TTB true); (ii) L4 significantly below 50% for the first-discriminator winner; (iii) monotone L-ladder crossing 50%; (iv) T1 > L0 > U2 depth ordering. TTB predicts all discriminating cells statistically indistinguishable at one common p* and cannot approximate any of these patterns even with its full beta in [0.1,20], epsilon in [0,0.5] range, because its per-trial prediction is parameterized by a single constant independent of cue depth, cue validity, and lower-cue configuration.

[4] This design adversarially dissociates Take-The-Best (TTB) from Sparse-Option Preference / Polarity-Inverted Evidence Integration (SPI) by exploiting their two structural differences: (i) TTB uses ONLY the single highest-validity discriminating cue with conventional polarity (1 = good), producing one constant choice probability p* = (1-eps)*sigmoid(beta) + eps/2 on EVERY trial where any cue discriminates; (ii) SPI integrates ALL cues with near-uniform weights and INVERTED polarity, preferring the option with FEWER 1s, producing a graded psychometric function of the weighted sparsity evidence E = sum_j w_j*(b_j - a_j). Validities are [0.95, 0.85, 0.75, 0.70, 0.65, 0.60]: well-separated and descending so TTB's cascade order is unambiguous, yet close enough that SPI's sign of E is robust across its entire gamma in [0, 2] (verified at gamma = 0, 1, 2). DISSOCIATION 1 (direction, sign-robust across BOTH parameter boxes): Trial T6 presents unanimous evidence (A = all 1s, B = all 0s). TTB must pick the unanimously endorsed option with p >= 0.51 at ANY (beta in [0.1,20], eps in [0,0.5]) because sigmoid(beta) >= 0.5. SPI must pick the all-zeros option: E = -1, and even at its worst case (rho = 0.35, eps = 0.35, beta = 0.5) p(endorsed option) <= 0.476 < 0.5. The same sign-robust conflict holds on T0, T1, T7, T10, where the first-discriminating-cue winner carries more 1s (E = -w1, -w3, -w2, all < 0 for any gamma since the deciding cue always has the largest weight among consulted cues). DISSOCIATION 2 (quantitative sweep TTB cannot fit): Trials T0, T1, T2, T3, T4, T4b, T5 hold cue 1 (0.95) as the first discriminating cue favoring the SAME option, while the lower-cue configuration sweeps SPI's inverted evidence monotonically: E(T0) = -(w1+w2+w3) in [-0.63, -0.50], E(T1) = -w1 in [-0.26, -0.17], E(T2) = -w1+w2 in [-0.05, 0], E(T3) = -w1+w2+w3 in [+0.11, +0.17], E(T4) in [+0.25, +0.32], E(T4b) in [+0.37, +0.49], E(T5) = 1-2*w1 in [+0.48, +0.67] — all sign-robust across gamma in [0,2]. TTB predicts ONE constant p* on every rung (cue 1 is always the sole basis); SPI predicts a monotone psychometric sweep crossing exactly 50% between T2 and T3 (e.g., at gamma=1, beta=6, eps=0.2, rho=0.1: approximately 0.20, 0.32, 0.48, 0.69, 0.80, 0.85, 0.88 for the cue-1 winner). No single constant can fit a ~0.7-wide sweep that straddles 50%, and conversely SPI cannot produce a flat confident p* because its output is strictly monotone in E. DISSOCIATION 3 (near-tie cell): on T2, SPI's evidence is in [-0.05, 0], forcing p ~ 50% (exactly 50% at gamma=0), while TTB is confident at p* > 0.51 — a confidence dissociation at exactly the point where SPI's evidence cancels. DISSOCIATION 4 (depth invariance): T7 (cue 3 decides) and T10 (cue 2 decides) are, to TTB, identical to T1 (same constant p*, only the discriminating cue's identity changes); SPI predicts slightly weaker anti-winner evidence (|E| = w2 or w3 < w1), a subtle ordering TTB forbids. CONTROLS: T9/T9b are identical-vector trials where both theories predict exactly 50% (key-bias check); every dissociating trial has a mirrored twin (A/B swapped) so side bias cancels, with analysis pooled by first-discriminating-cue winner. Decisive quantitative signatures: (a) p(unanimously endorsed option) < 0.5 on T6/T6m if SPI is true, > 0.5 if TTB is true — sign-robust over both full parameter boxes; (b) Spearman correlation between cell choice proportion and SPI's evidence E: strongly positive if SPI is true, ~0 if TTB is true; (c) monotone ladder T0 < T1 < T2 ~ 0.5 < T3 < T4 < T4b < T5 vs. TTB's flat equality of all ten discriminating cells; (d) T2 at ~50% while T1/T3 are confident. 24 unique pairs x K = 4 repetitions = exactly 96 trials, giving 8 observations per subject per pooled cell (4 unmirrored + 4 mirrored).

[5] ADVERSARIAL DESIGN: SPI (advocated) vs TTB (competing), built on a two-tier validity architecture with 8 cues: a dominant pair (0.95, 0.92) and a tight near-uniform low block of six (0.62 down to 0.52). This architecture is chosen so that (a) TTB's cascade is unambiguous (cue 1, then cue 2, then the low block), and (b) SPI's near-uniform weighting (gamma in [0,2]) makes the low block act as a coherent 'counting block': at gamma=0 SPI's evidence is exactly the 1s-count difference (n_B - n_A)/8; even at gamma=2 the whole low block (sum weight 0.528) outweighs either top cue, so block-mass sign conclusions are robust across SPI's entire parameter box. TTB NEVER consults the low block whenever cue 1 or cue 2 discriminates — so the low block is invisible to TTB and decisive for SPI. The design sweeps the low-block mass from the TTB winner's side to the loser's side while holding the TTB decision constant, generating a wide monotone psychometric that TTB's single constant p* cannot fit at any (beta, epsilon).

CORE DISSOCIATION 1 (TTB-flat vs SPI-sweep): Trials 1-6 (L1, L4, L5, L7, L8, L11) all have cue 1 (0.95) as the first and only-relevant discriminator favoring A, so TTB predicts ONE constant p(A) = (1-eps)*sigmoid(beta)+eps/2 on all of them. SPI's evidence for A sweeps monotonically: E = -0.875/-0.77 (L1), -0.25/-0.35 (L4), -0.125/-0.24 (L5), 0/-0.04 (L7, exact cancellation at gamma=0), +0.125/+0.05 (L8), +0.625/+0.28 (L11) (values at gamma=0/gamma=2). At gamma=1, beta=6, eps=0.2, rho=0.1, SPI predicts p(TTB winner) of roughly 0.19, 0.27, 0.29, 0.55, 0.66, 0.81 — a ~0.6-wide monotone sweep crossing 50% between L5 and L8, with L7 sitting near 50% (cancellation cell: SPI's weighted evidence nearly vanishes while TTB is at full p*). No constant fits a sweep this wide that straddles 50%.

CORE DISSOCIATION 2 (sign-robust direction conflicts): On X2 (trial 11: unanimous endorsement of A, B all zeros) TTB must pick A with p >= 0.51 at ANY (beta in [0.1,20], eps in [0,0.5]); SPI must pick B: E = -1, and even at its worst case (rho=0.35, eps=0.35, beta=0.5) p(A) <= 0.486 < 0.5. The same sign-robust conflict holds on L1 (E in [-0.875, -0.77]) and on the headline depth-2 cell Y1 (trial 9: cue 1 tied, cue 2 plus four low cues all endorse B, A empty — TTB winner B at p*, SPI p(TTB winner) <= 0.49 across its whole box because E in [+0.60, +0.63] makes empty A the sparse option). These cells dissociate the two theories in DIRECTION with sign robustness over both full parameter boxes.

CORE DISSOCIATION 3 (depth): D1 and D3 (trials 7-8) replicate L5 and L8 with cue 2 (0.92) as first discriminator. TTB predicts EXACTLY the same p* as the depth-1 twins (depth invariance); SPI predicts nearly identical evidence (E differs only by w1-w2), so depth cells double as replication while testing TTB's structural claim that cue identity is irrelevant. Y1 extends depth to a 5-cue mass against the cue-2 winner — the largest direction conflict available at depth 2.

ANCHORS: X1 (trial 10: cue 1 endorses A, all seven other cues endorse B) is an extreme AGREEMENT cell — both theories pick A — but with graded SPI confidence (E in [+0.51, +0.75]); it verifies that SPI subjects are not simply anti-endorsement robots, since here the sparser option coincides with the top-cue option. Controls C1/C2 (trials 23-24: identical vectors, all-0 and all-1) check both theories' exact-50% guessing prediction and catch key bias. Every dissociating trial has a mirrored twin (A/B swapped, trials 12-22) so side bias cancels; analysis pools by TTB-winner side. 24 unique pairs x K=4 = exactly 96 trials, 8 observations per pooled cell per subject.

DECISIVE QUANTITATIVE SIGNATURES: (a) Spearman correlation between cell choice proportion and SPI evidence E: strongly positive if SPI true, ~0 if TTB true; (b) p(TTB winner) < 0.5 on X2, L1, Y1 and their mirrors (sign-robust over both boxes) vs > 0.5 under TTB; (c) monotone ladder L1 < L4 < L5 < ~0.5 at L7 < L8 < L11 vs TTB's flat equality of all discriminating cells at one common p*; (d) L7 near 50% while L1/L11 are extreme — a confidence pattern TTB's two-level step function cannot produce, and conversely a flat p* that SPI's strictly monotone evidence-to-choice map cannot produce. This design differs from prior SPI-vs-TTB work by using the two-tier dominant-pair + counting-block architecture, the depth-2 mass-conflict cell Y1, and a cancellation cell placed exactly where SPI's weighted evidence vanishes.

[6] RFI-delta and SPI are nearly observationally equivalent at the pooled/trial level: both compute p = (1-eps)*[(1-m)*sigmoid(beta*E) + m*(1-sigmoid(beta*E))] + eps/2 over the same near-flat weighted sparsity evidence E, and SPI's parameter box (gamma in [0,2], beta in [0.5,20], rho in [0,0.35], eps in [0.05,0.35]) CONTAINS the shape any single RFI subject can produce. No pooled choice proportion can separate them. The ONE structural difference is the population architecture: RFI-delta has a stable per-subject trait with P(s=-1)=pi~0.94 plus a small per-trial slip delta<=0.12, so it predicts a BIMODAL subject population -- a dominant anti-endorsement cluster whose per-subject p(sparse option) on extreme trials is tightly boxed in [0.82, 0.92] (slip and lapse cap extremity: max = (1-eps)*[(1-delta)*sigma(beta)+delta*(1-sigma(beta))]+eps/2 <= ~0.92) and a ~6% conventional-minority cluster at p(sparse) ~ [0.08, 0.18] -- with an empirically EMPTY middle band (0.2, 0.8). SPI's box mathematically FORBIDS any subject below ~0.52 on a unanimous trial: min over eps=0.35, rho=0.35, beta=0.5 of (1-eps)*[(1-rho)*sigma(beta*E)+rho*(1-sigma(beta*E))]+eps/2 = 0.524 at E=1; more generally SPI's p_core >= 0.5 whenever E >= 0 (since rho <= 0.35 < 0.5), so NO SPI subject can prefer the endorsed side on ANY non-negative-evidence cell. Therefore the design must make per-subject p(sparse) precisely estimable: 3 extreme cells (E in [0.85, 1.0]) plus their mirrors give 6 pairs x K=6 = 36 high-|E| trials per subject (SE ~ 0.05), enough to classify each subject into majority (>0.8), minority (<0.2), or forbidden-middle bands. DESIGN (8 features, validities [0.95, 0.88, 0.78, 0.70, 0.62, 0.55, 0.52, 0.50], well spread, sum 5.50): (1) UNANIMOUS ladder T1-T6 sweeps weighted anti-endorsement evidence E for the all-zeros (sparse) option from 1.00 (T1: 8-vs-0) down through 0.89/0.85 (T2/T3: 7-vs-0 skipping the lowest vs highest cue -- a gamma probe: flat weighting (gamma=0) predicts T2 = T3 exactly, any gamma>0 predicts T2 > T3), 0.67 (T4: 5-vs-0), 0.42 (T5: 3-vs-0), to 0.15 (T6: 1-vs-0 on the top cue). RFI majority (mid-box: gamma=0.5, beta=5, delta=0.10, eps=0.04) predicts p(sparse) ~ 0.88, 0.87, 0.87, 0.86, 0.79, 0.61 -- a graded monotone psychometric; the RFI MINORITY shows the full MIRRORED ladder (~0.12 ... 0.39), which SPI cannot generate for any subject on the top rungs. (2) GAMMA cell T7: A endorsed only on cue 1 (0.95), B endorsed on cues 6-8 (0.55/0.52/0.50). Count says A is sparser (1 vs 3 ones); RFI's box (gamma <= 1) gives E in [+0.11, +0.25], so RFI sign-robustly predicts p(A) in [0.56, 0.79] -- never below 0.5. SPI at gamma=2 flips the weighted sign (0.9025 > 0.8229) and can prefer B; if observed p(A) < 0.5, RFI is falsified while steep-gamma SPI survives -- an accepted adversarial risk that tests RFI's flat-weighting commitment. (3) Tie controls T8/T16 (identical vectors) verify both theories' exact-50% prediction and catch key bias. Mirrors T9-T15 cancel side bias; analysis pools by sparse-option side. DECISIVE QUANTITATIVE SIGNATURES: (a) subject-level bimodality on the 36 extreme trials: dominant mode in [0.82, 0.92], small second mode in [0.08, 0.18], empty band 0.2-0.8 (RFI) vs a unimodal/continuous population with a hard floor at ~0.52 and NO subject below 0.5 (SPI); even ONE subject with per-subject p(sparse) < 0.5 on positive-E cells falsifies SPI, while RFI requires ~pi (~6%) of subjects there -- hence N >= 40-60 subjects (expect 2-4 conventional subjects in 50). (b) minority subjects show a complete mirrored psychometric (p(sparse) rising toward 0.5 as E shrinks: ~0.12 on T1, ~0.39 on T6, ~0.46 on T7), impossible under SPI where p_core >= 0.5 for all E >= 0. (c) RFI's extremity cap: NO majority subject may exceed ~0.92 on T1 (delta >= 0.08, eps <= 0.08); SPI permits up to 0.975, so a population of subjects at 0.95+ falsifies RFI. (d) monotone ladder T1 >= T2 > T3 > T4 > T5 > T6 > T7 with T2-T3 as a flat-vs-steep weighting probe. Both theories are thus sharply falsifiable, but only RFI predicts the two-attenuated-clusters signature, and only SPI is compatible with a homogeneous middle -- the population distribution over per-subject extreme-cell proportions is the arbiter.

[7] ADVOCATED: SPI. COMPETITOR: RFI-delta. The two theories are nearly observationally equivalent on any single pooled cell: both compute p(sparse) = (1-eps)*[(1-m)*sigmoid(beta*E) + m*(1-sigmoid(beta*E))] + eps/2 over the same near-flat weighted sparsity evidence E. The dissociation therefore lives entirely in their PARAMETER-BOX and POPULATION-ARCHITECTURE constraints, and the design is built to make those box boundaries empirically decisive. (1) RFI's per-subject box is narrow: beta in [3,8], epsilon in [0,0.08], delta in [0.08,0.12], gamma in [0,1], with a fixed per-subject polarity trait and a bimodal population (94-95% anti-endorsement majority, ~5% conventional minority). SPI's per-subject box is wide: beta in [0.5,20], epsilon in [0.05,0.35], rho in [0,0.35], gamma in [0,2], continuous population. Consequences exploited: (A) EXTREMITY FLOOR: on E~1 (unanimous) cells, an RFI majority subject must show p(sparse) >= 0.816 (min at beta=3, delta=0.12, eps=0.08) and <= ~0.92 (delta+eps cap); the RFI minority sits at <= 0.19; the band (0.2, 0.8) is EMPTY at the subject level, and RFI's POOLED p(sparse) on extreme cells is boxed in [0.78, 0.88]. SPI's central regime (e.g., gamma=0.5, beta=4, eps=0.20, rho=0.15) predicts p(sparse) ~= 0.77 on E=1 -- below RFI's per-subject floor and pooled floor -- and SPI's box places most subjects in 0.55-0.80, exactly RFI's forbidden middle. (B) WITHIN-SUBJECT SLOPE FLOOR: because RFI mandates beta >= 3, every RFI subject must show a rise of at least ~0.19 from the low-E cell (E~0.10-0.125) to the unanimous cell (E=1) (min rise ~0.19 at beta=8, delta=0.12, eps=0.08; typical ~0.25-0.28). SPI with beta <= 1.5 predicts rises of 0.06-0.15 -- shallow, attenuated psychometrics that RFI cannot generate for any subject. (C) GAMMA CEILING: RFI hard-caps gamma at 1; SPI allows gamma up to 2. Two validity-vs-count conflict cells (G1, G2) are constructed so that the count-sparser option's weighted evidence E is sign-robustly POSITIVE for all gamma in [0,1] (RFI must prefer the count-sparser option, p in [0.52, 0.76]) but NEGATIVE at gamma=2 (E = -0.096 and -0.121), so SPI subjects with gamma > ~1.3-1.55 prefer the count-DENSER option. Any observed p(count-sparser) < 0.5 on G1/G2 falsifies RFI while SPI survives. DESIGN: 8 cues with a wide, well-separated validity ladder [0.98 ... 0.50]. The E-ladder uses nested endorsement sets (A endorsed on cue 5; {5,6}; {4,5,6}; {3,4,5,6}; {2..6}; {2..7}; {1..7}; all 8) against an empty or low-tied opponent, so the rung ordering of E is invariant for EVERY gamma in [0,2] (E = .125/.116/.103; .125/.102/.080; .250/.218/.182; .375/.352/.317; .500/.497/.477; .625/.654/.665; .875/.915/.945; 1 -- values at gamma=0/1/2). Single mid-validity-cue differences (R1, R1c) have E ~= 0.08-0.125 nearly independent of gamma (weight normalization cancels), giving a gamma-robust low-E anchor; the 7-vs-0 and 8-vs-0 cells give gamma-robust extreme anchors. R1c and R5c are 'conflict-style' twins (both options carry endorsements; tied cues cancel exactly) so the sparse preference is not reducible to 'pick the empty bar'. Every informative cell has a mirrored twin (A/B swapped) so side bias cancels; analysis pools by sparse-option side. T0/T1 are identical-vector ties where both theories predict exactly 50% (key-bias check). 24 unique pairs x K=4 = 96 trials exactly; per subject this yields 16 observations on the pooled extreme cells (R7/R8 + mirrors) and 16 on the pooled low-E cells (R1/R1c + mirrors), enough (with N >= 60 subjects and a hierarchical mixture analysis) to locate each subject in the (p_low, p_high) plane. PREDICTED OUTCOMES -- if SPI is true: a CONTINUOUS per-subject distribution of p_high (sparse choice rate on extreme cells) with substantial mass in 0.55-0.80, i.e., inside RFI's forbidden middle band; per-subject psychometric rises spanning 0.05-0.30 (many below 0.19); pooled p_high on extreme cells ~0.70-0.80 (below RFI's 0.78 floor) or, if subjects are consistently sparse, possibly > 0.88 (above RFI's cap); no subject below 0.5 on positive-E cells (SPI's p_core >= 0.5 for E >= 0 since rho <= 0.35); G1/G2 can go below 0.5 for steep-gamma subjects. If RFI-delta is true: a BIMODAL subject population -- a dominant cluster with p_high in [0.82, 0.92], a ~5% cluster at p_high <= 0.19, an EMPTY band (0.2, 0.8); EVERY subject showing rise >= ~0.19 between low-E and extreme cells; pooled p_high on extreme cells within [0.78, 0.88]; p(count-sparser) > 0.5 on both gamma probes; ties at exactly 50%. DECISIVE QUANTITATIVE SIGNATURES: (a) population mass in the (0.25, 0.75) band of per-subject p_high (SPI) vs empty band with two tight clusters (RFI); (b) per-subject rise p_high - p_low < 0.19 for any subject (falsifies RFI, fits low-beta SPI); (c) pooled p_high outside [0.78, 0.88] (falsifies RFI in either direction); (d) p(count-sparser) < 0.5 on G1/G2 (falsifies RFI's gamma <= 1; SPI's gamma up to 2 survives); (e) any single subject with p_high < 0.45 on E >= 0.875 cells (falsifies SPI's 0.5 floor, supports RFI's conventional minority); (f) tie trials at 50% (shared control). The single-cue-difference and conflict trials genuinely require comparing the two rating vectors, satisfying the locked-prompt display requirement.

[8] ADVOCATED: RFI-delta. COMPETITOR: GSP (unnormalized-evidence variant). The two theories share the anti-endorsement sparse-preference core, so no single cell separates them in direction. The decisive structural difference is the EVIDENCE SCALE: RFI NORMALIZES its cue weights (w = v^gamma / sum(v^gamma), so evidence lives in [-1, +1] with slope = beta <= 8), while GSP uses RAW weights (effective slope in normalized units = beta_raw * S(gamma), where S(gamma) = sum(v_j^gamma)). I therefore choose 12 validities [0.98, 0.95, 0.88, 0.85, 0.82, 0.80, 0.78, 0.76, 0.74, 0.72, 0.62, 0.55] with S(0)=12, S(1)=9.45, S(1.5)=8.46. Since GSP's beta_raw >= 1.1, its effective slope is >= 1.1*8.46 = 9.3 > 8 for EVERY gamma in its box [0, 1.5]: the two theories' effective-slope sets are DISJOINT ([3,8] vs [9.3, 38.4]). Behavioral consequence: GSP saturates by endorsement margin 2, while RFI's confidence must keep climbing all the way to the unanimous trial. The design is a margin ladder (option A endorsed on the top-k cues, k = 1,2,3,4,6,8,12, vs an empty B; nested top-k sets make the rung ordering gamma-invariant for both theories), plus conflict-style replications with nonempty bars (tied cues cancel exactly in both), a count-tie validity-conflict cell, a gamma probe, and identical-vector ties. (1) SATURATION DISSOCIATION (headline): on the margin-2 cells (difference carried by the two top cues), RFI's normalized evidence is capped at E_norm = 1.93/9.45 = 0.204, so even at beta=8, delta=0.08, eps=0 its p(sparse) <= 0.78, while its unanimous p >= 0.816 — EVERY RFI majority subject shows a within-subject rise p(k=12) - p(k=2) >= ~0.11 (typically 0.15-0.25), and the ~5% minority shows the mirrored fall <= -0.11. GSP's raw margin-2 evidence is >= 1.895 for all gamma, so sigma >= 0.890 and p(k=2) >= 0.672 for every GSP subject, with rise <= (2m-1)(1-eps)(1-sigma2) <= 0.283*(p2 - 0.5) <= 0.08 at RFI's boundary p2 = 0.78. The (p2, rise) regions are therefore mathematically disjoint: RFI = {p2 in [0.55,0.78], rise >= 0.11} plus a mirrored low cluster {p2 <= 0.40, rise <= -0.11}; GSP = {p2 >= 0.67, rise <= 0.283*(p2-0.5)}. Pooled mid-box predictions: RFI ladder p(sparse) ~ 0.59/0.67/0.73/0.78/0.83/0.86/0.88 for k=1/2/3/4/6/8/12 (rise k2->k12 ~ 0.17); GSP ~ 0.75/0.83/0.85/0.85/0.85/0.85/0.85 (rise ~ 0.02) — GSP is FLAT above margin 2, RFI keeps climbing. (2) SINGLE-CUE CEILING: on k=1 trials RFI's E_norm <= 0.104, so p(sparse) <= 0.665 for every RFI subject (pooled ~0.59); GSP's raw single-cue evidence (~1) with beta_raw up to 3.2 allows p up to 0.95 (pooled ~0.75). Subjects or populations confident on single-cue trials falsify RFI. (3) POPULATION ARCHITECTURE: RFI predicts a bimodal per-subject distribution on extreme cells (94-95% anti-endorsement majority with p(k=12) in [0.82, 0.92], ~5% conventional minority at [0.08, 0.18], EMPTY middle band); GSP's continuous m in [0.75, 1] predicts p(k=12) in [0.72, 0.98] with NO subject below 0.5 on any positive-evidence cell (m >= 0.75 > 0.5 makes p_core >= 0.5 for E >= 0). Even one subject with a below-0.5 sparse rate on E>0 cells falsifies GSP while RFI requires ~5% of them; conversely, population mass in (0.72, 0.82) or (0.92, 1) falsifies RFI's narrow per-subject box. Requires N >= 40-60 subjects; per subject, pooling L2/L2c and mirrors gives 16 margin-2 observations and 8 unanimous observations, enough for a hierarchical (p2, p12) scatter. (4) GAMMA PROBE (G1): A endorsed only on cue 1 (0.98), B on cues 11-12 (0.62, 0.55). A is the count-sparser option; the weighted evidence E = 0.62^gamma + 0.55^gamma - 0.98^gamma is positive for all gamma <= 1 (RFI's box: at gamma=1, +0.19) but flips sign at gamma ~ 1.41, inside GSP's gamma box (at gamma=1.5, -0.074). RFI sign-robustly prefers the count-sparser option (p ~ 0.51-0.53); GSP's steep-gamma subjects prefer the count-denser one. Pooled p(count-sparser) < 0.47 falsifies RFI while GSP survives. (5) COUNT-TIE CONTROL (C2): A endorsed on the two top cues, B on the two bottom cues — a 2-2 count tie, so pure counting predicts 50%; both theories prefer the low-validity-endorsed (effectively sparser) option B, but RFI's E_norm <= 0.08 caps p(B) <= ~0.60 while GSP reaches up to 0.95 (pooled ~0.69 vs ~0.54). This cell simultaneously rules out count-based strategies and adds a second low-margin scale probe. (6) CONFLICT-STYLE REPLICATIONS: L2c and L6c replicate the margin-2 and margin-6 rungs with both bars nonempty (shared endorsements on tied cues cancel exactly in both theories), controlling any 'pick the empty bar' heuristic and doubling the observations on the decisive margin-2 cell. (7) TIES: T0/T1 are identical-vector trials where both theories predict exactly 50% (key-bias check). Every informative trial has a mirrored twin (A/B swapped) so side bias cancels; analysis pools by sparse-option side. 24 unique pairs x K = 4 = 96 trials exactly. DECISIVE QUANTITATIVE SIGNATURES: (a) pooled rise p(k=12) - p(k=2) ~ 0.17 (RFI) vs ~ 0.02 (GSP), and per-subject rise >= 0.11 (RFI, both mirrored clusters) vs GSP's bound rise <= 0.283*(p2 - 0.5) — disjoint regions in the (p2, rise) plane; (b) p(k=1) <= 0.665 for all subjects (RFI) vs substantial mass above 0.7 (GSP); (c) bimodal subject distribution with empty middle and ~5% sub-0.5 minority (RFI) vs continuous [0.72, 0.98] with a hard >= 0.5 floor (GSP); (d) p(count-sparser) >= 0.5 on G1 with a flip below 0.5 falsifying RFI's gamma <= 1; (e) monotone climbing ladder through k=12 (RFI) vs flat-above-k=2 saturation (GSP); (f) ties at exactly 50% (shared control).

[9] ADVOCATED: GSP (unnormalized raw-evidence variant). COMPETITOR: RFI-delta. Both theories share the anti-endorsement sparse-preference core and the same per-trial mixture algebra, so no direction dissociation exists; the design must attack their three structural differences quantitatively. (1) EVIDENCE SCALE. GSP's raw weights give every subject an effective slope (in normalized-evidence units) s = beta_raw * S(gamma), while RFI's normalized weights give s = beta <= 8. I choose 12 validities [0.99, 0.95, 0.88, 0.85, 0.82, 0.80, 0.78, 0.76, 0.74, 0.72, 0.52, 0.52] with S(0)=12, S(1)=9.33, S(1.5)=8.33, so GSP's slope support [1.1*8.33, 3.2*12] = [9.16, 38.4] is DISJOINT from RFI's [3, 8]. Behavioral consequence, read out on a 7-rung matched mid-block margin ladder (Slow: single 0.52-cue, L1..L6: nested cues 4-9 at 0.85-0.74, U: unanimous; rung evidence E_norm = 0.045-0.083 / 0.083-0.094 / 0.167-0.183 / 0.25-0.27 / 0.33-0.35 / 0.50-0.51 / 1.0, gamma-invariant for both theories): every GSP subject's psychometric transition midpoint sits at E_norm <= 0.109 (at or below the single mid-cue rung), while every RFI subject's sits at E_norm >= 0.125 (between L1 and L4). Per-subject half-plane test on the decisive margin-2 rung: rise = p(U) - p(L2) satisfies rise <= 0.556*(p(L2) - 0.5) for EVERY GSP subject (sigma_L2 >= sigma(9.16*0.167) = 0.821) and rise >= 0.626*(p(L2) - 0.5) for EVERY RFI subject (sigma_L2 <= sigma(8*0.179) = 0.807, sigma_U tied to the same beta) -- disjoint half-planes in the (p(L2), rise) plane. Pooled: GSP ~ 0.65/0.71/0.80/0.84/0.85/0.85/0.85 (flat above margin 2; rise L2->U ~ 0.05), RFI ~ 0.56/0.58/0.67/0.73/0.78/0.83/0.86 (monotone climb; pooled rise L2->U >= 0.10). Single-cue ceiling: no RFI subject exceeds p = 0.647 on L1 or 0.635 on Slow; GSP reaches 0.95+. (2) GAMMA CEILING. RFI caps gamma at 1; GSP allows up to 1.5. Flip cells F1/F2/F3 pit a single top cue (0.99 / 0.95 / 0.88) against the low pair (0.52, 0.52); the weighted evidence E(gamma) = 2*0.52^gamma - v_top^gamma is positive for all gamma <= 1 (RFI sign-robustly prefers the top-cue option, p in [0.48, 0.534], slip and lapse cannot cross 0.5) but flips sign at gamma = 1.07 / 1.15 / 1.32. Any subject preferring the low-pair option on any F-cell, in the monotone flip order F1 before F2 before F3, falsifies RFI categorically while remaining inside GSP's box (acknowledged adversarial risk: GSP does not guarantee steep-gamma mass, so absence of flips is also GSP-compatible -- the F-ladder is a one-sided weapon; the ladder signatures above are the two-sided core). (3) POPULATION ARCHITECTURE. RFI: 94-95% majority with per-subject attenuation (1-eps)(1-2*delta) in [0.699, 0.84] (p_U in [0.817, 0.92]) plus ~5% conventional minority with NEGATIVE attenuation (p_U in [0.08, 0.18]) and an empty middle band; GSP: continuous m in [0.75, 1] gives attenuation in [0.44, 0.98], never negative (p_core >= 0.5 whenever E >= 0), with stated mean effective attenuation ~0.70 -- i.e., GSP predicts substantial subject mass BELOW RFI's 0.699 majority floor and ABOVE its 0.84 majority cap (up to p_U = 0.99, past RFI's hard delta >= 0.08 cap of 0.92). Fitted per-subject attenuation from the 56 ladder trials is the arbiter: any subject with negative attenuation falsifies GSP (RFI requires ~5%); any subject with attenuation < 0.699 or > 0.84 falsifies RFI's narrow majority box. Controls: C2 (A endorsed on the top pair 0.99+0.95, B on the low pair 0.52+0.52; 2-2 count tie) -- both theories prefer the low-validity-endorsed option but GSP confidently (p up to 0.94) while RFI is capped at 0.64, simultaneously ruling out pure count-based strategies; T0/T1 identical-vector ties verify both theories' exact-50% prediction and catch key bias; every informative trial has a mirrored twin (A/B swapped) so side bias cancels, with analysis pooled by sparse-option side (ladder) or top-cue-option side (F-cells, C2). 24 unique pairs x K = 4 = 96 trials exactly: per subject 56 ladder observations (8 per pooled rung), 24 flip observations (8 per pooled flip cell), 8 C2, 8 ties. N >= 40-60 subjects with hierarchical per-subject psychometric fitting. DECISIVE QUANTITATIVE SIGNATURES: (a) per-subject (p(L2), rise) half-plane disjointness: rise <= 0.556*(p2-0.5) for all subjects (GSP) vs rise >= 0.626*(p2-0.5) (RFI); (b) per-subject transition-midpoint support E_mid <= 0.109 (GSP) vs >= 0.125 (RFI) -- disjoint slope supports [9.16, 38.4] vs [3, 8]; (c) flat-above-margin-2 ladder with pooled rise L2->U ~ 0.05 (GSP) vs monotone climb with rise >= 0.10 (RFI); (d) single-cue rungs: p(L1) pooled ~0.71 vs ~0.58, no RFI subject above 0.647; (e) attenuation histogram: continuous [0.44, 0.98] with mass below 0.699 and/or above 0.84, no negative values (GSP) vs bimodal with empty middle, tight majority band, ~5% negative-attenuation minority (RFI); (f) any F-cell flip (monotone in flip point 1.07/1.15/1.32) categorically falsifies RFI's gamma <= 1; (g) ties at exactly 50% (shared control); (h) C2 magnitude gap (GSP up to 0.94, RFI <= 0.64).

[10] STRUCTURAL ANALYSIS (why this design looks where it looks). Both theories share the identical anti-endorsement core p(A) = 0.5 + (1-eps)(2m-1)(sigmoid(beta*E)-0.5) with E = sum_j v_j^gamma (b_j - a_j), raw weights, and nearly identical m ([0.74,1] vs [0.75,1]) and epsilon boxes. Critically, RSD-vgp's 'two-component gamma mixture' is algebraically exactly Uniform[0,2]: the 65% component on [0,1.3] and the 35% component on [1.3,2] both have density 0.5, so CDF(t) = 0.5t on [0,2]. GSP's gamma is Uniform[0,1.5]. The theories are therefore THE SAME model with different parameter SUPPORTS, and only three support regions can ever dissociate them: (i) gamma in (1.5, 2.0] -- 25% of RSD subjects, 0% of GSP subjects; (ii) beta in [1.1, 1.8) -- 46% of GSP subjects (ln(1.8/1.1)/ln(3.2/1.1) = 0.461), 0% of RSD; (iii) beta in (3.2, 6.0] -- 52% of RSD subjects, 0% of GSP. The design is built to make exactly these three boundaries empirically decisive, with honest acknowledgment that pooled per-cell proportions differ only modestly (the decisive evidence is subject-level population structure, requiring N >= 60 and hierarchical fitting).

VALIDITY ARCHITECTURE (13 experts): two 'top' experts at 1.00 and 0.99 -- since 1.00^gamma = 1 and 0.99^gamma ~ 1, their combined weight T(t) = 1 + 0.99^t is ~1.98-2.00 for ALL gamma, so the top pair acts as a gamma-invariant weight-2 block; seven bottom experts at exactly 0.50, whose power weights halve with every unit of gamma; and four mid experts (0.92, 0.85, 0.75, 0.65) used for the beta probes and tied fillers.

DISSOCIATION 1 -- THE FLIP LADDER (gamma support, the headline). A flip cell endorses A on the two top experts and B on k of the 0.50-experts, all else tied at 0. E(gamma) = k*0.5^gamma - (1+0.99^gamma), which is strictly decreasing with a single sign change at t* = log2(k/2): F4 (k=4) t*=1.01; F5 (k=5) t*=1.33; F6 (k=6) t*=1.60; F7 (k=7) t*=1.82. Low-gamma subjects prefer the count-sparser side (A, 2 endorsements); gamma > t* subjects prefer the side endorsed by MORE but low-validity experts (B). Because both gamma distributions are uniform on their supports, the predicted fraction of subjects with a true inverted preference is EXACTLY the survival function: RSD (2-t*)/2 = 0.50 / 0.33 / 0.20 / 0.09 for F4/F5/F6/F7; GSP max(0,(1.5-t*)/1.5) = 0.33 / 0.11 / 0 / 0. The F6 and F7 zeros are HARD GSP predictions: E_F6(1.5) = +0.136 > 0 and E_F7(1.5) = +0.490 > 0, and since m >= 0.75 makes p_core >= 0.5 whenever E >= 0, NO GSP subject can truly prefer the low-validity side on F6/F7 (lapse keeps p >= 0.5 exactly). Any subject whose pooled F6 (or F7) rate is credibly below 50% categorically falsifies GSP, while RSD REQUIRES ~20% (F6) and ~9% (F7) of subjects there. F5 gives a 3x population contrast (33% vs 11% inverters) with a DEPTH dissociation: GSP inverters (gamma in (1.33,1.5]) are bounded below by p ~ 0.34 (worst case beta=3.2, A_eff=0.95; ~0.39 at typical A_eff=0.69, since E_F5(1.5) = -0.217), whereas RSD inverters reach p ~ 0.04 (E_F5(2) = -0.730, beta=6). Moreover the within-subject pattern is diagnostic: an RSD inverter with gamma > 1.60 inverts F4, F5 AND F6 monotonically, while a GSP inverter (gamma <= 1.5) inverts F4, F5 but NEVER F6. Estimated per-subject gamma from the ladder traces the survival curve: linear down to 2.0 (RSD) vs a hard edge at 1.5 (GSP).

DISSOCIATION 2 -- BETA FLOOR AND SUPPORT (P1/P2 probes). A single high-validity endorsement difference with tied fillers gives |E| = 0.92^gamma on P1 (range [0.846, 0.92], only 8% variation over gamma in [0,2]) and 0.85^gamma on P2 ([0.72, 0.85]) -- essentially gamma-invariant moderate evidence, isolating beta. With A_eff read out from the saturated unanimous cell U (S(gamma) >= 6.28, so beta*S >= 6.9 and p_U = 0.5 + A_eff/2 within 0.001), the normalized probe confidence c = (p-0.5)/A_eff is boxed: GSP c in [0.217, 0.450] (P1) and [0.189, 0.438] (P2); RSD c in [0.321, 0.496] (P1) and [0.286, 0.493] (P2). GSP predicts 46% of subjects BELOW RSD's hard floor c = 0.321; RSD predicts none. In the (p_U, p_P1) plane, RSD subjects lie on or above the line p_P1 = 0.5 + 0.321(2*p_U - 1); roughly 46% of GSP subjects fall strictly below it (down to 0.5 + 0.217(2*p_U - 1)). Symmetrically, 52% of RSD subjects exceed GSP's effective confidence ceiling, a weaker but corroborating upper-tail signal.

CONTROLS. C1 (2-2 count tie: tops vs lows): E(t) = 2*0.5^t - (1+0.99^t) < 0 for all t > 0, so both theories prefer the low-validity-endorsed side while a pure counting strategy predicts exactly 50% -- this rules out count-based responding and doubles as a low-gamma probe (E -> 0 as gamma -> 0). T (identical vectors): both theories predict exactly 50% (key-bias check). U (unanimous): saturated for every subject (beta*S >= 6.9), reading out A_eff needed to interpret the probes. Every informative cell has a mirrored twin (A/B swapped) so side bias cancels; F-cells pool by count-sparser side, P-cells by the non-endorsed side. 16 unique pairs x K = 6 = exactly 96 trials: per subject, 12 observations per pooled mirrored cell (F4-F7, P1, P2, C1), 6 on U, 6 on T.

PREDICTED OUTCOMES. If RSD-vgp is true: ~20% of subjects credibly below 50% on F6 and ~9% on F7 (down to rates ~0.05-0.25); F5 inversion ~33% with deep defections; every subject's normalized probe confidence c >= 0.32; per-subject gamma estimates fill (1.5, 2.0] with ~25% mass; every subject's beta >= 1.8. If GSP is true: ZERO subjects credibly below 50% on F6/F7 (inversion rate at binomial-noise level only); F5 inversion ~11% bounded below ~0.34; ~46% of subjects with c < 0.32 on P1; gamma histogram with a hard edge at 1.5; beta down to 1.1. Pooled per-cell proportions are close (e.g., F5 pooled p(sparse side) ~ 0.60 RSD vs ~0.67 GSP) because RSD's extra steep-gamma mass is offset by GSP's lower betas -- the arbiter is the subject-level population structure, so the analysis must be hierarchical (per-subject gamma, beta, A_eff recovery plus mixture model comparison, N >= 60, with simulation-based model recovery reported).

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "STRUCTURAL SETUP. GSP and RSD-vgp share the identical anti-endorsement core p(A) = 0.5 + (1-eps)(2m-1)(sigmoid(beta*E)-0.5) with E = sum_j v_j^gamma (b_j - a_j), raw weights, and nearly identical m ([0.75,1] vs [0.74,1]) and epsilon boxes. RSD-vgp's two-component gamma mixture is algebraically exactly Uniform[0,2] (both components have density 0.5), vs GSP's Uniform[0,1.5]; and the beta supports are log-uniform [1.1,3.2] (GSP) vs [1.8,6.0] (RSD). Only three support regions can ever dissociate them: gamma in (1.5,2] (25% of RSD subjects, 0% of GSP), beta in [1.1,1.8) (46% of GSP subjects, 0% of RSD), beta in (3.2,6] (52% of RSD, 0% of GSP). This design makes the first two decisive with a NEW validity architecture built around a gamma-INVARIANT probe ladder, unlike prior work whose probes carried 8-15% gamma variation.\n\nARCHITECTURE (13 experts): [1.00, 0.99, 0.99, 0.99, 0.85, 0.80, 0.75, 0.55x6]. The three 0.99 experts have weights in [0.9801, 1] for ALL gamma in [0,2] -- a 2%-variation, effectively gamma-invariant evidence unit. A single 0.99-cue difference gives |E| ~ 1 EXACTLY regardless of gamma; two such cues give |E| ~ 2. This isolates beta from gamma at the subject level, which no previous design achieved.\n\nDISSOCIATION 1 -- BETA FLOOR (headline, gamma-invariant). On P99 (single 0.99-cue difference, all else tied), per-subject normalized confidence c = (p-0.5)/A_eff (A_eff read from the saturated unanimous cell, where beta*E >= 1.1*7.68 = 8.4 for every subject) equals sigma(beta*E)-0.5 with E in [0.98, 1]: GSP c in [0.246, 0.461] with 46% of subjects (beta in [1.1,1.8)) at or below c ~ 0.355; RSD c in [0.354, 0.498] with a HARD floor at 0.354 (beta >= 1.8). GSP predicts ~46% of subjects below RSD's floor; RSD predicts zero. Corroborating probes P85 (c_GSP in [0.188,0.438] vs c_RSD in [0.286,0.494]) and P75 (c_GSP in [0.150,0.417] vs c_RSD in [0.233,0.489]) form a 4-rung validity ladder (P99, P99x2, P85, P75) enabling joint per-subject (gamma, beta, A_eff) recovery by hierarchical logistic fit; P99x2 (|E| ~ 2) confirms where each subject saturates (GSP floor c = 0.396 vs RSD floor 0.471).\n\nDISSOCIATION 2 -- GAMMA CEILING (flip ladder). Flip cells pit the top pair {1.00, 0.99} (combined weight T(gamma) in [1.980, 2], gamma-invariant) against k cues at 0.55: E(gamma) = k*0.55^gamma - T(gamma), flipping at t* = 1.17 (F4, k=4), 1.545 (F5, k=5), 1.855 (F6, k=6); F3 (k=3) flips at 0.685. Because m >= 0.74 makes p_core >= 0.5 whenever E >= 0, NO GSP subject can truly prefer the count-denser low-validity side on F5 or F6: E_F5(1.5) = +0.055 > 0 and E_F6(1.5) = +0.462 > 0 across GSP's entire box -- a hard floor at exactly 50%. RSD REQUIRES (2-t*)/2 = 41.6% / 22.7% / 7.2% of subjects to invert on F4/F5/F6, with deep defections (at gamma=2, beta=6: E_F5 = -0.448, p(sparse side) down to ~0.08). Any subject with pooled F5 rate credibly below 50% categorically falsifies GSP while RSD demands ~23% of subjects there. Within-subject monotonicity traces the survival curve: an RSD subject inverting F6 also inverts F5 and F4; a GSP subject may invert F4 (22% of subjects, gamma in (1.17,1.5]) but NEVER F5 -- and GSP F4 inversions are shallow (p >= ~0.25, since |E| <= 0.353 and beta <= 3.2), whereas RSD F4 inversions reach ~0.05. Per-subject gamma estimates show a hard edge at 1.5 (GSP) vs support filling (1.5,2] with 25% mass (RSD).\n\nPOOLED (mid-box) predictions, p(sparse side): GSP ~ F3 0.60 / F4 0.68 / F5 0.79 / F6 0.83 / P99 0.77 / P85 0.75 / P75 0.73 / U 0.85; RSD ~ F3 0.55 / F4 0.61 / F5 0.79 / F6 0.84 / P99 0.82 / P85 0.80 / P75 0.79 / U 0.84. Pooled differences are modest (the F4 gap ~0.07 and the ~0.05 probe gap are the largest); the arbiter is subject-level population structure, requiring N >= 60 and hierarchical fitting -- honestly acknowledged.\n\nCONTROLS. F5c replicates F5 with BOTH bars nonempty (tied mid cues e4-e6 cancel exactly in both theories), controlling any 'pick the empty bar' heuristic. C2 is a 2-2 count tie (top pair vs two 0.55s): E = 2*0.55^gamma - T < 0 for all gamma > 0, so both theories confidently prefer the low-validity-endorsed side while pure counting predicts exactly 50% -- ruling out count-based responding. T0/T1 (identical vectors) verify both theories' exact-50% prediction and catch key bias. U/Um saturate every subject (beta*E >= 8.4), reading out A_eff needed to normalize the probes. Every informative cell has a mirrored twin (A/B swapped) so side bias cancels; analysis pools by sparse-option side (F-cells), non-endorsed side (probes), all-0 side (U), low-validity side (C2). 24 unique pairs x K = 4 = exactly 96 trials: per subject 8 observations per pooled mirrored cell, 16 on the pooled F5+F5c cell, 32 across the probe ladder.\n\nDECISIVE QUANTITATIVE SIGNATURES: (a) fraction of subjects with normalized P99 confidence c < 0.35: ~46% (GSP) vs 0% (RSD, hard floor 0.354); (b) per-subject p(sparse) on pooled F5: >= 0.5 for ALL subjects (GSP, hard floor) vs ~23% of subjects below 0.5, reaching 0.05-0.25 (RSD); (c) same on F6: 0% vs ~7%; (d) F4 inversion depth: GSP inverters bounded below ~0.25, RSD inverters reach <= 0.10; (e) per-subject gamma histogram: hard edge at 1.5 (GSP) vs 25% mass in (1.5,2] (RSD); (f) monotone within-subject flip pattern F6 -> F5 -> F4 with no GSP-compatible F5 inversion; (g) ties at exactly 50% and C2 low-side preference > 50% (shared controls, the latter ruling out counting).",
  "validities": [
    1.0,
    0.99,
    0.99,
    0.99,
    0.85,
    0.8,
    0.75,
    0.75,
    0.75,
    0.75,
    0.75,
    0.75,
    0.75
  ],
  "trial_a_ratings": [
    [
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
      0,
      0,
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
      1,
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
      1,
      1,
      1,
      1,
      1,
      1,
      1
    ]
  ],
  "trial_b_ratings": [
    [
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
      1,
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
      0,
      0,
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
      1,
      1,
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
      1,
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
  "input_tokens": 17938,
  "output_tokens": 28929,
  "reasoning_tokens": 25532,
  "provider": "Sail Research"
}
```
