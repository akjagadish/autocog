# experiment_attempt_01

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
**Description:** Heavy-Tail Anti-Endorsement Integration, revision 6 (HT-AEI-r6). Subjects reading a product described by binary expert ratings treat an endorsement (rating = 1) as evidence AGAINST the option carrying it (a 'fewer red flags' reading), so the count-sparser option is perceived as higher quality. Within a trial, all cues are integrated additively with RAW (unnormalized) validity weights: the evidence for A over B is E = sum_j v_j^gamma (b_j - a_j), so the absolute endorsement-count margin sets the evidence scale independent of feature count, and identical rating vectors give exactly 50%. Choice follows p(A) = (1-eps)*[m*sigmoid(beta*E) + (1-m)*(1-sigmoid(beta*E))] + eps/2, with effective attenuation A_eff = (1-eps)(2m-1) concentrated near 0.69 — the 'steep normalized slope x moderate asymptote' family. The population is a three-segment mixture with SEGMENT-SPECIFIC PRECISION, all segments now at moderate-to-high precision (no soft-beta segment remains, since weakly identified slopes were shown to leak Exp 10/14 posteriors below threshold and to donate RFI-ward on Exp 9): (1) a 40% low bulk on gamma in [0, 1.0] with beta log-uniform on [2.2, 4.0] — the validated shared-core bulk, concentrated below the ~0.73/1.0 flip points of Exp 16's ladder, with the beta floor raised so every low-bulk subject's normalized slope is sharply identified; (2) a 25% mid segment on gamma in [1.05, 1.35] with beta log-uniform on [2.2, 3.2] — a band where RSD's uniform density (0.5) beats SGAI's (0.4) and EPAI's (0.375), sitting fully inside GSP's gamma box with GSP-compatible beta, so these subjects donate RSD-ward on Exp 15 while pulling Exp 11 and Exp 9 toward their observed values; (3) a 35% heavy tail on gamma in [2.0, 2.10] with beta log-uniform on [2.2, 2.8] — an extreme edge-hug of the RSD gamma boundary (average distance <= 0.05) at sharpened precision, so tail subjects' gamma posteriors concentrate just above 2.0 (feeding Exp 14), their slopes stay above the s>9.5 threshold (feeding Exp 10), and their near-deterministic past-flip-point behavior pulls Exp 16 down toward 0.4335. No per-trial gamma jitter. Falsifiable signatures: ~35% of subjects carry gamma within 0.10 of 2.0 with per-subject posteriors extending smoothly past 2.0; Exps 13/15 small and positive; Exp 11 in the 50-100 range; Exp 14 at or above 0.21; Exp 16 in [0.42, 0.45]; Exp 10 at or above 0.85.

**Parameters:**
- gamma_u: [0, 1]
- log_beta_low: [0.788, 1.386]
- log_beta_mid: [0.788, 1.163]
- log_beta_tail: [0.788, 1.030]
- epsilon: [0.03, 0.10]
- m: [0.75, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Heavy-Tail Anti-Endorsement Integration, revision 6 (HT-AEI-r6).
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are 0/1 cue values. History is ignored
    # (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"HT-AEI-r6 expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    # ---- Validity gating: three-segment mixture with SEGMENT-SPECIFIC
    # beta. All segments now carry moderate-to-high precision (no
    # soft-beta segment: weakly identified slopes were shown to leak
    # Exp 10/14 posteriors below threshold and donate RFI-ward on Exp 9).
    # gamma_u ~ U[0, 1] is mapped through:
    #   u in [0.00, 0.40): gamma ~ U[0.0, 1.0], beta log-U[2.2, 4.0]
    #            (40%; the validated low bulk, UNTOUCHED in gamma support.
    #            Beta floor raised 1.8 -> 2.2 per the pre-specified Exp 10
    #            contingency: sharpens per-subject slope identification
    #            without touching any other mechanism.)
    #   u in [0.40, 0.65): gamma ~ U[1.05, 1.35], beta log-U[2.2, 3.2]
    #            (25%; ENLARGED from 18%. On [1.05, 1.3] RSD's uniform
    #            density 0.5 beats SGAI's 0.4 and EPAI's 0.375, so these
    #            subjects donate RSD-ward on Exp 15; the band sits fully
    #            inside GSP's gamma box with GSP-compatible beta, pulling
    #            Exp 11 down toward the observed 52.6 and Exp 9 GSP-ward
    #            toward -1.76; it is above the 1.0 flip point
    #            (Exp-16-neutral); gamma near 1.2 matches the
    #            validity-proportional metrics of Exps 2 and 5.)
    #   u in [0.65, 1.00]: gamma ~ U[2.0, 2.10], beta log-U[2.2, 2.8]
    #            (35%; TRIMMED from 42% and RE-SHARPENED. Extreme edge-hug
    #            of the RSD gamma boundary (average distance <= 0.05), with
    #            beta raised from [1.9, 2.4] to [2.2, 2.8]: sharper gamma
    #            identification concentrates tail posteriors just above
    #            2.0 (Exp 14 recovers), keeps slopes above the s>9.5
    #            threshold (Exp 10), makes tail subjects near-deterministic
    #            past the flip points (Exp 16 pulled down toward 0.4335),
    #            and removes RFI-fittable soft slopes (Exp 9 back toward
    #            -1.76). The fraction trim cuts the residual Exp 13 edge
    #            cost ~17% while the mid segment's positive donations
    #            hold the total near the observed +6.)
    u = float(parameters["gamma_u"])
    if u < 0.40:
        gamma = 1.0 * (u / 0.40)
        log_b = float(parameters["log_beta_low"])
    elif u < 0.65:
        gamma = 1.05 + 0.30 * ((u - 0.40) / 0.25)
        log_b = float(parameters["log_beta_mid"])
    else:
        gamma = 2.0 + 0.10 * ((u - 0.65) / 0.35)
        log_b = float(parameters["log_beta_tail"])

    beta = float(np.exp(log_b))
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
    # (1-m) with the conventional polarity. Continuous, no empty middle,
    # no mirror cluster. NO per-trial gamma jitter (delta = 0): the
    # stochastic attention gate is a confirmed dead lever.
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

[11] STRUCTURAL SETUP. GSP and RSD-vgp share the identical anti-endorsement core: p(A) = 0.5 + (1-eps)(2m-1)(sigmoid(beta*E) - 0.5) with E = sum_j v_j^gamma (b_j - a_j), RAW weights, and nearly identical m ([0.75,1] vs [0.74,1]) and epsilon boxes. RSD's two-component gamma mixture is algebraically exactly Uniform[0,2] (both components have density 0.5). The theories are therefore the SAME model with different parameter SUPPORTS, and only three regions can ever dissociate them: (i) gamma in (1.5, 2] -- 25% of RSD subjects, 0% of GSP; (ii) beta in [1.1, 1.8) -- 46.1% of GSP subjects (ln(1.8/1.1)/ln(3.2/1.1) = 0.461), 0% of RSD; (iii) beta in (3.2, 6.0] -- 52.3% of RSD subjects, 0% of GSP. This design makes all three boundaries decisive using a NEW architecture built on EXACT-UNIT cues, unlike prior designs whose gamma-invariant probes relied on 0.99-validity cues (2% weight drift) and whose beta readout required A_eff normalization via a saturated anchor.

KEY ALGEBRAIC LEVER (new, exact, parameter-free). For BOTH theories, on EVERY trial, p - 0.5 = A_eff * (sigmoid(beta*E) - 0.5) exactly, where A_eff = (1-eps)(2m-1) > 0. Therefore the RATIO of choice deviations across cells with the same evidence sign is exactly A_eff-free AND lapse-free: r12 = (p_{E=2} - 0.5)/(p_{E=1} - 0.5) = (sigmoid(2*beta)-0.5)/(sigmoid(beta)-0.5), a pure function of beta alone. I construct cells with EXACTLY integer evidence E = 1, 2, 3 using cues of validity 1.00 (weight = 1 for every gamma in [0,2], perfectly gamma-invariant) plus exactly-canceling shared endorsements, so E is an exact integer for every subject of either theory at every gamma.

ARCHITECTURE (12 experts): three 1.00 cues (exact unit weights, gamma-invariant), then 0.50 x3, 0.55 x2, 0.60 x2, 0.65, 0.70. S(gamma) ranges over [5.99, 12], so the unanimous cell saturates every subject (beta*S >= 1.1*5.99 = 6.59, sigmoid >= 0.9986), reading out A_eff = 2*(p_U - 0.5).

DISSOCIATION 1 -- BETA SUPPORT VIA THE EXACT RATIO LADDER (headline). Cells B1/B2/B3 differ only on unit cues with exactly canceling shared endorsements: E = -1, -2, -3 exactly, for every gamma. Per-subject r12 = (sigmoid(2b)-0.5)/(sigmoid(b)-0.5): GSP support [1.082, 1.604] with 46.1% of subjects ABOVE 1.3216 (all beta < 1.8); RSD support [1.005, 1.3216] with a HARD ceiling at 1.3216 (beta >= 1.8) and 52.3% at or below 1.082 (beta >= 3.2). The regions are DISJOINT on both tails: any subject with r12 > 1.33 categorically falsifies RSD while GSP requires ~46% of subjects there; conversely any subject with r12 <= 1.082 falsifies GSP while RSD requires ~52% there. The corroborating ratio r13 = (sigmoid(3b)-0.5)/(sigmoid(b)-0.5) gives an independent second readout (GSP [1.085, 1.860], 46% above 1.384; RSD ceiling 1.384). No prior design exploited this exact cancellation-based ratio signature.

DISSOCIATION 2 -- GAMMA CEILING VIA THE FLIP LADDER. Flip cells pit the exact unit pair (A endorsed on two 1.00 cues, weight exactly 2 for all gamma) against low-validity blocks whose total weight W(gamma) crosses 2 at a designed flip point g*: E(gamma) = W(gamma) - 2, strictly decreasing. Flip points: F0 = 0.585 (3x0.50), F1 = 1.31 (0.50, 0.55, 0.60, 0.70), F2 = 1.47 (0.55, 0.60, 0.65, 0.70), F3 = 1.65 (0.50, 0.50, 0.55, 0.60, 0.70), F4 = 1.91 (0.50 x3, 0.55, 0.60, 0.70). Because m >= 0.74 > 0.5 makes p >= 0.5 whenever E >= 0, and E_F3(1.5) = +0.165 > 0 and E_F4(1.5) = +0.519 > 0, NO GSP subject (gamma <= 1.5) can truly prefer the low-validity-denser side on F3/F4 -- a hard floor at exactly 50%. RSD REQUIRES (2-g*)/2 = 17.5% of subjects below 0.5 on F3 and 4.5% on F4. Predicted population inversion rates (fraction of subjects with true p < 0.5): F0: 0.610 (GSP) / 0.707 (RSD); F1: 0.127 / 0.345; F2: 0.020 / 0.265; F3: 0.000 / 0.175; F4: 0.000 / 0.045. The inversion-rate curve vs flip point is piecewise-linear falling to zero AT 1.5 with a hard edge (GSP) vs continuing linearly to 2.0 (RSD); F2/F3 tightly bracket the 1.5 edge. Within-subject, an RSD inverter of F3 necessarily inverts F2, F1, F0 (monotone in gamma), and per-subject gamma estimates fill (1.5, 2.0] with 25% mass under RSD vs a hard edge at 1.5 under GSP. Inversion depth: on F1 an RSD subject at gamma=2, beta=6 has E = -0.60, p(sparse side) ~ 0.17; on F3 at gamma=2, p ~ 0.23.

CONTROLS. All beta-axis cells are CONFLICT-STYLE (both options carry endorsements; shared endorsements on identical validities cancel exactly), ruling out any 'pick the empty bar' heuristic; Uc (9 endorsements vs 3, |E| >= 3.44) is a near-saturated conflict twin of the unanimous cell, so p_U ~ p_Uc verifies that the A_eff anchor is not an empty-bar artifact. C2 is a 2-2 COUNT TIE (A endorsed on the two unit cues, B on two 0.50 cues): E = 2*0.5^gamma - 2 < 0 for all gamma > 0, so both theories prefer the low-validity-endorsed side (GSP mid-box p ~ 0.74, RSD ~ 0.82) while pure counting predicts exactly 50% -- ruling out count-based responding. T1/T2 are identical-vector ties where both theories predict exactly 50% (key-bias check). Every informative cell has a mirrored twin (A/B swapped) so side bias cancels; analysis pools by sparse-option side (ladder, U), unit-cue side (B-cells), or low-validity side (C2).

POOLED MID-BOX PREDICTIONS (GSP at gamma=0.75, beta=2.0, A_eff=0.70 vs RSD at gamma=1.0, beta=3.0, A_eff=0.70), p(sparse side): U 0.85/0.85; B1 0.77/0.82; B2 0.84/0.85; B3 0.85/0.85; F0 0.42/0.28; F1 0.71/0.67; F2 0.73/0.74; F3 0.75/0.75; F4 0.76/0.76; C2 0.74/0.82; ties 0.50/0.50. Pooled differences are modest (largest: F0 ~ 0.14, C2 ~ 0.08, B1 ~ 0.05) -- honestly acknowledged: the arbiter is SUBJECT-LEVEL population structure, requiring N >= 60-80 and hierarchical Bayesian fitting (per-subject gamma, beta, A_eff with population priors given by each theory, plus simulation-based model recovery), as in the strongest prior designs.

DECISIVE QUANTITATIVE SIGNATURES: (a) per-subject exact ratio r12 > 1.33 (equivalently hierarchical beta-hat < 1.8): ~46% of subjects (GSP) vs 0% hard (RSD); (b) r12 <= 1.082 (beta-hat >= 3.2): ~52% (RSD) vs ~0% (GSP); (c) any single subject with pooled F3 (or F4) rate credibly below 0.5 categorically falsifies GSP (E_F3(1.5) = +0.165 > 0 makes p >= 0.5 for every GSP parameter), while RSD requires 17.5% of subjects below 0.5 on F3; (d) inversion-rate survival curve over flip points 0.585/1.31/1.47/1.65/1.91: hard zero past 1.5 with a kink (GSP) vs linear decline to 2.0 with 25% of gamma-hat mass in (1.5, 2] (RSD); (e) within-subject monotone flip pattern (inverting F3 implies inverting F2, F1, F0); (f) per-subject 2D (gamma-hat, beta-hat) histogram with mass split across the three exclusive support regions [beta<1.8] vs [gamma>1.5] vs [beta>3.2]; (g) C2 low-side preference > 50% and ties at exactly 50% (shared controls, the former ruling out counting).

[12] STRUCTURAL ANALYSIS. RSD-vgp and EPAI-xp share the identical core: p(A) = 0.5 + A_eff*(sigmoid(beta*E) - 0.5), E = sum_j v_j^gamma*(b_j - a_j), raw (unnormalized) weights, anti-endorsement polarity. RSD-vgp's two-component gamma mixture is algebraically EXACTLY Uniform[0,2] (both components have density 0.5). EPAI-xp's gamma mixture has density 0.4583 on [0,1.2] and 0.375 on [1.2,2.4]. The theories are therefore the SAME model with different parameter SUPPORTS, and only three support regions can dissociate them: (i) gamma in (2.0, 2.4]: 15% of EPAI subjects, 0% of RSD; (ii) beta in [1.8, 2.0): 8.8% of RSD subjects (ln(2/1.8)/ln(6/1.8)), 0% of EPAI (the (6,7] region is behaviorally undetectable due to saturation -- acknowledged one-sidedness); (iii) A_eff = (1-eps)(2m-1): RSD support [0.427, 0.970] vs EPAI [0.504, 0.892] -- roughly 12% of RSD subjects fall below 0.504 and 8% above 0.892, while EPAI places exactly ZERO subjects outside its box. The design makes all three boundaries empirically decisive.

ARCHITECTURE (12 experts): three cues at validity 1.00 (power weight exactly 1 for EVERY gamma -- gamma-invariant unit cues) and nine near the validity floor: seven at 0.50, one at 0.52, one at 0.55. The unit trio enables cells with EXACT integer evidence E = -1, -2, -3 for every subject at every gamma; the near-0.5 block's weights halve approximately per unit of gamma, giving fine-grained, exactly computable flip points against the gamma-invariant weight-2 top pair. Unanimous total weight S(gamma) >= 4.77 for all gamma in [0, 2.4], so the unanimous cells saturate every subject of both theories (beta*|E| >= 8.6 even at RSD's lowest beta = 1.8).

DISSOCIATION 1 -- FLIP LADDER (gamma ceiling; headline). On flip cells, A is endorsed on the two 1.00 cues (endorsement mass exactly 2 for all gamma) plus a shared canceling endorsement on the third 1.00 cue; B is endorsed on a block of k low cues. E(gamma) = W_block(gamma) - 2, strictly decreasing, with a single sign change at the flip point g*: subjects with gamma < g* prefer A (the count-sparser, top-cue-endorsed option); gamma > g* prefer B. Flip points: F4 (k=4 at 0.50): g* = 1.000; F5 (k=5): 1.322; F6 (k=6): 1.585; F7 (k=7): 1.807; F8 (k=8: 7x0.50+0.52): 2.015; F9 (k=9: 7x0.50+0.52+0.55): 2.223. Predicted population inversion rates P(gamma > g*): RSD 0.500/0.339/0.207/0.097/0.000/0.000; EPAI 0.542/0.404/0.306/0.222/0.144/0.066. HARD CELLS: F8 and F9 have E(2.0) = +0.020 and +0.323 > 0, and E is strictly decreasing, so EVERY RSD subject (gamma <= 2) has E > 0 and therefore p(A) > 0.5 STRICTLY (m >= 0.74 > 0.5 keeps p_core > 0.5 whenever E > 0; lapse only pulls toward 0.5). No RSD subject can prefer B on F8/F9. EPAI-xp REQUIRES 14.4% of subjects below 0.5 on F8 (inversion depth down to p(A) ~ 0.26 at gamma = 2.4, beta = 3.7) and 6.6% on F9. Any subject with a credibly inverted preference on F8/F9 categorically falsifies RSD-vgp while remaining inside EPAI's box; conversely, a hard zero of inversions past g* = 2.0 where EPAI demands 14% falsifies EPAI. The full survival curve (inversion rate vs g*) is piecewise-linear hitting zero AT 2.0 (RSD) vs declining smoothly to zero at 2.4 (EPAI). Within-subject, inverting F9 implies inverting F8, F7, F6... (monotone in gamma). Corroboration: EPAI's gamma density drops 18% at 1.2 (0.4583 -> 0.375), predicting a visible kink in the per-subject gamma histogram; RSD's histogram is flat to its hard edge at 2.0.

DISSOCIATION 2 -- A_eff SUPPORT (m/epsilon boxes). The unanimous cells U/Um saturate every subject, reading out A_eff = 2*(p_sparse - 0.5) to within 0.0002. RSD support for p_U: [0.714, 0.985]; EPAI hard box: [0.752, 0.946]. RSD places ~12% of subjects below 0.752 and ~8% above 0.946 (its m extends to 1.0 with epsilon down to 0.03, allowing near-deterministic subjects; EPAI's m <= 0.96 and epsilon >= 0.03 cap A_eff at 0.892). Any subject with credibly p_U < 0.752 or > 0.946 falsifies EPAI-xp categorically; RSD requires ~20% of subjects in these forbidden zones.

DISSOCIATION 3 -- BETA FLOOR (exact ratio ladder). B1/B2/B3 use only the unit cues with exactly canceling shared endorsements (a shared endorsement on the same-validity cue contributes w*(1-1) = 0), giving E = -1, -2, -3 EXACTLY for every gamma. The per-subject ratio r12 = (0.5 - p_B2)/(0.5 - p_B1) = (sigmoid(2*beta) - 0.5)/(sigmoid(beta) - 0.5) is A_eff-free and lapse-free. RSD support: r12 in [1.005, 1.322] (beta in [1.8, 6]); EPAI: [1.003, 1.266] (beta in [2.0, 7.0]). The region r12 in (1.266, 1.322] (i.e., beta in [1.8, 2.0)) is RSD-EXCLUSIVE: 8.8% of RSD subjects, 0% of EPAI. The independent second ratio r13 = (0.5 - p_B3)/(0.5 - p_B1) has RSD-exclusive region (1.307, 1.384]. The upper-beta region (6, 7] produces only saturation-level differences in both ratios and is acknowledged as undetectable -- this weapon is one-sided.

CONTROLS. C/Cm: a 3-3 endorsement-count tie (A endorsed on the two 1.00 cues plus the shared cue; B on two 0.50 cues plus the shared cue): E = 2*0.5^gamma - 2 < 0 for all gamma > 0, so both theories prefer the low-validity-endorsed side (pooled ~0.79) while pure counting predicts exactly 50% -- ruling out count-based responding. T1/T2: identical vectors -- both theories predict exactly 50% (key-bias check). Every informative cell has a mirrored twin (A/B swapped) so side bias cancels; analysis pools by target side (top-pair side for flip cells, non-endorsed side for B-cells, sparse side for U). All informative trials carry distinct, non-trivially comparable rating vectors on both bars (locked-prompt requirement).

POWER AND ANALYSIS. 24 unique pairs x K = 4 = 96 trials exactly; 8 observations per pooled mirrored cell per subject. Pooled mid-box differences are modest and honestly acknowledged (approximate pooled p(top-pair side): F7 ~0.77 RSD vs ~0.71 EPAI; F8 ~0.77 vs ~0.71; F9 ~0.80 vs ~0.78; F4 ~0.50 both; U ~0.845 both; B1 ~0.20 both) because the populations' central tendencies nearly coincide -- the arbiter is SUBJECT-LEVEL population structure. Requires N >= 60 subjects with hierarchical Bayesian fitting of per-subject (gamma, beta, A_eff) under each theory's population prior, plus simulation-based model recovery.

DECISIVE QUANTITATIVE SIGNATURES: (a) any subject with credibly p(A) < 0.5 on F8 or F9 categorically falsifies RSD-vgp (E(2.0) > 0 over its entire gamma-box), while EPAI requires 14.4%/6.6% of subjects there; (b) inversion-rate survival curve over g* = 1.00/1.32/1.59/1.81/2.02/2.22: RSD 0.50/0.34/0.21/0.10/0/0 vs EPAI 0.54/0.40/0.31/0.22/0.14/0.07 -- a hard zero past 2.0 (RSD) vs a smooth decline to 2.4 (EPAI); (c) per-subject gamma histogram: flat with hard edge at 2.0 (RSD) vs an 18% density drop at 1.2 and 15% of mass in (2.0, 2.4] (EPAI); (d) per-subject p_U outside [0.752, 0.946]: ~20% (RSD) vs 0% (EPAI); (e) per-subject r12 in (1.266, 1.322]: ~9% (RSD) vs 0% (EPAI); (f) within-subject monotone flip pattern (inverting F9 implies inverting F8..F4); (g) count-tie C preferring the low-validity side above 50% and ties at exactly 50% (shared controls).

[13] STRUCTURAL SETUP. EPAI-xp and RSD-vgp share the identical anti-endorsement core: p(A) = 0.5 + A_eff*(sigmoid(beta*E) - 0.5), E = sum_j v_j^gamma*(b_j - a_j), RAW weights. RSD-vgp's two-component gamma mixture is algebraically EXACTLY Uniform[0,2] (both components have density 0.5). EPAI-xp's mixture has density 0.4583 on [0,1.2] and 0.375 on [1.2,2.4] (mean gamma 1.14 vs RSD's 1.0). The theories are the SAME model with different parameter SUPPORTS; only three regions can dissociate them: (i) gamma in (2.0, 2.4] -- 15% of EPAI subjects, 0% of RSD; (ii) beta in [1.8, 2.0) -- 8.8% of RSD subjects (ln(2/1.8)/ln(6/1.8)), 0% of EPAI (EPAI's (6,7] beta region is saturation-blind and acknowledged as undetectable -- one-sided weapon); (iii) A_eff = (1-eps)(2m-1): RSD support [0.427, 0.970] (p_unanimous in [0.714, 0.985], ~12% below 0.752 and ~8% above 0.946) vs EPAI hard box [0.504, 0.892] (p_unanimous in [0.752, 0.946], ZERO mass outside). A fourth, previously undeployed lever: EPAI's CDF is pointwise BELOW RSD's on all of (0,2] (0.458t < 0.5t below 1.2; 0.55+0.375(t-1.2) < 0.5t above), so EPAI's gamma distribution STRICTLY STOCHASTICALLY DOMINATES: EPAI predicts a HIGHER inversion rate at EVERY flip point, plus an 18% density drop at the 1.2 kink.

ARCHITECTURE (13 experts): three cues at validity 1.00 (power weight exactly 1 for EVERY gamma -- gamma-invariant unit cues) and a spread singleton ladder 0.74/0.72/0.68/0.63/0.56/0.53 plus four 0.50 cues. Unanimous total weight S(gamma) >= 5.88 for all gamma in [0, 2.4], so the unanimous cells saturate every subject of both theories (beta*|E| >= 1.8*5.88 = 10.6), reading out A_eff = 2*(p_sparse - 0.5) to within 0.0002.

DISSOCIATION 1 -- ENGINEERED FLIP LADDER (headline; unlike prior mass-2-vs-uniform-block constructions, this uses a mass-1 unit cue against MIXED-VALIDITY PAIRS whose combined weight crosses exactly 1 at a designed flip point g*, giving a denser, precisely placed ladder). On each flip cell A is endorsed on one 1.00 cue (weight exactly 1 for all gamma) and B on a pair/block with W(g*) = 1; E(gamma) = W(gamma) - 1 is strictly decreasing with a single sign change at g*, so subjects with gamma < g* prefer the count-sparser unit-cue side and gamma > g* prefer the low-validity-denser side. Flip points: F1 {0.50,0.56} g*=1.09; F2 {0.50,0.63} g*=1.22; F3 {0.50,0.72} g*=1.43; F4 {0.63,0.72} g*=1.78; F5 {0.68,0.72} g*=1.95; F6 {0.50 x4} g*=2.00; F7 {0.72,0.74} g*=2.20. Predicted population inversion rates P(gamma > g*): RSD 0.455/0.385/0.285/0.110/0.025/0.000/0.000; EPAI 0.500/0.443/0.364/0.233/0.169/0.150/0.075. EPAI predicts MORE inverters at every rung (stochastic dominance, a one-sided ordering test across the whole ladder). HARD CELLS: on F6, E(gamma) = 4*0.5^gamma - 1 >= 0 for ALL gamma <= 2 (zero only at the measure-zero point gamma = 2), and on F7, E(2.0) = 0.72^2 + 0.74^2 - 1 = +0.066 > 0 with E strictly decreasing, so since m >= 0.74 > 0.5 makes p >= 0.5 whenever E >= 0, NO RSD subject can truly prefer the low-validity side on F6/F7 -- a hard floor at exactly 50%. EPAI-xp REQUIRES 15.0% of subjects below 0.5 on F6 (inversion depth: typical tail subject gamma~2.2 has E = -0.131, beta~3.7, p(unit side) ~ 0.42) and 7.5% on F7 (shallow, p ~ 0.47-0.49 -- acknowledged weak per subject, but population-detectable). Any subject credibly below 50% on F6/F7 categorically falsifies RSD-vgp; conversely a hard zero of inversions past g* = 2.0 where EPAI demands 15% (with N = 80, ~12 expected tail subjects) falsifies EPAI. Within-subject, inverting F6 implies inverting F5, F4, ... (monotone in gamma); per-subject gamma-hat from the 7 rungs (56 observations) traces the survival curve: RSD linear to a hard edge at 2.0 vs EPAI's smoother decline to 2.4 with 15% of gamma-hat mass in (2.0, 2.4]. The ladder also rules out pure counting: counting predicts the unit-cue (fewer-endorsement) side at ~100% on EVERY rung, flat, which neither theory produces.

DISSOCIATION 2 -- THE 1.2 KINK. F1 (1.09) and F2 (1.22) bracket EPAI's gamma-density drop (0.4583 -> 0.375 at 1.2, an 18% drop) while RSD's density is flat 0.5 throughout. The survival-curve slope (inversion-rate drop per unit flip-point interval) is 0.5 everywhere under RSD vs ~0.46 below 1.2 and ~0.375 above under EPAI: fit a piecewise-density survival curve with a break at 1.2. Honest caveat: the kink is a secondary, N-hungry signature (N >= 100 recommended); the tail cells F6/F7 and the dominance ordering are the primary weapons.

DISSOCIATION 3 -- BETA FLOOR VIA EXACT-E RATIO CELLS. B1/B2 use only unit cues with exactly canceling shared endorsements (a shared endorsement on the same cue contributes w*(1-1) = 0), giving E = -1 and -2 EXACTLY for every subject at every gamma, with both bars nonempty (conflict-style). The per-subject ratio r12 = (p_sparse,B2 - 0.5)/(p_sparse,B1 - 0.5) = (sigmoid(2*beta) - 0.5)/(sigmoid(beta) - 0.5) is exactly A_eff-free and lapse-free. RSD support: r12 in [1.005, 1.322] (beta in [1.8, 6]); EPAI: [1.003, 1.266] (beta in [2.0, 7.0]). The region r12 in (1.266, 1.322] (beta in [1.8, 2.0)) is RSD-EXCLUSIVE: 8.8% of RSD subjects, 0% of EPAI. EPAI's own exclusive beta region (6, 7] moves r12 only between 1.0025 and 1.005 -- below measurement resolution, honestly acknowledged as one-sided. Per-subject r12 from 8 observations per cell is noisy; the arbiter is hierarchical population fitting under each theory's beta prior.

DISSOCIATION 4 -- A_eff SUPPORT. The unanimous cells U/Um saturate every subject, reading out A_eff. RSD support for p_sparse on U: [0.714, 0.985] with ~12% of subjects below 0.752 and ~8% above 0.946 (m up to 1.0 with eps down to 0.03 permits near-deterministic subjects; m down to 0.74 permits weak ones); EPAI's hard box: [0.752, 0.946], exactly zero mass outside. Any subject with credibly p_U < 0.752 or > 0.946 falsifies EPAI-xp categorically; RSD requires ~20% of subjects in these forbidden zones. EPAI additionally predicts a NARROWER population spread of p_U (A_eff concentrated near 0.68-0.72) than RSD's wide continuum.

CONTROLS. C/Cm: a 2-2 endorsement-count tie (A endorsed on two 1.00 cues, B on two 0.50 cues): E = 2*0.5^gamma - 2 < 0 for all gamma > 0, so both theories prefer the low-validity-endorsed side (pooled ~0.75-0.82) while pure counting predicts exactly 50% -- ruling out count-based responding. T1/T2: identical non-trivial vectors -- both theories predict exactly 50% (key-bias check). Every informative cell has a mirrored twin (A/B swapped) so side bias cancels; analysis pools by unit-cue side (flip cells), sparse side (B-cells, U), or low-validity side (C). All informative trials carry distinct, non-trivially comparable rating vectors on both bars (locked-prompt requirement).

POWER AND ANALYSIS. 24 unique pairs x K = 4 = 96 trials exactly; 8 observations per pooled mirrored cell per subject. Pooled mid-box differences are modest and honestly acknowledged (approximate pooled p(unit-cue side): F1 ~0.51 RSD vs ~0.48 EPAI; F4 ~0.64 vs ~0.60; F6 ~0.73 vs ~0.69 -- the populations' central tendencies nearly coincide) because the decisive evidence is SUBJECT-LEVEL population structure. Requires N >= 80-100 subjects with hierarchical Bayesian fitting of per-subject (gamma, beta, A_eff) under each theory's population prior, plus simulation-based model recovery.

DECISIVE QUANTITATIVE SIGNATURES: (a) any subject credibly below 50% on F6 or F7 categorically falsifies RSD-vgp (E >= 0 over its entire gamma box; hard floor at 0.5), while EPAI requires 15%/7.5% of subjects there; (b) inversion-rate survival curve over g* = 1.09/1.22/1.43/1.78/1.95/2.00/2.20: RSD 0.455/0.385/0.285/0.110/0.025/0/0 vs EPAI 0.500/0.443/0.364/0.233/0.169/0.150/0.075 -- EPAI higher at EVERY rung (stochastic dominance) with a hard zero past 2.0 under RSD; (c) per-subject gamma histogram: flat density 0.5 with hard edge at 2.0 (RSD) vs 18% density drop at 1.2 and 15% of mass in (2.0, 2.4] (EPAI); (d) per-subject p_U outside [0.752, 0.946]: ~20% (RSD) vs 0% (EPAI); (e) per-subject r12 in (1.266, 1.322]: ~9% (RSD) vs 0% (EPAI); (f) within-subject monotone flip pattern (inverting F6 implies inverting F5, F4, F3...); (g) count-tie C preferring the low-validity side above 50% and identical-vector ties at exactly 50% (shared controls).

[14] STRUCTURAL SETUP. RSD-vgp and SGAI-r7 share the identical anti-endorsement core: p(A) = 0.5 + A_eff*(sigmoid(beta*E) - 0.5), E = sum_j v_j^gamma*(b_j - a_j), RAW weights, and both compute a single per-trial p that is then Bernoulli-sampled (SGAI's per-trial gamma_t jitter and both theories' m-mixing are absorbed into the marginal p, so trials are i.i.d. per cell under BOTH -- the jitter cannot mimic anything, it only blurs SGAI's own signatures). The theories differ ONLY in population supports: (i) GAMMA. RSD's two-component mixture is algebraically EXACTLY Uniform[0,2] (density 0.5 throughout). SGAI's habitual gamma is a 20% low-precision cluster ~U[0,1] plus an 80% majority with density 0.32 on [0.8,1.3], 0.96 on [1.3,1.7], 0.64 on [1.7,2.0], and 0.32 on (2.0,2.2] -- population density 0.2/[0,0.8], 0.52/[0.8,1.0], 0.32/[1.0,1.3], 0.96/[1.3,1.7], 0.64/[1.7,2.0], 0.32/(2.0,2.2]. SGAI's gamma distribution FIRST-ORDER STOCHASTICALLY DOMINATES RSD's (CDF below it everywhere on (0,2)), and SGAI places 6.4% of subjects above gamma=2.0 where RSD places ZERO. (ii) BETA. RSD: log-uniform [1.8,6] for every subject. SGAI: 80% the same, 20% low-precision log-uniform [1.2,1.8] -- beta < 1.8 is a hard RSD-impossible region holding 20% of SGAI's population. (iii) A_eff = (1-eps)(2m-1). RSD: m in [0.74,1.0], eps in [0.03,0.11] -> p_unanimous in [0.714,0.985] with ~8-9% of subjects OUTSIDE [0.725,0.966]. SGAI: m in [0.80,0.98] (majority eps [0.03,0.10], low-precision eps [0.15,0.25]) -> p_unanimous HARD-BOXED in [0.725,0.966], zero mass outside. The design makes all three boundaries decisive.

ARCHITECTURE (13 experts): three cues at validity 1.00 (power weight EXACTLY 1 for every gamma -- gamma-invariant unit cues), nine cues at 0.50 (weights halve per unit gamma, giving the exact ladder E(gamma) = k*0.5^gamma - 2), and one cue at 0.58. Shared endorsements on identical cue indices cancel exactly (w*(1-1)=0) and keep both bars nonempty.

DISSOCIATION 1 -- THE FLIP LADDER (headline). On rungs R1-R7, A is endorsed on the two unit cues (endorsement mass exactly 2 for all gamma) plus shared cues; B is endorsed on a block of k 0.50-cues (R7: seven 0.50s plus the 0.58 cue). E(gamma) = W_block(gamma) - 2 is strictly decreasing with a single sign change at g* = log2(k/2): gamma < g* prefers the count-sparser pair-endorsed side A; gamma > g* prefers the low-validity-dense side B. Flip points: R1 (k=3) 0.585; R2 (k=4) 1.000; R3 (k=5) 1.322; R4 (k=6) 1.585; R5 (k=7) 1.807; R6 (k=8) 2.000; R7 (7x0.50+0.58) 2.055. Predicted TRUE inversion rates P(gamma > g*): RSD 0.707/0.500/0.339/0.207/0.097/0/0 vs SGAI 0.883/0.736/0.619/0.366/0.188/0.064/0.046 -- SGAI higher at EVERY rung (stochastic dominance, a one-sided ordering test across the whole ladder). HARD CELLS: on R6, E(2.0)=0 and E>0 for all gamma<2, so since m>=0.74>0.5 keeps p>=0.5 whenever E>=0, NO RSD subject can truly prefer B; on R7, E(2.0)=+0.086>0 and E is strictly decreasing, so again NO RSD subject can prefer B. SGAI REQUIRES 6.4% of subjects below 0.5 on R6 (depth: gamma=2.1 gives E=-0.134, p~0.43) and 4.6% on R7 (gamma=2.2 gives E=-0.175, p~0.40). Any subject credibly below 50% on R6/R7 categorically falsifies RSD-vgp; a hard zero where SGAI demands ~5 subjects in N=80 falsifies SGAI-r7. The per-subject gamma histogram recovered from the 7 rungs (56 observations) is the primary signature: RSD = FLAT density 0.5 with a hard edge at 2.0; SGAI = lumpy with 2.5x depletion on [0,0.585] (11.7% vs 29.3%), a 1.9x peak on [1.322,1.585] (25.2% vs 13.2%), and 6.4% of mass in (2.0,2.2]. Within-subject, inverting R6 implies inverting R5, R4, ... (monotone in gamma). Approximate pooled mid-box predictions, p(pair-endorsed side), RSD vs SGAI: R1 0.38/0.26; R2 0.53/0.37; R3 0.58/0.46; R4 0.68/0.60; R5 0.75/0.66; R6 0.80/0.77; R7 0.79/0.75 -- RSD predicts systematically STRONGER sparse-side preference at every rung, a 0.08-0.16-wide separation no parameterization inside SGAI's box can close because it is driven by the gamma distributions themselves.

DISSOCIATION 2 -- BETA FLOOR VIA EXACT-E CELLS. B1/B2 use only unit cues with exactly canceling shared endorsements: E = -1 and -2 EXACTLY for every subject at every gamma. The per-subject ratio r12 = (p_sparse,B2 - 0.5)/(p_sparse,B1 - 0.5) = (sigmoid(2*beta)-0.5)/(sigmoid(beta)-0.5) is exactly A_eff-free, lapse-free, and gamma-free. RSD support: r12 in [1.005, 1.322] (beta in [1.8,6]). SGAI: 80% in the same support PLUS 20% low-precision subjects with r12 in [1.322, 1.552] (beta in [1.2,1.8]; r12(1.2)=1.552, r12(1.5)~1.46). The region r12 > 1.322 is SGAI-EXCLUSIVE: any subject credibly there categorically falsifies RSD, while SGAI REQUIRES 20% of subjects there. Conversely a hard zero of low-beta subjects falsifies SGAI's low-precision cluster. Pooled B1 differs only modestly (~0.80 RSD vs ~0.78 SGAI) -- honestly acknowledged; the arbiter is the hierarchical per-subject beta posterior with a 20% low-beta mixture component (Bayes factor + simulation-based recovery).

DISSOCIATION 3 -- A_eff SUPPORT (extremity box). U/Um (one bar empty, one unanimous) saturate every subject (|E| >= 5.8, beta >= 1.2 -> sigmoid >= 0.999), reading out p_U = 0.5 + A_eff/2 to within 0.001. RSD support [0.714, 0.985] with ~8-9% of subjects OUTSIDE [0.725, 0.966] (m down to 0.74 gives weak subjects; m up to 1.0 with eps down to 0.03 gives near-deterministic subjects). SGAI hard box [0.725, 0.966], exactly ZERO mass outside (m <= 0.98 and eps >= 0.03 cap the majority; the low-precision cluster's eps >= 0.15 caps it at 0.908). Any subject with credibly p_U < 0.725 or > 0.966 falsifies SGAI-r7; RSD requires ~20% of subjects in these zones collectively. The low-precision cluster additionally predicts 20% of subjects JOINTLY low-beta (r12 > 1.322), shallow-gamma (inverting at most the R1 rung), and capped extremity (p_U <= 0.908) -- a specific 3-way joint signature absent from RSD's independent-support population.

CONTROLS. C/Cm: a 4-4 endorsement-count tie (A endorsed on the unit pair, B on two 0.50 cues; shared fillers): E = 2*0.5^gamma - 2 < 0 for all gamma > 0, so both theories prefer the low-validity-endorsed side (pooled ~0.79-0.82) while pure counting predicts exactly 50% -- ruling out count-based responding. T1/T2: identical non-trivial vectors -- both theories predict exactly 50% (key-bias check). Every informative cell has a mirrored twin (A/B swapped) so side bias cancels; analysis pools by pair-endorsed side (rungs), sparse side (B-cells, U), or low-validity side (C). All informative trials carry nonempty, distinct bars on both sides, satisfying the locked-prompt comparison requirement.

NOTE ON DELTA. SGAI's per-trial gamma_t jitter (prob ~0.10, +/-0.4 around habitual gamma, capped at 2.2) is marginalized into each cell's i.i.d. Bernoulli p, so it produces NO supra-binomial trial-to-trial inconsistency in its own generative code; its only effect is a 10%-weighted smoothing of per-subject rung psychometrics toward the +/-0.4-gamma window average. This slightly blurs (never mimics) the support edges the design attacks; a secondary likelihood-ratio test of sharp-vs-smoothed per-subject rung psychometrics is reported but not relied upon.

POWER AND ANALYSIS. 24 unique pairs x K = 4 = 96 trials exactly; 8 observations per pooled mirrored cell per subject. Requires N >= 80 subjects with hierarchical Bayesian fitting of per-subject (gamma, beta, A_eff) under each theory's population prior, plus simulation-based model recovery. DECISIVE QUANTITATIVE SIGNATURES: (a) any subject credibly below 50% on R6 or R7 categorically falsifies RSD-vgp (E >= 0 over its entire gamma box), while SGAI requires 6.4%/4.6% of subjects there; (b) inversion-rate survival curve over g* = 0.585/1.000/1.322/1.585/1.807/2.000/2.055: RSD 0.707/0.500/0.339/0.207/0.097/0/0 vs SGAI 0.883/0.736/0.619/0.366/0.188/0.064/0.046 -- SGAI higher at every rung; (c) per-subject gamma histogram flat with hard edge at 2.0 (RSD) vs 2.5x depletion below 0.585, 1.9x peak on [1.32,1.59], 6.4% tail past 2.0 (SGAI); (d) per-subject r12 > 1.322 (beta-hat < 1.8): 0% hard (RSD) vs 20% low-precision cluster (SGAI); (e) per-subject p_U outside [0.725, 0.966]: ~8-9% (RSD) vs 0% hard box (SGAI); (f) within-subject monotone flip pattern (inverting R6 implies inverting R5...R1); (g) count-tie C preferring the low-validity side above 50% and identical-vector ties at exactly 50% (shared controls); (h) pooled rung ladder: RSD above SGAI at every rung by 0.08-0.16.

[15] STRUCTURAL SETUP. SGAI-r7 and RSD-vgp share the identical anti-endorsement core: p(A) = 0.5 + A_eff*(sigmoid(beta*E) - 0.5), E = sum_j v_j^gamma*(b_j - a_j), RAW weights. Per-cell trials are i.i.d. under both (SGAI's per-trial gamma_t jitter and both theories' m-mixing marginalize into the cell p). The theories differ ONLY in population structure, and this design attacks the three differences with a NEW set of weapons that the prior SGAI-vs-RSD design ([14]) did not deploy: (i) the LOW-PRECISION CLUSTER'S COUPLED (beta, gamma, A_eff) signature -- SGAI's 20% LP cluster jointly has beta in [1.2,1.8], gamma in [0,1], epsilon in [0.15,0.25], whereas RSD draws beta (log-uniform [1.8,6]), gamma (Uniform[0,2]) and (m, eps) INDEPENDENTLY; (ii) GAMMA-DENSITY PROFILOMETRY -- SGAI's majority gamma density is 0.32 on [0.8,1.3), 0.96 on [1.3,1.7), 0.64 on [1.7,2.0), 0.32 on (2.0,2.2] (population density including the LP cluster: 0.2 on [0,0.8), 0.52 on [0.8,1.0)), while RSD's is exactly flat 0.5 on [0,2]; (iii) the DELTA JITTER -- SGAI re-samples gamma_t ~ U[gamma-0.4, gamma+0.4] (capped [0,2.2]) on ~10% of trials, a within-subject boxcar smoothing in gamma-space that RSD has no mechanism to produce.

ARCHITECTURE (15 experts): three cues at validity 1.00 (power weight EXACTLY 1 for every gamma -- gamma-invariant unit cues), seven cues at 0.50, and five tuning cues (0.80, 0.75, 0.65, 0.60, 0.56). Shared endorsements on identical cue indices cancel exactly (w*(1-1)=0), keeping both bars nonempty on every informative trial (conflict-style; locked-prompt requirement satisfied).

DISSOCIATION 1 -- THE FLIP LADDER (7 rungs at SGAI's density breakpoints). On rung Lk, A is endorsed on a high-validity reference block and B on a low-validity block; E(gamma) = W_B(gamma) - W_A(gamma) is strictly decreasing with a single engineered sign change at flip point g*: gamma < g* subjects prefer the count-sparser reference side A; gamma > g* prefer the low-validity-dense side B. Engineered flip points (verified by direct computation of E): L1: A={1.00, 0.75} vs B={0.5 x3}, g*=0.73; L2: A={1.00, 0.56} vs B={0.5 x3}, g*=0.92; L3: A={1.00,1.00} vs B={0.5 x3, 0.65}, g*=1.12; L4: A={1.00,1.00} vs B={0.5 x4, 0.65}, g*=1.45; L5: A={1.00,1.00} vs B={0.5 x4, 0.80}, g*=1.62; L6: A={1.00,1.00} vs B={0.5 x5, 0.75}, g*=1.83; L7: A={1.00,1.00} vs B={0.5 x7, 0.60}, g*=2.08. Predicted true inversion rates P(gamma > g*): RSD 0.635/0.540/0.440/0.275/0.190/0.085/0.000 vs SGAI 0.854/0.816/0.698/0.496/0.333/0.173/0.042 (plus ~0.01-0.03 jitter spillover). SGAI is higher at EVERY rung (its gamma distribution first-order stochastically dominates RSD's). HARD TAIL CELL: on L7, E(2.0) = 7*0.25 + 0.36 - 2 = +0.11 > 0 and E is strictly decreasing, so NO RSD subject (gamma <= 2) can truly prefer the block side -- a hard floor at exactly 50% -- while SGAI requires ~4.2% of subjects (habitual gamma in (2.08, 2.2]) plus jitter spillover from majority subjects with gamma in [1.68, 2.0] (window reaches 2.2). Any subject credibly below 50% on L7 categorically falsifies RSD-vgp; a hard zero where SGAI demands ~5 subjects in N=100 falsifies SGAI-r7. Approximate pooled mid-box p(reference side), RSD vs SGAI: L1 0.44/0.34; L2 0.47/0.37; L3 0.53/0.44; L4 0.66/0.51; L5 0.70/0.63; L6 0.74/0.68; L7 0.77/0.73 -- honestly flagged as crude planning numbers; the arbiter is subject-level structure.

DISSOCIATION 2 -- CONDITIONAL (SLOPE x GAMMA x EXTREMITY) COUPLING (headline, new). The exact-E cells B1/B2 use only unit cues with exactly canceling shared endorsements, giving E = +1 and +2 EXACTLY for every subject at every gamma. The per-subject ratio r12 = (p_sparse,B2 - 0.5)/(p_sparse,B1 - 0.5) = (sigmoid(2*beta)-0.5)/(sigmoid(beta)-0.5) is exactly A_eff-free, lapse-free, and gamma-free: RSD support [1.005, 1.322] (beta in [1.8,6]); SGAI is 80% identical PLUS a 20% LP cluster with r12 in [1.322, 1.552] (beta in [1.2,1.8]). The NEW weapon is what the LP cluster's coupling predicts CONDITIONALLY: under SGAI, every subject with r12 > 1.322 is an LP subject with gamma <= 1.0 and epsilon >= 0.15, so (a) NO such subject can invert rung L4 or higher even WITH jitter (LP habitual gamma <= 1.0, jitter capped at gamma+0.4 <= 1.4 < g*_L4 = 1.45 -- a hard zero), (b) their saturated-cell extremity is capped at p_U <= 0.908, (c) their lapse is at least 0.15. Under RSD, beta is independent of gamma: NO subject has r12 > 1.322 at all, and the shallowest-slope quintile (beta in [1.8, 2.17]) still has gamma ~ Uniform[0,2] -- so ~27.5% of them invert L4, ~19% invert L5, ~8.5% invert L6, and ~25% show p_U > 0.91. Concrete pre-registered test: among the 20% shallowest subjects (hierarchical beta-hat, or per-subject r12), SGAI predicts L4/L5/L6 inversion at ~0% (hard for the LP cluster; small contamination from misclassified majority subjects) and p_U <= 0.908 for all; RSD predicts ~28%/19%/9% inversion and a wide p_U spread reaching 0.985. Any shallow-slope subject with p_U > 0.91 falsifies SGAI-r7; any subject at all with r12 > 1.33 falsifies RSD-vgp categorically while SGAI REQUIRES 20% there. Conversely a hard zero of low-beta subjects falsifies SGAI's LP cluster. This three-way joint signature (shallow slope AND low gamma AND capped extremity) cannot be produced by RSD's independent-support population in any proportion.

DISSOCIATION 3 -- GAMMA-DENSITY PROFILOMETRY. Adjacent rungs estimate the local gamma density from survival-curve drops: SGAI predicts segment densities of ~0.59 ([0.92,1.12]), ~0.61 ([1.12,1.45]), ~0.96 ([1.45,1.62] -- inside the 48%-majority peak on [1.3,1.7]), ~0.75 ([1.62,1.83]), ~0.77 ([1.83,2.08]) -- a 1.6x local BUMP on [1.45,1.62] relative to the [0.92,1.12] segment, a depleted low end (density 0.2 below 0.8), and continued survival past 2.0 (L7 inversion ~4-5%). RSD predicts a globally LINEAR survival curve (constant density 0.5, hard edge at 2.0). The rung placement -- bracketing SGAI's density breakpoints at 1.0, 1.3, 1.7, 2.0 from both sides -- is what makes the density shape identifiable, unlike prior widely-spaced ladders.

DISSOCIATION 4 -- A_eff SUPPORT EDGES. U/Um saturate every subject (|E| = S(gamma) >= 6.6 even at gamma=2.2, beta >= 1.2), reading out A_eff = 2*(p_sparse - 0.5) to within 0.001. RSD support for p_U: [0.714, 0.985] with ~5% of subjects above 0.966 (m -> 1, eps -> 0.03); SGAI hard box [0.725, 0.966], exactly ZERO mass outside (m <= 0.98, majority eps >= 0.03; LP eps >= 0.15 caps that cluster at 0.908). Any subject with credibly p_U > 0.966 or < 0.725 falsifies SGAI-r7; RSD requires ~5-8% of subjects in these zones. B1/B2 additionally identify (A_eff, beta) jointly per subject, so U tests the empty-bar heuristic (a 'pick the empty bar' strategy would push p_U toward 1.0, outside both boxes).

DISSOCIATION 5 -- THE DELTA JITTER (tertiary, new, within-subject). Because B1/B2 pin (beta, A_eff) independently of gamma, each subject's rung psychometric has, under RSD, a transition width in g*-space FULLY determined by (beta, A_eff) and the block slopes. SGAI's per-trial gamma_t re-sampling adds a 0.1-weighted BOXCAR smoothing of half-width 0.4 in gamma-space around each subject's habitual gamma -- a different functional form (flat ramp with kinks at +/-0.4, unchanged tails) than any beta-driven (logistic-in-E) flattening, which is already pinned by the beta cells. Population-level fit of a shared delta (SGAI: delta ~ 0.10 recovers; RSD: delta = 0). Honest power note: per-subject detectability is low (deviations <= ~0.05 in p); this is an exploratory signature requiring N >= 100, reported with simulation-based calibration, not relied upon.

CONTROLS. C/Cm (3-3 endorsement-count tie: A on the two non-shared unit cues, B on two 0.50 cues, one shared unit cue): E = 2*0.5^gamma - 2 < 0 for all gamma > 0, so both theories prefer the low-validity-endorsed side (pooled ~0.77-0.82) while pure counting predicts exactly 50% -- ruling out count-based responding. The rung ladder itself also rules out counting (counting predicts flat high p(reference side) on every rung) and conventional pro-endorsement polarity (predicts flat low). T1/T2 (identical non-trivial vectors): both theories predict exactly 50% -- key-bias check. Every informative cell has a mirrored twin (A/B swapped) so side bias cancels; analysis pools by reference side (rungs), sparse side (B-cells, U), or low-validity side (C).

POWER AND ANALYSIS. 24 unique pairs x K = 4 = 96 trials exactly; 8 observations per pooled mirrored cell per subject (7 rung cells, B1, B2, U, C, T). Requires N >= 100 subjects (the conditional test needs ~20 shallow-slope subjects) with hierarchical Bayesian fitting of per-subject (gamma, beta, A_eff) under each theory's population prior, plus simulation-based model recovery. DECISIVE QUANTITATIVE SIGNATURES: (a) any subject credibly below 50% on L7 falsifies RSD-vgp (E(2.0) = +0.11 > 0 over its entire gamma box), while SGAI requires ~4-5% of subjects there; (b) any subject with r12 > 1.33 falsifies RSD while SGAI requires 20%; (c) among shallow-slope subjects, L4/L5/L6 inversion ~0% and p_U <= 0.908 (SGAI, hard for the LP cluster since jitter caps LP gamma_t at 1.4 < 1.45) vs ~28%/19%/9% inversion and p_U up to 0.985 (RSD); (d) inversion-rate survival curve over g* = 0.73/0.92/1.12/1.45/1.62/1.83/2.08: RSD 0.635/0.540/0.440/0.275/0.190/0.085/0 (linear, hard edge at 2.0) vs SGAI 0.854/0.816/0.698/0.496/0.333/0.173/~0.05 (higher at every rung, density bump on [1.45,1.62], survival past 2.0); (e) per-subject p_U outside [0.725, 0.966]: ~5-8% (RSD) vs 0% hard (SGAI); (f) within-subject monotone flip pattern (inverting L5 implies inverting L4..L1); (g) count-tie C preferring the low-validity side above 50% and identical-vector ties at exactly 50% (shared controls); (h) population delta-hat ~ 0.10 with boxcar shape (SGAI) vs 0 (RSD). Pooled per-cell differences are modest (0.04-0.15) and honestly acknowledged: the populations' central tendencies nearly coincide and the decisive evidence is subject-level population structure, especially the LP cluster's three-way coupling, which no independent-support population can mimic in any mixture proportion.

[16] STRUCTURAL SETUP. RSD-vgp and HT-AEI-r6 share the identical anti-endorsement core: p(A) = 0.5 + A_eff*(sigmoid(beta*E) - 0.5), E = sum_j v_j^gamma*(b_j - a_j), RAW weights, m in [0.74-0.75, 1], epsilon in [0.03, 0.10-0.11] (A_eff boxes nearly identical, so extremity is NOT a usable lever). The theories are the SAME model with different population SUPPORTS, and only three support regions can ever dissociate them: (i) GAMMA GAP (1.35, 2.0]: RSD's two-component mixture is algebraically EXACTLY Uniform[0,2] (density 0.5 throughout), placing 32.5% of subjects in (1.35, 2.0]; HT-AEI places exactly ZERO there (its segments are [0,1], [1.05,1.35], [2.0,2.10]). (ii) GAMMA TAIL [2.0, 2.10]: HT-AEI's 35% heavy-tail segment lives entirely here; RSD's gamma is hard-capped at 2.0, so ZERO RSD subjects exist above 2.0. (iii) BETA FLOOR: RSD beta log-uniform [1.8, 6.0]; every HT-AEI segment has beta >= 2.2 (low [2.2,4.0], mid [2.2,3.2], tail [2.2,2.8]). Beta in [1.8, 2.2) holds 16.7% of RSD subjects (ln(2.2/1.8)/ln(6/1.8) = 0.1667) and 0% of HT-AEI subjects. HT's beta ceiling 4.0 vs RSD's 6.0 is acknowledged one-sided (saturation-limited). The design makes all three boundaries empirically decisive using EXACT algebra.

ARCHITECTURE (14 experts): three cues at validity 1.00 (power weight EXACTLY 1 for every gamma -- gamma-invariant unit cues), nine cues at 0.50 (weights halve per unit gamma), and two filler cues (0.72, 0.63) used ONLY in exactly-canceling shared endorsements or unanimous/tied positions, so they never perturb any evidence value. Shared endorsements on identical cue indices cancel exactly (w*(1-1) = 0), keeping both bars nonempty on every informative trial (conflict-style; locked-prompt requirement satisfied).

DISSOCIATION 1 -- THE EXACT FLIP LADDER (headline). Rungs R3-R8: option A is endorsed on the two non-shared unit cues (endorsement mass EXACTLY 2 for every gamma); option B is endorsed on k cues at validity 0.50. The evidence is then EXACTLY E(gamma) = k*0.5^gamma - 2, strictly decreasing with a single sign change at flip point g* = log2(k/2): gamma < g* subjects prefer the count-sparser unit-cue side A; gamma > g* prefer the low-validity-dense side B. Flip points: R3 (k=3) 0.585; R4 (k=4) 1.000; R5 (k=5) 1.322; R6 (k=6) 1.585; R7 (k=7) 1.807; R8 (k=8) 2.000. Predicted TRUE inversion rates P(gamma > g*): RSD 0.707 / 0.500 / 0.339 / 0.207 / 0.097 / 0.000 vs HT-AEI 0.766 / 0.600 / 0.373 / 0.350 / 0.350 / 0.350. Three signatures: (a) THE PLATEAU: HT-AEI's inversion rate is FLAT at 0.35 across R5-R8 (all its inverters there are tail subjects, gamma >= 2.0) and drops to zero only at 2.10, whereas RSD's uniform density forces a strictly LINEAR decline 0.34 -> 0.21 -> 0.10 -> 0 hitting a hard zero AT 2.0. RSD cannot produce a flat plateau at any parameterization; HT-AEI cannot put mass in the gap. (b) HARD FLOOR AT R8: E(gamma) = 8*0.5^gamma - 2 >= 0 for ALL gamma <= 2.0 (zero only at the measure-zero point gamma = 2), and m >= 0.74 > 0.5 keeps p >= 0.5 whenever E >= 0, so NO RSD subject can truly prefer the dense side on R8. HT-AEI REQUIRES its 35% tail there (E in [-0.134, 0), p as low as ~0.44 at gamma = 2.10, beta = 2.8 -- shallow per subject, population-detectable). (c) CONDITIONAL COUPLING: under HT-AEI every R6 inverter has gamma >= 2.0 (the gap is empty), so P(inverts R7 | inverts R6) = 1.0 and P(R8 rate < 0.5 | inverts R6) ~ 1; under RSD, P(inverts R7 | inverts R6) = (2-1.807)/(2-1.585) = 0.465 and P(R8 < 0.5 | inverts R6) = 0 HARD. Similarly P(inverts R7) itself: 35% (HT) vs 9.7% (RSD). (d) THE GAP HISTOGRAM: per-subject gamma recovered from the six rungs (48 observations) lands in bins [1.322,1.585)/[1.585,1.807)/[1.807,2.0] with total mass ~34% under RSD vs ~2.3% under HT-AEI (the sliver (1.322,1.35]); and above 2.0: 0% (RSD, hard) vs 35% (HT). Within-subject monotonicity (inverting R6 implies inverting R5..R3) holds under both and is a data-quality check. (e) SECONDARY SHAPE: HT-AEI's survival slope is ~0.40 below g*=1.0, ~0.70 on [1.0,1.322] (mid-segment density 0.833 vs RSD's 0.5) vs RSD's constant 0.50 -- a slope crossover, plus a 1.67x density bump on [1.05,1.35]. Depth among R7 inverters also differs (HT inverters all have gamma in [2,2.1], p ~ 0.35-0.42; RSD inverters have gamma in (1.807,2], p ~ 0.36-0.50, mean ~0.44 vs ~0.38).

DISSOCIATION 2 -- BETA FLOOR VIA EXACT-E CELLS. B1/B2 use only the unit cues with exactly canceling shared endorsements: E = +1 and +2 EXACTLY for every subject at every gamma (B1: A endorsed on unit cue 1, B on unit cues 2-3; B2: B on unit cues 2-3, unit cue 1 shared). The per-subject ratio r12 = (p_sparse,B2 - 0.5)/(p_sparse,B1 - 0.5) = (sigmoid(2*beta)-0.5)/(sigmoid(beta)-0.5) is exactly A_eff-free, lapse-free, and gamma-free. RSD support: r12 in [1.005, 1.322] (beta in [1.8,6]); HT-AEI: r12 <= 1.219 for ALL subjects (beta >= 2.2). The region r12 in (1.219, 1.322] is RSD-EXCLUSIVE: 16.7% of RSD subjects, 0% of HT-AEI. Any subject credibly there categorically falsifies HT-AEI; a hard zero there falsifies RSD's beta floor. Corroborating one-sided lever: among high-gamma subjects (R7/R8 inverters), HT-AEI's tail caps beta at 2.8 (r12 >= 1.121) while RSD's independent supports put ~63% of them above beta = 2.8 (r12 < 1.121) -- population-level among the ~10-35 high-gamma subjects. HT's beta ceiling 4.0 (vs RSD 6.0) moves r12 only between 1.005 and 1.037 -- below per-subject resolution, honestly acknowledged as one-sided.

CONTROLS. U/Um (one bar all-0, one all-1) saturate every subject (|E| = S(gamma) >= 5.9 even at gamma = 2.1 with beta >= 1.8), reading out A_eff = 2*(p_sparse - 0.5) to within 0.0002 -- required to normalize the beta probes. Uc/Ucm are conflict-style saturated twins (A endorsed on a single 0.50 cue, B on all 13 others; |E| >= 5.5): they verify the A_eff anchor is not an empty-bar artifact. C/Cm is a 5-5 endorsement-count tie (A endorsed on the two unit cues, B on two 0.50 cues, shared fillers): E = 2*0.5^gamma - 2 < 0 for all gamma > 0, so both theories prefer the low-validity-endorsed side (pooled ~0.75-0.85) while pure counting predicts exactly 50% -- ruling out count-based responding. T1/T2 are identical non-trivial vectors: both theories predict exactly 50% (key-bias check). Every informative cell has a mirrored twin (A/B swapped) so side bias cancels; analysis pools by unit-cue side (rungs), sparse side (B-cells, U, Uc), or low-validity side (C).

POOLED PREDICTIONS (approximate mid-box planning values, p(unit-cue side), RSD vs HT-AEI): R3 ~0.31/0.35 (crossover: HT's shallow low-segment inverters pull its R3 up); R4 ~0.52/0.46; R5 ~0.60/0.54; R6 ~0.69/0.61; R7 ~0.73/0.67; R8 ~0.75/0.70. Pooled differences are modest (0.04-0.08) and honestly acknowledged: the populations' central tendencies nearly coincide, and the decisive evidence is SUBJECT-LEVEL population structure. Requires N >= 100 subjects with hierarchical Bayesian fitting of per-subject (gamma, beta, A_eff) under each theory's population prior (RSD: gamma exactly U[0,2], beta log-U[1.8,6]; HT-AEI: the three-segment gamma mixture with segment-specific beta boxes), plus simulation-based model recovery.

DECISIVE QUANTITATIVE SIGNATURES: (a) any subject credibly below 50% on R8 categorically falsifies RSD-vgp (E >= 0 over its entire gamma box; hard floor at exactly 0.5), while HT-AEI requires 35% of subjects there; (b) inversion-rate survival curve over g* = 0.585/1.000/1.322/1.585/1.807/2.000: RSD 0.707/0.500/0.339/0.207/0.097/0 (linear, hard edge at 2.0) vs HT-AEI 0.766/0.600/0.373/0.350/0.350/0.350 (flat 0.35 plateau on [1.32, 2.0], cliff only at 2.10); (c) per-subject gamma histogram: RSD flat density 0.5 with hard edge at 2.0 vs HT-AEI's ~34%-vs-2.3% gap contrast on (1.322, 2.0] and 35%-vs-0% mass above 2.0, with a 1.67x bump on [1.05,1.35] and depletion on [0,1]; (d) P(inverts R7 | inverts R6) = 1.0 and P(R8 < 0.5 | inverts R6) ~ 1 (HT) vs 0.465 and 0 (RSD); (e) per-subject r12 in (1.219, 1.322] (beta-hat in [1.8, 2.2)): ~16.7% (RSD) vs 0% hard (HT-AEI); (f) among R7/R8 inverters, r12 >= 1.121 for all (HT tail beta <= 2.8) vs ~63% below 1.121 (RSD); (g) within-subject monotone flip pattern (inverting R6 implies inverting R5..R3); (h) count-tie C preferring the low-validity side above 50% and identical-vector ties at exactly 50% (shared controls).

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "STRUCTURAL SETUP. HT-AEI-r6 and RSD-vgp share the identical anti-endorsement core: p(A) = 0.5 + A_eff*(sigmoid(beta*E) - 0.5), E = sum_j v_j^gamma*(b_j - a_j), RAW weights, and nearly identical m/epsilon boxes (A_eff support is NOT a usable lever). The theories are the SAME model with different population SUPPORTS: (i) GAMMA. RSD's two-component mixture is algebraically EXACTLY Uniform[0,2] (density 0.5 throughout, hard edge at 2.0). HT-AEI is a three-segment density: 0.40 on [0,1] (40% low bulk), 0.833 on [1.05,1.35] (25% mid segment), 3.5 on [2.0,2.10] (35% heavy tail), with EMPTY gaps (1.0,1.05) and (1.35,2.0) and a hard cliff at 2.10. (ii) BETA. RSD: log-uniform [1.8,6] independent of gamma. HT: every segment has beta >= 2.2 (low [2.2,4.0], mid [2.2,3.2], tail [2.2,2.8]) -- beta in [1.8,2.2) holds 16.7% of RSD subjects and 0% of HT subjects, and the tail's beta is capped at 2.8. Unlike the prior HT-vs-RSD design ([16]: a coarse 6-rung ladder at 0.585/1.0/1.322/1.585/1.807/2.0 that could see the plateau and the 2.0 edge but NOT the internal structure of any segment), this design is a FINE-GRAINED GAMMA-DENSITY PROFILOMETER: 8 rungs spaced 0.04-0.26 apart that recover the entire piecewise density -- the mid-segment bump, the empty gap, the tail's internal uniform density, and the exact location of the 2.10 cliff.\n\nARCHITECTURE (32 experts): six cues at validity 1.00 (power weight EXACTLY 1 for every gamma -- a gamma-invariant reference block of weight exactly 6) and 24 cues at 0.50, plus two tuning cues (0.64, 0.76). On every rung trial A is endorsed on the six unit cues (endorsement mass exactly 6 for all gamma) and B on a block of low cues, so E(gamma) = W_block(gamma) - 6 is strictly decreasing with a single engineered sign change at flip point g*: gamma < g* subjects prefer A (the count-sparser side, 6 endorsements); gamma > g* prefer B. The W=6 architecture makes the NATURAL all-0.50 rung grid hit exactly the points this dissociation needs: 13 cues -> g*=1.1155 (inside HT's mid segment); 15 -> 1.3219 (top of mid segment); 18 -> 1.585 and 21 -> 1.807 (inside HT's empty gap); 24 -> 2.000 (the tail's lower edge). Two TUNED rungs probe the tail's interior: 23x0.50 + 0.64 -> g*=2.040; 23x0.50 + 0.76 -> g*=2.082. A beyond-cliff control: 24x0.50 + 0.76 -> g*=2.141, just past HT's 2.10 tail edge (E(2.10) = +0.16 > 0, so every subject of BOTH theories has p >= 0.5 there).\n\nDISSOCIATION 1 -- DENSITY PROFILOMETRY (headline). Because both gamma distributions are mixtures of uniforms, the population inversion rate at a rung equals the survival function S(g*) = P(gamma > g*), and adjacent rungs estimate LOCAL GAMMA DENSITY. Predicted inversion rates, HT vs RSD: R1 (1.1155): 0.545 / 0.442; R2 (1.3219): 0.373 / 0.339; R3 (1.585): 0.350 / 0.207; R4 (1.807): 0.350 / 0.097; R5 (2.000): 0.350 / 0.000; R6 (2.040): 0.210 / 0.000; R7 (2.082): 0.063 / 0.000; R8 (2.141): 0.000 / 0.000. Local densities (survival drop per unit flip-point): [1.1155,1.3219]: 0.833 (HT) vs 0.5 (RSD) -- the mid-segment 1.67x BUMP; [1.3219,1.585]: 0.087 vs 0.5; [1.585,1.807] and [1.807,2.0]: 0.000 vs 0.5 -- the EMPTY GAP (a 3-point FLAT plateau at exactly 0.35 across R3/R4/R5 is a signature RSD's uniform density cannot produce at any parameterization); [2.0,2.04] and [2.04,2.082]: 3.5 vs 0 -- the tail's internal density, ~7x RSD's maximum; [2.082,2.141]: cliff to 0 exactly at HT's 2.10 edge vs RSD's edge already at 2.0. Per-subject crossing points (the rung interval where a subject's psychometric crosses 0.5) give the full histogram: HT predicts 45.5% below 1.1155, 17.2% in (1.1155,1.3219], 2.3% in (1.3219,1.585], 0% in (1.585,2.0], and 35% in (2.0,2.141] (14.0% / 14.7% / 6.3% across the three tail sub-intervals -- a uniform 3.5-density fill); RSD predicts 55.8% / 10.3% / 13.2% / 20.8% / 0% -- a 15x contrast on the gap mass (2.3% vs 34.0%) and a hard 35%-vs-0% contrast above 2.0.\n\nDISSOCIATION 2 -- HARD TAIL CELLS AND THE CLIFF. On R5-R8, E(gamma) >= 0 for EVERY gamma <= 2.0 (R5: E = 24*0.5^gamma - 6 >= 0 with equality only at the measure-zero point gamma = 2; R6-R8 flip above 2.0), and m >= 0.74 > 0.5 keeps p >= 0.5 whenever E >= 0, so NO RSD subject can truly prefer B on any of R5-R8 -- a hard floor at exactly 50%. HT-AEI REQUIRES 35% of subjects below 0.5 on R5 (tail depth: E in [-0.40, 0], p in [0.39, 0.50], mean ~0.44), 21% on R6 (p ~ 0.44-0.50), and 6.3% on R7. The WITHIN-TAIL LINEAR DECLINE 35% -> 21% -> 6.3% -> 0% across [2.0, 2.141] is a quantitative fingerprint no RSD parameterization can approximate (its gamma is hard-capped at 2.0), and rung R8 locates the cliff: inversions at R8 falsify BOTH theories (HT's tail ends at 2.10; RSD's at 2.0), making the design adversarial in both directions. CONDITIONAL COUPLING: under HT, P(p(R5) < 0.5 | p(R4) < 0.5) = 1.0 (every R4 inverter is a tail subject); under RSD it is 0 HARD (no RSD subject can be below 0.5 on R5). Under HT, P(invert R6 | invert R5) = 0.60; under RSD the conditioning event has probability 0. Within-subject monotonicity (inverting a higher rung implies inverting all lower rungs) holds under both and is a data-quality check.\n\nDISSOCIATION 3 -- BETA SUPPORT VIA EXACT-E CELLS. B1/B2 use only unit cues with exactly canceling shared endorsements (shared endorsements on identical cue indices contribute w*(1-1) = 0), giving E = +1 and +2 EXACTLY for every subject at every gamma. The per-subject ratio r12 = (p_sparse,B2 - 0.5)/(p_sparse,B1 - 0.5) = (sigmoid(2*beta)-0.5)/(sigmoid(beta)-0.5) is exactly A_eff-free, lapse-free, and gamma-free. RSD support: r12 in [1.005, 1.322] (beta in [1.8,6]); HT-AEI: r12 <= 1.219 for ALL subjects (beta >= 2.2 in every segment). The region r12 in (1.219, 1.322] is RSD-EXCLUSIVE: 16.7% of RSD subjects, 0% of HT -- any subject credibly there categorically falsifies HT-AEI; a hard zero there falsifies RSD's beta floor. SEGMENT-COUPLED BETA (new readout on this design's deep rungs): among R4/R5 inverters, HT-AEI's tail caps beta at 2.8 (r12 >= 1.121 for all of them), whereas RSD draws beta independently, so ~63% of its high-gamma subjects have beta > 2.8 (r12 < 1.121) -- a population-level split of the inverter group that HT's coupled supports forbid. HT's beta ceiling of 4.0 (vs RSD's 6.0) moves r12 only between 1.005 and 1.037, below per-subject resolution -- honestly acknowledged as one-sided and not relied upon.\n\nPOOLED PREDICTIONS (approximate mid-box planning values, p(A-side), HT vs RSD): R1 ~0.47/0.47 (the low/mid mass and inverter depths nearly cancel -- R1 is a shape probe, not a pooled probe; honestly acknowledged); R2 ~0.57/0.64; R3 ~0.61/0.72; R4 ~0.63/0.76; R5 ~0.70/0.80; R6 ~0.72/0.79; R7 ~0.73/0.82; R8 ~0.78/0.83; B1 ~0.80/0.80; U ~0.85/0.85; ties 0.50/0.50. Pooled differences of 0.05-0.13 on R2-R7 are corroborating; the decisive evidence is SUBJECT-LEVEL population structure.\n\nCONTROLS. U/Um (all-0 vs all-1, mirrored) saturate every subject (|E| = S(gamma) >= 12.5 at gamma = 2.1, beta >= 1.8), reading out A_eff = 2*(p_sparse - 0.5) to within 0.0002 -- required to normalize the beta probes and rung psychometrics. Uc is a CONFLICT-STYLE saturated twin with the OPPOSITE sparse side (A endorsed on 31 cues, B on a single 0.50 cue; |E| >= 11): p_sparse(Uc) must equal p_sparse(U), which detects any 'pick the empty bar' heuristic while its reversed side assignment cancels side bias against U. T1 (identical non-trivial vectors) verifies both theories' exact-50% prediction and catches key bias. Every rung and beta cell has a mirrored twin (A/B swapped) so side bias cancels; analysis pools by unit-cue side (rungs) or sparse side (B-cells, U, Uc). The ladder itself rules out pure counting (counting predicts flat ~100% preference for the 6-endorsement side on every rung) and conventional pro-endorsement polarity (flat ~0%). All informative trials carry nonempty, distinct bars on both sides, satisfying the locked-prompt comparison requirement.\n\nPOWER AND ANALYSIS. 24 unique pairs x K = 4 = 96 trials exactly; 8 observations per pooled mirrored cell per subject; 64 rung observations per subject. Requires N >= 100 subjects with hierarchical Bayesian fitting of per-subject (gamma, beta, A_eff) under each theory's population prior (RSD: gamma exactly U[0,2], beta log-U[1.8,6] independent; HT-AEI: the three-segment gamma mixture with segment-coupled beta boxes), plus simulation-based model recovery.\n\nDECISIVE QUANTITATIVE SIGNATURES: (a) any subject credibly below 50% on R5, R6, or R7 categorically falsifies RSD-vgp (E >= 0 over its entire gamma box; hard floor at exactly 0.5), while HT-AEI requires 35%/21%/6.3% of subjects there; (b) inversion-rate survival curve over g* = 1.1155/1.3219/1.585/1.807/2.000/2.040/2.082/2.141: HT 0.545/0.373/0.350/0.350/0.350/0.210/0.063/0 (flat 0.35 plateau on [1.35,2.0], linear 3.5-density decline inside the tail, cliff at 2.10) vs RSD 0.442/0.339/0.207/0.097/0/0/0/0 (constant 0.5 density, hard edge at 2.0); (c) per-subject crossing-point histogram: gap mass (1.3219,2.0] at 2.3% (HT) vs 34.0% (RSD), tail mass (2.0,2.141] at 35% (HT) vs 0% hard (RSD), mid-bump density 0.833 vs 0.5 on [1.1155,1.3219]; (d) P(p(R5) < 0.5 | p(R4) < 0.5) = 1.0 (HT) vs 0 hard (RSD); (e) per-subject r12 in (1.219, 1.322]: ~16.7% (RSD) vs 0% hard (HT-AEI); (f) among R4/R5 inverters, r12 >= 1.121 for all (HT tail beta <= 2.8) vs ~63% below 1.121 (RSD independent supports); (g) within-subject monotone flip pattern (inverting R5 implies inverting R4..R1); (h) inversions at R8 (g* = 2.141) falsify BOTH theories -- HT's tail edge is at 2.10, RSD's at 2.0; (i) U = Uc in sparse-side rate (empty-bar control) and identical-vector ties at exactly 50% (key-bias control).",
  "validities": [
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.64,
    0.76
  ],
  "trial_a_ratings": [
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
      1,
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
      1,
      1,
      1,
      1,
      1,
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
      1,
      1,
      1,
      0,
      0,
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
      1,
      0,
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
      1,
      1,
      1,
      1,
      0,
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
      1,
      1,
      1,
      1,
      0,
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
      1,
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
      0,
      0,
      0,
      1,
      0
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
      1,
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
      1,
      1,
      1,
      1,
      1,
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
      0,
      1
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
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
      0,
      0,
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
      1,
      0,
      0,
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
      1,
      1,
      1,
      1,
      0,
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
      1,
      0,
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
      0,
      0,
      0,
      1,
      0
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 36271,
  "output_tokens": 57835,
  "reasoning_tokens": 49451,
  "provider": "Sail Research"
}
```
