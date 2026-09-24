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
**Description:** Sparse-Option Preference / Polarity-Inverted Evidence Integration (SPI). When choosing between two options described by binary expert ratings, a substantial share of subjects systematically mis-map cue polarity: an endorsement (rating = 1) is treated as evidence AGAINST an option, so the option with FEWER 1s (the sparser option) is perceived as higher quality — consistent with a 'fewer red flags' reading of the display or a rarity-implies-quality inference. Subjects integrate ALL cues additively with near-uniform weights: the evidence for option A over B is E = sum_j w_j * (b_j - a_j), where w_j = v_j^gamma with gamma small (validity plays at most a mild role). Choice is a softmax over [E, -E] with inverse temperature beta, plus an independent lapse epsilon to uniform. A per-subject polarity-mixing weight rho captures heterogeneity: with probability 1-rho the subject responds according to the inverted (sparse-preferring) polarity, with probability rho according to the conventional polarity. This mixture attenuates the extreme pure-sparse predictions toward the moderately negative values observed. The theory is sharply distinguished from both incumbents: unlike Take-The-Best it uses all cues and inverts the evidence sign (so it is not pinned at ~0 on the depth/conflict-invariance and margin-slope metrics where TTB is structurally stuck), and unlike power-weighted integration it places humans at the FLAT end of the weighting ladder with inverted polarity, producing the negative signatures that steep-gamma PWEI cannot generate at any parameter setting.

**Parameters:**
- gamma: [0, 2]
- beta: [0.5, 20]
- epsilon: [0.05, 0.35]
- rho: [0, 0.35]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Sparse-option Preference / Polarity-Inverted Evidence Integration (SPI).
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are 0/1 cue values. History is ignored
    # (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"SPI expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    rho = float(parameters["rho"])

    # Near-uniform validity weighting: w_j = v_j^gamma, gamma in [0, 2].
    # Normalization is a per-experiment constant rescale (validities are
    # fixed), absorbed by beta; it keeps the evidence scale O(1) for any
    # gamma and any n_features.
    w = np.power(val, gamma)
    w_sum = w.sum()
    if w_sum > 0:
        w = w / w_sum
    else:
        w = np.full(n_features, 1.0 / n_features)

    a, b = stim[0], stim[1]
    # Polarity-INVERTED evidence for A over B: each cue where B carries
    # the endorsement (1) counts FOR A; each cue where A carries the
    # endorsement counts AGAINST A. Sparse options accumulate evidence.
    E = float(np.sum(w * (b - a)))

    # Softmax over [E, -E] with max-subtraction for numerical stability.
    # Inverted polarity: positive E favors A (the sparser option).
    scores_inv = np.array([E, -E])
    z = beta * (scores_inv - scores_inv.max())
    e = np.exp(z)
    p_inv = e / e.sum()

    # Conventional polarity (endorsement favors the option carrying it):
    # mirror of the inverted distribution.
    scores_norm = -scores_inv
    z2 = beta * (scores_norm - scores_norm.max())
    e2 = np.exp(z2)
    p_norm = e2 / e2.sum()

    # Per-subject polarity mixture: (1 - rho) inverted, rho conventional.
    p_core = (1.0 - rho) * p_inv + rho * p_norm

    # Independent lapse to uniform choice.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # Guard against float drift.
    return np.random.choice(len(probs), p=probs)
```


## COMPETING THEORY
**Description:** Red-Flag Integration with Trait Polarity and Attentional Slip (RFI-delta). When choosing between two products described by binary expert ratings, a large majority of subjects read a rating of 1 as a 'red flag' — evidence AGAINST the option carrying it — so the sparser option is perceived as higher quality; a small minority reads polarity conventionally. Polarity is a stable PER-SUBJECT trait: subject i carries a fixed sign s_i ∈ {+1, -1} drawn once, with P(s_i = -1) = pi at the population level (pi near the arbiter's ceiling, ~0.94). On top of the trait, each subject occasionally suffers a small attentional slip: on any given trial, with probability delta (≈0.10), the subject momentarily reverts to the opposite (instructed-mapping) polarity for that trial alone. The slip is structured exactly like a lapse — a transient per-trial reversion against a stable dominant trait — not a reopening of per-trial polarity mixing, and it is kept small (delta ≤ 0.12). Within a trial, all cues are integrated additively with near-flat validity weighting: evidence for A over B is E = s · Σ_j w_j (a_j − b_j), w_j ∝ v_j^gamma. Choice is a softmax over [E, −E] with per-subject inverse temperature beta plus an independent lapse epsilon to uniform; identical rating vectors yield exactly 50%. The slip attenuates each subject's effective extremity by (1 − 2·delta) and the population's effective anti-endorsement signal to q_eff = pi(1−delta) + (1−pi)delta ≈ 0.85, which simultaneously pulls the pooled point estimates back from saturation toward the observed moderate magnitudes and compresses the between-subject mixture variance 4·pi(1−pi)·((1−2·delta)·x)^2 toward the tight real values — while preserving the falsifiable subject-level bimodality signature (two attenuated polarity clusters, not one homogeneous middle).

**Parameters:**
- pi: [0.94, 0.95]
- polarity_u: [0, 1]
- delta: [0.08, 0.12]
- gamma: [0, 1]
- beta: [3, 8]
- epsilon: [0, 0.08]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Red-Flag Integration with Trait Polarity + attentional slip (RFI-delta).
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are 0/1 cue values. History is ignored
    # (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"RFI expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    pi = float(parameters["pi"])
    u = float(parameters["polarity_u"])
    delta = float(parameters["delta"])

    # Trait polarity: drawn ONCE per subject (both parameters are fixed
    # for the entire subject run). s = -1 -> red-flag reading (rating of 1
    # is evidence AGAINST the carrying option); s = +1 -> conventional
    # reading (endorsement favors the carrying option). At the population
    # level P(s = -1) = E[pi] with pi in [0.94, 0.95], i.e. an
    # anti-endorsement majority near the arbiter's ceiling and a small
    # (~5%) conventional minority.
    s = -1.0 if u < pi else 1.0

    # Near-flat validity weighting: w_j = v_j^gamma, gamma in [0, 1].
    # Normalization is a per-experiment constant rescale (validities are
    # fixed), absorbed by beta; it keeps the evidence scale O(1) for any
    # gamma and any n_features.
    w = np.power(val, gamma)
    w_sum = w.sum()
    if w_sum > 0:
        w = w / w_sum
    else:
        w = np.full(n_features, 1.0 / n_features)

    a, b = stim[0], stim[1]
    # Signed additive evidence for A over B. With s = -1 each endorsement
    # carried by A counts AGAINST A and each endorsement carried by B
    # counts FOR A (the sparser option accumulates evidence).
    E = s * float(np.sum(w * (a - b)))

    # Softmax over [E, -E] with max-subtraction for numerical stability.
    # Identical rating vectors give E = 0 -> exactly 50/50.
    scores = np.array([E, -E])
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_trait = e / e.sum()

    # Attentional slip: with probability delta the subject momentarily
    # reverts to the OPPOSITE polarity mapping for this trial alone.
    # The opposite-polarity distribution is the mirror of p_trait, so the
    # per-trial choice distribution is a (1-delta)/delta mixture of the
    # two mirrors. This is the expectation of a per-trial Bernoulli slip
    # against a stable dominant trait (lapse-structured, small delta),
    # NOT per-trial polarity mixing: the trait sign is still fixed per
    # subject and the slip only attenuates each subject's extremity by
    # (1 - 2*delta).
    p_slip = p_trait[::-1]
    p_core = (1.0 - delta) * p_trait + delta * p_slip

    # Independent lapse to uniform choice.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
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

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "ADVOCATED: SPI. COMPETITOR: RFI-delta. The two theories are nearly observationally equivalent on any single pooled cell: both compute p(sparse) = (1-eps)*[(1-m)*sigmoid(beta*E) + m*(1-sigmoid(beta*E))] + eps/2 over the same near-flat weighted sparsity evidence E. The dissociation therefore lives entirely in their PARAMETER-BOX and POPULATION-ARCHITECTURE constraints, and the design is built to make those box boundaries empirically decisive. (1) RFI's per-subject box is narrow: beta in [3,8], epsilon in [0,0.08], delta in [0.08,0.12], gamma in [0,1], with a fixed per-subject polarity trait and a bimodal population (94-95% anti-endorsement majority, ~5% conventional minority). SPI's per-subject box is wide: beta in [0.5,20], epsilon in [0.05,0.35], rho in [0,0.35], gamma in [0,2], continuous population. Consequences exploited: (A) EXTREMITY FLOOR: on E~1 (unanimous) cells, an RFI majority subject must show p(sparse) >= 0.816 (min at beta=3, delta=0.12, eps=0.08) and <= ~0.92 (delta+eps cap); the RFI minority sits at <= 0.19; the band (0.2, 0.8) is EMPTY at the subject level, and RFI's POOLED p(sparse) on extreme cells is boxed in [0.78, 0.88]. SPI's central regime (e.g., gamma=0.5, beta=4, eps=0.20, rho=0.15) predicts p(sparse) ~= 0.77 on E=1 -- below RFI's per-subject floor and pooled floor -- and SPI's box places most subjects in 0.55-0.80, exactly RFI's forbidden middle. (B) WITHIN-SUBJECT SLOPE FLOOR: because RFI mandates beta >= 3, every RFI subject must show a rise of at least ~0.19 from the low-E cell (E~0.10-0.125) to the unanimous cell (E=1) (min rise ~0.19 at beta=8, delta=0.12, eps=0.08; typical ~0.25-0.28). SPI with beta <= 1.5 predicts rises of 0.06-0.15 -- shallow, attenuated psychometrics that RFI cannot generate for any subject. (C) GAMMA CEILING: RFI hard-caps gamma at 1; SPI allows gamma up to 2. Two validity-vs-count conflict cells (G1, G2) are constructed so that the count-sparser option's weighted evidence E is sign-robustly POSITIVE for all gamma in [0,1] (RFI must prefer the count-sparser option, p in [0.52, 0.76]) but NEGATIVE at gamma=2 (E = -0.096 and -0.121), so SPI subjects with gamma > ~1.3-1.55 prefer the count-DENSER option. Any observed p(count-sparser) < 0.5 on G1/G2 falsifies RFI while SPI survives. DESIGN: 8 cues with a wide, well-separated validity ladder [0.98 ... 0.50]. The E-ladder uses nested endorsement sets (A endorsed on cue 5; {5,6}; {4,5,6}; {3,4,5,6}; {2..6}; {2..7}; {1..7}; all 8) against an empty or low-tied opponent, so the rung ordering of E is invariant for EVERY gamma in [0,2] (E = .125/.116/.103; .125/.102/.080; .250/.218/.182; .375/.352/.317; .500/.497/.477; .625/.654/.665; .875/.915/.945; 1 -- values at gamma=0/1/2). Single mid-validity-cue differences (R1, R1c) have E ~= 0.08-0.125 nearly independent of gamma (weight normalization cancels), giving a gamma-robust low-E anchor; the 7-vs-0 and 8-vs-0 cells give gamma-robust extreme anchors. R1c and R5c are 'conflict-style' twins (both options carry endorsements; tied cues cancel exactly) so the sparse preference is not reducible to 'pick the empty bar'. Every informative cell has a mirrored twin (A/B swapped) so side bias cancels; analysis pools by sparse-option side. T0/T1 are identical-vector ties where both theories predict exactly 50% (key-bias check). 24 unique pairs x K=4 = 96 trials exactly; per subject this yields 16 observations on the pooled extreme cells (R7/R8 + mirrors) and 16 on the pooled low-E cells (R1/R1c + mirrors), enough (with N >= 60 subjects and a hierarchical mixture analysis) to locate each subject in the (p_low, p_high) plane. PREDICTED OUTCOMES -- if SPI is true: a CONTINUOUS per-subject distribution of p_high (sparse choice rate on extreme cells) with substantial mass in 0.55-0.80, i.e., inside RFI's forbidden middle band; per-subject psychometric rises spanning 0.05-0.30 (many below 0.19); pooled p_high on extreme cells ~0.70-0.80 (below RFI's 0.78 floor) or, if subjects are consistently sparse, possibly > 0.88 (above RFI's cap); no subject below 0.5 on positive-E cells (SPI's p_core >= 0.5 for E >= 0 since rho <= 0.35); G1/G2 can go below 0.5 for steep-gamma subjects. If RFI-delta is true: a BIMODAL subject population -- a dominant cluster with p_high in [0.82, 0.92], a ~5% cluster at p_high <= 0.19, an EMPTY band (0.2, 0.8); EVERY subject showing rise >= ~0.19 between low-E and extreme cells; pooled p_high on extreme cells within [0.78, 0.88]; p(count-sparser) > 0.5 on both gamma probes; ties at exactly 50%. DECISIVE QUANTITATIVE SIGNATURES: (a) population mass in the (0.25, 0.75) band of per-subject p_high (SPI) vs empty band with two tight clusters (RFI); (b) per-subject rise p_high - p_low < 0.19 for any subject (falsifies RFI, fits low-beta SPI); (c) pooled p_high outside [0.78, 0.88] (falsifies RFI in either direction); (d) p(count-sparser) < 0.5 on G1/G2 (falsifies RFI's gamma <= 1; SPI's gamma up to 2 survives); (e) any single subject with p_high < 0.45 on E >= 0.875 cells (falsifies SPI's 0.5 floor, supports RFI's conventional minority); (f) tie trials at 50% (shared control). The single-cue-difference and conflict trials genuinely require comparing the two rating vectors, satisfying the locked-prompt display requirement.",
  "validities": [
    0.98,
    0.92,
    0.85,
    0.78,
    0.68,
    0.6,
    0.5,
    0.5
  ],
  "trial_a_ratings": [
    [
      0,
      0,
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
      0
    ],
    [
      0,
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
      1
    ],
    [
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
      0
    ],
    [
      0,
      0,
      0,
      0,
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
      0,
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
      1
    ],
    [
      0,
      0,
      0,
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
      0
    ],
    [
      0,
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
      1,
      1
    ],
    [
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
      1
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 10395,
  "output_tokens": 45627,
  "reasoning_tokens": 42678,
  "provider": "Sail Research"
}
```
