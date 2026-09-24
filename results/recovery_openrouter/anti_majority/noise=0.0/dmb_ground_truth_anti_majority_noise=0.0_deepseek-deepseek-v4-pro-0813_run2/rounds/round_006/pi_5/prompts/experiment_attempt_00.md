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
**Description:** Compressive Negative Validity-Weighted integration with subject-level saturation (CNVW-tau). Each subject forms D = sum_j v_j^gamma_i * (A_j - B_j), where v_j are the instructed expert validities and gamma_i is a subject-level validity exponent. The weighted difference is passed through a strongly saturating bounded transform psi(D) = D / (tau_i + abs(D)), with a small subject-level half-saturation constant tau_i. Most subjects apply a negative sign, S_i = -lambda_i * psi(D) + bias_i, while a minority apply a positive sign. Choice probability is p(A) = epsilon_i/2 + (1 - epsilon_i) * sigmoid(beta_i * S_i). There is no feature-tally term and no conflict-gating interaction.

**Parameters:**
- gamma: [1.5, 2.5]
- tau: [0.10, 0.25]
- lambda: [0.8, 2.0]
- beta: [0.8, 2.0]
- bias: [-0.25, 0.25]
- epsilon: [0.03, 0.10]
- p_positive: [0.03, 0.10]
- sign_draw: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
import numpy as np

def predict(parameters, state, history):
    """Compressive negative validity-weighted integration with subject-level tau.

    D = sum_j v_j^gamma * (A_j - B_j)
    psi(D) = D / (tau + abs(D))
    S = sign * lambda * psi(D) + bias
    p(A) = lapse/2 + (1-lapse) * sigmoid(beta * S)

    Most subjects use sign = -1; a minority draw sign = +1.
    history is intentionally ignored because there is no feedback.
    """
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            "CNVW-tau expects a (2, n_features) stimulus; "
            f"got shape {stim.shape}."
        )

    a = stim[0]
    b = stim[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.size != a.size:
        raise ValueError(
            f"validities length {v.size} does not match n_features {a.size}."
        )

    gamma = float(parameters["gamma"])
    lam = float(parameters["lambda"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    bias = float(parameters["bias"])
    p_positive = float(parameters["p_positive"])
    sign_draw = float(parameters["sign_draw"])
    tau = float(parameters["tau"])

    # Latent sign mixture: most subjects use negative-valence weighting;
    # a minority trust positive-valence evidence and get a positive sign.
    sign = 1.0 if sign_draw < p_positive else -1.0

    # Validity-weighted feature difference.
    d = float(np.dot(v ** gamma, a - b))

    # Subject-level strongly saturating bounded transform. The small
    # half-saturation constant tau forces most non-negligible validity-
    # weighted differences toward the sign bound, so sign and not
    # continuous magnitude drives the main choice contrast.
    psi = d / (tau + abs(d))

    score = sign * lam * psi + bias
    z = beta * score

    # Numerically stable sigmoid for p(A | score).
    if z >= 0.0:
        p_a_core = 1.0 / (1.0 + np.exp(-z))
    else:
        ez = np.exp(z)
        p_a_core = ez / (1.0 + ez)

    # Uniform lapse mixture over the two options.
    p_a = (1.0 - epsilon) * p_a_core + epsilon / 2.0
    p_a = float(np.clip(p_a, 0.0, 1.0))

    return np.array([p_a, 1.0 - p_a], dtype=float)
```

**`policy source code`:**
```python
import numpy as np

def policy(probs):
    p = np.asarray(probs, dtype=float)
    if p.sum() <= 0.0:
        p = np.ones_like(p) / len(p)
    else:
        p = p / p.sum()
    return int(np.random.choice(len(p), p=p))
```


## COMPETING THEORY
**Description:** Thresholded anti-validity utility with Gaussian threshold noise and a small validity-consistent soft leak (TAVU-tn). Subjects compute a validity-weighted difference D = sum_j v_j^gamma_i * (A_j - B_j). A subject-level latent sign dominates: most subjects use negative weighting and make the anti-validity choice when |D| exceeds a subject-level threshold tau_i, but threshold placement is noisy across trials, so the aggregate threshold boundary is graded rather than a step. Within the dead zone and below the sign-following ceiling, subjects fall back on a subject-level bias. A small plain-D sigmoidal leak in the validity-consistent direction weakly damps extreme anti choices at large |D|. A uniform lapse adds residual noise. The dominant process remains thresholded sign, not continuous magnitude accumulation.

**Parameters:**
- gamma: [1.8, 2.2]
- tau: [0.16, 0.19]
- tau_sd: [0.02, 0.04]
- p_negative: [0.88, 0.94]
- sign_draw: [0.0, 1.0]
- bias: [0.46, 0.54]
- lapse: [0.12, 0.16]
- soft_weight: [0.00, 0.03]
- soft_beta: [0.7, 1.1]
- follow_max: [0.66, 0.74]
- validities: validities

**`predict source code`:**
```python
import numpy as np
import math


def predict(parameters, state, history):
    """Thresholded anti-validity utility with Gaussian threshold noise (TAVU-tn).

    D = sum_j v_j^gamma * (A_j - B_j). A subject-level sign dominates:
    most subjects apply a negative validity weighting and choose the
    anti-validity option when |D| exceeds a subject-level threshold.
    Threshold placement is noisy across trials, implemented analytically
    as p_above = Phi((|D| - tau) / tau_sd). Inside the dead zone, and
    below the sign-following ceiling, choices fall back on a subject-level
    bias. A small plain-D sigmoidal leak weakly opposes the anti-validity
    sign at large |D|, and a uniform lapse adds residual noise.
    """
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            "TAVU-tn expects a (2, n_features) stimulus; "
            f"got shape {stim.shape}."
        )

    a = stim[0]
    b = stim[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.size != a.size:
        raise ValueError(
            f"validities length {v.size} does not match n_features {a.size}."
        )

    gamma = float(parameters["gamma"])
    tau = float(parameters["tau"])
    tau_sd = float(parameters["tau_sd"])
    p_negative = float(parameters["p_negative"])
    sign_draw = float(parameters["sign_draw"])
    bias = float(parameters["bias"])
    lapse = float(parameters["lapse"])
    w_soft = float(parameters["soft_weight"])
    beta_soft = float(parameters["soft_beta"])
    follow_max = float(parameters["follow_max"])

    # Latent validity-sign mixture: most subjects are negative weighters.
    sign = -1.0 if sign_draw < p_negative else 1.0

    # Validity-weighted feature difference.
    d = float(np.dot(v ** gamma, a - b))

    # Analytic Gaussian threshold noise: probability that |D| crosses tau
    # when the trial-level threshold is perturbed.
    z_th = (abs(d) - tau) / tau_sd
    p_above = 0.5 * (1.0 + math.erf(z_th / math.sqrt(2.0)))
    p_above = float(np.clip(p_above, 0.0, 1.0))

    # Thresholded anti-validity target for p(A).
    if sign < 0.0:
        if d > 0.0:
            target_a = 0.0
        elif d < 0.0:
            target_a = 1.0
        else:
            target_a = bias
    else:
        if d > 0.0:
            target_a = 1.0
        elif d < 0.0:
            target_a = 0.0
        else:
            target_a = bias

    # Dominant thresholded sign rule, with a follow ceiling that keeps
    # some residual bias even after threshold crossing.
    core_threshold = (
        follow_max * (p_above * target_a + (1.0 - p_above) * bias)
        + (1.0 - follow_max) * bias
    )

    # Small validity-consistent soft leak. This is deliberately very weak
    # below |D| = 0.2 and only mildly damps extreme anti choices.
    z = beta_soft * d
    if z >= 0.0:
        soft_a = 1.0 / (1.0 + np.exp(-z))
    else:
        ez = np.exp(z)
        soft_a = ez / (1.0 + ez)

    core_a = (1.0 - w_soft) * core_threshold + w_soft * soft_a
    p_a = (1.0 - lapse) * core_a + lapse / 2.0
    p_a = float(np.clip(p_a, 0.0, 1.0))

    return np.array([p_a, 1.0 - p_a], dtype=float)
```

**`policy source code`:**
```python
import numpy as np


def policy(probs):
    p = np.asarray(probs, dtype=float)
    if p.sum() <= 0.0:
        p = np.ones_like(p) / len(p)
    else:
        p = p / p.sum()
    return int(np.random.choice(len(p), p=p))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This design uses six cues with one clearly highest-validity cue (0.91) that discriminates on every trial, while the five lower-validity cues are varied so their tally advantage runs from strongly pro-A to strongly pro-B. In trials 1-4 the highest-validity cue favors A; in trials 5-8 it favors B. Take The Best stops at that highest cue, so it predicts one constant A-choice probability for trials 1-4 and one constant complementary A-choice probability for trials 5-8, with no effect of the lower-cue tallies. Tallying ignores validity and uses the net feature-win count, so it predicts a monotone sigmoid in the tally difference within each block: across trials 1-4 the net advantage changes from +4 to +2 to 0 to -2, and across trials 5-8 from +2 to 0 to -2 to -4. The +4 and +2 top-A conditions separate the theories quantitatively even when both prefer A, while the tie and negative-tally conditions produce neutral or opposite predictions. This creates a flat-vs-monotone signature rather than a single ambiguous pair. Each unique pair is repeated 12 times because 96 // 8 = 12.

[1] This design contrasts Tallying's use of the net feature-win count with TTB's use of only the first discriminating cue. Five cues have spread validities (0.90, 0.80, 0.65, 0.55, 0.50). The 12 unique pairs are grouped by net tally d = #(A>B) - #(B>A): two pairs each at d=+2, -2, +1, -1, and four at d=0. Within each same-d group, Tallying predicts identical A-choice probabilities (sigmoid(beta*d)), whereas TTB predicts opposite directions because the first discriminating cue changes: in one member the top cue discriminates; in the other the top cue ties and the second cue discriminates in the opposite direction. For example, trials 1 and 2 both have d=+2; Tallying predicts strong A for both, but TTB predicts A on trial 1 and B on trial 2. Across d, Tallying predicts a monotone sigmoid, while TTB predicts a flat probability depending only on the first-cue winner. This yields a double dissociation and also quantitative differences on same-direction trials (e.g., d=+2 gives Tallying sigmoid(2*beta) versus TTB sigmoid(beta)). Tie values on non-discriminating features are varied for display and do not affect either model.

[2] Six experts are split into three high-validity cues (0.90, 0.85, 0.80) and three low-validity cues (0.55, 0.50, 0.50). NWA assigns negative validity-scaled weights, so an option's advantage on a high-validity cue is strong evidence against that option, whereas Tallying only counts the number of feature-wise wins and ignores validity. The eight pairs are organized by tally difference d = #A wins - #B wins. Trials 1-2 have d=0: Tallying must predict 0.50 for both, but NWA predicts B on trial 1 and A on trial 2 because the high-validity wins switch sides. Trials 3-4 both have d=+1, so Tallying predicts the same A-choice probability for both; NWA predicts B on trial 3 (A wins the three high cues) and A on trial 4 (A wins only the three low cues while B wins two high cues). Trials 5-6 mirror this at d=-1. Trials 7-8 are d=+2 and d=-2, where Tallying predicts strong A and strong B respectively, but NWA predicts the opposite. With beta=2.7, gamma=2.25, NWA p_A is approximately .02, .98, .01, .90, .99, .10, .006, .995 for trials 1-8; Tallying p_A is .50, .50, sigmoid(beta), sigmoid(beta), sigmoid(-beta), sigmoid(-beta), sigmoid(2beta), sigmoid(-2beta). The within-d reversals are the critical quantitative dissociation: Tallying cannot produce different p_A for trials with the same tally difference, regardless of beta or epsilon.

[3] This design uses four experts with a wide validity spread (0.95, 0.85, 0.65, 0.55). The first six unique pairs are all 2-vs-2 splits, so Tallying's net feature-win difference is zero for every one of them; Tallying must predict p(A)=0.50 exactly regardless of beta or epsilon because the softmax over equal tallies is uniform and the lapse mixture remains uniform. NWA, by contrast, assigns negative validity-scaled weights, so the same zero tally produces a graded range of p(A) as the two A-won features move from the two highest-validity experts (NWA p_A approximately 0.07) to the two lowest-validity experts (NWA p_A approximately 0.93). The last two pairs are 3-vs-1 splits with tally differences +2 and -2. Tallying therefore predicts a strong A preference for the +2 pair and a strong B preference for the -2 pair, while NWA predicts the opposite direction on both because the majority wins are low-validity and the single minority win is the highest-validity expert (NWA p_A approximately 0.23 and 0.77, assuming beta=2.70, gamma=2.25, epsilon=0). This creates a flat-vs-graded signature on the d=0 trials and a sign reversal on the d=+/-2 trials, which no Tallying beta or epsilon can reproduce.

[4] This design isolates the conflict-gating mechanism of VWEA-CG from NWA's fixed negative validity-weighted additive rule. Six experts are used. Feature 2 validity is set to 0.8148 because at gamma=2.25 its weight (0.8148^2.25 ≈ 0.631) equals the sum of the three 0.50-validity weights (3 × 0.5^2.25 ≈ 0.631). Consequently, the two positive-U pairs have the same validity-weighted difference U = 0.90^2.25 + 0.80^2.25 ≈ 1.394: in pair 1 A wins features 1-3 and B wins features 4-6 (tally 0), while in pair 2 A wins features 1,3,4,5,6 and B wins feature 2 (tally +4). The negative-U pairs are the exact mirror image. NWA depends only on S = -U, so it must predict identical p(A) for pairs 1 and 2 (≈0.03) and identical p(A) for pairs 3 and 4 (≈0.97). VWEA-CG, however, computes conflict = |U|/(1+|tally|). Pair 1 has conflict ≈1.39 and is strongly attenuated toward chance (p_A≈0.38), whereas pair 2 has conflict ≈0.28 and is not attenuated (p_A≈0.08); likewise pair 3≈0.62 and pair 4≈0.92 under central parameter values. The critical dissociation is therefore the within-U tally effect: NWA predicts a flat two-point pattern, while VWEA-CG predicts a large conflict-driven separation within each matched-U pair.

[5] Seven experts with a wide validity spread. Every unique pair is a 3-vs-3 split with one tied expert, so the unweighted feature tally is exactly zero for all trials. The validity-weighted difference U = sum_j v_j^gamma (A_j - B_j) is varied across trials by moving the three A-won experts from a mixed set to the three highest-validity experts. NWA depends only on U and predicts p(A) = sigmoid(-beta U) plus lapse, so p(A) must decrease monotonically as U increases. VWEA-CG has conflict = |U|/(1+|tally|) = |U|, so its effective inverse temperature shrinks at large U; the negative-valence majority drives p(A) down for moderate U, but conflict attenuation pushes p(A) back toward chance for large U. Under central parameters the predicted p(A) values for U = 0.134, 0.247, 0.325, 0.437, 0.620, 0.824, 1.014, 1.191 are approximately: NWA .411, .341, .296, .238, .161, .102, .065, .043; VWEA-CG .391, .300, .251, .204, .179, .209, .268, .326. Thus NWA is monotone decreasing while VWEA-CG is U-shaped, a rank-order dissociation that no NWA beta/gamma/epsilon can reproduce.

[6] Five experts with a wide validity spread [0.92, 0.84, 0.74, 0.62, 0.52]. At gamma=2.25 their validity weights v^gamma are approximately [0.829, 0.675, 0.508, 0.341, 0.230]. The 16 unique pairs are built as eight positive-D pairs and their A/B mirrors. In the positive block, A wins a subset of experts so the validity-weighted difference D = sum_j v_j^gamma (A_j - B_j) takes the values 0.551, 0.773, 0.885, 0.925, 1.107, 1.233, 1.441, 1.567; the mirrored block has the same |D| values with D<0. NWA uses S = -D and a softmax with beta around 2.7, so p(A) falls steeply across the positive block (about 0.19, 0.12, 0.09, 0.08, 0.06, 0.04, 0.03, 0.02 with epsilon=0.02) and rises steeply across the negative block (about 0.81 to 0.98). CNVW-tau instead passes D through the saturating transform psi(D)=D/(tau+|D|) with tau in [0.10, 0.25]; for |D|>=0.55 this transform is already strongly saturated, so its p(A) is nearly flat within each sign: approximately 0.22 to 0.18 for D>0 and 0.78 to 0.82 for D<0 under central parameters. Thus both theories predict B for positive D and A for negative D, but NWA predicts a large within-sign graded effect while CNVW-tau predicts two flat plateaus. This is a quantitative dissociation that CNVW-tau's beta/lambda/bias parameters cannot remove, because those parameters shift the plateau levels but do not restore a steep D-magnitude slope.

[7] Five experts with validities [0.95, 0.90, 0.80, 0.60, 0.50] create a wide spread of validity weights. The 16 unique pairs are 8 positive validity-weighted difference values D = sum_j v_j^gamma (A_j - B_j) and their exact A/B mirrors. At gamma=2.25 the positive D values are approximately 0.077, 0.102, 0.209, 0.390, 0.548, 0.758, 0.968, 1.758. This spans both the near-zero region and the saturation region. CNVW-tau's saturating transform psi(D)=D/(tau+|D|) is steep near zero because psi is approximately D/tau, but it plateaus for large D, while NWA uses a single logistic in -D. Under central parameters (CNVW: gamma=2.25, tau=0.175, lambda=1.4, beta=1.4, bias=0, epsilon=0.05; NWA: gamma=2.25, beta=2.7, epsilon=0.01), the positive-D p(A) predictions are approximately: CNVW .362, .336, .268, .220, .201, .185, .177, .162; NWA .449, .432, .364, .261, .189, .118, .073, .014. Thus CNVW-tau predicts lower p(A) than NWA for D < 0.5 but higher p(A) for D > 0.5. This crossover is a quantitative signature that NWA cannot produce because its psychometric function is a single logistic with a very low lapse floor, whereas CNVW-tau combines a high near-zero slope with a bounded saturating transform. The mirrored negative-D trials test the same crossover on the opposite side and also allow bias/symmetry checks.

[8] The design pits a first-discriminating-cue inversion process (GSS-ITTB) against a continuous validity-weighted integration process with saturation (CNVW-tau). With validities [0.90, 0.80, 0.70, 0.60, 0.55], the 0.90 expert discriminates on every unique pair, so GSS-ITTB's attentional gate is zero and no later cue can affect the response. GSS-ITTB therefore predicts only two flat choice probabilities: about 0.27 for A when the 0.90 cue favors A, and about 0.73 for A when the 0.90 cue favors B, with no effect of the lower-cue balance. CNVW-tau instead aggregates D = sum_j v_j^gamma*(A_j-B_j) and most subjects invert its sign. In the six top-A pairs, D takes approximately +0.789, +0.184, +0.472, -0.264, -0.581, -0.841; under central CNVW-tau parameters this gives p(A) near .18, .28, .21, .75, .80, .82, crossing from choosing B to choosing A while GSS-ITTB stays at about .27 throughout. The six top-B mirrors have the complementary D signs and likewise force CNVW-tau to cross from high A choice to low A choice while GSS-ITTB stays at about .73. The critical dissociation is the top-A negative-D trials and the top-B positive-D trials: GSS-ITTB continues choosing opposite the first cue, whereas CNVW-tau reverses because the lower-validity majority flips the sign of D. This sign-reversal signature cannot be produced by a flat first-cue rule regardless of q_90, and cannot be flattened by any CNVW-tau tau/lambda/beta/bias/lapse tuning while preserving the D-dependence.

[9] This design runs a first-discriminating-cue validity sweep while keeping the validity-weighted difference D positive in the first six trials. The validities are [0.95, 0.92, 0.90, 0.62, 0.50], so at gamma=2.0 the CNVW-tau weights are approximately 0.9025, 0.8464, 0.8100, 0.3844, 0.2500. In trials 1-6 the first discriminating cue always favors A and D equals 0.9025, 0.8464, 0.8100, 0.6344, 0.2500, and 0.1344. With central CNVW-tau parameters (tau=0.175, lambda=1.4, beta=1.4, negative sign) the predicted p(A) values are about 0.18, 0.18, 0.18, 0.19, 0.25, 0.31, a low flat-to-graded pattern because psi(D)=D/(tau+abs(D)) saturates for large positive D and most subjects invert the sign. GSS-ITTB stops at the first discriminating cue and does not integrate the later cues continuously. Its first-cue q calibration is nonmonotonic across the first three validity levels, and it becomes strongly inverted for low-validity first cues, predicting p(A) of about 0.23, 0.45, 0.27, 0.55, 0.68, 0.55 for the same six trials. Trials 4 and 6 are especially diagnostic: they share the same first discriminator (0.62 favoring A), but trial 6 changes the later 0.50 cue against A. GSS-ITTB predicts about 0.55 on both trials, whereas CNVW-tau moves from about 0.19 to 0.31 because D drops from +0.63 to +0.13. The last six trials are the exact A/B mirrors, giving complementary high-side predictions and a symmetry check. This produces a rank-order and within-first-cue dissociation: CNVW-tau cannot produce the GSS zigzag or the majority-A choices on D-positive low-first-cue trials, while GSS-ITTB cannot produce the continuous D-driven separation between trials 4 and 6. There are 12 unique pairs, K=8, giving 96 trials.

[10] This six-expert design is a validity-weighted-difference sweep aimed at separating TAVU-tn's thresholded anti-validity process from CNVW-tau's continuous compressive integration. At the central validity exponent gamma=2.0, the validity weights v^2 are approximately [0.810, 0.640, 0.490, 0.330, 0.320, 0.250]. The positive-D block produces D = sum_j v_j^gamma (A_j - B_j) values of 0.010, 0.070, 0.240, 0.410, 0.710, and 1.040, with the negative-D block containing exact A/B mirrors. The near-zero pair is critical: D=0.010 comes from replacing the 0.5657-validity expert with the 0.5745-validity expert inside an otherwise balanced sum. TAVU-tn has a threshold plus fallback process, so even a tiny nonzero D switches the sign target and a substantial fraction of the threshold distribution crosses tau; it should then flatten near the anti-validity ceiling over the moderate D range. CNVW-tau uses psi(D)=D/(tau+abs(D)) with no threshold, so it should move only slightly at D=0.010 but keep declining across larger D values. Under illustrative central parameter settings, TAVU-tn predicts p(A) for the positive-D block of about [0.386, 0.327, 0.215, 0.202, 0.207, 0.212], while CNVW-tau predicts about [0.484, 0.416, 0.336, 0.306, 0.282, 0.271]. The signature is therefore an early large threshold-like step plus a plateau for TAVU-tn, versus a small continuous early step plus continued graded decline for CNVW-tau. The mirrored negative-D block doubles the data and provides symmetry and bias/lapse checks. There are 12 unique pairs, so each repeats 8 times for the 96-trial budget.

[11] This design isolates a structural property that separates the two theories: continuity across D = 0. The validity set contains two matched duplicate pairs (0.92, 0.92 and 0.65, 0.65), so swapping a 0.92 win between A and B and a 0.65 win between A and B gives an exact D = 0 trial for any gamma while the displayed ratings still differ. Two close low-validity pairs (0.57 vs 0.56 and 0.51 vs 0.50) then produce tiny signed D values: +0.0012, -0.0012, +0.0101, -0.0101, +0.0113, -0.0113, +0.0214, -0.0214 at gamma = 2. TAVU-tn thresholds on abs(D), so the +epsilon and -epsilon pairs have almost identical threshold-crossing probability, but the latent sign target flips discontinuously at D = 0. Under central TAVU-tn parameters this predicts a large side jump, approximately p(A) = 0.396 at D = +0.0012 and p(A) = 0.604 at D = -0.0012, versus p(A) = 0.50 at D = 0. CNVW-tau instead uses the continuous bounded transform psi(D) = D / (tau + abs(D)), which is approximately D / tau near zero, so the same pairs predict p(A) = 0.497 and p(A) = 0.503: only a very small linear shift. Thus CNVW-tau predicts a smooth near-zero psychometric function, while TAVU-tn predicts an immediate sign-driven step. The moderate and large D pairs (+/-0.25, +/-0.4225, +/-0.8464) add a second quantitative signature: after threshold crossing TAVU-tn plateaus around p(A) = 0.21 to 0.22 on the positive side, whereas CNVW-tau continues to decline from about 0.29 to 0.23. The combination is a continuity-versus-discontinuity signature at the zero crossing plus a plateau-versus-slope signature at larger D, which cannot be removed by TAVU-tn parameter tuning within its specified ranges.

[12] At gamma = 2.0 the six validity weights v^gamma are [0.9216, 0.8836, 0.81, 0.64, 0.36, 0.25]. The first eight rows use positive D = sum_j v_j^gamma * (A_j - B_j) values of 0.038, 0.0736, 0.1116, 0.17, 0.20, 0.2436, 0.2816, and 0.3116; the last eight rows are exact A/B mirrors. The staircase is built to bracket the TAVU threshold tau in [0.16, 0.19]: the first three D values lie in the TAVU dead zone, 0.17 sits at threshold, 0.20 just above it, and the remaining values are clearly above threshold. For the negative-weight majority, TAVU-tn predicts p(A) near 0.50 for the first three positive pairs, a sharp drop to about 0.26 at D = 0.20, and then a plateau near 0.20 to 0.21; the mirrored side gaps are about 0.00, 0.00, 0.01, 0.26, 0.47, 0.59, 0.59, and 0.59. CNVW-tau has no dead zone and predicts a smooth continuous decline in p(A) from about 0.42 at D = 0.038 to about 0.24 at D = 0.31, with mirrored side gaps growing monotonically from about 0.16 to 0.53. The critical dissociation is dead-zone flatness below threshold plus a threshold jump and plateau versus a smooth saturation curve, with the two side-gap functions crossing near D = 0.20. This quantitative interaction cannot be removed by beta, lambda, bias, or lapse within the stated ranges.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "This design isolates the shape of the D-to-choice mapping by using a one-hot validity staircase anchored on the 0.50 feature. With validities [0.95, 0.87, 0.75, 0.68, 0.65, 0.62, 0.57, 0.51, 0.50] and a central gamma of 2.0, the squared validity weights are approximately [0.9025, 0.7569, 0.5625, 0.4624, 0.4225, 0.3844, 0.3249, 0.2601, 0.2500]. Each positive-D trial lets A win one higher-validity feature and B win the 0.50 anchor, so D equals the simple difference between that feature's weight and 0.25. The positive D staircase is therefore 0.0101, 0.0749, 0.1344, 0.1725, 0.2124, 0.3125, 0.5069, 0.6525. The critical property is that TAVU-tn has a threshold and dead zone while CNVW-tau is continuous and compressive. At the three smallest positive D values, TAVU-tn remains inside its threshold dead zone because those values are roughly 2 to 3 threshold-noise units below tau, so it predicts p(A) close to the fallback level, about 0.50. CNVW-tau instead applies psi(D) = D / (tau + abs(D)) with tau around 0.175, so it predicts a graded anti-validity decline from about p(A) = 0.48 at D = 0.0101 down to about 0.33 at D = 0.1344. This pre-threshold gradient is a shape signature TAVU-tn cannot produce with tau_sd in [0.02, 0.04]. The 0.1725 pair sits at the TAVU threshold, producing a steep threshold-like drop for TAVU-tn, whereas CNVW-tau continues smoothly. The larger D values add a second signature: after crossing threshold, TAVU-tn plateaus near p(A) around 0.26 because follow_max and the soft leak bound its anti-validity ceiling, while CNVW-tau keeps compressing and reaches about p(A) = 0.22 at D = 0.6525. Under central parameters (CNVW-tau: gamma=2.0, tau=0.175, lambda=1.4, beta=1.4, bias=0, epsilon=0.05, p_positive=0.05; TAVU-tn: gamma=2.0, tau=0.17, tau_sd=0.03, p_negative=0.91, bias=0.5, lapse=0.14, soft_weight=0.02, soft_beta=0.9, follow_max=0.70), the positive-block p(A) values are approximately CNVW-tau [0.477, 0.378, 0.328, 0.307, 0.290, 0.262, 0.234, 0.223] versus TAVU-tn [0.500, 0.500, 0.472, 0.372, 0.275, 0.259, 0.258, 0.259]. This gives a rank-order crossover near D = 0.21 and a second ordering reversal at high D, so the design produces both a dead-zone-flatness dissociation and a plateau-versus-compression dissociation in one staircase. The eight negative-D mirror trials double the data and provide a symmetry check. There are 16 unique pairs, repeated 6 times each for the 96-trial budget.",
  "validities": [
    0.95,
    0.87,
    0.75,
    0.68,
    0.65,
    0.62,
    0.61,
    0.6,
    0.6
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
      0
    ],
    [
      0,
      0,
      0,
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
      0,
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
      0
    ],
    [
      0,
      0,
      0,
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
      1,
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
  "input_tokens": 9298,
  "output_tokens": 32988,
  "reasoning_tokens": 31171,
  "provider": "Baidu"
}
```
