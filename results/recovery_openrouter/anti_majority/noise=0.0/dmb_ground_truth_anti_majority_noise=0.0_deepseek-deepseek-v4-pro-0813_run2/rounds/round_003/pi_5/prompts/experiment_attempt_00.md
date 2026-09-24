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
**Description:** Negative Validity-Weighted Additive (NWA): people combine all binary feature differences linearly, but assign negative weight to a feature advantage, and the absolute weight magnitude increases with the feature's stated validity. Formally, choice is based on S = sum_j w_j * (A_j - B_j), with w_j = -v_j^gamma (gamma > 0), so an option's cue advantage makes that option less likely to be chosen. Choice probability uses a softmax/sigmoid over this score with inverse temperature beta, plus a small uniform lapse epsilon.

**Parameters:**
- beta: [2.65, 2.75]
- gamma: [2.22, 2.28]
- epsilon: [0.0, 0.02]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # NWA: negative validity-weighted additive integration of all cues.
    # state is shape (2, n_features): row 0 = option A, row 1 = option B.
    # history is ignored because there is no trial-by-trial feedback.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"NWA expects a (2, n_features) stimulus; got shape {stim.shape}.")

    a = stim[0]
    b = stim[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.size != a.size:
        raise ValueError(
            f"validities length {v.size} does not match n_features {a.size}."
        )

    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Negative validity-scaled weights: an advantage on a cue is evidence
    # against that option, with stronger validity cues carrying more weight.
    weights = -(v ** gamma)

    score = float(np.dot(weights, a - b))

    # Stable sigmoid for p(A | score).
    z = beta * score
    if z >= 0:
        p_a_core = 1.0 / (1.0 + np.exp(-z))
    else:
        ez = np.exp(z)
        p_a_core = ez / (1.0 + ez)

    # Uniform lapse mixture.
    p_a = (1.0 - epsilon) * p_a_core + epsilon / 2.0
    return np.array([p_a, 1.0 - p_a], dtype=float)
```

**`policy source code`:**
```python
def policy(probs):
    p = np.asarray(probs, dtype=float)
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

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "Five experts with validities [0.95, 0.90, 0.80, 0.60, 0.50] create a wide spread of validity weights. The 16 unique pairs are 8 positive validity-weighted difference values D = sum_j v_j^gamma (A_j - B_j) and their exact A/B mirrors. At gamma=2.25 the positive D values are approximately 0.077, 0.102, 0.209, 0.390, 0.548, 0.758, 0.968, 1.758. This spans both the near-zero region and the saturation region. CNVW-tau's saturating transform psi(D)=D/(tau+|D|) is steep near zero because psi is approximately D/tau, but it plateaus for large D, while NWA uses a single logistic in -D. Under central parameters (CNVW: gamma=2.25, tau=0.175, lambda=1.4, beta=1.4, bias=0, epsilon=0.05; NWA: gamma=2.25, beta=2.7, epsilon=0.01), the positive-D p(A) predictions are approximately: CNVW .362, .336, .268, .220, .201, .185, .177, .162; NWA .449, .432, .364, .261, .189, .118, .073, .014. Thus CNVW-tau predicts lower p(A) than NWA for D < 0.5 but higher p(A) for D > 0.5. This crossover is a quantitative signature that NWA cannot produce because its psychometric function is a single logistic with a very low lapse floor, whereas CNVW-tau combines a high near-zero slope with a bounded saturating transform. The mirrored negative-D trials test the same crossover on the opposite side and also allow bias/symmetry checks.",
  "validities": [
    0.95,
    0.9,
    0.8,
    0.6,
    0.5
  ],
  "trial_a_ratings": [
    [
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
      0,
      0
    ],
    [
      1,
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
      0
    ],
    [
      1,
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
      0
    ],
    [
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
      0,
      0
    ],
    [
      0,
      0,
      1,
      1,
      0
    ],
    [
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
      0,
      1
    ],
    [
      0,
      1,
      0,
      1,
      0
    ],
    [
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
      0
    ],
    [
      0,
      0,
      1,
      1,
      0
    ],
    [
      0,
      0,
      0,
      1,
      1
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      0,
      1,
      1,
      0
    ],
    [
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
      0,
      1
    ],
    [
      0,
      1,
      0,
      1,
      0
    ],
    [
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
      0
    ],
    [
      0,
      0,
      1,
      1,
      0
    ],
    [
      0,
      0,
      0,
      1,
      1
    ],
    [
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
      0,
      0
    ],
    [
      1,
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
      0
    ],
    [
      1,
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
      0
    ],
    [
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
      0,
      0
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 5748,
  "output_tokens": 31229,
  "reasoning_tokens": 20331,
  "provider": "CoreWeave"
}
```
