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
**Description:** People compare two options by consulting cues one at a time in order of validity, stopping at the first cue that discriminates between the two options. That cue alone determines the choice: the option with the higher value on the discriminating cue wins, and no other cue is consulted. Cues with lower validity are never reached when a higher-validity cue already discriminates, so Take The Best is a "one-reason" decision rule — only a single feature is ever used on any given choice. Because only the sign of the comparison on the top discriminating cue matters, TTB ignores both the magnitude of that difference and all information on lower-validity cues, making it maximally frugal in the use of evidence. Cue validities are subjective and can be learned, inferred from instructions, or fixed by task structure. When no cue discriminates (all feature-wise comparisons tie) the learner has no basis for preference and must guess. Response noise enters through a softmax over the binary TTB score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Take The Best (Gigerenzer & Goldstein 1996).
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B. Cue cascade: features are consulted in order
    # of descending validity; the first discriminating cue (strict
    # inequality) determines the winner; if no cue discriminates,
    # the model guesses uniformly. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )
    # Descending validity; argsort is stable so validity ties break
    # toward the earlier feature index.
    cue_order = np.argsort(-val, kind="stable").tolist()

    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    if winner is None:
        # No discriminating cue — pure guess.
        return np.ones(2) / 2.0

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. For the
    # binary TTB score this collapses to sigmoid(beta) for the winner,
    # giving a direct mapping from beta onto the paper's flip-noise
    # levels (beta=0 ↔ 50/50; beta ≫ 1 ↔ deterministic).
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Validity-Weighted Evidence Integration (VWEI) with subjective validity amplification. On each trial the decision maker computes, for every rating dimension j, the discriminant d_j = sign(a_j - b_j) in {-1, 0, +1}, weights it by a subjectively amplified validity w_j = [log(v_j/(1-v_j))]^gamma, and accumulates evidence E = sum_j w_j * d_j. Choice probability is a sigmoid of beta * E mixed with a small uniform lapse epsilon. gamma = 1 recovers Bayes-optimal log-odds integration; gamma -> infinity recovers Take-The-Best; the empirically human region sits at gamma near 3, where the top cue dominates coalitions of weaker cues on steep validity gradients (Exp 2) but lower cues still attenuate allegiance on shallow gradients (Exp 1). This round's refinement is purely parametric: the mechanism is unchanged, but the sampling region is re-centered on the hand-verified sweet spot (gamma = 3, beta = 0.2, epsilon = 0), the lapse range is restored to (a tightened version of) the arbiter's [0, 0.2], and the beta floor is raised so that generic flattening noise no longer competes with the gamma-structured weight profile for attenuation work.

**Parameters:**
- gamma: [2.8, 3.2]
- beta: [0.18, 0.23]
- epsilon: [0.0, 0.03]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Validity-Weighted Evidence Integration (VWEI) with subjective
    # validity amplification. Stimulus is the pair of option feature
    # vectors for the current trial: array-like of shape
    # (2, n_features), row 0 = option A, row 1 = option B.
    #
    # Evidence for A over B:  E = sum_j w_j * sign(a_j - b_j),
    # with w_j = [log(v_j / (1 - v_j))]^gamma.
    #   gamma = 1  -> Bayes-optimal log-odds weighting (full integration)
    #   gamma -> inf -> Take-The-Best (top cue dominates every coalition)
    # P(A) = sigmoid(beta * E), mixed with a uniform lapse epsilon.
    # History is ignored: validities are communicated in the
    # instructions, so weights are fixed for the whole block.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"VWEI expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n_features = stim.shape[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.shape[0] != n_features:
        raise ValueError(
            f"validities length {v.shape[0]} != n_features {n_features}."
        )
    # Guard the log-odds against v = 1 (infinite weight) and v < 0.5.
    v = np.clip(v, 0.5 + 1e-9, 1.0 - 1e-6)
    log_odds = np.log(v / (1.0 - v))

    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Subjective cue weights: amplified log-odds validity.
    w = np.sign(log_odds) * np.abs(log_odds) ** gamma

    # Per-cue discriminant: +1 favors A, -1 favors B, 0 is uninformative.
    d = np.sign(a - b)

    # Accumulated evidence for A (positive) vs B (negative).
    E = float(np.dot(w, d))

    # Numerically stable sigmoid: 0.5 * (1 + tanh(x/2)).
    # When E == 0 this is exactly 0.5 (uniform guessing), for any beta.
    x = beta * E
    p_a = 0.5 * (1.0 + np.tanh(0.5 * x))
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

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "This design exploits a structural chasm between TTB and VWEI: TTB's score is binary (winner=1, loser=0), so its softmax NEVER sees cue validity, coalition size, or coalition composition. Consequence: TTB predicts ONE constant choice probability, P(pick TTB's winner) = (1-eps)*sigmoid(beta) + eps/2, on ALL 16 pairs \u2014 whether the deciding cue is the 90% expert or the 76% expert, and whether 0 or 7 lower cues oppose it. VWEI, with its bounded parameters (gamma in [2.8,3.2], beta in [0.18,0.23]), predicts a specific, monotone, non-flat profile across the same 16 pairs. The validity vector [0.90, 0.78, 0.76, 0.74, 0.72, 0.70, 0.68, 0.66, 0.64] is engineered so that (a) cue 1 (v=0.90, w=10.6) dominates every possible coalition, (b) cue 2 (v=0.78, w=2.03) is beaten by coalitions of 2+ of the next cues (w3+w4=2.68) but not by any single lower cue or by weak cues (w8+w9=0.48) \u2014 creating sign flips that are robust across the entire gamma range [2.8, 3.2] and beta range [0.18, 0.23] (verified: pair-7 E ranges -0.93..-0.95; pair-11 E ranges -3.35..-3.41; pair-13 E ranges +0.96..+1.14). Four dissociation signatures: (1) COALITION-SIZE GRADIENT (pairs 5-12): cue 1 ties, cue 2 picks the TTB winner, and the opposing coalition below cue 2 grows from 1 to 2 to 3 to 7 cues. TTB: P(cue-2 winner) flat at sigmoid(beta). VWEI: monotone decline 0.525 -> 0.468 -> 0.426 -> 0.354, crossing 50% between one and two opponents. (2) COMPOSITION AT FIXED COUNT (pairs 13/14 vs 7/8): exactly two opposing cues, but weak ones (experts 8-9) vs strong ones (experts 3-4). TTB: identical predictions (cue 2 decides in all four; validity-blind). VWEI: an ~11-point gap (0.577 vs 0.468) \u2014 this weight-magnitude sensitivity at fixed count falsifies TTB (and any count-based heuristic) while confirming VWEI's amplified log-odds weighting. (3) DECIDING-CUE VALIDITY FLATNESS (pairs 1-4 vs 5-14 vs 15-16): when the 90% expert decides (pairs 1-4, agreement baselines), VWEI predicts P(winner)=0.85-0.89; when the 78% expert decides, ~0.35-0.58; when the 76% expert decides against a 3-cue coalition (pairs 15-16, cascade-depth test), 0.447. TTB predicts the SAME probability in all three regimes \u2014 a large cross-family validity effect falsifies TTB, flatness falsifies VWEI. (4) SIGN CONFLICTS: on pairs 7-12 and 15-16 the theories point in OPPOSITE directions (TTB follows cue 2/3; VWEI's evidence sum favors the coalition), so a single trial type can flip a subject's majority choice. Every pair is non-degenerate (a discriminating cue always exists), the design is perfectly side-balanced (8 pairs with TTB winner = A, 8 with TTB winner = B; pairs are exact A/B mirrors), and with 16 unique pairs, K = 96//16 = 6 gives 96 trials and 12 samples per mirrored cell \u2014 enough per-subject power to estimate the flat-vs-graded profile that adjudicates the theories. A TTB subject is a horizontal line; a VWEI subject is a specific descending curve that no constant can fit (VWEI's spread 0.89->0.35 exceeds anything TTB's two noise parameters can produce, since TTB has no mechanism for any stimulus-driven variation at all).",
  "validities": [
    0.9,
    0.78,
    0.76,
    0.74,
    0.72,
    0.7,
    0.68,
    0.66,
    0.64
  ],
  "trial_a_ratings": [
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
      1
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
      0,
      0,
      1,
      1,
      1,
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
      1
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
      1,
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
  "input_tokens": 4039,
  "output_tokens": 21829,
  "reasoning_tokens": 19880,
  "provider": "Sail Research"
}
```
