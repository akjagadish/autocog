# experiment_attempt_00

## System Prompt

You are a renowned cognitive scientist designing an experiment in the Decision Making (Binary Features) domain.

Your goal is to be an adversarial collaborator: propose a design whose outcomes would be predicted by your advocated theory but NOT by the competing theory. Both are provided below.

A useful proposal targets a *quantitative* dissociation between the two theories — how they respond differently to specific stimuli in addition to differences in overall performance.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

A multi-attribute decision-making experiment. On each trial the subject sees two options (A, B), each described by `n_features` integer expert ratings (`n_features` is set by the length of `validities` you propose). Choose `validities` — one per feature, each in [0.5, 1.0], order free — to fix each expert's advertised accuracy; subjects are told these values up front. Then choose `trial_a_ratings/trial_b_ratings` (each rating value in [0, 1]) so that the intended heuristics (e.g. TTB, EQW, Tallying, WADD) make distinguishable predictions: avoid degenerate pairs where every heuristic agrees, and prefer pairs that dissociate single-feature focus from feature-summing strategies. Validities and the trial ratings together define the design; they are fixed across all trials in this experiment. No trial-by-trial correctness feedback. The total number of trials per subject is held at roughly 96: each unique pair is repeated K = max(1, 96 // n_unique_pairs) times in an independently-randomized order per subject.

Subjects see the following instructions:
In this experiment you will repeatedly choose between two fictitious products, A and B. On every trial you will see `n_features` expert ratings for each product (the number of experts is fixed across all trials and is set by the length of `validities`).

Each rating is an integer in [0, 1]. The ratings are displayed as a horizontal filled bar with the numeric value (e.g. "0/1") shown next to it. Higher = more positive.

The same experts (in the same order) provide ratings for both products on every trial. Each expert's accuracy (their validity expressed as a percentage, e.g. "Expert 1 (80%)") is shown next to their rating on every trial AND is also listed up front in an "Expert accuracies" panel.

On each trial, decide which product is of higher quality and press A for product A or B for product B. There is no time limit and no feedback. Note that for the first ~`min_rt_ms` of each trial the answer prompt is hidden and the keys are locked, so subjects first see the full ratings and can answer once the A / B prompt appears — design pairs that actually require comparing the ratings.

Total trials per subject is roughly `MAX_TRIALS`: each unique pair you propose is repeated `K = max(1, MAX_TRIALS // n_unique_pairs)` times in an independently-randomized order per subject.

## ADVOCATED THEORY
**Description:** People decide by the PARITY of the evidence rather than its weight or direction. They count the number of cues on which option A strictly beats option B, and prefer A when that count is ODD and B when it is EVEN (zero counts as even). This is a deliberately non-monotone, XOR-like rule: adding one more cue in A's favor flips the preference rather than strengthening it, so option dominance does NOT imply choice — an option that wins on every cue (an even count, when the cue number is even) is rejected. The rule uses no validities and no magnitudes, only the parity of feature-wise wins, which makes it an adversarially hard recovery target: it is uncorrelated with any single cue and with the validity-weighted sum, yet perfectly deterministic, much like the anti-majority ensemble. Response noise enters through a softmax over the binary score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # Cue-Parity: prefer A iff the number of cues where A strictly beats B is
    # ODD, else prefer B (zero A-wins is even -> B). Non-monotone in the
    # evidence; uses neither validities nor magnitudes. Stimulus is the pair
    # of option feature vectors: (2, n_features), row 0 = A, row 1 = B.
    # History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Cue-Parity expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    a, b = stim[0], stim[1]
    a_wins = int(np.sum(a > b))
    # Odd number of A-favoring cues -> A (0); even (including zero) -> B (1).
    winner = 0 if (a_wins % 2 == 1) else 1

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. For the binary
    # score this collapses to sigmoid(beta) for the winner (beta=0 ↔ 50/50;
    # beta ≫ 1 ↔ deterministic).
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
**Description:** Top-K Parity: Decision makers rely on a strict Cue-Parity rule but apply it selectively to a stable subset of the most valid cues. Instead of using a floating validity threshold or rounding a fraction (which can arbitrarily drop a single cue and flip the parity sum in 6- and 12-cue tasks), they evaluate the top ceil(k_frac * N) cues. This ensures that for tasks with 5 or 6 cues, all cues are evaluated, preserving multi-cue parity effects without collapsing to random parity flips.

**Parameters:**
- validities: validities
- k_frac: [0.9, 1.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    import math
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Top-K Parity expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    k_frac = float(parameters["k_frac"])
    
    n_cues = len(validities)
    # Use math.ceil to avoid arbitrarily dropping exactly 1 cue in 5- or 6-cue tasks
    k = max(1, math.ceil(k_frac * n_cues))
    
    # Get indices of the top K validities
    # np.argsort sorts ascending, so we take the last k elements
    top_k_indices = np.argsort(validities)[-k:]
    
    a_filtered = a[top_k_indices]
    b_filtered = b[top_k_indices]
    
    # Strict Cue-Parity on the filtered subset of cues
    a_wins = int(np.sum(a_filtered > b_filtered))
    winner = 0 if (a_wins % 2 == 1) else 1
    
    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over binary score
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] To dissociate Cue-Parity from Tallying, we manipulate the number of cues on which option A strictly beats option B. Tallying predicts a monotonic preference for A as the number of A-wins increases relative to B-wins. Cue-Parity, in contrast, predicts a non-monotone, oscillating preference based solely on whether the number of A-wins is odd or even. For example, when A dominates B on 2 or 4 features, Tallying strongly predicts A, but Cue-Parity predicts B (because 2 and 4 are even). Conversely, when A wins on 1 feature and B wins on 2 or 3, Tallying predicts B, but Cue-Parity predicts A (because 1 is odd). We include these crucial dissociations as well as baseline trials where both models agree.

[1] To strongly dissociate Tallying from Cue-Parity, this design focuses on trials where option A and option B win on an EQUAL number of features (i.e., a_wins == b_wins). Tallying predicts complete indifference (50/50 guessing) whenever the tally is tied, because it merely compares the counts. Cue-Parity, however, makes completely deterministic predictions even when the counts are tied, because it only checks whether a_wins is odd or even. For example, if both options win on exactly 1 feature, Tallying predicts a tie, whereas Cue-Parity deterministically predicts A (since 1 is odd). If both options win on exactly 2 features, Tallying predicts a tie, whereas Cue-Parity deterministically predicts B (since 2 is even). We include these tied-tally trials alongside asymmetric trials (a_wins != b_wins) to ensure a robust quantitative dissociation across the full parameter space.

[2] To dissociate Cue-Parity (which computes parity over ALL features) from Lexicographic Parity (which computes parity only over the top K most valid features), we design trials where the parity of A-wins changes depending on whether the least valid features are included. By making A strictly beat B on specific lower-validity cues, the two models will predict opposite choices for most values of the k_frac parameter.

[3] To dissociate Lexicographic Parity from Cue-Parity, we manipulate whether the parity of A-wins changes when only the most valid cues are considered versus when all cues are considered. Lexicographic Parity restricts its parity calculation to the top K valid features (where K depends on k_frac), whereas Cue-Parity always computes parity across all features. By designing trials where option A strictly beats B on an odd number of the top features but an even number of features overall (and vice versa), the two models yield strictly opposite deterministic predictions. We vary the distribution of A-wins across the high-validity and low-validity features to ensure robust dissociation regardless of the exact k_frac value.

[4] To quantitatively dissociate Cue-Parity from Continuous Weighted Parity, we exploit the vulnerability of the latter to phase shifts caused by validity weighting. Continuous Weighted Parity computes a parity score using a cosine function over the validity-weighted sum of A-wins. When the weighted sum deviates sufficiently from an integer, the model enters a 'dead zone' (predicting 50/50 at half-integers) or undergoes a complete phase shift (flipping the predicted winner). By setting extreme validities (1.0s and 0.5s) and having option A win exclusively on either high-validity or low-validity cues, we push the weighted sum of wins into these phase-shift regions. Cue-Parity, which strictly counts the unweighted parity of A-wins, remains completely invariant to these validity-based perturbations, yielding a sharp dissociation.

[5] To quantitatively and qualitatively dissociate Continuous Weighted Parity from Cue-Parity, we exploit the fact that Continuous Weighted Parity modulates the parity signal based on the validity of the winning features, while Cue-Parity relies strictly on the unweighted integer count of wins. By using 10 features (5 high validity, 5 low validity), we can design trials where option A wins on 5 high-validity or 5 low-validity features. For Cue-Parity, 5 wins is always odd, leading to a deterministic preference for A. However, for Continuous Weighted Parity, the validity weighting shifts the effective sum of wins. For example, 5 low-validity wins can produce a weighted sum of ~4.375, and 5 high-validity wins can produce ~5.625. Both of these weighted sums cross a half-integer boundary (4.5 and 5.5, respectively), causing the continuous cosine parity function to undergo a phase shift and actually predict option B, creating a strict qualitative reversal. Additionally, trials with exactly 4 high or 4 low validity wins push the weighted sum exactly to half-integers (e.g., 4.5 or 3.5), where Continuous Weighted Parity predicts complete 50/50 indifference, whereas Cue-Parity deterministically predicts B (since 4 is even).

[6] To quantitatively dissociate Cue-Parity from Tallying-Biased Parity, we exploit the latter's fallback mechanism. Cue-Parity predicts that the probability of choosing Option A depends entirely and exclusively on whether the number of A-wins is odd or even, remaining completely invariant to the validities of those features or the number of B-wins. Tallying-Biased Parity, however, mixes the strict parity rule with a small probability of validity-weighted tallying. By holding the number of A-wins constant (e.g., exactly 1 win, which is odd) while drastically changing the validity-weighted tally (e.g., A wins on 1 high-validity cue vs. A wins on 1 low-validity cue while B wins on 3 high-validity cues), Tallying-Biased Parity predicts a systematic shift in choice probabilities between the trials. Cue-Parity predicts identical choice probabilities across these paired trials.

[7] To quantitatively dissociate Tallying-Biased Parity from pure Cue-Parity, we exploit the alignment between the primary Cue-Parity rule and the secondary validity-weighted tallying fallback. Pure Cue-Parity predicts that the probability of choosing A depends exclusively on whether the number of A-wins is odd or even, plus a constant uniform lapse rate. It is completely blind to the validities of the winning features. Tallying-Biased Parity, however, mixes the strict parity rule with a small probability of validity-weighted tallying. By designing pairs of trials where the unweighted parity of A-wins is identical (e.g., A wins 2 features, which is Even -> Parity favors B) but the validity-weighted tally is reversed (Trial A: A wins on 2 high-validity cues vs B's 3 low-validity cues; Trial B: A wins on 2 low-validity cues vs B's 2 high-validity cues), we can observe a systematic shift. Tallying-Biased Parity predicts a larger choice share for the parity-disfavored option when the tallying fallback supports it. Pure Cue-Parity predicts perfectly flat choice probabilities across these paired trials.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To dissociate Cue-Parity from Top-K Parity, we exploit the latter's selective exclusion of the lowest-validity cues. Top-K Parity computes parity over the top K valid features (where K is determined by k_frac in [0.9, 1.0]). For a 20-feature task, k_frac allows the model to evaluate the top 18, 19, or 20 cues, systematically dropping the 1 or 2 least valid cues. Cue-Parity, however, strictly computes parity over all 20 features. By holding the number of A-wins constant on the top 17 features (at an odd number, e.g., 5) and systematically varying whether A wins or loses on the 3 lowest-validity features, we create a matrix of trials where the parity of A-wins flips depending on exactly how many of the bottom features are included. Cue-Parity's predictions will perfectly track the parity of the full 20-feature set, while Top-K Parity's predictions will remain blind to the variations in the dropped lowest-validity features.",
  "validities": [
    0.51,
    0.52,
    0.53,
    0.54,
    0.55,
    0.56,
    0.57,
    0.58,
    0.59,
    0.6,
    0.61,
    0.62,
    0.63,
    0.64,
    0.65,
    0.66,
    0.67,
    0.68,
    0.69,
    0.99
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
      0,
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
      1
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
      0,
      0,
      0,
      0,
      0
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
      0
    ],
    [
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
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 4113,
  "candidates_token_count": 1402,
  "total_token_count": 8172
}
```
