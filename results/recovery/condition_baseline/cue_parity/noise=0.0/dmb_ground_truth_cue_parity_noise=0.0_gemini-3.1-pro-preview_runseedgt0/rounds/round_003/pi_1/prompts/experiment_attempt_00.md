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
**Description:** Tallying-Biased Parity (Strict Parity Dominance): Decision makers primarily rely on a strict Cue-Parity rule (preferring the option with an odd number of winning cues). However, instead of lapsing into purely uniform random noise, their secondary considerations consist of occasionally falling back to a standard validity-weighted tallying strategy. To ensure that tallying acts only as a weak bias and does not override parity in cases of option dominance, the fallback probability is strictly bounded to a low range.

**Parameters:**
- validities: validities
- beta_parity: [0.1, 20.0]
- beta_tally: [0.1, 10.0]
- epsilon: [0.0, 0.3]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Tallying-Biased Parity expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Core Strategy: Strict Cue-Parity
    a_wins = int(np.sum(a > b))
    parity_winner = 0 if (a_wins % 2 == 1) else 1
    parity_scores = np.array([1.0, 0.0]) if parity_winner == 0 else np.array([0.0, 1.0])
    
    # Fallback Strategy: Validity-weighted Tallying
    val_a = np.sum(validities * (a > b))
    val_b = np.sum(validities * (b > a))
    tally_scores = np.array([val_a, val_b])
    
    beta_parity = float(parameters["beta_parity"])
    beta_tally = float(parameters["beta_tally"])
    epsilon = float(parameters["epsilon"])
    
    # Parity probabilities via softmax
    z_p = beta_parity * (parity_scores - parity_scores.max())
    e_p = np.exp(z_p)
    p_parity = e_p / e_p.sum()
    
    # Tallying probabilities via softmax
    z_t = beta_tally * (tally_scores - tally_scores.max())
    e_t = np.exp(z_t)
    p_tally = e_t / e_t.sum()
    
    # Mixture of the two strategies
    return (1.0 - epsilon) * p_parity + epsilon * p_tally
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

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate Cue-Parity from Tallying-Biased Parity, we exploit the latter's fallback mechanism. Cue-Parity predicts that the probability of choosing Option A depends entirely and exclusively on whether the number of A-wins is odd or even, remaining completely invariant to the validities of those features or the number of B-wins. Tallying-Biased Parity, however, mixes the strict parity rule with a small probability of validity-weighted tallying. By holding the number of A-wins constant (e.g., exactly 1 win, which is odd) while drastically changing the validity-weighted tally (e.g., A wins on 1 high-validity cue vs. A wins on 1 low-validity cue while B wins on 3 high-validity cues), Tallying-Biased Parity predicts a systematic shift in choice probabilities between the trials. Cue-Parity predicts identical choice probabilities across these paired trials.",
  "validities": [
    0.95,
    0.9,
    0.85,
    0.6,
    0.55
  ],
  "trial_a_ratings": [
    [
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
      0,
      0
    ],
    [
      0,
      0,
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
      0
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
      0,
      0,
      0
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
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3638,
  "candidates_token_count": 484,
  "total_token_count": 6219
}
```
